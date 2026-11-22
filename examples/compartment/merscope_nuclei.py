#!/usr/bin/env python
"""merscope_nuclei.py -- nuclei of a MERSCOPE section from its DAPI plane with StarDist. A blockwise Otsu threshold merges packed tumor nuclei with the cytoplasm between them
(nucleus share 0.59 on QC crops); Xenium's own nuclei come from a neural network on DAPI, so this does the same.

  model      StarDist2D '2D_versatile_fluo' (pretrained; Schmidt et al. 2018), default thresholds
  scale      the plane is averaged in 2 x 2 blocks (0.216 um/px; tested on five 120 um crops at /2, /3, /4: /2 splits
             touching nuclei that /3 and /4 merge)
  normalize  percentiles 1 and 99.8 of the non-zero pixels of each tile (MERSCOPE fills unimaged mosaic
             areas with 0; with them, 5% of a tile set p1 to 0 and StarDist finds no nucleus in it) (with a floor:
             the tile's range is at least --floor of the
             whole plane's tissue range, so tiles with little tissue are not stretched into noise), clipped to [0, 1.2].
             With the whole plane's percentiles (--norm global), bright packed tissue loses contrast and
             nuclei (densest QC crop 80 nuclei against 154; brightest 121 against 169; an empty crop 4 and 4)
  tiles      --tile px (at /2) with --overlap px on each side; tiles without a cell center are skipped; a nucleus is
             kept by the tile whose core holds its centroid
  contrast   CLAHE (window --clahe-kernel 64 px at /2 = 14 um, clip 0.01), Gaussian sigma 1 px, objects
             under 15 um2 dropped. Tuned on six 324 um crops (merscope_crops.py): in hazy, out-of-focus tissue cells with
             a nucleus 17% -> 66%; dense tumor 67% -> 69%; at an unimaged edge 65% -> 73%. Clip 0.03 and no smoothing
             fragmented nuclei (densest crop: 442 objects < 15 um2 against 91)
  owner      each nucleus belongs to the cell whose center is nearest its centroid, within --max-owner-um
Writes into --out (refuses to overwrite):
  <sample>.nuclei.npy         uint32 labels at /2 (0 = none), a memory-mappable .npy
  <sample>.nuclei.csv.gz      nucleus id, centroid x, y (um), area (um2), owner cell id, distance to its center (um)
  <sample>.nuclei.json        settings, normalisation, counts, timing
  <sample>.nuclei_qc.png      outlines on crops (--qc-crops: row,col of full-resolution crop corners; default: the
                              five crops of merscope_nucleus_qc.py, read from --qc-dir if given)

  python merscope_nuclei.py --img .../images/<prefix>_images_mosaic_DAPI_z3.tif --meta .../<prefix>_cell_metadata.csv \
      --sample MS_UTERINE_RO --out .../merscope_uterine_cd45ro/nuclei
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def down2(a):
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:h, :w].astype(np.float32)
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=float, default=9.2594, help="full-resolution pixels per um (transform)")
    ap.add_argument("--tile", type=int, default=2048, help="tile side at /2")
    ap.add_argument("--overlap", type=int, default=96, help="overlap at /2 on each side (> a nucleus diameter)")
    ap.add_argument("--max-owner-um", type=float, default=8.0)
    ap.add_argument("--norm", choices=("tile", "global"), default="tile",
                    help="contrast from the tile's own percentiles (default) or from the whole plane ("
                         "lost nuclei in bright packed tissue: 80 against 154 on the densest QC crop)")
    ap.add_argument("--clahe-kernel", type=int, default=64,
                    help="CLAHE window at /2 px (64 = 14 um; 0 = off)")
    ap.add_argument("--clahe-clip", type=float, default=0.01, help="CLAHE clip limit (0.03 amplified noise into nuclei)")
    ap.add_argument("--smooth", type=float, default=1.0, help="Gaussian sigma (/2 px) after CLAHE")
    ap.add_argument("--min-area-um2", type=float, default=15.0, help="objects smaller than this are dropped")
    ap.add_argument("--floor", type=float, default=0.5,
                    help="tile contrast: p99.8 - p1 at least this share of the whole plane's range")
    ap.add_argument("--qc-crops", default="", help="r0,c0;r0,c0;... full-resolution corners of 120 um QC crops")
    ap.add_argument("--qc-dir", default="", help="folder with crops/<name>_r<row>_c<col>.tif (from the QC step)")
    ap.add_argument("--max-tiles", type=int, default=0, help="test only")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    f_lab, f_tab = out / f"{a.sample}.nuclei.npy", out / f"{a.sample}.nuclei.csv.gz"
    for p in (f_lab, f_tab):
        if p.exists():
            sys.exit(f"{p} exists; not overwritten")
    import tifffile
    from scipy.ndimage import gaussian_filter
    from skimage.exposure import equalize_adapthist
    from stardist.models import StarDist2D
    try:
        img = tifffile.memmap(str(Path(a.img).expanduser()), mode="r")
    except ValueError:            # compressed test crops (merscope_crops.py) are read whole
        img = tifffile.imread(str(Path(a.img).expanduser()))
    H, W = img.shape
    h2, w2 = H // 2, W // 2
    M = pd.read_csv(Path(a.meta).expanduser(), usecols=["EntityID", "center_x", "center_y"]).rename(columns={"EntityID": "cell"})
    cx2, cy2 = M.center_x.to_numpy() * a.px / 2, M.center_y.to_numpy() * a.px / 2
    T = a.tile
    ny, nx = -(-h2 // T), -(-w2 // T)
    occ = np.zeros((ny, nx), int)
    np.add.at(occ, (np.clip((cy2 // T).astype(int), 0, ny - 1), np.clip((cx2 // T).astype(int), 0, nx - 1)), 1)
    # a tile with a cell center is segmented, and so are its neighbors (nuclei near the edge of the tissue)
    from scipy.ndimage import binary_dilation
    tiles = np.argwhere(binary_dilation(occ > 0, np.ones((3, 3), bool)))
    print(f"DAPI {W:,} x {H:,} px -> /2 {w2:,} x {h2:,}; {len(tiles)} of {ny * nx} tiles of {T} px to segment", flush=True)
    # normalisation from tissue pixels: sample the centers of occupied tiles
    samp = []
    for i, j in np.argwhere(occ > 20)[:: max(1, int((occ > 20).sum() // 200))]:
        r, c = int(i * T * 2), int(j * T * 2)
        samp.append(np.asarray(img[r:r + 2 * T:16, c:c + 2 * T:16], np.float32).ravel())
    s = np.concatenate(samp)
    s = s[s > 0]                                  # unimaged areas (0) out
    s = s[s > np.percentile(s, 5)]
    p1, p998 = np.percentile(s, [1, 99.8])
    print(f"normalisation: p1 {p1:.0f}, p99.8 {p998:.0f} (from {len(s):,} sampled pixels)", flush=True)
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    lab = np.lib.format.open_memmap(f_lab, mode="w+", dtype=np.uint32, shape=(h2, w2))
    rows, nid, t0, n_zero_tiles, n_small = [], 0, time.time(), 0, 0
    O = a.overlap
    for k, (i, j) in enumerate(tiles):
        if a.max_tiles and k >= a.max_tiles:
            break
        r0, c0 = i * T, j * T
        R0, C0 = max(r0 - O, 0), max(c0 - O, 0)
        R1, C1 = min(r0 + T + O, h2), min(c0 + T + O, w2)
        x = down2(np.asarray(img[2 * R0:2 * R1, 2 * C0:2 * C1]))
        zero = float((x <= 0).mean())   # unimaged mosaic areas are 0 and must not set the contrast
        n_zero_tiles += zero > 0
        if a.norm == "tile":      # contrast per tile, with a floor so near-empty tiles are not stretched
            v = x[x > 0]
            if v.size < 1000:
                continue
            lo, hi = np.percentile(v, [1, 99.8])
            hi = max(hi, lo + a.floor * (p998 - p1))
        else:
            lo, hi = p1, p998
        x = np.clip((x - lo) / (hi - lo), 0, 1.2)
        if a.clahe_kernel > 0:    # local contrast (hazy, out-of-focus tissue), then light smoothing
            unimaged = x <= 0
            x = equalize_adapthist(np.clip(x, 0, 1), kernel_size=a.clahe_kernel, clip_limit=a.clahe_clip)
            if a.smooth > 0:
                x = gaussian_filter(x, a.smooth)
            x[unimaged] = 0
        L, _ = model.predict_instances(x, verbose=False, show_tile_progress=False)
        if a.min_area_um2 > 0 and L.max():      # fragments below a nucleus size are dropped
            area = np.bincount(L.ravel()) * (2.0 / a.px) ** 2
            small = area < a.min_area_um2
            small[0] = False
            n_small += int(small.sum())
            L = np.where(small[L], 0, L)
        n = int(L.max())
        if n:
            idx = np.nonzero(L.ravel())[0]
            lv = L.ravel()[idx]
            rr, cc = np.divmod(idx, L.shape[1])
            cnt = np.bincount(lv, minlength=n + 1)
            my = np.bincount(lv, rr, minlength=n + 1) / np.maximum(cnt, 1) + R0
            mx = np.bincount(lv, cc, minlength=n + 1) / np.maximum(cnt, 1) + C0
            core = (my >= r0) & (my < r0 + T) & (mx >= c0) & (mx < c0 + T) & (cnt > 0)
            core[0] = False
            ids = np.flatnonzero(core)
            new = np.zeros(n + 1, np.uint32)
            new[ids] = np.arange(nid + 1, nid + 1 + len(ids), dtype=np.uint32)
            Lg = new[L]
            sub = lab[R0:R1, C0:C1]
            put = Lg > 0
            sub[put] = Lg[put]
            for q in ids:
                rows.append((int(new[q]), mx[q], my[q], int(cnt[q])))
            nid += len(ids)
        if (k + 1) % 20 == 0 or k + 1 == len(tiles):
            el = time.time() - t0
            print(f"  tile {k + 1}/{len(tiles)}: {nid:,} nuclei, {n_zero_tiles} tiles with unimaged pixels ({el:.0f} s, about {el / (k + 1) * (len(tiles) - k - 1) / 60:.0f} "
                  f"min left)", flush=True)
    lab.flush()
    N = pd.DataFrame(rows, columns=["nucleus", "x2", "y2", "px2"])
    um2 = 2.0 / a.px
    N["x"], N["y"], N["area_um2"] = N.x2 * um2, N.y2 * um2, N.px2 * um2 ** 2
    d, k = cKDTree(np.c_[M.center_x.to_numpy(), M.center_y.to_numpy()]).query(N[["x", "y"]].to_numpy(), k=1)
    N["owner"] = np.where(d <= a.max_owner_um, M.cell.to_numpy()[k], -1)
    N["owner_dist_um"] = d
    N[["nucleus", "x", "y", "area_um2", "owner", "owner_dist_um"]].to_csv(f_tab, index=False)
    own = N[N.owner >= 0]
    per_cell = own.groupby("owner").size()
    summ = dict(sample=a.sample, model="StarDist2D 2D_versatile_fluo", scale=2, um_per_px=um2, tile=T, overlap=O,
                norm=a.norm, floor=a.floor, tiles_with_unimaged_pixels=int(n_zero_tiles),
                clahe_kernel=a.clahe_kernel, clahe_clip=a.clahe_clip, smooth=a.smooth, min_area_um2=a.min_area_um2,
                small_dropped=int(n_small),
                p1=float(p1), p998=float(p998), tiles=int(len(tiles)), nuclei=int(len(N)),
                area_um2_median=float(N.area_um2.median()) if len(N) else None,
                owned_share=float((N.owner >= 0).mean()) if len(N) else None,
                cells=int(len(M)), cells_with_nucleus=int(per_cell.size), cells_with_2plus=int((per_cell > 1).sum()),
                seconds=round(time.time() - t0))
    (out / f"{a.sample}.nuclei.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
    # QC outlines
    crops = []
    if a.qc_crops:
        crops = [tuple(map(int, x.split(","))) for x in a.qc_crops.split(";") if x]
    elif a.qc_dir:
        for f in sorted(Path(a.qc_dir).expanduser().glob("crops/*_r*_c*.tif")):
            m = re.search(r"_r(\d+)_c(\d+)\.tif$", f.name)
            crops.append((int(m.group(1)), int(m.group(2))))
    if crops:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
        s = int(120 * a.px)
        fig, axes = plt.subplots(1, len(crops), figsize=(4.2 * len(crops), 4.4))
        for ax, (r, c) in zip(np.atleast_1d(axes), crops):
            im = down2(np.asarray(img[r:r + s, c:c + s]))
            L = np.asarray(lab[r // 2:r // 2 + im.shape[0], c // 2:c // 2 + im.shape[1]])
            vlo, vhi = np.percentile(im, [1, 99.8])
            ax.imshow(np.clip((im - vlo) / max(vhi - vlo, 1), 0, 1), cmap="gray")
            ax.contour(find_boundaries(L), levels=[0.5], colors="red", linewidths=0.4)
            sel = (cy2 >= r / 2) & (cy2 < r / 2 + im.shape[0]) & (cx2 >= c / 2) & (cx2 < c / 2 + im.shape[1])
            ax.scatter(cx2[sel] - c / 2, cy2[sel] - r / 2, s=4, c="cyan")
            ax.set_title(f"r{r} c{c}: {len(np.unique(L)) - 1} nuclei, {int(sel.sum())} cells; cover {(L > 0).mean():.2f}",
                         fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / f"{a.sample}.nuclei_qc.png", dpi=130)
        print(f"wrote {out / (a.sample + '.nuclei_qc.png')}")
    print(f"wrote {f_lab} and {f_tab}")


if __name__ == "__main__":
    main()
