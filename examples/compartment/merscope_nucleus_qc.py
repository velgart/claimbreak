#!/usr/bin/env python
"""merscope_nucleus_qc.py -- does the DAPI threshold of merscope_to_h5ad.py outline nuclei?

Reads (never changes) the DAPI plane, the threshold grid written by merscope_to_h5ad.py and the cell metadata. Draws
crops of --size um: DAPI (log, contrast by crop percentiles), the nucleus mask (log1p of the 3 x 3 mean above the block
threshold) as a red outline, cell centers as cyan dots. Crops: the densest tissue, a typical block, a block at the
tissue edge (a block next to one filled with the median), and one block with the lowest and one with the highest
threshold. Per crop: share of tissue pixels called nucleus, and the threshold. Writes OUT/nucleus_qc.png and
OUT/nucleus_qc.csv.

  python merscope_nucleus_qc.py --img .../images/<prefix>_images_mosaic_DAPI_z3.tif \
      --thr .../h5ad/MS_UTERINE_RO.dapi_threshold.csv --meta .../<prefix>_cell_metadata.csv --out .../nucleus_qc
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, binary_erosion


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img", required=True)
    ap.add_argument("--thr", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--size", type=float, default=60.0, help="crop side, um")
    ap.add_argument("--px", type=float, default=9.2594, help="pixels per um")
    a = ap.parse_args()
    import tifffile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    img = tifffile.memmap(str(Path(a.img).expanduser()), mode="r")
    H, W = img.shape
    T = pd.read_csv(a.thr).to_numpy(float)
    med = np.median(T)
    M = pd.read_csv(a.meta, usecols=["center_x", "center_y"])
    cr, cc = (M.center_y.to_numpy() * a.px).astype(int), (M.center_x.to_numpy() * a.px).astype(int)
    # cells per block, to find dense, typical and edge blocks
    nb = np.zeros(T.shape, int)
    np.add.at(nb, (np.clip(cr // a.block, 0, T.shape[0] - 1), np.clip(cc // a.block, 0, T.shape[1] - 1)), 1)
    filled = np.isclose(T, med)
    tissue = (nb > 50) & ~filled
    cand = {}
    ij = np.argwhere(tissue)
    cand["densest"] = tuple(np.unravel_index(np.argmax(np.where(tissue, nb, -1)), T.shape))
    q = np.where(tissue, nb, np.nan)
    typ = np.nanmedian(q)
    cand["typical"] = tuple(ij[np.argmin(np.abs(nb[tissue] - typ))])
    edge = [tuple(x) for x in ij if filled[max(x[0] - 1, 0):x[0] + 2, max(x[1] - 1, 0):x[1] + 2].any()]
    if edge:
        cand["edge"] = max(edge, key=lambda x: nb[x])
    cand["lowest threshold"] = tuple(ij[np.argmin(T[tissue])])
    cand["highest threshold"] = tuple(ij[np.argmax(T[tissue])])
    s = int(a.size * a.px)
    fig, axes = plt.subplots(1, len(cand), figsize=(4.2 * len(cand), 4.6))
    rows = []
    for ax, (name, (bi, bj)) in zip(np.atleast_1d(axes), cand.items()):
        # crop centerd on the cell nearest the block center
        r0, c0 = bi * a.block + a.block // 2, bj * a.block + a.block // 2
        k = np.argmin((cr - r0) ** 2 + (cc - c0) ** 2)
        r0, c0 = int(np.clip(cr[k] - s // 2, 0, H - s)), int(np.clip(cc[k] - s // 2, 0, W - s))
        im = np.asarray(img[r0:r0 + s, c0:c0 + s], float)
        lv = np.log1p(uniform_filter(im, 3))
        thr = T[bi, bj]
        nuc = lv > thr
        lo, hi = np.percentile(np.log1p(im), [1, 99.5])
        ax.imshow(np.clip((np.log1p(im) - lo) / (hi - lo), 0, 1), cmap="gray")
        ax.contour(nuc, levels=[0.5], colors="red", linewidths=0.6)
        sel = (cr >= r0) & (cr < r0 + s) & (cc >= c0) & (cc < c0 + s)
        ax.scatter(cc[sel] - c0, cr[sel] - r0, s=6, c="cyan")
        tis = lv > np.log1p(20)
        share = float(nuc[tis].mean()) if tis.any() else np.nan
        ax.set_title(f"{name}\nthr {thr:.2f} (= {np.expm1(thr):.0f}); nucleus {share:.2f} of tissue px\n"
                     f"{int(sel.sum())} cells in {a.size:.0f} um", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        rows.append(dict(crop=name, block_row=int(bi), block_col=int(bj), threshold=thr, nucleus_share_px=share,
                         cells=int(sel.sum()), px_median=float(np.median(im)), px_p99=float(np.percentile(im, 99))))
    fig.tight_layout()
    fig.savefig(out / "nucleus_qc.png", dpi=150)
    R = pd.DataFrame(rows)
    R.to_csv(out / "nucleus_qc.csv", index=False)
    print(R.round(3).to_string(index=False))
    print(f"blocks: {int(tissue.sum())} with tissue, {int(filled.sum())} filled with the median ({med:.2f})")
    print(f"wrote {out}/nucleus_qc.png and nucleus_qc.csv")


if __name__ == "__main__":
    main()
