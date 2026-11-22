#!/usr/bin/env python
"""merscope_to_h5ad.py -- whole-cell and nucleus-only counts of the same cells from a Vizgen MERSCOPE run
(MERSCOPE FFPE human uterine cancer with CD45RO co-staining, a second
same-section protein truth). Same output layout as compartment_to_h5ad.py, so the label/truth chain applies unchanged.

with --nuclei (merscope_nuclei.py, StarDist on DAPI), a transcript is nuclear when it lies in a nucleus owned
by its own cell; this is the method to use (in QC, the threshold below merged packed tumor nuclei).
MERSCOPE gives no nucleus flag per transcript. Without --nuclei, a transcript is nuclear when the DAPI intensity at its position is
above a local threshold, as Xenium's overlaps_nucleus is a 2D call on its DAPI image:
  DAPI      one mosaic plane (default z3, the middle of seven), read in place (memory map; never loaded or changed)
  position  micron -> mosaic pixel by micron_to_mosaic_pixel_transform.csv; intensity = mean of a 3 x 3 window (0.3 um)
  threshold Otsu on log(1 + DAPI) per block of --block pixels (2048 = 221 um), from pixels sampled every --stride;
            blocks with too little tissue take the median threshold of all blocks
Transcripts with cell_id -1 (outside every cell) and control codewords (Blank-*) are dropped.

Output, in --out:
  <sample>.h5ad            X whole-cell counts, layers nucleus and cyto; obs x, y (cell center, um), volume, sample,
                           donor, panel, section, unit, tissue, n_whole, n_nucleus; uns source (files, sizes, SHA-256 of
                           the small inputs), uns check (whole-cell totals against cell_by_gene.csv)
  <sample>.protein.csv.gz  cell_id and each antibody's signal per um3 (<stain>_raw / volume; _hp for high_pass), the
                           input of xenium_protein.py truth
  <sample>.dapi_threshold.csv  the threshold grid; <sample>.nucleus_summary.json  shares in the nucleus, per gene too
Refuses to overwrite.

  python merscope_to_h5ad.py --dir .../merscope_uterine_cd45ro/download --prefix HumanUterineCancerPatient2-ROCostain \
      --sample MS_UTERINE_RO --out .../merscope_uterine_cd45ro/h5ad
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL = re.compile(r"^(Blank|NegControl|Negative|Unassigned)", re.I)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def open_image(p):
    """Memory map of an uncompressed TIFF (read only); otherwise a zarr view of it."""
    import tifffile
    try:
        return tifffile.memmap(str(p), mode="r")
    except Exception as e:
        print(f"  (no memory map: {e}; using zarr)", flush=True)
        import zarr
        return zarr.open(tifffile.imread(str(p), aszarr=True), mode="r")


def otsu(v, bins=256):
    h, e = np.histogram(v, bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def threshold_grid(img, block, stride, min_tissue=0.05):
    H, W = img.shape
    ny, nx = -(-H // block), -(-W // block)
    T = np.full((ny, nx), np.nan)
    floor = None
    t0 = time.time()
    for i in range(ny):
        for j in range(nx):
            b = np.asarray(img[i * block:(i + 1) * block:stride, j * block:(j + 1) * block:stride], float)
            lv = np.log1p(b.ravel())
            if floor is None and lv.size:
                floor = np.log1p(20.0)
            tissue = lv[lv > floor]
            if tissue.size >= min_tissue * lv.size and tissue.size > 500:
                T[i, j] = otsu(tissue)
        if i % 10 == 0:
            print(f"  threshold rows {i + 1}/{ny} ({time.time() - t0:.0f} s)", flush=True)
    med = np.nanmedian(T)
    n_fill = int(np.isnan(T).sum())
    T[np.isnan(T)] = med
    return T, med, n_fill


def dapi_at(img, r, c, H, W):
    acc = np.zeros(len(r), float)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = np.clip(r + dr, 0, H - 1)
            cc = np.clip(c + dc, 0, W - 1)
            o = np.lexsort((cc, rr))                      # read in file order
            v = np.empty(len(r), float)
            v[o] = np.asarray(img[rr[o], cc[o]], float) if hasattr(img, "dtype") and not hasattr(img, "oindex") \
                else np.asarray(img.vindex[rr[o], cc[o]], float)
            acc += v
    return acc / 9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="download folder: <prefix>_cell_metadata.csv, _cell_by_gene.csv, "
                                                 "_detected_transcripts.csv; images/<prefix>_mosaic_DAPI_z<z>.tif, "
                                                 "images/<prefix>_micron_to_mosaic_pixel_transform.csv")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--z", type=int, default=3)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=5_000_000)
    ap.add_argument("--donor", default="", help="donor id for obs (default: the prefix)")
    ap.add_argument("--tissue", default="uterine cancer")
    ap.add_argument("--nuclei", default="", help="folder of merscope_nuclei.py output (StarDist nuclei); "
                                                  "without it, a DAPI threshold (not recommended)")
    ap.add_argument("--max-rows", type=int, default=0, help="test only: stop after this many transcripts")
    a = ap.parse_args()
    d, out = Path(a.dir).expanduser().resolve(), Path(a.out).expanduser().resolve()
    if out == d or d in out.parents:
        sys.exit("--out must be outside the download folder")
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{a.sample}.h5ad"
    for p in (dst, out / f"{a.sample}.protein.csv.gz"):
        if p.exists():
            sys.exit(f"{p} exists; not overwritten")
    P = a.prefix
    f_meta, f_cbg, f_tx = (d / f"{P}_{s}" for s in ("cell_metadata.csv", "cell_by_gene.csv", "detected_transcripts.csv"))
    def one_image(name):          # <prefix>_<name> (bucket) or <prefix>_images_<name> (browser download)
        f = [p for p in sorted((d / "images").glob(f"{P}*{name}")) if not p.name.startswith("._")]
        if len(f) != 1:
            sys.exit(f"need one images/{P}*{name}, found {[p.name for p in f]}")
        return f[0]
    f_img = one_image(f"mosaic_DAPI_z{a.z}.tif")
    f_tr = one_image("micron_to_mosaic_pixel_transform.csv")
    for p in (f_meta, f_cbg, f_tx, f_img, f_tr):
        if p is None or not p.exists():
            sys.exit(f"missing input: {p}")
    A_tr = np.loadtxt(f_tr)
    print(f"transform: x {A_tr[0, 0]:.4f} px/um + {A_tr[0, 2]:.1f}, y {A_tr[1, 1]:.4f} px/um + {A_tr[1, 2]:.1f}", flush=True)

    M = pd.read_csv(f_meta)
    M = M.rename(columns={M.columns[0]: "cell"})
    M["cell"] = M.cell.astype(np.int64)
    cidx = pd.Series(np.arange(len(M)), index=M.cell.to_numpy())
    print(f"{len(M):,} cells in {f_meta.name}", flush=True)

    img = open_image(f_img)
    H, W = img.shape[-2:]
    print(f"DAPI z{a.z}: {W:,} x {H:,} px, {img.dtype}", flush=True)
    NL = owner = None
    nuc_info = {}
    if a.nuclei:
        # nuclei from merscope_nuclei.py (StarDist); a transcript is nuclear when it lies in a nucleus owned by
        # its own cell (as Xenium's overlaps_nucleus); transcripts in a neighbor's nucleus are counted apart
        nd = Path(a.nuclei).expanduser()
        f_nl = next(iter(sorted(nd.glob("*.nuclei.npy"))), None)
        f_nt = next(iter(sorted(nd.glob("*.nuclei.csv.gz"))), None)
        if f_nl is None or f_nt is None:
            sys.exit(f"no *.nuclei.npy / *.nuclei.csv.gz in {nd}")
        NL = np.load(f_nl, mmap_mode="r")
        NT = pd.read_csv(f_nt)
        f_ow = next(iter(sorted(nd.glob("*.owners.csv.gz"))), None)     # owners by cell outline
        owner_rule = "nearest cell centre within 8 um (merscope_nuclei.py)"
        if f_ow is not None:
            OW = pd.read_csv(f_ow)
            NT = NT.drop(columns="owner").merge(OW[["nucleus", "owner"]], on="nucleus", how="left")
            NT["owner"] = NT.owner.fillna(-1).astype(np.int64)
            owner_rule = f"cell outline containing the nucleus centroid ({f_ow.name}, merscope_owners.py)"
        print(f"nucleus owner: {owner_rule}", flush=True)
        owner = np.full(int(NT.nucleus.max()) + 1, -1, np.int64)
        oi = cidx.reindex(NT.owner.to_numpy()).to_numpy()
        ok_o = ~np.isnan(oi) & (NT.owner.to_numpy() >= 0)
        owner[NT.nucleus.to_numpy()[ok_o]] = oi[ok_o].astype(np.int64)
        nuc_area = NT[ok_o].groupby(oi[ok_o].astype(np.int64)).area_um2.sum()
        js = nd / f_nl.name.replace(".nuclei.npy", ".nuclei.json")
        nuc_info = {"rule": "in a StarDist nucleus owned by the transcript's cell (merscope_nuclei.py)",
                    "labels": f_nl.name, "table": f_nt.name, "owner_rule": owner_rule,
                    "scale": int(round(H / NL.shape[0])),
                    "nuclei": int(len(NT)), "owned": int(ok_o.sum()),
                    "segmentation": json.loads(js.read_text()) if js.exists() else {}}
        print(f"nuclei: {len(NT):,} from {f_nl.name} (labels {NL.shape[1]:,} x {NL.shape[0]:,}), {int(ok_o.sum()):,} "
              f"owned by a cell of the metadata", flush=True)
        T, med, n_fill = None, np.nan, 0
    else:
        T, med, n_fill = threshold_grid(img, a.block, a.stride)
        pd.DataFrame(T).to_csv(out / f"{a.sample}.dapi_threshold.csv", index=False)
        print(f"DAPI threshold log1p: median {med:.2f} (= {np.expm1(med):.0f}), range {T.min():.2f}-{T.max():.2f}; "
              f"{n_fill} of {T.size} blocks filled with the median", flush=True)
        nuc_info = {"rule": "log1p(mean 3x3 DAPI) > blockwise Otsu", "z": a.z, "block_px": a.block, "stride": a.stride,
                    "median_threshold": med, "blocks_filled": n_fill}

    import pyarrow.csv as pacsv
    head = pd.read_csv(f_tx, nrows=0).columns
    want = ["global_x", "global_y", "gene", "cell_id"]
    miss = [c for c in want if c not in head]
    if miss:
        sys.exit(f"{f_tx.name} lacks {miss}; columns {list(head)}")
    reader = pacsv.open_csv(f_tx, read_options=pacsv.ReadOptions(block_size=1 << 28),
                            convert_options=pacsv.ConvertOptions(include_columns=want,
                                                                 column_types={"cell_id": "int64", "gene": "string"}))
    genes, gidx = [], {}
    Wm = Nm = None
    n_rows = n_kept = n_ctrl = n_out = n_nuc = n_other = 0
    per_gene_w, per_gene_n = {}, {}
    buf = []
    t0 = time.time()

    def flush(df):
        nonlocal Wm, Nm, n_kept, n_ctrl, n_out, n_nuc, n_other
        cell = df.cell_id.to_numpy()
        ctrl = df.gene.str.match(CONTROL).to_numpy()
        inside = cell >= 0
        n_ctrl += int((ctrl & inside).sum())
        keep = inside & ~ctrl
        df = df[keep]
        ri = cidx.reindex(df.cell_id.to_numpy()).to_numpy()
        ok = ~np.isnan(ri)
        n_out += int((~ok).sum())
        df, ri = df[ok], ri[ok].astype(np.int64)
        col = np.rint(A_tr[0, 0] * df.global_x.to_numpy() + A_tr[0, 2]).astype(np.int64)
        row = np.rint(A_tr[1, 1] * df.global_y.to_numpy() + A_tr[1, 2]).astype(np.int64)
        if NL is not None:
            sc = nuc_info["scale"]
            rr, cc2 = np.clip(row // sc, 0, NL.shape[0] - 1), np.clip(col // sc, 0, NL.shape[1] - 1)
            o = np.lexsort((cc2, rr))
            lid = np.empty(len(rr), np.int64)
            lid[o] = np.asarray(NL[rr[o], cc2[o]], np.int64)
            own = owner[np.clip(lid, 0, len(owner) - 1)]
            nuc = (lid > 0) & (own == ri)
            n_other += int(((lid > 0) & (own != ri)).sum())
        else:
            v = np.log1p(dapi_at(img, row, col, H, W))
            thr = T[np.clip(row // a.block, 0, T.shape[0] - 1), np.clip(col // a.block, 0, T.shape[1] - 1)]
            nuc = v > thr
        g = df.gene.to_numpy()
        for x in pd.unique(g):
            if x not in gidx:
                gidx[x] = len(genes)
                genes.append(x)
        gc = pd.Series(g).map(gidx).to_numpy()
        shape = (len(M), 4000)
        w = sp.coo_matrix((np.ones(len(ri), np.int32), (ri, gc)), shape=shape).tocsr()
        nn = sp.coo_matrix((np.ones(int(nuc.sum()), np.int32), (ri[nuc], gc[nuc])), shape=shape).tocsr()
        Wm = w if Wm is None else Wm + w
        Nm = nn if Nm is None else Nm + nn
        n_kept += len(ri)
        n_nuc += int(nuc.sum())

    for b in reader:
        df = b.to_pandas()
        n_rows += len(df)
        buf.append(df)
        if sum(len(x) for x in buf) >= a.batch:
            flush(pd.concat(buf, ignore_index=True))
            buf = []
            print(f"  {n_rows / 1e6:,.0f} M transcripts read, {n_kept / 1e6:,.1f} M in cells, nucleus share "
                  f"{n_nuc / max(n_kept, 1):.3f} ({time.time() - t0:.0f} s)", flush=True)
        if a.max_rows and n_rows >= a.max_rows:
            break
    if buf:
        flush(pd.concat(buf, ignore_index=True))
    G = len(genes)
    order = np.argsort(genes)
    gn = np.array(genes)[order]
    Wm, Nm = Wm[:, :G].tocsr()[:, order], Nm[:, :G].tocsr()[:, order]
    print(f"{n_rows:,} transcripts read; {n_kept:,} in cells kept ({n_ctrl:,} control codewords in cells dropped, "
          f"{n_out:,} with a cell id missing from the metadata); nucleus share {n_nuc / max(n_kept, 1):.3f}"
          + (f"; in another cell's nucleus {n_other / max(n_kept, 1):.3f} (counted as cytoplasm)" if NL is not None else ""),
          flush=True)

    # check against the platform's cell_by_gene
    check = {}
    if not a.max_rows:
        C = pd.read_csv(f_cbg, index_col=0)
        C.index = C.index.astype(np.int64)
        C = C[[c for c in C.columns if not CONTROL.match(c)]]
        C = C.reindex(M.cell.to_numpy()).fillna(0)
        mine = np.asarray(Wm.sum(1)).ravel()
        theirs = C.sum(1).to_numpy()
        check = dict(cells=int(len(mine)), identical_totals=float(np.mean(mine == theirs)),
                     total_ratio=float(mine.sum() / max(theirs.sum(), 1)))
        print(f"check against {f_cbg.name}: totals identical in {check['identical_totals']:.4f} of cells, "
              f"sum ratio {check['total_ratio']:.4f}", flush=True)

    import anndata as ad
    obs = pd.DataFrame(index=M.cell.astype(str).to_numpy())
    obs["x"], obs["y"] = M.center_x.to_numpy(float), M.center_y.to_numpy(float)
    obs["volume"] = M.volume.to_numpy(float)
    obs["cell_area"], obs["nucleus_area"] = np.nan, np.nan
    if NL is not None:
        na = np.zeros(len(M))
        na[nuc_area.index.to_numpy()] = nuc_area.to_numpy()
        obs["nucleus_area"] = np.where(na > 0, na, np.nan)
    obs["sample"], obs["donor"], obs["panel"], obs["tissue"] = a.sample, a.donor or P, "MERSCOPE500", a.tissue
    obs["section"] = f"MERSCOPE500:{a.sample}"
    obs["unit"] = a.donor or P
    obs["n_whole"] = np.asarray(Wm.sum(1)).ravel()
    obs["n_nucleus"] = np.asarray(Nm.sum(1)).ravel()
    A = ad.AnnData(X=Wm.astype(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(gn, name="gene")))
    A.layers["nucleus"] = Nm.astype(np.float32)
    A.layers["cyto"] = (Wm - Nm).astype(np.float32)
    src = {p.name: {"bytes": p.stat().st_size, **({"sha256": sha(p)} if p.stat().st_size < 2e9 else {})}
           for p in (f_meta, f_cbg, f_tx, f_img, f_tr)}
    A.uns["source"] = json.dumps({"platform": "merscope", "files": src, "control_regex": CONTROL.pattern,
                                  "nucleus": nuc_info, "in_other_nucleus": n_other, "max_rows": a.max_rows})
    A.uns["check"] = json.dumps(check)
    for c in ("sample", "donor", "panel", "section", "unit", "tissue"):
        A.obs[c] = A.obs[c].astype(str).astype("category")
    A.write_h5ad(dst, compression="gzip")
    share = obs.n_nucleus.sum() / max(obs.n_whole.sum(), 1)
    print(f"wrote {dst}: {A.n_obs:,} cells x {A.n_vars:,} genes; median counts whole {np.median(obs.n_whole):.0f}, "
          f"nucleus {np.median(obs.n_nucleus):.0f} (share in nucleus {share:.3f})", flush=True)

    pr = pd.DataFrame({"cell_id": obs.index})
    for c in M.columns:
        if c.lower().startswith("anti") and c.endswith(("_raw", "_high_pass")):
            name = re.sub(r"^Anti-[A-Za-z]+-", "", c).replace("_raw", "").replace("_high_pass", "_hp")
            pr[name] = M[c].to_numpy(float) / M.volume.to_numpy(float)
    pr.to_csv(out / f"{a.sample}.protein.csv.gz", index=False)
    print(f"wrote {out / (a.sample + '.protein.csv.gz')}: columns {list(pr.columns[1:])}", flush=True)

    pw, pn = np.asarray(Wm.sum(0)).ravel(), np.asarray(Nm.sum(0)).ravel()
    g_share = pd.Series(pn / np.maximum(pw, 1), index=gn).sort_values()
    summ = dict(nucleus_share=share, in_other_nucleus_share=n_other / max(n_kept, 1), cell_share_median=float(np.median(obs.n_nucleus / np.maximum(obs.n_whole, 1))),
                cells_zero_nucleus=float(np.mean(obs.n_nucleus == 0)), check=check,
                lowest_genes=g_share.head(10).round(3).to_dict(), highest_genes=g_share.tail(10).round(3).to_dict())
    (out / f"{a.sample}.nucleus_summary.json").write_text(json.dumps(summ, indent=1))
    print("genes with the highest nucleus share: " + ", ".join(f"{k} {v:.2f}" for k, v in summ["highest_genes"].items()))
    print("genes with the lowest nucleus share:  " + ", ".join(f"{k} {v:.2f}" for k, v in summ["lowest_genes"].items()))


if __name__ == "__main__":
    main()
