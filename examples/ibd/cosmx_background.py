r"""cosmx_background.py -- why do CosMx cells fall into one broad class? (IBD pilot)

Read only. For one or more converted pieces (compartment .h5ad) and their labels (claimbreak compartment label):

1. Protein by label. CosMx stains every cell for PanCK (epithelium), CD45 (immune) and CD68 (macrophages); the means
   are in obs (Mean.PanCK, Mean.CD45, Mean.CD68, ...). Per transcript label: cells, median of each stain, and the
   share of cells above the slide's 75th percentile of each stain. If cells called "Neural" are PanCK- or CD45-bright,
   the label is wrong for them, whatever the transcript classifier says.
2. Markers by label. Per label, the share of counts in marker sets of the coarse classes (genes on the panel only).
   A class whose cells carry no more of their own markers than other cells do is being assigned by something other
   than those markers (for example a flat background).
3. Background (with --raw and --sample): negative-control probe counts per cell from the deposited exprMat
   (columns NegPrb* / Negative*; the converter drops them). Expected background counts in the panel genes =
   negative counts per probe x number of panel genes; background share = that / the cell's panel counts (capped at 1).
   Per label and overall: quartiles per cell, and the pooled share (summed over cells; per-cell negative counts are
   mostly 0, so the quartiles understate it). A share of several tenths means a 1% flat floor in the labeler is far
   too small for this platform.

Usage:
  python cosmx_background.py --h5ad H5AD/I0284RI0276R2119_p3.h5ad --labels LABELS_DIR [--arm L1_whole] \
      [--raw RAW/cosmx --sample I0284RI0276R2119] --out ~/claimbreak_results/cosmx_background_I0284_p3
Writes OUT/protein_by_label.csv, markers_by_label.csv, background_by_label.csv (with --raw), cells.csv.gz.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT20", "CDH1", "CLDN4", "PIGR", "LGALS4", "FABP1"],
    "T_NK": ["CD3E", "CD3D", "CD2", "CD247", "TRAC", "IL7R", "CCL5", "NKG7", "GZMA"],
    "B": ["MS4A1", "CD79A", "CD19", "BANK1", "CD22"],
    "Plasma": ["JCHAIN", "MZB1", "IGHA1", "IGKC", "XBP1", "SDC1"],
    "Myeloid": ["CD68", "LYZ", "CD163", "C1QA", "C1QB", "CD14", "FCGR3A", "AIF1"],
    "Mast": ["TPSAB1", "CPA3", "KIT", "MS4A2"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"],
    "Muscle": ["ACTA2", "DES", "MYH11", "CNN1", "RGS5", "PDGFRB"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "PLVAP", "CLDN5"],
    "Neural": ["PLP1", "S100B", "SOX10", "GFAP", "CDH19", "NRXN1", "PRPH", "ELAVL4", "UCHL1"],
}
NEG = re.compile(r"^(NegPrb|Negative|NegControl)", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", nargs="+", required=True)
ap.add_argument("--labels", required=True, help="folder with <piece>.labels.csv.gz")
ap.add_argument("--arm", default="L1_whole", help="label column (level_arm)")
ap.add_argument("--raw", help="folder with the CosMx <sample>_exprMat_file.csv(.gz)")
ap.add_argument("--sample", help="slide name for --raw")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import anndata as ad
import scipy.sparse as sp

out = Path(a.out).expanduser()
if out.exists() and any(out.iterdir()):
    sys.exit(f"{out} exists and is not empty; choose a new --out")
out.mkdir(parents=True, exist_ok=True)

neg = None
if a.raw:
    if not a.sample:
        sys.exit("--raw needs --sample")
    hits = sorted(Path(a.raw).expanduser().glob(f"*{a.sample}*_exprMat_file.csv*"))
    if not hits:
        sys.exit(f"no exprMat for {a.sample} in {a.raw}")
    E = pd.read_csv(hits[0])
    ncols = [c for c in E.columns if NEG.match(c)]
    if not ncols:
        sys.exit(f"no negative-probe columns in {hits[0].name}; columns start {list(E.columns[:8])}")
    if "cell" in E:
        key = E["cell"].astype(str).to_numpy()
    else:   # as compartment_to_h5ad.py: (fov, cell_ID) -> the metadata's cell id, which names the h5ad cells
        mf = sorted(Path(a.raw).expanduser().glob(f"*{a.sample}*_metadata_file.csv*"))
        if not mf:
            sys.exit(f"exprMat has no 'cell' column and no metadata_file for {a.sample} was found")
        meta = pd.read_csv(mf[0], usecols=lambda c: c in ("cell", "cell_id", "fov", "cell_ID"))
        idc = "cell" if "cell" in meta else "cell_id"
        mk = pd.Series(meta[idc].astype(str).to_numpy(), index=pd.MultiIndex.from_arrays([meta["fov"], meta["cell_ID"]]))
        key = mk.reindex(pd.MultiIndex.from_arrays([E["fov"], E["cell_ID"]])).to_numpy()
    neg = pd.DataFrame({"neg_counts": E[ncols].sum(axis=1).to_numpy(), "neg_probes": len(ncols)}, index=key)
    neg = neg.groupby(level=0).sum().assign(neg_probes=len(ncols))
    print(f"{hits[0].name}: {len(E):,} cells, {len(ncols)} negative probes, median negative counts per cell "
          f"{np.median(neg.neg_counts):.1f}", flush=True)

prot_rows, mark_rows, bg_rows, cells = [], [], [], []
for f in a.h5ad:
    f = Path(f).expanduser()
    piece = f.name.replace(".h5ad", "")
    lf = Path(a.labels).expanduser() / f"{piece}.labels.csv.gz"
    if not lf.exists():
        print(f"{piece}: no labels file {lf}; skipped", flush=True)
        continue
    A = ad.read_h5ad(f)
    L = pd.read_csv(lf, usecols=["cell_id", a.arm], keep_default_na=False, dtype={"cell_id": str})
    lab = L.set_index("cell_id")[a.arm].reindex(A.obs_names.astype(str)).fillna("").to_numpy()
    X = sp.csr_matrix(A.X)
    tot = np.asarray(X.sum(1)).ravel()
    df = pd.DataFrame({"piece": piece, "label": np.where(lab == "", "(none)", lab), "counts": tot, "n_genes": A.n_vars},
                      index=A.obs_names.astype(str))
    stains = [c for c in A.obs.columns if c.startswith("Mean.")]
    for c in stains:
        df[c] = A.obs[c].to_numpy(float)
    genes = pd.Index(A.var_names.astype(str))
    for cls, gl in MARKERS.items():
        gi = genes.get_indexer([g for g in gl if g in genes])
        gi = gi[gi >= 0]
        df[f"mk_{cls}"] = (np.asarray(X[:, gi].sum(1)).ravel() / np.maximum(tot, 1)) if len(gi) else np.nan
    if neg is not None:
        nb = neg.reindex(df.index)
        matched = nb.neg_counts.notna().mean()
        print(f"{piece}: negative probes matched for {matched:.3f} of cells", flush=True)
        if matched < 0.5:
            print("  cell ids do not match the exprMat; check the id form (metadata 'cell' column)", flush=True)
        df["neg_counts"] = nb.neg_counts.to_numpy()
        df["bg_share"] = np.minimum(nb.neg_counts.to_numpy() / nb.neg_probes.to_numpy() * A.n_vars / np.maximum(tot, 1), 1.0)
    cells.append(df)
    print(f"{piece}: {len(df):,} cells, labels from {a.arm}: {df.label.value_counts(normalize=True).round(3).head(6).to_dict()}",
          flush=True)

if not cells:
    sys.exit("no piece with labels")
D = pd.concat(cells)
D.to_csv(out / "cells.csv.gz", compression="gzip")
g = D.groupby("label")
stains = [c for c in D.columns if c.startswith("Mean.")]
if stains:
    P = g.size().rename("cells").to_frame()
    for c in stains:
        q75 = D[c].quantile(0.75)
        P[f"{c}_median"] = g[c].median()
        P[f"{c}_share_top25"] = g[c].apply(lambda v: float((v > q75).mean()))
    P = P.sort_values("cells", ascending=False)
    P.to_csv(out / "protein_by_label.csv")
    print("\nprotein by label (median; share of cells in the top quarter of the stain):")
    print(P.round(3).to_string())
mk = [c for c in D.columns if c.startswith("mk_")]
M = g[mk].mean().rename(columns=lambda c: c[3:])
M.insert(0, "cells", g.size())
M = M.sort_values("cells", ascending=False)
M.to_csv(out / "markers_by_label.csv")
print("\nmean share of a cell's counts in each class's markers (rows: label; columns: marker set):")
print(M.round(4).to_string())
if "bg_share" in D:
    B = g["bg_share"].describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]]
    B.loc["(all)"] = D["bg_share"].describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]]
    # pooled over cells (per-cell negative counts are mostly 0, so per-cell quantiles understate the background):
    # expected background in the panel = total negative counts / probes x panel genes, over total panel counts
    D["_bg_exp"] = D["neg_counts"] / neg["neg_probes"].iloc[0] * D["n_genes"]
    pooled = D.groupby("label").apply(lambda g: g["_bg_exp"].sum() / g["counts"].sum(), include_groups=False)
    B["pooled"] = pooled
    B.loc["(all)", "pooled"] = D["_bg_exp"].sum() / D["counts"].sum()
    B["neg_per_cell_mean"] = D.groupby("label")["neg_counts"].mean()
    B.loc["(all)", "neg_per_cell_mean"] = D["neg_counts"].mean()
    D.drop(columns="_bg_exp", inplace=True)
    B.to_csv(out / "background_by_label.csv")
    print("\nestimated background share of panel counts (negative probes), by label:")
    print(B.round(3).to_string())
print(f"\nwrote {out}")
