r"""cellscape_check.py -- is each CellScape protein row on the right CosMx cell? (read only)

Input: the .h5ad from seurat_to_h5ad.py (CosMx counts in X; CosMx's own stains in obs, e.g. Max.PanCK, Mean.CD45,
Mean.CD3, Mean.DAPI; CellScape protein as cs_<marker>, cs_Case, cs_Image).

1. Protein against protein: Spearman correlation, over cells, of a CellScape marker with CosMx's own stain of the
   same antigen on the same cell, and with the same CosMx stain on the k-th nearest neighbor (k = 1..5, by CosMx
   x/y). If the rows are on the right cells, same-cell correlation is clearly above every neighbor's.
2. Protein against RNA: Spearman correlation of a CellScape marker with its genes' counts per cell (log1p of counts
   per 1,000), same cell and k-th neighbor.
Both overall and per --group (cs_Case) and per cs_Image when present (the matching may fail in some images only).
0. Positive control: CosMx's own stains against CosMx's own RNA on the same cells (no CellScape involved).
3. Geometry: an affine map (least squares) from the CellScape nucleus centroids (cs_nuc_centroid_col/row) to the
   CosMx coordinates of the id-matched cells: R^2 per axis and median residual (um). A correct join has R^2 near 1
   and residuals of a few um; a scrambled join has R^2 near 0.

Usage: python cellscape_check.py li2026_cosmx_ref.h5ad --out ~/claimbreak_results/cellscape_check
Writes OUT/protein_protein.csv, OUT/protein_rna.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS_PROT = {"PanCK": ["Max.PanCK", "Mean.PanCK"], "CD45": ["Max.CD45", "Mean.CD45"], "CD3": ["Max.CD3", "Mean.CD3"],
              "DNA": ["Mean.DAPI", "Max.DAPI"]}
PAIRS_RNA = {"PanCK": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT20"], "E-Cadherin": ["CDH1"], "EpCAM": ["EPCAM"],
             "CD3": ["CD3E", "CD3D", "CD3G"], "CD20": ["MS4A1"], "CD19": ["CD19"], "CD138": ["SDC1"],
             "CD38": ["CD38"], "CD68": ["CD68"], "CD163": ["CD163"], "CD31": ["PECAM1"], "CD34": ["CD34"],
             "SMA": ["ACTA2"], "Podoplanin": ["PDPN"], "Vimentin": ["VIM"], "CD45": ["PTPRC"], "CD4": ["CD4"],
             "CD8": ["CD8A", "CD8B"], "Foxp3": ["FOXP3"], "Ki67": ["MKI67"], "Muc1": ["MUC1"]}

ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("--out", required=True)
ap.add_argument("--group", default="cs_Case")
ap.add_argument("--kmax", type=int, default=5)
a = ap.parse_args()

import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

out = Path(a.out).expanduser()
if out.exists() and any(out.iterdir()):
    sys.exit(f"{out} exists and is not empty; choose a new --out")
out.mkdir(parents=True, exist_ok=True)
A = ad.read_h5ad(Path(a.h5ad).expanduser())
if "x_slide_mm" in A.obs:
    xy = A.obs[["x_slide_mm", "y_slide_mm"]].to_numpy(float) * 1000
elif {"x", "y"} <= set(A.obs):
    xy = A.obs[["x", "y"]].to_numpy(float)
else:
    sys.exit("no CosMx coordinates (x_slide_mm/y_slide_mm or x/y) in obs")
_, nn = cKDTree(xy).query(xy, k=a.kmax + 1)                     # nn[:, 0] = the cell itself
X = sp.csr_matrix(A.X)
tot = np.asarray(X.sum(1)).ravel()
genes = pd.Index(A.var_names.astype(str))


def rna(gl):
    gi = genes.get_indexer([g for g in gl if g in genes])
    gi = gi[gi >= 0]
    if not len(gi):
        return None
    return np.log1p(np.asarray(X[:, gi].sum(1)).ravel() / np.maximum(tot, 1) * 1000)


def rho(x, y, m):
    m = m & np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[m], y[m]).statistic if m.sum() > 50 else np.nan


groups = {"(all)": np.ones(A.n_obs, bool)}
for col in (a.group, "cs_Image"):
    if col in A.obs:
        for g in sorted(A.obs[col].astype(str).unique()):
            if g not in ("nan", "None", ""):
                groups[f"{col}={g}"] = (A.obs[col].astype(str) == g).to_numpy()

CONTROL = {"Max.PanCK": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT20"], "Max.CD45": ["PTPRC"],
           "Max.CD3": ["CD3E", "CD3D", "CD3G"], "Mean.CD68": ["CD68"]}
crow = []
for stain, gl in CONTROL.items():
    if stain not in A.obs:
        continue
    t = rna(gl)
    if t is None:
        continue
    v = A.obs[stain].to_numpy(float)
    r = {"cosmx_stain": stain, "rna": "+".join(g for g in gl if g in genes)}
    for k in range(a.kmax + 1):
        r["same_cell" if k == 0 else f"nb{k}"] = rho(v, t[nn[:, k]], np.ones(A.n_obs, bool))
    crow.append(r)
if crow:
    C0 = pd.DataFrame(crow)
    C0.to_csv(out / "control_cosmx_stain_rna.csv", index=False)
    print("\npositive control: CosMx stain against CosMx RNA, same cell and k-th neighbour (all cells):")
    print(C0.round(3).to_string(index=False))

rows = []
for kind, pairs in (("protein", PAIRS_PROT), ("rna", PAIRS_RNA)):
    for marker, targets in pairs.items():
        c = f"cs_{marker}"
        if c not in A.obs:
            continue
        p = np.log1p(np.clip(A.obs[c].to_numpy(float), 0, None))
        if kind == "protein":
            tcol = next((t for t in targets if t in A.obs), None)
            if tcol is None:
                continue
            t = A.obs[tcol].to_numpy(float)
            tname = tcol
        else:
            t = rna(targets)
            if t is None:
                continue
            tname = "+".join(g for g in targets if g in genes)
        for gname, gm in groups.items():
            r = {"kind": kind, "cellscape": marker, "cosmx": tname, "group": gname, "cells": int((gm & np.isfinite(p)).sum())}
            for k in range(a.kmax + 1):
                r["same_cell" if k == 0 else f"nb{k}"] = rho(p, t[nn[:, k]], gm)
            rows.append(r)
R = pd.DataFrame(rows)
R["best_neighbour"] = R[[f"nb{k}" for k in range(1, a.kmax + 1)]].max(axis=1)
R["same_minus_best_nb"] = R.same_cell - R.best_neighbour
for kind in ("protein", "rna"):
    T = R[R.kind == kind]
    T.to_csv(out / f"protein_{kind}.csv", index=False)
    print(f"\n{kind}: Spearman of CellScape marker with CosMx {'stain' if kind == 'protein' else 'RNA'} on the same "
          f"cell and on its k-th nearest neighbour (all cells):")
    print(T[T.group == "(all)"][["cellscape", "cosmx", "cells", "same_cell"] + [f"nb{k}" for k in range(1, a.kmax + 1)]]
          .round(3).to_string(index=False))
sub = R[(R.group != "(all)") & R.cellscape.isin(["PanCK", "CD45", "CD3", "DNA"]) & (R.kind == "protein")]
if len(sub):
    print("\nper patient / image (CosMx stain): same cell minus best neighbour (should be clearly > 0):")
    print(sub.pivot_table(index="group", columns="cellscape", values="same_minus_best_nb").round(3).to_string())
if {"cs_nuc_centroid_col", "cs_nuc_centroid_row"} <= set(A.obs):
    geo = []
    for gname, gm in groups.items():
        cxy = A.obs[["cs_nuc_centroid_col", "cs_nuc_centroid_row"]].to_numpy(float)
        m = gm & np.isfinite(cxy).all(1) & np.isfinite(xy).all(1)
        if m.sum() < 50:
            continue
        D = np.column_stack([cxy[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(D, xy[m], rcond=None)
        res = xy[m] - D @ coef
        r2 = 1 - (res ** 2).sum(0) / ((xy[m] - xy[m].mean(0)) ** 2).sum(0)
        geo.append({"group": gname, "cells": int(m.sum()), "R2_x": r2[0], "R2_y": r2[1],
                    "median_residual_um": float(np.median(np.hypot(res[:, 0], res[:, 1]))),
                    "median_nn_distance_um": float(np.median(np.hypot(*(xy[nn[:, 1]] - xy).T)[m]))})
    G = pd.DataFrame(geo)
    G.to_csv(out / "geometry.csv", index=False)
    print("\ngeometry: affine map of CellScape nucleus centroids onto the CosMx coordinates of the id-matched cells:")
    print(G.round(3).to_string(index=False))
print(f"\nwrote {out}")
