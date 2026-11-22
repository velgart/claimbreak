r"""cellscape_classes.py -- major cell classes from CellScape protein on the same cells as CosMx.

Input: the .h5ad from seurat_to_h5ad.py (Li et al. 2026 figshare object: CosMx counts in X; CellScape per-cell
protein in obs as cs_<marker>, cs_Case, cs_Image, ...). Read only; writes three files.

Protein classes, independent of the transcripts (--method zscore, default):
  per marker and per --group (default cs_Case), log1p(intensity) as a robust z-score (median, 1.4826 x MAD); a
  class's score is the mean z of its markers (MARKERS below). A cell gets the class with the highest score when
  that score is >= --zmin and exceeds the second-highest by >= --margin; two or more classes >= --zmin without that
  margin -> 'mixed'; otherwise 'unassigned'. The first version (--method gmm: a two-component mixture per marker,
  class score = mean posterior of the brighter component, >= --pos with all others < --neg) split the bulk of
  heavy-tailed marker distributions on the real data (CD31 92%, CD38 87% 'positive') and called 81% of cells mixed.

Writes:
  OUT/protein_classes.csv     per cell: class, scores, the authors' ct_major_new and gate category
  OUT/protein_ref.h5ad        cells with a protein class: CosMx counts in X, obs protein_class, donor (= --group):
                              a CosMx-native reference for `claimbreak compartment label --ref ...
                              --ref-levels P=protein_class --ref-donor donor`
  OUT/protein_section.h5ad    all cells as a labeling input (X whole-cell counts; layers nucleus = X and cyto = 0,
                              because this object has no transcript compartments: use the 'whole' arm only), obs x, y
                              in um from x_slide_mm, y_slide_mm, panel 'CosMx1k', donor, protein_class
Prints marker positive rates, class composition, and protein class against the authors' labels.

Usage: python cellscape_classes.py li2026_cosmx_ref.h5ad --out ~/claimbreak_results/cellscape_classes
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARKERS = {
    "Epithelial": ["PanCK", "E-Cadherin", "EpCAM"],
    "T_NK": ["CD3"],
    "B": ["CD20", "CD19"],
    "Plasma": ["CD138", "CD38"],
    "Myeloid": ["CD68", "CD163"],
    "Endothelial": ["CD31", "CD105"],
    "Stromal": ["SMA", "Podoplanin"],
}

ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("--out", required=True)
ap.add_argument("--group", default="cs_Case", help="obs column: mixtures are fitted per group (patient)")
ap.add_argument("--method", default="zscore", choices=["zscore", "gmm"])
ap.add_argument("--zmin", type=float, default=2.0)
ap.add_argument("--margin", type=float, default=1.0)
ap.add_argument("--pos", type=float, default=0.7)
ap.add_argument("--neg", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

import anndata as ad
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture

out = Path(a.out).expanduser()
if out.exists() and any(out.iterdir()):
    sys.exit(f"{out} exists and is not empty; choose a new --out")
out.mkdir(parents=True, exist_ok=True)
A = ad.read_h5ad(Path(a.h5ad).expanduser())
cols = {m: f"cs_{m}" for ms in MARKERS.values() for m in ms}
miss = [c for c in cols.values() if c not in A.obs]
if miss:
    sys.exit(f"protein columns missing: {miss}")
has = A.obs[list(cols.values())].notna().all(axis=1).to_numpy()
grp = A.obs[a.group].astype(str).to_numpy() if a.group in A.obs else np.full(A.n_obs, "all")
print(f"{A.n_obs:,} cells; with CellScape protein {has.sum():,}; groups ({a.group}): "
      f"{pd.Series(grp[has]).value_counts().to_dict()}", flush=True)

qs = [0.01, 0.25, 0.5, 0.75, 0.9, 0.99]
print("\nmarker intensity quantiles (raw), cells with protein:")
print(A.obs.loc[has, list(cols.values())].quantile(qs).T.rename(index=lambda c: c[3:]).round(1).to_string())
pos = pd.DataFrame(np.nan, index=A.obs_names, columns=list(cols))
for m, c in cols.items():
    v = np.log1p(np.clip(A.obs[c].to_numpy(float), 0, None))
    for g in np.unique(grp[has]):
        idx = np.where(has & (grp == g))[0]
        x = v[idx]
        if a.method == "gmm":
            gm = GaussianMixture(2, random_state=a.seed).fit(x.reshape(-1, 1))
            hi = int(np.argmax(gm.means_.ravel()))
            val = gm.predict_proba(x.reshape(-1, 1))[:, hi]
        else:
            med = np.median(x)
            mad = 1.4826 * np.median(np.abs(x - med)) or np.std(x) or 1.0
            val = (x - med) / mad
        pos.iloc[idx, pos.columns.get_loc(m)] = val
thr_hi, thr_lo = (a.pos, a.neg) if a.method == "gmm" else (a.zmin, None)
print(f"\nmarker share above {thr_hi} ({'posterior' if a.method == 'gmm' else 'robust z'}), cells with protein:")
print((pos[has] >= thr_hi).mean().round(3).to_string())

score = pd.DataFrame({k: pos[ms].mean(axis=1) for k, ms in MARKERS.items()})
sc0 = score.fillna(-np.inf if a.method == "zscore" else 0.0)      # cells without protein: 'no_protein' below
n_hi = (sc0 >= thr_hi).sum(axis=1)
best = sc0.idxmax(axis=1)
second = pd.Series(np.sort(sc0.to_numpy(), axis=1)[:, -2], index=sc0.index)
top = sc0.max(axis=1)
if a.method == "gmm":
    ok_cls = (top >= a.pos) & (second < a.neg)
    mixed = n_hi >= 2
else:
    ok_cls = (top >= a.zmin) & (top - second >= a.margin)
    mixed = (n_hi >= 2) & ~ok_cls
cls = np.where(~has, "no_protein", np.where(ok_cls, best, np.where(mixed, "mixed", "unassigned")))
res = pd.DataFrame({"protein_class": cls}, index=A.obs_names)
res = res.join(score.add_prefix("score_").round(4))
for c in ("ct_major_new", "category", a.group, "cell_id"):
    if c in A.obs:
        res[c] = A.obs[c].astype(str).to_numpy()
res.to_csv(out / "protein_classes.csv")
print("\nprotein classes (all cells):")
print(res.protein_class.value_counts().to_string())
assigned = ~res.protein_class.isin(["no_protein", "mixed", "unassigned"])
print(f"\nassigned {assigned.mean():.3f} of all cells, {assigned[has].mean():.3f} of cells with protein")
if "ct_major_new" in res:
    print("\nprotein class (rows) against the authors' ct_major_new (columns), cells with protein:")
    print(pd.crosstab(res.protein_class[has], res.ct_major_new[has]).to_string())

X = sp.csr_matrix(A.X)
donor = grp
R = ad.AnnData(X[np.where(assigned.to_numpy())[0]],
               obs=pd.DataFrame({"protein_class": res.protein_class[assigned].to_numpy(),
                                 "donor": donor[assigned.to_numpy()]},
                                index=A.obs_names[assigned.to_numpy()]),
               var=pd.DataFrame(index=A.var_names))
R.write_h5ad(out / "protein_ref.h5ad")
print(f"\nwrote protein_ref.h5ad: {R.n_obs:,} cells; per class {R.obs.protein_class.value_counts().to_dict()}; "
      f"per donor {R.obs.donor.value_counts().to_dict()}")

xmm = A.obs["x_slide_mm"].to_numpy(float) if "x_slide_mm" in A.obs else np.zeros(A.n_obs)
ymm = A.obs["y_slide_mm"].to_numpy(float) if "y_slide_mm" in A.obs else np.zeros(A.n_obs)
n = np.asarray(X.sum(1)).ravel()
S = ad.AnnData(X, obs=pd.DataFrame({"x": xmm * 1000.0, "y": ymm * 1000.0, "n_whole": n, "n_nucleus": n,
                                    "panel": "CosMx1k", "donor": donor, "section": "li2026_cellscape",
                                    "protein_class": res.protein_class.to_numpy()}, index=A.obs_names),
               var=pd.DataFrame(index=A.var_names))
S.layers["nucleus"] = X.copy()
S.layers["cyto"] = sp.csr_matrix(X.shape, dtype=X.dtype)
S.write_h5ad(out / "protein_section.h5ad")
print(f"wrote protein_section.h5ad (label with --arms whole only) and protein_classes.csv in {out}")
