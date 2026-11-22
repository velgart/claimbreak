#!/usr/bin/env python
"""xenium_protein_check.py -- the cosmx_protein_check.py tests on the protein we trust: Xenium Protein, renal carcinoma
(the paper calibration). Same three tests and the same gene signatures, so the CosMx breast multiomic
numbers can be read against a protein that worked as truth (protein recall, Fig. 2b).

Inputs, read only: the compartment .h5ad of the section (whole-cell counts in X, obs x, y in um) and
<sample>.protein.csv.gz (cell_id + background-subtracted intensity per protein; xenium_protein.py extract).
Positive cells: as in xenium_protein.py, a two-component mixture on log(1 + x) of the cells with non-zero intensity,
highest component; cells at zero are negative. Ashman D between the two components.
For each protein against its genes and class signature: Spearman with the cell's own and its nearest neighbor's
transcripts (per 1,000 counts); fold of the signature in protein-high over protein-low cells, and the same fold for
their nearest neighbors. Pair test: each protein-high cell against its nearest
protein-LOW neighbor (pair_fold), as in merscope_protein_check.py. Writes OUT/summary.csv.

  python xenium_protein_check.py --h5ad .../h5ad/XP_RCC.h5ad --protein .../h5ad/XP_RCC.protein.csv.gz --out OUT
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

PAIRS = {"CD3E": ["CD3E", "CD3D", "CD3G"], "CD8A": ["CD8A", "CD8B"], "CD4": ["CD4"], "CD45": ["PTPRC"],
         "CD20": ["MS4A1"], "CD68": ["CD68"], "CD163": ["CD163"], "PANCK": ["KRT8", "KRT18", "KRT19", "KRT7", "KRT5"],
         "EPCAM": ["EPCAM"], "CD45RO": ["PTPRC"], "CD45RA": ["PTPRC"]}
SIGS = {"T": ["CD3E", "CD3D", "CD3G", "CD2", "CD247", "TRAC", "TRBC1", "TRBC2", "IL7R", "LCK", "ZAP70", "CD8A", "CD8B",
              "GZMK", "CD6", "CD5", "CD96", "CD28", "ITK", "SKAP1", "CD27", "LAT"],
        "B": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "PAX5", "CD22", "FCRL1", "FCRL5", "VPREB3", "BLK"],
        "myeloid": ["CD68", "CD163", "CD14", "LYZ", "C1QA", "C1QB", "C1QC", "FCGR3A", "CSF1R", "MS4A7", "AIF1", "TYROBP"],
        "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1", "CLDN4", "ELF3", "KRT5", "KRT17"],
        "immune": ["PTPRC", "CORO1A", "LCP1", "LAPTM5", "CD53", "CD52", "ARHGDIB", "CD37"]}
SIG_OF = {"CD3E": "T", "CD8A": "T", "CD4": "T", "CD45": "immune", "CD20": "B", "CD68": "myeloid", "CD163": "myeloid",
          "PANCK": "epithelial", "EPCAM": "epithelial", "CD45RO": "T", "CD45RA": "immune"}


def norm(x):
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def mixture_high(v):
    """Positive calls as xenium_protein.py: 2-component mixture on log1p of non-zero values; Ashman D."""
    hi = np.zeros(len(v), bool)
    nz = v > 0
    x = np.log1p(v[nz]).reshape(-1, 1)
    gm = GaussianMixture(2, random_state=0).fit(x)
    o = np.argsort(gm.means_.ravel())
    m, s = gm.means_.ravel()[o], np.sqrt(gm.covariances_.ravel()[o])
    hi[nz] = gm.predict(x) == o[-1]
    return abs(m[1] - m[0]) * np.sqrt(2) / np.sqrt(s[0] ** 2 + s[1] ** 2), hi


def pair_test(xy, hi, k=30):
    """For each stain-high cell, its nearest stain-LOW neighbor (within k nearest cells); -1 if none.
    Only pairs within 15 um are used (pair_stats), so the low neighbor shares the high cell's surroundings."""
    d, nn = cKDTree(xy).query(xy, k=k + 1)
    d, nn = d[:, 1:], nn[:, 1:]
    lowmask = ~hi[nn]
    first = np.where(lowmask.any(1), lowmask.argmax(1), -1)
    rows = np.arange(len(xy))
    pick = np.where(first >= 0, nn[rows, np.maximum(first, 0)], -1)
    dist = np.where(first >= 0, d[rows, np.maximum(first, 0)], np.nan)
    return pick, dist


def pair_stats(z, hi, pick, dist, max_um=15.0):
    """pair_fold: signature in high cells over their nearest low neighbors (an antibody that marks its own cell: >> 1;
    a regional stain: about 1). lownb_fold: those low neighbors over all low cells (how clustered the class is)."""
    s = hi & (pick >= 0) & (dist <= max_um)   # touching pairs only: a regional stain then gives about 1
    zl = z[~hi].mean()
    return dict(pair_fold=z[s].mean() / z[pick[s]].mean() if z[pick[s]].mean() > 0 else np.nan,
                lownb_fold=z[pick[s]].mean() / zl if zl > 0 else np.nan,
                pair_dist_um=float(np.nanmedian(dist[s])), pair_cells=int(s.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-counts", type=int, default=20)
    a = ap.parse_args()
    import anndata as ad
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    A = ad.read_h5ad(Path(a.h5ad).expanduser())
    names = pd.Index(A.var_names.astype(str))
    genes = sorted({g for gs in list(PAIRS.values()) + list(SIGS.values()) for g in gs if g in names})
    X = A.X
    idx = [names.get_loc(g) for g in genes]
    Xg = X[:, idx]
    Xg = Xg.toarray() if sp.issparse(Xg) else np.asarray(Xg)
    tot = np.asarray(X.sum(1)).ravel().astype(float)
    obs = A.obs[["x", "y"]].copy()
    obs["cell_id"] = A.obs_names.astype(str)
    R = pd.DataFrame(Xg, columns=["rna_" + g for g in genes])
    R["cell_id"] = obs.cell_id.to_numpy()
    R["total"] = tot
    R["x"], R["y"] = obs.x.to_numpy(float), obs.y.to_numpy(float)
    P = pd.read_csv(Path(a.protein).expanduser(), dtype={"cell_id": str})
    pc = {norm(c): c for c in P.columns if c != "cell_id"}
    D = R.merge(P, on="cell_id")
    D = D[D.total >= a.min_counts].reset_index(drop=True)
    print(f"{len(R):,} cells in the h5ad, {len(P):,} with protein, {len(D):,} joined with >= {a.min_counts} counts "
          f"(median {D.total.median():.0f}); proteins {sorted(pc)}", flush=True)
    xy = D[["x", "y"]].to_numpy(float)
    nb = cKDTree(xy).query(xy, k=2)[1][:, 1]
    t = D.total.to_numpy(float)
    sig = {k: 1000 * D[["rna_" + g for g in gs if "rna_" + g in D.columns]].sum(1).to_numpy(float) / t
           for k, gs in SIGS.items()}
    print("signature genes found: " + ", ".join(f"{k} {sum('rna_' + g in D.columns for g in gs)}/{len(gs)}"
                                                for k, gs in SIGS.items()), flush=True)
    rows = []
    for pr, gs0 in PAIRS.items():
        col = pc.get(pr) or (pc.get("PTPRC") if pr == "CD45" else None) or (pc.get("MS4A1") if pr == "CD20" else None)
        gs = [g for g in gs0 if "rna_" + g in D.columns]
        if col is None or not gs:
            print(f"  {pr}: protein or genes missing; skipped")
            continue
        v = D[col].to_numpy(float)
        dA, hi = mixture_high(v)
        own = D[["rna_" + g for g in gs]].sum(1).to_numpy(float)
        rate = 1000 * own / t
        r_own, r_nb = spearmanr(v, rate).statistic, spearmanr(v, rate[nb]).statistic
        sg = SIG_OF[pr]
        z = sig[sg]
        s_own, s_nb = spearmanr(v, z).statistic, spearmanr(v, z[nb]).statistic
        f_own = z[hi].mean() / z[~hi].mean() if z[~hi].mean() > 0 else np.nan
        f_nb = z[nb][hi].mean() / z[nb][~hi].mean() if z[nb][~hi].mean() > 0 else np.nan
        pick, pdist = pair_test(xy, hi)
        ps = pair_stats(z, hi, pick, pdist)
        rows.append(dict(protein=pr, column=col, genes="+".join(gs), cells=len(D), ashman_D=dA, high_share=hi.mean(),
                         rho_own=r_own, rho_neighbour=r_nb, signature=sg, sig_rho_own=s_own, sig_rho_neighbour=s_nb,
                         sig_fold_high_vs_low=f_own, sig_fold_neighbour=f_nb,
                         cell_specific_excess=f_own / f_nb if f_nb else np.nan,
                         marker_share_high=(own[hi] > 0).mean(), marker_share_low=(own[~hi] > 0).mean(), **ps))
        print(f"  {pr:6s} D={dA:.2f} high {hi.mean():.3f}; rho own {r_own:.3f} vs neighbour {r_nb:.3f}; {sg} signature "
              f"{f_own:.2f}x (neighbours {f_nb:.2f}x, excess {f_own / f_nb:.2f}); pair_fold {ps['pair_fold']:.2f} "
              f"(low neighbours {ps['lownb_fold']:.2f}x all low, {ps['pair_dist_um']:.1f} um)", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False)
    print("\n" + S.drop(columns=["column", "genes"]).round(3).to_string(index=False))
    print(f"\nwrote {out}/summary.csv")


if __name__ == "__main__":
    main()
