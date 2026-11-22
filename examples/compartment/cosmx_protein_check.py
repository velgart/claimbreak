#!/usr/bin/env python
"""cosmx_protein_check.py -- can CosMx same-cell protein (64-plex Immuno-Oncology panel) serve as protein truth?
(the paper feasibility check; CosMx Human Breast Multiomic dataset, RNA and protein on the same slide)

Reads, without changing them, the RNA and protein flat files of one CosMx Same-Cell Multiomics run: the protein
_exprMat_file (antibody intensity per cell), the RNA _exprMat_file (only the marker genes are read) and the RNA
_metadata_file (cell centers, nCount_RNA). Cells are joined on fov and cell_ID; both layers share one segmentation.
For each antibody against its genes, the same three tests as hnscc_stain_check.py:
  separation   two-component Gaussian mixture on log(1 + intensity); Ashman D (we use D >= 2 for Xenium Protein)
  own vs nbr   Spearman of the antibody with the cell's OWN marker transcripts per 1,000 counts, and with its NEAREST
               NEIGHBOUR's (and with stain-low neighbors only)
  enrichment   share of antibody-high cells with at least one marker transcript, against antibody-low cells
plus a gene-signature test (summed counts of a class's genes: stronger when single genes are sparse) and
the negative-control antibodies (Ms IgG1, Rb IgG) as a background scale. Writes OUT/summary.csv and histograms.

  python cosmx_protein_check.py --rna .../Flatfiles_RNA/flatFiles/BreastCancer \
      --protein .../Flatfiles_Protein/flatFiles/BreastCancer --out .../cosmx_breast_multiomic/protein_check
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

PAIRS = {"CD3": ["CD3E", "CD3D", "CD3G"], "CD8": ["CD8A", "CD8B"], "CD4": ["CD4"], "CD45": ["PTPRC"],
         "CD20": ["MS4A1"], "CD68": ["CD68"], "CD163": ["CD163"], "EpCAM": ["EPCAM"], "Her2": ["ERBB2"],
         "SMA": ["ACTA2"], "CD31": ["PECAM1"]}
NEG = ["Ms IgG1", "Rb IgG"]
# gene signatures (summed counts per 1,000), a stronger test than single genes when counts per gene are sparse
SIGS = {"T": ["CD3E", "CD3D", "CD3G", "CD2", "CD247", "TRAC", "TRBC1", "TRBC2", "IL7R", "LCK", "ZAP70", "CD8A", "CD8B",
              "GZMK", "CD6", "CD5", "CD96", "CD28", "ITK", "SKAP1", "CD27", "LAT"],
        "B": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "PAX5", "CD22", "FCRL1", "FCRL5", "VPREB3", "BLK"],
        "myeloid": ["CD68", "CD163", "CD14", "LYZ", "C1QA", "C1QB", "C1QC", "FCGR3A", "CSF1R", "MS4A7", "AIF1", "TYROBP"],
        "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1", "CLDN4", "ELF3", "KRT5", "KRT17"],
        "immune": ["PTPRC", "CORO1A", "LCP1", "LAPTM5", "CD53", "CD52", "ARHGDIB", "CD37"]}
SIG_OF = {"CD3": "T", "CD8": "T", "CD4": "T", "CD45": "immune", "CD20": "B", "CD68": "myeloid", "CD163": "myeloid",
          "EpCAM": "epithelial", "Her2": "epithelial"}


def one(d, pat):
    f = sorted(glob.glob(os.path.join(d, pat)))
    if not f:
        raise SystemExit(f"no {pat} in {d}")
    return f[0]


def ashman(x):
    gm = GaussianMixture(2, random_state=0).fit(x.reshape(-1, 1))
    o = np.argsort(gm.means_.ravel())
    m, s = gm.means_.ravel()[o], np.sqrt(gm.covariances_.ravel()[o])
    hi = gm.predict(x.reshape(-1, 1)) == o[-1]
    return abs(m[1] - m[0]) * np.sqrt(2) / np.sqrt(s[0] ** 2 + s[1] ** 2), hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rna", required=True, help="RNA flatFiles/<sample> folder")
    ap.add_argument("--protein", required=True, help="protein flatFiles/<sample> folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-counts", type=int, default=50)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(one(a.protein, "*_exprMat_file.csv*"))
    M = pd.read_csv(one(a.rna, "*_metadata_file.csv*"),
                    usecols=lambda c: c in ("fov", "cell_ID", "CenterX_global_px", "CenterY_global_px", "nCount_RNA"))
    rf = one(a.rna, "*_exprMat_file.csv*")
    head = pd.read_csv(rf, nrows=0).columns
    genes = sorted({g for gs in list(PAIRS.values()) + list(SIGS.values()) for g in gs if g in head})
    R = pd.read_csv(rf, usecols=["fov", "cell_ID"] + genes)
    R = R.rename(columns={g: "rna_" + g for g in genes})      # gene names such as CD4 or CD68 equal antibody names
    D = P.merge(R, on=["fov", "cell_ID"]).merge(M, on=["fov", "cell_ID"])
    D = D[(D.cell_ID > 0) & (D.nCount_RNA >= a.min_counts)].reset_index(drop=True)
    print(f"{len(P):,} protein cells, {len(R):,} RNA cells, {len(D):,} joined with >= {a.min_counts} RNA counts "
          f"(median {D.nCount_RNA.median():.0f})", flush=True)
    xy = D[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
    nb = cKDTree(xy).query(xy, k=2)[1][:, 1]
    neg = [c for c in NEG if c in D.columns]
    bg = D[neg].mean(1).to_numpy(float) if neg else np.zeros(len(D))
    print(f"negative-control antibodies {neg}: median {np.median(bg):.2f}", flush=True)
    tot = D.nCount_RNA.to_numpy(float)
    sig = {k: 1000 * D[["rna_" + g for g in gs if "rna_" + g in D.columns]].sum(1).to_numpy(float) / tot
           for k, gs in SIGS.items()}
    print("signature genes found: " + ", ".join(f"{k} {sum('rna_' + g in D.columns for g in gs)}/{len(gs)}"
                                                for k, gs in SIGS.items()), flush=True)
    rows = []
    for ab, gs0 in PAIRS.items():
        gs = [g for g in gs0 if "rna_" + g in D.columns]
        if ab not in D.columns or not gs:
            print(f"  {ab}: missing antibody or genes; skipped")
            continue
        v = D[ab].to_numpy(float)
        own = D[["rna_" + g for g in gs]].sum(1).to_numpy(float)
        rate = 1000 * own / D.nCount_RNA.to_numpy(float)
        dA, hi = ashman(np.log1p(np.clip(v, 0, None)))
        r_own = spearmanr(v, rate).statistic
        r_nb = spearmanr(v, rate[nb]).statistic
        low = ~hi[nb]
        r_nbl = spearmanr(v[low], rate[nb][low]).statistic
        sg = SIG_OF.get(ab)
        if sg:
            z = sig[sg]
            s_own, s_nb = spearmanr(v, z).statistic, spearmanr(v, z[nb]).statistic
            s_fold = float(z[hi].mean() / z[~hi].mean()) if z[~hi].mean() > 0 else np.nan
            s_fold_nb = float(z[nb][hi].mean() / z[nb][~hi].mean()) if z[nb][~hi].mean() > 0 else np.nan
        else:
            s_own = s_nb = s_fold = s_fold_nb = np.nan
        rows.append(dict(antibody=ab, genes="+".join(gs), cells=len(D), median_intensity=float(np.median(v)),
                         signature=sg or "", sig_rho_own=s_own, sig_rho_neighbour=s_nb, sig_fold_high_vs_low=s_fold,
                         sig_fold_neighbour=s_fold_nb,
                         over_negctrl=float(np.median(v) / np.median(bg)) if np.median(bg) > 0 else np.nan,
                         ashman_D=dA, high_share=float(hi.mean()), rho_own=r_own, rho_neighbour=r_nb,
                         rho_neighbour_lowonly=r_nbl, marker_share_high=float((own[hi] > 0).mean()),
                         marker_share_low=float((own[~hi] > 0).mean())))
        print(f"  {ab:6s} D={dA:.2f} high {hi.mean():.3f}; rho own {r_own:.3f} vs neighbour {r_nb:.3f} "
              f"(low neighbours {r_nbl:.3f}); with a {'/'.join(gs)} transcript: high {(own[hi] > 0).mean():.2f}, "
              f"low {(own[~hi] > 0).mean():.2f}", flush=True)
        if sg:
            print(f"         {sg} signature: rho own {s_own:.3f} vs neighbour {s_nb:.3f}; antibody-high / low cells "
                  f"{s_fold:.2f}x (their neighbours {s_fold_nb:.2f}x)", flush=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            lv = np.log1p(np.clip(v, 0, None))
            bins = np.linspace(np.percentile(lv, 0.5), np.percentile(lv, 99.5), 80)
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.hist(lv[own > 0], bins, alpha=0.6, label=f"with {'/'.join(gs)} transcript")
            ax.hist(lv[own == 0], bins, alpha=0.6, label="without")
            ax.set_xlabel(f"log(1 + {ab} intensity)"); ax.set_ylabel("cells"); ax.legend(fontsize=7)
            ax.set_title(f"{ab}: D = {dA:.2f}", fontsize=9)
            fig.tight_layout(); fig.savefig(out / f"{ab}.png", dpi=120); plt.close(fig)
        except Exception as e:
            print(f"  (no plot: {e})")
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False)
    cols = ["antibody", "ashman_D", "high_share", "rho_own", "rho_neighbour", "signature", "sig_rho_own",
            "sig_rho_neighbour", "sig_fold_high_vs_low", "sig_fold_neighbour", "marker_share_high", "marker_share_low"]
    print("\n" + S[cols].round(3).to_string(index=False))
    print(f"\nwrote {out}/summary.csv and the histograms")


if __name__ == "__main__":
    main()
