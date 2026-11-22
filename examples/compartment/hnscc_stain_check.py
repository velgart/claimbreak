#!/usr/bin/env python
"""hnscc_stain_check.py -- does the CosMx CD3 morphology stain identify T cells well enough to serve as protein truth?
(the paper feasibility check; HNSCC CosMx TMAs, Dryad doi:10.5061/dryad.n2z34tn7x)

Reads each TMA's _metadata_file.csv (stain intensities, cell centers) and _exprMat_file.csv (counts) straight from the
zip; nothing is extracted and the download folder is only read. For each stain (CD3, and CD45 and PanCK for
comparison) and each TMA:
  separation   two-component Gaussian mixture on log(1 + Mean intensity) of cells with non-zero intensity;
               Ashman D between the components (we use D >= 2 for Xenium Protein); share of cells in the high one
  own vs nbr   Spearman correlation of the stain with the cell's OWN marker transcripts (CD3: CD3E+CD3D+CD3G per
               1,000 counts; CD45: PTPRC; PanCK: keratins in the panel) and with its NEAREST NEIGHBOUR's
  enrichment   share of stain-high cells with at least one marker transcript, against stain-low cells
A usable stain separates (D >= 2), correlates clearly more with its own transcripts than with the neighbor's, and
is enriched for its marker transcripts. Writes OUT/summary.csv and OUT/<TMA>_<stain>.png (histograms).

  python hnscc_stain_check.py --zip .../download/CosMX_RNA_TMA_Data_Expr.zip --out .../hnscc_cosmx_dryad/stain_check
"""
import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

MARKERS = {"CD3": ["CD3E", "CD3D", "CD3G"], "CD45": ["PTPRC"],
           "PanCK": ["KRT5", "KRT6A", "KRT6B", "KRT7", "KRT8", "KRT13", "KRT14", "KRT15", "KRT17", "KRT18", "KRT19"]}


def ashman(gm):
    order = np.argsort(gm.means_.ravel())
    m = gm.means_.ravel()[order]
    s = np.sqrt(gm.covariances_.ravel()[order])
    w = gm.weights_[order]
    return abs(m[-1] - m[-2]) * np.sqrt(2) / np.sqrt(s[-1] ** 2 + s[-2] ** 2), w[-1], order[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--um-per-px", type=float, default=0.168)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(Path(a.zip).expanduser())
    names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
    tmas = sorted({re.sub(r"_metadata_file\.csv$", "", Path(n).name) for n in names if n.endswith("_metadata_file.csv")})
    rows = []
    for t in tmas:
        meta_n = next(n for n in names if Path(n).name == f"{t}_metadata_file.csv")
        expr_n = next(n for n in names if Path(n).name == f"{t}_exprMat_file.csv")
        M = pd.read_csv(z.open(meta_n))
        head = pd.read_csv(z.open(expr_n), nrows=0).columns
        genes = [g for gs in MARKERS.values() for g in gs if g in head]
        E = pd.read_csv(z.open(expr_n), usecols=lambda c: c in ("fov", "cell_ID") or c in genes)
        # total counts per cell (all gene columns), read in chunks to keep memory low
        tot = []
        gcols = [c for c in head if c not in ("fov", "cell_ID", "cell") and not c.startswith(("NegPrb", "Negative", "SystemControl", "FalseCode"))]
        for ch in pd.read_csv(z.open(expr_n), usecols=["fov", "cell_ID"] + gcols, chunksize=20000):
            tot.append(pd.DataFrame({"fov": ch.fov, "cell_ID": ch.cell_ID, "total": ch[gcols].sum(1)}))
        T = pd.concat(tot)
        D = M.merge(E, on=["fov", "cell_ID"]).merge(T, on=["fov", "cell_ID"])
        D = D[(D.cell_ID > 0) & (D.total >= 20)].reset_index(drop=True)
        xc = next(c for c in D.columns if re.fullmatch(r"CenterX_global_px", c))
        yc = next(c for c in D.columns if re.fullmatch(r"CenterY_global_px", c))
        xy = D[[xc, yc]].to_numpy(float) * a.um_per_px
        tree = cKDTree(xy)
        dist, nn = tree.query(xy, k=2)
        nb, nd = nn[:, 1], dist[:, 1]
        print(f"{t}: {len(D):,} cells with >= 20 counts; median nearest-neighbour distance {np.median(nd):.1f} um", flush=True)
        for stain, gs in MARKERS.items():
            sc = next((c for c in D.columns if re.fullmatch(rf"Mean[._]{stain}", c, flags=re.I)), None)
            gs = [g for g in gs if g in D.columns]
            if sc is None or not gs:
                print(f"  {stain}: no stain column or no marker genes; skipped")
                continue
            own = D[gs].sum(1).to_numpy(float)
            own_rate = 1000 * own / D.total.to_numpy(float)
            v = D[sc].to_numpy(float)
            pos = v > 0
            lv = np.log1p(v[pos]).reshape(-1, 1)
            gm = GaussianMixture(2, random_state=0).fit(lv)
            d_ash, w_hi, k_hi = ashman(gm)
            hi = np.zeros(len(v), bool)
            hi[pos] = gm.predict(lv) == k_hi
            r_own = spearmanr(v, own_rate).statistic
            r_nbr = spearmanr(v, own_rate[nb]).statistic
            # neighbors that are themselves high in the stain inflate r_nbr; also report it on stain-low neighbors
            low_nb = ~hi[nb]
            r_nbr_low = spearmanr(v[low_nb], own_rate[nb][low_nb]).statistic
            enr_hi = float((own[hi] > 0).mean()) if hi.any() else np.nan
            enr_lo = float((own[~hi] > 0).mean())
            rows.append(dict(tma=t, stain=stain, column=sc, markers="+".join(gs), cells=len(D), ashman_D=d_ash,
                             high_share=float(hi.mean()), rho_own=r_own, rho_neighbour=r_nbr,
                             rho_neighbour_lowonly=r_nbr_low, marker_share_high=enr_hi, marker_share_low=enr_lo))
            print(f"  {stain:5s} D={d_ash:.2f} high {hi.mean():.3f}; rho own {r_own:.3f} vs neighbour {r_nbr:.3f} "
                  f"(stain-low neighbours {r_nbr_low:.3f}); cells with a marker transcript: high {enr_hi:.2f}, "
                  f"low {enr_lo:.2f}", flush=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5, 3.2))
                bins = np.linspace(lv.min(), lv.max(), 80)
                lab = own[pos] > 0
                ax.hist(lv[lab, 0], bins, alpha=0.6, label=f"with {'/'.join(gs[:3])} transcript")
                ax.hist(lv[~lab, 0], bins, alpha=0.6, label="without")
                ax.set_xlabel(f"log(1 + {sc})"); ax.set_ylabel("cells"); ax.legend(fontsize=7)
                ax.set_title(f"{t} {stain}: D = {d_ash:.2f}", fontsize=9)
                fig.tight_layout(); fig.savefig(out / f"{t}_{stain}.png", dpi=120); plt.close(fig)
            except Exception as e:  # plotting is optional
                print(f"  (no plot: {e})")
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False)
    print("\n" + S.drop(columns=["column", "markers"]).round(3).to_string(index=False))
    print(f"\nwrote {out}/summary.csv and the histograms")


if __name__ == "__main__":
    main()
