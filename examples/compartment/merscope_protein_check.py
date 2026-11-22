#!/usr/bin/env python
"""merscope_protein_check.py -- does MERSCOPE antibody co-staining mark the cell itself, or its region?
(the paper calibration; Vizgen MERSCOPE FFPE human uterine cancer, CD45RO / CD45RA co-stain release)

The same test as xenium_protein_check.py (RCC Xenium Protein: CD3E excess 3.9-4.6) and cosmx_protein_check.py
(CosMx stains: about 1). Reads, without changing them, <prefix>_cell_metadata.csv (per-cell stain signal summed over
the cell volume; centers in um) and <prefix>_cell_by_gene.csv (counts). Intensity = <stain>_raw / volume (and the
_high_pass version). Positive cells: two-component mixture on log(intensity), highest component; Ashman D.
For the antibody and, as controls, the Cellbound membrane stains, PolyT and DAPI, against every gene signature:
fold of the signature (counts per 1,000) in stain-high over stain-low cells, the same fold for their nearest
neighbors, and the ratio (cell_specific_excess).
Pair test: each stain-high cell against its nearest stain-LOW neighbor (pair_fold); robust to classes that
sit in dense aggregates, where the nearest neighbor of a T cell is another T cell. A cell-specific immune antibody gives an excess well above 1 for the
immune and T signatures and clearly above the controls. Writes OUT/summary.csv.

  python merscope_protein_check.py --meta .../HumanUterineCancerPatient2-ROCostain_cell_metadata.csv \
      --cbg .../HumanUterineCancerPatient2-ROCostain_cell_by_gene.csv --out .../merscope_uterine_cd45ro/protein_check
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

SIGS = {"T": ["CD3E", "CD3D", "CD3G", "CD2", "CD247", "TRAC", "TRBC1", "TRBC2", "IL7R", "LCK", "ZAP70", "CD8A", "CD8B",
              "GZMK", "CD6", "CD5", "CD96", "CD28", "ITK", "SKAP1", "CD27", "LAT", "CD4"],
        "B": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "PAX5", "CD22", "FCRL1", "FCRL5", "VPREB3", "BLK"],
        "myeloid": ["CD68", "CD163", "CD14", "LYZ", "C1QA", "C1QB", "C1QC", "FCGR3A", "CSF1R", "MS4A7", "AIF1", "TYROBP"],
        "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1", "CLDN4", "ELF3", "KRT5", "KRT17", "MUC16",
                       "PAX8"],
        "immune": ["PTPRC", "CORO1A", "LCP1", "LAPTM5", "CD53", "CD52", "ARHGDIB", "CD37", "CXCR4", "CD48"],
        "PTPRC": ["PTPRC"]}


def mixture_high(lv):
    gm = GaussianMixture(2, random_state=0).fit(lv.reshape(-1, 1))
    o = np.argsort(gm.means_.ravel())
    m, s = gm.means_.ravel()[o], np.sqrt(gm.covariances_.ravel()[o])
    hi = gm.predict(lv.reshape(-1, 1)) == o[-1]
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
    ap.add_argument("--meta", required=True)
    ap.add_argument("--cbg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-counts", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(Path(a.meta).expanduser())
    M = M.rename(columns={M.columns[0]: "cell"})
    M["cell"] = M.cell.astype(str)
    cbg = Path(a.cbg).expanduser()
    head = pd.read_csv(cbg, nrows=0).columns
    cid = head[0]
    gcols = [c for c in head[1:] if not c.lower().startswith("blank")]
    want = sorted({g for gs in SIGS.values() for g in gs if g in gcols})
    parts = []
    for ch in pd.read_csv(cbg, usecols=[cid] + gcols, chunksize=50000, dtype={cid: str}):
        p = ch[[cid] + want].copy()
        p["total"] = ch[gcols].sum(1).to_numpy(float)
        parts.append(p)
    G = pd.concat(parts).rename(columns={cid: "cell"})
    G = G.rename(columns={g: "rna_" + g for g in want})
    print(f"{len(gcols)} genes in the panel; signature genes found: " +
          ", ".join(f"{k} {sum(g in want for g in gs)}/{len(gs)}" for k, gs in SIGS.items()), flush=True)
    D = M.merge(G, on="cell")
    D = D[(D.total >= a.min_counts) & (D.volume > 0)].reset_index(drop=True)
    print(f"{len(M):,} cells in metadata, {len(G):,} in cell_by_gene, {len(D):,} joined with >= {a.min_counts} counts "
          f"(median {D.total.median():.0f}, volume median {D.volume.median():.0f} um3)", flush=True)
    xy = D[["center_x", "center_y"]].to_numpy(float)
    dist, nn = cKDTree(xy).query(xy, k=2)
    nb = nn[:, 1]
    print(f"median nearest-neighbour distance {np.median(dist[:, 1]):.1f} um", flush=True)
    t = D.total.to_numpy(float)
    sig = {}
    for k, gs in SIGS.items():
        cols = ["rna_" + g for g in gs if "rna_" + g in D.columns]
        if cols:
            sig[k] = 1000 * D[cols].sum(1).to_numpy(float) / t
    stains = [c[:-4] for c in D.columns if c.endswith("_raw")]
    ab = [s for s in stains if s.lower().startswith("anti")]
    rows = []
    for st in ab + [s for s in stains if s not in ab]:
        for kind in ("raw", "high_pass"):
            col = f"{st}_{kind}"
            if col not in D.columns:
                continue
            v = D[col].to_numpy(float) / D.volume.to_numpy(float)
            ok = v > 0
            lv = np.log(np.where(ok, v, np.nan))
            dA, hi_ok = mixture_high(lv[ok])
            hi = np.zeros(len(v), bool)
            hi[ok] = hi_ok
            pick, pdist = pair_test(xy, hi)
            for k, z in sig.items():
                f_own = z[hi].mean() / z[~hi].mean() if z[~hi].mean() > 0 else np.nan
                f_nb = z[nb][hi].mean() / z[nb][~hi].mean() if z[nb][~hi].mean() > 0 else np.nan
                rows.append(dict(stain=st, intensity=kind + "/volume", control=st not in ab, signature=k,
                                 cells=len(D), ashman_D=dA, high_share=hi.mean(),
                                 rho_own=spearmanr(v, z).statistic, rho_neighbour=spearmanr(v, z[nb]).statistic,
                                 sig_fold_high_vs_low=f_own, sig_fold_neighbour=f_nb,
                                 cell_specific_excess=f_own / f_nb if f_nb else np.nan,
                                 **pair_stats(z, hi, pick, pdist)))
            r = {x["signature"]: x for x in rows if x["stain"] == st and x["intensity"] == kind + "/volume"}
            print(f"  {st:20s} {kind:9s} D={dA:.2f} high {hi.mean():.3f}; excess " +
                  ", ".join(f"{k} {r[k]['cell_specific_excess']:.2f} ({r[k]['sig_fold_high_vs_low']:.2f}x vs "
                            f"{r[k]['sig_fold_neighbour']:.2f}x)" for k in ("immune", "T", "myeloid", "epithelial")
                            if k in r), flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False)
    W = S[S.intensity == "raw/volume"].pivot_table(index="stain", columns="signature", values="cell_specific_excess")
    print("\ncell_specific_excess (raw/volume), stain x signature:\n" + W.round(2).to_string())
    for kind in ("raw/volume", "high_pass/volume"):
        W = S[S.intensity == kind].pivot_table(index="stain", columns="signature", values="pair_fold")
        print(f"\npair_fold ({kind}): stain-high cells over their nearest stain-low neighbour:\n" + W.round(2).to_string())
    L = S[(S.intensity == "raw/volume") & S.signature.isin(["T", "immune"])][["stain", "signature", "lownb_fold",
                                                                              "pair_dist_um", "pair_cells"]]
    print("\nlow neighbours of high cells against all low cells (class clustering), and pair distance:\n" +
          L.round(2).to_string(index=False))
    print(f"\nwrote {out}/summary.csv")


if __name__ == "__main__":
    main()
