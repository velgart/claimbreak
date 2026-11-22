#!/usr/bin/env python
"""liver_label_check.py -- are the liver labels plausible? A reference-free check of label runs on the CosMx pieces
. Read only.

For each label run (a `claimbreak compartment label` output folder), piece and arm:
  composition    share of each label among labeled cells
  marker_agree   share of labeled cells whose label is also the class whose canonical markers (below; panel genes
                 only) take the largest share of the cell's whole-cell counts, relative to the tissue average (a
                 crude, reference-free call: it needs no profiles and no platform factor, so it cannot collapse the
                 way a fitted classifier can; it is noisy for small classes)
  marker_calls   composition of the marker call itself (what the tissue looks like by canonical markers)
Writes OUT/label_check.csv (long) and prints, per run, the dominant labels and marker agreement per piece.

  python liver_label_check.py --h5ad ~/claimbreak_data/data/liver_mash/h5ad_qc \
      --runs ~/claimbreak_results/liver_mash/labels,~/claimbreak_results/liver_mash/grid/g0 --out ~/claimbreak_results/liver_mash/grid
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

MARKERS = {
    "Hepatocyte": ["ALB", "APOA1", "APOA2", "APOC3", "APOH", "CYP3A4", "CYP2E1", "SERPINA1", "HP", "TF", "FGB", "AHSG"],
    "Cholangiocyte": ["KRT19", "KRT7", "EPCAM", "SOX9", "SPP1", "CFTR", "ANXA4"],
    "Endothelial": ["PECAM1", "VWF", "CLEC4G", "STAB2", "FCN3", "CLEC1B", "CDH5", "KDR", "ENG"],
    "Stellate_Fib": ["ACTA2", "COL1A1", "COL1A2", "COL3A1", "DCN", "PDGFRB", "RGS5", "LUM"],
    "Myeloid": ["CD68", "CD163", "MARCO", "C1QA", "C1QB", "LYZ", "CD14", "VSIG4", "FCGR3A"],
    "T_NK": ["CD3E", "CD3D", "CD2", "NKG7", "GZMA", "GNLY", "CD8A", "IL7R"],
    "B": ["MS4A1", "CD79A", "CD19", "BANK1"],
    "Plasma": ["JCHAIN", "MZB1", "IGHG1", "IGKC", "IGHA1", "XBP1"],
}


def marker_call(A):
    import scipy.sparse as sp
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    tot = np.asarray(X.sum(1)).ravel()
    var = pd.Index(A.var_names.astype(str))
    cls, S, used = [], [], {}
    for c, genes in MARKERS.items():
        g = [x for x in genes if x in var]
        if not g:
            continue
        used[c] = g
        s = np.asarray(X[:, var.get_indexer(g)].sum(1)).ravel() / np.maximum(tot, 1)
        S.append(s / max(s.mean(), 1e-9))                   # relative to the tissue average
        cls.append(c)
    S = np.column_stack(S)
    call = np.array(cls)[S.argmax(1)]
    call[(tot < 10) | (S.max(1) < 1)] = ""                    # too few counts, or no class above its tissue average
    return call, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True, help="folder with the piece h5ad files")
    ap.add_argument("--runs", required=True, help="comma-separated label folders")
    ap.add_argument("--arms", default="nucleus,whole,nucB,wholeB")
    ap.add_argument("--level", default="L1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import anndata as ad
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    runs = [os.path.expanduser(r) for r in a.runs.split(",") if r]
    rows = []
    pieces = sorted(glob.glob(os.path.join(os.path.expanduser(a.h5ad), "*_p*.h5ad")))
    for f in pieces:
        s = os.path.basename(f)[:-5]
        A = ad.read_h5ad(f)
        call, used = marker_call(A)
        mc = pd.Series(call[call != ""]).value_counts(normalize=True)
        for k, v in mc.items():
            rows.append(dict(run="markers", piece=s, arm="whole", metric="composition", label=k, value=float(v)))
        cid = pd.Index(A.obs_names.astype(str))
        for r in runs:
            lf = os.path.join(r, f"{s}.labels.csv.gz")
            if not os.path.exists(lf):
                continue
            L = pd.read_csv(lf, keep_default_na=False, dtype={"cell_id": str}, low_memory=False).set_index("cell_id")
            L = L.reindex(cid)
            for arm in a.arms.split(","):
                col = f"{a.level}_{arm}"
                if col not in L:
                    continue
                lab = L[col].fillna("").to_numpy()
                ok = lab != ""
                comp = pd.Series(lab[ok]).value_counts(normalize=True)
                for k, v in comp.items():
                    rows.append(dict(run=os.path.basename(r.rstrip("/")), piece=s, arm=arm, metric="composition",
                                     label=k, value=float(v)))
                both = ok & (call != "")
                rows.append(dict(run=os.path.basename(r.rstrip("/")), piece=s, arm=arm, metric="marker_agree",
                                 label="", value=float((lab[both] == call[both]).mean()) if both.any() else np.nan))
        print(f"{s}: marker classes used {', '.join(f'{c}({len(g)})' for c, g in used.items())}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(os.path.expanduser(a.out), "label_check.csv"), index=False)
    short = lambda p: p.replace("GSM93517", "").replace("_Leuven", "L")
    C = T[T.metric == "composition"]
    print("\nmarker calls (reference-free), composition per piece:")
    print(C[C.run == "markers"].pivot_table(index="piece", columns="label", values="value").rename(index=short)
          .round(2).fillna(0).to_string())
    for r in [x for x in T.run.unique() if x != "markers"]:
        Q = T[(T.run == r)]
        ag = Q[Q.metric == "marker_agree"].pivot_table(index="piece", columns="arm", values="value").rename(index=short)
        w = Q[(Q.metric == "composition") & (Q.arm == "whole")].pivot_table(index="piece", columns="label",
                                                                             values="value").rename(index=short)
        print(f"\n== {r}: whole-cell composition and agreement with the marker call per arm")
        print(pd.concat([w.round(2).fillna(0), ag.round(2).add_prefix("agree_")], axis=1).to_string())
    print(f"\nwrote {os.path.join(a.out, 'label_check.csv')}")


if __name__ == "__main__":
    main()
