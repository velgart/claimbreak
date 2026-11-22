#!/usr/bin/env python
"""cosmx_pairs.py -- the two copies of each transcript in a doubled CosMx tx_file: where do they differ, and which
compartment call is more plausible? (IBD pair). Reads one slide; writes nothing.

In the 1k tx_files of GSE312415 every transcript appears twice, the two copies agree on position, z, gene and cell, but
in ~15% of pairs not on CellComp. This script pairs the copies (copy 1 = earlier row in the file) and reports:
  1. CellComp of copy 1 x CellComp of copy 2;
  2. how the copies are laid out in the file (blocks per FOV? distance in rows);
  3. per copy: share of in-cell transcripts called Nuclear, Cytoplasm, Membrane;
  4. per copy and call: distance from the cell center, in units of the cell's radius sqrt(Area/pi). A sound call set
     puts Nuclear closest to the center and Membrane farthest;
  5. the same for pairs whose calls disagree, by combination.

  python cosmx_pairs.py --raw ~/claimbreak_data/IBD/raw/cosmx --sample I0284RI0276R2119
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

KEYS = ["fov", "x_local_px", "y_local_px", "x_global_px", "y_global_px", "z", "target"]


def find(raw, s, suf):
    f = glob.glob(os.path.join(raw, f"{s}{suf}")) + glob.glob(os.path.join(raw, f"{s}{suf}.gz"))
    if not f:
        sys.exit(f"{s}{suf} not found")
    return f[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sample", required=True)
    a = ap.parse_args()
    T = pd.read_csv(find(a.raw, a.sample, "_tx_file.csv"),
                    usecols=lambda c: c in KEYS + ["cell", "CellComp"])
    T["CellComp"] = T["CellComp"].fillna("None").astype(str)
    T["row"] = np.arange(len(T))
    meta = pd.read_csv(find(a.raw, a.sample, "_metadata_file.csv"))
    meta["cell"] = meta["cell"].astype(str)
    M = meta.set_index("cell")[["CenterX_global_px", "CenterY_global_px", "Area"]].reindex(T["cell"].astype(str))
    T["r"] = (np.hypot(T.x_global_px.to_numpy() - M.CenterX_global_px.to_numpy(),
                       T.y_global_px.to_numpy() - M.CenterY_global_px.to_numpy()) / np.sqrt(M.Area.to_numpy() / np.pi))
    g = T.groupby(KEYS, sort=False)
    T["k"] = g.cumcount()
    T["n"] = g["row"].transform("size")
    print(f"{len(T):,} rows; copies per transcript: {T.groupby(KEYS, sort=False).size().value_counts().sort_index().to_dict()}")
    P = T[T.n == 2]
    c1 = P[P.k == 0].set_index(KEYS)
    c2 = P[P.k == 1].set_index(KEYS).loc[c1.index]
    print(f"pairs: {len(c1):,}")
    print("\n1. CellComp, copy 1 (rows) x copy 2 (columns):")
    X = pd.crosstab(c1["CellComp"].to_numpy(), c2["CellComp"].to_numpy())
    print(X.to_string())
    print(f"   agree: {np.trace(X.reindex(index=X.columns, fill_value=0).to_numpy()) / X.to_numpy().sum():.4f}")
    print("\n2. layout:")
    d = c2["row"].to_numpy() - c1["row"].to_numpy()
    print(f"   rows between copies: median {np.median(d):,.0f}, 5% {np.quantile(d, .05):,.0f}, 95% {np.quantile(d, .95):,.0f}")
    F = pd.DataFrame({"fov": c1.index.get_level_values("fov"), "r1": c1["row"].to_numpy(), "r2": c2["row"].to_numpy()})
    B = F.groupby("fov").agg(r1_max=("r1", "max"), r2_min=("r2", "min"), r1_min=("r1", "min"), r2_max=("r2", "max"))
    print(f"   FOVs where all copy-1 rows come before all copy-2 rows: {(B.r1_max < B.r2_min).sum()} of {len(B)}")
    print(B.head(5).to_string())
    rows = []
    for nm, C in (("copy 1", c1), ("copy 2", c2)):
        C = C[(C["CellComp"] != "None") & C["r"].notna()]
        for comp, S in C.groupby("CellComp"):
            rows.append(dict(copy=nm, call=comp, n=len(S), share=len(S) / len(C), r_q25=S.r.quantile(.25),
                             r_median=S.r.median(), r_q75=S.r.quantile(.75)))
    print("\n3-4. per copy and call (r = distance from the cell centre / cell radius):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    J = pd.DataFrame({"c1": c1["CellComp"].to_numpy(), "c2": c2["CellComp"].to_numpy(), "r": c1["r"].to_numpy()})
    J = J[(J.c1 != "None") & (J.c2 != "None") & J.r.notna()]
    J["pair"] = np.where(J.c1 == J.c2, "agree " + J.c1, J.c1 + " (1) / " + J.c2 + " (2)")
    print("\n5. pairs by combination of calls (r is the same for both copies):")
    print(J.groupby("pair").r.describe()[["count", "25%", "50%", "75%"]].round(3).to_string())


if __name__ == "__main__":
    main()
