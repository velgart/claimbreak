#!/usr/bin/env python
"""cosmx_tx_dups.py -- are transcripts repeated in a CosMx tx_file? (IBD pair)
Reads one slide's tx_file whole (a 1k slide of ~8 M rows needs ~2 GB of memory); writes nothing.

Reports: the columns; whether coordinates are whole pixels; rows, distinct rows, and distinct transcripts by
(fov, position, z, target); how often each transcript occurs; whether the copies agree on cell and CellComp; whether the
copies sit next to each other or the file is two stacked halves; the duplicate share per FOV.

  python cosmx_tx_dups.py --raw ~/claimbreak_data/IBD/raw/cosmx --sample I0284RI0276R2119
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

KEYS = ["fov", "x_local_px", "y_local_px", "x_global_px", "y_global_px", "z", "target"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sample", required=True)
    a = ap.parse_args()
    f = (glob.glob(os.path.join(a.raw, f"{a.sample}_tx_file.csv")) +
         glob.glob(os.path.join(a.raw, f"{a.sample}_tx_file.csv.gz")))
    if not f:
        sys.exit("tx_file not found")
    T = pd.read_csv(f[0])
    if "CellComp" in T:                       # empty CellComp (outside cells) is a value, not missing
        T["CellComp"] = T["CellComp"].fillna("None")
    print(f"{f[0]}: {len(T):,} rows; columns {list(T.columns)}")
    print(T.head(4).to_string())
    for c in ("x_local_px", "y_local_px", "z"):
        if c in T:
            print(f"  {c}: whole numbers {bool((T[c] % 1 == 0).all())}, {T[c].nunique():,} distinct values")
    keys = [c for c in KEYS if c in T]
    print(f"\nrows {len(T):,}; distinct whole rows {len(T.drop_duplicates()):,}; "
          f"distinct by ({', '.join(keys)}) {len(T.drop_duplicates(keys)):,}")
    g = T.groupby(keys, sort=False).size()
    print("copies per transcript:", g.value_counts().sort_index().to_dict())
    dup = T.duplicated(keys, keep=False)
    if dup.any():
        D = T[dup]
        for c in ("cell", "cell_ID", "CellComp"):
            if c in D:
                agree = D.groupby(keys, sort=False)[c].nunique().eq(1).mean()
                print(f"  copies agree on {c}: {agree:.4f}")
        h = len(T) // 2
        if len(T) % 2 == 0:
            same = (T.iloc[:h].reset_index(drop=True)[keys] == T.iloc[h:].reset_index(drop=True)[keys]).all(axis=1).mean()
            print(f"  first half row-by-row equal to second half: {same:.4f}")
        adj = (T[keys].iloc[1:].reset_index(drop=True) == T[keys].iloc[:-1].reset_index(drop=True)).all(axis=1)
        print(f"  rows equal to the row before: {adj.sum():,}")
        first = T.duplicated(keys, keep="first")
        P = pd.DataFrame({"rows": T.groupby("fov").size(), "later_copies": first.groupby(T["fov"]).sum()})
        P["share"] = P.later_copies / P.rows
        print("  per FOV (share of rows that are later copies):", P.share.describe()[["min", "50%", "max"]].round(4).to_dict())
        print(P.head(10).to_string())


if __name__ == "__main__":
    main()
