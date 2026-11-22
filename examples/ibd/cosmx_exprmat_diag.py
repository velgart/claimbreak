#!/usr/bin/env python
"""cosmx_exprmat_diag.py -- why do whole-cell totals from a CosMx tx_file disagree with the deposited exprMat?
(IBD pair). Reads one slide's tx_file, metadata_file and exprMat_file; writes nothing.

Reports:
  1. columns of exprMat that are not targets of the tx_file, and targets of the tx_file that are not exprMat columns;
  2. per cell: exprMat total vs tx_file totals counted several ways (genes only / with controls; with or without
     CellComp 'None'), the share of cells equal under each, and the median ratio exprMat / tx;
  3. per gene: exprMat column sum vs tx_file count (in cells), the genes with the largest disagreement.

  python cosmx_exprmat_diag.py --raw ~/claimbreak_data/IBD/raw/cosmx --sample I0284RI0276R2119
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

CONTROL = re.compile(r"^(NegControl|NegPrb|Negative|SystemControl|FalseCode|BLANK|Blank|Unassigned|antisense|"
                     r"DeprecatedCodeword|UnassignedCodeword|IntergenicRegion)", re.I)


def find(raw, s, suf):
    for x in ("", ".gz"):
        f = glob.glob(os.path.join(raw, f"{s}{suf}{x}"))
        if f:
            return f[0]
    sys.exit(f"{s}{suf} not found in {raw}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--batch", type=int, default=10_000_000)
    a = ap.parse_args()
    meta = pd.read_csv(find(a.raw, a.sample, "_metadata_file.csv"))
    E = pd.read_csv(find(a.raw, a.sample, "_exprMat_file.csv"))
    print(f"metadata: {len(meta):,} rows; columns {list(meta.columns[:8])} ...")
    print(f"exprMat: {E.shape[0]:,} rows x {E.shape[1]:,} columns; first columns {list(E.columns[:6])}")
    idcols = [c for c in E.columns if not pd.api.types.is_numeric_dtype(E[c]) or c in ("fov", "cell_ID", "cell")
              or c.startswith("Unnamed")]
    other = [c for c in E.columns if c not in idcols]
    print(f"exprMat id-like columns: {idcols}")
    print(f"exprMat value dtype(s): {sorted(set(str(E[c].dtype) for c in other[:50]))}; "
          f"non-integer values: {bool((E[other[:50]] % 1 != 0).any().any())}")
    # cell key -> metadata cell id
    if "cell" in E:
        key = E["cell"].astype(str)
    else:
        mk = pd.Series(meta["cell"].astype(str).to_numpy(), index=pd.MultiIndex.from_arrays([meta["fov"], meta["cell_ID"]]))
        key = pd.Series(mk.reindex(pd.MultiIndex.from_arrays([E["fov"], E["cell_ID"]])).to_numpy())
    print(f"exprMat rows matched to metadata cells: {key.notna().sum():,}; with cell_ID 0: "
          f"{int((E['cell_ID'] == 0).sum()) if 'cell_ID' in E else 'n/a'}")
    # tx_file: counts per (cell, target, comp)
    parts = []
    targets_all = set()
    for d in pd.read_csv(find(a.raw, a.sample, "_tx_file.csv"), chunksize=a.batch,
                         usecols=lambda c: c in ("cell", "target", "CellComp", "fov", "cell_ID")):
        targets_all |= set(d["target"].astype(str).unique())
        parts.append(d.groupby(["cell", "target", "CellComp"], observed=True).size())
        print(f"  tx chunk: {len(d):,} rows", flush=True)
    tx = pd.concat(parts).groupby(level=[0, 1, 2]).sum().rename("n").reset_index()
    tx["cell"] = tx["cell"].astype(str)
    tx["ctrl"] = tx["target"].astype(str).str.match(CONTROL)
    print(f"CellComp values: {tx.groupby('CellComp').n.sum().to_dict()}")
    gcols = [c for c in other if not CONTROL.match(c)]
    ccols = [c for c in other if CONTROL.match(c)]
    print(f"\n1. exprMat gene columns {len(gcols)}, control columns {len(ccols)}; tx targets {len(targets_all)} "
          f"({sum(bool(CONTROL.match(t)) for t in targets_all)} controls)")
    only_e = sorted(set(other) - targets_all)
    only_t = sorted(targets_all - set(other))
    print(f"   in exprMat only ({len(only_e)}): {only_e[:20]}")
    print(f"   in tx_file only ({len(only_t)}): {only_t[:20]}")
    # 2. per-cell totals
    tE_g = pd.Series(E[gcols].sum(axis=1).to_numpy(), index=key).groupby(level=0).sum()
    tE_all = pd.Series(E[other].sum(axis=1).to_numpy(), index=key).groupby(level=0).sum()
    ways = {
        "genes, no None": tx[~tx.ctrl & (tx.CellComp != "None")],
        "genes, with None": tx[~tx.ctrl],
        "genes+controls, no None": tx[tx.CellComp != "None"],
        "genes+controls, with None": tx,
    }
    print("\n2. per cell (cells in both):")
    for nm, sub in ways.items():
        tT = sub.groupby("cell").n.sum()
        for enm, tE in (("exprMat genes", tE_g), ("exprMat all columns", tE_all)):
            c = tE.index.intersection(tT.index)
            x, y = tE.loc[c].to_numpy(float), tT.loc[c].to_numpy(float)
            r = np.median(x[y > 0] / y[y > 0]) if (y > 0).any() else np.nan
            print(f"   tx {nm:26s} vs {enm:20s}: {len(c):,} cells, equal {np.mean(x == y):.4f}, "
                  f"median ratio {r:.4f}, median diff {np.median(x - y):+.1f}")
    # exprMat cells absent from tx and vice versa
    tT = ways["genes, no None"].groupby("cell").n.sum()
    print(f"   cells with exprMat total > 0: {(tE_g > 0).sum():,}; with tx counts: {len(tT):,}; "
          f"overlap {len(tE_g[tE_g > 0].index.intersection(tT.index)):,}")
    print("   example cells (exprMat genes total, tx genes no None):")
    c = tE_g.index.intersection(tT.index)[:8]
    print(pd.DataFrame({"exprMat": tE_g.loc[c], "tx": tT.loc[c]}).to_string())
    # 3. per gene
    gE = E[gcols].sum(axis=0)
    gT = ways["genes, no None"][ways["genes, no None"].cell.isin(set(key.dropna()))].groupby("target").n.sum()
    G = pd.DataFrame({"exprMat": gE, "tx": gT}).fillna(0)
    G["ratio"] = G.exprMat / G.tx.replace(0, np.nan)
    print(f"\n3. per gene: median ratio exprMat/tx {G.ratio.median():.4f}; genes equal {np.mean(G.exprMat == G.tx):.4f}")
    G["absdiff"] = (G.exprMat - G.tx).abs()
    print(G.sort_values("absdiff", ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
