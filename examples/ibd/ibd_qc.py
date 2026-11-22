#!/usr/bin/env python
"""ibd_qc.py -- one row per converted compartment h5ad: depth and how many cells survive a count threshold
(IBD pair). Reads obs only (backed); writes one small CSV.

  python ibd_qc.py --h5ad '~/claimbreak_data/IBD/h5ad/*.h5ad' --out ~/claimbreak_results/ibd_qc.csv
"""
import argparse
import glob
import os

import anndata as ad
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="10,20,50")
    a = ap.parse_args()
    th = [int(x) for x in a.thresholds.split(",")]
    rows = []
    for f in sorted(glob.glob(os.path.expanduser(a.h5ad))):
        A = ad.read_h5ad(f, backed="r")
        o = A.obs
        w, n = o["n_whole"].to_numpy(float), o["n_nucleus"].to_numpy(float)
        r = dict(file=os.path.basename(f), panel=str(o["panel"].iloc[0]), unit=str(o["unit"].iloc[0]),
                 genes=A.n_vars, cells=A.n_obs, zero=float(np.mean(w == 0)),
                 med_whole=float(np.median(w)), q90_whole=float(np.quantile(w, 0.9)),
                 med_nucleus=float(np.median(n)), share_nucleus=float(n.sum() / max(w.sum(), 1)))
        for t in th:
            r[f"cells_ge{t}"] = int((w >= t).sum())
            r[f"med_whole_ge{t}"] = float(np.median(w[w >= t])) if (w >= t).any() else np.nan
        rows.append(r)
        A.file.close()
    T = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(a.out))), exist_ok=True)
    T.to_csv(os.path.expanduser(a.out), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
    print(T.drop(columns=[c for c in T.columns if c.startswith("med_whole_ge")]).to_string(index=False))
    t = th[1] if len(th) > 1 else th[0]
    print(f"\nby panel (cells with >= {t} counts):")
    print(T.groupby("panel").agg(files=("file", "size"), cells=("cells", "sum"),
                                 kept=(f"cells_ge{t}", "sum"), med_whole=("med_whole", "median")).to_string())


if __name__ == "__main__":
    main()
