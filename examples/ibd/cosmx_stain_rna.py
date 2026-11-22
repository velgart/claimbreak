r"""cosmx_stain_rna.py -- do CosMx's own stains agree with CosMx RNA on the same cell? (read only)

The control for any CosMx object: Spearman correlation, over cells, of a CosMx immunofluorescence stain (Max.* or
Mean.* in the metadata) with the counts of its genes (log1p per 1,000 counts), on the same cell and on the k-th
nearest neighbor (by CenterX/Y_global_px). Rows of the exprMat and the metadata are joined on (fov, cell_ID).
Run on a raw GEO slide (--raw/--sample) to see what a correctly assembled slide gives; compare with
cellscape_check.py's control on a processed object.

Usage: python cosmx_stain_rna.py --raw RAW/cosmx --sample I0284RI0276R2119 [--kmax 5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = {"PanCK": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT20"], "CD45": ["PTPRC"], "CD3": ["CD3E", "CD3D", "CD3G"],
         "CD68": ["CD68"]}

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--sample", required=True)
ap.add_argument("--kmax", type=int, default=5)
a = ap.parse_args()

from scipy.spatial import cKDTree
from scipy.stats import spearmanr

raw = Path(a.raw).expanduser()
fe = sorted(raw.glob(f"*{a.sample}*_exprMat_file.csv*"))
fm = sorted(raw.glob(f"*{a.sample}*_metadata_file.csv*"))
if not fe or not fm:
    sys.exit(f"exprMat or metadata for {a.sample} not found in {raw}")
genes_needed = sorted({g for gl in PAIRS.values() for g in gl})
head = pd.read_csv(fe[0], nrows=0).columns
import re
CTRL = re.compile(r"^(NegPrb|Negative|NegControl|SystemControl|FalseCode)", re.I)
gcols = [c for c in head if c not in ("fov", "cell_ID", "cell") and not c.startswith("Unnamed") and not CTRL.match(c)]
use = ["fov", "cell_ID"] + [c for c in gcols if c in genes_needed or c.replace(".", "-") in genes_needed]
E = pd.read_csv(fe[0], usecols=lambda c: c in set(use))
tot = pd.read_csv(fe[0], usecols=lambda c: c in set(gcols) or c in ("fov", "cell_ID"))
tot = pd.Series(tot[[c for c in tot.columns if c not in ("fov", "cell_ID")]].sum(axis=1).to_numpy(),
                index=pd.MultiIndex.from_frame(tot[["fov", "cell_ID"]]))
M = pd.read_csv(fm[0])
stain_cols = [c for c in M.columns if c.startswith(("Max.", "Mean."))]
key = pd.MultiIndex.from_frame(M[["fov", "cell_ID"]])
E = E.set_index(["fov", "cell_ID"]).reindex(key)
n = tot.reindex(key).to_numpy(float)
ok = np.isfinite(n) & (n > 0)
print(f"{fe[0].name}: {len(M):,} cells in metadata, {ok.sum():,} matched in exprMat; stains: {stain_cols}", flush=True)
xy = M[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
_, nn = cKDTree(xy).query(xy, k=a.kmax + 1)
rows = []
for ab, gl in PAIRS.items():
    gs = [g for g in E.columns if g in gl or g.replace(".", "-") in gl]
    if not gs:
        continue
    r_ = np.log1p(E[gs].sum(axis=1).to_numpy(float) / np.maximum(n, 1) * 1000)
    for kind in ("Max", "Mean"):
        sc = f"{kind}.{ab}"
        if sc not in M:
            continue
        s = M[sc].to_numpy(float)
        r = {"stain": sc, "rna": "+".join(gs)}
        for k in range(a.kmax + 1):
            t = r_[nn[:, k]]
            m = ok & ok[nn[:, k]] & np.isfinite(s) & np.isfinite(t)
            r["same_cell" if k == 0 else f"nb{k}"] = spearmanr(s[m], t[m]).statistic
        rows.append(r)
print("\nCosMx stain against CosMx RNA, same cell and k-th nearest neighbour:")
print(pd.DataFrame(rows).round(3).to_string(index=False))
