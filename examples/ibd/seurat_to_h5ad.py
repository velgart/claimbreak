r"""seurat_to_h5ad.py -- files from seurat_export.R (+ optional CellScape tables) -> one .h5ad.

Makes the Li et al. 2026 CosMx object usable as a platform-matched reference for `claimbreak compartment label`
(--ref OUT.h5ad --ref-levels L1=ct_major_new) and as protein truth (CellScape per-cell tables keyed by CosMx_Id).
X = raw counts (cells x genes; negative probes and SystemControl codewords dropped), var index = gene symbols,
obs = the Seurat meta.data; CellScape columns are added with the prefix 'cs_' (matched on --cellscape-key).

Usage:
  python seurat_to_h5ad.py EXPORT_DIR --out li2026_cosmx_ref.h5ad \
      [--cellscape "cells_protein_expression(in).csv" "cell_phenotypes(in).csv" --cellscape-key CosMx_Id --id-col cell_id]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("export_dir")
ap.add_argument("--out", required=True)
ap.add_argument("--cellscape", nargs="*", default=[])
ap.add_argument("--cellscape-key", default="CosMx_Id")
ap.add_argument("--id-col", default="cell_id", help="meta.data column holding the CosMx cell id")
a = ap.parse_args()

import anndata as ad
import scipy.io
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

d = Path(a.export_dir).expanduser()
out = Path(a.out).expanduser()
if out.exists():
    sys.exit(f"{out} exists")
M = sp.csr_matrix(scipy.io.mmread(d / "counts.mtx").T)           # cells x genes
genes = (d / "genes.txt").read_text().splitlines()
cells = (d / "cells.txt").read_text().splitlines()
meta = pd.read_csv(d / "meta.csv", index_col=0, low_memory=False)
meta.index = meta.index.astype(str)
meta = meta.reindex(cells)
ctrl = re.compile(r"^(Negative|NegPrb|NegControl|SystemControl|FalseCode|BLANK)", re.I)
keep = np.array([not ctrl.match(g) for g in genes])
if not np.allclose(M.data, np.round(M.data)):
    print("WARNING: counts are not integers (normalised layer?); the labeller expects raw counts", flush=True)
for c in meta.columns:                                             # h5ad-safe obs
    if meta[c].dtype == object:
        meta[c] = meta[c].astype(str)
A = ad.AnnData(M[:, np.where(keep)[0]].astype(np.float32), obs=meta,
               var=pd.DataFrame(index=pd.Index([genes[i] for i in np.where(keep)[0]], name="gene")))
A.var_names_make_unique()
print(f"{A.n_obs:,} cells x {A.n_vars:,} genes ({int((~keep).sum())} control features dropped); "
      f"median counts {np.median(np.asarray(A.X.sum(1)).ravel()):.0f}", flush=True)
for f in a.cellscape:
    if not Path(f).expanduser().exists():
        print(f"WARNING: {f} not found; skipped", flush=True)
        continue
    T = pd.read_csv(f, low_memory=False)
    print(f"{Path(f).name}: columns {list(T.columns)}", flush=True)
    if a.cellscape_key not in T:
        sys.exit(f"{f}: no column {a.cellscape_key}; columns {list(T.columns[:10])}")
    T[a.cellscape_key] = T[a.cellscape_key].astype(str)
    T = T.drop_duplicates(a.cellscape_key).set_index(a.cellscape_key)
    ids = A.obs[a.id_col].astype(str) if a.id_col in A.obs else pd.Series(A.obs_names, index=A.obs_names)
    J = T.reindex(ids.to_numpy())
    J.index = A.obs_names
    print(f"{Path(f).name}: {len(T):,} rows, matched {J.notna().any(axis=1).mean():.3f} of cells", flush=True)
    for c in J.columns:
        v = J[c]
        A.obs[f"cs_{c}"] = v.astype(str) if v.dtype == object else v.to_numpy()
for c in ("ct_major_new", "ct_major", "category"):
    if c in A.obs:
        print(f"\n{c}:\n{A.obs[c].value_counts().to_string()}", flush=True)
A.write_h5ad(out)
print(f"wrote {out}")
