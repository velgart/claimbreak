#!/usr/bin/env python
# make_synth.py OUTDIR [random] -- synthetic test data for the correction chain (not for results). Then:
#   claimbreak compartment label --ref OUTDIR/ref.h5ad --inputs OUTDIR/h5ad/*.h5ad --out OUTDIR/labels --levels L2m --arms whole,nucleus
"""Synthetic lung-like data: reference h5ad, one section h5ad, transcripts parquet; T cells next to tumor receive
tumor transcripts in the cytoplasm (spill-over)."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from pathlib import Path
from scipy.spatial import cKDTree
import pyarrow as pa, pyarrow.parquet as pq
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
types = ["Tumour", "T_NK", "Myeloid", "Fibro_muscle"]
G = 80
genes = [f"G{i:02d}" for i in range(G)]
prof = np.full((4, G), 0.2)
for k in range(4):
    prof[k, k*20:(k+1)*20] = rng.gamma(2, 3, 20)
prof /= prof.sum(1, keepdims=True)
# reference: 600 cells per type, 2 donors, tumor named Tu_L1/Tu_L2
rows, lab, don = [], [], []
for k, t in enumerate(types):
    for d in (1, 2):
        n = 300
        rows.append(rng.multinomial(400, prof[k], size=n)); don += [str(d)] * n
        lab += [f"Tu_L{d}" if t == "Tumour" else t] * n
R = ad.AnnData(X=sp.csr_matrix(np.vstack(rows).astype(np.float32)),
               obs=pd.DataFrame({"Level2": lab, "donor_id": don}, index=[f"r{i}" for i in range(len(lab))]),
               var=pd.DataFrame(index=genes))
R.write_h5ad(root / "ref.h5ad")
# section: 40x40 grid, tumor block on the left, immune scattered
n_side = 50
gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
x = gx.ravel() * 10.0 + rng.normal(0, 1, gx.size); y = gy.ravel() * 10.0 + rng.normal(0, 1, gx.size)
if len(sys.argv) > 2 and sys.argv[2] == "random":      # irregular layout (registration tests need it)
    x, y = rng.uniform(0, 500, gx.size), rng.uniform(0, 500, gx.size)
n = len(x)
t = np.where(x < 250, 0, 3)
r = rng.random(n); t[r < 0.2] = 1; t[(r >= 0.2) & (r < 0.3)] = 2
_, nn = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=7)
ntum = (t[nn[:, 1:]] == 0).sum(1)
size = np.where(t == 0, 120, np.where(t == 3, 90, 50))
nuc = np.vstack([rng.multinomial(int(s * 0.4), prof[k]) for s, k in zip(size, t)])
cyt = np.vstack([rng.multinomial(int(s * 0.6), prof[k]) for s, k in zip(size, t)])
spill = (t != 0) & (ntum > 0)
for i in np.where(spill)[0]:
    cyt[i] += rng.multinomial(int(12 * ntum[i]), prof[0])
cid = [f"c{i}" for i in range(n)]
A = ad.AnnData(X=sp.csr_matrix((nuc + cyt).astype(np.float32)),
               obs=pd.DataFrame({"x": x, "y": y, "sample": "GSM1_NSCLC_lung_L1_L1", "donor": "L1", "panel": "Lung",
                                 "section": "Lung:x", "cell_area": size.astype(float), "nucleus_area": 20.0},
                                index=cid), var=pd.DataFrame(index=genes))
A.layers["nucleus"] = sp.csr_matrix(nuc.astype(np.float32)); A.layers["cyto"] = sp.csr_matrix(cyt.astype(np.float32))
A.obs["n_whole"] = np.asarray(A.X.sum(1)).ravel(); A.obs["n_nucleus"] = nuc.sum(1)
(root / "h5ad").mkdir(exist_ok=True); A.write_h5ad(root / "h5ad" / "GSM1_NSCLC_lung_L1_L1.h5ad")
pd.DataFrame({"cell": cid, "true": np.array(types)[t]}).to_csv(root / "truth.csv", index=False)
# transcripts parquet
recs = []
for i in range(n):
    for M, inn in ((nuc[i], 1), (cyt[i], 0)):
        g = np.repeat(np.arange(G), M)
        rad = 2.0 if inn else 5.0
        recs.append(pd.DataFrame({"cell_id": cid[i], "feature_name": np.array(genes)[g], "qv": 30.0,
                                  "overlaps_nucleus": inn, "x_location": x[i] + rng.normal(0, rad, len(g)),
                                  "y_location": y[i] + rng.normal(0, rad, len(g)), "z_location": rng.normal(5, 1, len(g))}))
T = pd.concat(recs, ignore_index=True)
T = pd.concat([T, pd.DataFrame({"cell_id": "UNASSIGNED", "feature_name": ["NegControlProbe_1", "G00"], "qv": 30.0,
                                "overlaps_nucleus": 0, "x_location": 1.0, "y_location": 1.0, "z_location": 5.0})])
(root / "raw").mkdir(exist_ok=True)
pq.write_table(pa.Table.from_pandas(T, preserve_index=False), root / "raw" / "GSM1_NSCLC_lung_L1_L1_transcripts.parquet")
print(n, "cells", len(T), "transcripts; spill cells", spill.sum())
