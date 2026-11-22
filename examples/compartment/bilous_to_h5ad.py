"""One .h5ad per Xenium section of GSE311609 (Bilous et al., Nat Methods 2026), with whole-cell and nucleus-only counts
of the same cells.

From transcripts.parquet(.gz) of a section (standard Xenium output): transcripts with qv >= --min-qv (default 20, as
Xenium's own cell_feature_matrix) on gene probes only (control and unassigned codewords dropped), assigned to a cell.
  X                  whole-cell counts (all transcripts assigned to the cell by the default segmentation)
  layers['nucleus']  counts of the transcripts of that cell with overlaps_nucleus == 1
  layers['cyto']     X - nucleus
obs: sample, donor, panel, section, x, y (cell centroid, um), cell_area, nucleus_area (from cells.parquet),
     n_whole, n_nucleus. var: gene names. uns['source']: input file names, sizes and SHA-256, filters.
Check: whole-cell counts are compared with cell_feature_matrix.h5 (Gene Expression features) when it is given; the
share of cells whose totals agree exactly is printed and stored (uns['check']).

The input folder is only read. The .gz files are decompressed to a temporary file in --tmp (deleted afterwards).
  python bilous_to_h5ad.py --raw ~/claimbreak_data/Bilous/raw --sample GSM9509145_NSCLC_lung_L1_L1 --out ~/claimbreak_data/Bilous_h5ad \
      --tmp ~/claimbreak_data/Bilous_tmp
Writes OUT/<sample>.h5ad; refuses to overwrite.
"""
import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

CONTROL = re.compile(r"^(NegControl|BLANK|Blank|Unassigned|Intergenic|DeprecatedCodeword|antisense|UnassignedCodeword)",
                     re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--sample", required=True, help="file prefix, e.g. GSM9509145_NSCLC_lung_L1_L1")
ap.add_argument("--out", required=True)
ap.add_argument("--tmp", required=True)
ap.add_argument("--min-qv", type=float, default=20.0)
ap.add_argument("--batch", type=int, default=5_000_000)
a = ap.parse_args()

raw, out, tmp = (Path(p).expanduser().resolve() for p in (a.raw, a.out, a.tmp))
if out == raw or raw in out.parents or out in raw.parents or tmp == raw or raw in tmp.parents:
    sys.exit("--out and --tmp must be outside the raw data folder")
dst = out / f"{a.sample}.h5ad"
if dst.exists():
    sys.exit(f"{dst} exists; not overwritten")
out.mkdir(parents=True, exist_ok=True)
tmp.mkdir(parents=True, exist_ok=True)


def find(stem):
    for suf in ("", ".gz"):
        p = raw / f"{a.sample}_{stem}{suf}"
        if p.exists():
            return p
    return None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def plain(p):
    """A seekable uncompressed copy in tmp for .gz inputs (parquet needs seeking)."""
    if p.suffix != ".gz":
        return p, False
    q = tmp / p.name[:-3]
    with gzip.open(p, "rb") as fi, open(q, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 24)
    return q, True


ftr, fcells, fmat = find("transcripts.parquet"), find("cells.parquet"), find("cell_feature_matrix.h5")
if ftr is None or fcells is None:
    sys.exit(f"transcripts or cells parquet missing for {a.sample} in {raw}")
src = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in (ftr, fcells, fmat) if p is not None}
m = re.match(r"GSM\d+_(NSCLC|breast)_(5k|chuvio|lung|breast)_(\w+?)_(\w+)$", a.sample)
panel = {"5k": "5K", "chuvio": "Custom IO", "lung": "Lung", "breast": "Breast"}.get(m.group(2), m.group(2)) if m else ""
donor = m.group(3) if m else ""
print(f"{a.sample}: panel {panel}, donor {donor}", flush=True)

# cells
pc, rmc = plain(fcells)
cells = pq.read_table(pc).to_pandas()
if rmc:
    pc.unlink()
cells["cell_id"] = cells["cell_id"].astype(str)
ci = pd.Index(cells["cell_id"])
if not ci.is_unique:
    sys.exit("cell_id not unique in cells.parquet")

# transcripts, in batches
pt, rmt = plain(ftr)
pf = pq.ParquetFile(pt)
cols = set(pf.schema_arrow.names)
need = {"cell_id", "feature_name", "qv", "overlaps_nucleus"}
if not need <= cols:
    sys.exit(f"transcripts.parquet lacks {sorted(need - cols)}; columns: {sorted(cols)}")
genes, gidx = [], {}
W = N = None
n_tot = n_kept = 0
for rb in pf.iter_batches(batch_size=a.batch, columns=sorted(need)):
    d = rb.to_pandas()
    n_tot += len(d)
    f = d["feature_name"].astype(str)
    cid = d["cell_id"].astype(str)
    keep = (d["qv"].to_numpy() >= a.min_qv) & ~f.str.match(CONTROL).to_numpy() & cid.isin(ci).to_numpy()
    if not keep.any():
        continue
    f, cid, nuc = f[keep], cid[keep], d["overlaps_nucleus"].to_numpy()[keep].astype(bool)
    for g in pd.unique(f):
        if g not in gidx:
            gidx[g] = len(genes); genes.append(g)
    r = ci.get_indexer(cid)
    c = f.map(gidx).to_numpy()
    n_kept += len(r)
    shape = (len(ci), 60000)
    w = sp.coo_matrix((np.ones(len(r), np.int32), (r, c)), shape=shape).tocsr()
    nn = sp.coo_matrix((np.ones(int(nuc.sum()), np.int32), (r[nuc], c[nuc])), shape=shape).tocsr()
    W = w if W is None else W + w
    N = nn if N is None else N + nn
    print(f"  {n_tot:,} transcripts read, {n_kept:,} kept", flush=True)
if rmt:
    pt.unlink()
G = len(genes)
W, N = W[:, :G].tocsr(), N[:, :G].tocsr()
order = np.argsort(genes)
genes = np.array(genes)[order]
W, N = W[:, order], N[:, order]

obs = pd.DataFrame(index=pd.Index(ci.astype(str), name="cell_id"))
obs["sample"], obs["donor"], obs["panel"] = a.sample, donor, panel
obs["section"] = f"{panel}:{a.sample.split('_', 1)[1]}"
xc = "x_centroid" if "x_centroid" in cells else None
obs["x"] = cells["x_centroid"].to_numpy(float) if xc else np.nan
obs["y"] = cells["y_centroid"].to_numpy(float) if xc else np.nan
for c in ("cell_area", "nucleus_area"):
    obs[c] = cells[c].to_numpy(float) if c in cells else np.nan
obs["n_whole"] = np.asarray(W.sum(1)).ravel()
obs["n_nucleus"] = np.asarray(N.sum(1)).ravel()

check = {}
if fmat is not None:
    import h5py
    pm, rmm = plain(fmat)
    with h5py.File(pm, "r") as h:
        mtx = h["matrix"]
        bc = np.array(mtx["barcodes"]).astype(str)
        ft = mtx["features"]
        names = np.array(ft["name"]).astype(str)
        ftype = np.array(ft["feature_type"]).astype(str)
        M = sp.csc_matrix((mtx["data"][:], mtx["indices"][:], mtx["indptr"][:]), shape=tuple(mtx["shape"][:]))
    if rmm:
        pm.unlink()
    ge = ftype == "Gene Expression"
    tot_h5 = pd.Series(np.asarray(M[ge].sum(0)).ravel(), index=bc).reindex(obs.index)
    ok = np.isfinite(tot_h5.to_numpy())
    agree = float(np.mean(tot_h5.to_numpy()[ok] == obs["n_whole"].to_numpy()[ok])) if ok.any() else float("nan")
    genes_h5 = set(names[ge])
    check = {"cells_in_h5": int(ok.sum()), "share_cells_total_equal": agree,
             "genes_here_not_in_h5": sorted(set(genes) - genes_h5)[:20], "genes_h5_not_here": len(genes_h5 - set(genes))}
    print(f"check vs cell_feature_matrix.h5: {ok.sum():,} cells matched; totals equal for {agree:.4f}; "
          f"genes h5 not here {check['genes_h5_not_here']}", flush=True)

import anndata as ad
A = ad.AnnData(X=W.astype(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
A.layers["nucleus"] = N.astype(np.float32)
A.layers["cyto"] = (W - N).astype(np.float32)
A.uns["source"] = json.dumps({"files": src, "min_qv": a.min_qv, "control_regex": CONTROL.pattern,
                              "transcripts_read": n_tot, "transcripts_kept": n_kept})
A.uns["check"] = json.dumps(check)
for c in ("sample", "donor", "panel", "section"):
    A.obs[c] = A.obs[c].astype("category")
A.write_h5ad(dst, compression="gzip")
print(f"wrote {dst}: {A.n_obs:,} cells x {A.n_vars:,} genes; median counts whole {np.median(obs.n_whole):.0f}, "
      f"nucleus {np.median(obs.n_nucleus):.0f} (share in nucleus {obs.n_nucleus.sum() / max(obs.n_whole.sum(), 1):.3f})",
      flush=True)
