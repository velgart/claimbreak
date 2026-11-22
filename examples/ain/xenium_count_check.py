#!/usr/bin/env python
"""xenium_count_check.py -- why do whole-cell totals from the transcript table differ from the platform's cell-feature
matrix for some Xenium regions? (read only)

For each region: per-cell totals from the converted .h5ad (X, transcripts with qv >= 20 assigned to the cell), from
<sample>_cell_feature_matrix.h5 (Gene Expression features) and, if --meta is given, the authors' nCount_RNA
(cells named Region_<k>_<cell_id>). Prints the share of cells with equal totals for each pair, quantiles of the ratio
h5ad / matrix, the genes whose column sums differ most, and a scan of the transcript table (columns, qv below 20,
codeword categories, cells assigned) so that a different qv threshold, extra codewords or a re-segmentation can be told
apart.

  python xenium_count_check.py --raw ~/claimbreak_data/AIN/raw \
      --h5ad-dir ~/claimbreak_data/AIN/h5ad_regions --meta ~/Downloads/ain_meta/meta.csv.gz \
      --samples GSM9844156_Region01,GSM9844164_Region15,GSM9844157_Region02
"""
import argparse
import gzip
import re
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def plain(p, tmp):
    p = Path(p)
    if p.suffix != ".gz":
        return p, False
    q = Path(tmp) / p.name[:-3]
    with gzip.open(p, "rb") as fi, open(q, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 24)
    return q, True


def first(raw, sample, stem):
    for suf in ("", ".gz"):
        p = Path(raw) / f"{sample}{stem}{suf}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--h5ad-dir", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--tmp", default="")
    ap.add_argument("--scan-rows", type=int, default=20_000_000, help="transcript rows scanned per region")
    a = ap.parse_args()
    M = None
    if a.meta:
        M = pd.read_csv(a.meta, usecols=["cell", "nCount_RNA"])
        m = M["cell"].astype(str).str.extract(r"^Region_(\d+)_(.+)$")
        M["region"], M["cell_id"] = pd.to_numeric(m[0]), m[1]
    tmp = Path(a.tmp or tempfile.mkdtemp())
    pd.set_option("display.width", 200)
    for s in a.samples.split(","):
        print(f"\n==================== {s}", flush=True)
        A = ad.read_h5ad(Path(a.h5ad_dir) / f"{s}.h5ad")
        ours = pd.Series(np.asarray(A.X.sum(1)).ravel(), index=A.obs_names.astype(str))
        og = pd.Series(np.asarray(A.X.sum(0)).ravel(), index=A.var_names.astype(str))
        import h5py
        p, rm = plain(first(a.raw, s, "_cell_feature_matrix.h5"), tmp)
        with h5py.File(p, "r") as h:
            m = h["matrix"]
            bc = np.array(m["barcodes"]).astype(str)
            ft = np.array(m["features"]["feature_type"]).astype(str)
            nm = np.array(m["features"]["name"]).astype(str)
            X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
        if rm:
            p.unlink()
        print("feature types in the matrix:", pd.Series(ft).value_counts().to_dict())
        ge = ft == "Gene Expression"
        mat = pd.Series(np.asarray(X[ge].sum(0)).ravel(), index=bc)
        mg = pd.Series(np.asarray(X[ge].sum(1)).ravel(), index=nm[ge])
        common = ours.index.intersection(mat.index)
        o, t = ours.loc[common], mat.loc[common]
        r = (o / t.replace(0, np.nan)).dropna()
        print(f"cells: h5ad {len(ours):,}, matrix {len(mat):,}, common {len(common):,}; h5ad == matrix for "
              f"{np.mean(o.to_numpy() == t.to_numpy()):.4f}; totals h5ad {o.sum():,.0f}, matrix {t.sum():,.0f}")
        print("ratio h5ad/matrix quantiles 1,5,25,50,75,95,99%:",
              np.round(np.percentile(r, [1, 5, 25, 50, 75, 95, 99]), 3).tolist())
        neq = o.to_numpy() != t.to_numpy()
        if neq.any():
            print(f"cells that differ: {neq.sum():,}; of them h5ad > matrix {np.mean(o[neq] > t[neq]):.3f}; "
                  f"median |difference| {np.median(np.abs(o[neq] - t[neq])):.0f} counts")
        g = pd.DataFrame({"h5ad": og, "matrix": mg.groupby(level=0).sum()}).fillna(0)
        g["diff"] = g.h5ad - g.matrix
        g["rel"] = g["diff"] / g.matrix.replace(0, np.nan)
        print(f"genes: {len(g):,}; equal column sums {np.mean(g['diff'] == 0):.3f}; largest absolute differences:")
        print(g.reindex(g["diff"].abs().sort_values(ascending=False).index).head(12).round(3).to_string())
        if M is not None:
            reg = int(re.search(r"Region(\d+)", s).group(1))
            au = M[M.region == reg].set_index("cell_id")["nCount_RNA"]
            c2 = au.index.intersection(common)
            print(f"authors' nCount_RNA: {len(au):,} cells, {len(c2):,} common; equal to h5ad "
                  f"{np.mean(au.loc[c2].to_numpy() == ours.loc[c2].to_numpy()):.4f}, to matrix "
                  f"{np.mean(au.loc[c2].to_numpy() == mat.loc[c2].to_numpy()):.4f}")
        ftr = first(a.raw, s, "_transcripts.parquet")
        p, rm = plain(ftr, tmp)
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(p)
        cols = pf.schema_arrow.names
        print(f"transcript columns: {cols}; rows {pf.metadata.num_rows:,}")
        want = [c for c in ("cell_id", "qv", "feature_name", "codeword_category", "is_gene", "overlaps_nucleus",
                            "fov_name") if c in cols]
        n = 0
        parts = []
        for rb in pf.iter_batches(batch_size=5_000_000, columns=want):
            d = rb.to_pandas()
            parts.append(d)
            n += len(d)
            if n >= a.scan_rows:
                break
        D = pd.concat(parts, ignore_index=True)
        if rm:
            p.unlink()
        assigned = ~D.cell_id.astype(str).isin(["UNASSIGNED", "-1", ""])
        print(f"scanned {len(D):,} rows: assigned to a cell {assigned.mean():.3f}; qv < 20 {np.mean(D.qv < 20):.3f}; "
              f"assigned and qv < 20 {np.mean(assigned & (D.qv < 20)):.3f}")
        for c in ("codeword_category", "is_gene"):
            if c in D:
                print(f"{c}:", D[c].astype(str).value_counts().to_dict())
        if "fov_name" in D:
            print(f"fields of view in the scan: {D.fov_name.nunique()}")
        qv = D.loc[assigned, "qv"].to_numpy()
        print("qv quantiles of assigned transcripts 1,5,25,50%:", np.round(np.percentile(qv, [1, 5, 25, 50]), 2).tolist())


if __name__ == "__main__":
    main()
