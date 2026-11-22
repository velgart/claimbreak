#!/usr/bin/env python
"""proseg_to_h5ad.py -- a compartment .h5ad (whole-cell, nucleus and cytoplasm counts of the same cells) from a Proseg
segmentation of a Xenium section ("does better
segmentation fix it?").

Proseg (Jones et al., Nat Methods 2025) reassigns transcripts to cells from the transcripts themselves. The compartment
test needs, per cell, the counts over its nucleus; those come from the Xenium transcripts (overlaps_nucleus == 1),
which Proseg does not change. Each Proseg cell is tied to one Xenium cell by its nuclear transcripts, so that the cell
ids (and the protein matches, which are made to Xenium nuclei) stay those of the deposited data:

  1. join Proseg's transcript assignments to transcripts.parquet: by transcript_id when Proseg writes it, otherwise
     by row order after the same filter (qv >= --min-qv), checked gene by gene (stops if they differ);
  2. Proseg cell -> the Xenium cell whose nuclear transcripts it holds most of (a Xenium nucleus claimed by two Proseg
     cells goes to the one holding more of it); Proseg cells holding no nuclear transcript are dropped (reported);
     where Proseg's cell metadata has an original cell id, the agreement is reported;
  3. X = all transcripts Proseg assigns to the cell (gene probes, qv >= --min-qv); layers['nucleus'] = those of them
     with overlaps_nucleus == 1 whose Xenium cell is the cell's own; layers['cyto'] = X - nucleus.
obs: as bilous_to_h5ad.py (sample, donor, panel, section, x, y = Xenium cell centroid, cell_area, nucleus_area,
n_whole, n_nucleus) plus proseg_cell, share_own_nucleus (share of the Xenium nucleus's transcripts kept);
uns['proseg']: mapping and join statistics.

  python proseg_to_h5ad.py probe   --proseg DIR                       # print the columns of Proseg's outputs
  python proseg_to_h5ad.py convert --proseg DIR --raw RAW --sample GSM9509145_NSCLC_lung_L1_L1 --out OUT --tmp TMP

DIR holds transcript-metadata.csv.gz (and optionally cell-metadata.csv.gz) written by
  proseg --xenium transcripts.parquet --output-transcript-metadata DIR/transcript-metadata.csv.gz \
         --output-cell-metadata DIR/cell-metadata.csv.gz ...
The raw folder is only read; OUT/<sample>.h5ad is never overwritten.
"""
import argparse
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL = re.compile(r"^(NegControl|BLANK|Blank|Unassigned|Intergenic|DeprecatedCodeword|antisense|UnassignedCodeword)",
                     re.I)
TX_ID = ("transcript_id", "transcript", "id")
ASSIGN = ("assignment", "cell", "cell_id", "assigned_cell")
GENE = ("gene", "feature_name", "gene_name")
CELL_KEY = ("cell", "cell_id", "id")
ORIG = ("original_cell_id", "xenium_cell_id", "initial_cell_id")


def pick(cols, cands, what, required=True):
    for c in cands:
        if c in cols:
            return c
    if required:
        sys.exit(f"no {what} column among {list(cols)} (looked for {cands}); check the input columns")
    return None


def read_table(p, **kw):
    p = Path(p)
    if p.suffix == ".parquet" or p.name.endswith(".parquet.gz"):
        import pyarrow.parquet as pq
        return pq.read_table(p, columns=kw.get("usecols")).to_pandas()
    return pd.read_csv(p, low_memory=False, **kw)


def find_out(d, stem):
    for suf in (".csv.gz", ".csv", ".parquet"):
        q = Path(d) / f"{stem}{suf}"
        if q.exists():
            return q
    return None


def cmd_probe(a):
    d = Path(a.proseg).expanduser()
    for stem in ("transcript-metadata", "cell-metadata", "gene-metadata"):
        p = find_out(d, stem)
        if p is None:
            print(f"{stem}: not found in {d}")
            continue
        h = read_table(p, nrows=5) if p.suffix != ".parquet" else read_table(p).head()
        print(f"{p.name}: columns {list(h.columns)}")
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(h.head(3).to_string())
    for p in sorted(d.glob("*.zarr")):
        print(f"spatialdata: {p.name}: {sorted(x.name for x in p.iterdir())}")


def plain(p, tmp):
    if p.suffix != ".gz":
        return p, False
    q = tmp / p.name[:-3]
    with gzip.open(p, "rb") as fi, open(q, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 24)
    return q, True


def find_raw(raw, sample, stem):
    for suf in ("", ".gz"):
        q = raw / f"{sample}_{stem}{suf}"
        if q.exists():
            return q
    return None


def cmd_convert(a):
    import anndata as ad
    import pyarrow.parquet as pq
    raw, out, tmp = (Path(p).expanduser().resolve() for p in (a.raw, a.out, a.tmp))
    pdir = Path(a.proseg).expanduser().resolve()
    for p in (out, tmp):
        if p == raw or raw in p.parents:
            sys.exit("--out and --tmp must be outside the raw data folder")
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    dst = out / f"{a.sample}.h5ad"
    if dst.exists():
        sys.exit(f"{dst} exists; not overwritten")
    m = re.match(r"GSM\d+_(NSCLC|breast)_(5k|chuvio|lung|breast)_(\w+?)_(\w+)$", a.sample)
    panel = {"5k": "5K", "chuvio": "Custom IO", "lung": "Lung", "breast": "Breast"}.get(m.group(2), m.group(2)) if m else ""
    donor = m.group(3) if m else ""

    # Xenium transcripts (same filter as bilous_to_h5ad.py, plus transcript_id)
    ftr, fcells = find_raw(raw, a.sample, "transcripts.parquet"), find_raw(raw, a.sample, "cells.parquet")
    if ftr is None or fcells is None:
        sys.exit(f"transcripts or cells parquet missing for {a.sample} in {raw}")
    pt_, rm = plain(ftr, tmp)
    pf = pq.ParquetFile(pt_)
    cols = set(pf.schema_arrow.names)
    need = {"cell_id", "feature_name", "qv", "overlaps_nucleus"} | ({"transcript_id"} & cols)
    X = pf.read(columns=sorted(need)).to_pandas()
    if rm:
        pt_.unlink()
    n_all = len(X)
    X["feature_name"] = X["feature_name"].astype(str)
    X["cell_id"] = X["cell_id"].astype(str)
    X["qvok"] = X["qv"].to_numpy() >= a.min_qv
    print(f"{a.sample}: {n_all:,} Xenium transcripts, {int(X.qvok.sum()):,} with qv >= {a.min_qv}", flush=True)

    # Proseg assignments
    ptm = find_out(pdir, "transcript-metadata")
    if ptm is None:
        sys.exit(f"no transcript-metadata.csv(.gz)/.parquet in {pdir}")
    T = read_table(ptm)
    a_col = pick(T.columns, ASSIGN, "assignment")
    g_col = pick(T.columns, GENE, "gene", required=False)
    t_col = pick(T.columns, TX_ID, "transcript id", required=False)
    print(f"Proseg transcript metadata: {len(T):,} rows; assignment '{a_col}', gene '{g_col}', id '{t_col}'", flush=True)
    stats = {"proseg_rows": int(len(T)), "assignment_column": a_col, "gene_column": g_col, "id_column": t_col}
    if t_col and "transcript_id" in X:
        T[t_col] = pd.to_numeric(T[t_col], errors="coerce")
        pos = pd.Index(X["transcript_id"].astype("int64"))
        ri = pos.get_indexer(T[t_col].astype("int64"))
        if (ri < 0).mean() > 0.01:
            sys.exit(f"{(ri < 0).mean():.3f} of Proseg transcript ids not in transcripts.parquet: check the input file")
        stats["join"] = "transcript_id"
        stats["ids_not_found"] = int((ri < 0).sum())
    else:
        cand = np.flatnonzero(X.qvok.to_numpy())
        if len(cand) != len(T):
            cand = np.arange(n_all) if len(T) == n_all else None
        if cand is None:
            sys.exit(f"no transcript id in Proseg's output and {len(T):,} rows match neither the qv-filtered "
                     f"({int(X.qvok.sum()):,}) nor all ({n_all:,}) Xenium transcripts; check the `probe` output")
        ri = cand
        stats["join"] = "row order"
    if g_col is not None:
        ok = ri >= 0
        same = (T[g_col].astype(str).to_numpy()[ok] == X["feature_name"].to_numpy()[ri[ok]]).mean()
        stats["gene_agreement"] = float(same)
        print(f"join by {stats['join']}: gene agreement {same:.5f}", flush=True)
        if same < 0.999:
            sys.exit("Proseg genes do not match the Xenium transcripts at the joined rows; check the `probe` output")
    asg = pd.to_numeric(T[a_col], errors="coerce").to_numpy(float)
    n_cells_ps = None
    pcm = find_out(pdir, "cell-metadata")
    C = read_table(pcm) if pcm is not None else None
    if C is not None:
        n_cells_ps = len(C)
    valid = np.isfinite(asg) & (asg >= 0) & ((asg < n_cells_ps) if n_cells_ps else (asg < 2 ** 31 - 1)) & (ri >= 0)
    keep_t = np.zeros(n_all, bool)
    pcell = np.full(n_all, -1, np.int64)
    pcell[ri[valid]] = asg[valid].astype(np.int64)
    keep_t[ri[valid]] = True
    gene_ok = ~X["feature_name"].str.match(CONTROL).to_numpy()
    keep_t &= X.qvok.to_numpy() & gene_ok
    stats["assigned_share"] = float(valid.mean())
    print(f"assigned by Proseg: {valid.mean():.3f} of its transcripts; kept (genes, qv): {int(keep_t.sum()):,}", flush=True)

    # Proseg cell -> Xenium cell, by nuclear transcripts
    cells = pq.read_table(plain(fcells, tmp)[0]).to_pandas()
    cells["cell_id"] = cells["cell_id"].astype(str)
    ci = pd.Index(cells["cell_id"])
    xcell = ci.get_indexer(X["cell_id"])                    # -1: unassigned in Xenium
    nuc = X["overlaps_nucleus"].to_numpy().astype(bool) & (xcell >= 0)
    sel = keep_t & nuc
    pair = pd.DataFrame({"p": pcell[sel], "x": xcell[sel]}).value_counts().rename("n").reset_index()
    nuc_tot = np.bincount(xcell[nuc & X.qvok.to_numpy() & gene_ok], minlength=len(ci))
    pair = pair.sort_values("n", ascending=False)
    best_p = pair.drop_duplicates("p")                       # each Proseg cell: its main Xenium nucleus
    best = best_p.drop_duplicates("x")                       # each Xenium nucleus: the Proseg cell holding most of it
    p2x = dict(zip(best.p, best.x))
    n_ps = int(len(np.unique(pcell[keep_t & (pcell >= 0)])))
    stats.update({"proseg_cells_with_counts": n_ps, "proseg_cells_mapped": int(len(best)),
                  "proseg_cells_without_nucleus": int(n_ps - len(best_p)),
                  "proseg_cells_lost_nucleus_to_another": int(len(best_p) - len(best)),
                  "xenium_cells": int(len(ci))})
    if C is not None:
        k_col = pick(C.columns, CELL_KEY, "cell", required=False)
        o_col = pick(C.columns, ORIG, "original cell id", required=False)
        if o_col is not None:
            key = C[k_col].to_numpy() if k_col else np.arange(len(C))
            orig = pd.Series(C[o_col].astype(str).to_numpy(), index=key)
            ours = pd.Series({p: ci[x] for p, x in p2x.items()})
            both = ours.index.intersection(orig.index)
            agree = float((orig.loc[both].to_numpy() == ours.loc[both].to_numpy()).mean()) if len(both) else float("nan")
            stats["original_id_agreement"] = agree
            print(f"mapping agrees with Proseg's '{o_col}' for {agree:.3f} of {len(both):,} cells", flush=True)
    print(f"Proseg cells with counts {n_ps:,}; mapped to a Xenium nucleus {len(best):,}; without nuclear transcripts "
          f"{stats['proseg_cells_without_nucleus']:,}; nucleus taken by another cell {stats['proseg_cells_lost_nucleus_to_another']:,}",
          flush=True)

    # counts
    order = np.array(sorted(best.p))
    row_of = pd.Series(np.arange(len(order)), index=order)
    xs = np.array([p2x[p] for p in order])
    genes = np.array(sorted(X.loc[gene_ok, "feature_name"].unique()))
    gi = pd.Index(genes).get_indexer(X["feature_name"])
    m = keep_t & np.isin(pcell, order)
    r = row_of.reindex(pcell[m]).to_numpy().astype(np.int64)
    c = gi[m]
    W = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(order), len(genes)))
    own = nuc[m] & (xcell[m] == xs[r])
    Nn = sp.csr_matrix((np.ones(int(own.sum()), np.float32), (r[own], c[own])), shape=W.shape)
    W.sum_duplicates(); Nn.sum_duplicates()
    cell_ids = ci[xs].astype(str)
    cx = cells.set_index("cell_id").loc[cell_ids]
    obs = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    obs["sample"], obs["donor"], obs["panel"] = a.sample, donor, panel
    obs["section"] = f"{panel}:{a.sample.split('_', 1)[1]}:proseg"
    obs["x"] = cx["x_centroid"].to_numpy(float)
    obs["y"] = cx["y_centroid"].to_numpy(float)
    for k in ("cell_area", "nucleus_area"):
        obs[k] = cx[k].to_numpy(float) if k in cx else np.nan
    obs["n_whole"] = np.asarray(W.sum(1)).ravel()
    obs["n_nucleus"] = np.asarray(Nn.sum(1)).ravel()
    obs["proseg_cell"] = order
    obs["share_own_nucleus"] = obs["n_nucleus"].to_numpy() / np.maximum(nuc_tot[xs], 1)
    A = ad.AnnData(W, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    A.layers["nucleus"] = Nn
    A.layers["cyto"] = (W - Nn).tocsr()
    for k in ("sample", "donor", "panel", "section"):
        A.obs[k] = A.obs[k].astype("category")
    stats.update({"cells": int(A.n_obs), "genes": int(A.n_vars), "median_whole": float(np.median(obs.n_whole)),
                  "median_nucleus": float(np.median(obs.n_nucleus)),
                  "share_in_nucleus": float(obs.n_nucleus.sum() / max(obs.n_whole.sum(), 1)),
                  "median_share_own_nucleus": float(np.median(obs.share_own_nucleus)), "min_qv": a.min_qv,
                  "proseg_dir": str(pdir)})
    A.uns["proseg"] = json.loads(json.dumps(stats, default=float))
    A.uns["sample"] = a.sample
    A.write_h5ad(dst)
    print(json.dumps(stats, indent=1, default=float))
    print(f"wrote {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--proseg", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--proseg", required=True); c.add_argument("--raw", required=True)
    c.add_argument("--sample", required=True); c.add_argument("--out", required=True); c.add_argument("--tmp", required=True)
    c.add_argument("--min-qv", type=float, default=20.0)
    a = ap.parse_args()
    {"probe": cmd_probe, "convert": cmd_convert}[a.cmd](a)


if __name__ == "__main__":
    main()
