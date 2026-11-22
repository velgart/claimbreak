"""Raw Xenium output (one .tar.gz per sample) -> one claimbreak-ready .h5ad per sample, a depth survey, and an
optional combined file (written for Bhamidipati et al. 2026, RA synovium, Xenium Prime 5K + 100-gene add-on).

Per sample, in one streaming pass over the archive (morphology images are skipped, nothing else is unpacked to disk
beyond the five files needed), the script reads the cell-feature matrix, cells.parquet and transcripts.parquet and
writes <out>/samples/<sample>.h5ad with:
  X                     raw integer counts, Gene Expression features only (control probes and codewords dropped)
  layers['nuclear']     counts of the transcripts that overlap the nucleus (same filter as the matrix: assigned to the
                        cell, gene feature, qv >= 20), for spillover analyses
  obs                   sample, patient, timepoint, cell_id, x, y (centroids, microns), segmentation_method,
                        transcript_counts, cell_area, nucleus_area, n_genes (genes detected), total (counts in X),
                        nuclear_total, passes_filter (n_genes > 50, the authors' Seurat filter: "cells with <= 50
                        transcript species removed")
The nuclear layer is checked against the matrix: per-sample agreement of the transcript-derived totals with X is
reported (`tx_match_share`), and a sample whose agreement is below 0.99 is flagged rather than silently used.

The survey (<out>/survey.csv and survey.md) has one row per sample: cells, share passing the filter, median counts
and genes for all and for filtered cells, segmentation shares, nuclear share of counts, and the transcript check.

The combined file (--combine) concatenates selected samples (e.g. --timepoint pre) into <out>/<name>.h5ad with
the same fields; claimbreak builds neighborhoods within each section (sample), so coordinates are not offset.
Labels are joined later by cell ID and sample (--labels FILE: csv with columns sample, cell_id and label columns).
Label columns may not share a name with an obs column (refused, so a label file cannot
overwrite patient, timepoint or any other key); coverage is computed over the combined samples only, on both sides
(label rows of those samples matched to a cell; cells with a label), per sample and overall, printed, stored in uns
and written to <out>/<name>_label_coverage.csv, with stop flags (more than 10% of cells
unlabeled overall or more than 25% in one sample, or more than 10% of the label rows unmatched).

  python xenium_to_h5ad.py /path/to/RA --out ~/claimbreak_data/ra_bhamidipati --jobs 3
  python xenium_to_h5ad.py /path/to/RA --out ~/claimbreak_data/ra_bhamidipati --combine ra_pre --timepoint pre
Requires pyarrow (pip install pyarrow).
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

WANT = ("cell_feature_matrix/matrix.mtx.gz", "cell_feature_matrix/barcodes.tsv.gz",
        "cell_feature_matrix/features.tsv.gz", "cells.parquet", "transcripts.parquet")
NAME = re.compile(r"xenium_output_patient(\d+)_(pre|post)_treatment")
MIN_QV = 20
FILTER_GENES = 50
MAX_UNLABELLED, MAX_UNLABELLED_SAMPLE, MAX_UNMATCHED = 0.10, 0.25, 0.10     # coverage stop flags


def sample_name(path: Path) -> tuple[str, str, str]:
    m = NAME.search(path.name)
    if not m:
        stem = path.name.split(".tar")[0]
        return stem, stem, ""
    return f"p{int(m.group(1)):02d}_{m.group(2)}", f"p{int(m.group(1)):02d}", m.group(2)


def read_members(path: Path, tmp: Path) -> dict:
    """One pass over the archive; returns {wanted suffix: local path}."""
    tmp.mkdir(parents=True, exist_ok=True)
    got = {}
    with tarfile.open(path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            hit = next((w for w in WANT if m.name.endswith("/" + w) or m.name == w), None)
            if hit is None:
                continue
            dst = tmp / hit.replace("/", "__")
            with tf.extractfile(m) as src, open(dst, "wb") as out:
                while True:
                    b = src.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
            got[hit] = dst
            if len(got) == len(WANT):
                break
    missing = [w for w in WANT if w not in got]
    if missing:
        raise RuntimeError(f"{path.name}: missing {missing}")
    return got


def nuclear_counts(tx_path: Path, cells: pd.Index, genes: pd.Index) -> tuple[sp.csr_matrix, np.ndarray]:
    """Counts of nucleus-overlapping transcripts (cells x genes) and all-transcript totals per cell, with the matrix
    filter (assigned, gene feature, qv >= MIN_QV)."""
    import pyarrow.parquet as pq
    cols = [c for c in ("cell_id", "overlaps_nucleus", "feature_name", "qv") if c in pq.read_schema(tx_path).names]
    T = pq.read_table(tx_path, columns=cols).to_pandas()
    for c in ("cell_id", "feature_name"):
        if T[c].dtype == object and len(T) and isinstance(T[c].iloc[0], (bytes, bytearray)):
            T[c] = T[c].str.decode("utf-8")
    keep = T.qv.to_numpy() >= MIN_QV if "qv" in T else np.ones(len(T), bool)
    ci = cells.get_indexer(T.cell_id.astype(str))
    gi = genes.get_indexer(T.feature_name.astype(str))
    keep &= (ci >= 0) & (gi >= 0)
    tot = np.bincount(ci[keep], minlength=len(cells))
    nuc = keep & (T.overlaps_nucleus.to_numpy().astype(bool))
    N = sp.csr_matrix((np.ones(int(nuc.sum()), np.int32), (ci[nuc], gi[nuc])), shape=(len(cells), len(genes)))
    N.sum_duplicates()
    return N, tot


def process(path: Path, out: Path, tmp_root: Path) -> dict:
    import anndata as ad
    t0 = time.time()
    sample, patient, tp = sample_name(path)
    dst = out / "samples" / f"{sample}.h5ad"
    if dst.exists():
        A = ad.read_h5ad(dst, backed="r")
        r = survey_row(A.obs.copy(), sample)
        r["tx_match_share"] = float(A.uns.get("xenium", {}).get("tx_match_share", np.nan))
        r["seconds"] = 0
        A.file.close()
        return r
    tmp = tmp_root / sample
    try:
        f = read_members(path, tmp)
        feats = pd.read_csv(f["cell_feature_matrix/features.tsv.gz"], sep="\t", header=None)
        bars = pd.read_csv(f["cell_feature_matrix/barcodes.tsv.gz"], sep="\t", header=None)[0].astype(str)
        with gzip.open(f["cell_feature_matrix/matrix.mtx.gz"], "rb") as fh:
            M = scipy.io.mmread(fh).tocsc()                       # features x cells
        isg = (feats[2] == "Gene Expression").to_numpy()
        X = M[isg].T.tocsr().astype(np.int32)                      # cells x genes
        genes = pd.Index(feats.loc[isg, 1].astype(str).to_numpy())
        ids = pd.Index(feats.loc[isg, 0].astype(str).to_numpy())
        C = pd.read_parquet(f["cells.parquet"])
        C["cell_id"] = C.cell_id.astype(str)
        C = C.set_index("cell_id").reindex(bars)
        cells = pd.Index(bars.to_numpy())
        N, tx_tot = nuclear_counts(f["transcripts.parquet"], cells, genes)
        total = np.asarray(X.sum(1)).ravel()
        n_genes = np.diff(X.indptr)
        match = float(np.mean(tx_tot == total)) if len(total) else np.nan
        obs = pd.DataFrame({
            "sample": sample, "patient": patient, "timepoint": tp, "cell_id": cells.to_numpy(),
            "x": C.x_centroid.to_numpy(float), "y": C.y_centroid.to_numpy(float),
            "segmentation_method": C.get("segmentation_method", pd.Series("", index=C.index)).astype(str).to_numpy(),
            "transcript_counts": C.transcript_counts.to_numpy(), "cell_area": C.cell_area.to_numpy(float),
            "nucleus_area": C.nucleus_area.to_numpy(float), "n_genes": n_genes, "total": total,
            "nuclear_total": np.asarray(N.sum(1)).ravel(), "passes_filter": n_genes > FILTER_GENES})
        obs.index = [f"{sample}:{c}" for c in cells]
        for c in ("sample", "patient", "timepoint", "segmentation_method"):
            obs[c] = obs[c].astype("category")
        var = pd.DataFrame({"gene_id": ids.to_numpy()}, index=genes)
        A = ad.AnnData(X=X, obs=obs, var=var)
        A.layers["nuclear"] = N.astype(np.int32)
        A.uns["xenium"] = {"archive": path.name, "tx_match_share": match, "min_qv": MIN_QV}
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_suffix(".h5ad.part")
        A.write_h5ad(part, compression="gzip")
        os.replace(part, dst)
        r = survey_row(obs, sample)
        r["tx_match_share"] = match
        r["seconds"] = round(time.time() - t0)
        return r
    finally:
        for p in tmp.glob("*") if tmp.exists() else []:
            p.unlink()
        if tmp.exists():
            tmp.rmdir()


def survey_row(obs: pd.DataFrame, sample: str) -> dict:
    o = obs
    f = o[o.passes_filter.astype(bool)]
    seg = o.segmentation_method.astype(str).value_counts(normalize=True)
    short = lambda s: ("interior" if "interior" in s else "boundary" if "boundary" in s else
                       "expansion" if "expansion" in s else s or "unknown")
    r = {"sample": sample, "patient": str(o.patient.iloc[0]) if len(o) else "",
         "timepoint": str(o.timepoint.iloc[0]) if len(o) else "", "n_cells": int(len(o)),
         "n_pass": int(len(f)), "share_pass": float(len(f) / max(len(o), 1)),
         "median_total_all": float(o.total.median()), "median_total_pass": float(f.total.median()) if len(f) else np.nan,
         "median_genes_all": float(o.n_genes.median()), "median_genes_pass": float(f.n_genes.median()) if len(f) else np.nan,
         "nuclear_share_counts": float(o.nuclear_total.sum() / max(o.total.sum(), 1)),
         "nuclear_share_counts_pass": float(f.nuclear_total.sum() / max(f.total.sum(), 1)) if len(f) else np.nan}
    for k, v in seg.items():
        r[f"seg_{short(k)}"] = r.get(f"seg_{short(k)}", 0.0) + float(v)
    return r


def _one(args):
    path, out, tmp = args
    try:
        return process(Path(path), Path(out), Path(tmp))
    except Exception as e:                                      # report and continue with the other samples
        return {"sample": sample_name(Path(path))[0], "error": f"{type(e).__name__}: {e}"}


def label_coverage(obs: pd.DataFrame, L: pd.DataFrame, label_cols: list) -> tuple[pd.DataFrame, dict]:
    """Coverage of a label join over the combined samples: per sample, cells, cells with a label (every label
    column non-missing), label rows of that sample, rows matched to a cell; overall shares and stop flags."""
    samples = sorted(set(obs["sample"].astype(str)))
    Ls = L[L["sample"].astype(str).isin(samples)]
    lab = obs[label_cols].notna().all(axis=1).to_numpy() if label_cols else np.zeros(len(obs), bool)
    rows = []
    for s in samples:
        m = (obs["sample"].astype(str) == s).to_numpy()
        ls = Ls[Ls["sample"].astype(str) == s]
        rows.append({"sample": s, "timepoint": str(obs.loc[m, "timepoint"].iloc[0]) if "timepoint" in obs else "",
                     "cells": int(m.sum()), "cells_labelled": int(lab[m].sum()),
                     "share_unlabelled": float(1 - lab[m].mean()) if m.any() else np.nan,
                     "label_rows": int(len(ls)), "label_rows_matched": int(ls.index.isin(obs.index).sum())})
    T = pd.DataFrame(rows)
    T["share_unmatched"] = 1 - T.label_rows_matched / T.label_rows.replace(0, np.nan)
    tot = {"cells": int(T.cells.sum()), "share_unlabelled": float(1 - T.cells_labelled.sum() / max(T.cells.sum(), 1)),
           "max_share_unlabelled_sample": float(T.share_unlabelled.max()),
           "label_rows": int(T.label_rows.sum()),
           "share_unmatched": float(1 - T.label_rows_matched.sum() / max(T.label_rows.sum(), 1)),
           "label_rows_other_samples": int(len(L) - len(Ls))}
    stop = []
    if tot["share_unlabelled"] > MAX_UNLABELLED:
        stop.append(f"{tot['share_unlabelled']:.3f} of cells unlabelled > {MAX_UNLABELLED}")
    if tot["max_share_unlabelled_sample"] > MAX_UNLABELLED_SAMPLE:
        stop.append(f"a sample has {tot['max_share_unlabelled_sample']:.3f} unlabelled > {MAX_UNLABELLED_SAMPLE}")
    if tot["share_unmatched"] > MAX_UNMATCHED:
        stop.append(f"{tot['share_unmatched']:.3f} of label rows unmatched > {MAX_UNMATCHED}")
    tot["stop"] = stop
    return T, tot


def combine(out: Path, name: str, timepoint: str | None, labels: str | None, only_pass: bool):
    import anndata as ad
    files = sorted((out / "samples").glob("*.h5ad"))
    parts = []
    for fpath in files:
        A = ad.read_h5ad(fpath)
        if timepoint and str(A.obs.timepoint.iloc[0]) != timepoint:
            continue
        if only_pass:
            A = A[A.obs.passes_filter.to_numpy()].copy()
        parts.append(A)
    if not parts:
        sys.exit("no samples selected")
    genes = parts[0].var_names
    for A in parts[1:]:
        if not A.var_names.equals(genes):
            sys.exit("samples have different gene lists; not combining")
    B = ad.concat(parts, join="outer", merge="first")
    B.uns["xenium_combined"] = {"samples": [str(A.obs["sample"].iloc[0]) for A in parts], "timepoint": timepoint or "all",
                                "only_pass": bool(only_pass)}
    if labels:
        L = pd.read_csv(labels)
        if not {"sample", "cell_id"} <= set(L.columns):
            sys.exit("labels file needs columns sample and cell_id")
        L.index = L["sample"].astype(str) + ":" + L["cell_id"].astype(str)
        if L.index.duplicated().any():
            sys.exit("labels file has duplicated (sample, cell_id) rows")
        cols = [c for c in L.columns if c not in ("sample", "cell_id")]
        clash = sorted(set(cols) & set(B.obs.columns))
        if clash:
            sys.exit(f"label columns {clash} share names with obs columns; rename them in the labels file")
        for c in cols:
            B.obs[c] = L[c].reindex(B.obs_names).to_numpy()
        T, tot = label_coverage(B.obs, L, cols)
        T.to_csv(out / f"{name}_label_coverage.csv", index=False)
        print(T.to_string(index=False), flush=True)
        print(f"labels: {tot['label_rows']} rows of the combined samples ({tot['label_rows_other_samples']} rows of other "
              f"samples ignored), {1 - tot['share_unmatched']:.4f} matched to cells; {tot['cells']} cells, "
              f"{tot['share_unlabelled']:.4f} unlabelled (largest per sample {tot['max_share_unlabelled_sample']:.4f})",
              flush=True)
        print("STOP (label coverage): " + "; ".join(tot["stop"]) if tot["stop"] else "label coverage within limits",
              flush=True)
        B.uns["xenium_combined"]["label_coverage"] = {k: v for k, v in tot.items() if k != "stop"}
        B.uns["xenium_combined"]["label_coverage_stop"] = "; ".join(tot["stop"])
    dst = out / f"{name}.h5ad"
    B.write_h5ad(dst, compression="gzip")
    print(f"combined: {B.n_obs} cells x {B.n_vars} genes from {len(parts)} samples -> {dst}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir", help="folder with xenium_output_*.tar.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", help="scratch folder for the five extracted files (default OUT/tmp)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", help="comma-separated sample names (e.g. p01_pre) to process")
    ap.add_argument("--combine", metavar="NAME", help="write OUT/NAME.h5ad from the per-sample files")
    ap.add_argument("--timepoint", choices=["pre", "post"], help="with --combine: only these samples")
    ap.add_argument("--only-pass", action="store_true", help="with --combine: only cells passing the authors' filter")
    ap.add_argument("--labels", help="with --combine: csv with sample, cell_id and label columns")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.combine:
        combine(out, a.combine, a.timepoint, a.labels, a.only_pass)
        return
    paths = sorted(Path(a.data_dir).expanduser().glob("xenium_output_*.tar.gz"))
    if a.only:
        want = set(a.only.split(","))
        paths = [p for p in paths if sample_name(p)[0] in want]
    if not paths:
        sys.exit(f"no xenium_output_*.tar.gz in {a.data_dir}")
    tmp = Path(a.tmp).expanduser() if a.tmp else out / "tmp"
    jobs = [(str(p), str(out), str(tmp)) for p in paths]
    print(f"{len(jobs)} archives -> {out} ({a.jobs} at a time)", flush=True)
    rows = []
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(_one, jobs):
                rows.append(r)
                print(r, flush=True)
    else:
        for j in jobs:
            r = _one(j)
            rows.append(r)
            print(r, flush=True)
    S = pd.DataFrame(rows).sort_values("sample")
    old = out / "survey.csv"
    if old.exists():
        prev = pd.read_csv(old)
        S = pd.concat([prev[~prev["sample"].isin(S["sample"])], S], ignore_index=True).sort_values("sample")
    S.to_csv(old, index=False, float_format="%.4g")
    flag = S[S.get("tx_match_share", pd.Series(1.0, index=S.index)).fillna(1.0) < 0.99] if len(S) else S
    L = ["# Xenium depth survey", "", S.round(3).to_string(index=False), ""]
    if "error" in S and S.error.notna().any():
        L.append("ERRORS: " + "; ".join(f"{s}: {e}" for s, e in zip(S["sample"], S["error"]) if isinstance(e, str)))
    if len(flag):
        L.append("transcript totals disagree with the matrix (< 0.99 of cells) in: " + ", ".join(flag["sample"]))
    (out / "survey.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
