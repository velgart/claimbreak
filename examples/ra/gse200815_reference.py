#!/usr/bin/env python
"""gse200815_reference.py -- a synovium single-cell reference from GEO GSE200815 (Floudas et al., Ann Rheum Dis 2022;
unsorted synovial tissue, 4 RA and 5 PsA patients, 10x 3' v3.1), annotated at the coarse level by canonical lineage
markers and, for information, by the AMP RA phase 2 cell states (Zhang et al., Nature 2023, Supplementary Table 5,
open aggregate marker table). For `claimbreak compartment label` on the RA Xenium data.

The GEO deposit holds unfiltered matrices (all 10x barcodes) and no cell annotation, so:
  cluster   extract the chosen samples from GSE200815_RAW.tar into WORK (the tar is only read); call cells (UMIs >=
            --min-umi, genes >= --min-genes, mitochondrial share < --max-mito, the authors' 25%); drop the top
            --doublet-pct % of cells by UMIs per sample (a crude doublet guard; the authors used DoubletFinder);
            normalize, PCA, Harmony by sample when harmonypy is installed, Leiden clusters; score every cluster on
            canonical lineage markers and on the AMP state markers; propose a coarse class per cluster (flagged for
            review when two lineages score high (doublets), when the margin is small, or for proliferating clusters).
            Writes WORK/cells.h5ad, WORK/clusters.csv (one row per cluster: size, samples, scores, proposal, flag,
            best AMP state) and prints the table. Edit the `coarse` column of a copy (clusters_reviewed.csv) to
            change a proposal; "" drops a cluster.
  build     from WORK/cells.h5ad and the reviewed (or proposed) cluster table: drop "" clusters, at most --per-type
            cells per coarse class, genes of the spatial panel (--genes-from), raw counts in X; obs donor_id, coarse,
            amp_state, cluster, sample, diagnosis. Writes OUT and <OUT stem>_types.csv.

  python gse200815_reference.py cluster --tar GSE200815_RAW.tar --amp-markers 41586_2023_6708_MOESM5_ESM.xlsx \
      --work ~/claimbreak_data/data/RA_ref/GSE200815/work [--diagnoses RA]
  python gse200815_reference.py build --work .../work [--clusters .../work/clusters_reviewed.csv] \
      --genes-from '~/claimbreak_data/data/RA_h5ad/compartment_pre/*.h5ad' \
      --out ~/claimbreak_data/data/RA_ref/gse200815_ref.h5ad
Needs scanpy (and igraph or leidenalg; harmonypy optional): pip install scanpy igraph leidenalg harmonypy
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

LINEAGE = {  # canonical lineage markers (coarse classes of the RA run)
    "T_NK": ["CD3E", "CD3D", "CD2", "TRAC", "CD247", "NKG7", "GNLY", "KLRD1"],
    "B": ["MS4A1", "CD79A", "CD19", "BANK1", "CD22"],
    "Plasma": ["MZB1", "JCHAIN", "SDC1", "TNFRSF17", "IGHG1", "XBP1"],
    "Myeloid": ["CD68", "C1QA", "C1QB", "LYZ", "CD14", "CSF1R", "AIF1", "FCER1G"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "PRG4", "PDPN", "DCN", "LUM", "THY1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "EMCN"],
    "Mural": ["RGS5", "ACTA2", "MYH11", "NOTCH3", "MCAM", "TAGLN"],
}
PROLIF = ["MKI67", "TOP2A", "CENPF", "STMN1"]
AMP_SHEETS = {"T_NK": ["T cell cluster marker", "NK cell cluster marker"], "B": ["B cell cluster marker"],
              "Plasma": ["B cell cluster marker"], "Myeloid": ["Myeloid cell cluster marker"],
              "Fibroblast": ["Stromal cluster marker"], "Mural": ["Stromal cluster marker"],
              "Endothelial": ["Endothelial cluster marker"], "Mast": []}
SAMPLE = re.compile(r"(GSM\d+)_((RA|PsA)\w+?)_(barcodes|features|matrix)\.(tsv|mtx)\.gz$")


def amp_markers(path, top=15):
    """{sheet: {state: [genes]}} from Supplementary Table 5 (header row found by 'Cluster number')."""
    X = pd.ExcelFile(path)
    out = {}
    for s in X.sheet_names:
        d = X.parse(s, header=None)
        h = d.index[d.iloc[:, 0].astype(str).str.contains("Cluster number")][0]
        d = X.parse(s, header=h)
        out[s] = {str(c): g.sort_values("FC", ascending=False).Gene.astype(str).head(top).tolist()
                  for c, g in d.groupby("Cluster annotation", sort=False)}
    return out


def read_sample(files):
    import scipy.io
    feats = pd.read_csv(files["features"], sep="\t", header=None)
    bars = pd.read_csv(files["barcodes"], sep="\t", header=None)[0].astype(str).to_numpy()
    with gzip.open(files["matrix"], "rb") as fh:
        M = sp.csc_matrix(scipy.io.mmread(fh))                       # genes x barcodes
    if feats.shape[1] > 2:
        keep = (feats[2].astype(str) == "Gene Expression").to_numpy()
        feats, M = feats[keep], M[np.flatnonzero(keep)]
    return M, feats[0].astype(str).to_numpy(), feats[1].astype(str).to_numpy(), bars


def score_clusters(A, groups, sets):
    """mean over genes of the cluster-mean z score (genes z-scored across clusters of log-normalized expression)."""
    import scanpy as sc
    cl = A.obs[groups].astype(str)
    names = sorted(cl.unique(), key=lambda x: (len(x), x))
    genes = sorted({g for v in sets.values() for g in v if g in A.var_names})
    if not genes:
        return pd.DataFrame(index=names)
    X = A[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    mean = pd.DataFrame(X, columns=genes).groupby(cl.to_numpy()).mean().reindex(names)
    z = (mean - mean.mean()) / mean.std(ddof=0).replace(0, 1)
    return pd.DataFrame({k: z[[g for g in v if g in z.columns]].mean(1) if any(g in z.columns for g in v) else np.nan
                         for k, v in sets.items()}, index=names)


def cmd_cluster(a):
    import anndata as ad
    import scanpy as sc
    work = Path(a.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    if (work / "cells.h5ad").exists():
        sys.exit(f"{work}/cells.h5ad exists; remove it to recluster")
    want = set(a.diagnoses.split(","))
    raw = work / "raw"
    raw.mkdir(exist_ok=True)
    samples = {}
    with tarfile.open(Path(a.tar).expanduser(), "r") as tf:
        for m in tf.getmembers():
            r = SAMPLE.search(m.name)
            if not r or r.group(3) not in want:
                continue
            dst = raw / Path(m.name).name
            if not dst.exists():
                with tf.extractfile(m) as src, open(dst, "wb") as fo:
                    fo.write(src.read())
            samples.setdefault((r.group(1), r.group(2), r.group(3)), {})[r.group(4)] = dst
    if not samples:
        sys.exit(f"no {a.diagnoses} samples in {a.tar}")
    parts = []
    for (gsm, name, dx), files in sorted(samples.items()):
        M, ids, sym, bars = read_sample(files)
        umi = np.asarray(M.sum(0)).ravel()
        ng = np.diff(M.indptr)
        mt = np.asarray(M[np.flatnonzero(pd.Series(sym).str.upper().str.startswith("MT-").to_numpy())].sum(0)).ravel()
        mito = mt / np.maximum(umi, 1)
        ok = (umi >= a.min_umi) & (ng >= a.min_genes) & (mito < a.max_mito)
        if ok.sum() and a.doublet_pct > 0:
            ok &= umi <= np.percentile(umi[ok], 100 - a.doublet_pct)
        idx = np.flatnonzero(ok)
        B = ad.AnnData(X=sp.csr_matrix(M[:, idx].T).astype(np.float32),
                       obs=pd.DataFrame({"sample": name, "gsm": gsm, "diagnosis": dx, "umi": umi[idx],
                                         "genes": ng[idx], "mito": mito[idx]},
                                        index=[f"{name}:{b}" for b in bars[idx]]),
                       var=pd.DataFrame({"gene_id": ids, "feature_name": sym}, index=ids))
        print(f"{name} ({dx}): {len(bars):,} barcodes; {ok.sum():,} cells (median UMIs {np.median(umi[idx]):.0f}, "
              f"genes {np.median(ng[idx]):.0f}, mito {np.median(mito[idx]):.3f})", flush=True)
        parts.append(B)
    if sum(p.n_obs for p in parts) == 0:
        sys.exit("no cells passed the thresholds (--min-umi, --min-genes, --max-mito)")
    A = ad.concat([p for p in parts if p.n_obs], join="inner", merge="same")
    A.var_names = A.var["feature_name"].astype(str).to_numpy()
    A.var_names_make_unique()
    A.layers["counts"] = A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    sc.pp.highly_variable_genes(A, n_top_genes=a.hvg, batch_key="sample")
    H = A[:, A.var.highly_variable].copy()
    sc.pp.scale(H, max_value=10)
    sc.tl.pca(H, n_comps=a.pcs, random_state=0)
    rep = "X_pca"
    if a.harmony:
        # harmonypy directly: scanpy's wrapper (<= 1.12) transposes Z_corr, which harmonypy >= 2 already
        # returns as cells x PCs, so the wrapper raises ValueError (shape).
        try:
            import harmonypy
            Z = np.asarray(harmonypy.run_harmony(H.obsm["X_pca"], H.obs, "sample", random_state=0).Z_corr)
            if Z.shape != H.obsm["X_pca"].shape:
                Z = Z.T
            H.obsm["X_pca_harmony"] = Z
            rep = "X_pca_harmony"
            print("Harmony by sample", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            print("no Harmony (traceback above); clustering on PCA", flush=True)
    else:
        print("Harmony off (--no-harmony); clustering on PCA", flush=True)
    A.uns["embedding"] = rep
    sc.pp.neighbors(H, use_rep=rep, n_neighbors=15, random_state=0)
    try:
        sc.tl.leiden(H, resolution=a.resolution, key_added="cluster", flavor="igraph", n_iterations=2,
                     directed=False, random_state=0)
    except Exception:
        sc.tl.leiden(H, resolution=a.resolution, key_added="cluster", random_state=0)
    A.obs["cluster"] = H.obs["cluster"].astype(str).to_numpy()
    sc_lin = score_clusters(A, "cluster", dict(LINEAGE, Proliferating=PROLIF))
    # a z score is relative to the other clusters, so a lineage nobody expresses (e.g. mast cells absent)
    # can still "win" a cluster; require the cluster's mean log-normalized expression of the set to reach --min-expr
    cl = A.obs["cluster"].astype(str).to_numpy()
    for k, genes in dict(LINEAGE, Proliferating=PROLIF).items():
        gg = [x for x in genes if x in A.var_names]
        if not gg:
            continue
        X = A[:, gg].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        absm = pd.Series(X.max(1), index=cl).groupby(level=0).mean()
        weak = absm.reindex(sc_lin.index).fillna(0) < a.min_expr
        sc_lin.loc[weak, k] = np.nan
    amp = amp_markers(a.amp_markers) if a.amp_markers else {}
    rows = []
    for c in sc_lin.index:
        s = sc_lin.loc[c, list(LINEAGE)].dropna().sort_values(ascending=False)
        if len(s) < 2:
            s = pd.concat([s, pd.Series([-np.inf] * (2 - len(s)), index=[f"none{i}" for i in range(2 - len(s))])])
        best, second = s.index[0], s.index[1]
        flag = []
        if s.iloc[0] - s.iloc[1] < a.margin:
            flag.append(f"close:{second}")
        if s.iloc[1] > a.doublet_z:
            flag.append(f"two-lineage:{second}")
        if pd.notna(sc_lin.loc[c, "Proliferating"]) and sc_lin.loc[c, "Proliferating"] > a.doublet_z:
            flag.append("proliferating")
        m = A.obs.cluster == c
        row = dict(cluster=c, cells=int(m.sum()), samples=int(A.obs.loc[m, "sample"].nunique()),
                   top_sample_share=float(A.obs.loc[m, "sample"].value_counts(normalize=True).iloc[0]),
                   proposed=best, coarse=best if not any(f.startswith("two-lineage") for f in flag) else "",
                   flag=";".join(flag), **{f"z_{k}": round(float(v), 2) for k, v in sc_lin.loc[c].items()})
        if amp and AMP_SHEETS.get(best):
            sets = {f"{st}": g for sh in AMP_SHEETS[best] for st, g in amp.get(sh, {}).items()}
            sa = score_clusters(A[A.obs.cluster.isin(sc_lin.index)], "cluster", sets).loc[c].sort_values(ascending=False)
            row["amp_state"] = sa.index[0] if len(sa) else ""
            row["amp_state_2"] = sa.index[1] if len(sa) > 1 else ""
        rows.append(row)
    T = pd.DataFrame(rows).sort_values(["proposed", "cells"], ascending=[True, False])
    T.to_csv(work / "clusters.csv", index=False)
    A.X = A.layers.pop("counts")
    A.obsm["X_umap_input"] = H.obsm[rep]
    A.write_h5ad(work / "cells.h5ad", compression="gzip")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    cols = ["cluster", "cells", "samples", "top_sample_share", "proposed", "coarse", "flag"] + \
           [f"z_{k}" for k in LINEAGE] + (["amp_state"] if amp else [])
    print("\n" + T[cols].to_string(index=False))
    print(f"\n{A.n_obs:,} cells, {T.shape[0]} clusters; proposed classes (cells):\n"
          + T.groupby("coarse").cells.sum().sort_values(ascending=False).to_string())
    print(f"wrote {work}/cells.h5ad and {work}/clusters.csv; review, copy to clusters_reviewed.csv if you change it")


def cmd_build(a):
    import anndata as ad
    work = Path(a.work).expanduser()
    out = Path(a.out).expanduser()
    if out.exists():
        sys.exit(f"{out} exists; not overwritten")
    ct = Path(a.clusters).expanduser() if a.clusters else work / "clusters.csv"
    T = pd.read_csv(ct, dtype={"cluster": str}, keep_default_na=False)
    A = ad.read_h5ad(work / "cells.h5ad")
    cmap = dict(zip(T.cluster.astype(str), T.coarse.astype(str)))
    smap = dict(zip(T.cluster.astype(str), T.get("amp_state", pd.Series([""] * len(T))).astype(str)))
    A.obs["coarse"] = A.obs.cluster.astype(str).map(cmap).fillna("")
    A.obs["amp_state"] = A.obs.cluster.astype(str).map(smap).fillna("")
    # coarse "X|Y" in the review splits a cluster cell by cell (e.g. B and plasma cells merged by Harmony):
    # each cell goes to the class whose lineage genes (log-normalized per cell, z-scored within the cluster) score higher;
    # cells scoring no higher on either than the cluster mean are dropped
    for cl, val in list(cmap.items()):
        if "|" not in val:
            continue
        classes = [c for c in val.split("|") if c]
        m = np.flatnonzero(A.obs.cluster.astype(str).to_numpy() == cl)
        S = []
        for k in classes:
            gg = [g for g in LINEAGE.get(k, []) if g in A.var_names]
            if not gg:
                sys.exit(f"cluster {cl}: no lineage genes for {k}")
            # X of cells.h5ad holds raw counts (cmd_cluster restores them before writing); z-scores of raw
            # counts follow each cell's depth, so normalize per cell (counts per 10,000, log1p) first
            tot = np.asarray(A[m].X.sum(1)).ravel()
            X = A[m, gg].X
            X = X.toarray() if sp.issparse(X) else np.asarray(X)
            X = np.log1p(X / np.maximum(tot, 1)[:, None] * 1e4)
            S.append(((X - X.mean(0)) / np.maximum(X.std(0), 1e-9)).mean(1))
        S = np.vstack(S)
        best = S.argmax(0)
        lab = np.where(S.max(0) > 0, np.array(classes)[best], "")
        A.obs.iloc[m, A.obs.columns.get_loc("coarse")] = lab
        print(f"cluster {cl} split: " + ", ".join(f"{k} {int((lab == k).sum())}" for k in classes)
              + f", dropped {int((lab == '').sum())}", flush=True)
    A = A[A.obs.coarse != ""].copy()
    rng = np.random.default_rng(a.seed)
    keep = []
    for k, g in A.obs.groupby("coarse", observed=True):
        idx = np.flatnonzero(A.obs.coarse.to_numpy() == k)
        keep.append(idx if len(idx) <= a.per_type else rng.choice(idx, a.per_type, replace=False))
    A = A[np.sort(np.concatenate(keep))].copy()
    if a.genes_from:
        files = sorted(glob.glob(os.path.expanduser(a.genes_from)))
        genes = set().union(*[set(ad.read_h5ad(f, backed="r").var_names.astype(str)) for f in files])
        A = A[:, [g for g in A.var_names if g in genes]].copy()
        print(f"{A.n_vars:,} panel genes kept (of {len(genes):,} in {len(files)} spatial files)", flush=True)
    A.obs["donor_id"] = A.obs["sample"].astype(str)
    A.var["feature_name"] = A.var_names.astype(str)
    for c in ("donor_id", "coarse", "amp_state", "cluster", "sample", "diagnosis"):
        A.obs[c] = A.obs[c].astype(str).astype("category")
    A.uns["source"] = json.dumps(dict(geo="GSE200815", paper="Floudas et al., Ann Rheum Dis 2022",
                                      amp_markers="Zhang et al., Nature 2023, Supplementary Table 5",
                                      clusters=str(ct), per_type=a.per_type, seed=a.seed))
    tab = (A.obs.groupby(["coarse", "amp_state"], observed=True)
           .agg(n=("donor_id", "size"), donors=("donor_id", "nunique")).reset_index())
    tab.to_csv(str(out.with_suffix("")) + "_types.csv", index=False)
    A.write_h5ad(out, compression="gzip")
    print(f"wrote {out}: {A.n_obs:,} cells x {A.n_vars:,} genes; donors {A.obs.donor_id.nunique()}")
    print(A.obs.coarse.value_counts().to_string())


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cluster")
    c.add_argument("--tar", required=True)
    c.add_argument("--work", required=True)
    c.add_argument("--amp-markers", default="")
    c.add_argument("--diagnoses", default="RA", help="RA, PsA or RA,PsA")
    c.add_argument("--min-umi", type=int, default=500)
    c.add_argument("--min-genes", type=int, default=250)
    c.add_argument("--max-mito", type=float, default=0.25)
    c.add_argument("--doublet-pct", type=float, default=1.0)
    c.add_argument("--hvg", type=int, default=2000)
    c.add_argument("--pcs", type=int, default=30)
    c.add_argument("--resolution", type=float, default=1.0)
    c.add_argument("--no-harmony", dest="harmony", action="store_false",
                   help="cluster on plain PCA, without Harmony")
    c.add_argument("--margin", type=float, default=0.5)
    c.add_argument("--doublet-z", type=float, default=1.0)
    c.add_argument("--min-expr", type=float, default=0.5,
                   help="minimum cluster mean of the per-cell maximum log-normalised expression over a lineage's genes")
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--clusters", default="")
    b.add_argument("--genes-from", default="")
    b.add_argument("--per-type", type=int, default=3000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cluster": cmd_cluster, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
