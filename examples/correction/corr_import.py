#!/usr/bin/env python
"""corr_import.py -- labels of corrected counts, as arms of the claimbreak labels files.

For every section exported by corr_export.py (window cells only) writes OUT/<sample>.labels.csv.gz with the columns
of the original labels file (arms nucleus, whole, nucA, nucB, wholeB, ...) and the new arms
  <level>_rerun     our labeler on the original whole-cell counts of the window (check: equals <level>_whole)
  <level>_split     our labeler on SPLIT-purified counts (run_split.R split_counts)
  <level>_splitsp   our labeler on spatially aware SPLIT counts (splitsp_counts)
  <level>_shiftc    our labeler on SPLIT-shift counts (shift_counts)
  <level>_admix     our labeler on cellAdmix-cleaned counts (run_celladmix.R admix_counts)
  <level>_rctd      RCTD first type (doublet mode; reject -> unlabeled)
  <level>_shift     SPLIT-shift label (first type after swapping)
Our labeler uses the reference, the labeling settings (<sample>.meta.json in --labels) and the whole arm's platform
factor (<sample>.gamma.csv, not refitted) of the original run, so a change of label comes from the counts alone. Cells absent from a tool's output are unlabeled in that arm.
Per cell, for the corrected count sets, the share of the cell's counts removed in the marker genes of its nucleus type
and of its whole-cell type (columns rm_nuc_<arm>, rm_whole_<arm>; markers: the 10 genes with the largest log ratio of
the type's reference profile over the next type's).

  python corr_import.py --input $C/input --split $C/split --admix $C/admix --labels ~/claimbreak_results/bilous_v4 \
      --ref ~/claimbreak_data/Bilous/ref/<file>.h5ad --out ~/claimbreak_results/bilous_corr_v1
"""
import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from claimbreak.compartment import LabelConfig, Reference, label_section


def read_counts(d, stem, genes_file=None, cells_file=None):
    """genes x cells Matrix Market -> (cells x genes csr, genes, cells)."""
    M = sp.csr_matrix(sio.mmread(d / f"{stem}.mtx").T)
    g = (d / (genes_file or f"{stem.replace('_counts', '')}_genes.txt")).read_text().split()
    c = (d / (cells_file or f"{stem.replace('_counts', '')}_cells.txt")).read_text().split()
    return M, np.array(g), np.array(c)


def align(M, genes, cells, to_genes, to_cells):
    """Reindex to the window's genes and cells (absent genes and cells: zero)."""
    gi = pd.Index(genes).get_indexer(to_genes)
    ci = pd.Index(cells).get_indexer(to_cells)
    Mc = M.tocsr()
    rows = np.where(ci >= 0)[0]
    cols = np.where(gi >= 0)[0]
    sub = Mc[ci[rows]][:, gi[cols]]
    R = sp.csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(len(to_cells), len(rows)))
    C = sp.csr_matrix((np.ones(len(cols)), (np.arange(len(cols)), cols)), shape=(len(cols), len(to_genes)))
    return sp.csr_matrix(R @ sub @ C, dtype=np.float32), np.isin(np.arange(len(to_cells)), rows)


def cfg_from_meta(meta):
    names = {f.name for f in fields(LabelConfig)}
    kw = {k: v for k, v in meta.items() if k in names}
    for k in ("gamma_clip", "exclude_genes"):
        if k in kw and kw[k] is not None:
            kw[k] = tuple(kw[k])
    return LabelConfig(**kw)


def markers(ref, level, genes, n=10):
    g, gidx, _ = ref.match(list(genes))
    types, mu = ref.profiles(level, gidx)
    lr = np.log2((mu + 1e-6) / (np.sort(mu, 0)[-2][None, :] + 1e-6))
    out = {}
    for k, t in enumerate(types):
        own = mu[k] >= mu.max(0)                      # genes where the type is highest
        cand = np.where(own)[0]
        top = cand[np.argsort(-lr[k, cand])[:n]]
        out[t] = [g[i] for i in top]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="corr_export.py output")
    ap.add_argument("--split", default="", help="folder of run_split.R outputs (one subfolder per section)")
    ap.add_argument("--admix", default="", help="folder of run_celladmix.R outputs")
    ap.add_argument("--labels", required=True, help="the original labels folder (with <sample>.meta.json)")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-level-col", default="Level2")
    ap.add_argument("--ref-merge", default=r"^Tu_L\d=>Tumour")
    ap.add_argument("--ref-donor", default="donor_id")
    ap.add_argument("--level", default="L2m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import anndata as ad
    inp, out, lab_dir = Path(a.input).expanduser(), Path(a.out).expanduser(), Path(a.labels).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    lv = a.level
    ref = Reference.load(a.ref, {lv: a.ref_level_col}, merge=a.ref_merge, donor=a.ref_donor)
    summary = []
    for sdir in sorted(p for p in inp.iterdir() if p.is_dir() and p.name != "ref"):
        sample = sdir.name
        meta = json.loads((lab_dir / f"{sample}.meta.json").read_text())
        cfg = cfg_from_meta(meta)
        cells = pd.read_csv(sdir / "cells.csv", dtype={"cell_id": str}, keep_default_na=False)
        W = sp.csr_matrix(sio.mmread(sdir / "counts.mtx").T, dtype=np.float32)
        genes = np.array((sdir / "genes.txt").read_text().split())
        cid = cells.cell_id.to_numpy()
        L = pd.read_csv(lab_dir / f"{sample}.labels.csv.gz", keep_default_na=False, low_memory=False,
                        dtype={"cell_id": str})
        L = L.set_index("cell_id").reindex(cid).reset_index()
        mk = markers(ref, lv, genes)
        gam = pd.read_csv(lab_dir / f"{sample}.gamma.csv")     # platform factor of the whole arm, original run
        gfix = {f"{lv}_whole": dict(zip(gam["gene"].astype(str), gam[f"{lv}_whole"]))}
        obs = pd.DataFrame({"x": cells.x.to_numpy(float), "y": cells.y.to_numpy(float)}, index=cid)

        def relabel(X, name):
            A = ad.AnnData(X=sp.csr_matrix(X, dtype=np.float32), obs=obs.copy(), var=pd.DataFrame(index=genes))
            A.layers["nucleus"] = sp.csr_matrix(X.shape, dtype=np.float32)
            A.obs["n_whole"] = np.asarray(A.X.sum(1)).ravel()
            A.obs["n_nucleus"] = 0.0
            res, _, _ = label_section(A, ref, [lv], ["whole"], cfg, a.seed, sample, gamma_fixed=gfix)
            L[f"{lv}_{name}"] = res[f"{lv}_whole"].to_numpy()
            L[f"{lv}_{name}_second"] = res[f"{lv}_whole_second"].to_numpy()
            L[f"{lv}_{name}_margin"] = res[f"{lv}_whole_margin"].to_numpy()
            L[f"n_{name}"] = np.asarray(X.sum(1)).ravel()

        def removed(X, name):
            gi = {g: i for i, g in enumerate(genes)}
            nucl, whol = L[f"{lv}_nucleus"].astype(str).to_numpy(), L[f"{lv}_whole"].astype(str).to_numpy()
            for col, labs in ((f"rm_nuc_{name}", nucl), (f"rm_whole_{name}", whol)):
                v = np.full(len(cid), np.nan)
                for t in set(labs) - {""}:
                    idx = [gi[g] for g in mk.get(t, []) if g in gi]
                    rows = np.where(labs == t)[0]
                    if not idx or not len(rows):
                        continue
                    o = np.asarray(W[rows][:, idx].sum(1)).ravel()
                    c = np.asarray(X[rows][:, idx].sum(1)).ravel()
                    v[rows] = np.where(o > 0, 1 - c / np.maximum(o, 1e-9), np.nan)
                L[col] = v

        relabel(W, "rerun")
        info = dict(sample=sample, cells=len(cid), rerun_equals_whole=float(np.mean(
            L[f"{lv}_rerun"].astype(str) == L[f"{lv}_whole"].astype(str))))
        sd = Path(a.split).expanduser() / sample if a.split else None
        if sd is not None and sd.exists():
            for stem, name in (("split_counts", "split"), ("splitsp_counts", "splitsp"), ("shift_counts", "shiftc")):
                if (sd / f"{stem}.mtx").exists():
                    M, g2, c2 = read_counts(sd, stem)
                    X, present = align(M, g2, c2, genes, cid)
                    relabel(X, name)
                    L.loc[~present, f"{lv}_{name}"] = ""
                    removed(X, name)
                    info[f"{name}_cells"] = int(present.sum())
            rd = pd.read_csv(sd / "rctd.csv", dtype={"cell_id": str}, keep_default_na=False)
            ft = rd.set_index("cell_id")
            lab = ft["first_type"].where(ft["spot_class"] != "reject", "")
            L[f"{lv}_rctd"] = lab.reindex(cid).fillna("").replace({"NA": ""}).to_numpy()
            L["rctd_spot_class"] = ft["spot_class"].reindex(cid).fillna("").to_numpy()
            if (sd / "shift_meta.csv").exists():
                sm = pd.read_csv(sd / "shift_meta.csv", dtype={"cell_id": str}, keep_default_na=False).set_index("cell_id")
                L[f"{lv}_shift"] = sm["first_type"].reindex(cid).fillna("").replace({"NA": ""}).to_numpy()
                L["shift_swap"] = sm["swap"].reindex(cid).astype(str).replace({"nan": ""}).to_numpy()
                info["shift_swapped"] = int((sm["swap"].astype(str).str.upper() == "TRUE").sum())
        ad_dir = Path(a.admix).expanduser() / sample if a.admix else None
        if ad_dir is not None and (ad_dir / "admix_counts.mtx").exists():
            M, g2, c2 = read_counts(ad_dir, "admix_counts")
            X, present = align(M, g2, c2, genes, cid)
            relabel(X, "admix")
            L.loc[~present, f"{lv}_admix"] = ""
            removed(X, "admix")
            info["admix_counts_removed"] = float(1 - X.sum() / max(W.sum(), 1))
        for arm in ("split", "splitsp", "shiftc", "admix", "rctd", "shift"):
            c = f"{lv}_{arm}"
            if c in L:
                ok = (L[c].astype(str) != "") & (L[f"{lv}_whole"].astype(str) != "")
                info[f"{arm}_equals_whole"] = float(np.mean(L.loc[ok, c].astype(str) == L.loc[ok, f"{lv}_whole"].astype(str)))
        L.to_csv(out / f"{sample}.labels.csv.gz", index=False, compression="gzip")
        (out / f"{sample}.meta.json").write_text(json.dumps({**meta, "correction_import": info}, indent=1))
        summary.append(info)
        print(json.dumps(info), flush=True)
    pd.DataFrame(summary).to_csv(out / "import_summary.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
