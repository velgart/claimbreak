#!/usr/bin/env python
"""uterus_reference.py -- a single-cell reference of the adult human uterus (endometrium, myometrium, endometrial
cancer) from the CZ CELLxGENE Census, for labeling the MERSCOPE uterine cancer data with `claimbreak compartment
label` (adapted from examples/ain/kidney_reference.py).

Cells: primary data, tissue_general uterus, tissues matching --tissues (uterus, endometrium, myometrium: the
Census group also holds ovary and fallopian tube) and not --exclude-tissue (adnexa of uterus: the ovarian
serous tumor data; decidua, placenta: pregnancy),
every disease unless --diseases is given, 10x assays, adult (prenatal excluded); at most --per-type
cells per fine cell type. Genes: those of the spatial panel (--genes-from). Counts: Census raw counts.
obs: donor_id, dataset_id, tissue, disease, assay, cell_type, coarse.
Coarse level (--ref-levels L1=coarse; cell types matching no rule are dropped):
  Epithelial (luminal, glandular, ciliated, malignant), Stromal (endometrial stroma, fibroblasts), Mural (smooth
  muscle of the myometrium and vessels, pericytes), Endothelial (incl. lymphatic), T_NK, B, Plasma, Myeloid, Mast,
  Neural; trophoblasts, germ and progenitor cells dropped
Run --list-types first: it prints every Census cell type with its count and coarse class without downloading counts.

  python uterus_reference.py --list-types --out ~/claimbreak_data/data/merscope_uterine_cd45ro/ref/uterus_census.h5ad
  python uterus_reference.py --genes-from '~/claimbreak_data/data/merscope_uterine_cd45ro/h5ad/*.h5ad' \
      --out ~/claimbreak_data/data/merscope_uterine_cd45ro/ref/uterus_census.h5ad
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

COARSE = [  # first match wins; "" = dropped
    ("", r"trophoblast|hematopoietic|erythro|platelet|megakaryocyte|progenitor|stem cell|oocyte|germ cell|granulosa"),
    ("Plasma", r"\bplasma cell|plasmablast"),
    ("B", r"\bB cell|\bB-cell"),
    ("T_NK", r"\bT cell|\bT-cell|\bT-helper|natural killer|\bNK\b|innate lymphoid"),
    ("Mast", r"\bmast cell"),
    ("Myeloid", r"macrophage|monocyte|dendritic|myeloid|phagocyte|neutrophil|eosinophil|basophil|granulocyte"),
    ("Endothelial", r"endothelial"),
    ("Mural", r"smooth muscle|pericyte|myocyte|muscle cell|myometrial"),
    ("Stromal", r"fibroblast|stromal|mesenchymal|decidual cell"),
    ("Neural", r"Schwann|neuron|glial"),
    ("Epithelial", r"epithelial|ciliated|glandular|luminal|secretory|malignant|tumou?r|cancer|carcinoma|mesothelial"),
]
EXACT = {}  # filled after --list-types if a type is mis-sorted
PRENATAL = r"post-fertilization|Carnegie|embryonic|fetal"


def coarse(ct):
    if ct in EXACT:
        return EXACT[ct]
    for name, rx in COARSE:
        if re.search(rx, ct, flags=re.I if name in ("Epithelial", "Stromal", "Mural") else 0):
            return name
    return ""

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--genes-from", default="", help="glob of compartment h5ad files; their genes are kept")
    ap.add_argument("--diseases", default="", help="| list (default: all); --list-types prints the diseases present")
    ap.add_argument("--tissues", default="uterus|endometri|myometri",
                    help="regex on tissue; only matching cells are kept (Census tissue_general 'uterus' also holds "
                         "ovary and fallopian tube)")
    ap.add_argument("--exclude-tissue", default="adnexa|ovar|fallopian|decidua|placent|chorion|amni|umbilical",
                    help="regex on tissue; matching cells are dropped")
    ap.add_argument("--list-types", action="store_true", help="print cell types, counts and coarse class; no download")
    ap.add_argument("--per-type", type=int, default=3000)
    ap.add_argument("--census-version", default="2025-11-08")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.list_types:
        sys.exit(f"{a.out} exists; not overwritten")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    genes = None
    if a.genes_from and not a.list_types:
        import anndata as ad
        files = sorted(glob.glob(os.path.expanduser(a.genes_from)))
        genes = sorted(set().union(*[set(ad.read_h5ad(f, backed="r").var_names.astype(str)) for f in files]))
        print(f"{len(genes):,} genes from {len(files)} spatial files", flush=True)
    import cellxgene_census
    flt = "is_primary_data == True and tissue_general == 'uterus'"
    if a.diseases:
        flt += f" and disease in {a.diseases.split('|')}"
    with cellxgene_census.open_soma(census_version=a.census_version) as c:
        hs = c["census_data"]["homo_sapiens"]
        cols = ["soma_joinid", "donor_id", "dataset_id", "tissue", "disease", "assay", "cell_type", "development_stage"]
        obs = hs.obs.read(value_filter=flt, column_names=cols).concat().to_pandas()
        obs = obs[obs.assay.str.contains("10x", case=False, na=False)]
        tis = obs.tissue.astype(str)
        keep_t = tis.str.contains(a.tissues, case=False, regex=True) if a.tissues else pd.Series(True, index=obs.index)
        drop_t = tis.str.contains(a.exclude_tissue, case=False, regex=True) if a.exclude_tissue else False
        keep_t = keep_t & ~drop_t
        tt = (pd.DataFrame({"tissue": tis, "disease": obs.disease.astype(str), "kept": keep_t})
              .groupby(["kept", "tissue", "disease"]).size().rename("cells").reset_index()
              .sort_values(["kept", "cells"], ascending=[False, False]))
        print(f"tissues (kept: matching '{a.tissues}' and not '{a.exclude_tissue}'):\n" + tt.to_string(index=False),
              flush=True)
        obs = obs[keep_t.to_numpy()]
        pre = obs.development_stage.astype(str).str.contains(PRENATAL, case=False, regex=True)
        print(f"prenatal cells excluded: {pre.sum():,} ({obs[pre].donor_id.nunique()} donors, "
              f"{obs[pre].dataset_id.nunique()} datasets)", flush=True)
        obs = obs[~pre]
        if a.list_types:
            allobs = hs.obs.read(value_filter="is_primary_data == True and tissue_general == 'uterus'",
                                 column_names=["disease"]).concat().to_pandas()
            print("diseases (all uterus primary cells):\n" + allobs.disease.astype(str).value_counts().to_string(), flush=True)
            t = (obs.groupby("cell_type", observed=True)
                 .agg(n=("donor_id", "size"), donors=("donor_id", "nunique"), datasets=("dataset_id", "nunique"))
                 .reset_index())
            t["coarse"] = [coarse(str(x)) or "(dropped)" for x in t.cell_type]
            t = t.sort_values(["coarse", "n"], ascending=[True, False])
            t.to_csv(os.path.splitext(a.out)[0] + "_list_types.csv", index=False)
            print(t.to_string(index=False), flush=True)
            print(f"wrote {os.path.splitext(a.out)[0]}_list_types.csv", flush=True)
            return
        print(f"{len(obs):,} cells, {obs.donor_id.nunique()} donors, {obs.dataset_id.nunique()} datasets, "
              f"{obs.cell_type.nunique()} cell types", flush=True)
        rng = np.random.default_rng(a.seed)
        pick = []
        for ct, g in obs.groupby("cell_type", observed=True):
            ids = g.soma_joinid.to_numpy()
            pick.append(ids if len(ids) <= a.per_type else rng.choice(ids, a.per_type, replace=False))
        ids = np.sort(np.concatenate(pick))
        print(f"sampling {len(ids):,} cells (at most {a.per_type} per cell type)", flush=True)
        vf = f"feature_name in {genes}" if genes else None
        A = cellxgene_census.get_anndata(c, organism="Homo sapiens", obs_coords=ids, var_value_filter=vf,
                                         obs_column_names=cols[1:])
    A.obs["coarse"] = [coarse(str(x)) for x in A.obs["cell_type"]]
    A.var_names = A.var["feature_name"].astype(str).to_numpy()
    A.var_names_make_unique()
    A.uns["source"] = json.dumps(dict(census_version=a.census_version, filter=flt, per_type=a.per_type,
                                      seed=a.seed, tissues=a.tissues, exclude_tissue=a.exclude_tissue, coarse_rules=COARSE, exact=EXACT, prenatal_excluded=PRENATAL, n_genes_requested=len(genes or [])))
    tab = (A.obs.groupby(["coarse", "cell_type"], observed=True)
           .agg(n=("donor_id", "size"), donors=("donor_id", "nunique"), datasets=("dataset_id", "nunique"))
           .reset_index())
    tab.to_csv(os.path.splitext(a.out)[0] + "_types.csv", index=False)
    A.write_h5ad(a.out, compression="gzip")
    print(f"wrote {a.out}: {A.n_obs:,} cells x {A.n_vars:,} genes", flush=True)
    print(A.obs.coarse.replace("", "(dropped)").value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
