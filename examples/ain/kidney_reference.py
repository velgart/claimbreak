#!/usr/bin/env python
"""kidney_reference.py -- a single-cell reference of the adult human kidney from the CZ CELLxGENE Census, for labeling
the acute interstitial nephritis Xenium data (Baker et al. 2026, GSE336890) with `claimbreak compartment label`
(adapted from examples/ibd/gut_reference.py v2).

Cells: primary data, tissue_general kidney, the diseases given, 10x assays, adult (prenatal excluded); at most
--per-type cells per fine cell type. Genes: those of the spatial panel (--genes-from). Counts: Census raw counts.
obs: donor_id, dataset_id, tissue, disease, assay, cell_type, coarse.
Coarse level (--ref-levels L1=coarse; cell types matching no rule are dropped):
  PT, LOH (thin limbs and thick ascending limb), Distal_CD (DCT, connecting tubule, collecting duct), Podocyte,
  PEC (parietal and other epithelium), Endothelial, Mural (vSMC, pericyte, mesangial), Stromal (fibroblasts),
  T_NK, B, Plasma, Myeloid (incl. granulocytes), Mast, Neural
Run --list-types first: it prints every Census cell type with its count and coarse class without downloading counts,
so the mapping can be checked (and EXACT extended) before the reference is built.

  pip install cellxgene-census
  python kidney_reference.py --list-types --out ~/claimbreak_data/AIN/ref/kidney_census.h5ad
  python kidney_reference.py --genes-from '~/claimbreak_data/AIN/h5ad/*.h5ad' \
      --out ~/claimbreak_data/AIN/ref/kidney_census.h5ad
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

COARSE = [  # first match wins; "" = dropped
    ("", r"hematopoietic|erythro|platelet|megakaryocyte|progenitor|stem cell|blastema|nephron progenitor"),
    ("Plasma", r"\bplasma cell|plasmablast"),
    ("B", r"\bB cell|\bB-cell"),
    ("T_NK", r"\bT cell|\bT-cell|\bT-helper|natural killer|\bNK\b|innate lymphoid"),
    ("Mast", r"\bmast cell"),
    ("Myeloid", r"macrophage|monocyte|dendritic|myeloid|phagocyte|neutrophil|eosinophil|basophil|granulocyte"),
    ("Podocyte", r"podocyte"),
    ("PT", r"proximal (convoluted |straight )?tubul|proximal tubule"),
    ("LOH", r"loop of Henle|thin (descending|ascending) limb|thick ascending|ascending limb"),
    ("Distal_CD", r"distal convoluted|connecting tubule|collecting duct|principal cell|intercalated cell|macula densa"),
    ("Endothelial", r"endothelial"),
    ("Mural", r"smooth muscle|pericyte|mesangial|juxtaglomerular"),
    ("Stromal", r"fibroblast|stromal|mesenchymal|interstitial cell"),
    ("Neural", r"Schwann|neuron|glial"),
    ("PEC", r"parietal epithelial|epithelial|urothelial"),
]
EXACT = {  # reviewed against --list-types (Census 2025-11-08); these win over COARSE
    "kidney epithelial cell": "",                  # generic, segment unknown
    "epithelial cell": "",                         # generic
    "vasa recta ascending limb cell": "Endothelial",   # vasa recta are vessels ("ascending limb" would match LOH)
    "vasa recta descending limb cell": "Endothelial",
    "myofibroblast cell": "Stromal",               # interstitial in kidney
}
PRENATAL = r"post-fertilization|Carnegie|embryonic|fetal"


def coarse(ct):
    if ct in EXACT:
        return EXACT[ct]
    for name, rx in COARSE:
        if re.search(rx, ct):
            return name
    return ""

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--genes-from", default="", help="glob of compartment h5ad files; their genes are kept")
    ap.add_argument("--diseases", default="normal|acute kidney failure|chronic kidney disease",
                    help="| list; --list-types prints the diseases present")
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
    dis = a.diseases.split("|")
    flt = f"is_primary_data == True and tissue_general == 'kidney' and disease in {dis}"
    with cellxgene_census.open_soma(census_version=a.census_version) as c:
        hs = c["census_data"]["homo_sapiens"]
        cols = ["soma_joinid", "donor_id", "dataset_id", "tissue", "disease", "assay", "cell_type", "development_stage"]
        obs = hs.obs.read(value_filter=flt, column_names=cols).concat().to_pandas()
        obs = obs[obs.assay.str.contains("10x", case=False, na=False)]
        pre = obs.development_stage.astype(str).str.contains(PRENATAL, case=False, regex=True)
        print(f"prenatal cells excluded: {pre.sum():,} ({obs[pre].donor_id.nunique()} donors, "
              f"{obs[pre].dataset_id.nunique()} datasets)", flush=True)
        obs = obs[~pre]
        if a.list_types:
            allobs = hs.obs.read(value_filter="is_primary_data == True and tissue_general == 'kidney'",
                                 column_names=["disease"]).concat().to_pandas()
            print("diseases (all kidney primary cells):\n" + allobs.disease.value_counts().to_string(), flush=True)
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
                                      seed=a.seed, coarse_rules=COARSE, exact=EXACT, prenatal_excluded=PRENATAL, n_genes_requested=len(genes or [])))
    tab = (A.obs.groupby(["coarse", "cell_type"], observed=True)
           .agg(n=("donor_id", "size"), donors=("donor_id", "nunique"), datasets=("dataset_id", "nunique"))
           .reset_index())
    tab.to_csv(os.path.splitext(a.out)[0] + "_types.csv", index=False)
    A.write_h5ad(a.out, compression="gzip")
    print(f"wrote {a.out}: {A.n_obs:,} cells x {A.n_vars:,} genes", flush=True)
    print(A.obs.coarse.replace("", "(dropped)").value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
