#!/usr/bin/env python
"""gut_reference.py -- a single-cell reference of the human small and large intestine from the CZ CELLxGENE Census,
for labeling the IBD spatial data with compartment_label.py.

Cells: primary data (is_primary_data), tissue_general small or large intestine, the diseases given (default normal,
Crohn disease, ulcerative colitis), 10x assays; at most --per-type cells per fine cell type (random, seed), so common
types do not swamp rare ones. Genes: those named in the spatial panels (--genes-from, the converted h5ad files), so the
file stays small. Counts are the Census raw counts.
obs: donor_id, dataset_id, tissue, disease, assay, cell_type (Cell Ontology), coarse (the regex map below).
Coarse level (for compartment_label.py --ref-levels L1=coarse; cell types matching no rule are dropped there):
  Epithelial, T_NK, B, Plasma, Myeloid (incl. granulocytes), Mast, Fibroblast, Muscle, Endothelial, Neural
EXACT names win over the regex rules. Prenatal donors (development_stage matching PRENATAL) are excluded: the IBD
spatial data are adult tissue, and fetal gut brings neuroblasts, progenitors and erythroblasts.
v2: EXACT overrides, prenatal filter, census version pinned (2025-11-08), types table with datasets.

  pip install cellxgene-census
  python gut_reference.py --genes-from '~/claimbreak_data/IBD/h5ad/*.h5ad' \
      --out ~/claimbreak_data/IBD/ref/gut_census.h5ad
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

COARSE = [  # case-sensitive; first match wins; order matters (plasma before B, mast before myeloid); "" = dropped
    ("", r"hematopoietic|erythro|platelet|megakaryocyte|progenitor"),
    ("Plasma", r"\bplasma cell|plasmablast"),
    ("B", r"\bB cell|\bB-cell"),
    ("T_NK", r"\bT cell|\bT-cell|\bT-helper|natural killer|\bNK\b|innate lymphoid|thymocyte"),
    ("Mast", r"\bmast cell"),
    ("Granulocyte", r"neutrophil|eosinophil|basophil|granulocyte"),
    ("Myeloid", r"macrophage|monocyte|dendritic|myeloid|[Ll]angerhans"),
    ("Endothelial", r"endothelial"),
    ("Muscle", r"smooth muscle|pericyte|myocyte"),
    ("Fibroblast", r"fibroblast|stromal|mesenchymal|mesothelial|reticular"),
    ("Neural", r"glial|neuron|Schwann|enteric glia"),
    ("Epithelial", r"enterocyte|epithelial|goblet|stem cell|transit amplifying|[Pp]aneth|enteroendocrine|tuft|"
                   r"colonocyte|\bM cell|BEST4|absorptive|secretory|\bL cell|enterochromaffin|crypt"),
]


EXACT = {  # checked one by one against the Census 2025-11-08 type list; these win over COARSE
    "intraepithelial lymphocyte": "T_NK",          # "epithelial" in the name, but a T cell
    "T follicular helper cell": "T_NK",
    "natural T-regulatory cell": "T_NK",
    "follicular dendritic cell": "Fibroblast",     # stromal, not myeloid
    "interstitial cell of Cajal": "Muscle",        # muscularis mesenchyme (KIT/ANO1)
    "GIP cell": "Epithelial",                      # enteroendocrine K cell
    "microfold cell of epithelium of small intestine": "Epithelial",
    "mononuclear phagocyte": "Myeloid",
    "neutrophil": "Myeloid", "eosinophil": "Myeloid", "granulocyte": "Myeloid",   # too few (< 200) for a class
    "stem cell": "",                               # ambiguous (haematopoietic or crypt)
    "epithelial cell of lung": "", "alveolar macrophage": "",
    "oligodendrocyte precursor cell": "",          # CNS type
    "mesenchymal lymphangioblast": "",
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
    ap.add_argument("--diseases", default="normal|Crohn disease|ulcerative colitis")
    ap.add_argument("--per-type", type=int, default=3000)
    ap.add_argument("--census-version", default="2025-11-08")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; not overwritten")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    genes = None
    if a.genes_from:
        import anndata as ad
        files = sorted(glob.glob(os.path.expanduser(a.genes_from)))
        genes = sorted(set().union(*[set(ad.read_h5ad(f, backed="r").var_names.astype(str)) for f in files]))
        print(f"{len(genes):,} genes from {len(files)} spatial files", flush=True)
    import cellxgene_census
    dis = a.diseases.split("|")
    flt = ("is_primary_data == True and tissue_general in ['small intestine', 'large intestine'] and "
           f"disease in {dis}")
    with cellxgene_census.open_soma(census_version=a.census_version) as c:
        hs = c["census_data"]["homo_sapiens"]
        cols = ["soma_joinid", "donor_id", "dataset_id", "tissue", "disease", "assay", "cell_type", "development_stage"]
        obs = hs.obs.read(value_filter=flt, column_names=cols).concat().to_pandas()
        obs = obs[obs.assay.str.contains("10x", case=False, na=False)]
        pre = obs.development_stage.astype(str).str.contains(PRENATAL, case=False, regex=True)
        print(f"prenatal cells excluded: {pre.sum():,} ({obs[pre].donor_id.nunique()} donors, "
              f"{obs[pre].dataset_id.nunique()} datasets)", flush=True)
        obs = obs[~pre]
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
