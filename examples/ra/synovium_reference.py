#!/usr/bin/env python
"""synovium_reference.py -- a single-cell reference of human synovium from the CZ CELLxGENE Census, for labeling the
RA Xenium data (Bhamidipati et al. 2026) with `claimbreak compartment label` (adapted from examples/ain/kidney_reference.py).

Three steps, each a separate call:
  --list-tissues   Census tissue terms matching --tissue-regex (default synov|joint|knee|hip), with cell counts, from
                   the Census summary table (no cell download). Choose the terms for --tissues.
  --list-types     for --tissues: diseases present, and every cell type with cells, donors, datasets and its coarse
                   class; writes <out stem>_list_types.csv. Review the mapping (COARSE, EXACT) before building.
  (build)          primary 10x cells of --tissues and --diseases, prenatal excluded, at most --per-type cells per fine
                   cell type, genes of the spatial panel (--genes-from); writes OUT (counts in X) and <out stem>_types.csv.
obs: donor_id, dataset_id, tissue, disease, assay, cell_type, coarse.
Coarse level (--ref-levels L1=coarse; cell types matching no rule are dropped):
  T_NK, B, Plasma, Myeloid (macrophages, monocytes, dendritic cells, granulocytes), Mast, Fibroblast (lining and
  sublining, synoviocytes), Endothelial, Mural (pericytes, smooth muscle), Adipocyte, Chondro (chondrocytes and
  related), Neural
The inputs are only read; OUT must not exist.

  pip install cellxgene-census
  python synovium_reference.py --list-tissues --out ~/claimbreak_data/data/RA_ref/synovium_census.h5ad
  python synovium_reference.py --list-types --tissues 'synovial membrane of synovial joint' --out ...
  python synovium_reference.py --tissues '...' --genes-from '~/claimbreak_data/data/RA_h5ad/compartment/*.h5ad' --out ...
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

COARSE = [  # first match wins; "" = dropped
    ("", r"hematopoietic|erythro|platelet|megakaryocyte|progenitor|stem cell"),
    ("Plasma", r"\bplasma cell|plasmablast"),
    ("B", r"\bB cell|\bB-cell"),
    ("T_NK", r"\bT cell|\bT-cell|\bT-helper|natural killer|\bNK\b|innate lymphoid|regulatory T"),
    ("Mast", r"\bmast cell"),
    ("Myeloid", r"macrophage|monocyte|dendritic|myeloid|phagocyte|neutrophil|eosinophil|basophil|granulocyte|osteoclast"),
    ("Endothelial", r"endothelial"),
    ("Mural", r"smooth muscle|pericyte|mural"),
    ("Adipocyte", r"adipocyte|fat cell"),
    ("Chondro", r"chondrocyte|chondroblast|osteoblast|osteocyte"),
    ("Neural", r"Schwann|neuron|glial"),
    ("Fibroblast", r"fibroblast|synoviocyte|synovial|stromal|mesenchymal"),
]
EXACT = {  # fine type -> coarse class ("" drops it); these win over COARSE; extend after reviewing --list-types
    "type A synoviocyte": "Myeloid",               # macrophage-like synovial lining cell
    "type B synoviocyte": "Fibroblast",            # fibroblast-like synovial lining cell
}
PRENATAL = r"post-fertilization|Carnegie|embryonic|fetal"


def coarse(ct):
    if ct in EXACT:
        return EXACT[ct]
    for name, rx in COARSE:
        if re.search(rx, ct, flags=re.I):
            return name
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list-tissues", action="store_true")
    ap.add_argument("--tissue-regex", default=r"synov|\bjoint|\bknee\b|\bhip\b")
    ap.add_argument("--list-types", action="store_true")
    ap.add_argument("--tissues", default="", help="| list of exact Census tissue labels; prefix general: to select by "
                                                  "tissue_general instead (composite reference, e.g. "
                                                  "'general:adipose tissue|skin of body|lymph node|musculature')")
    ap.add_argument("--diseases", default="", help="| list (default: all diseases present)")
    ap.add_argument("--genes-from", default="", help="glob of compartment h5ad files; their genes are kept")
    ap.add_argument("--per-type", type=int, default=3000)
    ap.add_argument("--census-version", default="2025-11-08")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    building = not (a.list_tissues or a.list_types)
    if building and os.path.exists(a.out):
        sys.exit(f"{a.out} exists; not overwritten")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    stem = os.path.splitext(a.out)[0]
    import cellxgene_census
    with cellxgene_census.open_soma(census_version=a.census_version) as c:
        if a.list_tissues:
            S = c["census_info"]["summary_cell_counts"].read().concat().to_pandas()
            S = S[(S.organism.str.contains("sapiens", case=False)) & S.category.isin(["tissue", "tissue_general"])]
            S = S[S.label.str.contains(a.tissue_regex, case=False, regex=True)]
            S = S.sort_values("unique_cell_count", ascending=False)
            S.to_csv(stem + "_list_tissues.csv", index=False)
            print(S[["category", "label", "ontology_term_id", "total_cell_count", "unique_cell_count"]].to_string(index=False))
            print(f"wrote {stem}_list_tissues.csv; pass the chosen labels (category tissue) as --tissues 'a|b'")
            return
        field = "tissue"
        sel = a.tissues
        if sel.startswith("general:"):
            field, sel = "tissue_general", sel[len("general:"):]
        tissues = [t for t in sel.split("|") if t]
        if not tissues:
            sys.exit("--tissues is required (run --list-tissues first)")
        flt = f"is_primary_data == True and {field} in {tissues}"
        dis = [d for d in a.diseases.split("|") if d]
        if dis:
            flt += f" and disease in {dis}"
        hs = c["census_data"]["homo_sapiens"]
        cols = ["soma_joinid", "donor_id", "dataset_id", "tissue", "disease", "assay", "cell_type", "development_stage"]
        obs = hs.obs.read(value_filter=flt, column_names=cols).concat().to_pandas()
        n0 = len(obs)
        obs = obs[obs.assay.str.contains("10x", case=False, na=False)]
        pre = obs.development_stage.astype(str).str.contains(PRENATAL, case=False, regex=True)
        print(f"{n0:,} primary cells; {len(obs):,} from 10x assays; prenatal excluded: {pre.sum():,}", flush=True)
        obs = obs[~pre]
        if a.list_types:
            print("tissues:\n" + obs.tissue.value_counts().head(30).to_string())
            print("diseases:\n" + obs.disease.value_counts().to_string())
            print("datasets:\n" + obs.groupby("dataset_id").agg(cells=("donor_id", "size"), donors=("donor_id", "nunique"))
                  .sort_values("cells", ascending=False).to_string())
            t = (obs.groupby("cell_type", observed=True)
                 .agg(n=("donor_id", "size"), donors=("donor_id", "nunique"), datasets=("dataset_id", "nunique"))
                 .reset_index())
            t["coarse"] = [coarse(str(x)) or "(dropped)" for x in t.cell_type]
            t = t.sort_values(["coarse", "n"], ascending=[True, False])
            t.to_csv(stem + "_list_types.csv", index=False)
            print(t.to_string(index=False))
            print("coarse totals:\n" + t.groupby("coarse").n.sum().sort_values(ascending=False).to_string())
            print(f"wrote {stem}_list_types.csv")
            return
        genes = None
        if a.genes_from:
            import anndata as ad
            files = sorted(glob.glob(os.path.expanduser(a.genes_from)))
            genes = sorted(set().union(*[set(ad.read_h5ad(f, backed="r").var_names.astype(str)) for f in files]))
            print(f"{len(genes):,} genes from {len(files)} spatial files", flush=True)
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
    A.uns["source"] = json.dumps(dict(census_version=a.census_version, filter=flt, per_type=a.per_type, seed=a.seed,
                                      coarse_rules=COARSE, exact=EXACT, prenatal_excluded=PRENATAL,
                                      n_genes_requested=len(genes or [])))
    tab = (A.obs.groupby(["coarse", "cell_type"], observed=True)
           .agg(n=("donor_id", "size"), donors=("donor_id", "nunique"), datasets=("dataset_id", "nunique"))
           .reset_index())
    tab.to_csv(stem + "_types.csv", index=False)
    A.write_h5ad(a.out, compression="gzip")
    print(f"wrote {a.out}: {A.n_obs:,} cells x {A.n_vars:,} genes", flush=True)
    print(A.obs.coarse.replace("", "(dropped)").value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
