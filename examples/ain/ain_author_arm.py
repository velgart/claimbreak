#!/usr/bin/env python
"""ain_author_arm.py -- add the authors' published labels (Baker et al. 2026, final_annotations2) as an arm
`<level>_author` of the claimbreak labels files, mapped to our coarse classes, so that `claimbreak compartment calls /
contacts / aggregates` can compare them with nucleus labels.

The mapping is fixed here and committed before the comparison is run. Authors' classes without a single coarse class
(Immune_mixed, Mixed_VIM_high, T_cell_LQ, ddTE) are left unlabeled ("").

  python ain_author_arm.py --labels ~/claimbreak_data/AIN/labels/v1 --h5ad-dir ~/claimbreak_data/AIN/h5ad \
      --out ~/claimbreak_data/AIN/labels/v1_author
"""
import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

AUTHOR_TO_L1 = {
    "TAL": "LOH", "tDL": "LOH",
    "PT-INJ": "PT", "PT-H": "PT", "PT-dd": "PT",
    "CD": "Distal_CD", "CNT": "Distal_CD", "DCT": "Distal_CD",
    "Podo": "Podocyte", "Parietal": "PEC",
    "EC": "Endothelial", "LEC": "Endothelial",
    "vSMC": "Mural", "Pericyte": "Mural", "JGA": "Mural", "Glomerular_Stromal": "Mural",
    "Fibroblasts": "Stromal", "Stromal": "Stromal", "Myofibroblast": "Stromal",
    "T_cell_CD8": "T_NK", "T_cell_CD4": "T_NK", "T_cell_DN": "T_NK", "T_cell_gd": "T_NK",
    "B_cell": "B", "Plasma_cell": "Plasma", "Mast_cell": "Mast",
    "Macrophage": "Myeloid", "Dendritic_cell": "Myeloid", "APC": "Myeloid", "Eosinophils": "Myeloid",
    "Immune_mixed": "", "Mixed_VIM_high": "", "T_cell_LQ": "", "ddTE": "",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--h5ad-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="L1")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    unknown = set()
    for f in sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz")):
        dst = out / f.name
        if dst.exists():
            print(f"{dst.name} exists; skipped", flush=True)
            continue
        sample = f.name.split(".labels")[0]
        obs = ad.read_h5ad(Path(a.h5ad_dir).expanduser() / f"{sample}.h5ad", backed="r").obs
        auth = obs["author_label"].astype(str)
        unknown |= set(auth.unique()) - set(AUTHOR_TO_L1)
        D = pd.read_csv(f, keep_default_na=False, low_memory=False)
        lab = auth.reindex(D["cell_id"].astype(str)).map(AUTHOR_TO_L1).fillna("").to_numpy()
        D[f"{a.level}_author"] = lab
        D.to_csv(dst, index=False, compression="gzip")
        ok = (D[f"{a.level}_author"] != "") & (D[f"{a.level}_whole"] != "")
        agree = (D.loc[ok, f"{a.level}_author"] == D.loc[ok, f"{a.level}_whole"]).mean()
        agree_n = (D.loc[ok, f"{a.level}_author"] == D.loc[ok, f"{a.level}_nucleus"]).mean()
        print(f"{sample}: author labels on {(D[f'{a.level}_author'] != '').mean():.3f} of cells; agreement with our "
              f"whole-cell labels {agree:.3f}, with our nucleus labels {agree_n:.3f}", flush=True)
    if unknown:
        print(f"authors' classes not in the mapping (left unlabelled): {sorted(unknown)}", flush=True)


if __name__ == "__main__":
    main()
