#!/usr/bin/env python
"""Fit the packaged flip models from the GSE311609 compartment labels (bilous_label.py output).

Reference labels: half B of the nucleus counts (nucB); test labels: whole-cell counts minus half A, thinned to the
depth of half B (wholeB); null: half A (nucA). Neighbors typed by the full-nucleus labels, contact = ranks 1-6
within 30 um, far = ranks 7-20 within 60 um. Units = donors (leave-one-donor-out AUC).

Writes two models:
  flipmodel_default.json     all features (nucleus share and areas included)
  flipmodel_no_nucleus.json  without outside_nucleus_share and the areas (for data without a nucleus split)
Copy both into src/claimbreak/data/ to replace the provisional default and enable the fallback.

  python examples/compartment/fit_flipmodel.py --labels ~/claimbreak_results/bilous_v2 \
      --h5ad-dir ~/claimbreak_data/Bilous_h5ad --out ~/claimbreak_results/flipmodel_fit
"""
import argparse, glob, json, os, re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from claimbreak import Tissue
from claimbreak import flipmodel as fm

LEVEL = "L2m"


def section(f, h5ad_dir):
    df = pd.read_csv(f)
    name = os.path.basename(f).split(".labels")[0]
    need = [f"{LEVEL}_{a}" for a in ("nucB", "wholeB", "nucA", "nucleus")] + [f"{LEVEL}_nucB_second",
                                                                             f"{LEVEL}_nucB_margin"]
    df = df[df[need].notna().all(axis=1)].reset_index(drop=True)
    m = re.search(r"_(L\d)_L\d", name)
    df["donor"] = m.group(1) if m else name
    df["section"] = name
    if h5ad_dir:
        c = glob.glob(os.path.join(h5ad_dir, name + "*.h5ad"))
        if len(c) == 1:
            import anndata as ad
            obs = ad.read_h5ad(c[0], backed="r").obs
            ids = df["cell_id"].astype(str)
            if ids.isin(obs.index.astype(str)).all():
                o = obs.loc[ids.to_numpy()]
                for k in ("cell_area", "nucleus_area"):
                    if k in o:
                        df[k] = o[k].to_numpy(float)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--h5ad-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-ratio", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(a.out)
    D = pd.concat([section(f, a.h5ad_dir) for f in sorted(glob.glob(os.path.join(a.labels, "*.labels.csv.gz")))],
                  ignore_index=True)
    print(f"{len(D):,} cells, {D.section.nunique()} sections, donors {sorted(D.donor.unique())}", flush=True)
    t = Tissue(counts=sp.csr_matrix((len(D), 1)), xy=D[["x", "y"]].to_numpy(float), labels=D[f"{LEVEL}_nucB"].astype(str),
               section=D.section.to_numpy(), unit=D.donor.to_numpy(), genes=np.array(["none"]))
    types = np.array(sorted(set(D[f"{LEVEL}_nucB"]) | set(D[f"{LEVEL}_wholeB"]) | set(D[f"{LEVEL}_nucleus"])))
    nb = fm.Neighbourhood.build(t, labels=D[f"{LEVEL}_nucleus"].astype(str).to_numpy(), types=types, k=20,
                                r_contact=30.0, r_far=60.0)
    second = np.array([list(types).index(x) if x in types else -1 for x in D[f"{LEVEL}_nucB_second"]])
    cell = {"margin": D[f"{LEVEL}_nucB_margin"].to_numpy(float), "log_counts": np.log1p(D["n_nucB"].to_numpy(float))}
    if {"n_whole", "n_nucleus"} <= set(D.columns):
        cell["outside_nucleus_share"] = 1 - D.n_nucleus.to_numpy(float) / np.maximum(D.n_whole.to_numpy(float), 1)
    if {"cell_area", "nucleus_area"} <= set(D.columns):
        ca, na = D.cell_area.to_numpy(float), D.nucleus_area.to_numpy(float)
        cell.update(log_cell_area=np.log(np.maximum(ca, 1e-6)), log_nucleus_area=np.log(np.maximum(na, 1e-6)),
                    nucleus_cell_ratio=na / np.maximum(ca, 1e-6))
    else:
        print("areas not found (pass --h5ad-dir): the default model is fitted without them", flush=True)
    rng = np.random.default_rng(a.seed)
    W, A = D[f"{LEVEL}_wholeB"].astype(str).to_numpy(), D[f"{LEVEL}_nucA"].astype(str).to_numpy()
    for variant, drop in (("default", ()), ("no_nucleus", ("outside_nucleus_share", "log_cell_area",
                                                          "log_nucleus_area", "nucleus_cell_ratio"))):
        feats = [f for f in fm.ALL_FEATURES if f not in drop]
        design = fm.Design(nb, {k: v for k, v in cell.items() if k not in drop}, second, feats)
        model, info = fm.fit(t, W, design, null_labels=A, neg_ratio=a.neg_ratio, rng=rng)
        model.note = (f"GSE311609 (Bilous et al. 2026; NSCLC Xenium, 12 sections, donors L1-L4, panels Lung, "
                      f"Custom IO, 5K), coarse labels. Reference nucleus half B, test whole-cell minus half A at "
                      f"half-B depth, null half A; neighbours typed by full-nucleus labels, contact ranks 1-6 "
                      f"within 30 um, far 7-20 within 60 um. Variant: {variant}.")
        model.provenance = dict(info, script="examples/compartment/fit_flipmodel.py", variant=variant)
        model.to_json(os.path.join(a.out, f"flipmodel_{variant}.json"))
        print(variant, json.dumps({k: round(v, 3) for k, v in model.beta.items()}),
              "AUC (leave one donor out):", round(info.get("auc_leave_one_unit_out", float("nan")), 3), flush=True)


if __name__ == "__main__":
    main()
