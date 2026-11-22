r"""label_drivers.py -- which genes make the labeler call a class? (IBD pilot)

Read only. For one labeled compartment file (claimbreak compartment label) and the reference used:

1. Drivers of a class: for cells labeled A (default: the most common label) with runner-up B, the summed
   log-likelihood contribution of each gene to A over B, sum over cells of x_g * (log q_Ag - log q_Bg), with the
   same profiles as the labeler (flat floor, no platform factor). Top genes per (A, B), with each gene's share of
   the total. If A wins through genes that are not A's markers (housekeeping, immunoglobulin, mitochondrial or
   high-background genes), the call reflects a platform/reference mismatch, not A.
2. Platform deviation per gene: observed tissue pseudobulk against the pseudobulk expected from the reference
   profiles at the labeled composition, log2(observed share / expected share), with the gene's share of the
   tissue counts. Genes detected far more (or less) than the reference predicts are what a platform factor fitted
   on hard labels bends toward (or what a platform-matched reference would remove).
3. Exclusion list (composition-free): log2 of a gene's tissue share over its highest share in any reference type.
   Above --exclude-log2 (default 1, i.e. twice what even the type that expresses it most would give) no mix of
   cell types can explain the gene: written to OUT/exclude_overdetected.txt for
   `claimbreak compartment label --exclude-genes`.

Usage:
  python label_drivers.py --ref REF.h5ad --ref-levels L1=coarse --h5ad PIECE.h5ad --labels LABELS_DIR \
      [--level L1 --arm whole --floor 0.01 --target Neural] --out ~/claimbreak_results/drivers_<piece>
Writes OUT/drivers.csv (per A, B, gene) and OUT/platform_deviation.csv (per gene).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--ref-levels", default="L1=coarse")
ap.add_argument("--ref-merge", default="")
ap.add_argument("--ref-map", default="")
ap.add_argument("--ref-raw", action="store_true")
ap.add_argument("--ref-layer", default="")
ap.add_argument("--h5ad", required=True)
ap.add_argument("--labels", required=True, help="folder with <piece>.labels.csv.gz")
ap.add_argument("--level", default="")
ap.add_argument("--arm", default="whole")
ap.add_argument("--floor", type=float, default=0.01, help="as used for the labels")
ap.add_argument("--target", default="", help="class A to explain (default: the most common label)")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--exclude-log2", type=float, default=1.0, help="threshold of the composition-free exclusion list")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import anndata as ad
import scipy.sparse as sp
from claimbreak.compartment import Reference

out = Path(a.out).expanduser()
if out.exists() and any(out.iterdir()):
    sys.exit(f"{out} exists and is not empty; choose a new --out")
levcol = dict(x.split("=", 1) for x in a.ref_levels.split(",") if x)
lv = a.level or list(levcol)[0]
ref = Reference.load(a.ref, {lv: levcol[lv]}, merge=a.ref_merge, map_file=a.ref_map, raw=a.ref_raw, layer=a.ref_layer)
A = ad.read_h5ad(Path(a.h5ad).expanduser())
piece = Path(a.h5ad).name.replace(".h5ad", "")
lf = Path(a.labels).expanduser() / f"{piece}.labels.csv.gz"
col = f"{lv}_{a.arm}"
L = pd.read_csv(lf, usecols=["cell_id", col, f"{col}_second"], keep_default_na=False, dtype={"cell_id": str})
L = L.set_index("cell_id").reindex(A.obs_names.astype(str)).fillna("")
genes, gidx, _ = ref.match(list(A.var_names))
types, mu = ref.profiles(lv, gidx)
q = (1 - a.floor) * mu + a.floor / mu.shape[1]
q = q / q.sum(1, keepdims=True)
lq = np.log(q)
tix = {t: i for i, t in enumerate(types)}
src = {"whole": None, "nucleus": "nucleus", "cyto": "cyto"}
if a.arm not in src:
    sys.exit("--arm must be whole, nucleus or cyto (count sets made from splits are not stored)")
M = A.X if src[a.arm] is None else A.layers[src[a.arm]]
X = sp.csr_matrix(M)[:, [A.var_names.get_loc(g) for g in genes]].astype(np.float64)
lab, sec = L[col].to_numpy(), L[f"{col}_second"].to_numpy()
out.mkdir(parents=True, exist_ok=True)

target = a.target or pd.Series(lab[lab != ""]).value_counts().index[0]
rows = []
for b in pd.Series(sec[lab == target]).value_counts().index[:4]:
    m = (lab == target) & (sec == b)
    if m.sum() < 20 or b not in tix:
        continue
    contrib = np.asarray(X[m].sum(0)).ravel() * (lq[tix[target]] - lq[tix[b]])
    tot = contrib.sum()
    order = np.argsort(-contrib)
    for r, j in enumerate(order[: a.top]):
        rows.append({"A": target, "B": b, "cells": int(m.sum()), "rank": r + 1, "gene": genes[j],
                     "contribution": contrib[j], "share_of_total": contrib[j] / tot if tot > 0 else np.nan,
                     "q_A": q[tix[target], j], "q_B": q[tix[b], j],
                     "mean_count": float(np.asarray(X[m][:, j].sum()) / m.sum())})
    print(f"\n{target} over {b} ({m.sum():,} cells): top genes by summed log-likelihood contribution "
          f"(total {tot:.0f} nats)")
    print(pd.DataFrame(rows)[lambda d: d.B == b][["rank", "gene", "share_of_total", "mean_count", "q_A", "q_B"]]
          .head(15).round(4).to_string(index=False))
pd.DataFrame(rows).to_csv(out / "drivers.csv", index=False)

ok = np.isin(lab, types)
comp = pd.Series(lab[ok]).value_counts()
n_by = pd.Series(np.asarray(X[ok].sum(1)).ravel()).groupby(lab[ok]).sum()
exp = sum(n_by[t] * q[tix[t]] for t in n_by.index)
obs = np.asarray(X[ok].sum(0)).ravel()
dev = pd.DataFrame({"gene": genes, "obs_share": obs / obs.sum(), "exp_share": exp / exp.sum()})
dev["log2_obs_over_exp"] = np.log2((dev.obs_share + 1e-7) / (dev.exp_share + 1e-7))
dev["max_type_share"] = q.max(0)
dev["log2_obs_over_any_type"] = np.log2((dev.obs_share + 1e-7) / (dev.max_type_share + 1e-7))
dev = dev.sort_values("log2_obs_over_exp", ascending=False)
dev.to_csv(out / "platform_deviation.csv", index=False)
print(f"\nlabelled composition: {comp.div(comp.sum()).round(3).head(8).to_dict()}")
print("\ngenes detected most above the reference expectation (share of tissue counts >= 0.1%):")
print(dev[dev.obs_share >= 0.001].head(20).round(4).to_string(index=False))
print("\nmost below:")
print(dev[dev.exp_share >= 0.001].tail(15).round(4).to_string(index=False))
print(f"\nshare of tissue counts in genes >4-fold above expectation: "
      f"{dev.loc[dev.log2_obs_over_exp > 2, 'obs_share'].sum():.3f}; below 1/4: "
      f"{dev.loc[dev.log2_obs_over_exp < -2, 'obs_share'].sum():.3f}")
ex = dev[dev.log2_obs_over_any_type > a.exclude_log2].sort_values("obs_share", ascending=False)
(out / "exclude_overdetected.txt").write_text("".join(f"{g}\n" for g in ex.gene))
print(f"\ngenes above any reference type's share by more than 2^{a.exclude_log2:g}: {len(ex)} of {len(dev)}, "
      f"holding {ex.obs_share.sum():.3f} of tissue counts -> exclude_overdetected.txt")
print(ex.head(15)[["gene", "obs_share", "max_type_share", "log2_obs_over_any_type"]].round(4).to_string(index=False))
print(f"wrote {out}")
