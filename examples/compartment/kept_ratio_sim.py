#!/usr/bin/env python
"""kept_ratio_sim.py -- does the label-only kept ratio recover the true loss? (a ground-truth check of the kept ratio).

Synthetic sections with known cell types. Each cell's nuclear and cytoplasmic counts are drawn from its own type's
profile; a planted share `psi` of the cytoplasmic counts comes from the touching cells instead (each touching cell
contributes in proportion to its RNA), and a smaller share `psi_nuc` of the nuclear counts likewise. The cells are
labeled by `compartment.label_section` exactly as the real sections (arms nucA, nucB, wholeB; same classifier,
platform factor and thinning), against a reference drawn from the same type profiles. Then, per focal type t:

  true recall ratio   P(wholeB = t | true t) / P(nucB = t | true t): the loss relative to the nucleus that protein
                      measures when protein marks exactly the true cells
  kept ratio          P(wholeB = t | nucA = t) / P(nucB = t | nucA = t): the label-only estimate

Optional failure mode (--fp-gene): one probe that reads as a strong T marker in the reference is detected in the
nuclei of fibroblasts (an off-target probe, as CCL5 in synovium). It makes nuclear false-positive T calls that half B
repeats, so the kept ratio understates what whole-cell counts keep (it overstates the loss).

Cell types (illustrative sizes): T (small, RNA-poor), myeloid, fibroblast, tumor (large, RNA-rich). Tumor nests and
fibroblast stroma as patches, T and myeloid scattered over both.

  python kept_ratio_sim.py --out ~/claimbreak_work/sim_kept [--psi 0,0.1,0.2,0.3,0.4 --seeds 0,1,2 --fp-gene]
Writes OUT/kept_ratio_sim.csv (one row per setting, seed and type) and prints the comparison. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

TYPES = ("T", "Myeloid", "Fibroblast", "Tumour")
NUC_COUNTS = {"T": 25, "Myeloid": 40, "Fibroblast": 45, "Tumour": 80}     # mean nuclear counts per cell
CYTO_FOLD = {"T": 0.8, "Myeloid": 1.5, "Fibroblast": 1.5, "Tumour": 2.0}  # cytoplasmic over nuclear counts


def profiles(G: int, rng):
    """Type profiles over G genes: a shared background plus 12 markers per type (about 4-8x their background)."""
    base = rng.gamma(0.6, 1.0, G)
    mu = np.tile(base, (len(TYPES), 1))
    for k in range(len(TYPES)):
        m = np.arange(k * 12, (k + 1) * 12)
        mu[k, m] *= rng.uniform(4, 8, len(m))
    return mu / mu.sum(1, keepdims=True)


def tissue(n: int, rng, L: float = 2500.0):
    xy = rng.uniform(0, L, (n, 2))
    centres = rng.uniform(0, L, (120, 2))
    patch = rng.choice([2, 3], len(centres), p=[0.45, 0.55])          # fibroblast stroma or tumor nest
    base = patch[cKDTree(centres).query(xy)[1]]
    u = rng.random(n)
    t = np.where(u < 0.12, 0, np.where(u < 0.22, 1, base))            # T 12%, myeloid 10%, scattered
    return xy, t


def counts(xy, t, mu, psi, psi_nuc, fp_gene, fp_rate, rng):
    n, G = len(t), mu.shape[1]
    lam_n = np.array([NUC_COUNTS[TYPES[k]] for k in t], float)
    lam_c = lam_n * np.array([CYTO_FOLD[TYPES[k]] for k in t])
    d, nn = cKDTree(xy).query(xy, k=7)
    d, nn = d[:, 1:], nn[:, 1:]
    touch = d <= 18.0                                                  # touching neighbors (cell spacing ~10-15 um)
    # foreign profile per cell: touching cells' profiles weighted by their RNA; isolated cells keep their own
    w = np.where(touch, (lam_n + lam_c)[nn], 0.0)
    has = w.sum(1) > 0
    foreign = np.einsum("nj,njg->ng", w, mu[t[nn]]) / np.where(has, w.sum(1), 1)[:, None]
    foreign[~has] = mu[t[~has]]
    own = mu[t]
    pc = (1 - psi) * own + psi * foreign
    pn = (1 - psi_nuc) * own + psi_nuc * foreign
    if fp_gene is not None:                                            # off-target probe in fibroblast nuclei
        fib = t == TYPES.index("Fibroblast")
        pn[fib, fp_gene] += fp_rate
        pn[fib] /= pn[fib].sum(1, keepdims=True)
    Xn = rng.poisson(pn * lam_n[:, None]).astype(np.float32)
    Xc = rng.poisson(pc * lam_c[:, None]).astype(np.float32)
    return sp.csr_matrix(Xn), sp.csr_matrix(Xc)


def reference(mu, rng, per_type: int = 400, depth: int = 2000):
    X = np.vstack([rng.multinomial(depth, mu[k], per_type) for k in range(len(TYPES))])
    lab = np.repeat(np.array(TYPES), per_type)
    return X, lab


def run_one(seed, psi, psi_nuc, fp, n, G, fp_rate):
    import anndata as ad
    from claimbreak import compartment as cm
    rng = np.random.default_rng(seed)
    mu = profiles(G, rng)
    xy, t = tissue(n, rng)
    fp_gene = 0 if fp else None                                        # gene 0: a T marker in the reference
    Xn, Xc = counts(xy, t, mu, psi, psi_nuc, fp_gene, fp_rate, rng)
    genes = [f"g{j}" for j in range(G)]
    RX, rlab = reference(mu, rng)
    ref = cm.Reference(RX, np.array(genes), {"L1": rlab}, np.repeat(np.arange(4), len(rlab) // 4), {})
    Xw = (Xn + Xc).tocsr()
    A = ad.AnnData(X=Xw, obs=pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                                           "n_whole": np.asarray(Xw.sum(1)).ravel(),
                                           "n_nucleus": np.asarray(Xn.sum(1)).ravel()},
                                          index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=genes), layers={"nucleus": Xn, "cyto": Xc})
    cfg = cm.LabelConfig(audit=False)
    res, _, _ = cm.label_section(A, ref, ["L1"], ["nucA", "nucB", "wholeB"], cfg, seed=seed, sample="sim")
    a_, b_, w_ = (res[f"L1_{x}"].to_numpy() for x in ("nucA", "nucB", "wholeB"))
    truth = np.array(TYPES)[t]
    rows = []
    for ty in ("T", "Myeloid"):
        tr = truth == ty
        okb, okw = (b_ != "") & tr, (w_ != "") & tr
        true_ratio = np.mean(w_[okw] == ty) / np.mean(b_[okb] == ty)
        sel = a_ == ty
        kb, kw = sel & (b_ != ""), sel & (w_ != "")
        kept = np.mean(w_[kw] == ty) / np.mean(b_[kb] == ty)
        rows.append(dict(seed=seed, psi=psi, psi_nuc=psi_nuc, fp_gene=fp, type=ty, true_cells=int(tr.sum()),
                         nucA_calls=int(sel.sum()), nucA_precision=float(np.mean(truth[sel] == ty)),
                         recall_nucB=float(np.mean(b_[okb] == ty)), recall_wholeB=float(np.mean(w_[okw] == ty)),
                         true_recall_ratio=float(true_ratio), kept_ratio=float(kept),
                         difference=float(kept - true_ratio)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--psi", default="0,0.1,0.2,0.3,0.4", help="planted foreign share of the cytoplasmic counts")
    ap.add_argument("--psi-nuc-fold", type=float, default=0.25, help="nuclear foreign share as a fraction of psi")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cells", type=int, default=30000); ap.add_argument("--genes", type=int, default=150)
    ap.add_argument("--fp-gene", action="store_true", help="also run with an off-target T-marker probe in fibroblasts")
    ap.add_argument("--fp-rate", type=float, default=0.06, help="share of fibroblast nuclear counts on that probe")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fp in ([False, True] if a.fp_gene else [False]):
        for psi in [float(x) for x in a.psi.split(",")]:
            for s in [int(x) for x in a.seeds.split(",")]:
                rows += run_one(s, psi, psi * a.psi_nuc_fold, fp, a.cells, a.genes, a.fp_rate)
                print(f"fp {fp} psi {psi} seed {s} done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "kept_ratio_sim.csv", index=False)
    S = (R.groupby(["fp_gene", "type", "psi"])[["nucA_precision", "true_recall_ratio", "kept_ratio", "difference"]]
         .mean().round(3))
    with pd.option_context("display.width", 160):
        print("\nmeans over seeds:\n" + S.to_string())
    print(f"wrote {out}/kept_ratio_sim.csv")


if __name__ == "__main__":
    main()
