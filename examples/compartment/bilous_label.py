"""Label Xenium cells of GSE311609 (Bilous et al. 2026) from the matched single-nucleus reference, four ways per cell
.

Arms (same cells, same classifier, same reference; only the counts differ):
  whole    X (all transcripts the default segmentation assigned to the cell)
  nucleus  layers['nucleus'] (transcripts overlapping the nucleus)
  thinned  X thinned binomially to each cell's nucleus depth (p = n_nucleus / n_whole, seed --seed):
           whole-cell composition at nucleus depth. whole vs thinned = depth; thinned vs nucleus = composition
           (neighbor contamination, plus genuine nuclear/cytoplasmic differences)
  cyto     layers['cyto']
  nucA, nucB  binomial 0.5 split of the nucleus counts (seed --seed): two independent nucleus samples
  cytoB    cyto thinned to each cell's nucB depth (cells with fewer cyto counts keep them all)
  wholeB   (whole - nucA) = nucB + cyto, thinned to each cell's nucB depth: whole-cell composition, independent of nucA
           nucB vs nucA = noise alone at that depth (the null); cytoB vs nucA = noise + compartment (contamination);
           wholeB vs nucA = the same for whole-cell composition (what whole-cell labels see).
  wB<k>    (whole - nucA) thinned to k% of each cell's nucB depth (e.g. wB50, wB200); wBall = all of (whole - nucA)
  nB<k>    nucB thinned to k% of its depth (k <= 100): the null at lower depth
           Depth series for the whole-cell excess: the null cannot go deeper than nucB, so deeper wB arms are read
           against nB100 (and a bound on how much the null could grow with depth).
           Both test arms are independent of nucA, so a wrong nucA label is "corrected" equally often in each.
Levels: L2m = reference Level2 with Tu_L1..Tu_L4 merged into Tumor (9 types); L3m = Level3, same merge (23 types).

Classifier: multinomial likelihood on the panel genes present in the reference (matched by symbol, var.feature_name).
  mu[k,g] = mean over reference cells of type k of their panel-gene proportions, mixed with a flat floor:
            (1 - floor) * mu + floor / G   (floor, default 0.01, absorbs background and off-target counts)
  platform factor gamma[g] per section and arm, fitted by --gamma-iter rounds of: label -> expected pseudobulk under
            the labels -> gamma *= observed / expected (geometric mean 1, clipped to [0.05, 20])
  label = argmax_k  sum_g x_g log(gamma_g mu_kg / sum_g' gamma_g' mu_kg')   (uniform prior: maximum likelihood)
  per cell and arm: label, second label, margin (log-likelihood of best minus second, nats), n (counts used).
  Cells with fewer than --min-counts panel counts in an arm get no label in that arm.

Reference check (--ref-cv, per panel and level): leave-one-donor-out accuracy on the reference itself, with test cells
thinned binomially to fixed depths (default 20, 50, 100, 200 counts on panel genes), so you can see which level a panel
can separate at Xenium depths. Writes ref_cv_<panel>.csv (per depth, level, true type: n, accuracy) and the confusion.

Usage:
  python bilous_label.py --ref REF.h5ad --xenium ~/claimbreak_data/Bilous_h5ad/*.h5ad --out ~/claimbreak_data/Bilous_labels [--ref-cv]
Writes OUT/<sample>.labels.csv.gz (one row per cell) and OUT/<sample>.gamma.csv; refuses to overwrite existing files.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--xenium", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--levels", default="L2m,L3m")
ap.add_argument("--arms", default="whole,nucleus,thinned,cyto,nucA,nucB,cytoB")
ap.add_argument("--floor", type=float, default=0.01)
ap.add_argument("--min-counts", type=int, default=10)
ap.add_argument("--gamma-iter", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ref-cv", action="store_true")
ap.add_argument("--cv-depths", default="20,50,100,200")
ap.add_argument("--ref-layer", default="", help="reference layer to use instead of X (e.g. SoupX); default X (raw)")
a = ap.parse_args()

import anndata as ad

out = Path(a.out).expanduser()
out.mkdir(parents=True, exist_ok=True)
levels = [s for s in a.levels.split(",") if s]
arms = [s for s in a.arms.split(",") if s]
LEVCOL = {"L2m": "Level2", "L3m": "Level3"}

# reference
R = ad.read_h5ad(Path(a.ref).expanduser())
RX = R.layers[a.ref_layer] if a.ref_layer else R.X
RX = sp.csr_matrix(RX)
rsym = R.var["feature_name"].astype(str).to_numpy() if "feature_name" in R.var else R.var_names.astype(str).to_numpy()
if len(set(rsym)) != len(rsym):
    sys.exit("duplicate gene symbols in reference")
rpos = {g: i for i, g in enumerate(rsym)}
rdonor = R.obs["donor_id"].astype(str).to_numpy()
rlab = {}
for lv in levels:
    s = R.obs[LEVCOL[lv]].astype(str).to_numpy()
    rlab[lv] = np.where(pd.Series(s).str.match(r"^Tu_L\d").to_numpy(), "Tumour", s)
print(f"reference: {R.n_obs:,} cells, {R.n_vars:,} genes; "
      + "; ".join(f"{lv}: {len(set(rlab[lv]))} types" for lv in levels), flush=True)


def profiles(gidx, lab, rows=None):
    """mean panel-gene proportion profile per type (types x genes), from reference rows."""
    M = RX[:, gidx] if rows is None else RX[rows][:, gidx]
    L = lab if rows is None else lab[rows]
    tot = np.asarray(M.sum(1)).ravel()
    ok = tot > 0
    P = sp.diags(1.0 / np.where(ok, tot, 1)) @ M
    types = np.array(sorted(set(L[ok])))
    mu = np.vstack([np.asarray(P[ok & (L == t)].mean(0)).ravel() for t in types])
    return types, mu


def loglik_matrix(mu, gamma):
    q = (1 - a.floor) * mu + a.floor / mu.shape[1]
    q = q * gamma[None, :]
    q = q / q.sum(1, keepdims=True)
    return np.log(q).T  # genes x types


def classify(X, mu, types, fit_gamma=True):
    """X: cells x genes (csr counts). Returns label idx, second idx, margin, n, gamma."""
    G = X.shape[1]
    gamma = np.ones(G)
    n = np.asarray(X.sum(1)).ravel()
    use = n >= a.min_counts
    Xu = X[use]
    obs_pb = np.asarray(Xu.sum(0)).ravel()
    for it in range(a.gamma_iter + 1 if fit_gamma else 1):
        LL = np.asarray(Xu @ loglik_matrix(mu, gamma))
        best = LL.argmax(1)
        if not fit_gamma or it == a.gamma_iter:
            break
        q = (1 - a.floor) * mu + a.floor / G
        q = q * gamma[None, :]
        q = q / q.sum(1, keepdims=True)
        exp_pb = np.asarray(sp.csr_matrix((n[use], (best, np.zeros(len(best), int))),
                                          shape=(len(types), 1)).toarray()).ravel() @ q
        r = (obs_pb + 0.5) / (exp_pb + 0.5)
        gamma = gamma * r
        gamma = np.clip(gamma / np.exp(np.mean(np.log(gamma))), 0.05, 20)
    srt = np.sort(LL, 1)
    second = np.argsort(LL, 1)[:, -2] if len(types) > 1 else best
    lab = np.full(X.shape[0], -1)
    sec = np.full(X.shape[0], -1)
    mar = np.full(X.shape[0], np.nan)
    lab[use], sec[use] = best, second
    mar[use] = srt[:, -1] - srt[:, -2] if len(types) > 1 else np.inf
    return lab, sec, mar, n, gamma


def thin(X, n_target, rng):
    X = sp.csr_matrix(X, copy=True)
    n = np.asarray(X.sum(1)).ravel()
    p = np.where(n > 0, np.minimum(n_target / np.maximum(n, 1), 1.0), 0.0)
    rowp = np.repeat(p, np.diff(X.indptr))
    X.data = rng.binomial(X.data.astype(np.int64), rowp).astype(np.float32)
    X.eliminate_zeros()
    return X


def thin_to_depth(X, depth, rng):
    """thin each row to about `depth` counts (rows with fewer stay as they are)."""
    return thin(X, np.full(X.shape[0], float(depth)), rng)


cv_done = set()
for f in a.xenium:
    f = Path(f).expanduser()
    sample = f.name.replace(".h5ad", "")
    dst = out / f"{sample}.labels.csv.gz"
    if dst.exists():
        print(f"{dst} exists; skipped", flush=True)
        continue
    A = ad.read_h5ad(f)
    panel = str(A.obs["panel"].iloc[0])
    genes = [g for g in A.var_names if g in rpos]
    miss = [g for g in A.var_names if g not in rpos]
    gidx = np.array([rpos[g] for g in genes])
    xidx = np.array([A.var_names.get_loc(g) for g in genes])
    print(f"{sample}: panel {panel}, {A.n_obs:,} cells, {len(genes)} of {A.n_vars} genes in reference", flush=True)

    # reference cross-validation, once per panel gene set
    key = (panel, tuple(genes))
    if a.ref_cv and key not in cv_done:
        cv_done.add(key)
        rng = np.random.default_rng(a.seed)
        rows_cv, conf_cv = [], []
        for lv in levels:
            for d in sorted(set(rdonor)):
                tr, te = rdonor != d, rdonor == d
                types, mu = profiles(gidx, rlab[lv], np.where(tr)[0])
                Xte_full = RX[np.where(te)[0]][:, gidx]
                truth = rlab[lv][te]
                for depth in [int(x) for x in a.cv_depths.split(",")]:
                    Xte = thin_to_depth(Xte_full, depth, rng)
                    lab, _, _, n, _ = classify(Xte, mu, types, fit_gamma=False)
                    pred = np.where(lab >= 0, types[np.maximum(lab, 0)], "NA")
                    df = pd.DataFrame({"true": truth, "pred": pred})
                    df = df[df.pred != "NA"]
                    for t, g in df.groupby("true"):
                        rows_cv.append({"panel": panel, "level": lv, "held_out_donor": d, "depth": depth, "true": t,
                                        "n": len(g), "accuracy": float((g.pred == t).mean())})
                    cc = df.groupby(["true", "pred"]).size().reset_index(name="n")
                    cc["panel"], cc["level"], cc["held_out_donor"], cc["depth"] = panel, lv, d, depth
                    conf_cv.append(cc)
        cvf = out / f"ref_cv_{panel.replace(' ', '_')}.csv"
        cv = pd.DataFrame(rows_cv)
        cv.to_csv(cvf, index=False)
        pd.concat(conf_cv).to_csv(out / f"ref_cv_confusion_{panel.replace(' ', '_')}.csv", index=False)
        summ = (cv.assign(c=cv.n * cv.accuracy).groupby(["level", "depth"])[["c", "n"]].sum()
                .assign(acc=lambda t: t.c / t.n)["acc"].unstack("depth").round(3))
        bal = cv.groupby(["level", "depth"]).accuracy.mean().unstack("depth").round(3)
        print(f"  reference leave-one-donor-out, panel {panel} (rows level, columns depth):\n"
              f"  overall accuracy\n{summ.to_string()}\n  mean per-type accuracy\n{bal.to_string()}", flush=True)

    # counts per arm, panel genes in reference
    def sub(M):
        return sp.csr_matrix(M)[:, xidx].astype(np.float32)
    Xw = sub(A.X)
    Xn = sub(A.layers["nucleus"])
    rng = np.random.default_rng(a.seed)
    countsets = {}
    if {"nucA", "nucB", "cytoB", "wholeB"} & set(arms) or any(re.match(r"^(wB|nB)", x) for x in arms):
        XB = sp.csr_matrix(Xn, copy=True)
        XB.data = np.random.default_rng(a.seed + 1).binomial(XB.data.astype(np.int64), 0.5).astype(np.float32)
        XA = (Xn - XB).tocsr()
        XA.eliminate_zeros(); XB.eliminate_zeros()
    for arm in arms:
        if arm == "whole":
            countsets[arm] = Xw
        elif arm == "nucleus":
            countsets[arm] = Xn
        elif arm == "cyto":
            countsets[arm] = sub(A.layers["cyto"])
        elif arm == "nucA":
            countsets[arm] = XA
        elif arm == "nucB":
            countsets[arm] = XB
        elif re.match(r"^wB(\d+|all)$", arm):
            rest = (Xw - XA).tocsr()
            if arm == "wBall":
                countsets[arm] = rest
            else:
                k = int(arm[2:]) / 100.0
                countsets[arm] = thin(rest, k * np.asarray(XB.sum(1)).ravel(), rng)
        elif re.match(r"^nB\d+$", arm):
            k = int(arm[2:]) / 100.0
            if k > 1:
                sys.exit(f"{arm}: nucB cannot be thinned above 100%")
            countsets[arm] = XB if k == 1 else thin(XB, k * np.asarray(XB.sum(1)).ravel(), rng)
        elif arm == "wholeB":
            countsets[arm] = thin((Xw - XA).tocsr(), np.asarray(XB.sum(1)).ravel(), rng)
        elif arm == "cytoB":
            countsets[arm] = thin(sub(A.layers["cyto"]), np.asarray(XB.sum(1)).ravel(), rng)
        elif arm == "thinned":
            countsets[arm] = thin(Xw, np.asarray(Xn.sum(1)).ravel(), rng)
        else:
            sys.exit(f"unknown arm {arm}")

    res = pd.DataFrame({"cell_id": A.obs_names.astype(str), "x": A.obs["x"].to_numpy(), "y": A.obs["y"].to_numpy(),
                        "n_whole": A.obs["n_whole"].to_numpy(), "n_nucleus": A.obs["n_nucleus"].to_numpy()})
    res.insert(0, "sample", sample)
    gam = {"gene": genes}
    for lv in levels:
        types, mu = profiles(gidx, rlab[lv])
        for arm, X in countsets.items():
            lab, sec, mar, n, gamma = classify(X, mu, types)
            res[f"{lv}_{arm}"] = np.where(lab >= 0, types[np.maximum(lab, 0)], "")
            res[f"{lv}_{arm}_second"] = np.where(sec >= 0, types[np.maximum(sec, 0)], "")
            res[f"{lv}_{arm}_margin"] = np.round(mar, 3)
            if lv == levels[0]:
                res[f"n_{arm}"] = n.astype(int)
            gam[f"{lv}_{arm}"] = np.round(gamma, 4)
        msg = []
        if True:
            for x, y in [("whole", "nucleus"), ("thinned", "nucleus"), ("whole", "thinned"), ("cyto", "nucleus"),
                         ("nucB", "nucA"), ("cytoB", "nucA"), ("wholeB", "nucA")] + \
                        [(x, "nucA") for x in arms if re.match(r"^(wB|nB)", x)]:
                if x in arms and y in arms:
                    ok = (res[f"{lv}_{x}"] != "") & (res[f"{lv}_{y}"] != "")
                    msg.append(f"{x}=={y} {np.mean(res.loc[ok, f'{lv}_{x}'] == res.loc[ok, f'{lv}_{y}']):.3f} "
                               f"(n {ok.sum():,})")
        print(f"  {lv}: " + "; ".join(msg), flush=True)
        comp = res[f"{lv}_whole"].value_counts(normalize=True).round(3) if "whole" in arms else None
        if comp is not None:
            print(f"    whole-cell composition: {dict(comp.head(12))}", flush=True)
    res.to_csv(dst, index=False, compression="gzip")
    pd.DataFrame(gam).to_csv(out / f"{sample}.gamma.csv", index=False)
    (out / f"{sample}.meta.json").write_text(json.dumps({
        "reference": str(a.ref), "ref_layer": a.ref_layer or "X", "genes_used": len(genes), "genes_missing": miss,
        "floor": a.floor, "min_counts": a.min_counts, "gamma_iter": a.gamma_iter, "seed": a.seed,
        "levels": levels, "arms": arms}, indent=1))
    print(f"  wrote {dst}", flush=True)
