r"""compartment_label.py -- label every cell of a compartment h5ad (compartment_to_h5ad.py or bilous_to_h5ad.py) from
a single-cell reference, several ways per cell. Generalises bilous_label.py
to any reference and platform; with the defaults it reproduces bilous_label.py on GSE311609.

Reference options:
  --ref-levels  LEVEL=column[,LEVEL=column]: output level name and the reference obs column it comes from
                (default L2m=Level2,L3m=Level3, the lung reference)
  --ref-merge   regex=>label[;regex=>label]: rename reference labels matching a regex (default ^Tu_L\d=>Tumor)
  --ref-map     CSV with columns level,from,to: exact renames per level (e.g. fine gut types to a coarse level);
                labels mapped to "" or "drop" are removed from the reference
  --ref-donor   obs column with the donor (for --ref-cv; default donor_id)
  --ref-raw     take counts from .raw.X (CELLxGENE files keep raw counts there)
  --ref-subset  column=value|value[;column=value]: keep only these reference cells (e.g. tissue=colon|ileum)
Output columns added: donor, panel, section, unit, tissue (from the h5ad obs), so later scripts need no file-name rules.

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

Reference check (--ref-cv, per panel and level): grouped K-fold accuracy on the reference itself (folds of
--cv-group, default the donor column, --cv-folds 5; groups are shuffled with --seed and dealt to folds; --cv-folds 0 =
one fold per group, i.e. leave-one-donor-out), with test cells thinned binomially to fixed depths (default 20, 50,
100, 200 counts on panel genes), so you can see which level a panel can separate at those depths. At most
--cv-per-type test cells per type and fold (default 500). Profiles are built once per panel from per-(fold, type) sums,
so a fold costs one subtraction, not a pass over the reference. Writes ref_cv_<panel>.csv (per depth, level, held-out
fold, true type: n, accuracy) and the confusion.
Speed: type profiles are cached per (level, gene set), so files with the same panel reuse them; panel genes are selected from the reference once per panel. Labels are identical to the earlier version.

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
ap.add_argument("--inputs", "--xenium", nargs="+", required=True, dest="xenium", help="compartment .h5ad files")
ap.add_argument("--out", required=True)
ap.add_argument("--levels", default="", help="levels to label (default: all of --ref-levels)")
ap.add_argument("--arms", default="whole,nucleus,thinned,cyto,nucA,nucB,cytoB")
ap.add_argument("--floor", type=float, default=0.01)
ap.add_argument("--min-counts", type=int, default=10)
ap.add_argument("--gamma-iter", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ref-cv", action="store_true")
ap.add_argument("--cv-depths", default="20,50,100,200")
ap.add_argument("--cv-folds", type=int, default=5, help="grouped folds for --ref-cv (0 = one fold per group)")
ap.add_argument("--cv-group", default="", help="reference obs column grouping cells for --ref-cv (default --ref-donor)")
ap.add_argument("--cv-per-type", type=int, default=500, help="test cells per type and fold in --ref-cv (0 = all)")
ap.add_argument("--ref-layer", default="", help="reference layer to use instead of X (e.g. SoupX); default X (raw)")
ap.add_argument("--ref-levels", default="L2m=Level2,L3m=Level3")
ap.add_argument("--ref-merge", default=r"^Tu_L\d=>Tumour")
ap.add_argument("--ref-map", default="")
ap.add_argument("--ref-donor", default="donor_id")
ap.add_argument("--ref-raw", action="store_true")
ap.add_argument("--ref-subset", default="")
a = ap.parse_args()

import anndata as ad

out = Path(a.out).expanduser()
out.mkdir(parents=True, exist_ok=True)
arms = [s for s in a.arms.split(",") if s]
LEVCOL = dict(x.split("=", 1) for x in a.ref_levels.split(",") if x)
levels = [s for s in a.levels.split(",") if s and s in LEVCOL] if a.levels else list(LEVCOL)
if not levels:
    sys.exit(f"--levels {a.levels} not among --ref-levels {list(LEVCOL)}")

# reference
R = ad.read_h5ad(Path(a.ref).expanduser())
keep_ref = np.ones(R.n_obs, bool)
for cond in [c for c in a.ref_subset.split(";") if c]:
    col, vals = cond.split("=", 1)
    keep_ref &= R.obs[col].astype(str).isin(vals.split("|")).to_numpy()
if a.ref_raw:
    if R.raw is None:
        sys.exit("--ref-raw: the reference has no .raw")
    RX = sp.csr_matrix(R.raw.X)
    rvar = R.raw.var
else:
    RX = sp.csr_matrix(R.layers[a.ref_layer] if a.ref_layer else R.X)
    rvar = R.var
rsym = rvar["feature_name"].astype(str).to_numpy() if "feature_name" in rvar else rvar.index.astype(str).to_numpy()
if len(set(rsym)) != len(rsym):                       # keep the first occurrence of a duplicated symbol
    first = ~pd.Series(rsym).duplicated().to_numpy()
    RX, rsym = RX[:, np.where(first)[0]], rsym[first]
    print(f"reference: {int((~first).sum())} duplicated gene symbols dropped", flush=True)
rpos = {g: i for i, g in enumerate(rsym)}
rdonor = R.obs[a.ref_donor].astype(str).to_numpy() if a.ref_donor in R.obs else np.full(R.n_obs, "all")
merges = [m.split("=>", 1) for m in a.ref_merge.split(";") if m]
rmap = pd.read_csv(a.ref_map) if a.ref_map else None
rlab = {}
for lv in levels:
    s = pd.Series(R.obs[LEVCOL[lv]].astype(str).to_numpy())
    for rx, to in merges:
        s = s.where(~s.str.match(rx), to)
    if rmap is not None:
        m = rmap[rmap.level == lv]
        if len(m):
            s = s.map(dict(zip(m["from"].astype(str), m["to"].fillna("").astype(str)))).fillna(s)
    s = s.where(~s.isin(["", "drop", "nan"]), "")
    rlab[lv] = s.to_numpy()
    keep_ref &= rlab[lv] != ""
cvcol = a.cv_group or a.ref_donor
rgroup = R.obs[cvcol].astype(str).to_numpy() if cvcol in R.obs else np.full(R.n_obs, "all")
RX, rdonor, rgroup = RX[np.where(keep_ref)[0]], rdonor[keep_ref], rgroup[keep_ref]
rlab = {lv: v[keep_ref] for lv, v in rlab.items()}
print(f"reference: {R.n_obs:,} cells, {R.n_vars:,} genes; "
      + "; ".join(f"{lv}: {len(set(rlab[lv]))} types" for lv in levels), flush=True)


PCACHE = {}         # gene tuple -> (row-normalized panel matrix P, ok)
MUCACHE = {}        # (level, gene tuple) -> (types, mu)


def panel_props(gidx):
    """per-cell panel-gene proportions (csr) and which cells have panel counts; cached per gene set."""
    key = tuple(int(i) for i in gidx)
    if key not in PCACHE:
        PCACHE.clear()                                   # one panel at a time in memory
        M = RX[:, gidx].tocsr()
        tot = np.asarray(M.sum(1)).ravel()
        ok = tot > 0
        PCACHE[key] = (sp.diags(1.0 / np.where(ok, tot, 1)) @ M, ok, M)
    return PCACHE[key]


def cached_profiles(lv, gidx):
    key = (lv, tuple(int(i) for i in gidx))
    if key not in MUCACHE:
        MUCACHE[key] = profiles(gidx, rlab[lv])
    return MUCACHE[key]


def group_sums(P, ok, lab, groups):
    """sums of proportions and cell counts per (group code, type): returns types, S (g x t x genes), C (g x t)."""
    types = np.array(sorted(set(lab[ok])))
    tix = {t: i for i, t in enumerate(types)}
    gu, gcode = np.unique(groups, return_inverse=True)
    tcode = np.array([tix.get(t, -1) for t in lab])
    sel = ok & (tcode >= 0)
    col = gcode[sel] * len(types) + tcode[sel]
    Ind = sp.csr_matrix((np.ones(sel.sum()), (col, np.where(sel)[0])), shape=(len(gu) * len(types), P.shape[0]))
    S = np.asarray((Ind @ P).todense()).reshape(len(gu), len(types), P.shape[1])
    C = np.asarray(Ind.sum(1)).ravel().reshape(len(gu), len(types))
    return types, gu, gcode, S, C


def profiles(gidx, lab, rows=None):
    """mean panel-gene proportion profile per type (types x genes), from reference rows."""
    P, ok, _ = panel_props(gidx)
    if rows is not None:
        P, ok, L = P[rows], ok[rows], lab[rows]
    else:
        L = lab
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
        P, ok, M = panel_props(gidx)
        gu = np.unique(rgroup)
        perm = rng.permutation(len(gu))
        K = len(gu) if a.cv_folds <= 0 else min(a.cv_folds, len(gu))
        if K < 2:
            sys.exit(f"--ref-cv needs at least two groups in '{cvcol}'")
        gfold = dict(zip(gu[perm], np.arange(len(gu)) % K))
        fold = np.array([gfold[g] for g in rgroup])
        for lv in levels:
            types, fu, fcode, S, C = group_sums(P, ok, rlab[lv], fold)
            St, Ct = S.sum(0), C.sum(0)
            for fi, fname in enumerate(fu):
                Ctr = Ct - C[fi]
                have = Ctr > 0
                ttypes = types[have]
                mu = (St - S[fi])[have] / Ctr[have, None]
                te = np.where((fcode == fi) & ok)[0]
                if a.cv_per_type > 0:          # at most --cv-per-type test cells per type
                    lab_te = rlab[lv][te]
                    keep = []
                    for t in np.unique(lab_te):
                        idx = te[lab_te == t]
                        keep.append(idx if len(idx) <= a.cv_per_type else rng.choice(idx, a.cv_per_type, replace=False))
                    te = np.sort(np.concatenate(keep)) if keep else te
                Xte_full = M[te]
                truth = rlab[lv][te]
                for depth in [int(x) for x in a.cv_depths.split(",")]:
                    Xte = thin_to_depth(Xte_full, depth, rng)
                    lab, _, _, n, _ = classify(Xte, mu, ttypes, fit_gamma=False)
                    pred = np.where(lab >= 0, ttypes[np.maximum(lab, 0)], "NA")
                    df = pd.DataFrame({"true": truth, "pred": pred})
                    df = df[df.pred != "NA"]
                    for t, g in df.groupby("true"):
                        rows_cv.append({"panel": panel, "level": lv, "held_out_fold": int(fname), "depth": depth,
                                        "true": t, "n": len(g), "accuracy": float((g.pred == t).mean()),
                                        "type_in_training": bool(t in set(ttypes))})
                    cc = df.groupby(["true", "pred"]).size().reset_index(name="n")
                    cc["panel"], cc["level"], cc["held_out_fold"], cc["depth"] = panel, lv, int(fname), depth
                    conf_cv.append(cc)
            print(f"  ref-cv {lv}: {K} folds of '{cvcol}' ({len(gu)} groups)", flush=True)
        cvf = out / f"ref_cv_{panel.replace(' ', '_')}.csv"
        cv = pd.DataFrame(rows_cv)
        cv.to_csv(cvf, index=False)
        pd.concat(conf_cv).to_csv(out / f"ref_cv_confusion_{panel.replace(' ', '_')}.csv", index=False)
        summ = (cv.assign(c=cv.n * cv.accuracy).groupby(["level", "depth"])[["c", "n"]].sum()
                .assign(acc=lambda t: t.c / t.n)["acc"].unstack("depth").round(3))
        bal = cv.groupby(["level", "depth"]).accuracy.mean().unstack("depth").round(3)
        print(f"  reference check, {K} folds of {cvcol}, panel {panel} (rows level, columns depth):\n"
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
    for c in ("donor", "panel", "section", "unit", "tissue"):
        if c in A.obs:
            res[c] = A.obs[c].astype(str).to_numpy()
    gam = {"gene": genes}
    for lv in levels:
        types, mu = cached_profiles(lv, gidx)
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
        "reference": str(a.ref), "ref_layer": "raw" if a.ref_raw else (a.ref_layer or "X"),
        "ref_levels": LEVCOL, "ref_merge": a.ref_merge, "ref_map": a.ref_map, "ref_subset": a.ref_subset, "genes_used": len(genes), "genes_missing": miss,
        "floor": a.floor, "min_counts": a.min_counts, "gamma_iter": a.gamma_iter, "seed": a.seed,
        "ref_cv": dict(folds=a.cv_folds, group=cvcol, per_type=a.cv_per_type, depths=a.cv_depths) if a.ref_cv else None,
        "levels": levels, "arms": arms}, indent=1))
    print(f"  wrote {dst}", flush=True)
