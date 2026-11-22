#!/usr/bin/env python
"""bilous_placement.py -- same flips, different placement, held out by donor.

For every section, the whole-cell labels (arm `wholeB`) differ from the half-B nucleus labels (`nucB`) by a
number of a->b changes beyond the noise null (`nucA` against `nucB`). This script applies exactly that number of
excess flips per ordered pair (a, b) to the nucB labels, in arms that differ ONLY in where the flips land:

  random      uniformly among the section's nucB-a cells
  ambiguity   the nucB-a cells with the lowest nucB margin whose second-best label is b (no spatial input)
  kernel      weighted by the excess flip rate of the cell's stratum for b (contact / far only / absent),
              with the rates fitted on the OTHER donors (leave one donor out)
  kernel_panel  (5K sections only) rates fitted on the Lung and Custom IO sections of all donors

It then compares, per pair, the change in kNN colocalization (log2 O/E, k = 10, fixed graph) from nucB to each arm
with the real change from nucB to wholeB, and the permutation neighborhood-enrichment call (depleted / neither /
enriched) of each label set. The counts of flips come from the test section's own labels; only their placement is
held out.

A criteria file (YAML; thresholds of the verdict) is required: its SHA-256 and modification time are recorded in the
output, and the verdict is evaluated from the values at the end. The script refuses to overwrite an output folder.

Usage
  python bilous_placement.py --labels ~/claimbreak_results/bilous_v2 --h5ad-dir ~/claimbreak_data/Bilous_h5ad \
      --rule placement_rule.yaml --out ~/claimbreak_results/bilous_placement
  python bilous_placement.py --labels ~/claimbreak_results/bilous_v2 --inspect      # print columns and labels, then stop
"""
import argparse, glob, hashlib, json, os, re, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ARMS_NEEDED = ("nucA", "nucB", "wholeB", "nucleus")


# ----------------------------------------------------------------------------------------------- input
def sha256(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def find_cols(cols, arm, level):
    """Return dict(label, second, margin) for one arm and level, tolerant of the column template."""
    toks = [c for c in cols if re.search(rf"(^|[_\.:]){re.escape(arm)}($|[_\.:])", c)]
    if level:
        lv = [c for c in toks if re.search(rf"(^|[_\.:]){re.escape(level)}($|[_\.:])", c)]
        toks = lv or toks
    out = {}
    for c in toks:
        rest = re.sub(rf"(^|[_\.:]){re.escape(arm)}($|[_\.:])", "_", c)
        if level:
            rest = re.sub(rf"(^|[_\.:]){re.escape(level)}($|[_\.:])", "_", rest)
        rest = rest.strip("_.:").lower()
        if rest in ("", "label", "lab"):
            out["label"] = c
        elif rest in ("second", "second_label", "label2", "second_best", "lab2"):
            out["second"] = c
        elif rest in ("margin", "marg", "delta"):
            out["margin"] = c
    return out


def read_section(label_file, level, h5ad_dir):
    df = pd.read_csv(label_file)
    cols = list(df.columns)
    spec = {}
    for arm in ARMS_NEEDED:
        c = find_cols(cols, arm, level)
        need = ("label", "second", "margin") if arm == "nucB" else ("label",)
        miss = [k for k in need if k not in c]
        if miss:
            raise SystemExit(f"{os.path.basename(label_file)}: no {miss} column for arm {arm} (level {level}).\n"
                             f"Columns: {cols}\nRun with --inspect and pass --level, or rename.")
        spec[arm] = c
    # coordinates and metadata
    xy = [c for c in ("x", "y") if c in cols]
    meta = {}
    if len(xy) == 2:
        x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)
        for k in ("donor", "panel"):
            if k in cols:
                meta[k] = df[k].iloc[0]
    else:
        stem = os.path.basename(label_file).split(".labels")[0]
        cands = glob.glob(os.path.join(h5ad_dir or "", stem + "*.h5ad"))
        if len(cands) != 1:
            raise SystemExit(f"no coordinates in {label_file} and {len(cands)} h5ad matches for {stem} in {h5ad_dir}")
        import anndata as ad
        obs = ad.read_h5ad(cands[0], backed="r").obs
        idcol = cols[0]
        if df[idcol].astype(str).isin(obs.index.astype(str)).all():
            obs = obs.loc[df[idcol].astype(str).to_numpy()]
        elif len(obs) != len(df):
            raise SystemExit(f"{label_file}: cannot match {len(df)} label rows to {len(obs)} cells of {cands[0]}")
        x, y = obs["x"].to_numpy(float), obs["y"].to_numpy(float)
        for k in ("donor", "panel"):
            if k in obs.columns:
                meta[k] = str(obs[k].iloc[0])
    lab = {arm: df[spec[arm]["label"]].astype("string").to_numpy() for arm in ARMS_NEEDED}
    second = df[spec["nucB"]["second"]].astype("string").to_numpy()
    margin = df[spec["nucB"]["margin"]].to_numpy(float)
    nB = df["n_nucB"].to_numpy(float) if "n_nucB" in cols else np.full(len(df), np.nan)
    extra = {k: df[k].to_numpy(float) for k in ("n_whole", "n_nucleus") if k in cols}
    if h5ad_dir and "cell_id" in cols:      # areas for the logit arm
        stem = os.path.basename(label_file).split(".labels")[0]
        cands = glob.glob(os.path.join(h5ad_dir, stem + "*.h5ad"))
        if len(cands) == 1:
            import anndata as ad
            obs = ad.read_h5ad(cands[0], backed="r").obs
            ids = df["cell_id"].astype(str)
            if ids.isin(obs.index.astype(str)).all():
                o = obs.loc[ids.to_numpy()]
                for k in ("cell_area", "nucleus_area"):
                    if k in o:
                        extra[k] = o[k].to_numpy(float)
    return dict(x=x, y=y, lab=lab, second=second, margin=margin, nB=nB, extra=extra, meta=meta, spec=spec)


def infer_meta(name, meta):
    donor = meta.get("donor")
    panel = meta.get("panel")
    m = re.search(r"_(L\d)_L\d", name)
    if donor is None and m:
        donor = m.group(1)
    if panel is None:
        panel = "5K" if "_5k_" in name else ("CustomIO" if "chuvio" in name else ("Lung" if "_lung_" in name else "?"))
    return str(donor), str(panel)


# ----------------------------------------------------------------------------------------------- geometry
def neighbours(x, y, k):
    tree = cKDTree(np.column_stack([x, y]))
    d, i = tree.query(np.column_stack([x, y]), k=k + 1)
    return d[:, 1:], i[:, 1:]


def strata(codes_nuc, d20, i20, K, n_contact=6, r_contact=30.0, r_far=60.0):
    """int8 array n x K: 2 if type present among contact neighbors, 1 if only among far ones, 0 absent.
    Neighbors are typed by the full-nucleus labels (as in bilous_compare.py)."""
    n = len(codes_nuc)
    S = np.zeros((n, K), np.int8)
    rows = np.repeat(np.arange(n), i20.shape[1]).reshape(i20.shape)
    nt = codes_nuc[i20]
    ok = nt >= 0
    rank = np.broadcast_to(np.arange(i20.shape[1]), i20.shape)
    far = ok & (rank >= n_contact) & (d20 <= r_far)
    con = ok & (rank < n_contact) & (d20 <= r_contact)
    S[rows[far], nt[far]] = 1
    S[rows[con], nt[con]] = 2
    return S


# ----------------------------------------------------------------------------------------------- statistics
def coloc(codes, i10, K, keep):
    """log2 O/E for ordered pairs (a index, b neighbor) on a fixed graph. NaN for types not in `keep`."""
    k = i10.shape[1]
    n = len(codes)
    la = np.repeat(codes, k)
    lb = codes[i10.ravel()]
    M = np.bincount(la * K + lb, minlength=K * K).reshape(K, K).astype(float)
    na = np.bincount(codes, minlength=K).astype(float)
    exp = k * (na[None, :] - np.eye(K)) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log2((M / na[:, None]) / exp)
    L[~keep, :] = np.nan
    L[:, ~keep] = np.nan
    return L


def enrich_z(codes, i10, K, nperm, rng):
    k = i10.shape[1]
    src = np.repeat(np.arange(len(codes)), k)
    dst = i10.ravel()

    def counts(c):
        return np.bincount(c[src] * K + c[dst], minlength=K * K).astype(float)

    obs = counts(codes)
    P = np.empty((nperm, K * K))
    for p in range(nperm):
        P[p] = counts(rng.permutation(codes))
    sd = P.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (obs - P.mean(0)) / sd
    return z.reshape(K, K)


def call(z, thr):
    return np.where(z <= -thr, -1, np.where(z >= thr, 1, 0))


# ----------------------------------------------------------------------------------------------- flips
def excess_counts(B, W, A, K):
    """n[a, b] = (# nucB=a & whole=b) - (# nucB=a & nucA=b), a != b, floored at 0, rounded."""
    cw = np.bincount(B * K + W, minlength=K * K).reshape(K, K)
    cn = np.bincount(B * K + A, minlength=K * K).reshape(K, K)
    n = np.clip(np.rint(cw - cn), 0, None).astype(int)
    np.fill_diagonal(n, 0)
    return n


def strata_count(codes_nuc, d20, i20, K, n_contact=6, r_contact=30.0, r_far=60.0):
    """Dose levels per (cell, type): 0 absent, 1 only among far neighbors, 1 + c for c = 1..6 cells of the type
    among the contact neighbors (diagnostic kernel_count)."""
    S = strata(codes_nuc, d20, i20, K, n_contact, r_contact, r_far)
    n = len(codes_nuc)
    nt = codes_nuc[i20[:, :n_contact]]
    con = (nt >= 0) & (d20[:, :n_contact] <= r_contact)
    C = np.zeros((n, K), np.int16)
    rows = np.repeat(np.arange(n), n_contact).reshape(nt.shape)
    np.add.at(C, (rows[con], nt[con]), 1)
    return np.where(C > 0, 1 + C, S).astype(np.int8)


def stratum_counts(B, W, A, S, K):
    """Per (a, b, stratum s): denominators and real / null change counts, for fitting the kernel."""
    L = int(S.max()) + 1 if S.size else 3
    L = max(L, 3)
    den = np.zeros((K, K, L))
    nw = np.zeros((K, K, L))
    nn = np.zeros((K, K, L))
    for b in range(K):
        s = S[:, b]
        den[:, b, :] += np.bincount(B * L + s, minlength=K * L).reshape(K, L)
        mw = W == b
        mn = A == b
        nw[:, b, :] += np.bincount(B[mw] * L + s[mw], minlength=K * L).reshape(K, L)
        nn[:, b, :] += np.bincount(B[mn] * L + s[mn], minlength=K * L).reshape(K, L)
    return den, nw, nn


def kernel_rates(den, nw, nn, min_den, floor):
    """Excess rate per (a, b, stratum); pair-specific where the stratum has >= min_den cells, else pooled."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (nw - nn) / den
        pooled = (nw - nn).sum((0, 1)) / den.sum((0, 1))
    r = np.where(den >= min_den, r, pooled[None, None, :])
    return np.clip(np.nan_to_num(r, nan=floor), floor, None)


def apply_flips(B, n, chooser, S=None):
    """Apply n[a, b] flips to a copy of B; `chooser(a, b, cand_idx, m)` returns the chosen cells.
    Pairs with more flips go first; a cell is flipped at most once. Returns new labels, shortfall and the share of
    flipped cells that have their new type among their contact neighbors (a check of the placement)."""
    out = B.copy()
    used = np.zeros(len(B), bool)
    short = 0
    contact = 0
    pairs = sorted(((n[a, b], a, b) for a in range(n.shape[0]) for b in range(n.shape[1]) if n[a, b] > 0),
                   reverse=True)
    for m, a, b in pairs:
        cand = np.flatnonzero((B == a) & ~used)
        if len(cand) == 0:
            short += m
            continue
        take = chooser(a, b, cand, min(m, len(cand)))
        short += m - len(take)
        out[take] = b
        used[take] = True
        if S is not None:
            contact += int((S[take, b] == 2).sum())
    return out, short, contact / max(int(used.sum()), 1)


def chooser_random(rng):
    return lambda a, b, cand, m: rng.choice(cand, m, replace=False)


CELL_FEATS = ("log_nucleus_area", "log_cell_area", "nucleus_cell_ratio", "outside_nucleus_share",
              "log_nucB_counts", "nucB_margin")
NB_FEATS = ("b_contact", "b_far", "b_share20", "a_contact", "second_is_b")
# feature sets of the logit arm (--logit-variants): what a user without a nucleus split could compute
LOGIT_VARIANTS = {
    "full": None,                                                     # every feature available
    "no_nucleus": ("log_nucleus_area", "nucleus_cell_ratio", "outside_nucleus_share"),   # dropped
    "no_areas": ("log_nucleus_area", "nucleus_cell_ratio", "outside_nucleus_share", "log_cell_area"),
    "neighbours": ("log_nucleus_area", "nucleus_cell_ratio", "outside_nucleus_share", "log_cell_area",
                   "log_nucB_counts", "nucB_margin", "second_is_b"),   # labels and positions only
}


def cell_features(extra, nB, margin):
    f = {}
    if "nucleus_area" in extra and "cell_area" in extra:
        na, ca = extra["nucleus_area"], extra["cell_area"]
        f["log_nucleus_area"] = np.log(np.maximum(na, 1e-3))
        f["log_cell_area"] = np.log(np.maximum(ca, 1e-3))
        f["nucleus_cell_ratio"] = na / np.maximum(ca, 1e-3)
    if "n_whole" in extra and "n_nucleus" in extra:
        f["outside_nucleus_share"] = 1 - extra["n_nucleus"] / np.maximum(extra["n_whole"], 1)
    if np.isfinite(nB).all():
        f["log_nucB_counts"] = np.log1p(nB)
    f["nucB_margin"] = margin
    return f


def neighbour_counts(codes_nuc, d20, i20, K, ncon=6, rcon=30.0, rfar=60.0):
    n = len(codes_nuc)
    nt = codes_nuc[i20]
    ok = nt >= 0
    rank = np.broadcast_to(np.arange(i20.shape[1]), i20.shape)
    con = ok & (rank < ncon) & (d20 <= rcon)
    far = ok & (rank >= ncon) & (d20 <= rfar)
    rows = np.repeat(np.arange(n), i20.shape[1]).reshape(i20.shape)
    Cc = np.zeros((n, K), np.int16); np.add.at(Cc, (rows[con], nt[con]), 1)
    Cf = np.zeros((n, K), np.int16); np.add.at(Cf, (rows[far], nt[far]), 1)
    C20 = np.zeros((n, K), np.int16); np.add.at(C20, (rows[ok], nt[ok]), 1)
    return Cc, Cf, C20


def pair_features(s, idx, a, b, cols):
    """Feature matrix for cells idx (all nucB = a) and destination b, in the order of cols."""
    out = []
    for c in cols:
        if c == "b_contact":
            out.append(s["Cc"][idx, b])
        elif c == "b_far":
            out.append((s["Cf"][idx, b] > 0).astype(float))
        elif c == "b_share20":
            out.append(s["C20"][idx, b] / 20.0)
        elif c == "a_contact":
            out.append(s["Cc"][idx, a])
        elif c == "second_is_b":
            out.append((s["second"][idx] == b).astype(float))
        else:
            out.append(s["cellF"][c][idx])
    return np.column_stack(out).astype(float)


def logit_rows(s, cols, neg_ratio, rng):
    """Case-control rows for the real outcome (wholeB == b) among nucB-a cells, pairs with positive excess."""
    B, W, n = s["B"], s["W"], s["n"]
    X, y, w, pr = [], [], [], []
    for a, b in zip(*np.nonzero(n > 0)):
        cand = np.flatnonzero(B == a)
        pos = cand[W[cand] == b]
        neg = cand[W[cand] != b]
        m = min(len(neg), neg_ratio * max(len(pos), 1))
        negs = rng.choice(neg, m, replace=False) if m < len(neg) else neg
        idx = np.concatenate([pos, negs])
        X.append(pair_features(s, idx, a, b, cols))
        y.append(np.r_[np.ones(len(pos)), np.zeros(len(negs))])
        w.append(np.r_[np.ones(len(pos)), np.full(len(negs), len(neg) / max(len(negs), 1))])
        pr.append(np.full(len(idx), a * 1000 + b))
    return np.vstack(X), np.concatenate(y), np.concatenate(w), np.concatenate(pr)


def fit_logit(train, cols):
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([t["lrow"][0] for t in train]); y = np.concatenate([t["lrow"][1] for t in train])
    w = np.concatenate([t["lrow"][2] for t in train]); pr = np.concatenate([t["lrow"][3] for t in train])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    P = pd.get_dummies(pd.Series(pr), dtype=float).to_numpy()
    m = LogisticRegression(max_iter=3000).fit(np.hstack([(X - mu) / sd, P]), y, sample_weight=w)
    return dict(beta=m.coef_[0][: len(cols)], mu=mu, sd=sd)


def chooser_logit(s, model, cols, rng):
    """Gumbel-top-k: sampling without replacement with probability proportional to the odds exp(z.beta)."""
    def f(a, b, cand, m):
        z = (pair_features(s, cand, a, b, cols) - model["mu"]) / model["sd"]
        keys = z @ model["beta"] + rng.gumbel(size=len(cand))
        return cand[np.argpartition(-keys, m - 1)[:m]] if m < len(cand) else cand
    return f


def chooser_kernel(S, rates, rng):
    def f(a, b, cand, m):
        w = rates[a, b, S[cand, b]]
        keys = np.log(rng.random(len(cand))) / w  # Efraimidis-Spirakis
        return cand[np.argpartition(-keys, m - 1)[:m]] if m < len(cand) else cand
    return f


def chooser_oracle(W, rng):
    """Diagnostic: the a->b flips placed on cells that really changed a->b (nucB a, wholeB b)."""
    def f(a, b, cand, m):
        real = cand[W[cand] == b]
        return rng.choice(real, min(m, len(real)), replace=False)
    return f


def chooser_ambiguity(second_codes, margin):
    def f(a, b, cand, m):
        first = cand[second_codes[cand] == b]
        first = first[np.argsort(margin[first], kind="stable")]
        if len(first) >= m:
            return first[:m]
        rest = np.setdiff1d(cand, first)
        rest = rest[np.argsort(margin[rest], kind="stable")]
        return np.concatenate([first, rest[: m - len(first)]])
    return f


def signed_excess(B, W, A, K):
    """Off-diagonal (wholeB - nucA) change counts from nucB: positive and negative parts, and all changes."""
    cw = np.bincount(B * K + W, minlength=K * K).reshape(K, K).astype(float)
    cn = np.bincount(B * K + A, minlength=K * K).reshape(K, K).astype(float)
    d = cw - cn
    np.fill_diagonal(d, 0)
    return dict(excess_pos=int(d[d > 0].sum()), excess_neg=int(-d[d < 0].sum()),
                changes_whole=int((B != W).sum()), changes_null=int((B != A).sum()))


def real_contact_share(B, W, n, S):
    """Share of the real nucB->wholeB changes on excess pairs whose new type is among the contact neighbors."""
    m = (B != W) & (n[B, W] > 0)
    return float((S[np.flatnonzero(m), W[m]] == 2).mean()) if m.any() else float("nan")


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="folder with <sample>.labels.csv.gz (bilous_label.py output)")
    ap.add_argument("--h5ad-dir", help="folder with the converted h5ad files (coordinates, donor, panel)")
    ap.add_argument("--level", default="L2m")
    ap.add_argument("--rule", help="criteria YAML (thresholds of the verdict)")
    ap.add_argument("--out")
    ap.add_argument("--immune", default="T/NK,Myeloid,B")
    ap.add_argument("--partners", default="Tumour,Fibro/muscle,Vessel")
    ap.add_argument("--min-cells", type=int, default=200, help="types with fewer nucB cells are not reported")
    ap.add_argument("--draws", type=int, default=20, help="draws of the random and kernel arms")
    ap.add_argument("--z-draws", type=int, default=3, help="draws on which the enrichment call is computed")
    ap.add_argument("--nperm", type=int, default=100)
    ap.add_argument("--z-thr", type=float, default=2.0)
    ap.add_argument("--min-den", type=int, default=50)
    ap.add_argument("--rate-floor", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--sections", help="comma-separated substrings to restrict sections (testing)")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--diagnostic", action="store_true",
                    help="diagnostic: add the oracle arm (the excess flips placed on the cells that really changed) and "
                         "the signed excess; the verdict is recorded as a diagnostic, not as the decision")
    ap.add_argument("--skip-calls", action="store_true", help="skip the enrichment calls (faster)")
    ap.add_argument("--logit", action="store_true",
                    help="with --diagnostic: placement by a per-cell logistic model of the real flips, fitted on the "
                         "other donors (neighbourhood and cell features; areas need --h5ad-dir)")
    ap.add_argument("--neg-ratio", type=int, default=10)
    ap.add_argument("--logit-variants", default="full",
                    help="comma list of logit feature sets: " + ", ".join(LOGIT_VARIANTS) + " (arm 'logit' = full, "
                         "others 'logit_<name>'); all fitted leaving one donor out")
    ap.add_argument("--insample", action="store_true",
                    help="with --diagnostic: also kernels fitted on all donors (separates transfer from features)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.labels, "*.labels.csv.gz")))
    if a.sections:
        keys = a.sections.split(",")
        files = [f for f in files if any(k in f for k in keys)]
    if not files:
        raise SystemExit(f"no *.labels.csv.gz in {a.labels}")

    if a.inspect:
        df = pd.read_csv(files[0], nrows=5)
        print(files[0]); print(list(df.columns)); print(df.head().to_string())
        for arm in ARMS_NEEDED:
            print(arm, find_cols(list(df.columns), arm, a.level))
        return

    if a.logit:
        a.diagnostic = True
    if not a.rule or not a.out:
        raise SystemExit("--rule and --out are required for a run (with --diagnostic too, to record the same rule)")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    with open(a.rule) as f:
        import yaml
        rule = yaml.safe_load(f)
    rule_rec = dict(path=os.path.abspath(a.rule), sha256=sha256(a.rule),
                    mtime=datetime.fromtimestamp(os.path.getmtime(a.rule), timezone.utc).isoformat())
    started = datetime.now(timezone.utc).isoformat()
    os.makedirs(a.out)

    # ---- pass 1: read sections, common label space, geometry, stratum counts
    secs = []
    names_all = set()
    for f in files:
        name = os.path.basename(f).split(".labels")[0]
        d = read_section(f, a.level, a.h5ad_dir)
        donor, panel = infer_meta(name, d["meta"])
        keep = np.ones(len(d["x"]), bool)
        for arm in ARMS_NEEDED:
            keep &= ~pd.isna(d["lab"][arm])
        keep &= ~pd.isna(d["second"]) & np.isfinite(d["margin"])
        for arm in ARMS_NEEDED:
            names_all.update(pd.unique(d["lab"][arm][keep]).tolist())
        secs.append(dict(name=name, donor=donor, panel=panel, d=d, keep=keep))
        print(f"read {name}: donor {donor}, panel {panel}, {keep.sum():,} of {len(keep):,} cells labelled in all arms",
              flush=True)
    names = sorted(names_all)
    K = len(names)
    code = {s: i for i, s in enumerate(names)}
    immune = a.immune.split(",")
    partners = a.partners.split(",")
    missing = [t for t in immune + partners if t not in code]
    if missing:
        raise SystemExit(f"types {missing} not in labels {names}; pass --immune / --partners")
    pairs_ip = [(i, p) for i in immune for p in partners]
    pairs_h = [(t, t) for t in names]

    def enc(arr):
        return np.array([code.get(v, -1) if not pd.isna(v) else -1 for v in arr], np.int64)

    for s in secs:
        d, keep = s["d"], s["keep"]
        x, y = d["x"][keep], d["y"][keep]
        s["B"], s["W"], s["A"] = enc(d["lab"]["nucB"][keep]), enc(d["lab"]["wholeB"][keep]), enc(d["lab"]["nucA"][keep])
        s["N"] = enc(d["lab"]["nucleus"][keep])
        s["second"], s["margin"] = enc(d["second"][keep]), d["margin"][keep]
        s["margin_pc"] = d["margin"][keep] / np.maximum(d["nB"][keep], 1)   # margin per nucB count (secondary)
        s["cellF"] = cell_features({k: v[keep] for k, v in d["extra"].items()}, d["nB"][keep], d["margin"][keep])
        d20, i20 = neighbours(x, y, 20)
        s["i10"] = i20[:, :10]
        s["S"] = strata(s["N"], d20, i20, K)
        s["cnt"] = stratum_counts(s["B"], s["W"], s["A"], s["S"], K)
        if a.diagnostic:
            s["Sc"] = strata_count(s["N"], d20, i20, K)
            s["cntc"] = stratum_counts(s["B"], s["W"], s["A"], s["Sc"], K)
        if a.logit:
            s["Cc"], s["Cf"], s["C20"] = neighbour_counts(s["N"], d20, i20, K)
        s["n"] = excess_counts(s["B"], s["W"], s["A"], K)
        del s["d"]

    rng = np.random.default_rng(a.seed)
    logit_cols, logit_models, lv_cols = None, {}, {}
    if a.logit:
        common = set.intersection(*[set(s["cellF"]) for s in secs])
        logit_cols = list(NB_FEATS) + [c for c in CELL_FEATS if c in common]
        print("logit features:", logit_cols, flush=True)
        for s in secs:
            s["lrow"] = logit_rows(s, logit_cols, a.neg_ratio, rng)      # one case-control sample, all variants
        for v in a.logit_variants.split(","):
            if v not in LOGIT_VARIANTS:
                raise SystemExit(f"unknown logit variant {v!r}: {', '.join(LOGIT_VARIANTS)}")
            drop = LOGIT_VARIANTS[v] or ()
            cols = [c for c in logit_cols if c not in drop]
            arm = "logit" if v == "full" else f"logit_{v}"
            lv_cols[arm] = cols
            j = [logit_cols.index(c) for c in cols]
            sub = lambda t: dict(lrow=(t["lrow"][0][:, j],) + tuple(t["lrow"][1:]))
            logit_models[arm] = {}
            for dn in sorted({s["donor"] for s in secs}):
                logit_models[arm][dn] = fit_logit([sub(t) for t in secs if t["donor"] != dn], cols)
                print(f"{arm} without {dn}: " + ", ".join(f"{c} {x:+.2f}" for c, x in
                                                       zip(cols, logit_models[arm][dn]["beta"])), flush=True)
    rows, zrows, flips = [], [], []
    rates_out = {}
    for s in secs:
        # kernel rates: leave this donor out; panel hold-out for 5K
        def fit(train, key="cnt"):
            L = max(t[key][0].shape[2] for t in train)
            pad = lambda x: np.pad(x, ((0, 0), (0, 0), (0, L - x.shape[2])))
            den = sum(pad(t[key][0]) for t in train); nw = sum(pad(t[key][1]) for t in train)
            nn = sum(pad(t[key][2]) for t in train)
            return kernel_rates(den, nw, nn, a.min_den, a.rate_floor)
        train = [t for t in secs if t["donor"] != s["donor"]]
        arm_rates = {"kernel": (fit(train), s["S"]) if train else None}
        if s["panel"] == "5K":
            tp = [t for t in secs if t["panel"] != "5K"]
            if tp:
                arm_rates["kernel_panel"] = (fit(tp), s["S"])
        if a.diagnostic and train:
            Rc = fit(train, "cntc")
            Sc = np.minimum(s["Sc"], Rc.shape[2] - 1)
            arm_rates["kernel_count"] = (Rc, Sc)
        if a.logit:
            for arm in logit_models:
                arm_rates[arm] = (arm, logit_models[arm][s["donor"]])
        if a.diagnostic and a.insample:   # rates fitted on all donors, the test donor included (not held out)
            arm_rates["kernel_insample"] = (fit(secs), s["S"])
            Ri = fit(secs, "cntc")
            arm_rates["kernel_count_insample"] = (Ri, np.minimum(s["Sc"], Ri.shape[2] - 1))
        rates_out[s["name"]] = {k: (v[0].tolist() if not isinstance(v[0], str) else
                                    dict(zip(lv_cols[v[0]], v[1]["beta"].tolist())))
                                for k, v in arm_rates.items() if v is not None}

        B, n, i10 = s["B"], s["n"], s["i10"]
        keepT = np.bincount(B, minlength=K) >= a.min_cells
        base = coloc(B, i10, K, keepT)
        real = coloc(s["W"], i10, K, keepT) - base
        arms = {"random": [], "ambiguity": [], "ambiguity_pc": []}
        if a.diagnostic:
            arms["oracle"] = []
        for k in arm_rates:
            if arm_rates[k] is not None:
                arms[k] = []
        labelsets = {"nucB": B, "wholeB": s["W"]}
        short, ccon = {}, {}
        lab, sh, cc = apply_flips(B, n, chooser_ambiguity(s["second"], s["margin"]), s["S"])
        arms["ambiguity"].append(coloc(lab, i10, K, keepT) - base); short["ambiguity"] = sh; ccon["ambiguity"] = [cc]
        labelsets["ambiguity#0"] = lab
        if a.diagnostic:
            for _ in range(min(a.draws, 5)):
                lab, sh, cc = apply_flips(B, n, chooser_oracle(s["W"], rng), s["S"])
                arms["oracle"].append(coloc(lab, i10, K, keepT) - base)
                short["oracle"] = sh; ccon.setdefault("oracle", []).append(cc)
        if np.isfinite(s["margin_pc"]).all():
            lab, sh, cc = apply_flips(B, n, chooser_ambiguity(s["second"], s["margin_pc"]), s["S"])
            arms["ambiguity_pc"].append(coloc(lab, i10, K, keepT) - base)
            short["ambiguity_pc"] = sh; ccon["ambiguity_pc"] = [cc]
        for dr in range(a.draws):
            lab, sh, cc = apply_flips(B, n, chooser_random(rng), s["S"])
            arms["random"].append(coloc(lab, i10, K, keepT) - base); short["random"] = sh
            ccon.setdefault("random", []).append(cc)
            if dr < a.z_draws:
                labelsets[f"random#{dr}"] = lab
            for k, RS in arm_rates.items():
                if RS is None:
                    continue
                R, Sk = RS
                ch = chooser_logit(s, Sk, lv_cols[R], rng) if isinstance(R, str) else chooser_kernel(Sk, R, rng)
                lab, sh, cc = apply_flips(B, n, ch, s["S"])
                arms[k].append(coloc(lab, i10, K, keepT) - base); short[k] = sh
                ccon.setdefault(k, []).append(cc)
                if dr < a.z_draws:
                    labelsets[f"{k}#{dr}"] = lab
        flips.append(dict(section=s["name"], donor=s["donor"], panel=s["panel"], cells=len(B),
                          excess_flips=int(n.sum()), excess_share=float(n.sum() / len(B)),
                          **signed_excess(s["B"], s["W"], s["A"], K),
                          real_contact_share=real_contact_share(B, s["W"], n, s["S"]),
                          **{f"shortfall_{k}": int(v) for k, v in short.items()},
                          **{f"contact_share_{k}": float(np.mean(v)) for k, v in ccon.items()}))
        for (ta, tb), group in [(p, "immune-partner") for p in pairs_ip] + [(p, "homotypic") for p in pairs_h]:
            ia, ib = code[ta], code[tb]
            r = dict(section=s["name"], donor=s["donor"], panel=s["panel"], pair=f"{ta}|{tb}", group=group,
                     baseline=base[ia, ib], real=real[ia, ib])
            for k, v in arms.items():
                r[k] = float(np.nanmean([m[ia, ib] for m in v])) if v else np.nan
            rows.append(r)
        # enrichment calls
        zc = {}
        for k, lab in ({} if a.skip_calls else labelsets).items():
            zc[k] = call(enrich_z(lab, i10, K, a.nperm, rng), a.z_thr)
        for (ta, tb), group in [(p, "immune-partner") for p in pairs_ip] + [(p, "homotypic") for p in pairs_h]:
            ia, ib = code[ta], code[tb]
            if not (keepT[ia] and keepT[ib]) or not zc:
                continue
            r = dict(section=s["name"], donor=s["donor"], panel=s["panel"], pair=f"{ta}|{tb}", group=group)
            for k, c in zc.items():
                r[k] = int(c[ia, ib])
            zrows.append(r)
        print(f"done {s['name']}: {n.sum():,} excess flips ({n.sum() / len(B):.3f} of cells)", flush=True)

    sec = pd.DataFrame(rows)
    sec.to_csv(os.path.join(a.out, "coloc_sections.csv"), index=False)
    pd.DataFrame(zrows).to_csv(os.path.join(a.out, "calls_sections.csv"), index=False)
    pd.DataFrame(flips).to_csv(os.path.join(a.out, "flips.csv"), index=False)
    with open(os.path.join(a.out, "kernel_rates.json"), "w") as f:
        json.dump(dict(types=names, strata=["absent", "far only", "contact"], rates=rates_out), f)

    # ---- donor-level summary: median over a donor's sections, then over donors
    armcols = [c for c in ("real", "oracle", "random", "ambiguity", "ambiguity_pc", "kernel", "kernel_count",
                           "kernel_insample", "kernel_count_insample", "logit", "kernel_panel")
               if c in sec.columns] + sorted(c for c in sec.columns if c.startswith("logit_"))
    don = sec.groupby(["pair", "group", "donor"])[["baseline"] + armcols].median().reset_index()
    don.to_csv(os.path.join(a.out, "coloc_donors.csv"), index=False)
    summ = don.groupby(["pair", "group"])[["baseline"] + armcols].median().reset_index()
    summ["donors"] = don.groupby(["pair", "group"]).size().to_numpy()
    summ.to_csv(os.path.join(a.out, "coloc_summary.csv"), index=False)
    if "kernel_panel" in sec.columns:
        k5 = sec[sec.panel == "5K"].groupby(["pair", "group"])[["real", "kernel", "kernel_panel", "random"]].median()
        k5.reset_index().to_csv(os.path.join(a.out, "coloc_5K_panel_holdout.csv"), index=False)

    # ---- call changes
    zc = pd.DataFrame(zrows)
    callsum = {}
    if len(zc):
        ks = [c for c in zc.columns if "#" in c]
        groups = {}
        for c in ks:
            groups.setdefault(c.split("#")[0], []).append(c)
        callsum["changed_vs_nucB"] = {"wholeB": float((zc["wholeB"] != zc["nucB"]).mean())}
        callsum["agree_with_wholeB"] = {}
        for g, cols in groups.items():
            sub = [zc[[c, "nucB", "wholeB"]].dropna() for c in cols]
            callsum["changed_vs_nucB"][g] = float(np.mean([(t[c] != t["nucB"]).mean() for t, c in zip(sub, cols)]))
            callsum["agree_with_wholeB"][g] = float(np.mean([(t[c] == t["wholeB"]).mean() for t, c in zip(sub, cols)]))
        whole_changed = zc[zc["wholeB"] != zc["nucB"]]
        callsum["reproduces_wholeB_change"] = {}
        for g, cols in groups.items():
            sub = [whole_changed[[c, "wholeB"]].dropna() for c in cols]
            callsum["reproduces_wholeB_change"][g] = (
                float(np.mean([(t[c] == t["wholeB"]).mean() for t, c in zip(sub, cols)])) if len(sub[0]) else None)
        callsum["pairs_where_wholeB_changes_call"] = whole_changed.groupby("pair").size().to_dict()
        callsum["n_pair_sections"] = int(len(zc))

    # ---- verdict
    verdict = evaluate(summ, rule)
    if a.diagnostic:
        verdict = dict(note="diagnostic run: additional comparisons", **verdict)
        if "oracle" in summ:
            inf = summ[summ["real"].abs() >= verdict["informative_floor"]]
            verdict["sign_match_oracle"] = int((np.sign(inf.oracle) == np.sign(inf.real)).sum())
            verdict["abs_error_oracle_median"] = float((inf.oracle - inf.real).abs().median())
        inf = summ[summ["real"].abs() >= verdict["informative_floor"]]
        for k in ("kernel_count", "kernel_insample", "kernel_count_insample", "logit",
                  *sorted(c for c in summ.columns if c.startswith("logit_"))):
            if k in summ:
                verdict[f"sign_match_{k}"] = int((np.sign(inf[k]) == np.sign(inf.real)).sum())
                verdict[f"abs_error_{k}_median"] = float((inf[k] - inf.real).abs().median())
    rec = dict(started=started, finished=datetime.now(timezone.utc).isoformat(), rule=rule_rec,
               script_sha256=sha256(os.path.abspath(__file__)), args=vars(a), types=names,
               sections=[dict(name=s["name"], donor=s["donor"], panel=s["panel"]) for s in secs],
               rule_before_run=rule_rec["mtime"] < started, calls=callsum, verdict=verdict)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(json.dumps(dict(verdict=verdict, calls=callsum), indent=1, default=str))
    if not rec["rule_before_run"]:
        print("WARNING: the criteria file was modified after the run started", file=sys.stderr)


def evaluate(summ, rule):
    r = rule.get("rule", rule)
    floor = float(r.get("informative_floor", 0.02))
    inf = summ[summ["real"].abs() >= floor]
    out = dict(informative_floor=floor, n_pairs=int(len(summ)), n_informative=int(len(inf)))
    if len(inf) == 0:
        out["outcome"] = "no informative pairs"
        return out
    for arm in ("kernel", "random", "ambiguity"):
        if arm not in inf:
            continue
        out[f"sign_match_{arm}"] = int((np.sign(inf[arm]) == np.sign(inf["real"])).sum())
        out[f"abs_error_{arm}_median"] = float((inf[arm] - inf["real"]).abs().median())
    if "kernel" in inf and "random" in inf:
        out["kernel_closer_than_random"] = int(((inf.kernel - inf.real).abs() < (inf.random - inf.real).abs()).sum())
    if "kernel" in inf and "ambiguity" in inf:
        out["kernel_closer_than_ambiguity"] = int(((inf.kernel - inf.real).abs() < (inf.ambiguity - inf.real).abs()).sum())
    m = len(inf)
    ok = m >= int(r.get("min_informative", 12))
    k_sign = out.get("sign_match_kernel", 0) / m >= float(r.get("kernel_sign_min", 0.8))
    r_sign = out.get("sign_match_random", m) / m <= float(r.get("random_sign_max", 1 / 3))
    k_size = out.get("kernel_closer_than_random", 0) / m > float(r.get("kernel_closer_min", 0.5))
    a_sign = out.get("sign_match_ambiguity", 0) / m >= float(r.get("kernel_sign_min", 0.8))
    out.update(enough_informative=ok, kernel_sign=k_sign, random_sign=r_sign, kernel_size=k_size, ambiguity_sign=a_sign)
    if not ok:
        out["outcome"] = "inconclusive: too few informative pairs"
    elif k_sign and r_sign and k_size:
        out["outcome"] = ("placement matters; expression ambiguity also suffices" if a_sign
                          else "placement matters; contact kernel needed, ambiguity alone does not suffice")
    elif r_sign and not k_sign:
        out["outcome"] = "random fails, but the held-out kernel does not reproduce the real change"
    else:
        out["outcome"] = "neither criterion met"
    return out


if __name__ == "__main__":
    main()
