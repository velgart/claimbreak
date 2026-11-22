#!/usr/bin/env python
"""bilous_flipcells.py -- which cells do whole-cell counts relabel?

The placement diagnostics showed that the real a->b changes (nucB a, wholeB b) reproduce the sign of the
colocalization change while contact-presence, contact-count and expression-ambiguity placements do not, and that
fitting those kernels on all donors barely helps. So the cells that flip differ from other contact cells in something
those kernels do not see. This script asks what, per cell:

  rows      cells labeled a by nucB, for every destination b with positive excess in the section
            (all positives; up to --neg-ratio sampled negatives per positive, case-control)
  outcome   real:  wholeB == b        null: nucA == b   (same rows; the null is what noise alone does)
  features  neighborhood: b among contact neighbors (count 0-6), b among far ones (0/1), share of b among 20 NN,
                           a among contact neighbors (count)
            the cell:      log nucleus area, log cell area, nucleus/cell area ratio, share of counts outside the
                           nucleus, log nucB counts, nucB margin, second-best nucB label == b
  model     logistic, standardised features, one-hot pair; leave-one-donor-out AUC for: neighborhood only,
            cell only, both; coefficients from all donors, for the real and the null outcome.

A feature that predicts the real flips but not the null flips is a property of whole-cell labeling, not of noise.

Usage
  python bilous_flipcells.py --labels ~/claimbreak_results/bilous_v2 --h5ad-dir ~/claimbreak_data/Bilous_h5ad \
      --out ~/claimbreak_results/bilous_flipcells
"""
import argparse, glob, os, re, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def infer(name):
    m = re.search(r"_(L\d)_L\d", name)
    panel = "5K" if "_5k_" in name else ("CustomIO" if "chuvio" in name else ("Lung" if "_lung_" in name else "?"))
    return (m.group(1) if m else "?"), panel


def load(f, level, h5ad_dir):
    df = pd.read_csv(f)
    name = os.path.basename(f).split(".labels")[0]
    col = lambda arm, suf="": f"{level}_{arm}{suf}"
    need = [col("nucB"), col("wholeB"), col("nucA"), col("nucleus"), col("nucB", "_second"), col("nucB", "_margin")]
    miss = [c for c in need if c not in df]
    if miss:
        raise SystemExit(f"{f}: missing {miss}")
    keep = df[need].notna().all(1).to_numpy()
    out = dict(name=name, df=df[keep].reset_index(drop=True))
    cands = glob.glob(os.path.join(h5ad_dir, name + "*.h5ad")) if h5ad_dir else []
    if len(cands) == 1:
        import anndata as ad
        obs = ad.read_h5ad(cands[0], backed="r").obs
        ids = out["df"]["cell_id"].astype(str)
        if ids.isin(obs.index.astype(str)).all():
            o = obs.loc[ids.to_numpy()]
            for k in ("cell_area", "nucleus_area"):
                if k in o:
                    out["df"][k] = o[k].to_numpy(float)
        else:
            print(f"  {name}: cell ids not in h5ad obs; area features skipped")
    return out


def features(d, names, level, ncon=6, rcon=30.0, rfar=60.0):
    df = d["df"]
    code = {s: i for i, s in enumerate(names)}
    enc = lambda c: df[c].map(code).fillna(-1).astype(int).to_numpy()
    B, W, A, N = enc(f"{level}_nucB"), enc(f"{level}_wholeB"), enc(f"{level}_nucA"), enc(f"{level}_nucleus")
    sec = enc(f"{level}_nucB_second")
    xy = df[["x", "y"]].to_numpy(float)
    dd, ii = cKDTree(xy).query(xy, k=21)
    dd, ii = dd[:, 1:], ii[:, 1:]
    K = len(names)
    n = len(df)
    nt = N[ii]
    con = (np.arange(20) < ncon)[None, :] & (dd <= rcon) & (nt >= 0)
    far = (np.arange(20) >= ncon)[None, :] & (dd <= rfar) & (nt >= 0)
    rows = np.repeat(np.arange(n), 20).reshape(n, 20)
    Ccon = np.zeros((n, K), np.int16); np.add.at(Ccon, (rows[con], nt[con]), 1)
    Cfar = np.zeros((n, K), np.int16); np.add.at(Cfar, (rows[far], nt[far]), 1)
    ok = nt >= 0
    C20 = np.zeros((n, K), np.int16); np.add.at(C20, (rows[ok], nt[ok]), 1)
    cell = {}
    if "nucleus_area" in df and "cell_area" in df:
        na, ca = df["nucleus_area"].to_numpy(float), df["cell_area"].to_numpy(float)
        cell["log_nucleus_area"] = np.log(np.maximum(na, 1e-3))
        cell["log_cell_area"] = np.log(np.maximum(ca, 1e-3))
        cell["nucleus_cell_ratio"] = na / np.maximum(ca, 1e-3)
    if "n_whole" in df and "n_nucleus" in df:
        nw, nn = df["n_whole"].to_numpy(float), df["n_nucleus"].to_numpy(float)
        cell["outside_nucleus_share"] = 1 - nn / np.maximum(nw, 1)
    if "n_nucB" in df:
        cell["log_nucB_counts"] = np.log1p(df["n_nucB"].to_numpy(float))
    cell["nucB_margin"] = df[f"{level}_nucB_margin"].to_numpy(float)
    return dict(B=B, W=W, A=A, second=sec, Ccon=Ccon, Cfar=Cfar, C20=C20, cell=cell)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--h5ad-dir")
    ap.add_argument("--level", default="L2m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-ratio", type=int, default=10)
    ap.add_argument("--min-pos", type=int, default=30, help="pairs with fewer real flips over all sections are dropped")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(a.out)
    rng = np.random.default_rng(a.seed)
    files = sorted(glob.glob(os.path.join(a.labels, "*.labels.csv.gz")))
    secs = [load(f, a.level, a.h5ad_dir) for f in files]
    names = sorted(set().union(*[set(s["df"][f"{a.level}_nucB"].dropna()) | set(s["df"][f"{a.level}_wholeB"].dropna())
                                 for s in secs]))
    K = len(names)
    parts = []
    for s in secs:
        donor, panel = infer(s["name"])
        F = features(s, names, a.level)
        B, W, A = F["B"], F["W"], F["A"]
        cw = np.bincount(B * K + W, minlength=K * K).reshape(K, K)
        cn = np.bincount(B * K + A, minlength=K * K).reshape(K, K)
        ex = cw - cn
        np.fill_diagonal(ex, 0)
        for ia in range(K):
            cand = np.flatnonzero(B == ia)
            for ib in np.flatnonzero(ex[ia] > 0):
                pos = cand[(W[cand] == ib) | (A[cand] == ib)]
                neg = cand[(W[cand] != ib) & (A[cand] != ib)]
                m = min(len(neg), a.neg_ratio * max(len(pos), 1))
                neg = rng.choice(neg, m, replace=False) if m < len(neg) else neg
                idx = np.concatenate([pos, neg])
                wneg = len(cand[(W[cand] != ib) & (A[cand] != ib)]) / max(len(neg), 1)
                r = pd.DataFrame(dict(section=s["name"], donor=donor, panel=panel, pair=f"{names[ia]}>{names[ib]}",
                                      y_real=(W[idx] == ib).astype(int), y_null=(A[idx] == ib).astype(int),
                                      weight=np.where((W[idx] == ib) | (A[idx] == ib), 1.0, wneg),
                                      b_contact=F["Ccon"][idx, ib], b_far=(F["Cfar"][idx, ib] > 0).astype(int),
                                      b_share20=F["C20"][idx, ib] / 20.0, a_contact=F["Ccon"][idx, ia],
                                      second_is_b=(F["second"][idx] == ib).astype(int)))
                for k, v in F["cell"].items():
                    r[k] = v[idx]
                parts.append(r)
        print(f"{s['name']}: donor {donor}, panel {panel}, {int((ex > 0).sum())} pairs with positive excess", flush=True)
    X = pd.concat(parts, ignore_index=True)
    npos = X.groupby("pair").y_real.sum()
    X = X[X.pair.isin(npos[npos >= a.min_pos].index)].reset_index(drop=True)
    X.to_csv(os.path.join(a.out, "rows.csv.gz"), index=False)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    nb = ["b_contact", "b_far", "b_share20", "a_contact"]
    ce = [c for c in ("log_nucleus_area", "log_cell_area", "nucleus_cell_ratio", "outside_nucleus_share",
                      "log_nucB_counts", "nucB_margin", "second_is_b") if c in X]
    P = pd.get_dummies(X["pair"], prefix="pair", dtype=float)

    def design(cols):
        Z = X[cols].astype(float)
        Z = (Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)
        return pd.concat([Z, P], axis=1)

    res, coefs = [], []
    for y in ("y_real", "y_null"):
        for setname, cols in (("neighbourhood", nb), ("cell", ce), ("both", nb + ce)):
            D = design(cols).to_numpy()
            pred = np.full(len(X), np.nan)
            for dn in X.donor.unique():
                te = (X.donor == dn).to_numpy()
                m = LogisticRegression(max_iter=3000, C=1.0).fit(D[~te], X[y][~te], sample_weight=X.weight[~te])
                pred[te] = m.predict_proba(D[te])[:, 1]
            auc = roc_auc_score(X[y], pred, sample_weight=X.weight)
            res.append(dict(outcome=y, features=setname, auc_lodo=auc))
            if setname == "both":
                m = LogisticRegression(max_iter=3000, C=1.0).fit(D, X[y], sample_weight=X.weight)
                for c, v in zip(cols, m.coef_[0][: len(cols)]):
                    coefs.append(dict(outcome=y, feature=c, coef_per_sd=v))
            print(y, setname, round(auc, 3), flush=True)
    res = pd.DataFrame(res); res.to_csv(os.path.join(a.out, "auc.csv"), index=False)
    co = pd.DataFrame(coefs).pivot(index="feature", columns="outcome", values="coef_per_sd")
    co["real_minus_null"] = co["y_real"] - co["y_null"]
    co.to_csv(os.path.join(a.out, "coefs.csv"))
    # per panel, coefficient of the key features for the real outcome (both features)
    per = []
    for pan, g in X.groupby("panel"):
        idx = g.index.to_numpy()
        D = design(nb + ce).to_numpy()[idx]
        for y in ("y_real", "y_null"):
            m = LogisticRegression(max_iter=3000).fit(D, g[y], sample_weight=g.weight)
            per += [dict(panel=pan, outcome=y, feature=c, coef_per_sd=v) for c, v in zip(nb + ce, m.coef_[0])]
    pd.DataFrame(per).to_csv(os.path.join(a.out, "coefs_by_panel.csv"), index=False)
    summary = dict(rows=len(X), pairs=int(X.pair.nunique()), positives_real=int(X.y_real.sum()),
                   positives_null=int(X.y_null.sum()), auc=res.to_dict("records"))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(co.round(3).to_string())


if __name__ == "__main__":
    main()
