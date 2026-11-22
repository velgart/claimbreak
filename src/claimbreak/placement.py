"""Where do the compartment test's excess label changes land, and does it matter?

For every section, the whole-cell labels (arm `wholeB`) differ from the half-B nucleus labels (`nucB`) by a number
of a -> b changes beyond the noise null (`nucA` against `nucB`). The same number of excess changes per ordered pair
is applied to the nucB labels in arms that differ only in where the changes land:

  random            uniformly among the section's nucB-a cells
  ambiguity         lowest nucB margin among cells whose second-best label is b (no spatial input)
  oracle            the cells that really changed a -> b (diagnostic upper reference)
  kernel            excess rate by contact stratum of b (absent / far only / contact), fitted on the other units
  kernel_count      the same with the number of contact neighbors of type b as strata
  transfer_<name>   a fixed flip model fitted elsewhere (packaged: GSE311609 lung, `default`, `no_nucleus`)
  louo_<variant>    a flip model fitted on the other units of this dataset (`full`, `no_nucleus`)

Flip models use claimbreak.flipmodel: features standardised over all cells of the dataset analyzed, odds
exp(z . beta), Gumbel top-k sampling without replacement. The readout is the change in kNN colocalization
(log2 O/E, k = 10, fixed graph) from nucB to each arm against the real change (nucB to wholeB), by pair, section and
unit (patient or donor): sign agreement on pairs whose real change is at least `informative` in absolute value.

As in the lung analysis, starting labels, margin and second label come from nucleus counts (half B), because the
errors are placed on them and compared with the real change. Neighbors are typed by the full-nucleus labels by
default; `nbr_arm="whole"` (or "wholeB") types them by whole-cell labels, as a user without a nucleus split would,
and with the no_nucleus feature set that is the setting of claimbreak's stress test on such data.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import flipmodel as fm

ARMS_NEEDED = ("nucA", "nucB", "wholeB", "nucleus")
NUCLEUS_FEATURES = ("outside_nucleus_share", "log_cell_area", "log_nucleus_area", "nucleus_cell_ratio")
LOUO_VARIANTS = {"full": (), "no_nucleus": NUCLEUS_FEATURES}


def unit_from(sample: str, pattern: str, fallback: str | None = None) -> str:
    """Unit of a section: the regex match on the sample name (the named group `unit` if the pattern has one, else
    the whole match; plain groups such as (AIN|ATI) do not select), else the fallback (a unit / donor column), else
    the sample. Same rule as compartment.unit_of."""
    m = re.search(pattern, str(sample)) if pattern else None
    if m:
        return m.group("unit") if "unit" in m.groupdict() else m.group(0)
    return str(fallback) if fallback not in (None, "", "nan") else str(sample)


# --------------------------------------------------------------------------- input

def read_section(f: Path, level: str, h5ad_dir=None, unit_pattern: str = r"I0\d{3}", nbr_arm: str = "nucleus") -> dict:
    arms = tuple(dict.fromkeys(("nucA", "nucB", "wholeB", nbr_arm)))
    D = pd.read_csv(f, keep_default_na=False, low_memory=False)
    sample = str(D["sample"].iloc[0]) if "sample" in D else f.name.split(".labels")[0]
    need = [f"{level}_{a}" for a in arms] + [f"{level}_nucB_second", f"{level}_nucB_margin", "x", "y"]
    miss = [c for c in need if c not in D]
    if miss:
        raise KeyError(f"{f.name}: missing columns {miss} (level {level}); columns: {list(D.columns)[:30]}")
    mg = pd.to_numeric(D[f"{level}_nucB_margin"], errors="coerce").to_numpy(float)
    keep = np.isfinite(mg) & (D[f"{level}_nucB_second"].astype(str) != "").to_numpy()
    for a in arms:
        keep &= (D[f"{level}_{a}"].astype(str) != "").to_numpy()
    D = D.loc[keep].reset_index(drop=True)
    extra = {}
    for c in ("n_whole", "n_nucleus", "n_nucB"):
        if c in D:
            extra[c] = pd.to_numeric(D[c], errors="coerce").to_numpy(float)
    for c in ("cell_area", "nucleus_area"):
        if c in D:
            extra[c] = pd.to_numeric(D[c], errors="coerce").to_numpy(float)
    if h5ad_dir and "cell_id" in D and not {"cell_area", "nucleus_area"} <= set(extra):
        p = Path(h5ad_dir).expanduser() / f"{sample}.h5ad"
        if not p.exists():
            c = sorted(Path(h5ad_dir).expanduser().glob(f"{sample}*.h5ad"))
            p = c[0] if len(c) == 1 else None
        if p is not None:
            import anndata as ad
            obs = ad.read_h5ad(p, backed="r").obs
            ids = D["cell_id"].astype(str)
            if ids.isin(obs.index.astype(str)).all():
                o = obs.loc[ids.to_numpy()]
                for c in ("cell_area", "nucleus_area"):
                    if c in o and np.isfinite(o[c].to_numpy(float)).any():
                        extra[c] = o[c].to_numpy(float)
    fb = D["unit"].iloc[0] if "unit" in D else (D["donor"].iloc[0] if "donor" in D else None)
    return dict(name=sample, unit=unit_from(sample, unit_pattern, fb), x=D["x"].to_numpy(float),
                y=D["y"].to_numpy(float), lab={a: D[f"{level}_{a}"].astype(str).to_numpy() for a in arms},
                second=D[f"{level}_nucB_second"].astype(str).to_numpy(), margin=mg[keep], extra=extra)


# --------------------------------------------------------------------------- geometry and statistics
# (as examples/compartment/bilous_placement.py, bd4502a7)

def strata(codes_nuc, d20, i20, K, n_contact=6, r_contact=30.0, r_far=60.0):
    """n x K: 2 if the type is among the contact neighbors, 1 if only among far ones, 0 absent."""
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


def neighbour_counts(codes_nuc, d20, i20, K, n_contact=6, r_contact=30.0, r_far=60.0):
    n = len(codes_nuc)
    nt = codes_nuc[i20]
    ok = nt >= 0
    rank = np.broadcast_to(np.arange(i20.shape[1]), i20.shape)
    con = ok & (rank < n_contact) & (d20 <= r_contact)
    far = ok & (rank >= n_contact) & (d20 <= r_far)
    rows = np.repeat(np.arange(n), i20.shape[1]).reshape(i20.shape)
    out = []
    for m in (con, far, ok):
        C = np.zeros((n, K), np.int16)
        np.add.at(C, (rows[m], nt[m]), 1)
        out.append(C)
    return out


def coloc_keep(codes, i10, K, keep):
    """log2 O/E for ordered pairs on a fixed graph; NaN for types not in `keep`."""
    k = i10.shape[1]
    n = len(codes)
    M = np.bincount(np.repeat(codes, k) * K + codes[i10.ravel()], minlength=K * K).reshape(K, K).astype(float)
    na = np.bincount(codes, minlength=K).astype(float)
    exp = k * (na[None, :] - np.eye(K)) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log2((M / na[:, None]) / exp)
    L[~keep, :] = np.nan
    L[:, ~keep] = np.nan
    return L


def excess_counts(B, W, A, K):
    cw = np.bincount(B * K + W, minlength=K * K).reshape(K, K)
    cn = np.bincount(B * K + A, minlength=K * K).reshape(K, K)
    n = np.clip(np.rint(cw - cn), 0, None).astype(int)
    np.fill_diagonal(n, 0)
    return n


def stratum_counts(B, W, A, S, K, L):
    den, nw, nn = (np.zeros((K, K, L)) for _ in range(3))
    for b in range(K):
        s = np.minimum(S[:, b], L - 1)
        den[:, b, :] += np.bincount(B * L + s, minlength=K * L).reshape(K, L)
        for M, m in ((nw, W == b), (nn, A == b)):
            M[:, b, :] += np.bincount(B[m] * L + s[m], minlength=K * L).reshape(K, L)
    return den, nw, nn


def kernel_rates(den, nw, nn, min_den=50, floor=1e-4):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (nw - nn) / den
        pooled = (nw - nn).sum((0, 1)) / den.sum((0, 1))
    r = np.where(den >= min_den, r, pooled[None, None, :])
    return np.clip(np.nan_to_num(r, nan=floor), floor, None)


def apply_flips(B, n, chooser, S):
    """Apply n[a, b] changes to a copy of B (larger pairs first, a cell changes at most once). Returns labels,
    shortfall and the share of changed cells with their new type among the contact neighbors."""
    out = B.copy()
    used = np.zeros(len(B), bool)
    short = contact = 0
    pairs = sorted(((n[a, b], a, b) for a, b in zip(*np.nonzero(n > 0))), reverse=True)
    for m, a, b in pairs:
        cand = np.flatnonzero((B == a) & ~used)
        if len(cand) == 0:
            short += m
            continue
        take = chooser(a, b, cand, min(m, len(cand)))
        short += m - len(take)
        out[take] = b
        used[take] = True
        contact += int((S[take, b] == 2).sum())
    return out, short, contact / max(int(used.sum()), 1)


def _topk(keys, cand, m):
    return cand[np.argpartition(-keys, m - 1)[:m]] if m < len(cand) else cand


def chooser_random(rng):
    return lambda a, b, cand, m: rng.choice(cand, m, replace=False)


def chooser_ambiguity(second, margin):
    def f(a, b, cand, m):
        first = cand[second[cand] == b]
        first = first[np.argsort(margin[first], kind="stable")]
        if len(first) >= m:
            return first[:m]
        rest = np.setdiff1d(cand, first)
        rest = rest[np.argsort(margin[rest], kind="stable")]
        return np.concatenate([first, rest[: m - len(first)]])
    return f


def chooser_oracle(W, rng):
    def f(a, b, cand, m):
        real = cand[W[cand] == b]
        return rng.choice(real, min(m, len(real)), replace=False)
    return f


def chooser_kernel(S, rates, rng):
    def f(a, b, cand, m):
        w = rates[a, b, np.minimum(S[cand, b], rates.shape[2] - 1)]
        return _topk(np.log(rng.random(len(cand))) / w, cand, m)   # Efraimidis-Spirakis
    return f


def chooser_model(design, off, cols, beta, rng):
    def f(a, b, cand, m):
        z = design.rows(cand + off, a, b)[:, cols]
        return _topk(z @ beta + rng.gumbel(size=len(cand)), cand, m)
    return f


# --------------------------------------------------------------------------- flip-model rows and fits

def model_rows(s, design, off, neg_ratio, rng, min_pos=10, max_pos=2000):
    """Case-control rows of the real outcome (wholeB == b among nucB-a cells) for pairs with excess changes: up to
    max_pos positives and neg_ratio negatives per positive per section and pair, weighted back to the full counts."""
    B, W, n = s["B"], s["W"], s["n"]
    X, y, w, pr = [], [], [], []
    for a, b in zip(*np.nonzero(n > 0)):
        cand = np.flatnonzero(B == a)
        pos_all = cand[W[cand] == b]
        if len(pos_all) < min_pos:
            continue
        pos = rng.choice(pos_all, max_pos, replace=False) if len(pos_all) > max_pos else pos_all
        neg = cand[W[cand] != b]
        m = min(len(neg), neg_ratio * len(pos))
        negs = rng.choice(neg, m, replace=False) if m < len(neg) else neg
        idx = np.concatenate([pos, negs])
        X.append(design.rows(idx + off, a, b))
        y.append(np.r_[np.ones(len(pos)), np.zeros(len(negs))])
        w.append(np.r_[np.full(len(pos), len(pos_all) / len(pos)), np.full(len(negs), len(neg) / max(len(negs), 1))])
        pr.append(np.full(len(idx), a * 1000 + b))
    if not X:
        return None
    return np.vstack(X), np.concatenate(y), np.concatenate(w), np.concatenate(pr)


def fit_rows(rows, cols):
    """Logistic fit with pair intercepts (discarded), as claimbreak.flipmodel.fit."""
    from sklearn.linear_model import LogisticRegression
    rows = [r for r in rows if r is not None]
    X = np.vstack([r[0][:, cols] for r in rows])
    y, w, pr = (np.concatenate([r[i] for r in rows]) for i in (1, 2, 3))
    import scipy.sparse as sp
    _, g = np.unique(pr, return_inverse=True)
    P = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g)))          # pair intercepts, sparse
    w = w * (len(w) / w.sum())                                            # mean weight 1: same optimum, better scaled
    m = LogisticRegression(max_iter=3000).fit(sp.hstack([sp.csr_matrix(X), P]).tocsr(), y, sample_weight=w)
    return m.coef_[0][: len(cols)], int(len(y)), int(y.sum())


def pair_auc(rows, cols, beta):
    """Mean within-pair AUC of z . beta on held-out rows, weighted by positives (pairs with both classes)."""
    from sklearn.metrics import roc_auc_score
    rows = [r for r in rows if r is not None]
    if not rows:
        return np.nan
    X = np.vstack([r[0][:, cols] for r in rows])
    y, w, pr = (np.concatenate([r[i] for r in rows]) for i in (1, 2, 3))
    sc = X @ beta
    num = den = 0.0
    for p in np.unique(pr):
        m = pr == p
        if y[m].min() == y[m].max():
            continue
        k = float(y[m].sum())
        num += k * roc_auc_score(y[m], sc[m], sample_weight=w[m])
        den += k
    return num / den if den else np.nan


# --------------------------------------------------------------------------- main entry

def placement(labels_dir, out, level: str, immune, partners, h5ad_dir=None, unit_pattern: str = r"I0\d{3}",
              use_sections=None, transfer=("default", "no_nucleus"), models: dict | None = None,
              louo=("full", "no_nucleus"), kernels: bool = True, oracle: bool = True, ambiguity: bool = True,
              draws: int = 20, min_cells: int = 200, informative: float = 0.02, neg_ratio: int = 10,
              max_pos: int = 2000,
              min_den: int = 50, k: int = 10, seed: int = 20261013, save_models: bool = False, nbr_arm: str = "nucleus",
              verbose: bool = True):
    """Run every arm on every section of LABELS_DIR (compartment label output with arms nucA, nucB, wholeB, nucleus)
    and write coloc_sections.csv, coloc_units.csv, coloc_summary.csv, sign_match.csv, sign_match_units.csv,
    flips.csv, models.json and meta.json into OUT (refuses to overwrite). use_sections: an iterable of sample names,
    or the path of a 'compartment summary' sections.csv (its `used` column) or a text file of names.
    transfer: packaged flip-model names; models: extra {name: JSON path}. With save_models, the flip models fitted
    on all units are written as flipmodel_<variant>.json (for transfer to another dataset)."""
    say = print if verbose else (lambda *a, **k: None)
    out = Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    files = sorted(Path(labels_dir).expanduser().glob("*.labels.csv.gz"))
    if use_sections is not None:
        if isinstance(use_sections, (str, Path)):
            p = Path(use_sections).expanduser()
            if p.suffix == ".csv":
                t = pd.read_csv(p)
                names = set(t.loc[t["used"].astype(bool), "sample"].astype(str)) if "used" in t else set(t["sample"])
            else:
                names = {ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
        else:
            names = set(map(str, use_sections))
        files = [f for f in files if f.name.split(".labels")[0] in names]
    if not files:
        raise FileNotFoundError(f"no labels files to use in {labels_dir}")
    immune, partners = list(immune), list(partners)
    started = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(seed)

    # ---- pass 1: read, common type list
    secs = [read_section(f, level, h5ad_dir, unit_pattern, nbr_arm) for f in files]
    types = sorted(set().union(*[set(np.concatenate(list(s["lab"].values()))) for s in secs]))
    missing = [t for t in immune + partners if t not in types]
    if missing:
        raise ValueError(f"types {missing} not in the labels {types}; check --immune / --partners / --level")
    K = len(types)
    code = {t: i for i, t in enumerate(types)}
    units = sorted({s["unit"] for s in secs})
    say(f"{len(secs)} sections, {len(units)} units ({', '.join(units)}), types {types}", flush=True)
    if len(units) < 2:
        say("one unit only: no held-out kernels or flip models", flush=True)

    # ---- pass 2: geometry, counts, features
    enc = lambda v: np.array([code[x] for x in v], np.int64)
    Lc = 8                               # kernel_count strata: 0 absent, 1 far only, 1 + c (c = 1..6 in contact)
    off = 0
    for s in secs:
        s["B"], s["W"], s["A"], s["N"] = (enc(s["lab"][a]) for a in ("nucB", "wholeB", "nucA", nbr_arm))
        s["sec2"] = np.array([code.get(x, -1) for x in s["second"]], np.int64)
        xy = np.column_stack([s["x"], s["y"]])
        d20, i20 = cKDTree(xy).query(xy, k=21)
        d20, i20 = d20[:, 1:], i20[:, 1:]
        s["i10"] = i20[:, :k]
        s["S"] = strata(s["N"], d20, i20, K)
        s["Cc"], s["Cf"], s["C20"] = neighbour_counts(s["N"], d20, i20, K)
        s["Sc"] = np.where(s["Cc"] > 0, 1 + s["Cc"], s["S"]).astype(np.int8)
        s["n"] = excess_counts(s["B"], s["W"], s["A"], K)
        s["cnt"] = stratum_counts(s["B"], s["W"], s["A"], s["S"], K, 3)
        s["cntc"] = stratum_counts(s["B"], s["W"], s["A"], s["Sc"], K, Lc)
        e = s["extra"]
        cf = {"margin": s["margin"]}
        if "n_nucB" in e:
            cf["log_counts"] = np.log1p(e["n_nucB"])
        if "n_whole" in e and "n_nucleus" in e:
            cf["outside_nucleus_share"] = np.clip(1 - e["n_nucleus"] / np.maximum(e["n_whole"], 1), 0, 1)
        if "cell_area" in e and "nucleus_area" in e and np.isfinite(e["nucleus_area"]).any():
            ca, na = e["cell_area"], e["nucleus_area"]
            cf.update(log_cell_area=np.log(np.maximum(ca, 1e-6)), log_nucleus_area=np.log(np.maximum(na, 1e-6)),
                      nucleus_cell_ratio=na / np.maximum(ca, 1e-6))
        s["cf"], s["off"] = cf, off
        off += len(s["B"])
        del s["lab"], s["second"], s["extra"], s["x"], s["y"]
        say(f"read {s['name']} ({s['unit']}): {len(s['B']):,} cells, {int(s['n'].sum()):,} excess changes "
            f"({s['n'].sum() / len(s['B']):.3f})", flush=True)

    feats_cell = sorted(set.intersection(*[set(s["cf"]) for s in secs]))
    nb = fm.Neighbourhood(contact=np.vstack([s["Cc"] for s in secs]), far=np.vstack([s["Cf"] for s in secs]),
                          k20=np.vstack([s["C20"] for s in secs]), types=np.array(types), k=20)
    cell = {f: np.concatenate([s["cf"][f] for s in secs]) for f in feats_cell}
    design = fm.Design(nb, cell, np.concatenate([s["sec2"] for s in secs]), fm.ALL_FEATURES)
    feats = list(design.features)
    say(f"flip-model features available: {feats}", flush=True)
    for s in secs:
        del s["Cc"], s["Cf"], s["C20"], s["cf"]

    # ---- flip models: fixed (transfer) and leave one unit out
    rec_models = {"features_available": feats, "transfer": {}, "louo": {}}
    fixed = {}
    for name in transfer or ():
        fixed[f"transfer_{name}"] = fm.FlipModel.default(name)
    for name, path in (models or {}).items():
        fixed[f"transfer_{name}"] = fm.FlipModel.from_json(path)
    rows = {s["name"]: model_rows(s, design, s["off"], neg_ratio, rng, max_pos=max_pos) for s in secs}
    by_unit = {u: [rows[s["name"]] for s in secs if s["unit"] == u] for u in units}
    plan = {}
    for arm, mdl in fixed.items():
        cols_f = [f for f in feats if f in mdl.beta]
        beta = np.array([mdl.beta[f] for f in cols_f])
        j = [feats.index(f) for f in cols_f]
        auc = {u: pair_auc(by_unit[u], j, beta) for u in units}
        rec_models["transfer"][arm] = dict(features=cols_f, missing=[f for f in mdl.beta if f not in feats],
                                           beta=dict(zip(cols_f, beta.tolist())), note=mdl.note,
                                           auc_by_unit=auc, auc_median=float(np.nanmedian(list(auc.values()))))
        plan[arm] = {u: (j, beta) for u in units}
        say(f"{arm}: features {cols_f}; not available {rec_models['transfer'][arm]['missing']}; within-pair AUC by "
            f"unit median {rec_models['transfer'][arm]['auc_median']:.3f}", flush=True)
    for v in (louo or ()) if len(units) > 1 else ():
        drop = LOUO_VARIANTS[v]
        cols_v = [f for f in feats if f not in drop]
        j = [feats.index(f) for f in cols_v]
        arm = f"louo_{v}"
        plan[arm], rec = {}, {"features": cols_v, "by_unit": {}}
        for u in units:
            beta, nr, npos = fit_rows([r for uu in units if uu != u for r in by_unit[uu]], j)
            auc = pair_auc(by_unit[u], j, beta)
            plan[arm][u] = (j, beta)
            rec["by_unit"][u] = dict(beta=dict(zip(cols_v, beta.tolist())), rows=nr, positives=npos, auc=auc)
        rec["auc_median"] = float(np.nanmedian([r["auc"] for r in rec["by_unit"].values()]))
        rec_models["louo"][arm] = rec
        bm = pd.DataFrame({u: r["beta"] for u, r in rec["by_unit"].items()}).median(axis=1)
        say(f"{arm}: held-out within-pair AUC median {rec['auc_median']:.3f}; median beta "
            + ", ".join(f"{c} {x:+.2f}" for c, x in bm.items()), flush=True)
        if save_models:
            beta, nr, npos = fit_rows([r for u in units for r in by_unit[u]], j)
            out.mkdir(parents=True, exist_ok=True)
            fm.FlipModel(beta=dict(zip(cols_v, map(float, beta))),
                         note=f"fitted by claimbreak.placement on {labels_dir} (level {level}), variant {v}",
                         provenance=dict(rows=nr, positives=npos, units=units, sections=[s["name"] for s in secs],
                                         features=cols_v)).to_json(out / f"flipmodel_{v}.json")

    # ---- arms per section
    pairs = [(i, p, "immune-partner") for i in immune for p in partners] + [(t, t, "homotypic") for t in types]
    crow, frow = [], []
    for s in secs:
        B, W, n, S, i10 = s["B"], s["W"], s["n"], s["S"], s["i10"]
        keepT = np.bincount(B, minlength=K) >= min_cells
        base = coloc_keep(B, i10, K, keepT)
        res = {"real": [coloc_keep(W, i10, K, keepT) - base]}
        short, ccon = {}, {}

        def run(arm, ch):
            lab, sh, cc = apply_flips(B, n, ch, S)
            res.setdefault(arm, []).append(coloc_keep(lab, i10, K, keepT) - base)
            short[arm] = sh
            ccon.setdefault(arm, []).append(cc)
        if ambiguity:
            run("ambiguity", chooser_ambiguity(s["sec2"], s["margin"]))
        if oracle:
            for _ in range(min(draws, 5)):
                run("oracle", chooser_oracle(W, rng))
        train = [t for t in secs if t["unit"] != s["unit"]]
        kr = {}
        if kernels and train:
            for key, arm in (("cnt", "kernel"), ("cntc", "kernel_count")):
                den, nw, nn = (sum(t[key][i] for t in train) for i in range(3))
                kr[arm] = (kernel_rates(den, nw, nn, min_den), S if arm == "kernel" else s["Sc"])
        for _ in range(draws):
            run("random", chooser_random(rng))
            for arm, (R, Sk) in kr.items():
                run(arm, chooser_kernel(Sk, R, rng))
            for arm, per in plan.items():
                j, beta = per[s["unit"]]
                run(arm, chooser_model(design, s["off"], j, beta, rng))
        m = (B != W) & (n[B, W] > 0)
        frow.append(dict(section=s["name"], unit=s["unit"], cells=len(B), excess_changes=int(n.sum()),
                         excess_share=float(n.sum() / len(B)), changes_whole=int((B != W).sum()),
                         changes_null=int((B != s["A"]).sum()),
                         real_contact_share=float((S[np.flatnonzero(m), W[m]] == 2).mean()) if m.any() else np.nan,
                         **{f"shortfall_{a}": int(v) for a, v in short.items()},
                         **{f"contact_share_{a}": float(np.mean(v)) for a, v in ccon.items()}))
        for ta, tb, group in pairs:
            ia, ib = code[ta], code[tb]
            r = dict(section=s["name"], unit=s["unit"], pair=f"{ta}|{tb}", group=group, baseline=base[ia, ib])
            for a, v in res.items():
                r[a] = float(np.nanmean([x[ia, ib] for x in v])) if np.isfinite([x[ia, ib] for x in v]).any() \
                    else np.nan
            crow.append(r)
        say(f"done {s['name']}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    sec = pd.DataFrame(crow)
    arms = [c for c in sec.columns if c not in ("section", "unit", "pair", "group", "baseline", "real")]
    sec.to_csv(out / "coloc_sections.csv", index=False)
    pd.DataFrame(frow).to_csv(out / "flips.csv", index=False)
    uni = sec.groupby(["pair", "group", "unit"])[["baseline", "real"] + arms].median().reset_index()
    uni.to_csv(out / "coloc_units.csv", index=False)
    summ = uni.groupby(["pair", "group"])[["baseline", "real"] + arms].median()
    summ["units"] = uni.dropna(subset=["real"]).groupby(["pair", "group"]).size()
    summ = summ.reset_index()
    summ.to_csv(out / "coloc_summary.csv", index=False)

    def match(T, by=None):
        T = T[T.real.abs() >= informative].dropna(subset=["real"])
        rows_ = []
        for key, g in (T.groupby(by) if by else [(None, T)]):
            r = {by: key} if by else {}
            r["informative_pairs"] = len(g)
            for a in arms:
                ok = g[a].notna()
                r[f"sign_{a}"] = int((np.sign(g.loc[ok, a]) == np.sign(g.loc[ok, "real"])).sum())
                r[f"abs_err_{a}"] = float((g.loc[ok, a] - g.loc[ok, "real"]).abs().median()) if ok.any() else np.nan
            rows_.append(r)
        return pd.DataFrame(rows_)
    sm = match(summ)
    sm.to_csv(out / "sign_match.csv", index=False)
    smu = match(uni, "unit")
    smu.to_csv(out / "sign_match_units.csv", index=False)
    (out / "models.json").write_text(json.dumps(rec_models, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(
        started=started, finished=datetime.now(timezone.utc).isoformat(), labels=str(labels_dir), level=level,
        immune=immune, partners=partners, h5ad_dir=str(h5ad_dir) if h5ad_dir else None, unit_pattern=unit_pattern,
        sections=[dict(name=s["name"], unit=s["unit"]) for s in secs], types=types, arms=arms,
        transfer=list(transfer or ()), models=models or {}, louo=list(louo or ()), draws=draws, min_cells=min_cells,
        nbr_arm=nbr_arm, informative=informative, neg_ratio=neg_ratio, max_pos=max_pos, min_den=min_den, k=k, seed=seed), indent=1))
    return dict(sections=sec, units=uni, summary=summ, sign_match=sm, sign_match_units=smu, flips=pd.DataFrame(frow),
                models=rec_models, arms=arms)
