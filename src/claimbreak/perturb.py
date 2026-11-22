"""Controlled perturbations of labels and counts.

Label errors (label_errors): each eligible cell is relabeled with probability eps * w_i, where w_i is a spatial
weight with mean 1 over eligible cells (random, boundary-concentrated, or condition-dependent); the wrong label is
drawn from a confusion matrix (native, within-lineage, uniform, or supplied), optionally biased toward the labels
typical of the cell's neighborhood (context bias, the "destination" effect).

Contamination (contaminate): each molecule leaves its cell with probability s (optionally type-dependent) and is
assigned to one of the k nearest cells with weights exp(-d / lam) (optionally receiver-type dependent). Total
counts are conserved.

Granularity (merge): collapse labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .tissue import Tissue, neighbour_counts


# --------------------------------------------------------------------------- confusion matrices
def uniform_confusion(types) -> pd.DataFrame:
    T = len(types)
    M = (1 - np.eye(T)) / max(T - 1, 1)
    return pd.DataFrame(M, index=types, columns=types)


def lineage_confusion(types, lineage) -> pd.DataFrame:
    """Errors only within a lineage (dict type -> group, or callable); types alone in their lineage fall back to
    uniform over all types."""
    types = list(types)
    grp = [lineage(t) if callable(lineage) else lineage.get(t, t) for t in types]
    M = np.array([[1.0 if (grp[i] == grp[j] and i != j) else 0.0 for j in range(len(types))]
                  for i in range(len(types))])
    rs = M.sum(1, keepdims=True)
    M = np.where(rs > 0, M / np.where(rs == 0, 1, rs), (1 - np.eye(len(types))) / max(len(types) - 1, 1))
    return pd.DataFrame(M, index=types, columns=types)


# --------------------------------------------------------------------------- spatial weights
def heterogeneity(t: Tissue, k: int = 10, labels=None) -> np.ndarray:
    """Fraction of a cell's k nearest neighbors with a different label."""
    lab = t.labels if labels is None else labels
    idx, _ = t.graph.knn(k)
    return (lab[idx] != lab[:, None]).mean(1)


def spatial_weights(t: Tissue, mode: str = "random", eligible=None, k: int = 10, gamma: float = 2.0,
                    condition_mult: dict | None = None, type_mult: dict | None = None,
                    cell_mult=None) -> np.ndarray:
    """Per-cell weights with mean 1 over eligible cells.

    random      all 1
    boundary    (heterogeneity + 0.05) ** gamma: errors concentrate where neighbors differ
    interior    (1.05 - heterogeneity) ** gamma
    condition_mult multiplies weights by condition (e.g. {"DSS": 1.5, "water": 1.0}) before normalising.
    type_mult multiplies weights by reference label (e.g. native per-type flip rates), so that eps stays the mean
    error rate but is distributed across types as the labeler distributes it.
    """
    eligible = np.ones(t.n, bool) if eligible is None else eligible
    if mode == "random":
        w = np.ones(t.n)
    elif mode == "boundary":
        w = (heterogeneity(t, k) + 0.05) ** gamma
    elif mode == "interior":
        w = (1.05 - heterogeneity(t, k)) ** gamma
    else:
        raise ValueError(f"unknown spatial mode {mode}")
    if condition_mult:
        if t.condition is None:
            raise ValueError("condition_mult given but the tissue has no condition")
        w = w * np.array([condition_mult.get(c, 1.0) for c in t.condition])
    if type_mult:
        w = w * np.array([type_mult.get(x, 1.0) for x in t.labels])
    if cell_mult is not None:                  # e.g. per-condition, per-type native flip rates
        w = w * np.asarray(cell_mult, float)
    w = np.where(eligible, w, 0.0)
    return w / w[eligible].mean()


# --------------------------------------------------------------------------- label errors
def label_errors(t: Tissue, eps: float, confusion, rng, weights=None, context_bias: float = 0.0,
                 k: int = 10, types=None) -> tuple[np.ndarray, dict]:
    """Return (new labels, info). eps is the target error rate among eligible cells (types, default all).

    confusion: a DataFrame (rows true, columns called, off-diagonal, row-normalized) used for every cell, or a
    dict {condition: DataFrame} applied to the cells of each condition (condition-specific error structure).
    """
    lab = t.labels.copy()
    by_cond = isinstance(confusion, dict)
    if by_cond and t.condition is None:
        raise ValueError("a per-condition confusion needs a tissue with a condition")
    ref = next(iter(confusion.values())) if by_cond else confusion
    tl = list(ref.index)
    codes = np.array([tl.index(x) if x in tl else -1 for x in lab])
    eligible = codes >= 0
    if types is not None:
        eligible &= np.isin(lab, list(types))
    w = np.ones(t.n) if weights is None else np.asarray(weights, float)
    p = np.clip(eps * w, 0, 1) * eligible
    flip = rng.random(t.n) < p
    if context_bias:
        idx, _ = t.graph.knn(k)
        nb = neighbour_counts(np.where(codes >= 0, codes, 0), idx, len(tl)) / k
    groups = t.condition if by_cond else np.zeros(t.n, dtype=object)
    for g in pd.unique(groups[flip]):
        M = confusion[g].reindex(index=tl, columns=tl).to_numpy() if by_cond else ref.to_numpy()
        for c in np.unique(codes[flip & (groups == g)]):
            ii = np.flatnonzero(flip & (codes == c) & (groups == g))
            P = np.repeat(M[c][None, :], len(ii), 0)
            if context_bias:
                P = P * (1 + context_bias * nb[ii])
            P[:, c] = 0
            s = P.sum(1, keepdims=True)
            P = np.where(s > 0, P / np.where(s == 0, 1, s), 0)
            u = rng.random(len(ii))[:, None]
            dest = (np.cumsum(P, 1) < u).sum(1).clip(0, len(tl) - 1)
            lab[ii] = np.array(tl)[dest]
    realised = float((lab[eligible] != t.labels[eligible]).mean()) if eligible.any() else 0.0
    info = {"target_eps": float(eps), "realised_eps": realised, "n_eligible": int(eligible.sum())}
    if t.condition is not None:
        for c in pd.unique(t.condition):
            m = eligible & (t.condition == c)
            info[f"realised_eps_{c}"] = float((lab[m] != t.labels[m]).mean()) if m.any() else np.nan
    return lab, info


# --------------------------------------------------------------------------- contamination
def margin_probabilities(scores: np.ndarray, labels, types, target_flip: float):
    """Labeler posterior per cell: softmax of centroid correlations with a temperature chosen so that the mean
    probability of not being called the current label equals target_flip (the labeler's resampling flip rate).
    Returns (P, q): P (n x T) posterior; q (n,) = 1 - P[current label]."""
    tl = list(types)
    ref = np.array([tl.index(x) for x in labels])
    S = np.where(np.isnan(scores), -np.inf, scores)

    def post(tau):
        Z = S / tau
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    lo, hi = 1e-6, 10.0
    for _ in range(60):                                   # bisection on log tau: mean(q) increases with tau
        mid = np.sqrt(lo * hi)
        if (1 - post(mid)[np.arange(len(ref)), ref]).mean() < target_flip:
            lo = mid
        else:
            hi = mid
    P = post(np.sqrt(lo * hi))
    return P, 1 - P[np.arange(len(ref)), ref]


def margin_errors(labels, P: np.ndarray, q: np.ndarray, types, eps: float, rng, eligible=None):
    """Profile-aware label errors: cell i flips with probability min(1, eps * q_i / mean(q)) (errors fall on cells
    the labeler finds ambiguous), to type j with probability proportional to P[i, j], j != current label."""
    tl = np.array(list(types), dtype=object)
    lab = np.asarray(labels, dtype=object).copy()
    el = np.ones(len(lab), bool) if eligible is None else eligible
    w = np.zeros(len(lab)); w[el] = q[el] / max(q[el].mean(), 1e-12)
    # probabilities above 1 are clipped; rescale so the expected rate over eligible cells is still eps
    lo, hi = 1.0, 1e6
    if np.clip(eps * w[el], 0, 1).mean() < eps and eps > 0:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.clip(mid * eps * w[el], 0, 1).mean() < eps:
                lo = mid
            else:
                hi = mid
        w = w * hi
    pf = np.clip(eps * w, 0, 1)
    flip = rng.random(len(lab)) < pf
    ref = np.array([list(tl).index(x) for x in lab])
    for i in np.flatnonzero(flip):
        d = P[i].copy(); d[ref[i]] = 0
        if d.sum() <= 0:
            continue
        lab[i] = tl[rng.choice(len(tl), p=d / d.sum())]
    return lab, {"realised_eps": float((lab != np.asarray(labels, dtype=object)).mean()),
                 "clipped_share": float((eps * w[el] >= 1).mean())}


def patch_errors(t: Tissue, eps: float, confusion, rng, size: float = 200.0, within: float = 0.5, eligible=None):
    """Spatially clustered, coherent label errors (a mis-annotated region or field of view): square tiles of side
    `size` are chosen per section until they hold eps / within of the eligible cells; in each tile every type gets
    one wrong destination (drawn from its confusion row, per condition if confusion is a dict) and each of its
    cells takes it with probability `within`."""
    lab = np.asarray(t.labels, dtype=object).copy()
    el = np.ones(t.n, bool) if eligible is None else eligible
    phi = min(eps / within, 1.0)
    chosen = np.zeros(t.n, bool)
    for s in pd.unique(t.section):
        m = np.flatnonzero(t.section == s)
        tile = np.floor(t.xy[m] / size).astype(np.int64)
        key = tile[:, 0] * 1_000_003 + tile[:, 1]
        tiles, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv, weights=el[m].astype(float), minlength=len(tiles))
        n = 0.0
        for j in rng.permutation(len(tiles)):
            if n >= phi * el[m].sum():
                break
            cells = m[inv == j]
            C = confusion[t.condition[cells[0]]] if isinstance(confusion, dict) else confusion
            for ty in pd.unique(lab[cells]):
                ii = cells[(t.labels[cells] == ty) & el[cells]]
                if not len(ii) or ty not in C.index:
                    continue
                row = C.loc[ty].drop(ty, errors="ignore").clip(lower=0)
                if row.sum() <= 0:
                    continue
                dest = rng.choice(row.index.to_numpy(), p=(row / row.sum()).to_numpy())
                hit = ii[rng.random(len(ii)) < within]
                lab[hit] = dest
            chosen[cells] = True
            n += cnt[j]
    return lab, {"realised_eps": float((lab != t.labels).mean()), "patch_share": float(chosen.mean())}


def contaminate(t: Tissue, s: float, rng, k: int = 6, lam: float | None = None, source_mult: dict | None = None,
                receiver_mult: dict | None = None) -> tuple[Tissue, dict]:
    """Move molecules between neighboring cells; conserves total counts."""
    X = t.counts.tocsr()
    idx, dist = t.graph.knn(k)
    s_cell = np.full(t.n, float(s))
    if source_mult:
        s_cell *= np.array([source_mult.get(x, 1.0) for x in t.labels])
    s_cell = np.clip(s_cell, 0, 1)
    W = np.ones_like(dist) if lam is None else np.exp(-dist / lam)
    if receiver_mult:
        W = W * np.vectorize(lambda x: receiver_mult.get(x, 1.0))(t.labels[idx])
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    rows = np.repeat(np.arange(t.n), np.diff(X.indptr))
    leave = rng.binomial(X.data.astype(np.int64), s_cell[rows])
    stay = X.data - leave
    remaining = leave.copy()
    tail = np.cumsum(W[:, ::-1], 1)[:, ::-1]                  # sum of weights from slot m onwards
    add_r, add_c, add_v = [], [], []
    for m in range(k):
        if m < k - 1:
            q = np.where(tail[:, m] > 0, W[:, m] / np.maximum(tail[:, m], 1e-12), 0)
            x = rng.binomial(remaining, np.clip(q[rows], 0, 1))
        else:
            x = remaining
        remaining = remaining - x
        nz = x > 0
        add_r.append(idx[rows[nz], m]); add_c.append(X.indices[nz]); add_v.append(x[nz])
    Y = sp.csr_matrix((stay.astype(np.float64), X.indices.copy(), X.indptr.copy()), shape=X.shape)
    A = sp.csr_matrix((np.concatenate(add_v).astype(np.float64),
                       (np.concatenate(add_r), np.concatenate(add_c))), shape=X.shape)
    Y = (Y + A).tocsr()
    Y.eliminate_zeros()
    moved = float(leave.sum() / max(X.data.sum(), 1))
    return t.replace(counts=Y), {"s": float(s), "moved_fraction": moved, "k": k, "lam": lam}


def merge(labels: np.ndarray, groups: dict) -> np.ndarray:
    """Collapse labels: groups maps new label -> list of old labels."""
    lut = {old: new for new, olds in groups.items() for old in olds}
    return np.array([lut.get(x, x) for x in labels])
