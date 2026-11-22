"""Where real label errors land: a per-cell model of whole-cell relabeling, and placement of label errors by it.

Background (GSE311609 compartment test). On Xenium sections in which every transcript is placed inside or
outside the nucleus, labels called from whole-cell counts differ from labels called from nucleus counts of the same
depth in a spatially organized way: small cells take the type of the larger neighbors that touch them. The cells
that change are not simply the ambiguous ones (that is what counting noise changes) nor every cell with a contact
of the destination type: they are cells with a large share of their counts outside the nucleus, touching cells of
the destination type, called with moderate confidence. A logistic model of these features, fitted on three donors,
placed held-out errors so that kNN colocalization changed in the direction of the real change for 15 of 18 pairs
(random placement: 3).

This module provides
  cell_features      per-cell features from a Tissue and optional nucleus counts and areas
  Neighborhood      contact (ranks 1-6), far (7-20) and 20-NN counts of each type around every cell
  FlipModel          coefficients per standard deviation of each feature; load / save JSON; the packaged default
  fit                fit a FlipModel from a reference labeling and a test labeling of the same cells
  draw_counts        how many a -> b errors to make at rate eps (confusion matrix), shared between placements
  place              put a given number of a -> b errors on cells: at random, or with odds exp(z . beta)

Features are standardised within the dataset they are applied to (z-scores), so coefficients transfer between
labelers whose margins are on different scales. Features missing from a dataset are dropped (and reported); the
packaged coefficients were fitted jointly, so a model without the nucleus-based features is an approximation:
refit it where a nucleus split exists (examples/compartment/fit_flipmodel.py).
"""
from __future__ import annotations

import dataclasses
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .tissue import Tissue

NB_FEATURES = ("b_contact", "b_far", "b_share20", "a_contact", "second_is_b")
CELL_FEATURES = ("log_nucleus_area", "log_cell_area", "nucleus_cell_ratio", "outside_nucleus_share",
                 "log_counts", "margin")
ALL_FEATURES = NB_FEATURES + CELL_FEATURES


# --------------------------------------------------------------------------- features
@dataclasses.dataclass
class Neighbourhood:
    contact: np.ndarray   # (n, T) cells of each type among ranks 1..n_contact (within r_contact if given)
    far: np.ndarray       # (n, T) cells of each type among ranks n_contact+1..k (within r_far if given)
    k20: np.ndarray       # (n, T) cells of each type among the k nearest
    types: np.ndarray
    k: int

    @classmethod
    def build(cls, t: Tissue, labels=None, types=None, k: int = 20, n_contact: int = 6,
              r_contact: float | None = None, r_far: float | None = None) -> "Neighbourhood":
        """Neighbors are typed by `labels` (default: the tissue's reference labels). Distances in the units of
        t.xy; without r_contact / r_far only ranks are used."""
        codes, types = t.codes(labels, types)
        idx, dist = t.graph.knn(k)
        nt = codes[idx]
        ok = nt >= 0
        rank = np.broadcast_to(np.arange(k), idx.shape)
        con = ok & (rank < n_contact)
        far = ok & (rank >= n_contact)
        if r_contact is not None:
            con &= dist <= r_contact
        if r_far is not None:
            far &= dist <= r_far
        rows = np.broadcast_to(np.arange(t.n)[:, None], idx.shape)
        T = len(types)

        def tally(m):
            out = np.zeros((t.n, T), np.int16)
            np.add.at(out, (rows[m], nt[m]), 1)
            return out
        return cls(contact=tally(con), far=tally(far), k20=tally(ok), types=np.asarray(types), k=k)


def margin_second(scores: np.ndarray, ref_codes: np.ndarray):
    """From labeler scores (n x T, higher = more like that type): the margin of the reference label over the
    best other label, and the best other label (code)."""
    S = np.where(np.isnan(scores), -np.inf, scores).copy()
    own = S[np.arange(len(S)), ref_codes].copy()
    S[np.arange(len(S)), ref_codes] = -np.inf
    second = S.argmax(1)
    other = S[np.arange(len(S)), second]
    margin = np.where(np.isfinite(own) & np.isfinite(other), own - other, np.nan)
    return margin, second


def cell_features(t: Tissue, margin=None, nucleus_counts=None, cell_area=None, nucleus_area=None) -> dict:
    """Per-cell features (raw scale). nucleus_counts: total counts over the nucleus per cell (or a cells x genes
    matrix); cell_area, nucleus_area: per cell, any consistent units."""
    f = {}
    tot = np.asarray(t.counts.sum(1)).ravel()
    f["log_counts"] = np.log1p(tot)
    if margin is not None:
        f["margin"] = np.asarray(margin, float)
    if nucleus_counts is not None:
        nc = nucleus_counts
        nc = np.asarray(nc.sum(1)).ravel() if getattr(nc, "ndim", 1) == 2 else np.asarray(nc, float)
        f["outside_nucleus_share"] = np.clip(1 - nc / np.maximum(tot, 1), 0, 1)
    if cell_area is not None and nucleus_area is not None:
        ca, na = np.asarray(cell_area, float), np.asarray(nucleus_area, float)
        f["log_cell_area"] = np.log(np.maximum(ca, 1e-6))
        f["log_nucleus_area"] = np.log(np.maximum(na, 1e-6))
        f["nucleus_cell_ratio"] = na / np.maximum(ca, 1e-6)
    return f


class Design:
    """Standardised features for (cell, a, b) rows. Cell features are z-scored over cells; neighborhood counts
    over all (cell, type) entries."""

    def __init__(self, nb: Neighbourhood, cell: dict, second: np.ndarray | None, features):
        self.nb, self.second = nb, second
        self.features = [f for f in features if (f in cell) or (f in NB_FEATURES and (f != "second_is_b" or
                                                                                      second is not None))]
        self.cell = {}
        for k, v in cell.items():
            v = np.asarray(v, float)
            fin = np.isfinite(v)
            mu, sd = (v[fin].mean(), v[fin].std()) if fin.any() else (0.0, 1.0)
            self.cell[k] = np.where(fin, (v - mu) / (sd if sd > 0 else 1.0), 0.0)
        frac20 = nb.k20 / float(nb.k)
        self._std = {"b_contact": (nb.contact.mean(), nb.contact.std() or 1.0),
                     "a_contact": (nb.contact.mean(), nb.contact.std() or 1.0),
                     "b_far": ((nb.far > 0).mean(), (nb.far > 0).std() or 1.0),
                     "b_share20": (frac20.mean(), frac20.std() or 1.0)}
        q = 1.0 / max(len(nb.types) - 1, 1)                 # each cell has one second-best label among T - 1
        self._std["second_is_b"] = (q, np.sqrt(q * (1 - q)) or 1.0)

    def rows(self, idx, a: int, b: int) -> np.ndarray:
        cols = []
        for f in self.features:
            if f == "b_contact":
                v = self.nb.contact[idx, b]
            elif f == "a_contact":
                v = self.nb.contact[idx, a]
            elif f == "b_far":
                v = (self.nb.far[idx, b] > 0).astype(float)
            elif f == "b_share20":
                v = self.nb.k20[idx, b] / float(self.nb.k)
            elif f == "second_is_b":
                v = (self.second[idx] == b).astype(float)
            else:
                cols.append(self.cell[f][idx])
                continue
            mu, sd = self._std[f]
            cols.append((v - mu) / sd)
        return np.column_stack(cols) if cols else np.zeros((len(idx), 0))


# --------------------------------------------------------------------------- model
@dataclasses.dataclass
class FlipModel:
    beta: dict                    # feature -> coefficient per standard deviation
    note: str = ""
    provenance: dict = dataclasses.field(default_factory=dict)

    def coef(self, features) -> np.ndarray:
        return np.array([self.beta[f] for f in features], float)

    def to_json(self, path):
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=1))

    @classmethod
    def from_json(cls, path) -> "FlipModel":
        d = json.loads(Path(path).read_text())
        return cls(beta=d["beta"], note=d.get("note", ""), provenance=d.get("provenance", {}))

    @classmethod
    def default(cls, variant: str = "default") -> "FlipModel":
        """A packaged model: "default" (all features) or "no_nucleus" (fitted without the nucleus-based features,
        for data without a nucleus split; present once fitted with examples/compartment/fit_flipmodel.py)."""
        f = resources.files("claimbreak").joinpath(f"data/flipmodel_{variant}.json")
        if not f.is_file():
            raise FileNotFoundError(f"no packaged flip model '{variant}'")
        d = json.loads(f.read_text())
        return cls(beta=d["beta"], note=d.get("note", ""), provenance=d.get("provenance", {}))

    @staticmethod
    def has_variant(variant: str) -> bool:
        return resources.files("claimbreak").joinpath(f"data/flipmodel_{variant}.json").is_file()


def fit(t: Tissue, test_labels, design: Design, ref_labels=None, null_labels=None, neg_ratio: int = 10,
        rng=None, min_pos: int = 30) -> tuple[FlipModel, dict]:
    """Fit a FlipModel: among cells labeled a by the reference labeling, which take label b in the test
    labeling? Rows: every (cell, b) with b a destination whose count exceeds the null's (if null_labels are
    given) or is positive; all positives and up to neg_ratio sampled negatives per positive, weighted back.
    Pair intercepts are fitted and discarded (they set how many errors, not where). Returns (model, info) with the
    leave-one-unit-out AUC when the tissue has more than one unit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = rng or np.random.default_rng(0)
    types = design.nb.types
    ref = t.codes(ref_labels, types)[0]
    tst = t.codes(test_labels, types)[0]
    T = len(types)
    cw = np.bincount(ref[(ref >= 0) & (tst >= 0)] * T + tst[(ref >= 0) & (tst >= 0)], minlength=T * T).reshape(T, T)
    if null_labels is not None:
        nul = t.codes(null_labels, types)[0]
        ok = (ref >= 0) & (nul >= 0)
        cw = cw - np.bincount(ref[ok] * T + nul[ok], minlength=T * T).reshape(T, T)
    np.fill_diagonal(cw, 0)
    X, y, w, grp, unit = [], [], [], [], []
    for a, b in zip(*np.nonzero(cw > 0)):
        cand = np.flatnonzero(ref == a)
        pos = cand[tst[cand] == b]
        if len(pos) < min_pos:
            continue
        neg = cand[tst[cand] != b]
        m = min(len(neg), neg_ratio * len(pos))
        negs = rng.choice(neg, m, replace=False) if m < len(neg) else neg
        idx = np.concatenate([pos, negs])
        X.append(design.rows(idx, a, b))
        y.append(np.r_[np.ones(len(pos)), np.zeros(len(negs))])
        w.append(np.r_[np.ones(len(pos)), np.full(len(negs), len(neg) / max(len(negs), 1))])
        grp.append(np.full(len(idx), a * T + b))
        unit.append(t.unit[idx])
    if not X:
        raise ValueError("no pair with enough relabelled cells to fit a flip model")
    X, y, w, grp, unit = map(np.concatenate, (X, y, w, grp, unit))
    P = pd.get_dummies(pd.Series(grp), dtype=float).to_numpy()
    D = np.hstack([X, P])
    m = LogisticRegression(max_iter=3000).fit(D, y, sample_weight=w)
    beta = dict(zip(design.features, map(float, m.coef_[0][: len(design.features)])))
    info = {"rows": int(len(y)), "positives": int(y.sum()), "pairs": int(len(np.unique(grp))),
            "features": list(design.features)}
    units = np.unique(unit)
    if len(units) > 1:
        pred = np.full(len(y), np.nan)
        for u in units:
            te = unit == u
            if y[~te].min() == y[~te].max():
                continue
            pred[te] = LogisticRegression(max_iter=3000).fit(D[~te], y[~te], sample_weight=w[~te]).predict_proba(
                D[te])[:, 1]
        ok = np.isfinite(pred)
        info["auc_leave_one_unit_out"] = float(roc_auc_score(y[ok], pred[ok], sample_weight=w[ok]))
    return FlipModel(beta=beta, note="fitted by claimbreak.flipmodel.fit", provenance=info), info


# --------------------------------------------------------------------------- errors
def draw_counts(labels, eps: float, confusion: pd.DataFrame, rng, eligible=None) -> np.ndarray:
    """Number of a -> b errors (T x T, rows = current label) at rate eps: Binomial(n_a, eps) errors per type,
    split over destinations by the type's (off-diagonal, row-normalized) confusion row."""
    types = list(confusion.index)
    codes = np.array([types.index(x) if x in types else -1 for x in np.asarray(labels)])
    el = codes >= 0 if eligible is None else (codes >= 0) & np.asarray(eligible, bool)
    T = len(types)
    M = confusion.to_numpy(float).copy()
    np.fill_diagonal(M, 0)
    rs = M.sum(1, keepdims=True)
    M = np.where(rs > 0, M / np.where(rs == 0, 1, rs), 0)
    n = np.zeros((T, T), int)
    for a in range(T):
        na = int((el & (codes == a)).sum())
        if na == 0 or M[a].sum() == 0:
            continue
        n[a] = rng.multinomial(rng.binomial(na, eps), M[a])
    return n


def place(labels, n: np.ndarray, types, rng, design: Design | None = None, model: FlipModel | None = None,
          eligible=None):
    """Apply n[a, b] errors to a copy of labels. Without a model, cells are chosen uniformly among the eligible
    cells of type a; with a model, with probability proportional to exp(z . beta) (Gumbel-top-k sampling without
    replacement). Larger pairs are placed first and a cell changes at most once.
    Returns (labels, info): realised rate and, with a design, the share of changed cells touching their new type."""
    types = list(types)
    lab = np.asarray(labels, dtype=object).copy()
    codes = np.array([types.index(x) if x in types else -1 for x in lab])
    el = codes >= 0 if eligible is None else (codes >= 0) & np.asarray(eligible, bool)
    used = np.zeros(len(lab), bool)
    beta = model.coef(design.features) if (model is not None and design is not None) else None
    short, touched, changed = 0, 0, 0
    order = sorted(((n[a, b], a, b) for a in range(n.shape[0]) for b in range(n.shape[1]) if n[a, b] > 0),
                   reverse=True)
    for m, a, b in order:
        cand = np.flatnonzero(el & (codes == a) & ~used)
        k = min(m, len(cand))
        short += m - k
        if k == 0:
            continue
        if beta is None:
            take = rng.choice(cand, k, replace=False)
        else:
            keys = design.rows(cand, a, b) @ beta + rng.gumbel(size=len(cand))
            take = cand[np.argpartition(-keys, k - 1)[:k]] if k < len(cand) else cand
        lab[take] = types[b]
        used[take] = True
        changed += k
        if design is not None:
            touched += int((design.nb.contact[take, b] > 0).sum())
    info = {"realised_eps": float(used[el].mean()) if el.any() else 0.0, "shortfall": int(short)}
    if design is not None and changed:
        info["contact_share"] = touched / changed
    return lab.astype(str), info
