"""Labeling helpers.

NearestCentroid: leave-one-section-out classifier (centroids from reference labels of the other sections, on the
counts they are given), assigning the label with the highest correlation. Used for the "re-called" error arm and
to estimate the native confusion structure. Any function f(counts_csr, section_array) -> labels can be used instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .tissue import Tissue, lognorm, thin


def _zscore_rows(Z):
    Z = Z - Z.mean(1, keepdims=True)
    s = np.sqrt((Z ** 2).sum(1, keepdims=True))
    s[s == 0] = 1
    return Z / s


class NearestCentroid:
    """Leave-one-section-out nearest-centroid labeler (correlation on log-normalized counts)."""

    def __init__(self, reference: Tissue, genes=None, scale: float | None = None):
        self.types = reference.types
        self.genes = genes
        tot = np.asarray(reference.counts.sum(1)).ravel()
        self.scale = float(np.median(tot)) if scale is None else scale
        codes, _ = reference.codes()
        self.sections = np.array(sorted(pd.unique(reference.section)))
        T = len(self.types)
        G = reference.counts.shape[1] if genes is None else len(genes)
        # per-section sums and counts (computed section by section to bound memory)
        self._sum = np.zeros((len(self.sections), T, G))
        self._n = np.zeros((len(self.sections), T))
        for si, s in enumerate(self.sections):
            m = np.flatnonzero(reference.section == s)
            Z = lognorm(reference.counts[m], self.scale, genes)
            np.add.at(self._sum[si], codes[m], Z)
            self._n[si] = np.bincount(codes[m], minlength=T)

    def centroids(self, exclude=None) -> np.ndarray:
        keep = self.sections != exclude if exclude is not None else np.ones(len(self.sections), bool)
        if keep.sum() == 0:
            keep[:] = True
        S, N = self._sum[keep].sum(0), self._n[keep].sum(0)
        C = S / np.maximum(N, 1)[:, None]
        C[N == 0] = np.nan
        return C

    def predict(self, counts: sp.csr_matrix, section: np.ndarray) -> np.ndarray:
        out = np.empty(counts.shape[0], dtype=object)
        for s in pd.unique(section):
            m = np.flatnonzero(section == s)
            C = self.centroids(exclude=s if s in self.sections else None)
            ok = ~np.isnan(C).any(1)
            Cz = _zscore_rows(C[ok])
            for c0 in range(0, len(m), 50000):                      # chunks bound memory
                mm = m[c0:c0 + 50000]
                Z = _zscore_rows(lognorm(counts[mm], self.scale, self.genes))
                out[mm] = self.types[ok][np.argmax(Z @ Cz.T, 1)]
        return out.astype(str)


def centroid_scores(lab: "NearestCentroid", counts, section) -> np.ndarray:
    """Correlation of each cell with every type centroid (n x T, columns in lab.types; NaN where a type has no
    centroid), leave-one-section-out as in predict."""
    out = np.full((counts.shape[0], len(lab.types)), np.nan)
    for s in pd.unique(section):
        m = np.flatnonzero(section == s)
        C = lab.centroids(exclude=s if s in lab.sections else None)
        ok = ~np.isnan(C).any(1)
        Cz = _zscore_rows(C[ok])
        for c0 in range(0, len(m), 50000):
            mm = m[c0:c0 + 50000]
            Z = _zscore_rows(lognorm(counts[mm], lab.scale, lab.genes))
            out[np.ix_(mm, np.flatnonzero(ok))] = Z @ Cz.T            # unit-norm rows: Pearson correlation
    return out


def recall_labels(t: Tissue, p: float, rng, labeller=None, reference: Tissue | None = None) -> np.ndarray:
    """Re-call labels on counts thinned to fraction p (noise-driven, native-structured errors)."""
    lab = labeller or NearestCentroid(reference or t)
    tt = thin(t, p, rng) if p < 1 else t
    if hasattr(lab, "predict"):
        return lab.predict(tt.counts, tt.section)
    return np.asarray(lab(tt.counts, tt.section)).astype(str)


def confusion(truth, called, types) -> pd.DataFrame:
    """Row-normalized confusion counts (rows true, columns called)."""
    C = pd.crosstab(pd.Categorical(truth, types), pd.Categorical(called, types), dropna=False)
    return C.reindex(index=types, columns=types, fill_value=0)


def native_confusion(t: Tissue, p: float, rng, labeller=None, mask=None, types=None,
                     fallback: pd.DataFrame | None = None, min_errors: int = 20) -> tuple[pd.DataFrame, float]:
    """Confusion of labels re-called at depth p against the reference labels.

    Returns (off-diagonal row-normalized confusion matrix, disagreement rate). The disagreement rate mixes
    noise-driven error with systematic labeler differences.
    mask: restrict to these cells (e.g. one condition); types: fixed label list (default: the tissue's);
    rows with fewer than min_errors errors take the corresponding row of `fallback` (e.g. the pooled matrix),
    or a uniform row if no fallback is given.
    """
    types = t.types if types is None else np.asarray(types)
    lab = labeller or NearestCentroid(t)
    tt = thin(t, p, rng) if p < 1 else t
    m = np.ones(t.n, bool) if mask is None else np.asarray(mask, bool)
    called = lab.predict(tt.counts[m], tt.section[m])
    C = confusion(t.labels[m], called, types).astype(float)
    rate = 1 - np.trace(C.to_numpy()) / max(C.to_numpy().sum(), 1)
    off = C.to_numpy().copy()
    np.fill_diagonal(off, 0)
    rs = off.sum(1, keepdims=True)
    uniform = (1 - np.eye(len(off))) / max(len(off) - 1, 1)
    fb = uniform if fallback is None else fallback.reindex(index=types, columns=types).fillna(0).to_numpy()
    norm = off / np.where(rs == 0, 1, rs)
    off = np.where(rs >= min_errors, norm, fb)          # too few errors to estimate: fall back
    return pd.DataFrame(off, index=types, columns=types), float(rate)


def native_flip(t: Tissue, rng, labeller=None, n_resample: int = 3, by_type: bool = False, mask=None):
    """Noise-only instability of labels: fraction of cells whose label changes when their counts are resampled
    (Poisson, mean = observed counts), averaged over resamples. Unlike the disagreement with the reference labels,
    it excludes systematic labeler differences; it is the denominator of the fragility index.
    by_type: also return {reference label: flip rate of its cells}. mask: restrict to these cells."""
    lab = labeller or NearestCentroid(t)
    m = np.ones(t.n, bool) if mask is None else np.asarray(mask, bool)
    sec, ref = t.section[m], t.labels[m]
    X = t.counts[m].tocsr()
    base = lab.predict(X, sec)
    changed = np.zeros(X.shape[0])
    for _ in range(n_resample):
        Y = X.copy()
        Y.data = rng.poisson(X.data).astype(np.float64)
        Y.eliminate_zeros()
        changed += lab.predict(Y, sec) != base
    changed /= n_resample
    rate = float(changed.mean())
    if not by_type:
        return rate
    per = pd.Series(changed).groupby(ref).mean()
    return rate, {str(k): float(v) for k, v in per.items()}
