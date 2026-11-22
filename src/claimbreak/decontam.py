"""Neighbor-aware labeling.

The exposure analyses showed that a cell's mask is a catchment for other cells' RNA: the counts outside
the nucleus carry RNA of the touching cells (in proportion to the share of the border they hold) and, less, of cells
within 20-30 um, and the more of the mask lies outside the nucleus, the more foreign RNA it holds. This module labels
a cell with that contamination in the likelihood instead of removing counts:

  log L_i(k) = sum_g n_ig log q_kg  +  sum_g o_ig log[(1 - psi_i) q_kg + psi_i c_ig]

  n_i     the cell's counts in the nucleus (none for arms without a nucleus split)
  o_i     its counts outside the nucleus
  q_k     the profile of candidate type k (reference mean, flat floor, platform factor gamma: as the plain labeler)
  c_i     the cell's expected contamination profile, sum_b a_ib q_b, built from geometry and the neighbors' labels
          only (never from the cell's own label); a_i is the composition of the foreign RNA F_i reaching the cell:
            F_ib = w_contact * (RNA of touching type-b neighbors, weighted by the share of the border each holds,
                                border shares at a nucleus gap of `contact_gap` as in the exposure analysis)
                 + w_region  * (mean RNA per cell of type b among the cells within `region` um)
                 + w_ambient * median RNA * (section share of type b by RNA)
          RNA = each cell's whole-cell counts; types from the neighbors' full-nucleus labels.
  psi_i   the foreign share of the outside counts: psi_i = kappa E_i s_i / (1 + kappa E_i s_i), E_i = sum_b F_ib / m_i
          the foreign RNA per unit of the cell's own (m_i its whole-cell counts), s_i = (r_i / median r)^beta with r_i
          the odds of a count lying outside the nucleus, capped at psi_max. kappa and beta per section and
          arm by maximum likelihood (best candidate per cell; or the labels of another count set, fit_kappa).

psi is the same for every candidate type. Letting it float per cell and candidate (psi_mode "profile") lets a wrong
type absorb the cell's own counts as 'contamination': on synthetic tissue at low depth that cost tumor cells and T
cells inside T aggregates, which the exposure-driven psi does not (tests/test_decontam.py).

Outside counts support type k only as far as the neighbors cannot explain them; the nucleus anchors the call. A
cell surrounded by its own type is barely affected (c_i is close to q_k). The weights w set the composition of c_i and,
with kappa, its amount.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from . import geometry as geo

PSI_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass
class DecontamConfig:
    contact_gap: float = 10.0      # um nucleus gap counted as mask contact (two 5 um masks touch up to about 10 um)
    region: float = 20.0           # um radius of the regional term
    w_contact: float = 1.0
    w_region: float = 0.5
    w_ambient: float = 0.1
    psi_max: float = 0.9           # largest foreign share of the outside counts allowed
    bins: int = 72                 # angular bins for border shares
    chunk: int = 2000              # cells per block in the likelihood
    psi_mode: str = "fit"          # "fit": psi_i = kappa E_i / (1 + kappa E_i), one kappa per section and arm,
                                   # the same psi for every candidate; "profile": psi free per cell and candidate


def _by_type(n_cells, i, j, w, code, K):
    """sum over pairs (i, j) of w into (i, type of j); unknown types left out. Dense cells x K."""
    ok = code[j] >= 0
    M = sp.csr_matrix((w[ok], (i[ok], code[j][ok])), shape=(n_cells, K))
    return np.asarray(M.toarray(), float)


def contamination_weights(xy, nucleus_area, nbr_code, mass, K: int, cfg: DecontamConfig = DecontamConfig()):
    """Expected foreign RNA reaching each cell, by type of origin:

      F_ib = w_contact * sum_{touching j of type b} share_ij m_j          (border share at the contact gap)
           + w_region  * sum_{j of type b within region} m_j / N_i          (mean RNA of the cells around, by type)
           + w_ambient * median(m) * p_b                                    (section composition by RNA)

    Returns A (cells x K, F normalized per cell: the composition of the contamination), E (sum_b F_ib / m_i: foreign
    RNA per unit of the cell's own, the exposure that sets psi), the covered border share and a note."""
    xy = np.asarray(xy, float)
    n = len(xy)
    code = np.asarray(nbr_code, int)
    m = np.nan_to_num(np.asarray(mass, float), nan=0.0).clip(min=0)
    r, note = geo.nucleus_radius(nucleus_area)
    i, j, gap, d = geo.gap_pairs(xy, r, cfg.contact_gap)
    ip, jp, sh, cov = geo.border_shares(n, xy, r, i, j, gap, d, cfg.contact_gap, bins=cfg.bins)
    P_c = _by_type(n, ip, jp, sh * m[jp], code, K)
    pr = cKDTree(xy).query_pairs(cfg.region, output_type="ndarray")      # the cell itself excluded
    ri, rj = (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]]) if len(pr) else (np.array([], int),) * 2
    N = np.maximum(np.bincount(ri, minlength=n), 1)
    P_r = _by_type(n, ri, rj, m[rj], code, K) / N[:, None]
    ok = code >= 0
    p_a = np.bincount(code[ok], weights=m[ok], minlength=K).astype(float)
    p_a = p_a / p_a.sum() if p_a.sum() > 0 else np.full(K, 1.0 / K)
    med = float(np.median(m[m > 0])) if (m > 0).any() else 1.0
    F = cfg.w_contact * P_c + cfg.w_region * P_r + cfg.w_ambient * med * p_a[None, :]
    tot = F.sum(1)
    A = np.divide(F, tot[:, None], out=np.tile(p_a, (n, 1)), where=tot[:, None] > 0)
    E = tot / np.maximum(m, 1.0)
    return A, E, cov, note


def psi_of(E, kappa: float, psi_max: float):
    """Foreign share of the outside counts from the exposure: kappa E / (1 + kappa E), capped."""
    return np.minimum(kappa * E / (1 + kappa * E), psi_max)


def mixture_loglik(Xn, Xo, q, A, cfg: DecontamConfig = DecontamConfig(), psi=None, psi_grid=PSI_GRID):
    """Log-likelihood (cells x K) of every candidate type and the psi used per (cell, type). Xn: nuclear counts (csr
    or None), Xo: outside counts (csr), q: K x G profiles (rows sum to 1), A: cells x K contamination composition.
    psi: per-cell foreign share (fixed, the same for every candidate); None = profiled per cell and candidate on the
    grid (more flexible: a wrong candidate can absorb the cell's own counts as 'contamination', see psi_mode)."""
    Xo = sp.csr_matrix(Xo)
    n_cells, G = Xo.shape
    K = q.shape[0]
    grid = np.array([p for p in psi_grid if p <= cfg.psi_max + 1e-9], float)
    logq = np.log(q)
    LL = np.zeros((n_cells, K))
    if Xn is not None:
        LL += np.asarray(sp.csr_matrix(Xn) @ logq.T)
    out_psi = np.zeros((n_cells, K))
    for lo in range(0, n_cells, cfg.chunk):
        hi = min(n_cells, lo + cfg.chunk)
        B = Xo[lo:hi]
        if B.nnz == 0:
            if psi is not None:
                out_psi[lo:hi] = np.asarray(psi)[lo:hi, None]
            continue
        rows = np.repeat(np.arange(hi - lo), np.diff(B.indptr))
        g, x = B.indices, B.data.astype(np.float64)
        c = np.einsum("nk,kn->n", A[lo:hi][rows], q[:, g])               # c_i at each count entry
        qk = q[:, g].T                                                    # entries x K
        R = sp.csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(hi - lo, len(rows)))
        if psi is not None:
            p = np.asarray(psi, float)[lo:hi]
            pe = p[rows][:, None]
            LL[lo:hi] += R @ (x[:, None] * np.log((1 - pe) * qk + pe * c[:, None]))
            out_psi[lo:hi] = p[:, None]
            continue
        S = np.empty((hi - lo, K, len(grid)))
        for t, p in enumerate(grid):
            S[:, :, t] = R @ (x[:, None] * np.log((1 - p) * qk + p * c[:, None]))
        best = S.argmax(2)
        LL[lo:hi] += np.take_along_axis(S, best[:, :, None], 2)[:, :, 0]
        out_psi[lo:hi] = grid[best]
    return LL, out_psi


KAPPAS = (0.0, 0.03, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
BETAS = (0.0, 0.5, 1.0, 2.0)


def outside_index(outside_share, beta: float):
    """(r_i / median r)^beta with r = o / (1 - o), o the share of the cell's counts outside the nucleus (clipped):
    the more of the mask lies outside the nucleus, the more foreign RNA it catches (exposure runs, scheme 'outside')."""
    o = np.clip(np.nan_to_num(np.asarray(outside_share, float), nan=0.5), 0.05, 0.95)
    r = o / (1 - o)
    return (r / np.median(r)) ** beta


def fit_kappa(Xn, Xo, q, A, E, use, cfg: DecontamConfig = DecontamConfig(), kappas=KAPPAS, max_cells: int = 20000,
              seed: int = 0, ref=None, outside=None, betas=(0.0,)):
    """Catchment constant kappa (and outside-share exponent beta) per section and arm, on (a sample of) the labeled
    cells, with psi_i = kappa E_i s_i / (1 + kappa E_i s_i), s_i = outside_index(outside_i, beta).

    ref = None (default): maximise sum_i max_k log L_i(k). ref = type codes from a different count set (e.g. the other
    nuclear half): maximise sum_i log L_i(ref_i) over cells with a reference label. Checked against the
    suspicion that the first form inflates kappa; on synthetic tissue with reference mismatch both gave the same kappa
    at moderate depth, and at low depth the noisy reference labels inflated kappa instead (cells whose reference label
    is wrong are fitted better with more 'contamination'). Returns kappa, beta and the table."""
    idx = np.flatnonzero(use if ref is None else (use & (np.asarray(ref) >= 0)))
    if len(idx) > max_cells:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_cells, replace=False))
    if len(idx) == 0:
        return 0.0, 0.0, []
    Xs = None if Xn is None else sp.csr_matrix(Xn)[idx]
    Os = sp.csr_matrix(Xo)[idx]
    r = None if ref is None else np.asarray(ref)[idx]
    tab = []
    for be in (betas if outside is not None else (0.0,)):
        Ei = E[idx] * (outside_index(outside, be)[idx] if outside is not None else 1.0)
        for k in kappas:
            LL, _ = mixture_loglik(Xs, Os, q, A[idx], cfg, psi=psi_of(Ei, k, cfg.psi_max))
            obj = LL.max(1).sum() if r is None else LL[np.arange(len(idx)), r].sum()
            tab.append((float(k), float(be), float(obj)))
    kappa, beta, _ = max(tab, key=lambda t: t[2])
    return kappa, beta, tab


def decide(LL, use):
    """Best and second type, margin (nats) from a log-likelihood table; rows not in `use` get -1 / NaN."""
    n, K = LL.shape
    lab = np.full(n, -1)
    sec = np.full(n, -1)
    mar = np.full(n, np.nan)
    o = np.argsort(LL[use], 1)
    lab[use] = o[:, -1]
    sec[use] = o[:, -2] if K > 1 else o[:, -1]
    srt = np.take_along_axis(LL[use], o, 1)
    mar[use] = srt[:, -1] - srt[:, -2] if K > 1 else np.inf
    return lab, sec, mar
