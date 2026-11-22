"""Correctors for niche-DE claims under label error (the "corrector arm").

Each corrector sees what the analysis sees (the tissue and the possibly erroneous labels) plus whatever its method
assumes is available, and returns claims with the same identifiers as the question it corrects, so they are scored
against the planting exactly like the uncorrected claims.

  twode   2DE (Levy et al., MLGenX 2025), adapted to proximity claims: for a call A|B|g, G1 = A cells with at least
          one B among their k nearest cells, G2 = A cells with none, N1 = cells of other types among the
          nearest neighbors of G1 cells. Genes higher in G1 than G2 are labeled local markers if significantly
          higher in G1 than in N1, else neighborhood genes; a Gaussian-process classifier on
          (LFC G1 vs G2, LFC N1 vs G2) gives p_g; calls with p_g < tau are removed. Down-calls: the same with G2,
          G1 and N2. Needs: labels only.
  simex   misclassification SIMEX (Kuechenhoff et al. 2006): relabel the observed labels with Pi^lambda
          (per condition), refit the per-unit slopes at several lambda, extrapolate each unit's slope
          quadratically to lambda = -1, then test across units. pi: "oracle" = cross-tabulation of reference
          against observed labels in this run (an upper bound: needs the truth); "native" = the labeler's
          resampling flips (flip rate per type and confusion, per condition): what a user can estimate.
  ppi     prediction-powered (difference) estimator from revealed patches: square patches covering `fraction` of
          each section get their reference labels revealed; per unit,
          slope_PP = slope(observed labels, all cells) - [slope(observed, patch) - slope(reference, patch)],
          with patch slopes on index cells whose k nearest cells all lie in revealed patches; then the usual test
          across units. For OLS slopes this is the PPI rectifier in difference form (approximate for a ratio
          estimator). Needs: true labels in the patches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg, stats

from .tissue import Tissue, lognorm
from .questions import bh


def make(spec: dict):
    spec = dict(spec)
    name = spec.pop("name")
    label = spec.pop("label", None)
    cls = {"twode": TwoDE, "simex": Simex, "ppi": PPI}.get(name)
    if cls is None:
        raise ValueError(f"unknown corrector {name}; available: twode, simex, ppi")
    c = cls(**spec)
    c.label = label or c.default_label()
    return c


def _with_filter(claims: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    """Remove calls (decision 0 and p = 1, so fixed-threshold scoring respects the filter)."""
    out = claims.copy()
    drop = (out.decision.to_numpy() != 0) & ~keep
    out.loc[drop, "decision"] = 0
    out.loc[drop, "p"] = 1.0
    out["filtered"] = drop
    return out


# --------------------------------------------------------------------------- 2DE
class TwoDE:
    def __init__(self, tau=0.9, k_exposure=None, k_neighbours=6, alpha=0.05, restarts=4, seed=0):
        self.tau, self.k_exposure, self.k_nb, self.alpha = tau, k_exposure, k_neighbours, alpha
        self.restarts, self.seed = restarts, seed

    def default_label(self):
        return "twode"

    @staticmethod
    def _welch(Z1, Z2):
        """Welch t-test on log1p expression; LFC as Scanpy's rank_genes_groups reports it (log2 of the ratio of
        back-transformed means, pseudo-count 1e-9), which is the scale 2DE's kernel bounds were set for."""
        m1, m2 = Z1.mean(0), Z2.mean(0)
        v1, v2 = Z1.var(0, ddof=1) / len(Z1), Z2.var(0, ddof=1) / len(Z2)
        se = np.sqrt(np.maximum(v1 + v2, 1e-12))
        tt = (m1 - m2) / se
        df = (v1 + v2) ** 2 / np.maximum(v1 ** 2 / max(len(Z1) - 1, 1) + v2 ** 2 / max(len(Z2) - 1, 1), 1e-24)
        lfc = np.log2((np.expm1(m1) + 1e-9) / (np.expm1(m2) + 1e-9))
        return lfc, 2 * stats.t.sf(np.abs(tt), np.maximum(df, 1))

    def _pg(self, l12, ln2, local):
        """Local-marker probability per gene from (LFC G vs other group, LFC N vs other group)."""
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
        train = l12 > 0
        y = local[train].astype(int)
        X = np.c_[l12, ln2]
        if train.sum() < 4 or y.min() == y.max():
            return local.astype(float), "rule"          # one class only: the defining rule itself
        kern = ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=10.0, alpha=0.1,
                                                                    length_scale_bounds=(10, 100),
                                                                    alpha_bounds=(1e-3, 1))
        gp = GaussianProcessClassifier(kernel=kern, n_restarts_optimizer=self.restarts, random_state=self.seed)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # bound-hitting hyperparameters are expected here
            gp.fit(X[train], y)
        return gp.predict_proba(X)[:, 1], "gp"

    def apply(self, q, t: Tissue, labels, claims, ctx, rng):
        c = claims[claims.decision != 0]
        if c.empty:
            return _with_filter(claims, np.ones(len(claims), bool))
        k_exp = self.k_exposure or q.k
        idx_e, _ = t.graph.knn(k_exp)
        idx_n, _ = t.graph.knn(self.k_nb)
        Z = lognorm(t.counts, q.scale, q.gidx, getattr(q, 'size_idx', None)).astype(np.float32)
        gpos = {g: i for i, g in enumerate(q.gnames)}
        parts = c.claim.str.split("|", expand=True)
        keep_claim = {}
        for (A, B), grp in c.groupby([parts[0], parts[1]]):
            isA = labels == A
            exposed = (labels[idx_e] == B).any(1)
            G1, G2 = np.flatnonzero(isA & exposed), np.flatnonzero(isA & ~exposed)
            if len(G1) < 10 or len(G2) < 10:
                continue                                       # cannot assess: keep the calls
            for direction, sub in grp.groupby(np.sign(grp.decision.to_numpy())):
                Ga, Gb = (G1, G2) if direction > 0 else (G2, G1)
                nb = np.unique(idx_n[Ga].ravel())
                Na = nb[labels[nb] != A]
                if len(Na) < 10:
                    continue
                lab_ab, _ = self._welch(Z[Ga], Z[Gb])
                lnb, _ = self._welch(Z[Na], Z[Gb])
                lan, pan = self._welch(Z[Ga], Z[Na])
                qan = bh(pan)
                local = (lan > 0) & (qan < self.alpha)
                pg, how = self._pg(lab_ab, lnb, local)
                for cl, g in zip(sub.claim, sub.claim.str.split("|").str[2]):
                    keep_claim[cl] = bool(pg[gpos[g]] >= self.tau)
        keep = np.array([keep_claim.get(cl, True) for cl in claims.claim])
        return _with_filter(claims, keep)


# --------------------------------------------------------------------------- MC-SIMEX
def _frac_power(P: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Pi^lam as a stochastic matrix; returns the matrix and the probability mass that had to be clipped."""
    if lam == 0:
        return np.eye(len(P)), 0.0
    M = np.real(linalg.fractional_matrix_power(P, lam)) if lam != round(lam) else np.linalg.matrix_power(P, int(lam))
    neg = float(np.clip(-M, 0, None).sum(1).max())
    M = np.clip(M, 0, None)
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
    return M, neg


def _relabel(labels, groups, Pis: dict, types, lam, rng):
    out = labels.copy()
    tl = list(types)
    code = np.array([tl.index(x) if x in tl else -1 for x in labels])
    clipped = 0.0
    for g, P in Pis.items():
        M, neg = _frac_power(P, lam)
        clipped = max(clipped, neg)
        for c in range(len(tl)):
            ii = np.flatnonzero((groups == g) & (code == c))
            if not len(ii) or M[c, c] >= 1:
                continue
            u = rng.random(len(ii))[:, None]
            dest = (np.cumsum(M[c][None, :], 1) < u).sum(1).clip(0, len(tl) - 1)
            out[ii] = np.array(tl, dtype=object)[dest]
    return out, clipped


class Simex:
    def __init__(self, pi="oracle", lambdas=(0.5, 1.0, 1.5, 2.0), B=2, extrapolant="quadratic", rate="flip",
                 rate_scale=1.0, then=None, mechanism="matrix"):
        if mechanism not in ("matrix", "margin"):
            raise ValueError("simex mechanism must be 'matrix' (Pi^lambda) or 'margin' (labeller-posterior flips)")
        self.mechanism = mechanism
        if pi not in ("oracle", "native"):
            raise ValueError("simex pi must be 'oracle' or 'native'")
        if rate not in ("flip", "disagreement"):
            raise ValueError("simex rate must be 'flip' or 'disagreement'")
        self.pi, self.lambdas, self.B, self.extrapolant = pi, tuple(lambdas), int(B), extrapolant
        self.rate, self.rate_scale = rate, float(rate_scale)
        self.then = make(then) if then else None            # a filter applied to the SIMEX claims (e.g. twode)

    def default_label(self):
        lab = f"simex_{self.pi}"
        if self.pi == "native" and self.rate == "disagreement":
            lab += "_dis"
        if self.pi == "native" and self.rate_scale != 1.0:
            lab += f"_x{self.rate_scale:g}"
        if self.mechanism == "margin":
            lab += "_margin"
        if self.then is not None:
            lab += f"+{self.then.label}"
        return lab

    def _rate(self, t, labels, ctx):
        """Error rate the margin mechanism adds per unit of lambda: oracle = realised error of the observed labels;
        native = the labeler's flip rate (or disagreement), times rate_scale."""
        if self.pi == "oracle":
            return float((np.asarray(labels, dtype=object) != np.asarray(t.labels, dtype=object)).mean())
        r = ctx.get("native_disagreement") if self.rate == "disagreement" else ctx.get("native_flip")
        return float(r or 0.0) * self.rate_scale

    def matrices(self, t, labels, ctx):
        types = list(ctx["types"])
        groups = t.condition if t.condition is not None else np.zeros(t.n, dtype=object)
        Pis = {}
        for g in pd.unique(groups):
            m = groups == g
            if self.pi == "oracle":
                ct = pd.crosstab(pd.Categorical(t.labels[m], types), pd.Categorical(labels[m], types),
                                 dropna=False).to_numpy().astype(float)
                rs = ct.sum(1, keepdims=True)
                P = np.where(rs > 0, ct / np.where(rs == 0, 1, rs), np.eye(len(types)))
            else:
                nb = (ctx.get("native_by_cond") or {}).get(g)
                C = (nb["confusion"] if nb else ctx["native_C"]).reindex(index=types, columns=types).fillna(0)
                fr = nb["flip_rate_by_type"] if nb else ctx["native_type_flip"]
                f0 = nb["flip_rate"] if nb else ctx["native_flip"]
                f = np.array([fr.get(ty, f0) for ty in types])
                if self.rate == "disagreement":           # scale flips up to the measured disagreement rate
                    dis = nb.get("disagreement") if nb else ctx.get("native_disagreement")
                    if dis is not None and f0 > 0:
                        f = f * (dis / f0)
                f = np.clip(f * self.rate_scale, 0, 0.95)
                Cm = C.to_numpy().copy(); np.fill_diagonal(Cm, 0)
                rs = Cm.sum(1, keepdims=True)
                Cm = np.where(rs > 0, Cm / np.where(rs == 0, 1, rs), 0)
                P = Cm * f[:, None]
                P[np.arange(len(types)), np.arange(len(types))] = 1 - f * (rs.ravel() > 0)
            Pis[g] = P
        return Pis, groups

    def apply(self, q, t, labels, claims, ctx, rng):
        if self.mechanism == "margin":
            return self._apply_margin(q, t, labels, claims, ctx, rng)
        Pis, groups = self.matrices(t, labels, ctx)
        if all(np.allclose(P, np.eye(len(P))) for P in Pis.values()):      # no estimated error: nothing to do
            res = claims.assign(simex_clipped=0.0)
            return self.then.apply(q, t, labels, res, ctx, rng).assign(simex_clipped=0.0) if self.then else res
        base, eff_scale = q.slopes(t, labels)
        curves = {key: {0.0: [(u, S)]} for key, (u, S, _) in base.items()}
        clipped = 0.0
        for lam in self.lambdas:
            for b in range(self.B):
                lab2, neg = _relabel(labels, groups, Pis, ctx["types"], lam, rng)
                clipped = max(clipped, neg)
                sl, _ = q.slopes(t, lab2)
                for key, (u, S, _) in sl.items():
                    if key in curves:
                        curves[key].setdefault(lam, []).append((u, S))
        lams = np.array([0.0, *self.lambdas])
        deg = 2 if self.extrapolant == "quadratic" else 1
        X = np.vander(lams, deg + 1, increasing=True)
        w = np.vander(np.array([-1.0]), deg + 1, increasing=True) @ np.linalg.pinv(X)   # weights over lambdas
        out = {}
        for key, cur in curves.items():
            if set(cur) != set(lams):
                continue
            units = set(cur[0.0][0][0])
            for lam in self.lambdas:
                for u, _ in cur[lam]:
                    units &= set(u)
            units = sorted(units, key=str)
            if not units:
                continue
            Y = []
            for lam in lams:
                mats = [S[[list(u).index(x) for x in units]] for u, S in cur[lam]]
                Y.append(np.mean(mats, 0))
            ext = np.tensordot(w.ravel(), np.stack(Y), axes=1)
            out[key] = (units, ext, None)
        res = q.claims_from(out, eff_scale, n_units_total=len(pd.unique(t.unit)))
        res["simex_clipped"] = clipped
        res["simex_curve"] = _curve_summary(curves)
        if self.then is not None:
            res = self.then.apply(q, t, labels, res, ctx, rng).assign(simex_clipped=clipped)
        return res


def _simex_margin(self, q, t, labels, claims, ctx, rng):
    """SIMEX whose simulated errors follow the labeler's mechanism: at each lambda, flip lambda x eps more cells
    with probability proportional to the posterior probability that their observed label is wrong, to labels drawn
    from that posterior (perturb.margin_errors), then extrapolate to lambda = -1 as in the matrix version."""
    from . import perturb
    eps = self._rate(t, labels, ctx)
    if eps <= 0:
        res = claims.assign(simex_clipped=0.0)
        return self.then.apply(q, t, labels, res, ctx, rng).assign(simex_clipped=0.0) if self.then else res
    P, _ = ctx["margin"]()
    tl = list(ctx["types"])
    code = np.array([tl.index(x) if x in tl else 0 for x in labels])
    qobs = 1 - P[np.arange(t.n), code]
    base, eff_scale = q.slopes(t, labels)
    curves = {key: {0.0: [(u, S)]} for key, (u, S, _) in base.items()}
    for lam in self.lambdas:
        for b in range(self.B):
            lab2, _ = perturb.margin_errors(labels, P, qobs, ctx["types"], min(lam * eps, 0.95), rng)
            sl, _ = q.slopes(t, lab2)
            for key, (u, S, _) in sl.items():
                if key in curves:
                    curves[key].setdefault(lam, []).append((u, S))
    res = self._extrapolate(q, t, curves, eff_scale)
    res["simex_clipped"] = 0.0
    res["simex_curve"] = _curve_summary(curves)
    if self.then is not None:
        res = self.then.apply(q, t, labels, res, ctx, rng).assign(simex_clipped=0.0)
    return res


def _curve_summary(curves) -> str:
    """Median over (pair, gene) of |mean slope at lambda - slope at 0|, per lambda: a flat curve means the simulated
    errors do not move the estimates (saturation) and extrapolation returns the naive value."""
    lams = sorted({l for cur in curves.values() for l in cur})
    out = []
    for lam in lams:
        if lam == 0:
            continue
        d = []
        for cur in curves.values():
            if lam not in cur or 0.0 not in cur:
                continue
            u0, S0 = cur[0.0][0]
            for u, S in cur[lam]:
                common = [x for x in u if x in u0]
                if common:
                    a = S[[list(u).index(x) for x in common]].mean(0)
                    b = S0[[list(u0).index(x) for x in common]].mean(0)
                    d.append(np.abs(a - b))
        out.append(f"{lam:g}:{np.median(np.concatenate(d)) if d else float('nan'):.3g}")
    return ";".join(out)


def _simex_extrapolate(self, q, t, curves, eff_scale):
    lams = np.array([0.0, *self.lambdas])
    deg = 2 if self.extrapolant == "quadratic" else 1
    X = np.vander(lams, deg + 1, increasing=True)
    w = np.vander(np.array([-1.0]), deg + 1, increasing=True) @ np.linalg.pinv(X)
    out = {}
    for key, cur in curves.items():
        if set(cur) != set(lams):
            continue
        units = set(cur[0.0][0][0])
        for lam in self.lambdas:
            for u, _ in cur[lam]:
                units &= set(u)
        units = sorted(units, key=str)
        if not units:
            continue
        Y = [np.mean([S[[list(u).index(x) for x in units]] for u, S in cur[lam]], 0) for lam in lams]
        out[key] = (units, np.tensordot(w.ravel(), np.stack(Y), axes=1), None)
    return q.claims_from(out, eff_scale, n_units_total=len(pd.unique(t.unit)))


Simex._apply_margin = _simex_margin
Simex._extrapolate = _simex_extrapolate


# --------------------------------------------------------------------------- PPI from revealed patches
def _ppi_additive(self, q, t, labels, mixed, rev, ctx):
    """PPI++ for the per-section slope of expression on neighbor density among cells of type A (Angelopoulos,
    Duchi & Zrnic 2023, regression form), additive in per-cell scores. Per section and gene, with sums S of the
    regression scores w * [1, x, x^2, z, x z] scaled by 1 / n_section:
        S = lam * S_pred(all cells) + S_true(validated, weight 1/pi) - lam * S_pred(validated, weight 1/pi),
    slope = (S0 S_xz - S1 S_z) / (S0 S2 - S1^2). pred: observed own labels (or the labeler's posterior, predictor
    soft) and neighbor density from observed labels; true: validator's own labels and, for patches, neighbor
    density from validated labels on cells whose neighborhood is fully validated. A cell whose labels are right
    contributes zero to S_true - S_pred. lam: tuned per gene from the validated cells (PPI++ power tuning, clipped
    to [0, 1]); 0 = validated cells only (classical); 1 = plain PPI. Unbiased for any lam; inference across units."""
    from .questions import kernel_densities
    if not hasattr(q, "sigma") or q.__class__.__name__ == "ProximityDEJoint":
        raise ValueError("the additive PPI estimator is implemented for proximity_de")
    tl = list(q.types)
    T = len(tl)
    cobs = np.array([tl.index(x) if x in tl else -1 for x in labels])
    ctru = np.array([tl.index(x) if x in tl else -1 for x in mixed])
    Xp = kernel_densities(t, cobs, T, q.k, q.sigma)
    if self.reveal == "patch":
        idx, _ = t.graph.knn(q.k)
        use = rev & rev[idx].all(1)
        Xt = kernel_densities(t, ctru, T, q.k, q.sigma)
        pi = np.full(t.n, max(float(use.mean()), 1e-9))
    else:
        use = rev
        Xt = Xp                                            # index-only: neighbor labels stay observed
        pi = self._pi if self.sampling != "uniform" else np.full(t.n, self.fraction)
    w_rev = np.where(use, 1.0 / np.maximum(pi, 1e-12), 0.0)
    Psoft = None
    if self.predictor == "soft":
        P, _ = ctx["margin"]()
        ct = list(ctx["types"])
        Psoft = np.zeros((t.n, T))
        for j, ty in enumerate(tl):
            if ty in ct:
                Psoft[:, j] = P[:, ct.index(ty)]
    Z = lognorm(t.counts, q.scale, q.gidx, getattr(q, 'size_idx', None)).astype(np.float64)
    G = Z.shape[1]
    itypes = q.index_types or tl
    ntypes = q.neighbour_types or tl
    eff_scale = float(q.k) if q.sigma is None else float(np.median(Xp.sum(1)))
    out, lam_used = {}, []
    sections = pd.unique(t.section)
    for A in itypes:
        a = tl.index(A)
        wp_all = (cobs == a).astype(float) if Psoft is None else Psoft[:, a]
        wt_all = (ctru == a).astype(float)
        per_unit = {b_: {} for b_ in ntypes}
        for s in sections:
            ms = t.section == s
            ns = ms.sum()
            cp = np.flatnonzero(ms & (wp_all > 1e-3))                    # cells in the predicted sums
            cr = np.flatnonzero(ms & use & ((wp_all > 1e-3) | (wt_all > 0)))   # validated cells that matter
            if len(cp) < q.min_cells or (wt_all[ms & use] > 0).sum() < self.min_cells:
                continue
            Zp, Zr = Z[cp], Z[cr]
            wp, wpr, wtr, vr = wp_all[cp], wp_all[cr], wt_all[cr], w_rev[cr]
            for B in ntypes:
                b = tl.index(B)
                xp, xpr, xtr = Xp[cp, b], Xp[cr, b], Xt[cr, b]

                def sums(w, x, Zm, v=None):
                    ww = w if v is None else w * v
                    return (ww.sum() / ns, (ww * x).sum() / ns, (ww * x * x).sum() / ns,
                            (ww @ Zm) / ns, ((ww * x) @ Zm) / ns)

                P_all = sums(wp, xp, Zp)
                P_rev = sums(wpr, xpr, Zr, vr)
                T_rev = sums(wtr, xtr, Zr, vr)
                den_p = P_all[0] * P_all[2] - P_all[1] ** 2
                if den_p <= 0:
                    continue
                if self.mode == "classical":
                    lam = np.zeros(G)
                elif self.lam == "tuned":
                    a0 = (P_all[0] * P_all[4] - P_all[1] * P_all[3]) / den_p      # pred slope
                    al = (P_all[3] - a0 * P_all[1]) / P_all[0]                    # pred intercept
                    mx = P_all[1] / P_all[0]
                    sxx = P_all[2] / P_all[0] - mx ** 2
                    ef = (wpr * (xpr - mx))[:, None] * (Zr - al - np.outer(xpr, a0)) / max(sxx, 1e-12)
                    et = (wtr * (xtr - mx))[:, None] * (Zr - al - np.outer(xtr, a0)) / max(sxx, 1e-12)
                    n_r = max(float(use[ms].sum()), 1.0)
                    num = (vr[:, None] * ef * et).sum(0)
                    dnm = (1 + n_r / ns) * (vr[:, None] * ef * ef).sum(0)
                    lam = np.clip(np.where(dnm > 0, num / np.maximum(dnm, 1e-300), 1.0), 0, 1)
                else:
                    lam = np.full(G, float(self.lam))
                S0 = lam * P_all[0] + T_rev[0] - lam * P_rev[0]
                S1 = lam * P_all[1] + T_rev[1] - lam * P_rev[1]
                S2 = lam * P_all[2] + T_rev[2] - lam * P_rev[2]
                Sz = lam * P_all[3] + T_rev[3] - lam * P_rev[3]
                Sxz = lam * P_all[4] + T_rev[4] - lam * P_rev[4]
                den = S0 * S2 - S1 ** 2
                ok = den > 1e-12 * max(abs(den_p), 1e-300)
                slope = np.where(ok, (S0 * Sxz - S1 * Sz) / np.where(ok, den, 1.0), np.nan)
                if np.isnan(slope).all():
                    continue
                lam_used.append(float(np.mean(lam)))
                per_unit[B].setdefault(t.unit[cp[0]], []).append(slope)
        for B in ntypes:
            if per_unit[B]:
                units = list(per_unit[B])
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        S = np.vstack([np.nanmean(per_unit[B][u], 0) for u in units])
                out[(A, B)] = (units, S, None)      # NaN = gene not estimable in that unit: dropped per gene
    res = q.claims_from(out, eff_scale, n_units_total=len(pd.unique(t.unit)))
    full, _ = q.slopes(t, labels)
    total = sum(len(u) for u, _, _ in full.values())
    res["ppi_revealed"] = float(rev.mean())
    res["ppi_interior"] = float(use.mean())
    res["ppi_units_kept"] = sum(len(u) for u, _, _ in out.values()) / max(total, 1)
    res["ppi_lambda"] = float(np.mean(lam_used)) if lam_used else np.nan
    return res


class PPI:
    """reveal "patch": reference labels revealed for all cells in square patches; the rectifier uses index cells
    whose k nearest cells are all revealed, so both the own and the neighbor role are corrected.
    reveal "index": reference labels revealed for a random `fraction` of cells, own label only (as an expert
    validating individual cells, CSDE-like); neighbor features keep the observed labels, so only the own-label
    route is corrected. Units (and pairs) without enough revealed index cells are dropped; ppi_units_kept reports
    the share of (pair, unit) estimates retained."""

    def __init__(self, fraction=0.05, patch=200.0, min_cells=10, reveal="patch", mode="ppi", validation_error=0.0,
                 sampling="uniform", validation_copy=0.0, floor=0.1, estimator="additive", lam="tuned",
                 predictor="hard"):
        if estimator not in ("additive", "ratio"):
            raise ValueError("ppi estimator must be 'additive' (PPI++ on per-cell scores) or 'ratio' (0.3-0.4 form)")
        if predictor not in ("hard", "soft"):
            raise ValueError("ppi predictor must be 'hard' (observed labels) or 'soft' (labeller posterior)")
        self.estimator, self.lam, self.predictor = estimator, lam, predictor
        if reveal not in ("patch", "index"):
            raise ValueError("ppi reveal must be 'patch' or 'index'")
        if mode not in ("ppi", "classical"):
            raise ValueError("ppi mode must be 'ppi' (rectified) or 'classical' (validated cells only)")
        if sampling not in ("uniform", "stratified", "uncertainty"):
            raise ValueError("ppi sampling must be 'uniform', 'stratified' or 'uncertainty'")
        if sampling != "uniform" and reveal != "index":
            raise ValueError("ppi sampling other than uniform needs reveal: index")
        self.fraction, self.patch, self.min_cells, self.reveal = float(fraction), float(patch), int(min_cells), reveal
        self.mode, self.validation_error = mode, float(validation_error)
        self.sampling, self.validation_copy, self.floor = sampling, float(validation_copy), float(floor)

    def default_label(self):
        pct = f"{self.fraction * 100:g}".replace(".", "p")
        lab = f"{'cls' if self.mode == 'classical' else 'ppi'}{'_idx' if self.reveal == 'index' else ''}"
        lab += {"uniform": "", "stratified": "_str", "uncertainty": "_unc"}[self.sampling] + f"_{pct}"
        if self.validation_error:
            lab += f"_ve{self.validation_error * 100:g}".replace(".", "p")
        if self.validation_copy:
            lab += f"_cp{self.validation_copy * 100:g}".replace(".", "p")
        if self.estimator == "ratio":
            lab += "_ratio"
        if self.predictor == "soft":
            lab += "_soft"
        if self.lam != "tuned" and self.estimator == "additive" and self.mode != "classical":
            lab += f"_lam{self.lam}"
        return lab

    def inclusion(self, t, labels, ctx):
        """Inclusion probability pi_i of each cell in the validated sample (index mode), mean = fraction.
        uniform: pi = fraction. stratified: equal expected count per observed label (water-filled, capped at 1).
        uncertainty: pi proportional to the labeler's posterior probability that the observed label is wrong,
        plus floor x its mean (keeps every cell eligible), capped at 1 (active inference: Zrnic & Candes 2024)."""
        n = t.n
        if self.sampling == "uniform":
            return np.full(n, self.fraction)
        if self.sampling == "stratified":
            u = np.zeros(n)
            labs, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
            u = 1.0 / cnt[inv]
        else:
            P, _ = ctx["margin"]()
            tl = list(ctx["types"])
            code = np.array([tl.index(x) if x in tl else -1 for x in labels])
            u = 1 - np.where(code >= 0, P[np.arange(n), np.clip(code, 0, None)], 0.0)
            u = u + self.floor * u.mean()
        lo, hi = 0.0, 1.0 / max(u.min(), 1e-12)
        for _ in range(80):                                   # scale so that mean(min(1, c u)) = fraction
            c = 0.5 * (lo + hi)
            if np.minimum(1, c * u).mean() < self.fraction:
                lo = c
            else:
                hi = c
        return np.minimum(1, hi * u)

    def validation_labels(self, t, ctx, rng):
        """Labels the validator assigns: the reference, or (validation_error > 0) the reference with profile-aware
        errors at that rate (the validator struggles with the cells the labeler finds ambiguous)."""
        if not self.validation_error:
            return t.labels
        from . import perturb
        P, q = ctx["margin"]()
        lab, _ = perturb.margin_errors(t.labels, P, q, ctx["types"], self.validation_error, rng)
        return lab

    def revealed(self, t, rng, labels=None, ctx=None):
        if self.reveal == "index":
            pi = self.inclusion(t, labels, ctx) if self.sampling != "uniform" else np.full(t.n, self.fraction)
            U = np.random.default_rng(ctx["sample_seed"]).random(t.n) if ctx and "sample_seed" in ctx \
                else rng.random(t.n)                          # common stream: nested samples across budgets
            self._pi = pi
            return U < pi
        rev = np.zeros(t.n, bool)
        for s in pd.unique(t.section):
            m = np.flatnonzero(t.section == s)
            tile = np.floor(t.xy[m] / self.patch).astype(np.int64)
            key = tile[:, 0] * 1_000_003 + tile[:, 1]
            tiles, inv = np.unique(key, return_inverse=True)
            order = rng.permutation(len(tiles))
            counts = np.bincount(inv, minlength=len(tiles))
            take, n = [], 0
            for j in order:
                if n >= self.fraction * len(m):
                    break
                take.append(j); n += counts[j]
            rev[m[np.isin(inv, take)]] = True
        return rev

    def apply(self, q, t, labels, claims, ctx, rng):
        rev = self.revealed(t, rng, labels, ctx)
        truth = np.asarray(self.validation_labels(t, ctx, rng), dtype=object).copy()
        if self.validation_copy:                              # validator repeats the labeler's mistake
            wrong = rev & (np.asarray(labels, dtype=object) != np.asarray(t.labels, dtype=object))
            cp = wrong & (rng.random(t.n) < self.validation_copy)
            truth[cp] = np.asarray(labels, dtype=object)[cp]
        mixed = np.where(rev, truth, labels)                  # validator's labels where revealed
        if self.estimator == "additive":
            return self._additive(q, t, labels, mixed, rev, ctx)
        wts = None
        if self.reveal == "index" and self.sampling != "uniform":
            wts = np.where(rev, 1.0 / np.maximum(self._pi, 1e-12), 0.0)
        full, eff_scale = q.slopes(t, labels)
        if self.reveal == "patch":
            idx, _ = t.graph.knn(q.k)
            use = rev & rev[idx].all(1)
            pn, _ = q.slopes(t, labels, cell_mask=use, min_cells=self.min_cells)
            pt, _ = q.slopes(t, mixed, cell_mask=use, min_cells=self.min_cells)
        else:
            use = rev
            pn, _ = q.slopes(t, labels, cell_mask=use, min_cells=self.min_cells, nb_labels=labels, weights=wts)
            pt, _ = q.slopes(t, mixed, cell_mask=use, min_cells=self.min_cells, nb_labels=labels, weights=wts)
        out, kept, total = {}, 0, 0
        if self.mode == "classical":                          # validated cells only: no use of unvalidated cells
            for key, (u, S, pooled) in full.items():
                total += len(u)
            for key, (u, S, pooled) in pt.items():
                kept += len(u)
                out[key] = (u, S, None)
            res = q.claims_from(out, eff_scale, n_units_total=len(pd.unique(t.unit)))
            res["ppi_revealed"] = float(rev.mean())
            res["ppi_interior"] = float(use.mean())
            res["ppi_units_kept"] = kept / max(total, 1)
            return res
        for key, (u, S, pooled) in full.items():
            total += len(u)
            if key not in pn or key not in pt:
                continue
            units = [x for x in u if x in pn[key][0] and x in pt[key][0]]
            if not units:
                continue
            kept += len(units)
            g = lambda d: d[key][1][[list(d[key][0]).index(x) for x in units]]
            Sf = S[[list(u).index(x) for x in units]]
            out[key] = (units, Sf - (g(pn) - g(pt)), None)
        res = q.claims_from(out, eff_scale, n_units_total=len(pd.unique(t.unit)))
        res["ppi_revealed"] = float(rev.mean())
        res["ppi_interior"] = float(use.mean())
        res["ppi_units_kept"] = kept / max(total, 1)
        return res


PPI._additive = _ppi_additive
