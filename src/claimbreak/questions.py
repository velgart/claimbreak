"""Question types and the claims they produce.

Every question maps (tissue with analysis counts, labels) to a table of claims with columns
    claim     stable identifier (e.g. "A|B" or "A|B|gene")
    stat      the statistic behind the decision
    p         p-value (NaN where not applicable)
    decision  +1, -1 or 0 under the question's standard decision rule
Questions are "prepared" once on the baseline (reference labels, unperturbed counts), which fixes anything that must
stay constant across runs (gene set, niche centroids used to match niche identities).

Implemented
    composition      differential abundance of each label between two conditions (per-unit proportions, Welch t)
    colocalization   neighborhood enrichment z-scores for every ordered pair of labels (labels permuted within
                     sections)
    niches           cellular neighborhoods by k-means on neighbor composition; claims: label t enriched in
                     niche c (>= 2-fold), and niche abundance differences between conditions
    proximity_de     within-type expression vs number of neighbors of another type (SpatioMark / niche-DE-like),
                     per section OLS, aggregated over units, BH over all tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

from .tissue import Tissue, lognorm, neighbour_counts


def kernel_densities(t: Tissue, codes: np.ndarray, T: int, k: int, sigma) -> np.ndarray:
    """(n, T) neighborhood features: counts of each label among the k nearest cells (sigma None), or
    Gaussian-weighted counts, weights exp(-d^2 / (2 sigma^2)), truncated at k (sigma in coordinate units)."""
    idx, dist = t.graph.knn(k)
    if sigma is None:
        return neighbour_counts(codes, idx, T).astype(float)
    w = np.exp(-dist ** 2 / (2 * float(sigma) ** 2))
    n, kk = idx.shape
    flat = np.repeat(np.arange(n), kk) * T + codes[idx].ravel()
    return np.bincount(flat, weights=w.ravel(), minlength=n * T).reshape(n, T)


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return q
    pp = p[ok]
    o = np.argsort(pp)
    r = pp[o] * len(pp) / (np.arange(len(pp)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty_like(pp)
    out[o] = np.clip(r, 0, 1)
    q[ok] = out
    return q


def _two_conditions(base, given, name):
    """The two conditions a contrast compares: [reference, alternative]."""
    present = sorted(pd.unique(base.condition))
    if given is None:
        if len(present) != 2:
            raise ValueError(f"{name}: the data have conditions {present}; give conditions: [reference, "
                             "alternative] in the question's settings")
        return present
    given = [str(c) for c in given]
    if len(given) != 2 or not set(given) <= set(present):
        raise ValueError(f"{name}: conditions must be two of {present}, got {given}")
    return given


def _claims(claim, stat, p, decision, support=None, effect=None, se=None, df=None) -> pd.DataFrame:
    """Claims table. support: cells behind the claim (for separating rarity from fragility); effect: effect size
    on the question's natural scale; se, df: standard error of stat and degrees of freedom of its test (niche DE:
    the interval stat +/- t_df * se is the claim's confidence interval)."""
    n = len(claim)
    out = pd.DataFrame({"claim": claim, "stat": stat, "p": p, "decision": np.asarray(decision, int),
                        "effect": np.full(n, np.nan) if effect is None else np.asarray(effect, float),
                        "support": np.full(n, np.nan) if support is None else np.asarray(support, float)})
    if se is not None:
        out["se"] = np.asarray(se, float)
        out["df"] = np.asarray(df, float)
    return out


class Question:
    name = "question"

    def __init__(self, label=None, **kw):
        """label: name under which the claims are reported (to run one question twice with different settings,
        e.g. proximity_de with and without an effect-size rule)."""
        if label:
            self.name = label
        self.kw = kw

    def prepare(self, base: Tissue):
        pass

    def run(self, t: Tissue, labels: np.ndarray, rep: int = 0) -> pd.DataFrame:
        """rep: replicate index; questions with random elements vary their seed with it (vary_seed)."""
        raise NotImplementedError


class Composition(Question):
    """Label proportions per unit compared between two conditions."""
    name = "composition"

    def __init__(self, conditions=None, alpha=0.05, **kw):
        super().__init__(**kw)
        self.conditions, self.alpha = conditions, alpha

    def prepare(self, base):
        if base.condition is None:
            raise ValueError("composition needs a condition")
        self.types = base.types
        self.conditions = _two_conditions(base, self.conditions, self.name)

    def run(self, t, labels, rep=0):
        df = pd.DataFrame({"unit": t.unit, "cond": t.condition, "lab": labels})
        df = df[df.cond.isin(self.conditions)]
        P = pd.crosstab(df.unit, df.lab, normalize="index").reindex(columns=self.types, fill_value=0)
        cond = df.groupby("unit").cond.first().reindex(P.index)
        a, b = P[cond == self.conditions[1]], P[cond == self.conditions[0]]
        rows = []
        for ty in self.types:
            if len(a) < 2 or len(b) < 2:
                d, p = a[ty].mean() - b[ty].mean(), np.nan
            else:
                d = a[ty].mean() - b[ty].mean()
                p = stats.ttest_ind(a[ty], b[ty], equal_var=False).pvalue
            rows.append((ty, d, p))
        c = pd.DataFrame(rows, columns=["claim", "stat", "p"])
        dec = np.where(c.p < self.alpha, np.sign(c.stat), 0)
        counts = df.lab.value_counts().reindex(self.types, fill_value=0).to_numpy()
        eff = [np.log2((a[ty].mean() + 1e-4) / (b[ty].mean() + 1e-4)) for ty in self.types]
        return _claims(c.claim, c.stat, c.p, np.nan_to_num(dec), support=counts, effect=eff)


class Colocalization(Question):
    """Neighborhood enrichment: observed neighbor-label pair counts vs labels permuted within sections.

    Decision: +1 if z > z and observed/expected >= min_ratio; -1 if z < -z and observed/expected <= 1/min_ratio.
    The ratio rule matters: with 10^5 cells almost every pair passes |z| > 2. Self-pairs are excluded unless
    include_self. vary_seed: permutations use seed + rep, so replicates include permutation noise.
    """
    name = "colocalization"

    def __init__(self, k=10, n_perm=50, z=2.0, min_ratio=1.5, include_self=False, seed=0, vary_seed=True, **kw):
        super().__init__(**kw)
        self.k, self.n_perm, self.z, self.seed = k, n_perm, z, seed
        self.min_ratio, self.include_self, self.vary_seed = min_ratio, include_self, vary_seed

    def prepare(self, base):
        self.types = base.types
        self.secs = [base.graph.section_index(s) for s in pd.unique(base.section)]

    def run(self, t, labels, rep=0):
        codes, types = t.codes(labels, self.types)
        T = len(types)
        idx, _ = t.graph.knn(self.k)
        k = idx.shape[1]

        def pair_counts(c):
            flat = np.repeat(c, k) * T + c[idx].ravel()
            return np.bincount(flat, minlength=T * T).reshape(T, T).astype(float)

        obs = pair_counts(codes)
        rng = np.random.default_rng(self.seed + (rep if self.vary_seed else 0))
        sims = np.empty((self.n_perm, T, T))
        for r in range(self.n_perm):
            pc = codes.copy()
            for ii in self.secs:
                pc[ii] = codes[rng.permutation(ii)]
            sims[r] = pair_counts(pc)
        mu, sd = sims.mean(0), sims.std(0) + 1e-9
        Z = (obs - mu) / sd
        R = (obs + 1) / (mu + 1)
        n_type = np.bincount(codes, minlength=T)
        a, b = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
        a, b = a.ravel(), b.ravel()
        keep = np.ones(len(a), bool) if self.include_self else a != b
        a, b = a[keep], b[keep]
        z, r = Z[a, b], R[a, b]
        dec = np.where((z > self.z) & (r >= self.min_ratio), 1,
                       np.where((z < -self.z) & (r <= 1 / self.min_ratio), -1, 0))
        claim = [f"{types[i]}|{types[j]}" for i, j in zip(a, b)]
        return _claims(claim, z, 2 * stats.norm.sf(np.abs(z)), dec,
                       support=np.minimum(n_type[a], n_type[b]), effect=np.log2(r))


class Niches(Question):
    """Cellular neighborhoods (k-means on neighbor composition), as a consensus over several k-means seeds.

    Claims
      co|t|u             labels t and u are both enriched (>= enrich-fold, >= min_cells) in some niche
                         (independent of niche identities)
      enriched|nicheN|t  label t enriched in niche N (niches matched to reference niches by centroid assignment)
      abundance|nicheN   niche abundance differs between two conditions (per-unit fractions, Welch t)
    Each run fits k-means with n_seeds seeds (n_init per seed); a claim is made when at least `consensus` (0.7) of
    the seeds make it (stability selection). stat = share of seeds making the claim. Claims made by more than
    `unstable_low` (0.3) but fewer than `consensus` of the seeds are flagged `unstable` (no decision): the niche
    solution itself is undecided about them. The reference niches are those
    of the best-scoring baseline fit. vary_seed: the seed set shifts with the replicate, so level-0 replicates
    measure the consensus's own variability. k-means objectives can have several near-equal solutions, which is
    why a single seed is not used.
    """
    name = "niches"

    def __init__(self, k=20, n_niches=8, enrich=2.0, min_cells=20, alpha=0.05, seed=0, conditions=None,
                 n_seeds=20, n_init=1, consensus=0.7, unstable_low=0.3, vary_seed=True, **kw):
        super().__init__(**kw)
        self.k, self.n_niches, self.enrich, self.min_cells = k, n_niches, enrich, min_cells
        self.alpha, self.seed, self.conditions = alpha, seed, conditions
        self.n_seeds, self.n_init, self.consensus, self.vary_seed = n_seeds, n_init, consensus, vary_seed
        self.unstable_low = unstable_low

    def _features(self, t, labels):
        codes, types = t.codes(labels, self.types)
        idx, _ = t.graph.knn(self.k)
        return codes, neighbour_counts(codes, idx, len(types)) / self.k

    def _fits(self, F, seeds):
        from sklearn.cluster import KMeans
        for sd in seeds:
            km = KMeans(self.n_niches, n_init=self.n_init, random_state=int(sd)).fit(F)
            yield km.labels_, km.cluster_centers_, km.inertia_

    def prepare(self, base):
        self.types = base.types
        _, F = self._features(base, base.labels)
        best = min(self._fits(F, range(self.seed, self.seed + self.n_seeds)), key=lambda x: x[2])
        self.centers = best[1]
        n_cond = 0 if base.condition is None else len(pd.unique(base.condition))
        self.contrast = None
        if n_cond >= 2 and (self.conditions is not None or n_cond == 2):
            self.contrast = _two_conditions(base, self.conditions, self.name)

    def _one(self, t, codes, niche):
        T = len(self.types)
        overall = np.bincount(codes, minlength=T) / len(codes)
        rows = []
        E = np.zeros((self.n_niches, T), bool)
        for n in range(self.n_niches):
            m = niche == n
            cnt = np.bincount(codes[m], minlength=T)
            ratio = (cnt / max(m.sum(), 1)) / np.maximum(overall, 1e-12)
            E[n] = (ratio >= self.enrich) & (cnt >= self.min_cells)
            for ti, ty in enumerate(self.types):
                rows.append((f"enriched|niche{n}|{ty}", int(E[n, ti]), np.log2(ratio[ti] + 1e-9), cnt[ti], np.nan))
        n_type = np.bincount(codes, minlength=T)
        for i in range(T):
            for j in range(i + 1, T):
                rows.append((f"co|{self.types[i]}|{self.types[j]}", int((E[:, i] & E[:, j]).any()), np.nan,
                             min(n_type[i], n_type[j]), np.nan))
        if self.contrast is not None:
            cs = self.contrast
            df = pd.DataFrame({"unit": t.unit, "cond": t.condition, "niche": niche})
            P = pd.crosstab(df.unit, df.niche, normalize="index").reindex(columns=range(self.n_niches),
                                                                            fill_value=0)
            cond = df.groupby("unit").cond.first().reindex(P.index)
            a, b = P[cond == cs[1]], P[cond == cs[0]]
            for n in range(self.n_niches):
                d = a[n].mean() - b[n].mean()
                pv = stats.ttest_ind(a[n], b[n], equal_var=False).pvalue if len(a) > 1 and len(b) > 1 \
                    else np.nan
                dec = int(np.sign(d)) if (pv == pv and pv < self.alpha) else 0
                rows.append((f"abundance|niche{n}", dec, d, int((niche == n).sum()), pv))
        return pd.DataFrame(rows, columns=["claim", "decision", "effect", "support", "p"])

    def run(self, t, labels, rep=0):
        codes, F = self._features(t, labels)
        start = self.seed + (rep * 1000 if self.vary_seed else 0)
        per = []
        for cl, C, _ in self._fits(F, range(start, start + self.n_seeds)):
            D = ((C[:, None, :] - self.centers[None]) ** 2).sum(2)
            r, c = linear_sum_assignment(D)
            lut = dict(zip(r, c))
            per.append(self._one(t, codes, np.array([lut[x] for x in cl])))
        A = pd.concat(per, keys=range(len(per)), names=["seed"]).reset_index(level=0)
        g = A.groupby("claim", sort=False)
        up = g.decision.apply(lambda d: (d > 0).mean())
        dn = g.decision.apply(lambda d: (d < 0).mean())
        dec = np.where(up >= self.consensus, 1, np.where(dn >= self.consensus, -1, 0))
        stat = np.where(dn > up, dn, up)
        out = _claims(list(up.index), stat, g.p.median().reindex(up.index).to_numpy(), dec,
                      support=g.support.mean().reindex(up.index).to_numpy(),
                      effect=g.effect.mean().reindex(up.index).to_numpy())
        out["unstable"] = (dec == 0) & (stat > self.unstable_low)
        return out


class ProximityDE(Question):
    """Expression of each gene in index-type cells vs the number of neighbor-type cells among its k nearest.

    Per section: OLS slope. aggregate="units" (default, needs >= 3 units): one-sample t-test of per-unit mean
    slopes, which respects independent units but has little power with few units. aggregate="pooled": one
    section-centerd regression over all cells, as many published analyses do (cells treated as replicates).
    BH over all (index, neighbor, gene) tests.
    effect = slope * k: change in log1p-normalized expression from no neighbors of type B to all k of them
    (about a natural-log fold change for expressed genes). Decision: q < alpha and |effect| >= min_effect.
    sigma (optional): use the Gaussian-weighted neighborhood of proximity_de_joint instead of plain counts
    (same features as the joint model, so the two differ only in marginal vs joint fitting); effect is then
    slope x median total weighted density.
    support = index cells in the test.
    """
    name = "proximity_de"

    def __init__(self, k=10, n_genes=200, genes=None, alpha=0.05, min_cells=30, index_types=None,
                 neighbour_types=None, aggregate="units", min_effect=0.0, sigma=None, **kw):
        super().__init__(**kw)
        self.min_effect, self.sigma = min_effect, sigma
        if aggregate not in ("units", "pooled"):
            raise ValueError("aggregate must be 'units' or 'pooled'")
        self.aggregate = aggregate
        self.k, self.n_genes, self.genes, self.alpha, self.min_cells = k, n_genes, genes, alpha, min_cells
        self.index_types, self.neighbour_types = index_types, neighbour_types

    def prepare(self, base):
        """Fixes the gene set and normalisation: the n_genes genes with the highest mean
        among the analyzed genes (the panel, base.meta["panel"], if set; else all genes), plus every planted gene;
        size factors from the analyzed genes minus the planted genes."""
        self.types = base.types
        panel = base.meta.get("panel")
        universe = np.flatnonzero(np.isin(base.genes, panel)) if panel is not None else np.arange(len(base.genes))
        if self.genes is None:
            mean = np.asarray(base.counts[:, universe].mean(0)).ravel()
            self.gidx = np.sort(universe[np.argsort(mean)[::-1][: self.n_genes]])
        else:
            self.gidx = np.array([np.flatnonzero(base.genes == g)[0] for g in self.genes])
        planted = [g for rec in base.meta.get("planted", []) for g in rec.get("genes", [])]
        extra = np.array([np.flatnonzero(base.genes == g)[0] for g in planted], int)
        self.gidx = np.unique(np.concatenate([self.gidx, extra]))
        self.gnames = base.genes[self.gidx]
        self.size_idx = np.setdiff1d(universe, extra)
        tot = np.asarray(base.counts[:, self.size_idx].sum(1)).ravel()
        self.scale = float(np.median(tot))
        # reference profiles (for the neighbor-marker audit), on baseline labels
        Z = np.asarray(base.counts[:, self.gidx].todense())
        self.profiles = pd.DataFrame({ty: Z[base.labels == ty].mean(0) for ty in self.types},
                                     index=self.gnames)

    def slopes(self, t, labels, cell_mask=None, min_cells=None, nb_labels=None, weights=None):
        """Per-unit slopes for every (index, neighbor) pair. Neighbor features use `labels` on all cells; index
        cells are the cells labeled A (restricted to cell_mask if given). Returns ({(A, B): (units, S, pooled)},
        eff_scale), S = units x genes (mean over a unit's sections), pooled = (sxx, sxy, syy, n, n_sections)."""
        codes, types = t.codes(labels, self.types)
        T = len(types)
        ncodes = codes if nb_labels is None else t.codes(nb_labels, self.types)[0]
        Nb = kernel_densities(t, ncodes, T, self.k, self.sigma)
        eff_scale = float(self.k) if self.sigma is None else float(np.median(Nb.sum(1)))
        mc = self.min_cells if min_cells is None else min_cells
        itypes = self.index_types or list(types)
        icodes = [int(np.flatnonzero(types == A)[0]) for A in itypes]
        sel = np.isin(codes, icodes) if cell_mask is None else np.isin(codes, icodes) & cell_mask
        rows = np.flatnonzero(sel)
        Zr = lognorm(t.counts[rows], self.scale, self.gidx, getattr(self, 'size_idx', None)).astype(np.float32)
        pos = np.full(t.n, -1); pos[rows] = np.arange(len(rows))
        ntypes = self.neighbour_types or list(types)
        out = {}
        for A in itypes:
            ai = int(np.flatnonzero(types == A)[0])
            for B in ntypes:
                bi = int(np.flatnonzero(types == B)[0])
                per_unit, pooled = {}, [0.0, np.zeros(len(self.gidx)), np.zeros(len(self.gidx)), 0, 0]
                for s in pd.unique(t.section):
                    m = (t.section == s) & (codes == ai) & sel
                    if m.sum() < mc:
                        continue
                    x = Nb[m, bi]
                    Zm = Zr[pos[m]].astype(np.float64)
                    if weights is None:
                        xc = x - x.mean()
                        sxx = xc @ xc
                        if sxx <= 0:
                            continue
                        Zc = Zm - Zm.mean(0)
                        sxy = xc @ Zc
                    else:                                  # inverse-probability weighted slope
                        w = weights[m]
                        xc = x - (w @ x) / w.sum()
                        sxx = (w * xc) @ xc
                        if sxx <= 0:
                            continue
                        Zc = Zm - (w @ Zm) / w.sum()
                        sxy = (w * xc) @ Zc
                    u = t.unit[m][0]
                    per_unit.setdefault(u, []).append(sxy / sxx)
                    pooled[0] += sxx; pooled[1] += sxy; pooled[2] += (Zc ** 2).sum(0)
                    pooled[3] += m.sum(); pooled[4] += 1
                if pooled[4] == 0:
                    continue
                units = list(per_unit)
                S = np.vstack([np.mean(per_unit[u], 0) for u in units])
                out[(A, B)] = (units, S, tuple(pooled))
        return out, eff_scale

    def claims_from(self, sl, eff_scale, n_units_total=None):
        """Claims from per-unit slopes (as returned by slopes, possibly corrected)."""
        use_units = self.aggregate == "units" and (n_units_total or 3) >= 3
        claim, stat, pv, sup, ses, dfs = [], [], [], [], [], []
        G = len(self.gnames)
        for (A, B), (units, S, pooled) in sl.items():
            if use_units and len(units) >= 3:
                if np.isnan(S).any():                  # per-gene units (corrected estimates may miss some)
                    nu = (~np.isnan(S)).sum(0)
                    import warnings
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)   # nu < 3 is set to NaN below
                        mu = np.nanmean(S, 0)
                        sd = np.nanstd(S, 0, ddof=1)
                        se = sd / np.sqrt(np.maximum(nu, 1))
                        tstat = mu / np.maximum(se, 1e-12)
                        p = 2 * stats.t.sf(np.abs(tstat), np.maximum(nu - 1, 1))
                    ok = nu >= 3                           # fewer than 3 units: not testable
                    p = np.where(ok, p, np.nan)
                    b = np.where(ok, mu, np.nan)
                    se = np.where(ok, se, np.nan)
                    df = np.where(ok, nu - 1, np.nan).astype(float)
                else:
                    mu, sd = S.mean(0), S.std(0, ddof=1)
                    se = sd / np.sqrt(len(S))
                    tstat = mu / np.maximum(se, 1e-12)
                    p = 2 * stats.t.sf(np.abs(tstat), len(S) - 1)
                    b = mu
                    df = np.full(G, float(len(S) - 1))
            elif pooled is not None:            # as before: pooled when fewer than 3 units
                sxx, sxy, syy, n, ns = pooled
                b = sxy / sxx
                dfp = max(n - 1 - ns, 1)
                se = np.sqrt(np.maximum(syy - b * sxy, 0) / dfp / sxx)
                tstat = b / np.maximum(se, 1e-12)
                p = 2 * stats.t.sf(np.abs(tstat), dfp)
                df = np.full(G, float(dfp))
            else:
                continue
            claim += [f"{A}|{B}|{g}" for g in self.gnames]
            stat += list(b)
            pv += list(p)
            ses += list(se)
            dfs += list(df)
            sup += [pooled[3] if pooled is not None else len(units)] * len(self.gnames)
        pv = np.asarray(pv, float)
        stat = np.asarray(stat, float)
        q = np.full(len(pv), np.nan)
        okp = ~np.isnan(pv)
        if okp.any():
            q[okp] = bh(pv[okp])
        eff = stat * eff_scale
        with np.errstate(invalid="ignore"):
            dec = np.where((q < self.alpha) & (np.abs(eff) >= self.min_effect), np.sign(np.nan_to_num(stat)), 0)
        out = _claims(claim, stat, pv, dec, support=sup, effect=eff, se=ses, df=dfs)
        out["q"] = q
        return out

    def run(self, t, labels, rep=0):
        sl, eff_scale = self.slopes(t, labels)
        return self.claims_from(sl, eff_scale, n_units_total=len(pd.unique(t.unit)))

    def marker_share(self, claims: pd.DataFrame, ratio: float = 2.0) -> dict:
        """Share of up- and down-called genes that are markers of the neighbor type (> ratio vs index type)."""
        c = claims[claims.decision != 0]
        if c.empty:
            return {"marker_share_up": np.nan, "marker_share_down": np.nan}
        parts = c.claim.str.split("|", expand=True)
        A, B, g = parts[0].to_numpy(), parts[1].to_numpy(), parts[2].to_numpy()
        P = self.profiles
        mk = np.array([(P.at[gg, b] + 1e-3) / (P.at[gg, a] + 1e-3) > ratio for a, b, gg in zip(A, B, g)])
        up, dn = c.decision.to_numpy() > 0, c.decision.to_numpy() < 0
        return {"marker_share_up": float(mk[up].mean()) if up.any() else np.nan,
                "marker_share_down": float(mk[dn].mean()) if dn.any() else np.nan}


class ProximityDEJoint(ProximityDE):
    """Niche DE with all neighbor types in one regression (as Niche-DE fits its effective niche).

    Each cell's neighborhood is the kernel-weighted count of every label among its k nearest cells
    (weights exp(-d^2 / (2 sigma^2)), sigma in coordinate units; truncated at k). Per section and index type, the
    expression of every gene is regressed on all neighbor-type densities at once (section-centerd OLS with a tiny
    ridge), so an effect of one neighbor type is estimated holding the others fixed, unlike proximity_de, which
    tests one neighbor type at a time. aggregate "units": t-test of per-unit mean coefficients; "pooled": one
    regression over all sections with OLS standard errors. BH over all tests.
    effect = coefficient x median total density (change from none to all neighbors of that type).
    Claims use the same identifiers as proximity_de ("A|B|gene").
    After each run, `diagnostics` holds, per index type A and neighbor type B, the variance inflation factor of
    B's density among A cells (pooled section-centerd design; VIF = 1 when B is uncorrelated with the other
    neighbor types) and the neighbor type most correlated with B. High VIF = the joint model cannot separate B
    from its partners (wide standard errors), which is where the one-at-a-time model misattributes.
    """
    name = "proximity_de_joint"

    def __init__(self, k=50, sigma=20.0, ridge=1e-6, **kw):
        super().__init__(k=k, **kw)
        self.sigma, self.ridge = sigma, ridge

    def _densities(self, t, codes, T):
        return kernel_densities(t, codes, T, self.k, self.sigma)

    def run(self, t, labels, rep=0):
        codes, types = t.codes(labels, self.types)
        T = len(types)
        D = self._densities(t, codes, T)
        scale_eff = float(np.median(D.sum(1)))
        itypes = self.index_types or list(types)
        ntypes = self.neighbour_types or list(types)
        ncols = np.array([int(np.flatnonzero(types == B)[0]) for B in ntypes])
        icodes = [int(np.flatnonzero(types == A)[0]) for A in itypes]
        rows = np.flatnonzero(np.isin(codes, icodes))
        Zr = lognorm(t.counts[rows], self.scale, self.gidx, getattr(self, 'size_idx', None)).astype(np.float32)
        pos = np.full(t.n, -1); pos[rows] = np.arange(len(rows))
        G, P = len(self.gidx), len(ncols)
        use_units = self.aggregate == "units" and len(pd.unique(t.unit)) >= 3
        claim, stat, pv, sup, diag, ses, dfs = [], [], [], [], [], [], []
        for A, ai in zip(itypes, icodes):
            per_unit, sec_vif, exposed, n_exp = {}, [], np.zeros(P), 0
            XtX, XtZ, ZtZ, n_tot, n_sec = np.zeros((P, P)), np.zeros((P, G)), np.zeros(G), 0, 0
            for s in pd.unique(t.section):
                m = (t.section == s) & (codes == ai)
                if m.sum() < max(self.min_cells, P + 2):
                    continue
                X = D[m][:, ncols]
                Xc = X - X.mean(0)
                Z = Zr[pos[m]].astype(np.float64)
                Zc = Z - Z.mean(0)
                a, b = Xc.T @ Xc, Xc.T @ Zc
                sda = np.sqrt(np.maximum(np.diag(a), 1e-12))
                sec_vif.append(np.diag(np.linalg.pinv(a / np.outer(sda, sda) + 1e-9 * np.eye(P))))
                exposed += (X >= 0.5).sum(0); n_exp += X.shape[0]
                lam = self.ridge * max(np.trace(a) / P, 1e-12)
                beta = np.linalg.solve(a + lam * np.eye(P), b)
                per_unit.setdefault(t.unit[m][0], []).append(beta)
                XtX += a; XtZ += b; ZtZ += (Zc ** 2).sum(0); n_tot += int(m.sum()); n_sec += 1
            if n_sec == 0:
                continue
            sd_ = np.sqrt(np.maximum(np.diag(XtX), 1e-12))
            R = XtX / np.outer(sd_, sd_)
            vif = np.diag(np.linalg.pinv(R + 1e-9 * np.eye(P)))
            Roff = R - np.diag(np.diag(R))
            SV = np.vstack(sec_vif)
            for j, B in enumerate(ntypes):
                jj = int(np.argmax(np.abs(Roff[j])))
                diag.append({"index": A, "neighbour": B, "vif": float(vif[j]), "partner": ntypes[jj],
                             "partner_r": float(Roff[j, jj]), "n_cells": n_tot,
                             "vif_section_median": float(np.median(SV[:, j])),
                             "vif_section_max": float(SV[:, j].max()),
                             "share_exposed": float(exposed[j] / max(n_exp, 1))})
            if use_units and len(per_unit) >= 3:
                S = np.stack([np.mean(v, 0) for v in per_unit.values()])        # units x P x G
                mu, sd = S.mean(0), S.std(0, ddof=1)
                semat = sd / np.sqrt(len(S))
                tstat = mu / np.maximum(semat, 1e-12)
                p = 2 * stats.t.sf(np.abs(tstat), len(S) - 1)
                bmat = mu
                dfv = float(len(S) - 1)
            else:
                lam = self.ridge * max(np.trace(XtX) / P, 1e-12)
                inv = np.linalg.pinv(XtX + lam * np.eye(P))
                bmat = inv @ XtZ
                rss = np.maximum(ZtZ - (bmat * XtZ).sum(0), 0)
                df = max(n_tot - n_sec - P, 1)
                semat = np.sqrt(np.outer(np.diag(inv), rss / df))
                tstat = bmat / np.maximum(semat, 1e-12)
                p = 2 * stats.t.sf(np.abs(tstat), df)
                dfv = float(df)
            for j, B in enumerate(ntypes):
                claim += [f"{A}|{B}|{g}" for g in self.gnames]
                stat += list(bmat[j]); pv += list(p[j]); sup += [n_tot] * G
                ses += list(semat[j]); dfs += [dfv] * G
        self.diagnostics = pd.DataFrame(diag)
        pv = np.asarray(pv)
        stat = np.asarray(stat, float)
        q = bh(pv)
        eff = stat * scale_eff
        dec = np.where((q < self.alpha) & (np.abs(eff) >= self.min_effect), np.sign(stat), 0)
        out = _claims(claim, stat, pv, dec, support=sup, effect=eff, se=ses, df=dfs)
        out["q"] = q
        return out


def _joint_slopes(self, t, labels, cell_mask=None, min_cells=None, nb_labels=None, weights=None):
    """Per-unit coefficients of the joint model for every (index, neighbor) pair, in the format of
    ProximityDE.slopes (pooled statistics are not kept: correctors use the units test)."""
    codes, types = t.codes(labels, self.types)
    T = len(types)
    ncodes = codes if nb_labels is None else t.codes(nb_labels, self.types)[0]
    D = self._densities(t, ncodes, T)
    scale_eff = float(np.median(D.sum(1)))
    itypes = self.index_types or list(types)
    ntypes = self.neighbour_types or list(types)
    ncols = np.array([int(np.flatnonzero(types == B)[0]) for B in ntypes])
    icodes = [int(np.flatnonzero(types == A)[0]) for A in itypes]
    sel = np.isin(codes, icodes) if cell_mask is None else np.isin(codes, icodes) & cell_mask
    rows = np.flatnonzero(sel)
    Zr = lognorm(t.counts[rows], self.scale, self.gidx, getattr(self, 'size_idx', None)).astype(np.float32)
    pos = np.full(t.n, -1); pos[rows] = np.arange(len(rows))
    P = len(ncols)
    mc = self.min_cells if min_cells is None else min_cells
    out = {}
    for A, ai in zip(itypes, icodes):
        per_unit = {}
        for s in pd.unique(t.section):
            m = (t.section == s) & (codes == ai) & sel
            if m.sum() < max(mc, P + 2):
                continue
            X = D[m][:, ncols]
            Z = Zr[pos[m]].astype(np.float64)
            w = np.ones(int(m.sum())) if weights is None else weights[m]
            Xc = X - (w @ X) / w.sum()
            Zc = Z - (w @ Z) / w.sum()
            a, b = Xc.T @ (w[:, None] * Xc), Xc.T @ (w[:, None] * Zc)
            lam = self.ridge * max(np.trace(a) / P, 1e-12)
            per_unit.setdefault(t.unit[m][0], []).append(np.linalg.solve(a + lam * np.eye(P), b))
        if not per_unit:
            continue
        units = list(per_unit)
        S = np.stack([np.mean(per_unit[u], 0) for u in units])          # units x P x G
        for j, B in enumerate(ntypes):
            out[(A, B)] = (units, S[:, j, :], None)
    return out, scale_eff


ProximityDEJoint.slopes = _joint_slopes


REGISTRY = {q.name: q for q in (Composition, Colocalization, Niches, ProximityDE, ProximityDEJoint)}


def make(spec) -> Question:
    """spec: a name or a dict {"name": ..., **parameters}."""
    if isinstance(spec, str):
        return REGISTRY[spec]()
    spec = dict(spec)
    return REGISTRY[spec.pop("name")](**spec)
