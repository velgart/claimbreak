"""Compartment analyses: label cells from different subsets of their own counts and ask where labels change.

Inputs are "compartment" AnnData files, one per section (made by examples/compartment/compartment_to_h5ad.py or
bilous_to_h5ad.py): X = whole-cell counts, layers['nucleus'] = counts over the nucleus, layers['cyto'] = the rest;
obs x, y (um), n_whole, n_nucleus, and optionally donor, panel, section, unit, tissue.

Three steps, each the packaged form of an example script with identical output (tests/test_compartment.py):

  label        (compartment_label.py)  every cell labeled from several count sets ("arms") against a single-cell
               reference, by one multinomial classifier with a per-gene platform factor gamma
  compare      (bilous_compare.py)     do label changes between two arms land on the types of contact neighbors,
               read against a null comparison (two independent halves of the nucleus counts)
  consequence  (bilous_consequence.py) kNN colocalization from two label sets, against random relabeling at the
               same confusion rates

Arms (same cells, classifier and reference; only the counts differ):
  whole, nucleus, cyto      X, layers['nucleus'], layers['cyto']
  thinned                   X thinned binomially to each cell's nucleus depth
  nucA, nucB                binomial 0.5 split of the nucleus counts (seed + 1): two independent nucleus samples
  cytoB, wholeB             cyto, and whole minus nucA, thinned to each cell's nucB depth
  wB<k>, wBall              whole minus nucA thinned to k% of the nucB depth, or not thinned
  nB<k>                     nucB thinned to k% (k <= 100)
  nucA3, nucB3, nucC3       multinomial thirds of the nucleus counts (own stream, seed + 2)
  wholeB3, cytoB3           whole minus nucA3 and nucC3, and cyto, thinned to each cell's nucB3 depth: the
                            compartment test conditioned on nucA3 == nucC3 (compare --agree nucC3) keeps cells whose
                            reference label an independent third confirms, so test changes are departures, not
                            corrections of nucA3

Classifier: mu[k, g] = mean panel-gene proportions of reference type k, mixed with a flat floor; per section and arm a
platform factor gamma[g] is refitted for `gamma_iter` rounds (observed over expected pseudobulk under the labels,
geometric mean 1, clipped); label = maximum likelihood (uniform prior). Options beyond the scripts, all off by
default (defaults reproduce the scripts exactly): `gamma_clip`, `gamma_damp` (gamma *= ratio ** damp) and
`gamma_fit="soft"` (expected pseudobulk from posteriors instead of hard labels), for platforms where hard-label
refitting can collapse onto one broad profile. `markers` restricts labeling to type-specific genes (each gene can
mark only the type with its highest mean, so ambient transcripts of one type in another's profile are not used, nor
are probes that mark no type); `exclude_genes` removes listed genes (both off by default).

Label audit: `label_audit` checks the nucleus and whole-cell arms of every labeled section (genes over-
detected beyond every reference type, calls that exist only through such a gene, marker support of each type) and
writes <sample>.audit.csv, meta.json `label_audit` and warnings; `audit_summary` (`compartment audit`) pools the
sections into an exclusion list for `exclude_genes`. The audit never changes a label (`audit=False` skips it).
`audit_existing` (`compartment audit --recompute H5AD_DIR`) runs the same audit on labels written
without it, from their stored labels, runner-ups, gamma and meta.json, without relabeling.

Second labeler: `labeler="celltypist"` labels every arm with a CellTypist logistic-regression model
trained on the same reference, restricted to the panel genes (and to the markers, if `markers` > 0). Training cells
(at most `ct_per_type` per type) are thinned to each depth in `ct_train_depths` (empty: full depth), normalized to
10,000 counts and log1p, as CellTypist expects; arms are normalized the same way. No platform factor (gamma is NaN);
the margin is the difference between the two highest decision scores. Models are cached per level and gene set.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

COMPARE_DEFAULT = "nucB|nucA,cytoB|nucA,wholeB|nucA,whole|nucleus,thinned|nucleus,cyto|nucleus,whole|thinned"
AGREEMENT_PAIRS = [("whole", "nucleus"), ("thinned", "nucleus"), ("whole", "thinned"), ("cyto", "nucleus"),
                   ("nucB", "nucA"), ("cytoB", "nucA"), ("wholeB", "nucA"),
                   ("nucB3", "nucA3"), ("nucC3", "nucA3"), ("wholeB3", "nucA3"), ("cytoB3", "nucA3")]


# --------------------------------------------------------------------------- classifier

@dataclass
class LabelConfig:
    floor: float = 0.01                  # flat share mixed into every profile (background, off-target counts)
    min_counts: int = 10                 # cells with fewer panel counts in an arm stay unlabeled in that arm
    gamma_iter: int = 5                  # rounds of platform-factor refitting (0 = none)
    gamma_clip: tuple = (0.05, 20.0)
    gamma_damp: float = 1.0              # gamma *= (observed / expected) ** gamma_damp
    gamma_fit: str = "hard"              # "hard": expected pseudobulk under the labels; "soft": under the posteriors
    collapse_warn: float = 0.6           # warn when one class exceeds this share of a labeled arm
    prior: str = "uniform"               # "uniform" (default) or "fit": class prior per section and arm
                                         # estimated by EM from the arm's own posteriors (floor prior_min)
    prior_min: float = 1e-4
    markers: int = 0                     # > 0: label on the N most specific genes per type only (Reference.markers)
    marker_min: float = 1e-4             # a marker's mean proportion in its own type is at least this
    marker_min_log2: float = 1.0         # and at least 2**this times the next type's mean
    exclude_genes: tuple = ()            # genes never used (e.g. probes detected far above the reference)
    labeller: str = "multinomial"        # "multinomial" (default) or "celltypist"
    ct_train_depths: tuple = (20, 50, 100, 200)   # celltypist: training cells thinned to these depths; () = full
    ct_per_type: int = 500               # celltypist: training cells per type (and per depth)
    ct_max_iter: int = 500               # celltypist: logistic-regression iterations
    ct_jobs: int = 1                     # celltypist: parallel jobs in training
    # neighbor-aware arms wholeM / cytoM / wholeMX (claimbreak.decontam)
    decon_gap: float = 10.0              # nucleus gap counted as mask contact (um)
    decon_region: float = 20.0           # radius of the regional term (um)
    decon_w: tuple = (1.0, 0.5, 0.1)     # weights of contact, regional and ambient terms in the contamination profile
    decon_psi_max: float = 0.9           # largest foreign share of the outside counts
    decon_psi_mode: str = "fit"          # "fit": psi from exposure with one fitted constant; "profile": free psi
    decon_fit_on: str = "max"            # labels kappa is fitted to: "max" (best candidate), "auto" (cytoM: nucB, else nucleus), an arm
    decon_betas: tuple = (0.0, 0.5, 1.0, 2.0)   # outside-share exponents tried (0 alone: no outside-share term)
    # label audit (label_audit): checks written to meta.json and <sample>.audit.csv; labels are unchanged
    audit: bool = True
    audit_arms: tuple = ("nucleus", "whole")    # arms audited when labeled
    audit_log2: float = 1.0              # over-detected: tissue share above 2**this times the highest type's share
    audit_min_share: float = 0.005       # ... and at least this share of the arm's counts
    audit_driver_share: float = 0.8      # a class rests on one gene when it carries this share of the class's evidence
                                         # (a strong genuine marker can carry ~0.5; CCL5 in synovium carried 1.26)
    audit_markers: int = 20              # markers per type for the support check (as Reference.markers)
    audit_min_support: float = 0.5       # warn when fewer of a class's calls carry one of its markers
    audit_decided_share: float = 0.05    # warn when this share of a class's calls exists only through one
                                         # over-detected gene (its counts outweigh the margin over the runner-up)
    audit_min_cells: int = 50            # classes with fewer calls are not audited
    audit_support_ratio: float = 0.7     # support warns only when observed support is also below this share
                                         # of the support expected at each call's depth from the reference profile


M_ARMS = ("wholeM", "cytoM", "wholeMX", "wBallM")   # neighbor-aware arms


def _q(mu, gamma, floor):
    q = (1 - floor) * mu + floor / mu.shape[1]
    q = q * gamma[None, :]
    return q / q.sum(1, keepdims=True)


def classify(X, mu, cfg: LabelConfig = LabelConfig(), fit_gamma: bool = True, gamma0=None):
    """X: cells x genes counts (csr), mu: types x genes profiles. Returns label index, second index, margin (nats),
    counts n and gamma; cells below cfg.min_counts get -1 / NaN. gamma0: starting platform factor (with
    fit_gamma=False, the factor used; NaN entries are 1)."""
    G = X.shape[1]
    K = mu.shape[0]
    gamma = np.ones(G) if gamma0 is None else np.nan_to_num(np.asarray(gamma0, float), nan=1.0)
    n = np.asarray(X.sum(1)).ravel()
    use = n >= cfg.min_counts
    Xu = X[use]
    obs_pb = np.asarray(Xu.sum(0)).ravel()
    lo, hi = cfg.gamma_clip
    if cfg.prior not in ("uniform", "fit"):
        raise ValueError(f"prior must be 'uniform' or 'fit', not {cfg.prior!r}")
    logpi = np.zeros(K)
    for it in range(cfg.gamma_iter + 1 if fit_gamma else 1):
        LL = np.asarray(Xu @ np.log(_q(mu, gamma, cfg.floor)).T)
        if cfg.prior == "fit" and len(LL):
            for _ in range(50):         # EM on the class shares with the likelihoods fixed
                Z = LL + logpi
                W = np.exp(Z - Z.max(1, keepdims=True))
                W /= W.sum(1, keepdims=True)
                pi = np.maximum(W.mean(0), cfg.prior_min)
                new = np.log(pi / pi.sum())
                if np.max(np.abs(new - logpi)) < 1e-6:
                    logpi = new
                    break
                logpi = new
            LL = LL + logpi
        best = LL.argmax(1)
        if not fit_gamma or it == cfg.gamma_iter:
            break
        q = _q(mu, gamma, cfg.floor)
        if cfg.gamma_fit == "soft":
            W = np.exp(LL - LL.max(1, keepdims=True))
            W /= W.sum(1, keepdims=True)
            exp_pb = (n[use][:, None] * W).sum(0) @ q
        elif cfg.gamma_fit == "hard":
            exp_pb = np.asarray(sp.csr_matrix((n[use], (best, np.zeros(len(best), int))),
                                              shape=(K, 1)).toarray()).ravel() @ q
        else:
            raise ValueError(f"gamma_fit must be 'hard' or 'soft', not {cfg.gamma_fit!r}")
        r = (obs_pb + 0.5) / (exp_pb + 0.5)
        gamma = gamma * (r if cfg.gamma_damp == 1.0 else r ** cfg.gamma_damp)
        gamma = np.clip(gamma / np.exp(np.mean(np.log(gamma))), lo, hi)
    srt = np.sort(LL, 1)
    second = np.argsort(LL, 1)[:, -2] if K > 1 else best
    lab = np.full(X.shape[0], -1)
    sec = np.full(X.shape[0], -1)
    mar = np.full(X.shape[0], np.nan)
    lab[use], sec[use] = best, second
    mar[use] = srt[:, -1] - srt[:, -2] if K > 1 else np.inf
    classify.last_prior = np.exp(logpi)          # fitted class shares (uniform: all equal)
    return lab, sec, mar, n, gamma


def _markers_from_mu(mu, n: int, min_prop: float = 1e-4, min_log2: float = 1.0):
    """Reference.markers on given profiles: per type k, positions of its n most specific genes (highest mean in k,
    at least min_prop there, at least 2**min_log2 over the next type), as a list of arrays."""
    if mu.shape[0] < 2:
        return [np.arange(mu.shape[1])]
    top = mu.argmax(0)
    second = np.sort(mu, 0)[-2]
    ratio = np.log2((mu.max(0) + 1e-9) / (second + 1e-9))
    out = []
    for k in range(mu.shape[0]):
        cand = np.where((top == k) & (mu[k] >= min_prop) & (ratio >= min_log2))[0]
        out.append(cand[np.argsort(-ratio[cand], kind="stable")][:n])
    return out


def label_audit(X, mu, types, genes, lab, sec=None, gamma=None, cfg: LabelConfig = LabelConfig()):
    """Three checks of one labeled arm, each detectable without protein:

    overdetected  genes whose share of the arm's counts (cells with >= cfg.min_counts) exceeds every reference type's
                  share by more than 2**cfg.audit_log2 while holding at least cfg.audit_min_share of the counts. No
                  mix of cell types explains such a gene; to the classifier it acts as a strong marker of whichever
                  type has its highest share (synovium: CCL5, 4-14% of nuclear counts, read as T/NK).
    drivers       (multinomial, needs sec) per class with >= cfg.audit_min_cells calls: each gene's share of the
                  summed evidence for the call over the runner-up, sum over cells of x_g (log q_kg - log q_jg); the
                  class rests on one gene when the top share reaches cfg.audit_driver_share. Pooled over the class,
                  so true members dilute it: a report, rarely a warning.
    decided       (multinomial) per over-detected gene: the share of calls of the type it reads as whose margin over
                  the runner-up is no larger than the gene's own contribution, i.e. calls that exist only through it
                  (warning at cfg.audit_decided_share). The direct measure of the damage.
    support       per class: share of its calls with at least one count in its cfg.audit_markers markers, over-
                  detected genes not counted (warning only with at least 5 markers). Also the support
                  expected of true members at each call's depth (support_expected); the warning needs the observed
                  share below cfg.audit_support_ratio of it, so shallow calls on large panels do not warn.
    decided also `misread`: on the runner-up types' markers, where the decided calls sit between true
                  members (0) and the runner-up types (1) at their depth. Reported, not interpreted: on real
                  data it did not separate misread from genuine calls, while the class support ratio did.
    Returns (dict for meta.json, tidy DataFrame, warnings)."""
    X = sp.csr_matrix(X)
    types = np.asarray(types)
    genes = list(genes)
    n = np.asarray(X.sum(1)).ravel()
    use = (n >= cfg.min_counts) & (np.asarray(lab) >= 0)
    rows, warn = [], []
    pb = np.asarray(X[use].sum(0)).ravel()
    obs = pb / max(pb.sum(), 1.0)
    mx = mu.max(0)
    l2 = np.log2((obs + 1e-12) / (mx + 1e-12))
    od = np.where((l2 > cfg.audit_log2) & (obs >= cfg.audit_min_share))[0]
    od = od[np.argsort(-obs[od])]
    over = [dict(gene=genes[j], obs_share=round(float(obs[j]), 5), max_type_share=round(float(mx[j]), 6),
                 log2=round(float(l2[j]), 3), type_of_max=str(types[mu[:, j].argmax()])) for j in od]
    for o in over:
        rows.append(dict(check="overdetected", type=o["type_of_max"], gene=o["gene"], value=o["obs_share"],
                         detail=o["log2"]))
        warn.append(f"{o['gene']} is {o['obs_share']:.3f} of counts, {2 ** o['log2']:.0f}x its highest reference "
                    f"share ({o['type_of_max']}): over-detected, reads as a {o['type_of_max']} marker "
                    f"(consider --exclude-genes)")
    lab = np.asarray(lab)
    drivers = {}
    if sec is not None and gamma is not None:
        logq = np.log(_q(mu, np.nan_to_num(np.asarray(gamma, float), nan=1.0), cfg.floor))
        sec = np.asarray(sec)
        for k, t in enumerate(types):
            mk = use & (lab == k) & (sec >= 0) & (sec != k)
            if mk.sum() < cfg.audit_min_cells:
                continue
            c = np.zeros(len(genes))
            for j in np.unique(sec[mk]):
                m = mk & (sec == j)
                c += np.asarray(X[m].sum(0)).ravel() * (logq[k] - logq[j])
            tot = c.sum()
            if tot <= 0:
                continue
            o = np.argsort(-c)[:5]
            drivers[str(t)] = dict(cells=int(mk.sum()), top=[[genes[g], round(float(c[g] / tot), 4)] for g in o],
                                   top_share=round(float(c[o[0]] / tot), 4))
            rows.append(dict(check="driver", type=str(t), gene=genes[o[0]], value=float(c[o[0]] / tot),
                             detail=int(mk.sum())))
            if c[o[0]] / tot >= cfg.audit_driver_share:
                warn.append(f"{t} calls rest on {genes[o[0]]} ({c[o[0]] / tot:.2f} of their evidence over the "
                            f"runner-up, {int(mk.sum()):,} cells)"
                            + (", an over-detected gene" if o[0] in set(od.tolist()) else ""))
    decided = {}
    mk_sets = [list(m.tolist()) for m in _markers_from_mu(mu, cfg.audit_markers, cfg.marker_min, cfg.marker_min_log2)]
    if sec is not None and gamma is not None and len(od):
        for g in od:                       # calls that exist only through an over-detected gene
            k = int(mu[:, g].argmax())
            mk = use & (lab == k) & (sec >= 0) & (sec != k)
            if mk.sum() < cfg.audit_min_cells:
                continue
            ii = np.where(mk)[0]
            Xk = X[ii]
            d = logq[k][None, :] - logq[sec[ii]]                         # cells x genes
            margin = np.asarray(Xk.multiply(d).sum(1)).ravel()
            xg = np.asarray(Xk[:, g].todense()).ravel()
            dep = xg * d[:, g] >= margin
            share = float(dep.mean())
            # are the decided calls misread cells of their runner-up type? Misread cells (synovial fibroblasts
            # called T through CCL5) carry the runner-up type's markers as often as that type does at their depth;
            # true members whose own marker is detected more efficiently than in the reference (TRAC in gut T cells)
            # carry them only by chance. misread = (observed - expected as type k) / (expected as the runner-up -
            # expected as type k), on the runner-up's markers (over-detected genes excluded): ~0 genuine, ~1
            # misread. The calls' own-class markers cannot separate the two: the decided calls are, by
            # construction, those whose other evidence does not favor k. NaN when the two expectations differ by
            # less than 0.1 or fewer than 20 calls are decided.
            misread = np.nan
            if dep.sum() >= 20:
                Xd = Xk[dep]
                sj = sec[ii][dep]
                q_ = _q(mu, np.nan_to_num(np.asarray(gamma, float), nan=1.0), cfg.floor)
                nd = np.asarray(Xd.sum(1)).ravel()
                obs, ek, ej = [], [], []
                odl = set(od.tolist())
                for j in np.unique(sj):
                    mj = np.array([x for x in mk_sets[j] if x not in odl], dtype=int)
                    if not len(mj):
                        continue
                    r_ = sj == j
                    obs.append((np.asarray(Xd[r_][:, mj].sum(1)).ravel() > 0).astype(float))
                    ek.append(1.0 - (1.0 - min(q_[k, mj].sum(), 1.0)) ** nd[r_])
                    ej.append(1.0 - (1.0 - min(q_[j, mj].sum(), 1.0)) ** nd[r_])
                if obs:
                    o_, k_, j_ = (float(np.concatenate(v).mean()) for v in (obs, ek, ej))
                    if j_ - k_ >= 0.1:
                        misread = (o_ - k_) / (j_ - k_)
            decided[genes[g]] = dict(type=str(types[k]), calls=int(mk.sum()), decided=int(dep.sum()),
                                     share=round(share, 4),
                                     misread=None if not np.isfinite(misread) else round(float(misread), 3))
            rows.append(dict(check="decided", type=str(types[k]), gene=genes[g], value=share, detail=int(mk.sum())))
            if np.isfinite(misread):
                rows.append(dict(check="decided_misread", type=str(types[k]), gene=genes[g], value=float(misread),
                                 detail=int(dep.sum())))
            if share >= cfg.audit_decided_share:
                tail = ""
                if np.isfinite(misread):
                    # number only. On real data it did not separate misread calls (CCL5 in synovium,
                    # 0.50-0.76 per section) from genuine ones (TRAC in gut, 0.42-0.79): decided calls are those
                    # whose other evidence leans to the runner-up, which runner-up marker counts and neighbors'
                    # spilled RNA both supply. The class support ratio separated them (0.41 against 0.92-1.00).
                    tail = (f"; runner-up marker score {misread:.2f} (0 like true {types[k]} cells, 1 like the "
                            f"runner-up types; not a verdict, see the class support ratio)")
                warn.append(f"{share:.2f} of {types[k]} calls ({int(dep.sum()):,} of {int(mk.sum()):,}) exist only "
                            f"through over-detected {genes[g]}: without its counts they fall below the runner-up"
                            + tail)
    support, support_exp = {}, {}
    odset = set(od.tolist())
    # support expected at depth. A true member of type k with n counts has at least one count in k's markers
    # with probability 1 - (1 - s_k)^n, s_k the markers' share of k's profile (platform factor and floor applied as
    # in the classifier). On a 5,000-gene panel with a few dozen nuclear counts that is often well below 0.5, so
    # observed support is judged against it: support / expected near 1 means the calls behave like true members.
    qk = _q(mu, np.nan_to_num(np.asarray(gamma, float), nan=1.0) if gamma is not None else np.ones(mu.shape[1]),
            cfg.floor)
    for k, (t, mpos) in enumerate(zip(types, _markers_from_mu(mu, cfg.audit_markers, cfg.marker_min,
                                                               cfg.marker_min_log2))):
        mk = use & (lab == k)
        mpos = np.array([j for j in mpos if j not in odset], dtype=int)
        if mk.sum() < cfg.audit_min_cells or not len(mpos):
            continue
        s = float((np.asarray(X[mk][:, mpos].sum(1)).ravel() > 0).mean())
        sk = float(min(qk[k, mpos].sum(), 1.0))
        e = float(np.mean(1.0 - (1.0 - sk) ** n[mk]))
        support[str(t)] = round(s, 4)
        support_exp[str(t)] = round(e, 4)
        rows.append(dict(check="support", type=str(t), gene="", value=s, detail=int(mk.sum())))
        rows.append(dict(check="support_expected", type=str(t), gene="", value=e, detail=int(mk.sum())))
        if s < cfg.audit_min_support and len(mpos) >= 5 and s < cfg.audit_support_ratio * e:
            warn.append(f"only {s:.2f} of {t} calls have a {t} marker transcript, against {e:.2f} expected of true "
                        f"{t} cells at their depth ({len(mpos)} markers; over-detected genes not counted)")
    return (dict(overdetected=over, drivers=drivers, decided=decided, support=support, support_expected=support_exp),
            pd.DataFrame(rows), warn)


def thin(X, n_target, rng):
    """Binomial thinning of each row to about n_target counts (rows with fewer stay as they are)."""
    X = sp.csr_matrix(X, copy=True)
    n = np.asarray(X.sum(1)).ravel()
    p = np.where(n > 0, np.minimum(n_target / np.maximum(n, 1), 1.0), 0.0)
    rowp = np.repeat(p, np.diff(X.indptr))
    X.data = rng.binomial(X.data.astype(np.int64), rowp).astype(np.float32)
    X.eliminate_zeros()
    return X


# --------------------------------------------------------------------------- second labeler: CellTypist

def _lognorm(X, target: float = 1e4):
    """Counts (csr) -> log1p of counts per `target` (csr, float32); rows without counts stay zero."""
    X = sp.csr_matrix(X, dtype=np.float32, copy=True)
    n = np.asarray(X.sum(1)).ravel()
    X = sp.diags((target / np.where(n > 0, n, 1)).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def ct_fit(M, labels, genes, cfg: LabelConfig = LabelConfig(), seed: int = 0):
    """Train a CellTypist model on reference counts M (cells x genes, csr) with labels: at most cfg.ct_per_type cells
    per type, each thinned to every depth of cfg.ct_train_depths (full depth if empty). Returns the model."""
    import logging
    import celltypist
    logging.getLogger("celltypist").setLevel(logging.WARNING)
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = np.asarray(M.sum(1)).ravel()
    rows = []
    for t in np.unique(labels):
        idx = np.where((labels == t) & (n > 0))[0]
        if len(idx) > cfg.ct_per_type > 0:
            idx = rng.choice(idx, cfg.ct_per_type, replace=False)
        rows.append(idx)
    rows = np.sort(np.concatenate(rows))
    blocks, ys = [], []
    for depth in (cfg.ct_train_depths or (None,)):
        Xd = M[rows] if depth is None else thin(M[rows], np.full(len(rows), float(depth)), rng)
        keep = np.asarray(Xd.sum(1)).ravel() > 0
        blocks.append(_lognorm(Xd[keep]))
        ys.append(labels[rows][keep])
    Xtr, ytr = sp.vstack(blocks).tocsr(), np.concatenate(ys)
    try:
        model = celltypist.train(Xtr, labels=ytr, genes=list(genes), check_expression=False,
                                 max_iter=cfg.ct_max_iter, n_jobs=cfg.ct_jobs, feature_selection=False)
        model.description["trainer"] = "celltypist.train"
    except TypeError as e:              # CellTypist <= 1.7.1 passes multi_class, removed in scikit-learn 1.8
        if "multi_class" not in str(e):
            raise
        model = _ct_train_recipe(Xtr, ytr, genes, cfg)
    return model


def _ct_train_recipe(X, y, genes, cfg: LabelConfig):
    """CellTypist's own training steps (celltypist/train.py, 1.7.1: drop genes without counts, StandardScaler with
    mean, clip at 10, one-vs-rest logistic regression with C = 1, lbfgs below 50,000 cells and sag above), with
    OneVsRestClassifier in place of LogisticRegression(multi_class='ovr'), which scikit-learn 1.8 removed. Returns a
    celltypist Model, so prediction is the same either way."""
    from celltypist.models import Model
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    genes = np.asarray(genes)
    keep = np.asarray(X.sum(0)).ravel() > 0
    scaler = StandardScaler(with_mean=True)
    Z = scaler.fit_transform(X[:, keep].toarray())
    Z[Z > 10] = 10
    lr = LogisticRegression(C=1.0, solver="sag" if Z.shape[0] > 50000 else "lbfgs", max_iter=cfg.ct_max_iter)
    clf = OneVsRestClassifier(lr, n_jobs=cfg.ct_jobs).fit(Z, y)
    clf.features = genes[keep]
    return Model(clf, scaler, {"trainer": "celltypist recipe (OneVsRestClassifier; scikit-learn >= 1.8)",
                               "number_celltypes": len(clf.classes_)})


def ct_classify(X, genes, model, cfg: LabelConfig = LabelConfig(), chunk: int = 5000):
    """Label counts X (cells x genes, csr; columns named by `genes`) with a CellTypist model, as celltypist.annotate
    does: log1p of counts per 10,000 over all of X's genes, then the model's features (training drops genes without
    counts), scaled with the model's scaler, clipped at 10, best match. Returns (types, label index, second index,
    margin, n) with -1 / NaN for cells below cfg.min_counts."""
    types = np.asarray(model.classifier.classes_).astype(str)
    pos = {str(g): j for j, g in enumerate(genes)}
    feat = np.array([pos[str(g)] for g in model.features], dtype=int)
    n = np.asarray(X.sum(1)).ravel()
    use = np.where(n >= cfg.min_counts)[0]
    lab = np.full(X.shape[0], -1)
    sec = np.full(X.shape[0], -1)
    mar = np.full(X.shape[0], np.nan)
    mean, scale = model.scaler.mean_, model.scaler.scale_
    scale = np.where(scale > 0, scale, 1.0)
    for i in range(0, len(use), chunk):
        r = use[i:i + chunk]
        Z = (_lognorm(X[r])[:, feat].toarray() - mean) / scale
        np.clip(Z, None, 10, out=Z)
        S = model.classifier.decision_function(Z)
        if S.ndim == 1:
            S = np.column_stack([-S, S])
        o = np.argsort(S, 1)
        lab[r] = o[:, -1]
        sec[r] = o[:, -2] if S.shape[1] > 1 else o[:, -1]
        mar[r] = np.take_along_axis(S, o[:, -1:], 1).ravel() - (np.take_along_axis(S, o[:, -2:-1], 1).ravel()
                                                                 if S.shape[1] > 1 else 0)
    return types, lab, sec, mar, n


# --------------------------------------------------------------------------- reference

class Reference:
    """A single-cell reference: raw counts, gene symbols, labels per level and a grouping column (for CV)."""

    def __init__(self, X, symbols, labels: dict, groups, source: dict | None = None):
        self.X = sp.csr_matrix(X)
        self.symbols = np.asarray(symbols)
        self.labels = labels
        self.groups = np.asarray(groups)
        self.source = source or {}
        self.pos = {g: i for i, g in enumerate(self.symbols)}
        self._panel, self._mu, self._ct = {}, {}, {}

    @classmethod
    def load(cls, path, levels: dict, merge: str = "", map_file: str = "", donor: str = "donor_id",
             raw: bool = False, layer: str = "", subset: str = "", cv_group: str = "", verbose: bool = True):
        """levels: {output level: reference obs column}. merge: "regex=>label;...". map_file: CSV level,from,to
        (to "" or "drop" removes cells). subset: "column=v1|v2;...". As compartment_label.py's --ref-* options."""
        import anndata as ad
        R = ad.read_h5ad(Path(path).expanduser())
        keep = np.ones(R.n_obs, bool)
        for cond in [c for c in subset.split(";") if c]:
            col, vals = cond.split("=", 1)
            keep &= R.obs[col].astype(str).isin(vals.split("|")).to_numpy()
        if raw:
            if R.raw is None:
                raise ValueError("raw=True: the reference has no .raw")
            RX, rvar = sp.csr_matrix(R.raw.X), R.raw.var
        else:
            RX, rvar = sp.csr_matrix(R.layers[layer] if layer else R.X), R.var
        sym = rvar["feature_name"].astype(str).to_numpy() if "feature_name" in rvar else rvar.index.astype(str).to_numpy()
        n_dup = 0
        if len(set(sym)) != len(sym):                      # keep the first occurrence of a duplicated symbol
            first = ~pd.Series(sym).duplicated().to_numpy()
            n_dup = int((~first).sum())
            RX, sym = RX[:, np.where(first)[0]], sym[first]
            if verbose:
                print(f"reference: {n_dup} duplicated gene symbols dropped", flush=True)
        merges = [m.split("=>", 1) for m in merge.split(";") if m]
        rmap = pd.read_csv(map_file) if map_file else None
        labs = {}
        for lv, col in levels.items():
            s = pd.Series(R.obs[col].astype(str).to_numpy())
            for rx, to in merges:
                s = s.where(~s.str.match(rx), to)
            if rmap is not None:
                m = rmap[rmap.level == lv]
                if len(m):
                    s = s.map(dict(zip(m["from"].astype(str), m["to"].fillna("").astype(str)))).fillna(s)
            s = s.where(~s.isin(["", "drop", "nan"]), "")
            labs[lv] = s.to_numpy()
            keep &= labs[lv] != ""
        gcol = cv_group or donor
        groups = R.obs[gcol].astype(str).to_numpy() if gcol in R.obs else np.full(R.n_obs, "all")
        rows = np.where(keep)[0]
        src = dict(reference=str(path), ref_layer="raw" if raw else (layer or "X"), ref_levels=dict(levels),
                   ref_merge=merge, ref_map=map_file, ref_subset=subset, cv_group=gcol, n_cells_file=int(R.n_obs),
                   n_genes_file=int(R.n_vars), duplicated_symbols=n_dup)
        ref = cls(RX[rows], sym, {lv: v[keep] for lv, v in labs.items()}, groups[keep], src)
        if verbose:
            print(f"reference: {R.n_obs:,} cells, {R.n_vars:,} genes; "
                  + "; ".join(f"{lv}: {len(set(ref.labels[lv]))} types" for lv in levels), flush=True)
        return ref

    def match(self, genes):
        """Genes of a panel found in the reference: (genes, reference indices, missing genes)."""
        g_in = [g for g in genes if g in self.pos]
        return g_in, np.array([self.pos[g] for g in g_in], dtype=int), [g for g in genes if g not in self.pos]

    def panel(self, gidx):
        """Per-cell panel-gene proportions (csr), cells with panel counts, raw panel counts. One panel cached."""
        key = tuple(int(i) for i in gidx)
        if key not in self._panel:
            self._panel.clear()
            M = self.X[:, gidx].tocsr()
            tot = np.asarray(M.sum(1)).ravel()
            ok = tot > 0
            self._panel[key] = (sp.diags(1.0 / np.where(ok, tot, 1)) @ M, ok, M)
        return self._panel[key]

    def profiles(self, level, gidx):
        """(types, mu): mean panel-gene proportion profile per type. Cached per (level, gene set)."""
        key = (level, tuple(int(i) for i in gidx))
        if key not in self._mu:
            P, ok, _ = self.panel(gidx)
            L = self.labels[level]
            types = np.array(sorted(set(L[ok])))
            mu = np.vstack([np.asarray(P[ok & (L == t)].mean(0)).ravel() for t in types])
            self._mu[key] = (types, mu)
        return self._mu[key]

    def ct_model(self, level, gidx, genes, sel=None, cfg: LabelConfig = LabelConfig(), seed: int = 0):
        """CellTypist model for one level on the panel genes (positions sel only, if given). Cached."""
        key = (level, tuple(int(i) for i in gidx), None if sel is None else tuple(int(i) for i in sel),
               tuple(cfg.ct_train_depths), cfg.ct_per_type, cfg.ct_max_iter, seed)
        if key not in self._ct:
            _, ok, M = self.panel(gidx)
            cols = np.arange(len(gidx)) if sel is None else np.asarray(sel)
            g = [genes[j] for j in cols]
            self._ct[key] = ct_fit(M[ok][:, cols], self.labels[level][ok], g, cfg, seed)
        return self._ct[key]

    def markers(self, level, gidx, n: int, min_prop: float = 1e-4, min_log2: float = 1.0):
        """Positions (into gidx) of the n most specific genes of every type: genes whose highest mean proportion is
        in that type, at least min_prop there and at least 2**min_log2 times the second-highest type's, ranked by
        that log2 ratio. A probe detected far above the reference that happens to be highest in one type can still
        pass: remove such genes with exclude_genes (examples/ibd/label_drivers.py lists them)."""
        types, mu = self.profiles(level, gidx)
        if len(types) < 2:
            return np.arange(mu.shape[1]), {types[0]: mu.shape[1]} if len(types) else {}
        top = mu.argmax(0)
        second = np.sort(mu, 0)[-2]
        ratio = np.log2((mu.max(0) + 1e-9) / (second + 1e-9))
        sel, per = [], {}
        for k, t in enumerate(types):
            cand = np.where((top == k) & (mu[k] >= min_prop) & (ratio >= min_log2))[0]
            cand = cand[np.argsort(-ratio[cand], kind="stable")][:n]
            per[t] = len(cand)
            sel.extend(cand.tolist())
        return np.array(sorted(set(sel)), dtype=int), per

    def cross_validate(self, gidx, panel: str, levels, cfg: LabelConfig = LabelConfig(), folds: int = 5,
                       per_type: int = 500, depths=(20, 50, 100, 200), seed: int = 0):
        """Grouped K-fold accuracy on the reference, test cells thinned to fixed depths (no platform factor).
        Returns (per fold/type/depth accuracy, confusion counts, K)."""
        rng = np.random.default_rng(seed)
        P, ok, M = self.panel(gidx)
        gu = np.unique(self.groups)
        perm = rng.permutation(len(gu))
        K = len(gu) if folds <= 0 else min(folds, len(gu))
        if K < 2:
            raise ValueError("cross-validation needs at least two groups")
        gfold = dict(zip(gu[perm], np.arange(len(gu)) % K))
        fold = np.array([gfold[g] for g in self.groups])
        rows, conf = [], []
        for lv in levels:
            lab_all = self.labels[lv]
            # markers chosen once from all reference cells (a mild optimism of the CV when cfg.markers > 0)
            msel = (self.markers(lv, gidx, cfg.markers, cfg.marker_min, cfg.marker_min_log2)[0]
                    if cfg.markers > 0 else None)
            types, fu, fcode, S, C = _group_sums(P, ok, lab_all, fold)
            St, Ct = S.sum(0), C.sum(0)
            for fi, fname in enumerate(fu):
                Ctr = Ct - C[fi]
                have = Ctr > 0
                ttypes = types[have]
                mu = (St - S[fi])[have] / Ctr[have, None]
                if msel is not None:
                    mu = mu[:, msel] / np.maximum(mu[:, msel].sum(1, keepdims=True), 1e-12)
                te = np.where((fcode == fi) & ok)[0]
                if per_type > 0:
                    lab_te = lab_all[te]
                    keep = []
                    for t in np.unique(lab_te):
                        idx = te[lab_te == t]
                        keep.append(idx if len(idx) <= per_type else rng.choice(idx, per_type, replace=False))
                    te = np.sort(np.concatenate(keep)) if keep else te
                Xte_full = M[te] if msel is None else M[te][:, msel]
                truth = lab_all[te]
                if cfg.labeller == "celltypist":
                    tr = np.where((fcode != fi) & ok)[0]
                    cols = np.arange(M.shape[1]) if msel is None else msel
                    model = ct_fit(M[tr][:, cols], lab_all[tr], [str(j) for j in cols], cfg, seed + fi)
                for depth in depths:
                    Xte = thin(Xte_full, np.full(Xte_full.shape[0], float(depth)), rng)
                    if cfg.labeller == "celltypist":
                        ct_types, lab, _, _, _ = ct_classify(Xte, [str(j) for j in cols], model, cfg)
                        pred = np.where(lab >= 0, ct_types[np.maximum(lab, 0)], "NA")
                    else:
                        lab, _, _, _, _ = classify(Xte, mu, cfg, fit_gamma=False)
                        pred = np.where(lab >= 0, ttypes[np.maximum(lab, 0)], "NA")
                    df = pd.DataFrame({"true": truth, "pred": pred})
                    df = df[df.pred != "NA"]
                    for t, g in df.groupby("true"):
                        rows.append({"panel": panel, "level": lv, "held_out_fold": int(fname), "depth": int(depth),
                                     "true": t, "n": len(g), "accuracy": float((g.pred == t).mean()),
                                     "type_in_training": bool(t in set(ttypes))})
                    cc = df.groupby(["true", "pred"]).size().reset_index(name="n")
                    cc["panel"], cc["level"], cc["held_out_fold"], cc["depth"] = panel, lv, int(fname), int(depth)
                    conf.append(cc)
        return pd.DataFrame(rows), pd.concat(conf) if conf else pd.DataFrame(), K


def _group_sums(P, ok, lab, groups):
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


# --------------------------------------------------------------------------- labeling a section

def _minus(X, Y):
    """X - Y (csr) with negative entries set to zero. A nucleus layer that is not a subset of the whole-cell
    counts (other segmentations, rounding) would otherwise give negative counts, which binomial thinning rejects."""
    R = (X - Y).tocsr()
    if R.nnz and R.data.min() < 0:
        R.data = np.maximum(R.data, 0)
    R.eliminate_zeros()
    return R


def count_sets(Xw, Xn, Xc, arms, seed: int = 0) -> dict:
    """Count matrices per arm (cells x genes, csr). Xc may be None if no cyto arm is asked for."""
    rng = np.random.default_rng(seed)
    out = {}
    need_split = {"nucA", "nucB", "cytoB", "wholeB"} & set(arms) or any(re.match(r"^(wB|nB)", x) for x in arms)
    if need_split:
        XB = sp.csr_matrix(Xn, copy=True)
        XB.data = np.random.default_rng(seed + 1).binomial(XB.data.astype(np.int64), 0.5).astype(np.float32)
        XA = (Xn - XB).tocsr()
        XA.eliminate_zeros(); XB.eliminate_zeros()
        nB = np.asarray(XB.sum(1)).ravel()
    if {"nucA3", "nucB3", "nucC3", "wholeB3", "cytoB3"} & set(arms):
        # three independent nuclear thirds (own stream, seed + 2, so nucA / nucB stay as before). nucC3
        # labels condition the compartment test (compare --agree nucC3); wholeB3 contains neither nucA3 nor nucC3.
        r3 = np.random.default_rng(seed + 2)
        N = sp.csr_matrix(Xn, copy=True)
        k = N.data.astype(np.int64)
        a3 = r3.binomial(k, 1.0 / 3.0)
        b3 = r3.binomial(k - a3, 0.5)
        thirds = {}
        for nm, v in (("nucA3", a3), ("nucB3", b3), ("nucC3", k - a3 - b3)):
            M = sp.csr_matrix((v.astype(np.float32), N.indices.copy(), N.indptr.copy()), shape=N.shape)
            M.eliminate_zeros()
            thirds[nm] = M
        nB3 = np.asarray(thirds["nucB3"].sum(1)).ravel()
    for arm in arms:
        if arm in ("nucA3", "nucB3", "nucC3"):
            out[arm] = thirds[arm]
        elif arm == "wholeB3":
            rest = _minus(_minus(Xw, thirds["nucA3"]), thirds["nucC3"])
            out[arm] = thin(rest, nB3, r3)
        elif arm == "cytoB3":
            out[arm] = thin(Xc, nB3, r3)
        elif arm == "whole":
            out[arm] = Xw
        elif arm == "nucleus":
            out[arm] = Xn
        elif arm == "cyto":
            out[arm] = Xc
        elif arm == "nucA":
            out[arm] = XA
        elif arm == "nucB":
            out[arm] = XB
        elif re.match(r"^wB(\d+|all)$", arm):
            rest = _minus(Xw, XA)
            out[arm] = rest if arm == "wBall" else thin(rest, int(arm[2:]) / 100.0 * nB, rng)
        elif re.match(r"^nB\d+$", arm):
            k = int(arm[2:]) / 100.0
            if k > 1:
                raise ValueError(f"{arm}: nucB cannot be thinned above 100%")
            out[arm] = XB if k == 1 else thin(XB, k * nB, rng)
        elif arm == "wholeB":
            out[arm] = thin(_minus(Xw, XA), nB, rng)
        elif arm == "cytoB":
            out[arm] = thin(Xc, nB, rng)
        elif arm == "thinned":
            out[arm] = thin(Xw, np.asarray(Xn.sum(1)).ravel(), rng)
        else:
            raise ValueError(f"unknown arm {arm}")
    return out


def panel_genes(ref: Reference, var_names, cfg: LabelConfig = LabelConfig()):
    """Panel genes found in the reference and not excluded: (genes, reference indices, missing, excluded)."""
    genes, gidx, missing = ref.match(list(var_names))
    ex = set(cfg.exclude_genes)
    keep = [i for i, g in enumerate(genes) if g not in ex]
    return [genes[i] for i in keep], gidx[keep], missing, sorted(ex & set(genes))


def _arm_info(res, info, lv, arm, lab, cfg):
    """Labelled share, composition and warnings of one labeled arm (both labelers)."""
    share_lab = float(np.mean(lab >= 0))
    info.setdefault("labelled", {})[f"{lv}_{arm}"] = round(share_lab, 4)
    if share_lab < 0.5:
        info["warnings"].append(f"{lv}_{arm}: only {share_lab:.2f} of cells have >= {cfg.min_counts} counts in the "
                                f"genes used" + (" (markers only: consider a lower --min-counts)"
                                                 if cfg.markers else ""))
    comp = res.loc[res[f"{lv}_{arm}"] != "", f"{lv}_{arm}"].value_counts(normalize=True)
    info["composition"][f"{lv}_{arm}"] = {k: round(float(v), 4) for k, v in comp.items()}
    if len(comp) and comp.iloc[0] > cfg.collapse_warn:
        info["warnings"].append(f"{lv}_{arm}: {comp.index[0]} is {comp.iloc[0]:.2f} of labelled cells "
                                f"(possible label collapse; check before any downstream statistic)")


def _audit_arm(info, lv, arm, X, mu, types, genes, lab, sec, gamma, cfg):
    """label_audit of one arm into info['audit'] (meta.json), info['audit_rows'] (<sample>.audit.csv) and warnings."""
    d, rows, warn = label_audit(X, mu, types, genes, lab, sec, gamma, cfg)
    info.setdefault("audit", {})[f"{lv}_{arm}"] = d
    if len(rows):
        info.setdefault("audit_rows", []).append(rows.assign(level=lv, arm=arm))
    info["warnings"].extend(f"{lv}_{arm} audit: {w}" for w in warn)


def _label_m_arms(A, res, info, gam, lv, types, mu, sel, Xn, m_sets, cfg: LabelConfig):
    """Neighbor-aware arms of one level (claimbreak.decontam): neighbors typed from their full nuclear counts
    (the nucleus arm's labels if present), contamination profile from geometry, profiled mixture likelihood."""
    from . import decontam as dc
    pick = (lambda M: M) if sel is None else (lambda M: None if M is None else M[:, sel])
    if f"{lv}_nucleus" in res:
        idx = {t: k for k, t in enumerate(types)}
        code = np.array([idx.get(t, -1) for t in res[f"{lv}_nucleus"]])
    else:
        code = classify(pick(Xn), mu, cfg)[0]
    area = A.obs["nucleus_area"].to_numpy(float) if "nucleus_area" in A.obs else None
    dcfg = dc.DecontamConfig(contact_gap=cfg.decon_gap, region=cfg.decon_region, w_contact=cfg.decon_w[0],
                             w_region=cfg.decon_w[1], w_ambient=cfg.decon_w[2], psi_max=cfg.decon_psi_max,
                             psi_mode=cfg.decon_psi_mode)
    if dcfg.psi_mode not in ("fit", "profile"):
        raise ValueError(f"decon_psi_mode must be 'fit' or 'profile', not {dcfg.psi_mode!r}")
    W, E, cov, note = dc.contamination_weights(A.obs[["x", "y"]].to_numpy(float), area, code,
                                            A.obs["n_whole"].to_numpy(float), len(types), dcfg)
    if note:
        info["warnings"].append(f"{lv} neighbour-aware arms: {note}")
    nw = A.obs["n_whole"].to_numpy(float)
    outside = 1 - A.obs["n_nucleus"].to_numpy(float) / np.maximum(nw, 1)
    for arm, (Xa, Xo) in m_sets.items():
        Xa, Xo = pick(Xa), pick(Xo)
        tot = Xo if Xa is None else (Xa + Xo).tocsr()
        _, _, _, n, gamma = classify(tot, mu, cfg)        # platform factor as for the plain arm on the same counts
        q = _q(mu, gamma, cfg.floor)
        use = n >= cfg.min_counts
        if dcfg.psi_mode == "fit":
            on = cfg.decon_fit_on
            if on == "auto":
                on = "nucB" if arm == "cytoM" and f"{lv}_nucB" in res else "nucleus"
            if on == "max":
                ref = None
            elif on == "nucleus" and f"{lv}_nucleus" not in res:
                ref = code
            elif f"{lv}_{on}" in res:
                idx = {t: k for k, t in enumerate(types)}
                ref = np.array([idx.get(t, -1) for t in res[f"{lv}_{on}"]])
            else:
                raise ValueError(f"--decon-fit-on {on}: no such arm labelled at level {lv} (add it to --arms)")
            kappa, beta, tab = dc.fit_kappa(Xa, Xo, q, W, E, use, dcfg, ref=ref, outside=outside,
                                            betas=tuple(cfg.decon_betas))
            info.setdefault("decon_kappa", {})[f"{lv}_{arm}"] = kappa
            info.setdefault("decon_beta", {})[f"{lv}_{arm}"] = beta
            info.setdefault("decon_fit_on", {})[f"{lv}_{arm}"] = on
            if kappa == dc.KAPPAS[-1]:
                info["warnings"].append(f"{lv}_{arm}: catchment constant at the top of its grid ({kappa})")
            Ei = E * dc.outside_index(outside, beta)
            LL, psi = dc.mixture_loglik(Xa, Xo, q, W, dcfg, psi=dc.psi_of(Ei, kappa, dcfg.psi_max))
        else:
            LL, psi = dc.mixture_loglik(Xa, Xo, q, W, dcfg)
        lab, sec, mar = dc.decide(LL, use)
        res[f"{lv}_{arm}"] = np.where(lab >= 0, types[np.maximum(lab, 0)], "")
        res[f"{lv}_{arm}_second"] = np.where(sec >= 0, types[np.maximum(sec, 0)], "")
        res[f"{lv}_{arm}_margin"] = np.round(mar, 3)
        res[f"{lv}_{arm}_psi"] = np.where(lab >= 0, psi[np.arange(len(lab)), np.maximum(lab, 0)], np.nan).round(2)
        if f"n_{arm}" not in res:
            res[f"n_{arm}"] = n.astype(int)
        if sel is not None:
            g_full = np.full(len(gam["gene"]), np.nan)
            g_full[sel] = gamma
            gamma = g_full
        gam[f"{lv}_{arm}"] = np.round(gamma, 4)
        _arm_info(res, info, lv, arm, lab, cfg)
    if "border_covered" not in res:
        res["border_covered"] = np.round(cov, 3)
        res["foreign_exposure"] = np.round(E, 3)


def label_section(A, ref: Reference, levels, arms, cfg: LabelConfig = LabelConfig(), seed: int = 0,
                  sample: str | None = None, gamma_fixed: dict | None = None):
    """Label one compartment AnnData. Returns (labels table, gamma table, info dict with agreement and composition).
    gamma_fixed: {"<level>_<arm>": {gene: platform factor}} (e.g. from an earlier run's <sample>.gamma.csv): that arm
    is labeled with the given factor instead of refitting it."""
    sample = sample or str(A.uns.get("sample", "section"))
    genes, gidx, missing, excluded = panel_genes(ref, A.var_names, cfg)
    xidx = np.array([A.var_names.get_loc(g) for g in genes], dtype=int)

    def sub(M):
        return sp.csr_matrix(M)[:, xidx].astype(np.float32)
    Xw, Xn = sub(A.X), sub(A.layers["nucleus"])
    Xc = sub(A.layers["cyto"]) if any(x in ("cyto", "cytoB", "cytoB3", "cytoM", "wBallM") for x in arms) else None
    m_arms = [x for x in arms if x in M_ARMS]
    arms = [x for x in arms if x not in M_ARMS]
    sets = count_sets(Xw, Xn, Xc, arms, seed)
    if m_arms and cfg.labeller != "multinomial":
        raise ValueError(f"arms {', '.join(m_arms)} need the multinomial labeller")
    m_sets = {}
    for x in m_arms:      # (nuclear counts or None, counts outside the nucleus)
        if x == "wholeM":
            m_sets[x] = (Xn, (Xw - Xn).tocsr())
        elif x == "wholeMX":
            m_sets[x] = (None, Xw)
        elif x == "wBallM":   # the other nuclear half plus the outside counts: same counts as wBall, judged against nucA
            XB = sets["nucB"] if "nucB" in sets else count_sets(Xw, Xn, Xc, ["nucB"], seed)["nucB"]
            m_sets[x] = (XB, (Xw - Xn).tocsr())
        else:             # cytoM: the same counts as cytoB (include cytoB in the arms to pair them)
            m_sets[x] = (None, sets["cytoB"] if "cytoB" in sets else count_sets(Xw, Xn, Xc, ["cytoB"], seed)["cytoB"])
        m_sets[x][1].eliminate_zeros()
    res = pd.DataFrame({"cell_id": A.obs_names.astype(str), "x": A.obs["x"].to_numpy(), "y": A.obs["y"].to_numpy(),
                        "n_whole": A.obs["n_whole"].to_numpy(), "n_nucleus": A.obs["n_nucleus"].to_numpy()})
    res.insert(0, "sample", sample)
    for c in ("donor", "panel", "section", "unit", "tissue"):
        if c in A.obs:
            res[c] = A.obs[c].astype(str).to_numpy()
    gam = {"gene": genes}
    info = {"genes_used": len(genes), "genes_missing": missing, "genes_excluded": excluded, "agreement": {},
            "composition": {}, "warnings": [], "markers": {}}
    for lv in levels:
        types, mu = ref.profiles(lv, gidx)
        sel = None
        if cfg.markers > 0:
            sel, per = ref.markers(lv, gidx, cfg.markers, cfg.marker_min, cfg.marker_min_log2)
            owner = mu[:, sel].argmax(0)
            info["markers"][lv] = {"n_genes": int(len(sel)), "per_type": per,
                                   "genes": {t: [genes[j] for j, o in zip(sel, owner) if o == k]
                                             for k, t in enumerate(types)}}
            mu = mu[:, sel] / np.maximum(mu[:, sel].sum(1, keepdims=True), 1e-12)
            few = [t for t, c in per.items() if c < 5]
            if few:
                info["warnings"].append(f"{lv}: fewer than 5 markers on this panel for {', '.join(few)}")
        if cfg.labeller == "celltypist":
            model = ref.ct_model(lv, gidx, genes, sel, cfg, seed)
            info.setdefault("ct_trainer", {})[lv] = model.description.get("trainer", "")
        elif cfg.labeller != "multinomial":
            raise ValueError(f"labeller must be 'multinomial' or 'celltypist', not {cfg.labeller!r}")
        for arm, X in sets.items():
            if cfg.labeller == "celltypist":
                ct_types, lab, sec, mar, n = ct_classify(X if sel is None else X[:, sel],
                                                         genes if sel is None else [genes[j] for j in sel], model, cfg)
                res[f"{lv}_{arm}"] = np.where(lab >= 0, ct_types[np.maximum(lab, 0)], "")
                res[f"{lv}_{arm}_second"] = np.where(sec >= 0, ct_types[np.maximum(sec, 0)], "")
                res[f"{lv}_{arm}_margin"] = np.round(mar, 3)
                if lv == levels[0]:
                    res[f"n_{arm}"] = n.astype(int)
                gam[f"{lv}_{arm}"] = np.full(len(genes), np.nan)
                _arm_info(res, info, lv, arm, lab, cfg)
                if cfg.audit and arm in cfg.audit_arms:
                    _audit_arm(info, lv, arm, X if sel is None else X[:, sel], mu,
                               types, genes if sel is None else [genes[j] for j in sel], lab, None, None, cfg)
                continue
            g0 = None
            if gamma_fixed and f"{lv}_{arm}" in gamma_fixed:     # platform factor of an earlier run, not refitted
                g0 = pd.Series(gamma_fixed[f"{lv}_{arm}"]).reindex(genes).to_numpy(float)
                g0 = g0 if sel is None else g0[sel]
            lab, sec, mar, n, gamma = classify(X if sel is None else X[:, sel], mu, cfg, fit_gamma=g0 is None,
                                               gamma0=g0)
            if cfg.prior == "fit":
                info.setdefault("prior", {})[f"{lv}_{arm}"] = {str(t): round(float(v), 5)
                                                              for t, v in zip(types, classify.last_prior)}
            if cfg.audit and arm in cfg.audit_arms:
                _audit_arm(info, lv, arm, X if sel is None else X[:, sel], mu, types,
                           genes if sel is None else [genes[j] for j in sel], lab, sec, gamma, cfg)
            if sel is not None:
                g_full = np.full(len(genes), np.nan)
                g_full[sel] = gamma
                gamma = g_full
            res[f"{lv}_{arm}"] = np.where(lab >= 0, types[np.maximum(lab, 0)], "")
            res[f"{lv}_{arm}_second"] = np.where(sec >= 0, types[np.maximum(sec, 0)], "")
            res[f"{lv}_{arm}_margin"] = np.round(mar, 3)
            if lv == levels[0]:
                res[f"n_{arm}"] = n.astype(int)
            gam[f"{lv}_{arm}"] = np.round(gamma, 4)
            _arm_info(res, info, lv, arm, lab, cfg)
        if m_arms:
            _label_m_arms(A, res, info, gam, lv, types, mu, sel, Xn, m_sets, cfg)
        all_arms = arms + m_arms
        pairs = AGREEMENT_PAIRS + [(x, "nucA") for x in arms if re.match(r"^(wB|nB)", x)] + \
            [("wholeM", "nucleus"), ("wholeMX", "nucleus"), ("cytoM", "nucA"), ("wholeM", "whole"),
             ("wBall", "nucA"), ("wBallM", "nucA")]
        for x, y in pairs:
            if x in all_arms and y in all_arms:
                ok = (res[f"{lv}_{x}"] != "") & (res[f"{lv}_{y}"] != "")
                info["agreement"][f"{lv}:{x}=={y}"] = (float(np.mean(res.loc[ok, f"{lv}_{x}"] == res.loc[ok, f"{lv}_{y}"])),
                                                     int(ok.sum()))
    return res, pd.DataFrame(gam), info


def label_files(ref: Reference, inputs, out, levels, arms, cfg: LabelConfig = LabelConfig(), seed: int = 0,
                ref_cv: bool = False, cv_folds: int = 5, cv_per_type: int = 500, cv_depths=(20, 50, 100, 200),
                verbose: bool = True):
    """Label every input file into OUT (refuses to overwrite; files already labeled are skipped). Writes, per
    sample, <sample>.labels.csv.gz, .gamma.csv and .meta.json, and with ref_cv ref_cv_<panel>.csv and the confusion."""
    import anndata as ad
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    cv_done, written = set(), []
    for f in inputs:
        f = Path(f).expanduser()
        sample = f.name.replace(".h5ad", "")
        dst = out / f"{sample}.labels.csv.gz"
        if dst.exists():
            say(f"{dst} exists; skipped", flush=True)
            continue
        A = ad.read_h5ad(f)
        panel = str(A.obs["panel"].iloc[0])
        genes, gidx, _, _ = panel_genes(ref, A.var_names, cfg)
        say(f"{sample}: panel {panel}, {A.n_obs:,} cells, {len(genes)} of {A.n_vars} genes in reference", flush=True)
        key = (panel, tuple(genes))
        if ref_cv and key not in cv_done:
            cv_done.add(key)
            cv, conf, K = ref.cross_validate(gidx, panel, levels, cfg, cv_folds, cv_per_type, cv_depths, seed)
            cv.to_csv(out / f"ref_cv_{panel.replace(' ', '_')}.csv", index=False)
            conf.to_csv(out / f"ref_cv_confusion_{panel.replace(' ', '_')}.csv", index=False)
            summ = (cv.assign(c=cv.n * cv.accuracy).groupby(["level", "depth"])[["c", "n"]].sum()
                    .assign(acc=lambda t: t.c / t.n)["acc"].unstack("depth").round(3))
            bal = cv.groupby(["level", "depth"]).accuracy.mean().unstack("depth").round(3)
            say(f"  reference check, {K} folds of {ref.source.get('cv_group')}, panel {panel}:\n  overall accuracy\n"
                f"{summ.to_string()}\n  mean per-type accuracy\n{bal.to_string()}", flush=True)
        res, gam, info = label_section(A, ref, levels, arms, cfg, seed, sample)
        for lv in levels:
            msg = [f"{k.split(':', 1)[1]} {v[0]:.3f} (n {v[1]:,})" for k, v in info["agreement"].items()
                   if k.startswith(lv + ":")]
            say(f"  {lv}: " + "; ".join(msg), flush=True)
            if f"{lv}_whole" in info["composition"]:
                c = info["composition"][f"{lv}_whole"]
                say(f"    whole-cell composition: {dict(list(c.items())[:12])}", flush=True)
        for w in info["warnings"]:
            say(f"  WARNING {w}", flush=True)
        res.to_csv(dst, index=False, compression="gzip")
        gam.to_csv(out / f"{sample}.gamma.csv", index=False)
        if info.get("audit_rows"):
            (pd.concat(info["audit_rows"], ignore_index=True).assign(sample=sample)
             [["sample", "level", "arm", "check", "type", "gene", "value", "detail"]]
             .to_csv(out / f"{sample}.audit.csv", index=False))
        meta = {**ref.source, "genes_used": info["genes_used"], "genes_missing": info["genes_missing"],
                "genes_excluded": info["genes_excluded"], "marker_genes": info["markers"],
                "labelled_share": info.get("labelled", {}), "ct_trainer": info.get("ct_trainer"),
                "decon_kappa": info.get("decon_kappa"), "decon_beta": info.get("decon_beta"),
                "decon_fit_on": info.get("decon_fit_on"), "prior_fitted": info.get("prior"),
                "label_audit": info.get("audit"),
                **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}, "seed": seed,
                "ref_cv": dict(folds=cv_folds, per_type=cv_per_type, depths=list(cv_depths)) if ref_cv else None,
                "levels": list(levels), "arms": list(arms), "agreement": info["agreement"],
                "composition": info["composition"], "warnings": info["warnings"]}
        (out / f"{sample}.meta.json").write_text(json.dumps(meta, indent=1))
        say(f"  wrote {dst}", flush=True)
        written.append(dst)
    return written


def audit_summary(labels_dir, out: str = "", arms=("nucleus", "whole"), min_sections: float = 0.5,
                  verbose: bool = True):
    """Pool the <sample>.audit.csv files of a labels folder. A gene goes on the exclusion list when it is
    over-detected in at least `min_sections` (a share of the sections, or a count when >= 1) in any of `arms`.
    Writes OUT/exclude_overdetected.txt (for `compartment label --exclude-genes`), overdetected.csv, drivers.csv and
    support.csv (one row per section, level, arm and class). Returns (exclusion list, overdetected, drivers)."""
    labels_dir = Path(labels_dir).expanduser()
    files = sorted(labels_dir.glob("*.audit.csv"))
    if not files:
        raise ValueError(f"no *.audit.csv in {labels_dir}: labels written with the audit off (see --recompute)")
    D = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    D = D[D.arm.isin(arms)]
    nsec = D["sample"].nunique()
    need = min_sections if min_sections >= 1 else np.ceil(min_sections * nsec)
    O = D[D.check == "overdetected"]
    og = (O.groupby(["gene", "type"]).agg(sections=("sample", "nunique"), median_share=("value", "median"),
                                          max_share=("value", "max"), median_log2=("detail", "median"))
          .reset_index().sort_values(["sections", "median_share"], ascending=False))
    excl = og.loc[og.sections >= need, "gene"].drop_duplicates().tolist()
    Dr = D[D.check == "driver"].rename(columns={"value": "top_share", "detail": "cells"})
    Su = D[D.check == "support"].rename(columns={"value": "support", "detail": "cells"})
    Se = D[D.check == "support_expected"]
    if len(Se):                                         # support expected at the calls' depth
        Su = Su.merge(Se.rename(columns={"value": "expected"})[["sample", "level", "arm", "type", "expected"]],
                      on=["sample", "level", "arm", "type"], how="left")
        Su["ratio"] = Su.support / Su.expected.where(Su.expected > 0)
    if out:
        o = Path(out).expanduser()
        o.mkdir(parents=True, exist_ok=True)
        (o / "exclude_overdetected.txt").write_text("".join(f"{g}\n" for g in excl))
        og.to_csv(o / "overdetected.csv", index=False)
        Dr.to_csv(o / "drivers.csv", index=False)
        Su.to_csv(o / "support.csv", index=False)
    if verbose:
        print(f"{len(files)} audited sections; over-detected in >= {int(need)} of {nsec}: {len(excl)} genes")
        if len(og):
            print(og.head(15).round(4).to_string(index=False))
        if len(Dr):
            t = (Dr.groupby(["level", "arm", "type"])
                 .agg(sections=("sample", "nunique"), median_top_share=("top_share", "median"),
                      top_gene=("gene", lambda g: g.value_counts().index[0])).reset_index())
            print("\nshare of a class's evidence carried by its top gene (median over sections):")
            print(t.round(3).to_string(index=False))
        if len(Su):
            print("\ncalls with a marker transcript of their class (median over sections):")
            print(Su.groupby(["level", "arm", "type"]).support.median().round(3).unstack("type").to_string())
        if "ratio" in Su:
            j = Su
            print("\nobserved over expected support at the calls' depth (~1: calls behave like true members; "
                  "median over sections):")
            print(j.groupby(["level", "arm", "type"]).ratio.median().round(2).unstack("type").to_string())
    return excl, og, Dr


def audit_existing(labels_dir, h5ad_dir, out, arms=("nucleus", "whole"), reference: str = "",
                   ref_levels: str = "", samples: str = "", min_sections: float = 0.5, verbose: bool = True,
                   verbose_overdetected: bool = False):
    """Label audit of labels written without it, without relabeling. For each <sample>.labels.csv.gz
    of LABELS: the counts of <sample>.h5ad in H5AD_DIR, the reference and settings of <sample>.meta.json, the genes
    and platform factor of <sample>.gamma.csv and the stored labels and runner-ups of each arm go into
    `label_audit`, exactly as `compartment label` would have run it. Writes OUT/<sample>.audit.csv, the pooled
    tables of `audit_summary` and OUT/reproduce.csv: per sample and arm, how well the stored margins are recomputed
    from the stored labels and gamma (a check that the reconstruction is the labeling's own; multinomial only).
    reference / ref_levels ("L1=coarse,L2=fine") override meta.json, for labels made on another machine or before
    the reference was recorded. Read-only on LABELS and H5AD_DIR; refuses a non-empty OUT."""
    import anndata as ad
    from dataclasses import fields
    labels_dir, h5ad_dir, out = Path(labels_dir).expanduser(), Path(h5ad_dir).expanduser(), Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} exists and is not empty")
    files = sorted(labels_dir.glob("*.labels.csv.gz"))
    if samples:
        keep = set(samples.split(","))
        files = [f for f in files if f.name.replace(".labels.csv.gz", "") in keep]
    if not files:
        raise ValueError(f"no *.labels.csv.gz in {labels_dir}")
    out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    cfgnames = {f.name for f in fields(LabelConfig)}
    refs, rep, skipped = {}, [], []
    for f in files:
        sample = f.name.replace(".labels.csv.gz", "")
        mf, gf, hf = labels_dir / f"{sample}.meta.json", labels_dir / f"{sample}.gamma.csv", h5ad_dir / f"{sample}.h5ad"
        miss = [x.name for x in (mf, gf, hf) if not x.exists()]
        if miss:
            skipped.append((sample, "missing " + ", ".join(miss)))
            say(f"{sample}: skipped (missing {', '.join(miss)})", flush=True)
            continue
        m = json.loads(mf.read_text())
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in m.items() if k in cfgnames}
        cfg = LabelConfig(**kw)
        rpath = reference or m.get("reference", "")
        lvmap = dict(x.split("=", 1) for x in ref_levels.split(",") if x) if ref_levels else m.get("ref_levels")
        if not rpath or not lvmap:
            raise ValueError(f"{mf.name}: no reference or ref_levels recorded; pass --reference and --ref-levels")
        layer = m.get("ref_layer", "X")
        key = (str(rpath), json.dumps(lvmap, sort_keys=True), m.get("ref_merge", ""), m.get("ref_map", ""),
               m.get("ref_subset", ""), layer, m.get("cv_group", ""))
        if key not in refs:
            refs[key] = Reference.load(rpath, lvmap, merge=m.get("ref_merge", ""), map_file=m.get("ref_map", ""),
                                       raw=(layer == "raw"), layer="" if layer in ("X", "raw") else layer,
                                       subset=m.get("ref_subset", ""), cv_group=m.get("cv_group", ""),
                                       verbose=verbose)
        ref = refs[key]
        G = pd.read_csv(gf)
        genes = G["gene"].astype(str).tolist()             # the genes the labeling used (exclusions applied)
        g_in, gidx, g_miss = ref.match(genes)
        if g_miss:
            raise ValueError(f"{sample}: {len(g_miss)} genes of {gf.name} not in the reference ({g_miss[:5]}): "
                             f"not the labelling's reference")
        A = ad.read_h5ad(hf)
        absent = [g for g in genes if g not in A.var_names]
        if absent:
            raise ValueError(f"{sample}: {len(absent)} genes of {gf.name} not in {hf.name} ({absent[:5]})")
        xidx = np.array([A.var_names.get_loc(g) for g in genes], dtype=int)

        def sub(Mx):
            return sp.csr_matrix(Mx)[:, xidx].astype(np.float32)
        L = pd.read_csv(f, keep_default_na=False, dtype={"cell_id": str})
        L = L.set_index("cell_id").reindex(A.obs_names.astype(str))
        lost = int(L["x"].isna().sum()) if "x" in L else 0
        Xw, Xn = sub(A.X), sub(A.layers["nucleus"])
        all_arms = [x for x in m.get("arms", list(arms)) if x not in M_ARMS]
        want = [x for x in arms if x in all_arms]
        if not want:
            skipped.append((sample, f"none of {arms} labelled"))
            continue
        if set(want) <= {"nucleus", "whole"}:
            sets = {"nucleus": Xn, "whole": Xw}
        else:                                               # split arms: same seed and arm order as the labeling
            Xc = sub(A.layers["cyto"]) if "cyto" in A.layers else None
            sets = count_sets(Xw, Xn, Xc, all_arms, int(m.get("seed", 0)))
        rows = []
        for lv in m.get("levels", []):
            types, mu = ref.profiles(lv, gidx)
            tix = {t: k for k, t in enumerate(types)}
            sel = None
            if cfg.markers > 0:
                sel, _ = ref.markers(lv, gidx, cfg.markers, cfg.marker_min, cfg.marker_min_log2)
                mu = mu[:, sel] / np.maximum(mu[:, sel].sum(1, keepdims=True), 1e-12)
            gl = genes if sel is None else [genes[j] for j in sel]
            for arm in want:
                col = f"{lv}_{arm}"
                if col not in L:
                    continue
                lab = np.array([tix.get(t, -1) for t in L[col].fillna("").astype(str)])
                bad = sorted(set(L[col].fillna("").astype(str)) - set(types) - {""})
                if bad:
                    raise ValueError(f"{sample} {col}: labels {bad[:5]} not among the reference types {list(types)}")
                X = sets[arm] if sel is None else sets[arm][:, sel]
                multi = cfg.labeller == "multinomial" and f"{col}_second" in L and col in G
                sec = gam = None
                if multi:
                    sec = np.array([tix.get(t, -1) for t in L[f"{col}_second"].fillna("").astype(str)])
                    gam = G[col].to_numpy(float)
                    gam = gam if sel is None else gam[sel]
                    # reproduction: stored margin against the margin recomputed from the stored gamma
                    ok = (lab >= 0) & (sec >= 0)
                    if ok.any() and f"{col}_margin" in L:
                        logq = np.log(_q(mu, np.nan_to_num(gam, nan=1.0), cfg.floor))
                        LLk = np.asarray(X[ok] @ logq.T)
                        lp = (m.get("prior_fitted") or {}).get(col)
                        if lp:
                            LLk = LLk + np.log(np.array([lp.get(str(t), 1.0) for t in types]))[None, :]
                        ii = np.arange(ok.sum())
                        re_m = LLk[ii, lab[ok]] - LLk[ii, sec[ok]]
                        st = pd.to_numeric(L[f"{col}_margin"], errors="coerce").to_numpy(float)[ok]
                        d = np.abs(re_m - st)
                        tol = 0.002 + 1e-3 * np.abs(st)            # gamma is stored to 4 decimals
                        rep.append(dict(sample=sample, level=lv, arm=arm, cells=int(ok.sum()),
                                        median_abs_diff=float(np.nanmedian(d)), share_within_tol=float(np.mean(d <= tol)),
                                        label_is_max=float(np.mean(np.argmax(LLk, 1) == lab[ok])),
                                        cells_not_in_labels=lost))
                d_, r_, w_ = label_audit(X, mu, types, gl, lab, sec, gam, cfg)
                if len(r_):
                    rows.append(r_.assign(level=lv, arm=arm))
                # over-detected genes are listed in the tables; only those that decide calls, rest a class
                # on one gene or leave calls without markers are printed
                n_od = sum("over-detected, reads as" in w for w in w_)
                for w in w_:
                    if verbose_overdetected or "over-detected, reads as" not in w:
                        say(f"  {sample} {col} audit: {w}", flush=True)
                if n_od and not verbose_overdetected:
                    say(f"  {sample} {col} audit: {n_od} over-detected genes (overdetected.csv)", flush=True)
        if rows:
            (pd.concat(rows, ignore_index=True).assign(sample=sample)
             [["sample", "level", "arm", "check", "type", "gene", "value", "detail"]]
             .to_csv(out / f"{sample}.audit.csv", index=False))
        say(f"{sample}: audited {', '.join(want)}", flush=True)
    R = pd.DataFrame(rep)
    R.to_csv(out / "reproduce.csv", index=False)
    (out / "audit_existing.json").write_text(json.dumps(dict(labels=str(labels_dir), h5ad=str(h5ad_dir),
                                                             arms=list(arms), reference_override=reference,
                                                             skipped=skipped), indent=1))
    if len(R):
        say("\nreproduction of the stored margins (share of cells within tolerance; label is the maximum):\n"
            + R.groupby(["level", "arm"])[["share_within_tol", "label_is_max", "median_abs_diff"]].median()
            .round(4).to_string(), flush=True)
        if (R.label_is_max < 0.99).any():
            say("WARNING: stored labels are not the maximum under the reconstructed profiles in some sections: "
                "the reference, genes or settings differ from the labelling's; do not use this audit", flush=True)
    if any(out.glob("*.audit.csv")):
        audit_summary(out, out, arms=tuple(arms), min_sections=min_sections, verbose=verbose)
    return R


# --------------------------------------------------------------------------- where do label changes land

def regional_shares(xy, code, K: int, ring=(30.0, 100.0), px: float = 5.0):
    """per cell and type b, the share of b among the labeled cells whose centers lie in the ring
    ring[0] < d <= ring[1] um around the cell (the cell itself and its touching neighbors fall inside ring[0]).
    Cells are binned to a `px`-um grid and counted with an FFT convolution against the ring, so a 700,000-cell
    section takes seconds; the ring edges are exact to about one pixel. Returns (cells x K shares, NaN where the ring
    holds no labeled cell; cells in the ring)."""
    from scipy.signal import fftconvolve
    xy = np.asarray(xy, float)
    code = np.asarray(code, int)
    r_in, r_out = float(ring[0]), float(ring[1])
    pad = int(np.ceil(r_out / px)) + 1
    ij = np.floor((xy - xy.min(0)) / px).astype(np.int64) + pad
    shape = tuple(ij.max(0) + pad + 1)
    h = np.arange(-pad, pad + 1) * px
    dd = np.hypot(h[:, None], h[None, :])
    ker = ((dd > r_in) & (dd <= r_out)).astype(float)
    ok = code >= 0
    flat = np.ravel_multi_index((ij[:, 0], ij[:, 1]), shape)
    tot = np.zeros(len(xy))
    S = np.zeros((len(xy), K))
    for b in range(K):
        m = ok & (code == b)
        if not m.any():
            continue
        H = np.bincount(flat[m], minlength=int(np.prod(shape))).reshape(shape).astype(float)
        Cb = fftconvolve(H, ker, mode="same")
        S[:, b] = np.maximum(Cb[ij[:, 0], ij[:, 1]], 0.0)
    tot = S.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(tot[:, None] > 0.5, S / np.maximum(tot[:, None], 1e-12), np.nan)
    return share, np.round(tot).astype(int)


def _share_bins(share, eligible, nbins: int = 5):
    """bin each column of `share` (cells x K): bin 0 = no cell of that type in the ring, bins 1..nbins-1 =
    quantiles of the positive shares among eligible cells, bin nbins = no labeled cell in the ring (NaN)."""
    n, K = share.shape
    out = np.zeros((n, K), np.int64)
    for b in range(K):
        s = share[:, b]
        pos = eligible & np.isfinite(s) & (s > 0)
        if nbins > 1 and pos.sum() >= nbins:
            edges = np.unique(np.quantile(s[pos], np.linspace(0, 1, nbins)[1:-1]))
        else:
            edges = np.array([])
        k = np.where(np.isfinite(s) & (s > 0), 1 + np.searchsorted(edges, np.nan_to_num(s), side="right"), 0)
        out[:, b] = np.where(np.isfinite(s), np.minimum(k, nbins - 1), nbins)
    return out


def neighbour_shells(xy, k_contact: int = 6, k_far: int = 20, radius: float = 30.0, radius_far: float = 60.0):
    """kNN indices (ranks 1..k_far) and masks of contact (ranks <= k_contact within radius) and far neighbors."""
    dist, idx = cKDTree(xy).query(xy, k=k_far + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]
    rank = np.arange(k_far)[None, :]
    return idx, (rank < k_contact) & (dist <= radius), (rank >= k_contact) & (dist <= radius_far)


def compare_section(D: pd.DataFrame, levels, comparisons, nbr_arm: str = "nucleus", k_contact: int = 6,
                    k_far: int = 20, radius: float = 30.0, radius_far: float = 60.0, min_margin: float = 0.0,
                    common_cells=(), tile: float = 200.0, tiles_out: list | None = None, types_out: list | None = None,
                    agree: str = "", edges=None, region=(30.0, 100.0), region_bins: int = 10, region_px: float = 5.0,
                    match_degree: bool = True, region_min: int = 20):
    """One labels table. Returns (pair rows, summary rows): per comparison T|R, level, pair (a, b) and stratum
    (contact / far_only / absent of type b around the cell), the rate of changes a -> b; and the observed over
    expected share of changes whose new type is a contact (or far-only) neighbor's.
    with tiles_out / types_out (lists), also the same sums per square tile of `tile` um (cells, changes,
    changes landing on a contact / far-only neighbor's type and their expected number) and, per tile and reference
    type, cells and cells keeping the type: the input of the block bootstrap in `summarise`.
    `agree` (an arm, e.g. nucC3): only cells whose reference label equals that arm's label are eligible, in
    every comparison (null included), so changes are departures from a confirmed label rather than corrections of a
    wrong one. `edges` (n x 2 indices, e.g. contacts.boundary_edges): contact = touching segmentation boundaries
    instead of the 6 nearest within 30 um; far only = kNN ranks 1..k_far within radius_far that do not touch.
    `region` (inner, outer radius in um; None = off) adds a regionally matched expected share,
    exp_contact_reg: for a change a -> b, the share of type-a cells touching b among type-a cells with the same
    regional share of b (cells in the ring, `region_bins` bins) and, with `match_degree`, the same number of contacts,
    leaving the changed cell out of its own stratum; strata with fewer than `region_min` other cells fall back to the
    regional bin alone, then to the plain expectation.
    The plain expectation compares a changed cell with type-a cells anywhere in the section, so changes that follow
    regional composition (cells 30-100 um away) score above one at contact without touching mattering; matched on the
    region they score about one. obs_contact_reg equals obs_contact; the two expectations sit side by side."""
    sample = D["sample"].iloc[0]
    xy_ = D[["x", "y"]].to_numpy(float)
    if tile and tile > 0:               # collision-free tile keys (x * 100,000 + y collided for negative y)
        tx_, ty_ = (np.floor(xy_[:, j] / tile).astype(np.int64) for j in (0, 1))
        tkey = (tx_ - tx_.min()) * (int(ty_.max() - ty_.min()) + 1) + (ty_ - ty_.min())
    else:
        tkey = np.zeros(len(D), np.int64)
    idx, contact, far = neighbour_shells(D[["x", "y"]].to_numpy(float), k_contact, k_far, radius, radius_far)
    if edges is not None:
        E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        n_ = len(D)
        Adj = sp.csr_matrix((np.ones(2 * len(E), bool), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                            shape=(n_, n_))
        _, _, far = neighbour_shells(xy_, 0, k_far, radius, radius_far)   # every kNN rank within radius_far
        touch = np.asarray(Adj[np.repeat(np.arange(n_), k_far), idx.ravel()]).ravel().reshape(idx.shape)
        far = far & ~touch
        deg = np.bincount(E.ravel(), minlength=n_) if len(E) else np.zeros(n_, np.int64)
    else:
        deg = contact.sum(1)
    pair_rows, summ_rows = [], []
    for lv in levels:
        nb = D[f"{lv}_{nbr_arm}"].to_numpy()
        types = np.array(sorted(set(nb) - {""}))
        tix = {t: i for i, t in enumerate(types)}
        nbi = np.array([tix.get(t, -1) for t in nb])
        K = len(types)
        nbr_t = nbi[idx]
        C = np.zeros((len(D), K), bool)
        F = np.zeros((len(D), K), bool)
        for j in range(k_far):
            ok = nbr_t[:, j] >= 0
            if edges is None:
                rows = np.where(ok & contact[:, j])[0]
                C[rows, nbr_t[rows, j]] = True
            rows = np.where(ok & far[:, j])[0]
            F[rows, nbr_t[rows, j]] = True
        if edges is not None and len(E):
            for u, v in ((E[:, 0], E[:, 1]), (E[:, 1], E[:, 0])):
                ok = nbi[v] >= 0
                C[u[ok], nbi[v[ok]]] = True
        Fo = F & ~C
        if region is not None:          # strata for the regionally matched expectation
            share_r, _ = regional_shares(xy_, nbi, K, region, region_px)
            RB = _share_bins(share_r, nbi >= 0, region_bins)
            dg = np.minimum(deg, 8).astype(np.int64) if match_degree else np.zeros(len(D), np.int64)
            ND = int(dg.max()) + 1
        for T, Rr in comparisons:
            if f"{lv}_{T}" not in D or f"{lv}_{Rr}" not in D:
                continue
            t, r = D[f"{lv}_{T}"].to_numpy(), D[f"{lv}_{Rr}"].to_numpy()
            mg = pd.to_numeric(D[f"{lv}_{Rr}_margin"], errors="coerce").to_numpy()
            el = (t != "") & (r != "") & (np.nan_to_num(mg, nan=-1) >= min_margin) & np.isin(r, types)
            for cx in common_cells:
                el &= D[f"{lv}_{cx}"].to_numpy() != ""
            if agree:
                if f"{lv}_{agree}" not in D:
                    raise KeyError(f"--agree {agree}: no column {lv}_{agree} in the labels of {sample}")
                el &= r == D[f"{lv}_{agree}"].to_numpy()
            ri = np.array([tix.get(x, -1) for x in r])
            ti = np.array([tix.get(x, -1) for x in t])
            chg = el & (ti != ri) & (ti >= 0)
            for ai in range(K):
                ca = el & (ri == ai)
                na = int(ca.sum())
                if na == 0:
                    continue
                for bi in range(K):
                    if bi == ai:
                        continue
                    hit = ca & (ti == bi)
                    for sname, S in (("contact", C[:, bi]), ("far_only", Fo[:, bi]), ("absent", ~C[:, bi] & ~F[:, bi])):
                        m = ca & S
                        n = int(m.sum())
                        if n:
                            pair_rows.append({"sample": sample, "level": lv, "comparison": f"{T}|{Rr}",
                                              "a": types[ai], "b": types[bi], "stratum": sname, "n": n,
                                              "changes": int((hit & S).sum()), "rate": (hit & S).sum() / n})
            pC = np.zeros((K, K))
            pF = np.zeros((K, K))
            for ai in range(K):
                ca = el & (ri == ai)
                if ca.any():
                    pC[ai] = C[ca].mean(0)
                    pF[ai] = Fo[ca].mean(0)
            cr, ct = ri[chg], ti[chg]
            expR = np.full(len(D), np.nan)
            if region is not None and chg.any():
                # expected contact share of each change a -> b among type-a cells in the same regional-share
                # bin of b (and contact count); the changed cell is in its own stratum, so the share is defined
                ci_ = np.where(chg)[0]
                for ai in np.unique(cr):
                    ca = el & (ri == ai)
                    for bi in np.unique(ct[cr == ai]):
                        sel = ci_[(cr == ai) & (ct == bi)]
                        own = C[sel, bi].astype(float)              # leave the changed cell out of its stratum
                        e_ = np.full(len(sel), pC[ai, bi])
                        for s in (RB[:, bi], RB[:, bi] * ND + dg):  # coarse first, finer overwrites where it can
                            m_ = int(s.max()) + 1
                            den = np.bincount(s[ca], minlength=m_)[s[sel]] - 1.0
                            num = np.bincount(s[ca], weights=C[ca, bi].astype(float), minlength=m_)[s[sel]] - own
                            okk = den >= region_min
                            e_[okk] = num[okk] / den[okk]
                        expR[sel] = e_
            if tiles_out is not None or types_out is not None:
                ci = np.where(chg)[0]
                tk_u, tk_inv = np.unique(tkey, return_inverse=True)
                nT = len(tk_u)
                cnt = lambda m, w=None: np.bincount(tk_inv[m] if w is None else tk_inv[m], weights=w, minlength=nT)
                if tiles_out is not None:
                    hc = np.zeros(nT); ec = np.zeros(nT); hf = np.zeros(nT); ef = np.zeros(nT); er = np.zeros(nT)
                    if len(ci):
                        hc = np.bincount(tk_inv[ci], weights=C[ci, ct].astype(float), minlength=nT)
                        ec = np.bincount(tk_inv[ci], weights=pC[cr, ct], minlength=nT)
                        hf = np.bincount(tk_inv[ci], weights=Fo[ci, ct].astype(float), minlength=nT)
                        ef = np.bincount(tk_inv[ci], weights=pF[cr, ct], minlength=nT)
                        er = np.bincount(tk_inv[ci], weights=np.nan_to_num(expR[ci]), minlength=nT)
                    nc, ncg = cnt(el), cnt(chg)
                    for k_ in np.flatnonzero(nc > 0):
                        row_ = {"sample": sample, "level": lv, "comparison": f"{T}|{Rr}", "tile": int(tk_u[k_]),
                                "cells": int(nc[k_]), "changed": int(ncg[k_]), "hit_contact": float(hc[k_]),
                                "exp_contact": float(ec[k_]), "hit_far_only": float(hf[k_]),
                                "exp_far_only": float(ef[k_])}
                        if region is not None:
                            row_["hit_contact_reg"], row_["exp_contact_reg"] = float(hc[k_]), float(er[k_])
                        tiles_out.append(row_)
                if types_out is not None:
                    for ai in range(K):
                        ca = el & (ri == ai)
                        if not ca.any():
                            continue
                        na_, ka_ = cnt(ca), cnt(ca & (ti == ai))
                        for k_ in np.flatnonzero(na_ > 0):
                            types_out.append({"sample": sample, "level": lv, "comparison": f"{T}|{Rr}",
                                              "tile": int(tk_u[k_]), "type": types[ai], "cells": int(na_[k_]),
                                              "kept": int(ka_[k_])})
            obsC = C[np.where(chg)[0], ct].mean() if chg.any() else np.nan
            expC = pC[cr, ct].mean() if chg.any() else np.nan
            obsF = Fo[np.where(chg)[0], ct].mean() if chg.any() else np.nan
            expF = pF[cr, ct].mean() if chg.any() else np.nan
            row_ = {"sample": sample, "level": lv, "comparison": f"{T}|{Rr}", "cells": int(el.sum()),
                    "change_rate": chg.sum() / max(el.sum(), 1), "changed": int(chg.sum()),
                    "obs_contact": obsC, "exp_contact": expC, "ratio_contact": obsC / expC if expC else np.nan,
                    "obs_far_only": obsF, "exp_far_only": expF,
                    "ratio_far_only": obsF / expF if expF else np.nan}
            if region is not None:
                expCr = float(np.nanmean(expR[chg])) if chg.any() else np.nan
                row_.update(obs_contact_reg=obsC, exp_contact_reg=expCr,
                            ratio_contact_reg=obsC / expCr if expCr else np.nan)
            summ_rows.append(row_)
    return pair_rows, summ_rows


def null_tables(P: pd.DataFrame, S: pd.DataFrame, null: str = "nucB|nucA"):
    """Tests sharing the null's reference arm, read against the null: (did, did_pairs) or (None, None)."""
    if not len(S) or null not in set(S.comparison):
        return None, None
    ref_arm = null.split("|")[1]
    rows = []
    for (smp, lv), g in S.groupby(["sample", "level"]):
        nl = g[g.comparison == null]
        if nl.empty:
            continue
        nl = nl.iloc[0]
        for _, t in g[(g.comparison != null) & g.comparison.str.endswith("|" + ref_arm)].iterrows():
            rows.append({"sample": smp, "level": lv, "test": t.comparison, "null": null,
                         "change_test": t.change_rate, "change_null": nl.change_rate,
                         "ratio_test": t.ratio_contact, "ratio_null": nl.ratio_contact,
                         "ratio_of_ratios": t.ratio_contact / nl.ratio_contact})
    did = pd.DataFrame(rows)
    pairs = None
    if len(P):
        W = P.pivot_table(index=["sample", "level", "comparison", "a", "b"], columns="stratum",
                          values=["rate", "n"], aggfunc="first")
        W.columns = [f"{x}_{y}" for x, y in W.columns]
        W = W.reset_index()
        for c in ("rate_contact", "rate_absent", "rate_far_only", "n_contact", "n_absent", "n_far_only"):
            if c not in W:              # a stratum no cell fell into (e.g. touching boundaries leave no 'absent')
                W[c] = np.nan
        W["contrast"] = W["rate_contact"] - W["rate_absent"]
        nul = W[W.comparison == null][["sample", "level", "a", "b", "contrast"]].rename(columns={"contrast": "c_null"})
        tst = W[(W.comparison != null) & W.comparison.str.endswith("|" + ref_arm)]
        pairs = tst.merge(nul, on=["sample", "level", "a", "b"], how="inner")
        pairs["excess"] = pairs.contrast - pairs.c_null
    return did, pairs


def compare(labels_dir, out, levels, comparisons=COMPARE_DEFAULT, null: str = "nucB|nucA", samples: str = "",
            verbose: bool = True, contact: str = "knn", boundaries=None, tol: float = 1.0, **kw):
    """All labels files of a folder -> OUT/pairs.csv, summary.csv, did.csv, did_pairs.csv (as bilous_compare.py).
    contact="boundary" reads <sample>_cell_boundaries.parquet from `boundaries` (contacts.find_boundaries)
    and counts cells whose boundaries come within `tol` um as contacts; settings.json records the options."""
    src, out = Path(labels_dir).expanduser(), Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; choose a new output folder")
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.labels.csv.gz"))
    if samples:
        keys = samples.split(",")
        files = [f for f in files if any(k in f.name for k in keys)]
    comps = [tuple(c.split("|")) for c in comparisons.split(",")] if isinstance(comparisons, str) else comparisons
    pair_rows, summ_rows, tile_rows, type_rows = [], [], [], []
    if contact not in ("knn", "boundary"):
        raise ValueError(f"contact must be 'knn' or 'boundary', not {contact!r}")
    if contact == "boundary" and not boundaries:
        raise ValueError("contact='boundary' needs the folder of cell_boundaries.parquet files (boundaries)")
    (out / "settings.json").write_text(json.dumps({"labels": str(src), "levels": list(levels), "null": null,
                                                   "comparisons": [f"{a}|{b}" for a, b in comps], "contact": contact,
                                                   "boundaries": str(boundaries) if boundaries else None, "tol": tol,
                                                   **{k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}},
                                                  indent=1, default=str))
    edge_rows = []
    for f in files:
        D = pd.read_csv(f, keep_default_na=False, low_memory=False)
        if contact == "boundary":
            from .contacts import boundary_edges, find_boundaries, read_parquet_any
            smp = f.name.replace(".labels.csv.gz", "")
            bp = find_boundaries(Path(boundaries).expanduser(), smp)
            if bp is None:
                raise FileNotFoundError(f"no cell_boundaries.parquet for {smp} in {boundaries}")
            E = boundary_edges(read_parquet_any(bp), D["cell_id"].astype(str).to_numpy(), tol)
            deg = np.bincount(E.ravel(), minlength=len(D))
            edge_rows.append({"sample": smp, "cells": len(D), "edges": len(E), "touching_median": float(np.median(deg)),
                              "no_contact_share": float(np.mean(deg == 0))})
            if verbose:
                print(f"{smp}: {len(E):,} boundary contacts (tol {tol} um), median {np.median(deg):.0f} per cell, "
                      f"{np.mean(deg == 0):.3f} of cells touch none", flush=True)
            kw_ = dict(kw, edges=E)
        else:
            kw_ = kw
        p, s = compare_section(D, levels, comps, tiles_out=tile_rows, types_out=type_rows, **kw_)
        pair_rows += p
        summ_rows += s
        if verbose:
            for r in s:
                print(f"{r['sample']} {r['level']} {r['comparison']}: cells {r['cells']:,}, changed "
                      f"{r['change_rate']:.3f}; contact x{r['ratio_contact']:.2f}, far only x{r['ratio_far_only']:.2f}",
                      flush=True)
    P, S = pd.DataFrame(pair_rows), pd.DataFrame(summ_rows)
    P.to_csv(out / "pairs.csv", index=False)
    did, dpairs = null_tables(P, S, null)
    if did is not None:
        did.to_csv(out / "did.csv", index=False)
    if dpairs is not None:
        dpairs.to_csv(out / "did_pairs.csv", index=False)
    S.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(tile_rows).to_csv(out / "tiles.csv.gz", index=False, compression="gzip")
    pd.DataFrame(type_rows).to_csv(out / "tiles_types.csv.gz", index=False, compression="gzip")
    if edge_rows:
        pd.DataFrame(edge_rows).to_csv(out / "boundary_contacts.csv", index=False)
    return P, S, did, dpairs


# --------------------------------------------------------------------------- consequence for colocalization

def coloc(lab, idx, K: int):
    """log2 observed / expected kNN colocalization of all ordered pairs; lab: int codes. Returns (matrix, counts)."""
    n = len(lab)
    cnt = np.bincount(lab, minlength=K).astype(float)
    nb = lab[idx]
    M = np.zeros((K, K))
    for j in range(idx.shape[1]):
        np.add.at(M, (lab, nb[:, j]), 1)
    O = M / np.maximum(cnt[:, None], 1)
    # expected type-b neighbors of a type-a cell: k (n_b - [a == b]) / (n - 1); the cell itself is left out of
    # its own type's count (k n_b / (n - 1) for every pair would slightly understate same-type enrichment)
    E = idx.shape[1] * (cnt[None, :] - np.eye(K)) / max(n - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(O / E), cnt


def consequence_section(D: pd.DataFrame, level: str, ref_arm: str, test_arm: str, rng, k: int = 10, draws: int = 20,
                        min_cells: int = 200):
    """Colocalization from the reference arm, the test arm and random relabeling of the reference labels with the
    section's own confusion P(test | ref), independently of position (draws averaged). Rows per ordered pair."""
    c_ref, c_test = f"{level}_{ref_arm}", f"{level}_{test_arm}"
    D = D[(D[c_ref] != "") & (D[c_test] != "")].reset_index(drop=True)
    sample = D["sample"].iloc[0]
    types = np.array(sorted(set(D[c_ref]) | set(D[c_test])))
    code = {t: i for i, t in enumerate(types)}
    r, t = D[c_ref].map(code).to_numpy(), D[c_test].map(code).to_numpy()
    K = len(types)
    xy = D[["x", "y"]].to_numpy(float)
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = idx[:, 1:]
    C = np.zeros((K, K))
    np.add.at(C, (r, t), 1)
    cum = np.cumsum(C / np.maximum(C.sum(1, keepdims=True), 1), 1)
    cr, cnt_r = coloc(r, idx, K)
    ct, _ = coloc(t, idx, K)
    acc = np.zeros((K, K))
    for _ in range(draws):
        u = rng.random(len(r))
        tl = np.minimum((u[:, None] > cum[r]).sum(1), K - 1)
        acc += coloc(tl, idx, K)[0]
    cl = acc / draws
    dis = float(np.mean(r != t))
    rows = []
    for i in range(K):
        for j in range(K):
            if cnt_r[i] < min_cells or cnt_r[j] < min_cells:
                continue
            rows.append(dict(sample=sample, a=types[i], b=types[j], coloc_ref=cr[i, j], coloc_test=ct[i, j],
                             coloc_typelevel=cl[i, j], n_a_ref=int(cnt_r[i]), n_b_ref=int(cnt_r[j]), disagreement=dis))
    return rows, len(D), dis


def consequence(labels_dir, out, level: str, ref_arm: str = "nucleus", test_arm: str = "whole", k: int = 10,
                draws: int = 20, min_cells: int = 200, seed: int = 0, verbose: bool = True):
    """All labels files of a folder -> OUT/coloc.csv and summary.csv (as bilous_consequence.py)."""
    out = Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    cols = ["sample", "x", "y", f"{level}_{ref_arm}", f"{level}_{test_arm}"]
    for f in sorted(Path(labels_dir).expanduser().glob("*.labels.csv.gz")):
        D = pd.read_csv(f, usecols=cols, keep_default_na=False)
        rr, n, dis = consequence_section(D, level, ref_arm, test_arm, rng, k, draws, min_cells)
        rows += rr
        if verbose:
            print(f"{f.name}: {n:,} cells, disagreement {dis:.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "coloc.csv", index=False)
    R["d_test"] = R.coloc_test - R.coloc_ref
    R["d_type"] = R.coloc_typelevel - R.coloc_ref
    S = R.groupby(["a", "b"]).agg(sections=("d_test", "size"), d_test=("d_test", "median"),
                                  d_type=("d_type", "median"),
                                  test_below_type=("d_test", lambda v: int((v < R.loc[v.index, "d_type"]).sum()))
                                  ).reset_index()
    S.to_csv(out / "summary.csv", index=False)
    return R, S


# --------------------------------------------------------------------------- summary over sections and units

def extra_ratio(o_t, e_t, r_t, o_0, e_0, r_0):
    """Observed over expected share of a contact (or far-only) neighbor's type among the label changes a test makes
    beyond the null: (o_T r_T - o_0 r_0) / (e_T r_T - e_0 r_0) (Methods)."""
    den = e_t * r_t - e_0 * r_0
    return (o_t * r_t - o_0 * r_0) / den if den > 0 else np.nan


def extra_ratio_draws(t, n0, which: str = "contact", draws: int = 0, rng=None):
    """Numerator and denominator of the extra-change ratio of one section, in cells ((o_T r_T - o_0 r_0) n and
    (e_T r_T - e_0 r_0) n, n = the smaller cell count), and with draws > 0 their parametric bootstrap: label changes
    drawn binomially from the cells (rate r) and changes landing on a contact (or far-only) neighbor's type drawn
    binomially from the changes (share o); the expected shares e, which come from the neighbor layout, are held fixed.
    Returns (num, den, num_draws, den_draws)."""
    n = float(min(t.cells, n0.cells))
    o_t, e_t, o_0, e_0 = (float(x) for x in (t[f"obs_{which}"], t[f"exp_{which}"], n0[f"obs_{which}"], n0[f"exp_{which}"]))
    r_t, r_0 = float(t.change_rate), float(n0.change_rate)
    if not all(np.isfinite([o_t, e_t, o_0, e_0, r_t, r_0])):
        return np.nan, np.nan, None, None
    num, den = (o_t * r_t - o_0 * r_0) * n, (e_t * r_t - e_0 * r_0) * n
    if not draws:
        return num, den, None, None
    rng = rng or np.random.default_rng(0)
    ct = rng.binomial(int(t.cells), min(max(r_t, 0), 1), draws)
    c0 = rng.binomial(int(n0.cells), min(max(r_0, 0), 1), draws)
    ht = rng.binomial(ct, min(max(o_t, 0), 1))
    h0 = rng.binomial(c0, min(max(o_0, 0), 1))
    rt, r0 = ct / max(t.cells, 1), c0 / max(n0.cells, 1)
    return num, den, (ht / max(t.cells, 1) - h0 / max(n0.cells, 1)) * n, (e_t * rt - e_0 * r0) * n


def unit_of(sample: str, pattern: str = ""):
    """Unit (donor, patient) of a section from its sample name: the named group `unit` of the regex, else the whole
    match; the sample itself when the pattern is empty or does not match. The default '' gives one unit per
    section."""
    m = re.search(pattern, str(sample)) if pattern else None
    if not m:
        return str(sample)
    return m.group("unit") if "unit" in m.groupdict() else m.group(0)


def unit_note(samples, pattern: str = "") -> str:
    """a warning when units come out as sections: no pattern given, or a pattern some samples do not match."""
    samples = [str(x) for x in samples]
    if not samples:
        return ""
    if not pattern:
        return (f"every section is its own unit ({len(samples)} sections): pass --unit-pattern to pool the sections "
                f"of a donor or patient")
    miss = [x for x in samples if not re.search(pattern, x)]
    if miss:
        return (f"--unit-pattern {pattern!r} does not match {len(miss)} of {len(samples)} sample names (e.g. "
                f"{miss[0]}); those sections are their own units")
    return ""


def _block_weights(n_tiles: int, draws: int, rng):
    """draws x n_tiles resampling counts (tiles drawn with replacement, as many as there are)."""
    return rng.multinomial(n_tiles, np.full(n_tiles, 1.0 / n_tiles), size=draws).astype(float)


def block_bootstrap(cdir, test: str, null: str, level: str | None, samples, unit_pattern: str, draws: int = 1000,
                    seed: int = 0, focus: str = r"(?i)(?:^|[^a-z])(?:t|nk|tnk)(?:[^a-z]|$)"):
    """Block bootstrap over square tiles (tiles.csv.gz / tiles_types.csv.gz of `compare`, 200 um by default):
    tiles of all sections of a unit are drawn with replacement, so spatially correlated label changes move together;
    the parametric bootstrap of `summarise` treats cells as independent. Returns (per unit: pooled extra-change ratio
    from the tile sums and its 95% interval, contact and far only; per unit and focal type: kept ratio of the test
    over the null, (kept_T / cells_T) / (kept_0 / cells_0) on cells whose reference arm gives the type, and its
    interval). Units with fewer than 10 tiles get no interval."""
    cdir = Path(cdir)
    out_u, out_k = [], []
    if not (cdir / "tiles.csv.gz").exists():
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    Tl = pd.read_csv(cdir / "tiles.csv.gz")
    if len(Tl):
        Tl = Tl[Tl.comparison.isin([test, null]) & Tl["sample"].isin(samples)]
        if level:
            Tl = Tl[Tl.level == level]
    stats = ["cells", "changed", "hit_contact", "exp_contact", "hit_far_only", "exp_far_only"]
    if len(Tl) and "exp_contact_reg" in Tl:                 # regionally matched expectation
        stats += ["hit_contact_reg", "exp_contact_reg"]
    ws = ("contact", "far_only") + (("contact_reg",) if "exp_contact_reg" in stats else ())
    if len(Tl):
        W = Tl.pivot_table(index=["level", "sample", "tile"], columns="comparison", values=stats, aggfunc="sum",
                           fill_value=0)
        W.columns = [f"{x}|{c}" for x, c in W.columns]
        W = W.reset_index()
        for x in stats:
            for c in (test, null):
                if f"{x}|{c}" not in W:
                    W[f"{x}|{c}"] = 0.0
        W["unit"] = W["sample"].map(lambda v: unit_of(v, unit_pattern))
        for (lv, u), g in W.groupby(["level", "unit"]):
            sec = pd.factorize(g["sample"])[0]
            nS, nT = sec.max() + 1, len(g)
            G = sp.csr_matrix((np.ones(nT), (np.arange(nT), sec)), shape=(nT, nS))
            Wt = np.vstack([np.ones(nT), _block_weights(nT, draws, rng)]) if nT >= 10 else np.ones((1, nT))
            agg = {k: np.asarray(G.T @ (Wt * g[k].to_numpy(float)[None, :]).T).T for k in g.columns if "|" in k}
            row = {"unit": u, "level": lv, "tiles": nT}
            Nt, N0 = agg[f"cells|{test}"], agg[f"cells|{null}"]
            n = np.minimum(Nt, N0)
            with np.errstate(divide="ignore", invalid="ignore"):
                for w in ws:
                    num = (agg[f"hit_{w}|{test}"] / Nt - agg[f"hit_{w}|{null}"] / N0) * n
                    den = (agg[f"exp_{w}|{test}"] / Nt - agg[f"exp_{w}|{null}"] / N0) * n
                    num, den = np.nansum(num, 1), np.nansum(den, 1)
                    row[f"block_{w}"] = num[0] / den[0] if den[0] > 0 else np.nan
                    if len(num) > 1:
                        ok = den[1:] > 0
                        rr = num[1:][ok] / den[1:][ok]
                        lo, hi = np.percentile(rr, [2.5, 97.5]) if ok.mean() > 0.9 else (np.nan, np.nan)
                    else:
                        lo = hi = np.nan
                    row[f"block_{w}_lo"], row[f"block_{w}_hi"] = lo, hi
            row["block_ci_excludes_1"] = bool(np.isfinite(row["block_contact_lo"]) and row["block_contact_lo"] > 1)
            out_u.append(row)
    if (cdir / "tiles_types.csv.gz").exists():
        Ty = pd.read_csv(cdir / "tiles_types.csv.gz")
        if len(Ty):
            rx = re.compile(focus)
            Ty = Ty[Ty.comparison.isin([test, null]) & Ty["sample"].isin(samples)
                    & Ty["type"].astype(str).map(lambda v: bool(rx.search(v)))]
            if level:
                Ty = Ty[Ty.level == level]
        if len(Ty):
            V = Ty.pivot_table(index=["level", "type", "sample", "tile"], columns="comparison",
                               values=["cells", "kept"], aggfunc="sum", fill_value=0)
            V.columns = [f"{x}|{c}" for x, c in V.columns]
            V = V.reset_index()
            for x in ("cells", "kept"):
                for c in (test, null):
                    if f"{x}|{c}" not in V:
                        V[f"{x}|{c}"] = 0.0
            V["unit"] = V["sample"].map(lambda v: unit_of(v, unit_pattern))
            for (lv, t, u), g in V.groupby(["level", "type", "unit"]):
                nT = len(g)
                Wt = np.vstack([np.ones(nT), _block_weights(nT, draws, rng)]) if nT >= 10 else np.ones((1, nT))
                s_ = {k: Wt @ g[k].to_numpy(float) for k in g.columns if "|" in k}
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = (s_[f"kept|{test}"] / s_[f"cells|{test}"]) / (s_[f"kept|{null}"] / s_[f"cells|{null}"])
                fin = r[1:][np.isfinite(r[1:])]
                lo, hi = (np.percentile(fin, [2.5, 97.5]) if len(r) > 1 and len(fin) > 0.9 * (len(r) - 1)
                          else (np.nan, np.nan))
                out_k.append({"unit": u, "level": lv, "type": t, "tiles": nT, "cells_test": int(s_[f"cells|{test}"][0]),
                              "cells_null": int(s_[f"cells|{null}"][0]), "kept_ratio": r[0], "kept_ratio_lo": lo,
                              "kept_ratio_hi": hi})
    return pd.DataFrame(out_u), pd.DataFrame(out_k)


def summarise(compare_dir, out, conseq_dir=None, test: str = "wholeB|nucA", null: str = "nucB|nucA",
              level: str | None = None, min_null_cells: int = 1000, min_stratum: int = 50,
              unit_pattern: str = "", top: int = 10, min_extra: int = 50, draws: int = 2000, seed: int = 0,
              block_draws: int = 1000, focus: str = r"(?i)(?:^|[^a-z])(?:t|nk|tnk)(?:[^a-z]|$)"):
    """Per section and per unit (patient; `unit_pattern` applied to the sample name): the extra-change ratio of the
    test over the null (contact and far only), per-pair excess (contact minus absent, minus the null's) with the
    reverse pair, and, with conseq_dir, the colocalization change of real against random errors. Writes
    sections.csv, units.csv, pairs.csv, pairs_units.csv and (with conseq_dir) coloc_pairs.csv into OUT; returns the
    tables.

    The extra-change ratio is a ratio of differences and is unstable when the test changes few more labels than the
    null. sections.csv therefore also gives the number of extra changes and `extra_ok` (at least min_extra);
    units.csv gives, per unit, the pooled ratio (sum of numerators over sum of denominators of its used sections, in
    cells) with a 95% parametric-bootstrap interval (draws), next to the median of the sections with extra_ok. The
    pooled ratio is the one to report per unit. A unit is evaluable only if its denominator (extra changes
    expected at contact under indifference, `expected_extra_contact`) is stable, i.e. positive in more than 90% of
    the draws; `ci_excludes_1` marks evaluable units whose interval lies above one."""
    cdir, out = Path(compare_dir).expanduser(), Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(cdir / "summary.csv")
    if level:
        S = S[S.level == level]
    rows, parts = [], {}
    rng = np.random.default_rng(seed)
    for (smp, lv), g in S.groupby(["sample", "level"]):
        t, n0 = g[g.comparison == test], g[g.comparison == null]
        if t.empty or n0.empty:
            continue
        t, n0 = t.iloc[0], n0.iloc[0]
        ws = ("contact", "far_only") + (("contact_reg",) if "exp_contact_reg" in S else ())
        parts[(smp, lv)] = {w: extra_ratio_draws(t, n0, w, draws, rng) for w in ws}
        n_extra = (float(t.change_rate) - float(n0.change_rate)) * min(float(t.cells), float(n0.cells))
        rows.append({"sample": smp, "unit": unit_of(smp, unit_pattern), "level": lv, "null_cells": int(n0.cells),
                     "extra_changes": int(round(n_extra)) if np.isfinite(n_extra) else np.nan,
                     "change_test": t.change_rate, "change_null": n0.change_rate,
                     "extra_ratio_contact": extra_ratio(t.obs_contact, t.exp_contact, t.change_rate,
                                                        n0.obs_contact, n0.exp_contact, n0.change_rate),
                     "extra_ratio_far_only": extra_ratio(t.obs_far_only, t.exp_far_only, t.change_rate,
                                                         n0.obs_far_only, n0.exp_far_only, n0.change_rate),
                     "ratio_test": t.ratio_contact, "ratio_null": n0.ratio_contact,
                     **({"extra_ratio_contact_reg": extra_ratio(t.obs_contact_reg, t.exp_contact_reg, t.change_rate,
                                                                n0.obs_contact_reg, n0.exp_contact_reg,
                                                                n0.change_rate)} if "exp_contact_reg" in S else {})})
    sec = pd.DataFrame(rows)
    note = unit_note(sorted(set(sec["sample"])) if len(sec) else [], unit_pattern)
    if note:
        print(f"WARNING: {note}", flush=True)
    sec["used"] = sec.null_cells >= min_null_cells if len(sec) else []
    if len(sec):
        # the extra-change ratio needs the test to change more labels than the null; a full-depth test against the
        # half-depth null often does not, and then the ratio is undefined (NaN), not evidence either way
        sec["test_changes_more"] = sec.change_test > sec.change_null
        sec["extra_ok"] = sec.extra_changes >= min_extra
    sec.to_csv(out / "sections.csv", index=False)
    used = set(sec.loc[sec.used, "sample"]) if len(sec) else set()

    # per unit: pooled extra-change ratio with a bootstrap interval, and the median of sections with enough changes
    urows = []
    if len(sec):
        for (u, lv), g in sec[sec.used].groupby(["unit", "level"]):
            row = {"unit": u, "level": lv, "sections": len(g), "sections_extra_ok": int(g.extra_ok.sum()),
                   "extra_changes": int(np.nansum(g.extra_changes))}
            ok_draws = {}
            for w in ("contact", "far_only") + (("contact_reg",) if "extra_ratio_contact_reg" in sec else ()):
                pp = [parts[(smp, lv)][w] for smp in g["sample"]]
                pp = [x for x in pp if np.isfinite(x[0]) and np.isfinite(x[1])]
                num, den = sum(x[0] for x in pp), sum(x[1] for x in pp)
                row[f"pooled_{w}"] = num / den if pp and den > 0 else np.nan
                # the denominator is the number of extra changes expected at a contact (far-only) neighbor's
                # type if changes ignored the neighbors; the ratio is unstable when it is small, whatever the number
                # of extra changes (RA p06: 365 extra changes, 22.1)
                row[f"expected_extra_{w}"] = den if pp else np.nan
                if draws and pp and all(x[2] is not None for x in pp):
                    nd, dd = sum(x[2] for x in pp), sum(x[3] for x in pp)
                    ok = dd > 0
                    rr = nd[ok] / dd[ok]
                    row[f"pooled_{w}_lo"], row[f"pooled_{w}_hi"] = (np.percentile(rr, [2.5, 97.5]) if ok.mean() > 0.9
                                                                    else (np.nan, np.nan))
                    ok_draws[w] = float(ok.mean())
                row[f"median_{w}_extra_ok"] = g.loc[g.extra_ok, f"extra_ratio_{w}"].median()
            # evaluable = enough extra changes AND a stable denominator: positive in more than 90% of the
            # bootstrap draws (finite interval); without draws, at least min_extra / 4 expected extra contact changes
            stable = (np.isfinite(row.get("pooled_contact_lo", np.nan)) if draws
                      else row["expected_extra_contact"] >= min_extra / 4)
            row["evaluable"] = bool(row["extra_changes"] >= min_extra and np.isfinite(row["pooled_contact"]) and stable)
            row["ci_excludes_1"] = bool(row["evaluable"] and draws and row["pooled_contact_lo"] > 1)
            urows.append(row)
    U = pd.DataFrame(urows)
    # block bootstrap over tiles (compare folders with tile sums)
    BU, BK = (block_bootstrap(cdir, test, null, level, used, unit_pattern, block_draws, seed, focus)
              if block_draws and len(sec) else (pd.DataFrame(), pd.DataFrame()))
    if len(BU) and len(U):
        U = U.merge(BU, on=["unit", "level"], how="left")
    U.to_csv(out / "units.csv", index=False)
    if len(BK):
        BK.to_csv(out / "units_kept.csv", index=False)

    dp = cdir / "did_pairs.csv"
    P = pd.read_csv(dp) if dp.exists() else pd.DataFrame(columns=["comparison"])
    if test not in set(P.comparison):
        # the test does not share the null's reference arm (e.g. author|nucleus against nucB|nucA): no null to subtract;
        # the pair excess is the test's own contrast (rate at contact minus rate with b absent), c_null left empty
        R = pd.read_csv(cdir / "pairs.csv")
        R = R[R.comparison == test]
        if len(R):
            W = R.pivot_table(index=["sample", "level", "comparison", "a", "b"], columns="stratum",
                              values=["rate", "n"], aggfunc="first")
            W.columns = [f"{x}_{y}" for x, y in W.columns]
            P = W.reset_index()
            P["contrast"] = P.get("rate_contact") - P.get("rate_absent")
            P["c_null"] = np.nan
            P["excess"] = P.contrast
    P = P[(P.comparison == test) & P["sample"].isin(used)]
    if level:
        P = P[P.level == level]
    P = P[(P.get("n_contact", 0) >= min_stratum) & (P.get("n_absent", 0) >= min_stratum)].copy()
    P["unit"] = P["sample"].map(lambda s: unit_of(s, unit_pattern))
    pr = P.groupby(["level", "a", "b"]).agg(sections=("excess", "size"), excess_median=("excess", "median"),
                                            sections_pos=("excess", lambda v: int((v > 0).sum())),
                                            rate_contact=("rate_contact", "median"),
                                            rate_absent=("rate_absent", "median"),
                                            null_contrast=("c_null", "median")).reset_index()
    rev = pr.rename(columns={"a": "b", "b": "a", "excess_median": "reverse_excess",
                             "sections_pos": "reverse_pos", "sections": "reverse_sections"})[
        ["level", "a", "b", "reverse_excess", "reverse_pos", "reverse_sections"]]
    pr = pr.merge(rev, on=["level", "a", "b"], how="left").sort_values("excess_median", ascending=False)
    pr.to_csv(out / "pairs.csv", index=False)
    pu = (P.groupby(["level", "a", "b", "unit"]).excess.median().reset_index()
          .groupby(["level", "a", "b"]).agg(units=("excess", "size"), unit_median=("excess", "median"),
                                            units_pos=("excess", lambda v: int((v > 0).sum()))).reset_index()
          .sort_values("unit_median", ascending=False))
    pu.to_csv(out / "pairs_units.csv", index=False)
    res = {"sections": sec, "units": U, "pairs": pr, "pairs_units": pu, "units_kept": BK}

    if conseq_dir:
        R = pd.read_csv(Path(conseq_dir).expanduser() / "coloc.csv")
        R = R[R["sample"].isin(used)].copy() if used else R.iloc[0:0].copy()
        R["unit"] = R["sample"].map(lambda s: unit_of(s, unit_pattern))
        R["d_test"] = R.coloc_test - R.coloc_ref
        R["d_type"] = R.coloc_typelevel - R.coloc_ref
        R = R[np.isfinite(R.d_test) & np.isfinite(R.d_type)]
        U = R.groupby(["a", "b", "unit"])[["d_test", "d_type"]].median().reset_index()
        cp = R.groupby(["a", "b"]).agg(sections=("d_test", "size"), d_test=("d_test", "median"),
                                       d_type=("d_type", "median"),
                                       sections_test_below=("d_test", lambda v: int((v < R.loc[v.index, "d_type"]).sum()))
                                       ).reset_index()
        cu = U.groupby(["a", "b"]).agg(units=("d_test", "size"), unit_d_test=("d_test", "median"),
                                       unit_d_type=("d_type", "median"),
                                       units_test_below=("d_test", lambda v: int((v < U.loc[v.index, "d_type"]).sum()))
                                       ).reset_index()
        cp = cp.merge(cu, on=["a", "b"], how="left")
        cp.to_csv(out / "coloc_pairs.csv", index=False)
        res["coloc_pairs"] = cp
    return res
