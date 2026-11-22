"""Plant known truth into a real tissue.

null        permute whole count profiles among cells with the same reference label in the same section: keeps
            positions, composition, labels and library sizes, removes every neighborhood effect beyond identity.
spike       raise (or lower) a gene set in receiver cells as a function of distance to the nearest sender cell,
            f = exp(-d / lam); fold change exp(effect * f). Raising genes adds transcripts, so library size rises
            with them; after library-size normalisation every other gene of those cells falls slightly. That
            compositional shift is a real property of normalized analyses; the truth record lists only the
            planted genes, so large effects on many genes will show it as down-called "false positives".
shift       move receiver cells near senders part of the way toward a partner type's mean profile (a state change
            that a labeler may read as a different type: tests false negatives through relabeling).

Each function returns (new tissue, truth record). Truth records are collected in tissue.meta["planted"].
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .tissue import Tissue, _binom_sparse


def _add_planted(t: Tissue, rec: dict) -> dict:
    meta = dict(t.meta)
    meta["planted"] = list(meta.get("planted", [])) + [rec]
    return meta


def null(t: Tissue, rng, within=("section", "labels")) -> tuple[Tissue, dict]:
    keys = [getattr(t, w) for w in within]
    grp = np.unique(np.vstack(keys).T.astype(str), axis=0, return_inverse=True)[1].ravel()
    perm = np.arange(t.n)
    for g in np.unique(grp):
        ii = np.flatnonzero(grp == g)
        perm[ii] = rng.permutation(ii)
    rec = {"kind": "null", "within": list(within)}
    return t.replace(counts=t.counts[perm], meta=_add_planted(t, rec)), rec


def distance_to_type(t: Tissue, cells: np.ndarray, target_type: str, labels=None) -> np.ndarray:
    """Distance from each of `cells` to the nearest cell of target_type in the same section (inf if none)."""
    from scipy.spatial import cKDTree
    lab = t.labels if labels is None else labels
    d = np.full(len(cells), np.inf)
    for s in np.unique(t.section[cells]):
        tgt = np.flatnonzero((t.section == s) & (lab == target_type))
        if len(tgt) == 0:
            continue
        m = t.section[cells] == s
        d[m] = cKDTree(t.xy[tgt]).query(t.xy[cells[m]])[0]
    return d


def _type_means(t: Tissue):
    return {ty: np.asarray(t.counts[t.labels == ty].mean(0)).ravel() for ty in t.types}


def _candidates(t: Tissue, sender: str | None, receiver: str, rule: str = "all", min_mean: float = 0.2,
                max_ratio: float = 2.0, means=None) -> np.ndarray:
    """Genes eligible for a spike-in (indices).

    rule "all"  mean in receivers >= min_mean, no other type above max_ratio x the
                receiver mean, and the receiver not above max_ratio x every other type.
    rule "pair" mean in receivers >= min_mean, sender and receiver means within max_ratio of each other (the gene
                is not a marker of the sender, so mislabeled or contaminating sender cells cannot fake the
                planted effect), and the receiver not above max_ratio x every other type (not a receiver marker).
                Genes high in some third type are allowed; on a 480-gene panel with 11 types "all" can leave too
                few candidates for receivers with a distinctive profile.
    rule "sender" mean in receivers >= min_mean and sender mean >= max_ratio x the receiver mean: markers of the
                sender, as a truly induced gene that the sender also expresses (tests filters that remove such genes).
    rule "none" mean in receivers >= min_mean only.
    """
    M = means or _type_means(t)
    mean_r = M[receiver]
    ok = mean_r >= min_mean
    others = [M[ty] for ty in t.types if ty != receiver]
    mx = np.max(others, 0) if others else np.zeros_like(mean_r)
    not_receiver_marker = (mx + 1e-3) / (mean_r + 1e-3) > 1 / max_ratio
    if rule == "all":
        ratio = (mx + 1e-3) / (mean_r + 1e-3)
        ok &= (ratio > 1 / max_ratio) & (ratio < max_ratio)
    elif rule == "pair":
        if sender is None or sender not in M:
            raise ValueError(f"marker_rule pair needs a sender label; got {sender}")
        rs = (M[sender] + 1e-3) / (mean_r + 1e-3)
        ok &= (rs > 1 / max_ratio) & (rs < max_ratio) & not_receiver_marker
    elif rule == "sender":
        if sender is None or sender not in M:
            raise ValueError(f"marker_rule sender needs a sender label; got {sender}")
        ok &= (M[sender] + 1e-3) / (mean_r + 1e-3) >= max_ratio
    elif rule != "none":
        raise ValueError(f"marker_rule must be 'all', 'pair', 'sender' or 'none', got {rule}")
    return np.flatnonzero(ok)


def candidate_counts(t: Tissue, sender, receiver, min_mean=0.2, max_ratio=2.0) -> dict:
    """Number of eligible spike-in genes under each marker rule."""
    for ty in (sender, receiver):
        if ty not in t.types:
            raise ValueError(f"spike: label {ty!r} not in the data (labels: {', '.join(map(str, t.types))})")
    M = _type_means(t)
    return {r: int(len(_candidates(t, sender, receiver, r, min_mean, max_ratio, M))) for r in ("all", "pair", "sender", "none")}


def _pick_genes(t: Tissue, genes, receiver: str, rng, exclude_markers=True, min_mean=0.2, sender=None,
                marker_rule=None, max_ratio=2.0):
    if not isinstance(genes, (int, np.integer)):
        return np.asarray([np.flatnonzero(t.genes == g)[0] if isinstance(g, str) else int(g) for g in genes])
    rule = marker_rule or ("all" if exclude_markers else "none")
    cand = _candidates(t, sender, receiver, rule, min_mean, max_ratio)
    if len(cand) < genes:
        n = candidate_counts(t, sender, receiver, min_mean, max_ratio) if sender in t.types else {}
        raise ValueError(f"only {len(cand)} candidate genes for the spike-in {sender} -> {receiver} "
                         f"(marker_rule {rule}, min_mean {min_mean}); candidates by rule: {n}. "
                         "Lower genes or min_mean, or use another marker_rule")
    return np.sort(rng.choice(cand, genes, replace=False))


def spike(t: Tissue, sender: str, receiver: str, genes, effect: float, lam: float, rng,
          exclude_markers=True, marker_rule=None, min_mean=0.2, max_ratio=2.0) -> tuple[Tissue, dict]:
    """Planted neighborhood effect: genes change in receivers by exp(effect * exp(-d/lam)).
    Gene choice (when genes is a number): see _candidates; marker_rule overrides exclude_markers."""
    for ty in (sender, receiver):
        if ty not in t.types:
            raise ValueError(f"spike: label {ty!r} not in the data (labels: {', '.join(map(str, t.types))})")
    rule = marker_rule or ("all" if exclude_markers else "none")
    g = _pick_genes(t, genes, receiver, rng, min_mean=min_mean, sender=sender, marker_rule=rule,
                    max_ratio=max_ratio)
    cells = np.flatnonzero(t.labels == receiver)
    f = np.exp(-distance_to_type(t, cells, sender) / lam)
    fold = np.exp(effect * f)                                   # per receiver cell
    old = t.counts[cells][:, g].toarray()
    if effect >= 0:
        tot = np.asarray(t.counts[cells].sum(1)).ravel()
        prof = old.sum(0) / max(tot.sum(), 1)
        new = old + rng.poisson(tot[:, None] * prof[None, :] * (fold - 1)[:, None])
    else:
        new = rng.binomial(old.astype(np.int64), np.clip(fold, 0, 1)[:, None])
    delta = sp.coo_matrix(new - old)
    D = sp.csr_matrix((delta.data, (cells[delta.row], g[delta.col])), shape=t.counts.shape)
    X = (t.counts + D).tocsr()
    X.eliminate_zeros()
    M = _type_means(t)
    others = [ty for ty in t.types if ty not in (receiver,)]
    prof = {}
    for gi in g:
        top = max(others, key=lambda ty: M[ty][gi]) if others else None
        prof[str(t.genes[gi])] = {"receiver": round(float(M[receiver][gi]), 4),
                                  "sender": round(float(M[sender][gi]), 4),
                                  "top_other": str(top), "top_other_mean": round(float(M[top][gi]), 4) if top else None}
    rec = {"kind": "spike", "sender": sender, "receiver": receiver, "genes": [str(x) for x in t.genes[g]],
           "profiles": prof,
           "effect": float(effect), "lam": float(lam), "sign": int(np.sign(effect)),
           "marker_rule": rule if isinstance(genes, (int, np.integer)) else "given"}
    return t.replace(counts=X, meta=_add_planted(t, rec)), rec


def spike_set(t: Tissue, n_pairs: int, genes: int, effect: float, lam: float, rng, marker_rule="pair",
              min_cells: int = 500, max_per_receiver: int = 3, types=None, min_mean=0.2, max_ratio=2.0):
    """Many planted interactions (the many-pair design): n_pairs (sender, receiver) pairs drawn uniformly among
    labels with >= min_cells cells (optionally restricted to `types`), at most max_per_receiver per receiver, each
    with `genes` genes chosen by marker_rule and disjoint from genes already planted in that receiver. Pairs whose
    candidates run out are skipped and redrawn. Each pair is recorded as an ordinary spike."""
    lab, cnt = np.unique(t.labels, return_counts=True)
    pool = [l for l, c in zip(lab, cnt) if c >= min_cells and (types is None or l in types)]
    used, per_recv, recs, tries = {}, {}, [], 0
    while len(recs) < n_pairs and tries < 50 * n_pairs:
        tries += 1
        s_, r_ = rng.choice(pool, 2, replace=False)
        if per_recv.get(r_, 0) >= max_per_receiver or any(x["sender"] == s_ and x["receiver"] == r_ for x in recs):
            continue
        cand = _candidates(t, s_, r_, marker_rule, min_mean, max_ratio)
        cand = np.setdiff1d(cand, used.get(r_, np.array([], int)))
        if len(cand) < genes:
            continue
        g = np.sort(rng.choice(cand, genes, replace=False))
        t, rec = spike(t, s_, r_, [str(x) for x in t.genes[g]], effect, lam, rng)
        rec["marker_rule"] = marker_rule
        used[r_] = np.union1d(used.get(r_, np.array([], int)), g)
        per_recv[r_] = per_recv.get(r_, 0) + 1
        recs.append(rec)
    if len(recs) < n_pairs:
        raise ValueError(f"spike_set: planted only {len(recs)} of {n_pairs} pairs; lower n_pairs, genes or min_cells")
    return t, {"kind": "spike_set", "n_pairs": len(recs)}


def shift(t: Tissue, sender: str, receiver: str, partner: str, w_max: float, lam: float,
          rng) -> tuple[Tissue, dict]:
    """Receivers near senders take a fraction w = w_max * exp(-d/lam) of their counts from the partner profile."""
    cells = np.flatnonzero(t.labels == receiver)
    w = w_max * np.exp(-distance_to_type(t, cells, sender) / lam)
    tot = np.asarray(t.counts[cells].sum(1)).ravel()
    part = t.labels == partner
    prof = np.asarray(t.counts[part].sum(0)).ravel()
    prof = prof / max(prof.sum(), 1)
    kept = _binom_sparse(t.counts[cells], 1 - w, rng)
    add = sp.csr_matrix(rng.poisson((w * tot)[:, None] * prof[None, :]).astype(np.float64))
    keep_mask = np.ones(t.n)
    keep_mask[cells] = 0
    P = sp.csr_matrix((np.ones(len(cells)), (cells, np.arange(len(cells)))), shape=(t.n, len(cells)))
    X = (sp.diags(keep_mask) @ t.counts + P @ (kept + add)).tocsr()
    X.eliminate_zeros()
    rec = {"kind": "shift", "sender": sender, "receiver": receiver, "partner": partner,
           "w_max": float(w_max), "lam": float(lam)}
    return t.replace(counts=X, meta=_add_planted(t, rec)), rec
