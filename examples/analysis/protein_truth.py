"""Errors, not only disagreements: label error against protein identity on the same cells.

Dataset: Li et al. 2026 (IBD CosMx, Figshare 31595527), one section, 43,034 cells, 1,000 genes. Its published
major classes were gated on the CosMx immunofluorescence stains (Epithelial = PanCK+, Immune = CD45+ PanCK-,
Others = both negative). A CellScape multiplex protein image (about 40 proteins, separate
segmentation) covers the same region. Three subcommands:

  gate       Labels from the transcripts only (nearest centroid on half A of the counts, trained leave-one-fold-out
             over spatial folds of fields of view; flat and natural class priors), mapped to the gated class of
             each published subtype. A change of class is an ERROR against protein. Readouts for those errors, with
             the protein class as reference: error rate and class confusion; destination enrichment (do errors go to
             the class of the cell's neighbors?); the exact contact-versus-far fit (6 nearest neighbors vs ranks
             7-20). Control: class-level errors at the same rate placed at random (type-level). Robustness: the same
             on cells whose stain intensities lie far from the gate (protein-confident cells).
  register   Aligns the CellScape nuclear centroids (pixels) to the CosMx cell centroids (slide µm): orientation
             (8 axis flips/rotations) and translation by cross-correlation of density images, then iterative
             mutual-nearest-neighbor refinement (similarity, then affine), then matching within --max-dist µm.
             Chance level: the same matching after a 100 µm shift.
  cellscape  With a registration: protein classes from the CellScape panel (two-component mixture per marker on
             log intensity; priority rules below), then (i) agreement of the published major labels with the
             CellScape classes, and (ii) the gate readouts with CellScape classes as reference, on matched cells.

CellScape classes (priority order; a cell takes the first that fits): Plasma (CD138+), B (CD20+ or CD19+), T (CD3+),
Myeloid (CD68+ or CD163+), Epithelial (PanCK+ or E-Cadherin+), Endothelial (CD31+), Mesenchymal (SMA+ or
Podoplanin+ or Vimentin+, and CD45-), Immune other (CD45+), else unassigned. Published major labels map to these
classes (T/NK -> T); Neuron has no protein marker in the panel and is left out.

Readouts (written to gate_rules.json, registration.json and cellscape_rules.json):
  E1  gate: destination enrichment of class errors (natural-prior labeler) >= 1.2 with the control in [0.9, 1.1]
      -> errors against protein are spatially organized.
  E2  gate: beta_contact > 0.1 and > 3 SE, far / contact <= 0.5 -> contact-dominated errors.
  E3  gate: E1 on protein-confident cells.
  R0  register: matched share of the smaller point set >= 0.5 within 5 µm and matched minus chance share >= 0.35
      -> same section, registration valid. The search covers rotations and picks the candidate with the largest
      match excess over chance. The chance share at 5 µm depends on cell density (about 0.15 in this tissue even
      without alignment), so a fixed chance ceiling cannot be met by a correct registration of dense tissue; the
      stricter criterion (chance share <= 0.1) is reported beside it as R0_as_written.
  R1  register, local check: per field of view, best local translation (+-40 µm) against the median over
      translations, matches within 2 µm. Median excess >= 0.3 -> same section with local distortion (cell matching
      possible with per-field registration); < 0.1 -> a different (serial) section or unregistrable; else
      undetermined.
  P0  cellscape: CellScape classes agree with the CosMx protein gate (Epithelial / Immune / Others) on >= 0.8 of
      matched cells -> the CellScape gating is credible.
  P1  cellscape: the published major labels' agreement with CellScape classes (no threshold).
  P2  cellscape: E1 with CellScape classes as reference (errors of the natural-prior labeler against CellScape).

  python protein_truth.py gate DIR/li2026_cosmx_cellscape.h5ad --out OUT/gate
  python protein_truth.py register DIR/cosmx_meta.csv "DIR/cells_protein_expression(in).csv" --out OUT/register
  python protein_truth.py cellscape DIR/li2026_cosmx_cellscape.h5ad "DIR/cells_protein_expression(in).csv" \\
      --matches OUT/register/matches.csv --out OUT/cellscape
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

GATE = {"Epithelial": "Epithelial", "Immune": "Immune", "Others": "Others"}
CS_MAP = {"Epithelial": "Epithelial", "Endothelial": "Endothelial", "Mesenchymal": "Mesenchymal",
          "Myeloid": "Myeloid", "T/NK": "T", "B": "B", "Plasma": "Plasma"}


def _coloc():
    p = Path(__file__).resolve().parent / "coloc.py"
    spec = importlib.util.spec_from_file_location("coloc_pt", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------------------------- shared readouts
def spatial_readouts(xy, ref_class, lab_class, P=None, k=10, seed=0):
    """Destination enrichment and the exact contact/far fit of lab_class against ref_class (one section)."""
    from claimbreak.tissue import Tissue
    C = _coloc()
    n = len(ref_class)
    T = Tissue(counts=sp.csr_matrix((n, 1)), xy=np.asarray(xy, float), labels=np.asarray(ref_class).astype(str),
               section=np.repeat("s", n), unit=np.repeat("s", n), genes=np.array(["g"]))
    types = list(T.types)
    lab = np.asarray(lab_class).astype(str)
    if not set(lab) <= set(types):
        raise ValueError(f"labels outside the reference classes: {sorted(set(lab) - set(types))}")
    if P is None:
        P = np.eye(len(types))[[types.index(x) for x in lab]]
    idx, _ = T.graph.knn(k)
    dsum, dtab = C.destination_test(T, lab, P, idx)
    sp_, _ = C.spillover_test(T, lab)
    out = {"error_rate": float(np.mean(lab != T.labels)), "n_cells": int(n)}
    out.update({k_: v for k_, v in dsum.items() if k_.startswith(("dest_", "own_"))})
    out.update(sp_)
    return out, dtab, T


def random_control(T, lab_class, seed=0):
    """Class errors at the realised rate, cells chosen at random, destinations from the realised confusion."""
    from claimbreak import perturb
    types = list(T.types)
    ref = T.labels
    lab = np.asarray(lab_class).astype(str)
    C = pd.crosstab(pd.Categorical(ref, types), pd.Categorical(lab, types), dropna=False).astype(float)
    for a in types:
        C.loc[a, a] = 0.0
        s = C.loc[a].sum()
        C.loc[a] = C.loc[a] / s if s > 0 else 1.0 / max(len(types) - 1, 1)
        C.loc[a, a] = 0.0
    eps = float(np.mean(lab != ref))
    labels, _ = perturb.label_errors(T, eps, C, np.random.default_rng(seed))
    return np.asarray(labels).astype(str)


def verdict_E(r, ctrl):
    org = r["dest_enrichment_all"] >= 1.2 and 0.9 <= ctrl["dest_enrichment_all"] <= 1.1
    fit = r["sp_beta_contact"] > 0.1 and r["sp_beta_contact"] > 3 * r["sp_se_contact"]
    return {"E1": "organised" if org else "not organised",
            "E2": ("contact-dominated" if (fit and r["sp_ratio"] <= 0.5) else
                   "not contact-dominated" if fit else "undetermined")}


# --------------------------------------------------------------------------------------------- labeler
def transcript_labels(A_all, fold, seed=0):
    """Nearest centroid (leave-one-fold-out) on half A, flat and natural priors. Returns {variant: minor labels}."""
    from claimbreak.tissue import split
    from claimbreak.labeling import native_flip, NearestCentroid
    C = _coloc()
    T = A_all.replace(section=np.asarray(fold).astype(str))
    T = T.replace(graph=None)
    T.__post_init__()
    A, _ = split(T, 0.5, np.random.default_rng([seed, 1]))
    flip, _ = native_flip(A, np.random.default_rng([seed, 12]), labeller=NearestCentroid(A), by_type=True)
    own, P = C.nc_posterior(A, list(A.types), float(flip))
    Pc = C.prior_correct(P, C.class_prior(A.labels, list(A.types)))
    types = np.array(list(A.types))
    return {"flat": np.asarray(own).astype(str), "natural": types[np.argmax(Pc, 1)]}, float(flip)


def spatial_folds(xy, fov, n_folds=5, seed=0):
    """Folds of whole fields of view, contiguous along the longer axis (spatially separated training)."""
    f = pd.DataFrame({"fov": fov, "x": xy[:, 0], "y": xy[:, 1]}).groupby("fov")[["x", "y"]].mean()
    ax = "x" if (f.x.max() - f.x.min()) >= (f.y.max() - f.y.min()) else "y"
    order = f.sort_values(ax).index.to_numpy()
    fold_of = {v: i * n_folds // len(order) for i, v in enumerate(order)}
    return np.array([fold_of[v] for v in fov])


def load_cosmx(path, label_key="ct_minor_new", class_key="category", x="x_um", y="y_um", min_cells=50):
    import anndata as ad
    from claimbreak.tissue import Tissue
    A = ad.read_h5ad(path)
    obs = A.obs
    X = A.layers["counts"] if "counts" in A.layers else A.X
    lab = obs[label_key].astype(str).to_numpy()
    vc = pd.Series(lab).value_counts()
    keep = np.isin(lab, vc[vc >= min_cells].index)
    obs = obs[keep]
    T = Tissue(counts=sp.csr_matrix(X)[keep], xy=obs[[x, y]].to_numpy(float), labels=lab[keep],
               section=np.repeat("s", keep.sum()), unit=np.repeat("s", keep.sum()), genes=np.asarray(A.var_names),
               obs_names=np.asarray(obs.index))
    # class of each published subtype (its most frequent gate class)
    m = pd.crosstab(obs[label_key].astype(str), obs[class_key].astype(str))
    sub2class = m.idxmax(1).to_dict()
    return T, obs, sub2class


def confident_mask(obs, q=0.25):
    """Cells whose gating stains lie far from the gate. The gate is taken as the midpoint between the largest
    negative and the smallest positive scaled maximum (or their medians if the classes overlap); a cell is confident
    if its distance to the gate exceeds the q-quantile of distances, for both PanCK and CD45."""
    ok = np.ones(len(obs), bool)
    for m, flag in (("Max.PanCK.scaled", "PanCK_pos"), ("Max.CD45.scaled", "CD45_pos")):
        if m not in obs or flag not in obs:
            continue
        v, f = obs[m].to_numpy(float), obs[flag].astype(bool).to_numpy()
        lo, hi = v[~f].max(), v[f].min()
        t = (lo + hi) / 2 if lo < hi else (np.median(v[~f]) + np.median(v[f])) / 2
        d = np.abs(v - t)
        ok &= d > np.quantile(d, q)
    return ok


def cmd_gate(a):
    T, obs, sub2class = load_cosmx(a.h5ad, min_cells=a.min_cells)
    prot = obs["category"].astype(str).to_numpy()
    fold = spatial_folds(T.xy, obs["fov"].to_numpy(), a.folds)
    labs, flip = transcript_labels(T, fold)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conf = confident_mask(obs)
    rows, verd = [], {}
    for v, lab in labs.items():
        cls = np.array([sub2class.get(x, "unmapped") for x in lab])
        r, dtab, Tsp = spatial_readouts(T.xy, prot, cls)
        ctrl, _, _ = spatial_readouts(T.xy, prot, random_control(Tsp, cls))
        rc, _, _ = spatial_readouts(T.xy[conf], prot[conf], cls[conf])
        rows += [dict(variant=v, subset="all", arm="labeller", **r), dict(variant=v, subset="all", arm="random", **ctrl),
                 dict(variant=v, subset="protein_confident", arm="labeller", **rc)]
        dtab.to_csv(out / f"destination_{v}.csv", index=False, float_format="%.4g")
        pd.crosstab(pd.Series(prot, name="protein"), pd.Series(cls, name="labeller")).to_csv(out / f"confusion_{v}.csv")
        verd[v] = verdict_E(r, ctrl)
        verd[v]["E3"] = "holds" if rc["dest_enrichment_all"] >= 1.2 else "fails"
        verd[v]["subtype_disagreement"] = float(np.mean(lab != T.labels))
    S = pd.DataFrame(rows)
    S.to_csv(out / "gate_summary.csv", index=False, float_format="%.4g")
    J = {"rules": verd, "flip_rate": flip, "n_cells": int(T.n), "n_confident": int(conf.sum()),
         "folds": int(a.folds), "primary": "natural"}
    (out / "gate_rules.json").write_text(json.dumps(J, indent=1, default=float))
    cols = ["variant", "subset", "arm", "error_rate", "dest_enrichment_all", "own_share_ratio_all", "sp_beta_contact",
            "sp_se_contact", "sp_beta_far", "sp_ratio"]
    L = ["# label errors against the CosMx protein gate (class level)", "", S[[c for c in cols if c in S]].round(4)
         .to_string(index=False), "", json.dumps(J, default=float)]
    (out / "gate.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# --------------------------------------------------------------------------------------------- registration
ORIENT = [(1, 0, 0, 1), (-1, 0, 0, 1), (1, 0, 0, -1), (-1, 0, 0, -1), (0, 1, 1, 0), (0, -1, 1, 0), (0, 1, -1, 0),
          (0, -1, -1, 0)]


def _density(pts, origin, shape, bin_um):
    ij = np.floor((pts - origin) / bin_um).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    D = np.zeros(shape)
    np.add.at(D, (ij[ok, 0], ij[ok, 1]), 1.0)
    return D - D.mean()


def _xcorr_shift(P, Q, bin_um=20.0):
    """Translation of Q onto P maximising the cross-correlation of their density images; returns (shift, score)."""
    lo = np.minimum(P.min(0), Q.min(0)) - 200
    hi = np.maximum(P.max(0), Q.max(0)) + 200
    shape = tuple(int(x) for x in np.ceil((hi - lo) / bin_um))
    from scipy.fft import next_fast_len
    shape2 = tuple(next_fast_len(2 * s) for s in shape)
    A = np.zeros(shape2)
    B = np.zeros(shape2)
    A[:shape[0], :shape[1]] = _density(P, lo, shape, bin_um)
    B[:shape[0], :shape[1]] = _density(Q, lo, shape, bin_um)
    Cc = np.fft.irfft2(np.fft.rfft2(A) * np.conj(np.fft.rfft2(B)), s=shape2)
    i, j = np.unravel_index(np.argmax(Cc), Cc.shape)
    i = i - shape2[0] if i > shape2[0] // 2 else i
    j = j - shape2[1] if j > shape2[1] // 2 else j
    score = Cc.max() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12)
    return np.array([i, j], float) * bin_um, float(score)


def _umeyama(src, dst, scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1, d])
    R = U @ E @ Vt
    c = (sig * np.diag(E)).sum() / S.var(0).sum() if scale else 1.0
    M = c * R
    return M, mu_d - M @ mu_s


def _mutual_pairs(P, Q, r):
    tp, tq = cKDTree(P), cKDTree(Q)
    dq, iq = tp.query(Q, distance_upper_bound=r)          # for each Q, nearest P
    dp, ip = tq.query(P, distance_upper_bound=r)          # for each P, nearest Q
    ok = np.isfinite(dq)
    qi = np.flatnonzero(ok)
    pi = iq[ok]
    mutual = ip[pi] == qi
    return pi[mutual], qi[mutual], dq[ok][mutual]


def _rot(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _match_excess(P, Q1, max_dist, shift=100.0):
    pi, qi, d = _mutual_pairs(P, Q1, max_dist)
    pi0, _, _ = _mutual_pairs(P, Q1 + np.array([shift, 0.0]), max_dist)
    n_small = min(len(P), len(Q1))
    return (len(pi) - len(pi0)) / n_small, (pi, qi, d), len(pi0) / n_small


def _refine(P, Qpx, M, t, radii=(20, 10, 6, 4), iters=4):
    for r in radii:
        for _ in range(iters):
            Q1 = Qpx @ M.T + t
            pi, qi, _ = _mutual_pairs(P, Q1, r)
            if len(pi) < 20:
                return M, t
            M, t = _umeyama(Qpx[qi], P[pi])
    return M, t


def register(P, Qpx, max_dist=5.0, angles=np.arange(-45, 46, 3), scales=np.arange(0.8, 1.21, 0.05), bin_um=20.0,
             fine_bin_um=10.0, top=3, sub=20000, seed=0, s0=None):
    """Q (pixels) -> P frame (µm). Coarse: every orientation x rotation x scale, translation by cross-correlation of
    density images (bin_um); the best `top` candidates are refined (finer cross-correlation, then mutual nearest
    neighbors at decreasing radii, similarity then affine) and the one with the largest match EXCESS over chance
    (matches within max_dist minus matches after a 100 µm shift) wins. Returns (M, t, info, (pi, qi, d)).
    s0: the image scale in µm per pixel when it is known (e.g. from OME metadata); `scales` are then factors around
    it. Default (None): s0 from the ratio of the two point clouds' bounding boxes, which fails when the image crop and
    the section do not cover the same tissue."""
    rng = np.random.default_rng(seed)
    Ps = P[rng.choice(len(P), min(sub, len(P)), replace=False)]
    Qs = Qpx[rng.choice(len(Qpx), min(sub, len(Qpx)), replace=False)]
    rP = P.max(0) - P.min(0)
    cands = []
    for o in ORIENT:
        O = np.array(o, float).reshape(2, 2)
        for th in angles:
            Rm = _rot(th) @ O
            Qo = Qs @ Rm.T
            rQ = Qo.max(0) - Qo.min(0)
            s_0 = float(s0) if s0 is not None else float(np.sqrt(rP.prod() / max(rQ.prod(), 1e-12)))
            for f in scales:
                M = s_0 * f * Rm
                Q1 = Qs @ M.T
                c0 = P.mean(0) - Q1.mean(0)
                sh, sc = _xcorr_shift(Ps, Q1 + c0, bin_um)
                cands.append((sc, M, c0 + sh, o, float(th), float(f)))
    cands.sort(key=lambda c: -c[0])
    best = None
    for sc, M0, t0, o, th, f in cands[:top]:
        # local search: rotation +-2 degrees (0.25), scale +-3% (1%); translation by finer cross-correlation;
        # candidates judged by match excess over chance on the subsample
        lb = None
        for dth in np.arange(-2.0, 2.01, 0.25):
            for df in (0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03):
                M = df * _rot(dth) @ M0
                Q1 = Qs @ M.T
                c0 = (Qs @ M0.T + t0).mean(0) - Q1.mean(0)
                sh, _ = _xcorr_shift(Ps, Q1 + c0, fine_bin_um)
                tt = c0 + sh
                e, _, _ = _match_excess(Ps, Qs @ M.T + tt, max_dist * 2)
                if lb is None or e > lb[0]:
                    lb = (e, M, tt)
        _, M, t = lb
        M, t = _refine(P, Qpx, M, t, radii=(10, 6, 4, 3))
        Q1 = Qpx @ M.T + t
        pi, qi, _ = _mutual_pairs(P, Q1, max_dist * 2)
        if len(pi) >= 50:
            X = np.c_[Qpx[qi], np.ones(len(qi))]
            coef, *_ = np.linalg.lstsq(X, P[pi], rcond=None)
            M, t = coef[:2].T, coef[2]
        exc, pairs, chance = _match_excess(P, Qpx @ M.T + t, max_dist)
        if best is None or exc > best[0]:
            best = (exc, M, t, pairs, chance, o, th, f, sc)
    exc, M, t, (pi, qi, d), chance, o, th, f, sc = best
    n_small = min(len(P), len(Qpx))
    info = {"orientation": list(o), "rotation_deg_coarse": th, "scale_factor_grid": f, "xcorr_score": sc,
            "n_cosmx": int(len(P)), "n_cellscape": int(len(Qpx)), "n_matched": int(len(pi)),
            "matched_share_small": len(pi) / n_small, "chance_share_small": chance, "excess_share": exc,
            "median_residual_um": float(np.median(d)) if len(d) else np.nan,
            "um_per_px": float(np.sqrt(abs(np.linalg.det(M)))), "max_dist_um": max_dist,
            "n_candidates": len(cands), "n_refined": min(top, len(cands))}
    info["R0_as_written"] = ("valid" if info["matched_share_small"] >= 0.5 and info["chance_share_small"] <= 0.1
                             else "invalid")
    info["R0"] = "valid" if info["matched_share_small"] >= 0.5 and exc >= 0.35 else "invalid"   # see docstring
    return M, t, info, (pi, qi, d)


def local_check(P, fov, Q1, search_um=40.0, step_um=1.0, match_um=2.0, margin_um=50.0, min_pts=50):
    """Same section with local distortion, or a different section? For each field of view, the CosMx cells and the
    globally registered CellScape nuclei in its box (plus margin) are matched within match_um over a grid of local
    translations (+-search_um). Best = largest number of matches; chance = median over the grid. A same section
    with distortion shows best >> chance in most fields; a serial section shows best close to chance everywhere."""
    rows = []
    shifts = np.arange(-search_um, search_um + step_um / 2, step_um)
    for f in pd.unique(fov):
        Pf = P[fov == f]
        lo, hi = Pf.min(0) - margin_um, Pf.max(0) + margin_um
        Qf = Q1[(Q1 >= lo).all(1) & (Q1 <= hi).all(1)]
        if len(Pf) < min_pts or len(Qf) < min_pts:
            continue
        tp = cKDTree(Pf)
        counts = np.zeros((len(shifts), len(shifts)), int)
        for i, dx in enumerate(shifts):
            for j, dy in enumerate(shifts):
                d, _ = tp.query(Qf + np.array([dx, dy]), distance_upper_bound=match_um)
                counts[i, j] = int(np.isfinite(d).sum())
        bi, bj = np.unravel_index(np.argmax(counts), counts.shape)
        n = min(len(Pf), len(Qf))
        rows.append({"fov": f, "n_cosmx": len(Pf), "n_cellscape": len(Qf), "best_dx": shifts[bi], "best_dy": shifts[bj],
                     "best_share": counts[bi, bj] / n, "chance_share": float(np.median(counts)) / n})
    L = pd.DataFrame(rows)
    if len(L):
        L["excess"] = L.best_share - L.chance_share
    return L


def nn_profile(P, Q1, bins=np.arange(0, 20.5, 0.5), shift=100.0):
    """Distances from each Q point to its nearest P point, aligned and shifted (chance), as a histogram table."""
    tp = cKDTree(P)
    d1, _ = tp.query(Q1)
    d0, _ = tp.query(Q1 + np.array([shift, 0.0]))
    h1, _ = np.histogram(d1, bins)
    h0, _ = np.histogram(d0, bins)
    return pd.DataFrame({"lo_um": bins[:-1], "hi_um": bins[1:], "aligned": h1, "shifted": h0,
                         "excess": h1 - h0})


def cmd_register(a):
    M_ = pd.read_csv(a.cosmx_meta, low_memory=False)
    N = pd.read_csv(a.cellscape)
    P = np.c_[M_[a.cosmx_x].to_numpy(float), M_[a.cosmx_y].to_numpy(float)] * a.cosmx_unit_um
    Q = np.c_[N[a.cs_x].to_numpy(float), N[a.cs_y].to_numpy(float)]
    M, t, info, (pi, qi, d) = register(P, Q, max_dist=a.max_dist)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = nn_profile(P, Q @ M.T + t)
    if a.cosmx_fov in M_:
        Lc = local_check(P, M_[a.cosmx_fov].to_numpy(), Q @ M.T + t)
        Lc.to_csv(out / "local_check.csv", index=False, float_format="%.4g")
        if len(Lc):
            info["local_median_excess"] = float(Lc.excess.median())
            info["local_median_best_share"] = float(Lc.best_share.median())
            info["local_n_fov"] = int(len(Lc))
            info["R1"] = ("same section" if Lc.excess.median() >= 0.3 else
                          "different section or unregistrable" if Lc.excess.median() < 0.1 else "undetermined")
    prof.to_csv(out / "nn_distance_profile.csv", index=False)
    info["excess_within_3um"] = float(prof.loc[prof.hi_um <= 3, "excess"].sum() / min(len(P), len(Q)))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        ax[0].step(prof.lo_um, prof.aligned, where="post", label="aligned")
        ax[0].step(prof.lo_um, prof.shifted, where="post", label="shifted 100 µm (chance)")
        ax[0].set_xlabel("distance from CellScape nucleus to nearest CosMx cell (µm)")
        ax[0].set_ylabel("nuclei")
        ax[0].legend(frameon=False)
        Q1 = Q @ M.T + t
        c = np.median(P, 0)
        w = 150
        mp = (np.abs(P - c) < w).all(1)
        mq = (np.abs(Q1 - c) < w).all(1)
        ax[1].scatter(P[mp, 0], P[mp, 1], s=6, label="CosMx cells")
        ax[1].scatter(Q1[mq, 0], Q1[mq, 1], s=6, marker="x", label="CellScape nuclei (registered)")
        ax[1].set_aspect("equal")
        ax[1].legend(frameon=False, fontsize=7)
        ax[1].set_title("300 µm window at the median CosMx position", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "registration_check.png", dpi=150)
    except Exception as e:                                  # the figure is a convenience
        print(f"figure skipped: {e}", flush=True)
    pd.DataFrame({"cosmx_cell_id": M_["cell_id"].astype(str).to_numpy()[pi],
                  "cellscape_row": qi, "cellscape_nuc_label": N[a.cs_id].to_numpy()[qi], "distance_um": d}) \
        .to_csv(out / "matches.csv", index=False, float_format="%.4g")
    info.update({"affine": M.tolist(), "offset_um": t.tolist()})
    (out / "registration.json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, indent=1, default=float))


# --------------------------------------------------------------------------------------------- CellScape classes
MARKERS = {"Plasma": ["CD138"], "B": ["CD20", "CD19"], "T": ["CD3"], "Myeloid": ["CD68", "CD163"],
           "Epithelial": ["PanCK", "E-Cadherin"], "Endothelial": ["CD31"],
           "Mesenchymal": ["SMA", "Podoplanin", "Vimentin"]}


def marker_positive(v, seed=0, min_ashman=2.0, return_d=False):
    """Two-component Gaussian mixture on log1p intensity; positive = higher-mean component (posterior > 0.5).
    A marker whose two components are not separated (Ashman's D < min_ashman) has no positive population: a
    mixture fitted to unimodal noise would otherwise call half the cells positive."""
    from sklearn.mixture import GaussianMixture
    x = np.log1p(np.clip(np.asarray(v, float), 0, None)).reshape(-1, 1)
    g = GaussianMixture(2, random_state=seed).fit(x)
    mu, var = g.means_.ravel(), g.covariances_.ravel()
    d = float(np.sqrt(2) * abs(mu[0] - mu[1]) / np.sqrt(var[0] + var[1]))
    hi = int(np.argmax(mu))
    pos = (g.predict_proba(x)[:, hi] > 0.5) if d >= min_ashman else np.zeros(len(x), bool)
    return (pos, d) if return_d else pos


def cellscape_classes(N):
    fits = {m: marker_positive(N[m], return_d=True) for ms in MARKERS.values() for m in ms if m in N}
    pos = {m: f[0] for m, f in fits.items()}
    cd45 = marker_positive(N["CD45"]) if "CD45" in N else np.zeros(len(N), bool)
    cls = np.full(len(N), "unassigned", dtype=object)
    done = np.zeros(len(N), bool)
    for c, ms in MARKERS.items():
        hit = np.zeros(len(N), bool)
        for m in ms:
            if m in pos:
                hit |= pos[m]
        if c == "Mesenchymal":
            hit &= ~cd45
        new = hit & ~done
        cls[new] = c
        done |= hit
    cls[(~done) & cd45] = "Immune other"
    return cls.astype(str), {m: {"positive_share": float(f[0].mean()), "ashman_d": f[1]} for m, f in fits.items()}


def cmd_cellscape(a):
    reg = Path(a.matches).parent / "registration.json"
    if reg.exists() and json.loads(reg.read_text()).get("R0") != "valid" and not a.force:
        sys.exit("registration R0 is not valid; cellscape readouts would compare unrelated cells (use --force to "
                 "override, e.g. for a serial-section analysis at region level)")
    T, obs, sub2class = load_cosmx(a.h5ad, min_cells=a.min_cells)
    N = pd.read_csv(a.cellscape)
    Mt = pd.read_csv(a.matches)
    cs_cls, pos_rates = cellscape_classes(N)
    id2row = {c: i for i, c in enumerate(np.asarray(T.obs_names).astype(str))}
    Mt = Mt[Mt.cosmx_cell_id.astype(str).isin(id2row)]
    ci = np.array([id2row[c] for c in Mt.cosmx_cell_id.astype(str)])
    ref = cs_cls[Mt.cellscape_row.to_numpy()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gate = obs["category"].astype(str).to_numpy()[ci]
    coarse = np.where(np.isin(ref, ["Plasma", "B", "T", "Myeloid", "Immune other"]), "Immune",
                      np.where(ref == "Epithelial", "Epithelial", np.where(ref == "unassigned", "unassigned", "Others")))
    ok = coarse != "unassigned"
    p0 = float(np.mean(coarse[ok] == gate[ok]))
    major = obs["ct_major_new"].astype(str).to_numpy()[ci]
    pub = np.array([CS_MAP.get(x, "none") for x in major])
    usable = (pub != "none") & (ref != "unassigned")
    p1 = float(np.mean(pub[usable] == ref[usable]))
    fold = spatial_folds(T.xy, obs["fov"].to_numpy(), a.folds)
    labs, _ = transcript_labels(T, fold)
    sub2major = obs.groupby(obs["ct_minor_new"].astype(str))["ct_major_new"].agg(lambda s: s.astype(str).mode()[0])
    lab = np.array([CS_MAP.get(sub2major.get(x, ""), "none") for x in labs["natural"][ci]])
    u2 = usable & (lab != "none") & np.isin(lab, np.unique(ref[usable]))   # destinations must exist as classes
    r, dtab, Tsp = spatial_readouts(T.xy[ci][u2], ref[u2], lab[u2])
    ctrl, _, _ = spatial_readouts(T.xy[ci][u2], ref[u2], random_control(Tsp, lab[u2]))
    rp, _, _ = spatial_readouts(T.xy[ci][u2], ref[u2], pub[u2])
    J = {"P0_agreement_with_cosmx_gate": p0, "P0": "credible" if p0 >= 0.8 else "not credible",
         "P1_published_major_vs_cellscape": p1, "P1_n": int(usable.sum()), "P2": verdict_E(r, ctrl),
         "labeller_error_vs_cellscape": r["error_rate"], "marker_positive_rates": pos_rates,
         "n_matched": int(len(ci))}
    S = pd.DataFrame([dict(arm="labeller (natural prior)", **r), dict(arm="random control", **ctrl),
                      dict(arm="published major labels", **rp)])
    S.to_csv(out / "cellscape_summary.csv", index=False, float_format="%.4g")
    pd.crosstab(pd.Series(pub[usable], name="published"), pd.Series(ref[usable], name="cellscape")) \
        .to_csv(out / "published_vs_cellscape.csv")
    pd.Series(cs_cls).value_counts().to_csv(out / "cellscape_class_counts.csv")
    (out / "cellscape_rules.json").write_text(json.dumps(J, indent=1, default=float))
    cols = ["arm", "error_rate", "dest_enrichment_all", "own_share_ratio_all", "sp_beta_contact", "sp_se_contact",
            "sp_ratio"]
    L = ["# label errors against CellScape protein classes (matched cells)", "",
         S[[c for c in cols if c in S]].round(4).to_string(index=False), "", json.dumps(J, default=float)]
    (out / "cellscape.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("h5ad")
    g.add_argument("--out", required=True)
    g.add_argument("--folds", type=int, default=5)
    g.add_argument("--min-cells", type=int, default=50)
    r = sub.add_parser("register")
    r.add_argument("cosmx_meta")
    r.add_argument("cellscape")
    r.add_argument("--out", required=True)
    r.add_argument("--max-dist", type=float, default=5.0)
    r.add_argument("--cosmx-x", default="x_slide_mm")
    r.add_argument("--cosmx-y", default="y_slide_mm")
    r.add_argument("--cosmx-unit-um", type=float, default=1000.0)
    r.add_argument("--cs-x", default="nuc_centroid_col")
    r.add_argument("--cs-y", default="nuc_centroid_row")
    r.add_argument("--cs-id", default="nuc_label")
    r.add_argument("--cosmx-fov", default="fov")
    c = sub.add_parser("cellscape")
    c.add_argument("h5ad")
    c.add_argument("cellscape")
    c.add_argument("--matches", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--min-cells", type=int, default=50)
    c.add_argument("--force", action="store_true", help="run even if the registration is invalid")
    a = ap.parse_args(argv)
    {"gate": cmd_gate, "register": cmd_register, "cellscape": cmd_cellscape}[a.cmd](a)


if __name__ == "__main__":
    main()
