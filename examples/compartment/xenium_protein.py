#!/usr/bin/env python
"""xenium_protein.py -- protein on the same cells as the labels (Xenium
Protein, 10x FFPE human renal cell carcinoma, XOA 4.0, 377+28 genes and 27 proteins on one section).

Xenium Onboard Analysis 4.0 stores, per cell, a background-subtracted protein intensity as a second feature type
("Protein Expression") of cell_feature_matrix.h5, next to the genes. Protein and RNA come from the same cells of the
same section, so there is no registration and no chance matching (unlike the post-Xenium IHC of the lung data).

  extract  cell_feature_matrix.h5 -> <sample>.protein.csv.gz (cell_id + one column per protein) and markers.json
  truth    protein classes against the label arms of a labels file:
             positivity per marker: Gaussian mixture (2 or 3 components) on log1p(intensity) of the cells
             with non-zero intensity; positive = the highest component with posterior > 0.5, when it is separated
             from the next (Ashman's D >= 2), else no call; zeros are negative;
             classes (immune classes also reported without PanCK+, as "clean"):
               T = CD3E+, CD8 = CD8A+, CD4 = CD4+ CD3E+, B = CD20+, Myeloid = CD68+, Epithelial = PanCK+;
             expected label: T, CD8, CD4 -> T/NK type; B -> B; Myeloid -> Myeloid;
             recall per arm: share of class cells (labeled in the arm) given the expected label, overall and by
             contact with a PanCK+ cell (6 nearest within 30 um, neighbors typed by protein, so contact does not
             depend on any label); precision: share of cells given a label that are protein-positive for its class;
             fate: labels taken by class cells that miss the expected one.
  The comparison of interest is nucB against wholeB (equal depth) and
  nucleus against whole; the expectation from the RNA analyses is lower recall of protein T and CD8 cells from
  whole-cell labels, more so at PanCK+ contact.

  python xenium_protein.py extract --outs RCC_OUTS --sample XP_RCC --out DIR
  python xenium_protein.py truth --labels LABELS --protein DIR/XP_RCC.protein.csv.gz --sample XP_RCC --out DIR2 --level L1
  --arms nucleus,whole,nucB,wholeB,cytoB,cytoM,wBall,wBallM,wholeM
                   --pairs nucB:wholeB,nucleus:whole,nucB:wBall,nucB:wBallM,nucleus:wholeM --gaps nucB:wBall:wBallM
  border   loss of the expected label (wholeB vs nucB) for protein classes,
           binned by the share of the cell's polygon perimeter held by non-immune neighbors (CD45-negative by
           protein; shared perimeter from the segmentation polygons, claimbreak.contacts.boundary_shares), with
           block-bootstrap intervals per bin and a weighted slope of loss on the share. A loss that rises with the
           non-immune share confirms the contact mechanism on protein-typed neighbors; one that is flat does not.
  truth --strict --posterior 0.9: lineage-consistent classes (CD3E+ cells that are CD45+, CD68- and PanCK-)
           and a stricter positive call, to show that the recall ratios do not rest on loose positivity or on protein
           signal from neighbors; class_overlap.csv gives each class's share of CD45+, CD68+ and PanCK+ cells.
  truth --alias CD3E=CD45RO_hp: a class marker read from another column (MERSCOPE uterine cancer has only
           CD45RO; its T class is CD45RO+ cells, memory T cells and some activated myeloid cells).
  groups   a published labeling of the same cells (e.g. 10x cell_groups.csv: cell_id, group) scored against
           protein: per group, the share of cells positive for each class marker; and, with the class regexes of
           --map, recall and precision of each protein class.
  python xenium_protein.py border --labels LABELS --protein P.csv.gz --boundaries DIR --sample S --out DIR3
  border --base outline: the share over the whole outline, as on the label side; cells that barely touch
           one tumor cell no longer fall in the top bin.
  groups-border the published labeling and the claimbreak arms scored on the same (grouped) cells by the
           non-immune share of the outline, over the touched part (as border) and over the whole outline; recall per
           bin, loss per pair (nucleus:pub, nucleus:whole, nucB:wholeB) with block intervals, weighted slopes.
           Whether the published labels used protein is not known; if they did, their recall is not independent.
  python xenium_protein.py groups --groups cell_groups.csv --protein P.csv.gz --out DIR4
  python xenium_protein.py groups-border --groups cell_groups.csv --labels LABELS --protein P.csv.gz --boundaries DIR \
         --sample S --out DIR5
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CLASSES = {          # class: (required positive markers, expected label regex)
    "T": (["CD3E"], r"(?i)(^|[^a-z])(t|nk|tnk|t_nk)([^a-z]|$)"),
    "CD8": (["CD8A"], r"(?i)(^|[^a-z])(t|nk|tnk|t_nk)([^a-z]|$)"),
    "CD4": (["CD4", "CD3E"], r"(?i)(^|[^a-z])(t|nk|tnk|t_nk)([^a-z]|$)"),
    "B": (["CD20"], r"(?i)^b$|(^|_)b(_|$)|b_?cell"),
    "Myeloid": (["CD68"], r"(?i)myeloid|macro|mono"),
    # any immune label (CosMx CD45 stain; Xenium Protein panels with CD45)
    "Immune": (["CD45"], r"(?i)(^|[^a-z])(t|nk|tnk|t_nk|b|plasma|myeloid|macro|mono|mast|immune|lymph)([^a-z]|$)"),
    "Epithelial": (["PANCK"], None),
}


# lineage-consistent classes, so
# that a cell whose protein signal could come from a neighbor (CD3E on a CD45-negative, CD68+ or PanCK+ cell) is
# not counted as a T cell. truth --strict adds them next to the classes above: (positive markers, negative markers,
# expected label regex).
STRICT = {
    "T_strict": (["CD3E", "CD45"], ["CD68", "PANCK"], CLASSES["T"][1]),
    "CD8_strict": (["CD8A", "CD3E", "CD45"], ["CD68", "PANCK"], CLASSES["CD8"][1]),
    "CD4_strict": (["CD4", "CD3E", "CD45"], ["CD68", "PANCK"], CLASSES["CD4"][1]),
    "B_strict": (["CD20", "CD45"], ["CD3E", "CD68", "PANCK"], CLASSES["B"][1]),
}


def norm(name):
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _fresh(out):
    out = Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_protein(h5):
    import h5py
    import scipy.sparse as sp
    with h5py.File(h5, "r") as f:
        m = f["matrix"]
        ft = m["features"]
        names = [x.decode() for x in ft["name"][:]]
        types = [x.decode() for x in ft["feature_type"][:]]
        bc = [x.decode() for x in m["barcodes"][:]]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).tocsr()
    rows = [i for i, t in enumerate(types) if t == "Protein Expression"]
    if not rows:
        raise SystemExit(f"{h5}: no 'Protein Expression' features (feature types: {sorted(set(types))})")
    P = pd.DataFrame(X[rows].T.toarray(), columns=[names[i] for i in rows])
    P.insert(0, "cell_id", bc)
    return P


def cmd_extract(a):
    outs = Path(a.outs).expanduser()
    h5 = outs / "cell_feature_matrix.h5"
    if not h5.exists():
        hits = list(outs.glob("*/cell_feature_matrix.h5"))
        if len(hits) != 1:
            raise SystemExit(f"no cell_feature_matrix.h5 in {outs}")
        h5 = hits[0]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{a.sample}.protein.csv.gz"
    if dst.exists():
        raise SystemExit(f"{dst} exists; not overwritten")
    P = read_protein(h5)
    P.to_csv(dst, index=False, compression="gzip")
    summ = {c: {"zero_share": float((P[c] == 0).mean()), "median": float(P[c].median()),
                "q99": float(P[c].quantile(0.99))} for c in P.columns[1:]}
    (out / f"{a.sample}.protein_markers.json").write_text(json.dumps({"source": str(h5), "cells": len(P),
                                                                      "proteins": summ}, indent=1))
    print(f"{len(P):,} cells, {P.shape[1] - 1} proteins: {', '.join(P.columns[1:])}\nwrote {dst}")


def positive(v, seed=0, min_d=2.0, post=0.5):
    """Positive cells for one marker: (mask, Ashman D, threshold on v).
    XOA 4.0 intensities are background-subtracted, so most cells read exactly 0 and many negatives read small
    non-zero values. A mixture over all cells separates zero from non-zero (it calls most cells CD3E+); here
    the mixture (2 or 3 components by BIC) is fitted on log1p(v) of the non-zero cells only, and positive = the highest
    component (posterior > 0.5) when it is separated from the next (Ashman D >= min_d); zeros are negative."""
    from sklearn.mixture import GaussianMixture
    v = np.maximum(np.asarray(v, float), 0)
    nz = v > 0
    pos = np.zeros(len(v), bool)
    if nz.sum() < 50 or np.unique(v[nz]).size < 5:
        return pos, 0.0, np.nan
    x = np.log1p(v[nz]).reshape(-1, 1)
    best = None
    for k in (2, 3):
        g = GaussianMixture(k, random_state=seed, n_init=3).fit(x)
        mu, var = g.means_.ravel(), g.covariances_.ravel()
        o = np.argsort(mu)
        hi, nx = o[-1], o[-2]
        d = float(np.sqrt(2) * abs(mu[hi] - mu[nx]) / np.sqrt(var[hi] + var[nx]))
        bic = g.bic(x)
        if best is None or bic < best[0]:
            best = (bic, g, hi, d)
    _, g, hi, d = best
    if d >= min_d:
        pos[np.flatnonzero(nz)] = g.predict_proba(x)[:, hi] > post
    thr = float(v[pos].min()) if pos.any() else np.nan
    return pos, d, thr


PAIRS = (("nucB", "wholeB"), ("nucleus", "whole"))


def block_ci(D, cls, hits, strata, tile, draws, seed, pairs=PAIRS, gaps=()):
    """block bootstrap over square tiles (side `tile` um) for one protein class. hits: {arm: bool array, cell has
    the expected label}; strata: {name: bool mask}. Returns rows (metric, stratum, estimate, lo, hi): recall per arm,
    loss = 1 - recall(test) / recall(reference) per pair, and per partner the loss at contact minus without.
    gaps, triples (reference, base, model): share of the recall gap between base and reference that the model
    closes, (recall(model) - recall(base)) / (recall(reference) - recall(base)) (metric gap_closed_<model>_<base>)."""
    xy = D[["x", "y"]].to_numpy(float)
    key = np.floor(xy[:, 0] / tile).astype(np.int64) * 1_000_000 + np.floor(xy[:, 1] / tile).astype(np.int64)
    tk, inv = np.unique(key, return_inverse=True)
    T = len(tk)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(T, np.full(T, 1.0 / T), size=draws).astype(float) if draws else np.ones((0, T))
    W = np.vstack([np.ones((1, T)), W])                     # row 0: the point estimate
    rec, rec_all, lab_share = {}, {}, {}
    for sname, sm in strata.items():
        n_all = np.bincount(inv[cls & sm], minlength=T).astype(float)
        for arm, h in hits.items():
            ok = cls & sm & h[1]                             # labeled in the arm
            n = np.bincount(inv[ok], minlength=T).astype(float)
            k = np.bincount(inv[ok & h[0]], minlength=T).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                rec[(sname, arm)] = (W @ k) / (W @ n)
                # unlabeled cells (too few counts in the arm, or no nucleus) counted as misses, and the
                # labeled share; arms of different depth (nucleus against whole) differ in who is labeled
                rec_all[(sname, arm)] = (W @ k) / (W @ n_all)
                lab_share[(sname, arm)] = (W @ n) / (W @ n_all)
    rows = []

    def add(metric, stratum, v):
        b = v[1:][np.isfinite(v[1:])]
        rows.append(dict(metric=metric, stratum=stratum, estimate=float(v[0]),
                         lo=float(np.quantile(b, 0.025)) if len(b) else np.nan,
                         hi=float(np.quantile(b, 0.975)) if len(b) else np.nan, tiles=T, draws=draws))
    loss = {}
    for (sname, arm), v in rec.items():
        add(f"recall_{arm}", sname, v)
        add(f"recallall_{arm}", sname, rec_all[(sname, arm)])
        add(f"labelled_{arm}", sname, lab_share[(sname, arm)])
    for ref, test in pairs:
        for sname in strata:
            if (sname, ref) in rec and (sname, test) in rec:
                loss[(sname, test)] = 1 - rec[(sname, test)] / rec[(sname, ref)]
                add(f"loss_{test}_vs_{ref}", sname, loss[(sname, test)])
                with np.errstate(invalid="ignore", divide="ignore"):
                    add(f"lossall_{test}_vs_{ref}", sname, 1 - rec_all[(sname, test)] / rec_all[(sname, ref)])
    for ref, base, model in gaps:
        for sname in strata:
            if all((sname, x) in rec for x in (ref, base, model)):
                with np.errstate(invalid="ignore", divide="ignore"):
                    g = (rec[(sname, model)] - rec[(sname, base)]) / (rec[(sname, ref)] - rec[(sname, base)])
                add(f"gap_closed_{model}_{base}", sname, g)
    for sname in strata:
        if sname.endswith("_contact") and not sname.startswith("no_"):
            other = "no_" + sname
            for ref, test in pairs:
                if (sname, test) in loss and (other, test) in loss:
                    add(f"loss_{test}_contact_minus_no", sname, loss[(sname, test)] - loss[(other, test)])
    return rows


def cmd_truth(a):
    out = _fresh(a.out)
    lv = a.level
    L = pd.read_csv(Path(a.labels).expanduser() / f"{a.sample}.labels.csv.gz", keep_default_na=False,
                    dtype={"cell_id": str}, low_memory=False)
    P = pd.read_csv(a.protein, dtype={"cell_id": str})
    cols = {norm(c): c for c in P.columns if c != "cell_id"}
    for al in [x for x in getattr(a, "alias", "").split(",") if x]:     # e.g. CD3E=CD45RO_hp (MERSCOPE)
        key, src = (norm(y) for y in al.split("="))
        if src not in cols:
            raise SystemExit(f"--alias {al}: no protein column {src} (have {sorted(cols)})")
        print(f"alias: marker {key} read from protein column {cols[src]}")
        cols[key] = cols[src]
    D = L.merge(P, on="cell_id", how="inner")
    print(f"{len(D):,} of {len(L):,} labelled cells have protein values")
    # hybrid arms A+B: the label of arm A where A labeled the cell, else that of arm B (e.g. nucleus+whole:
    # nuclear labels for cells with enough nuclear counts, whole-cell labels for the rest)
    hyb = []
    for h in [x for x in (getattr(a, "hybrid", "") or "").split(",") if x]:
        first, second = h.split(":")
        c1, c2 = f"{lv}_{first}", f"{lv}_{second}"
        if c1 not in D or c2 not in D:
            raise SystemExit(f"--hybrid {h}: needs columns {c1} and {c2}")
        name = f"{first}+{second}"
        D[f"{lv}_{name}"] = np.where(D[c1].astype(str) != "", D[c1].astype(str), D[c2].astype(str))
        hyb.append(name)
        print(f"hybrid arm {name}: {first} where labelled ({(D[c1].astype(str) != '').mean():.3f} of cells), else {second}")
    pos, mrows = {}, []
    for key in sorted({m for req, _ in CLASSES.values() for m in req}):
        if key not in cols:
            mrows.append(dict(marker=key, protein=None, ashman_d=np.nan, positive_share=np.nan, threshold=np.nan))
            continue
        p, d, thr = positive(D[cols[key]].to_numpy(), seed=a.seed, post=getattr(a, "posterior", 0.5))
        pos[key] = p
        mrows.append(dict(marker=key, protein=cols[key], ashman_d=d, positive_share=float(p.mean()), threshold=thr))
    M = pd.DataFrame(mrows)
    M.to_csv(out / "markers.csv", index=False)
    classes = {c: (req, [], rx) for c, (req, rx) in CLASSES.items()}
    if getattr(a, "strict", False):
        classes.update({c: v for c, v in STRICT.items() if all(m in pos for m in v[0] + v[1])})

    def cmask(req, neg):
        m = np.logical_and.reduce([pos[x] for x in req])
        for x in neg:
            m = m & ~pos[x]
        return m
    # how lineage-consistent each class is (CD3E+ cells without CD45, or with CD68 or PanCK, point to protein
    # signal from neighbors or to a loose positivity call)
    ov = []
    for cname, (req, neg, rx) in classes.items():
        if not all(m in pos for m in req + neg):
            continue
        cm_ = cmask(req, neg)
        row = dict(protein_class=cname, cells=int(cm_.sum()), share_of_cells=float(cm_.mean()))
        for x in ("CD45", "CD68", "PANCK", "CD3E", "CD20"):
            if x in pos and cm_.any():
                row[f"share_{x}+"] = float(pos[x][cm_].mean())
        ov.append(row)
    pd.DataFrame(ov).to_csv(out / "class_overlap.csv", index=False)
    epi = pos.get("PANCK", np.zeros(len(D), bool))
    xy = D[["x", "y"]].to_numpy(float)
    dist, idx = cKDTree(xy).query(xy, k=a.k + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]
    # contact partners typed by protein (default PanCK and CD68); a partner marker that does not separate gives
    # no contact strata. Contact with a marker excludes the cell itself.
    partners = [norm(x) for x in getattr(a, "partners", "PANCK").split(",") if x]
    touch = {}
    for pm in partners:
        if pm in pos and pos[pm].any():
            touch[pm] = (pos[pm][idx] & (dist <= a.r)).any(1)
    touch_epi = touch.get("PANCK", np.zeros(len(D), bool))
    arms = [x for x in a.arms.split(",") if f"{lv}_{x}" in D] + hyb
    labels_all = sorted(set().union(*[set(D[f"{lv}_{x}"]) for x in arms]) - {""})
    rec, prec, fate = [], [], []
    for cname, (req, neg, rx) in classes.items():
        if not all(m in pos for m in req + neg):
            continue
        cls = cmask(req, neg)
        if rx is None:
            for arm in arms:
                lab = D[f"{lv}_{arm}"].to_numpy()
                vc = pd.Series(lab[cls & (lab != "")]).value_counts(normalize=True)
                for k, v in vc.items():
                    fate.append(dict(protein_class=cname, variant="all", arm=arm, label=k, share=float(v)))
            continue
        target = [t for t in labels_all if re.search(rx, t)]
        for variant, cm in (("all", cls), ("clean", cls & ~epi)):
            for arm in arms:
                lab = D[f"{lv}_{arm}"].to_numpy()
                ok = lab != ""
                hit = np.isin(lab, target)
                strata = [("all", np.ones(len(D), bool))]
                for pm, tm in touch.items():
                    strata += [(f"{pm.lower()}_contact", tm), (f"no_{pm.lower()}_contact", ~tm)]
                for stratum, sm in strata:
                    m = cm & ok & sm
                    m_all = cm & sm
                    # recall_all counts unlabeled cells as misses; labeled = share of the class labeled
                    rec.append(dict(protein_class=cname, variant=variant, arm=arm, stratum=stratum, n=int(m.sum()),
                                    recall=float(hit[m].mean()) if m.any() else np.nan,
                                    n_all=int(m_all.sum()),
                                    recall_all=float((hit & ok)[m_all].mean()) if m_all.any() else np.nan,
                                    labelled=float(ok[m_all].mean()) if m_all.any() else np.nan))
                if variant == "all":
                    called = ok & hit
                    prec.append(dict(protein_class=cname, arm=arm, called=int(called.sum()),
                                     precision=float(cls[called].mean()) if called.any() and cls.any() else np.nan))
                    vc = pd.Series(lab[cm & ok & ~hit]).value_counts(normalize=True)
                    for k, v in vc.items():
                        fate.append(dict(protein_class=cname, variant="missed", arm=arm, label=k, share=float(v)))
    # block bootstrap of recall, loss and contact difference (immune classes, all cells of the class)
    # --pairs (reference:test) and --gaps (reference:base:model) restricted to the arms present
    pairs = tuple(tuple(x.split(":")) for x in getattr(a, "pairs", "nucB:wholeB,nucleus:whole").split(",") if x)
    pairs = tuple(p for p in pairs if len(p) == 2 and all(x in arms for x in p))
    gaps = tuple(tuple(x.split(":")) for x in (getattr(a, "gaps", "") or "").split(",") if x)
    bad = [g for g in gaps if len(g) != 3]
    if bad:
        raise SystemExit(f"--gaps takes reference:base:model triples, not {bad}")
    gaps = tuple(g for g in gaps if all(x in arms for x in g))
    ci_rows = []
    draws = getattr(a, "block_draws", 1000)
    if draws:
        strata_ci = {"all": np.ones(len(D), bool)}
        for pm, tm in touch.items():
            strata_ci[f"{pm.lower()}_contact"] = tm
            strata_ci[f"no_{pm.lower()}_contact"] = ~tm
        for cname, (req, neg, rx) in classes.items():
            if rx is None or not all(m in pos for m in req + neg):
                continue
            cls = cmask(req, neg)
            target = [t for t in labels_all if re.search(rx, t)]
            hits = {arm: (np.isin(D[f"{lv}_{arm}"].to_numpy(), target), D[f"{lv}_{arm}"].to_numpy() != "")
                    for arm in arms}
            for r_ in block_ci(D, cls, hits, strata_ci, getattr(a, "tile", 200.0), draws, a.seed, pairs, gaps):
                ci_rows.append(dict(protein_class=cname, **r_))
    CI = pd.DataFrame(ci_rows)
    CI.to_csv(out / "recall_ci.csv", index=False)
    R, Pr, F = pd.DataFrame(rec), pd.DataFrame(prec), pd.DataFrame(fate)
    R.to_csv(out / "recall.csv", index=False)
    Pr.to_csv(out / "precision.csv", index=False)
    F.to_csv(out / "fate.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(labels=str(a.labels), protein=str(a.protein), sample=a.sample,
                                                   level=lv, arms=arms, k=a.k, r=a.r, cells=len(D),
                                                   pairs=[list(p) for p in pairs], gaps=[list(g) for g in gaps],
                                                   partners=partners,
                                                   contact_share={k: float(v.mean()) for k, v in touch.items()}),
                                         indent=1))
    with pd.option_context("display.width", 220):
        print("markers:\n" + M.round(3).to_string(index=False))
        if ov:
            print("\nclass size and lineage overlap (share of class cells positive for each marker):\n"
                  + pd.DataFrame(ov).round(3).to_string(index=False))
        if len(R):
            print("\nrecall of protein classes by the expected label (all cells / at contact with a protein partner / without):")
            print(R[R.variant == "all"].pivot_table(index=["protein_class", "stratum"], columns="arm",
                                                    values="recall").round(3).to_string())
            print("\nrecall with unlabelled cells counted as misses (all cells of the class), and the "
                  "labelled share per arm:")
            q = R[(R.variant == "all") & (R.stratum == "all")]
            print(q.pivot_table(index="protein_class", columns="arm", values=["recall_all", "labelled"])
                  .round(3).to_string())
            print("\nclean (no PanCK on the cell itself):")
            print(R[R.variant == "clean"].pivot_table(index=["protein_class", "stratum"], columns="arm",
                                                      values="recall").round(3).to_string())
        if len(Pr):
            print("\nprecision (protein-positive share of cells given the label):")
            print(Pr.pivot_table(index="protein_class", columns="arm", values="precision").round(3).to_string())
        if len(CI) and CI.metric.str.startswith("gap_closed").any():
            q = CI[CI.metric.str.startswith("gap_closed") & (CI.stratum == "all")]
            print("\nshare of the recall gap closed (block bootstrap):")
            print(q[["protein_class", "metric", "estimate", "lo", "hi"]].round(3).to_string(index=False))
        if len(CI):
            q = CI[CI.metric.str.startswith("loss_wholeB")]
            print(f"\nloss of the expected label, wholeB vs nucB, block bootstrap over {int(CI.tiles.iloc[0])} tiles "
                  f"of {getattr(a, 'tile', 200.0):.0f} um ({draws} draws):")
            print(q[["protein_class", "metric", "stratum", "estimate", "lo", "hi"]].round(3).to_string(index=False))
    print(f"wrote {out}")


SHARE_BINS = [0.0, 1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9, 1.0 + 1e-9]
SHARE_NAMES = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1)", "1"]
SHARE_MID = dict(zip(SHARE_NAMES, [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]))


def _nonimm_share(D, nonimm, bp, tol):
    """per cell, the share of the polygon outline held by non-immune neighbors, over the touched part of the
    outline (as in `border`) and over the whole outline; and the touched (covered) share. NaN when untouched."""
    from claimbreak.contacts import boundary_shares, read_parquet_any
    ip, jp, sh, cov = boundary_shares(read_parquet_any(bp), D["cell_id"].to_numpy(), tol)
    held_non = np.bincount(ip[nonimm[jp]], weights=sh[nonimm[jp]], minlength=len(D))
    with np.errstate(invalid="ignore", divide="ignore"):
        touched = np.where(cov > 0, held_non / cov, np.nan)
    outline = np.where(cov > 0, np.clip(held_non, 0, 1), np.nan)
    return touched, outline, cov


def _share_bins(share):
    k = np.searchsorted(SHARE_BINS, np.nan_to_num(share, nan=-1), side="left")
    k = np.where(share <= 0, 1, k)
    return np.where(np.isfinite(share), np.array(SHARE_NAMES, dtype=object)[np.clip(k - 1, 0, 5)], "none")


def _wslope(g, prefix, min_cells):
    g = g[g.stratum.isin([f"{prefix}{b}" for b in SHARE_NAMES])]
    x = g.stratum.str.slice(len(prefix)).map(SHARE_MID).to_numpy(float)
    y, w = g.estimate.to_numpy(float), g.cells.to_numpy(float)
    ok = np.isfinite(y) & (w >= min_cells)
    return (float(np.polyfit(x[ok], y[ok], 1, w=np.sqrt(w[ok]))[0]), int(ok.sum())) if ok.sum() >= 3 else (np.nan, int(ok.sum()))


def _positives(D, cols, seed):
    pos, rows = {}, []
    for key in sorted({m for req, _ in CLASSES.values() for m in req}):
        if key in cols:
            p, d, thr = positive(D[cols[key]].to_numpy(), seed=seed)
            pos[key] = p
            rows.append(dict(marker=key, ashman_d=d, positive_share=float(p.mean()), threshold=thr))
    return pos, pd.DataFrame(rows)


def cmd_border(a):
    """Loss of the expected label by the non-immune (CD45-negative) share of the polygon perimeter."""
    from claimbreak.contacts import find_boundaries
    out = _fresh(a.out)
    lv = a.level
    L = pd.read_csv(Path(a.labels).expanduser() / f"{a.sample}.labels.csv.gz", keep_default_na=False,
                    dtype={"cell_id": str}, low_memory=False)
    P = pd.read_csv(a.protein, dtype={"cell_id": str})
    cols = {norm(c): c for c in P.columns if c != "cell_id"}
    D = L.merge(P, on="cell_id", how="inner").reset_index(drop=True)
    pos, M = _positives(D, cols, a.seed)
    M.to_csv(out / "markers.csv", index=False)
    imm_key = norm(a.immune_marker)
    if imm_key not in pos:
        raise SystemExit(f"immune marker {a.immune_marker} not among the proteins")
    bp = find_boundaries(a.boundaries, a.sample)
    if bp is None:
        raise SystemExit(f"no cell_boundaries.parquet for {a.sample} in {a.boundaries}")
    touched, outline, cov = _nonimm_share(D, ~pos[imm_key], bp, a.tol)
    base = getattr(a, "base", "touched")
    share = outline if base == "outline" else touched
    bins = _share_bins(share)
    strata = {f"nonimm_{b}": bins == b for b in SHARE_NAMES + ["none"]}
    strata["all"] = np.ones(len(D), bool)
    # the same bins within bins of the regional non-immune share (protein; cells whose centers lie
    # region[0]-region[1] um away), so the outline dose-response is read among cells with the same surroundings: if the
    # loss followed the region rather than the touching cells, the slope would vanish within regional bins
    nreg = int(getattr(a, "region_bins", 3) or 0)
    reg_edges, rbin = [], None
    if nreg > 1 and getattr(a, "region", ""):
        from claimbreak.compartment import regional_shares
        ring = tuple(float(x) for x in a.region.split(","))
        sh_r, _ = regional_shares(D[["x", "y"]].to_numpy(float), (~pos[imm_key]).astype(int), 2, ring)
        reg = sh_r[:, 1]
        reg_edges = list(np.nanquantile(reg, np.linspace(0, 1, nreg + 1)[1:-1]))
        rbin = np.where(np.isfinite(reg), np.searchsorted(reg_edges, np.nan_to_num(reg), side="right"), -1)
        for r in range(nreg):
            for b in SHARE_NAMES:
                strata[f"reg{r}_nonimm_{b}"] = (rbin == r) & (bins == b)
    arms = [x for x in ("nucB", "wholeB", "cytoB") if f"{lv}_{x}" in D]
    labels_all = sorted(set().union(*[set(D[f"{lv}_{x}"]) for x in arms]) - {""})
    rows = []
    for cname in [c.strip() for c in a.classes.split(",") if c.strip()]:
        req, rx = CLASSES[cname]
        if rx is None or not all(m in pos for m in req):
            continue
        cls = np.logical_and.reduce([pos[m] for m in req])
        target = [t for t in labels_all if re.search(rx, t)]
        hits = {arm: (np.isin(D[f"{lv}_{arm}"].to_numpy(), target), D[f"{lv}_{arm}"].to_numpy() != "") for arm in arms}
        for r_ in block_ci(D, cls, hits, strata, a.tile, a.block_draws, a.seed, pairs=(("nucB", "wholeB"),)):
            n = int((cls & strata[r_["stratum"]]).sum())
            rows.append(dict(protein_class=cname, cells=n, **r_))
    C = pd.DataFrame(rows)
    C.to_csv(out / "loss_by_share.csv", index=False)
    fits, rfits = [], []
    for cname, g in C[C.metric == "loss_wholeB_vs_nucB"].groupby("protein_class"):
        b1, nb = _wslope(g, "nonimm_", a.min_cells)
        if np.isfinite(b1):
            fits.append(dict(protein_class=cname, slope=b1, bins=nb))
            rfits.append(dict(protein_class=cname, region_bin="all", slope=b1, bins=nb,
                              cells=int(g[g.stratum.str.match(r"^nonimm_")].cells.sum())))
        wsum = wn = 0.0
        for r in range(nreg if rbin is not None else 0):
            gr = g[g.stratum.str.startswith(f"reg{r}_")]
            b1, nb = _wslope(gr, f"reg{r}_nonimm_", a.min_cells)
            n_r = int(gr.cells.sum())
            rfits.append(dict(protein_class=cname, region_bin=f"{r}", slope=b1, bins=nb, cells=n_r))
            if np.isfinite(b1):
                wsum, wn = wsum + b1 * n_r, wn + n_r
        if wn:
            rfits.append(dict(protein_class=cname, region_bin="within (cell-weighted mean)", slope=wsum / wn,
                              bins=np.nan, cells=int(wn)))
    F = pd.DataFrame(fits)
    FR = pd.DataFrame(rfits)
    if rbin is not None:
        FR.to_csv(out / "slope_regional.csv", index=False)
    F.to_csv(out / "slope.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(labels=str(a.labels), protein=str(a.protein), sample=a.sample,
                                                   boundaries=str(bp), tol=a.tol, immune_marker=a.immune_marker,
                                                   base=base, cells=len(D), covered=float(np.mean(cov > 0)),
                                                   region=getattr(a, "region", ""), region_edges=reg_edges,
                                                   share_median=float(np.nanmedian(share))), indent=1))
    with pd.option_context("display.width", 220):
        q = C[C.metric == "loss_wholeB_vs_nucB"][["protein_class", "stratum", "cells", "estimate", "lo", "hi"]]
        print(f"loss of the expected label (wholeB vs nucB) by the non-immune share of the "
              f"{'whole outline' if base == 'outline' else 'touched part of the outline'}:")
        print(q.round(3).to_string(index=False))
        if len(F):
            print("\nweighted slope of loss on the non-immune share:\n" + F.round(3).to_string(index=False))
        if rbin is not None and len(FR):
            print(f"\nthe same slope within bins of the regional non-immune share ({a.region} um ring; "
                  f"edges {np.round(reg_edges, 3).tolist()}); a slope that survives within bins follows the touching "
                  f"cells, not the region (slope_regional.csv):\n" + FR.round(3).to_string(index=False))
    print(f"wrote {out}")


GROUP_MAP = "T=(?i)\\bT\\b|T cell|T_cell|CD8|CD4|Treg|lymphocyte;B=(?i)\\bB cell|B_cell|\\bB\\b;" \
            "Myeloid=(?i)macrophage|monocyte|myeloid|dendritic|DC\\b"


def cmd_groups(a):
    """A published labeling scored against protein: marker shares per group, and class recall / precision."""
    out = _fresh(a.out)
    G = pd.read_csv(a.groups, dtype={"cell_id": str})
    gcol = [c for c in G.columns if c != "cell_id"][0]
    P = pd.read_csv(a.protein, dtype={"cell_id": str})
    cols = {norm(c): c for c in P.columns if c != "cell_id"}
    D = G.merge(P, on="cell_id", how="inner").reset_index(drop=True)
    print(f"{len(D):,} of {len(G):,} grouped cells have protein values; {D[gcol].nunique()} groups")
    pos, M = _positives(D, cols, a.seed)
    M.to_csv(out / "markers.csv", index=False)
    cls_masks = {c: np.logical_and.reduce([pos[m] for m in req]) for c, (req, rx) in CLASSES.items()
                 if rx is not None and all(m in pos for m in req)}
    share = pd.DataFrame({c: m for c, m in cls_masks.items()}).groupby(D[gcol]).mean()
    share.insert(0, "cells", D[gcol].value_counts())
    share.sort_values("cells", ascending=False).to_csv(out / "group_marker_shares.csv")
    rows = []
    for item in a.map.split(";"):
        if "=" not in item:
            continue
        cname, rx = item.split("=", 1)
        if cname not in cls_masks:
            continue
        hit = D[gcol].astype(str).str.contains(rx, regex=True).to_numpy()
        cls = cls_masks[cname]
        rows.append(dict(protein_class=cname, groups="|".join(sorted(D.loc[hit, gcol].astype(str).unique())),
                         recall=float(hit[cls].mean()) if cls.any() else np.nan,
                         precision=float(cls[hit].mean()) if hit.any() else np.nan,
                         called=int(hit.sum()), positive=int(cls.sum())))
    R = pd.DataFrame(rows)
    R.to_csv(out / "recall_precision.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print("share of each group positive for each protein class:")
        print(share.sort_values("cells", ascending=False).round(3).to_string())
        print("\nrecall and precision with the group map:\n" + R.drop(columns="groups").round(3).to_string(index=False))
    print(f"wrote {out}")


def cmd_groups_border(a):
    """a published labeling (e.g. 10x cell_groups.csv) and the claimbreak arms on the same cells, recall of
    protein classes by the non-immune share of the polygon outline. The share is computed on all cells (neighbors
    complete); recall only on cells the published labeling covers, so every arm is scored on the same cells."""
    from claimbreak.contacts import find_boundaries
    out = _fresh(a.out)
    lv = a.level
    L = pd.read_csv(Path(a.labels).expanduser() / f"{a.sample}.labels.csv.gz", keep_default_na=False,
                    dtype={"cell_id": str}, low_memory=False)
    P = pd.read_csv(a.protein, dtype={"cell_id": str})
    G = pd.read_csv(a.groups, dtype={"cell_id": str})
    gcol = [c for c in G.columns if c != "cell_id"][0]
    cols = {norm(c): c for c in P.columns if c != "cell_id"}
    D = L.merge(P, on="cell_id", how="inner").merge(G.rename(columns={gcol: "_group"}), on="cell_id", how="left")
    D = D.reset_index(drop=True)
    D["_group"] = D["_group"].fillna("").astype(str)
    grouped = (D["_group"] != "").to_numpy()
    print(f"{len(D):,} cells with labels and protein; {grouped.sum():,} in the published groups ({grouped.mean():.1%})")
    pos, M = _positives(D, cols, a.seed)
    M.to_csv(out / "markers.csv", index=False)
    imm_key = norm(a.immune_marker)
    if imm_key not in pos:
        raise SystemExit(f"immune marker {a.immune_marker} not among the proteins")
    bp = find_boundaries(a.boundaries, a.sample)
    if bp is None:
        raise SystemExit(f"no cell_boundaries.parquet for {a.sample} in {a.boundaries}")
    touched, outline, cov = _nonimm_share(D, ~pos[imm_key], bp, a.tol)
    strata = {"all": grouped.copy()}
    for base, sh in (("touched_", touched), ("outline_", outline)):
        bins = _share_bins(sh)
        for b in SHARE_NAMES + ["none"]:
            strata[base + b] = grouped & (bins == b)
    arms = [x for x in a.arms.split(",") if x and f"{lv}_{x}" in D]
    labels_all = sorted(set().union(*[set(D[f"{lv}_{x}"]) for x in arms]) - {""}) if arms else []
    gmap = dict(item.split("=", 1) for item in a.map.split(";") if "=" in item)
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",") if ":" in x]
    rows = []
    for cname in [c.strip() for c in a.classes.split(",") if c.strip()]:
        req, rx = CLASSES[cname]
        if rx is None or cname not in gmap or not all(m in pos for m in req):
            continue
        cls = np.logical_and.reduce([pos[m] for m in req])
        target = [t for t in labels_all if re.search(rx, t)]
        hits = {arm: (np.isin(D[f"{lv}_{arm}"].to_numpy(), target), D[f"{lv}_{arm}"].to_numpy() != "") for arm in arms}
        hits["pub"] = (D["_group"].str.contains(gmap[cname], regex=True).to_numpy() & grouped, grouped)
        pp = tuple(p for p in pairs if p[0] in hits and p[1] in hits)
        for r_ in block_ci(D, cls, hits, strata, a.tile, a.block_draws, a.seed, pairs=pp):
            rows.append(dict(protein_class=cname, cells=int((cls & strata[r_["stratum"]]).sum()), **r_))
    C = pd.DataFrame(rows)
    C.to_csv(out / "recall_by_share.csv", index=False)
    fits = []
    for (cname, metric), g in C[C.metric.str.startswith(("recall_", "loss_"))].groupby(["protein_class", "metric"]):
        for base in ("touched_", "outline_"):
            b1, nb = _wslope(g, base, a.min_cells)
            fits.append(dict(protein_class=cname, metric=metric, base=base.rstrip("_"), slope=b1, bins=nb))
    F = pd.DataFrame(fits)
    F.to_csv(out / "slope.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(labels=str(a.labels), protein=str(a.protein), groups=str(a.groups),
                                                   sample=a.sample, boundaries=str(bp), tol=a.tol,
                                                   immune_marker=a.immune_marker, arms=arms, pairs=[list(p) for p in pp],
                                                   map=gmap, cells=len(D), grouped=int(grouped.sum()),
                                                   covered=float(np.mean(cov > 0))), indent=1))
    with pd.option_context("display.width", 240, "display.max_rows", 400):
        for base in ("touched_", "outline_"):
            q = C[C.metric.str.startswith("recall_") & C.stratum.str.startswith((base, "all"))]
            print(f"\nrecall by the non-immune share, {'touched part of the outline' if base == 'touched_' else 'whole outline'} (grouped cells):")
            print(q.pivot_table(index=["protein_class", "stratum"], columns="metric", values="estimate",
                                sort=False).round(3).to_string())
        q = C[C.metric.str.startswith("loss_") & C.stratum.str.startswith(("outline_", "all"))]
        print("\nloss by the non-immune share of the whole outline (block bootstrap):")
        print(q[["protein_class", "metric", "stratum", "cells", "estimate", "lo", "hi"]].round(3).to_string(index=False))
        print("\nweighted slope on the share (recall: negative = falls with non-immune border):")
        print(F.round(3).to_string(index=False))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--outs", required=True); e.add_argument("--sample", required=True); e.add_argument("--out", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--labels", required=True); t.add_argument("--protein", required=True)
    t.add_argument("--sample", required=True); t.add_argument("--out", required=True)
    t.add_argument("--level", default="L1"); t.add_argument("--arms", default="nucleus,whole,nucA,nucB,wholeB,cytoB")
    t.add_argument("--k", type=int, default=6); t.add_argument("--r", type=float, default=30.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--posterior", type=float, default=0.5,
                   help="posterior of the highest mixture component needed for a positive call (0.5 as before; "
                        "0.9 for a stricter call)")
    t.add_argument("--strict", action="store_true",
                   help="add lineage-consistent classes (T_strict = CD3E+ CD45+ CD68- PanCK-, CD8_strict, "
                        "CD4_strict, B_strict) and write class_overlap.csv")
    t.add_argument("--partners", default="PanCK,CD68", help="protein markers whose positive cells define contact strata")
    t.add_argument("--alias", default="",
                   help="MARKER=COLUMN pairs (comma separated): read a class marker from another protein column, "
                        "e.g. CD3E=CD45RO_hp for MERSCOPE CD45RO co-staining (T class = CD45RO+ cells)")
    t.add_argument("--tile", type=float, default=200.0, help="tile side (um) for the block bootstrap")
    t.add_argument("--block-draws", type=int, default=1000, help="block bootstrap draws (0: none)")
    t.add_argument("--pairs", default="nucB:wholeB,nucleus:whole",
                   help="reference:test arm pairs for the loss and its block interval")
    t.add_argument("--hybrid", default="",
                   help="A:B pairs (comma separated): add the arm A+B, A's label where A labelled the cell, else "
                        "B's (e.g. nucleus:whole)")
    t.add_argument("--gaps", default="",
                   help="reference:base:model triples; share of the base-to-reference recall gap the model closes")
    b = sub.add_parser("border")
    b.add_argument("--labels", required=True); b.add_argument("--protein", required=True)
    b.add_argument("--boundaries", required=True); b.add_argument("--sample", required=True)
    b.add_argument("--out", required=True); b.add_argument("--level", default="L1")
    b.add_argument("--immune-marker", default="CD45"); b.add_argument("--tol", type=float, default=1.0)
    b.add_argument("--classes", default="T,CD8,CD4,B")
    b.add_argument("--tile", type=float, default=200.0); b.add_argument("--block-draws", type=int, default=1000)
    b.add_argument("--min-cells", type=int, default=200, help="cells a bin needs to enter the slope")
    b.add_argument("--base", choices=("touched", "outline"), default="touched",
                   help="share over the touched part of the outline or over the whole outline (as "
                        "the label-side border share of compartment exposure --boundaries)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--region", default="30,100",
                   help="ring (um) for the regional non-immune share; slopes are also fitted within its bins")
    b.add_argument("--region-bins", type=int, default=3, help="quantile bins of the regional share (0: off)")
    g = sub.add_parser("groups")
    g.add_argument("--groups", required=True, help="CSV: cell_id and one group column (e.g. 10x cell_groups.csv)")
    g.add_argument("--protein", required=True); g.add_argument("--out", required=True)
    g.add_argument("--map", default=GROUP_MAP, help="class=regex pairs separated by ';' mapping groups to classes")
    g.add_argument("--seed", type=int, default=0)
    gb = sub.add_parser("groups-border")
    gb.add_argument("--groups", required=True); gb.add_argument("--labels", required=True)
    gb.add_argument("--protein", required=True); gb.add_argument("--boundaries", required=True)
    gb.add_argument("--sample", required=True); gb.add_argument("--out", required=True)
    gb.add_argument("--level", default="L1"); gb.add_argument("--map", default=GROUP_MAP)
    gb.add_argument("--arms", default="nucleus,whole,nucB,wholeB")
    gb.add_argument("--pairs", default="nucleus:pub,nucleus:whole,nucB:wholeB",
                    help="reference:test pairs for the loss; 'pub' is the published labelling")
    gb.add_argument("--immune-marker", default="CD45"); gb.add_argument("--tol", type=float, default=1.0)
    gb.add_argument("--classes", default="T,CD8,CD4,B,Myeloid")
    gb.add_argument("--tile", type=float, default=200.0); gb.add_argument("--block-draws", type=int, default=1000)
    gb.add_argument("--min-cells", type=int, default=200); gb.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"extract": cmd_extract, "truth": cmd_truth, "border": cmd_border, "groups": cmd_groups,
     "groups-border": cmd_groups_border}[a.cmd](a)


if __name__ == "__main__":
    main()
