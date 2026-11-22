#!/usr/bin/env python
"""protein_contacts.py -- T-cell contacts scored against protein on the same cells (28: the contact result of Fig. 4 rests on labels only; here the truth is protein).

On a Xenium Protein section (renal carcinomas 1 and 2), each cell has RNA labels from every arm and protein
intensities from the same cell. This script counts T-cell contacts with non-immune cells in three ways that share one
contact graph (6 nearest within 30 um over all cells with protein) and one partner set (cells CD45-negative by
protein, so the partners do not depend on any label):

  protein   T = CD3E+ by protein (xenium_protein.positive, the mixture call of `truth`); --t-class T_strict for
            CD3E+ CD45+ CD68- PanCK- cells
  <arm>     T = cells whose RNA label in the arm matches the T/NK pattern of xenium_protein.CLASSES["T"]
            (nucleus, whole, nucB, wholeB, and hybrid arms A+B from --hybrid, as in `truth --hybrid`)

For each it reports contacts per 100 partner cells (T cells touching >= 1 partner, per 100 partners; the readout of
`claimbreak compartment contacts`) and the contact share (share of T cells touching >= 1 partner), and the ratio of
each arm to protein, with block-bootstrap intervals over square tiles (as xenium_protein.block_ci). A label arm that
loses T cells at contact gives a ratio below 1 for per100 and a lower contact share than protein.

Variants: "all" (every cell with protein; a cell an arm did not label is not T in that arm) and "common" (cells
labeled in every arm, so nuclear and whole-cell arms are scored on the same cells).

  python protein_contacts.py --labels $R/xp_rcc/labels --protein $D/data/xenium_protein/h5ad/XP_RCC.protein.csv.gz \
         --sample XP_RCC --out $R/xp_rcc/protein_contacts --hybrid nucleus:whole

the label arms also against each other, with the same paired draws: wholeB / nucB (equal depth), whole /
nucleus, nucleus+whole / whole and nucleus+whole / nucleus (metrics per100_ratio_to_<ref>, share_ratio_to_<ref>). The
ratio to protein mixes two things, how many T cells a labeling calls and where it loses them; the contact share and the
ratios between arms separate them.

--confusion N adds the arm 'conf', the half-B labels relabeled at random N times with the section's own
confusion P(wholeB | nucB) (claimbreak.contacts.confusion_relabel, as in `compartment contacts --confusion` and Fig. 4a),
and the metrics change_pct (wholeB and conf against nucB) and position_pct = 100 (wholeB - conf) / nucB, the part of
the change that comes from where the label changes fall. --partner-from labels types partners by each arm's own labels
(non-immune), the label-only version of the same count; with protein partners, the partner set is fixed.

Writes contacts.csv (variant, arm, metric, estimate, lo, hi, tiles, draws), counts.csv (variant, arm, n_t, n_partner,
touching) and meta.json. Refuses to overwrite.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xenium_protein import CLASSES, STRICT, _fresh, norm, positive  # noqa: E402

from claimbreak.contacts import knn_edges  # noqa: E402


def _cb_version():
    try:
        import claimbreak
        return claimbreak.__version__
    except ImportError:
        return None


def read_inputs(labels, protein, sample):
    L = pd.read_csv(Path(labels).expanduser() / f"{sample}.labels.csv.gz", keep_default_na=False,
                    dtype={"cell_id": str}, low_memory=False)
    P = pd.read_csv(Path(protein).expanduser(), dtype={"cell_id": str})
    D = L.merge(P, on="cell_id", how="inner")
    if D.empty:
        raise SystemExit("no labelled cell has protein values (cell_id mismatch?)")
    return D, len(L), {norm(c): c for c in P.columns if c != "cell_id"}


def add_hybrids(D, lv, spec):
    names = []
    for h in [x for x in (spec or "").split(",") if x]:
        first, second = h.split(":")
        c1, c2 = f"{lv}_{first}", f"{lv}_{second}"
        if c1 not in D or c2 not in D:
            raise SystemExit(f"--hybrid {h}: needs columns {c1} and {c2}")
        name = f"{first}+{second}"
        D[f"{lv}_{name}"] = np.where(D[c1].astype(str) != "", D[c1].astype(str), D[c2].astype(str))
        names.append(name)
    return names


def protein_masks(D, cols, markers, seed, post):
    pos = {}
    for m in markers:
        if m not in cols:
            raise SystemExit(f"protein marker {m} missing (have {sorted(cols)})")
        pos[m] = positive(D[cols[m]].to_numpy(), seed=seed, post=post)[0]
    return pos


def parse_partner(spec):
    """'CD45-' or 'CD45-,PANCK+' -> [(marker, wanted sign)]"""
    out = []
    for x in [y.strip() for y in spec.split(",") if y.strip()]:
        if x[-1] not in "+-":
            raise SystemExit(f"--partner {x}: end each marker with + or -")
        out.append((norm(x[:-1]), x[-1] == "+"))
    return out


def touching(edges, t, part):
    """cells of t that touch >= 1 cell of part"""
    hit = np.zeros(len(t), bool)
    a, b = edges[:, 0], edges[:, 1]
    hit[a[part[b]]] = True
    hit[b[part[a]]] = True
    return t & hit


def tiles(xy, side):
    key = np.floor(xy[:, 0] / side).astype(np.int64) * 1_000_000 + np.floor(xy[:, 1] / side).astype(np.int64)
    return np.unique(key, return_inverse=True)[1]


PAIRS = (("wholeB", "nucB"), ("whole", "nucleus"), ("nucleus+whole", "whole"), ("nucleus+whole", "nucleus"),
         ("conf", "nucB"))


def score(arms_t, part, edges, inv, keep, draws, seed, pairs=PAIRS):
    """arms_t: {arm: T mask, or (T mask, partner mask), or a list of such pairs whose values are averaged (the
    confusion draws)}; part: the partner mask for arms given as a plain mask; keep: cells in the variant.
    Returns (rows, counts) with block intervals; the bootstrap weights are shared by every arm, so all ratios are
    paired."""
    T = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    W = rng.multinomial(T, np.full(T, 1.0 / T), size=draws).astype(float) if draws else np.ones((0, T))
    W = np.vstack([np.ones((1, T)), W])
    per, shr, rows, cnt = {}, {}, [], []
    for arm, spec in arms_t.items():
        items = spec if isinstance(spec, list) else [spec if isinstance(spec, tuple) else (spec, part)]
        ps, ss, c0 = [], [], None
        for t, pm in items:
            pk = pm & keep
            tk = t & keep
            tt = touching(edges, tk, pm) & keep
            n_p = W @ np.bincount(inv[pk], minlength=T)
            n_t, n_tt = W @ np.bincount(inv[tk], minlength=T), W @ np.bincount(inv[tt], minlength=T)
            with np.errstate(invalid="ignore", divide="ignore"):
                ps.append(100 * n_tt / n_p); ss.append(n_tt / n_t)
            if c0 is None:
                c0 = dict(arm=arm, n_t=int(tk.sum()), n_partner=int(pk.sum()), touching=int(tt.sum()))
        per[arm], shr[arm] = np.mean(ps, axis=0), np.mean(ss, axis=0)
        cnt.append(c0)

    def add(arm, metric, v):
        b = v[1:][np.isfinite(v[1:])]
        rows.append(dict(arm=arm, metric=metric, estimate=float(v[0]),
                         lo=float(np.quantile(b, 0.025)) if len(b) else np.nan,
                         hi=float(np.quantile(b, 0.975)) if len(b) else np.nan, tiles=T, draws=draws))
    for arm in arms_t:
        add(arm, "per100", per[arm])
        add(arm, "contact_share", shr[arm])
        if arm != "protein" and "protein" in arms_t:
            with np.errstate(invalid="ignore", divide="ignore"):
                add(arm, "per100_ratio_to_protein", per[arm] / per["protein"])
                add(arm, "share_ratio_to_protein", shr[arm] / shr["protein"])
    # label arms against each other (e.g. wholeB against nucB at equal depth), paired over the same draws
    for test, ref in pairs:
        if test in arms_t and ref in arms_t:
            with np.errstate(invalid="ignore", divide="ignore"):
                add(test, f"per100_ratio_to_{ref}", per[test] / per[ref])
                add(test, f"share_ratio_to_{ref}", shr[test] / shr[ref])
    # the part of the wholeB change that comes from where the changes fall, as in per100_conf.zsh:
    # 100 (wholeB - conf) / nucB, percentage points of the half-B value
    if all(x in arms_t for x in ("wholeB", "nucB", "conf")):
        with np.errstate(invalid="ignore", divide="ignore"):
            add("wholeB", "change_pct", 100 * (per["wholeB"] / per["nucB"] - 1))
            add("conf", "change_pct", 100 * (per["conf"] / per["nucB"] - 1))
            add("wholeB", "position_pct", 100 * (per["wholeB"] - per["conf"]) / per["nucB"])
    return rows, cnt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True); ap.add_argument("--protein", required=True)
    ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="L1"); ap.add_argument("--arms", default="nucleus,whole,nucB,wholeB")
    ap.add_argument("--hybrid", default="nucleus:whole", help="A:B pairs: arm A+B, A's label where A labelled the cell, else B's")
    ap.add_argument("--t-class", default="T", choices=["T", "T_strict"])
    ap.add_argument("--partner", default="CD45-", help="protein definition of partner cells, e.g. CD45- or PANCK+")
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--r", type=float, default=30.0)
    ap.add_argument("--tile", type=float, default=200.0); ap.add_argument("--block-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--posterior", type=float, default=0.5)
    ap.add_argument("--confusion", type=int, default=0,
                    help="draws of the half-B labels relabelled at random with the section's own confusion "
                         "P(wholeB | nucB) (arm 'conf'; as many changes of each kind as wholeB, placed at random)")
    ap.add_argument("--partner-from", default="protein", choices=["protein", "labels", "nucB"],
                    help="'labels' types partners by each arm's own labels (labelled and not immune), as the "
                         "label-only contact analyses do; the protein arm is then left out. 'nucB': partners "
                         "fixed at the non-immune half-B labels for every arm, the label-only analogue of protein partners")
    a = ap.parse_args(argv)

    D, n_lab, cols = read_inputs(a.labels, a.protein, a.sample)
    lv = a.level
    hyb = add_hybrids(D, lv, a.hybrid)
    arms = [x for x in a.arms.split(",") if f"{lv}_{x}" in D] + hyb
    missing = [x for x in a.arms.split(",") if f"{lv}_{x}" not in D]
    if not arms:
        raise SystemExit(f"none of the arms {a.arms} at level {lv} in the labels")
    out = _fresh(a.out)
    partner = parse_partner(a.partner)
    if a.t_class == "T":
        req, neg = CLASSES["T"][0], []
    else:
        req, neg = STRICT["T_strict"][0], STRICT["T_strict"][1]
    rx = CLASSES["T"][1]
    pos = protein_masks(D, cols, sorted(set(req + neg + [m for m, _ in partner])), a.seed, a.posterior)
    t_prot = np.logical_and.reduce([pos[m] for m in req])
    for m in neg:
        t_prot &= ~pos[m]
    part = np.ones(len(D), bool)
    for m, want in partner:
        part &= pos[m] if want else ~pos[m]
    arms_t = {"protein": t_prot}
    labelled = np.ones(len(D), bool)
    tlabels = set()
    for arm in arms:
        lab = D[f"{lv}_{arm}"].astype(str).to_numpy()
        hit = np.array([bool(re.search(rx, s)) if s else False for s in lab])
        tlabels |= set(lab[hit])
        arms_t[arm] = hit
        labelled &= lab != ""
    immune_rx = CLASSES["Immune"][1]
    nonimm = lambda lab: np.array([bool(x) and not re.search(immune_rx, x) for x in lab])
    if a.partner_from == "labels":
        del arms_t["protein"]
        for arm in list(arms_t):
            arms_t[arm] = (arms_t[arm], nonimm(D[f"{lv}_{arm}"].astype(str).to_numpy()))
    if a.partner_from == "nucB":
        if "nucB" not in arms:
            raise SystemExit("--partner-from nucB needs the arm nucB")
        part = nonimm(D[f"{lv}_nucB"].astype(str).to_numpy())
        del arms_t["protein"]
    if a.confusion:
        if not {"nucB", "wholeB"} <= set(arms):
            raise SystemExit("--confusion needs the arms nucB and wholeB")
        from claimbreak.contacts import confusion_relabel
        rb, tb = D[f"{lv}_nucB"].astype(str).to_numpy(), D[f"{lv}_wholeB"].astype(str).to_numpy()
        cats = np.array(sorted(set(rb) | set(tb)))
        rc, tcode = np.searchsorted(cats, rb), np.searchsorted(cats, tb)
        crng = np.random.default_rng([a.seed, 1788])
        conf = []
        for _ in range(a.confusion):
            lab = cats[confusion_relabel(rc, tcode, len(cats), crng)]
            tm = np.array([bool(re.search(rx, x)) if x else False for x in lab])
            conf.append((tm, nonimm(lab)) if a.partner_from == "labels" else (tm, part))
        arms_t["conf"] = conf
    xy = D[["x", "y"]].to_numpy(float)
    edges = knn_edges(xy, k=a.k, r=a.r)
    inv = tiles(xy, a.tile)
    rows, cnts = [], []
    for variant, keep in (("all", np.ones(len(D), bool)), ("common", labelled)):
        r_, c_ = score(arms_t, part, edges, inv, keep, a.block_draws, a.seed)
        rows += [dict(variant=variant, **x) for x in r_]
        cnts += [dict(variant=variant, **x) for x in c_]
    R = pd.DataFrame(rows)
    R.to_csv(out / "contacts.csv", index=False)
    pd.DataFrame(cnts).to_csv(out / "counts.csv", index=False)
    meta = dict(script="protein_contacts.py", version=_cb_version(), sample=a.sample, level=lv, arms=arms,
                missing_arms=missing, labelled_cells=n_lab, cells_with_protein=len(D),
                common_cells=int(labelled.sum()), t_class=a.t_class, t_labels=sorted(tlabels), partner=a.partner,
                partner_share=float(part.mean()), protein_t_share=float(t_prot.mean()), k=a.k, r=a.r, tile=a.tile,
                draws=a.block_draws, seed=a.seed, posterior=a.posterior, edges=int(len(edges)),
                confusion=a.confusion, partner_from=a.partner_from)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    show = R[(R.variant == "all") & (R.metric.isin(["per100", "contact_share", "change_pct", "position_pct"])
                                     | R.metric.str.contains("_ratio_to_"))]
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        print(show.drop(columns=["tiles", "draws"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
