#!/usr/bin/env python
"""protein_region.py -- the region analysis of region_loss.py with protein truth (are protein-confirmed T cells inside tumor (or non-immune) regions the ones whose labels are lost?).

On a Xenium Protein section: T cells by protein (CD3E+, or --t-class T_strict); for each, the share of region cells
within --radius um among the other cells, region cells typed by protein (--region PANCK+ for tumor, CD45- for all
non-immune cells); bins as region_loss.py (away < 0.2, border, inside >= 0.5). Per bin and label arm: recall (share of
protein T cells the arm labels T/NK), and the recall ratio wholeB / nucB (equal depth) and whole / nucleus, with
95% tile-bootstrap intervals (paired: the same draws for every bin and arm), and the difference inside minus away of
the loss 1 - ratio. No label enters the T calls or the regions, so the "inside" T cells cannot be partner nuclei
mislabeled T.

  python protein_region.py --labels $R/xp_ccrcc/labels --protein .../XP_CCRCC.protein.csv.gz --sample XP_CCRCC \
         --out $R/xp_ccrcc/region_panck --region PANCK+

Writes region.csv (bin, arm, metric, estimate, lo, hi), cells.csv (bin: protein T cells) and meta.json.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protein_contacts import parse_partner, protein_masks, read_inputs, tiles  # noqa: E402
from region_loss import region_share  # noqa: E402
from xenium_protein import CLASSES, STRICT, _fresh  # noqa: E402


def _cb_version():
    try:
        import claimbreak
        return claimbreak.__version__
    except ImportError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True); ap.add_argument("--protein", required=True)
    ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="L1"); ap.add_argument("--arms", default="nucleus,whole,nucB,wholeB")
    ap.add_argument("--t-class", default="T", choices=["T", "T_strict"])
    ap.add_argument("--region", default="PANCK+", help="protein definition of region cells, e.g. PANCK+ or CD45-")
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--bins", default="0,0.2,0.5,1.0001"); ap.add_argument("--bin-names", default="away,border,inside")
    ap.add_argument("--tile", type=float, default=200.0); ap.add_argument("--block-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--posterior", type=float, default=0.5)
    a = ap.parse_args(argv)
    D, n_lab, cols = read_inputs(a.labels, a.protein, a.sample)
    lv = a.level
    arms = [x for x in a.arms.split(",") if f"{lv}_{x}" in D]
    if not arms:
        raise SystemExit(f"none of the arms {a.arms} in the labels")
    out = _fresh(a.out)
    reg_spec = parse_partner(a.region)
    req, neg = (CLASSES["T"][0], []) if a.t_class == "T" else (STRICT["T_strict"][0], STRICT["T_strict"][1])
    pos = protein_masks(D, cols, sorted(set(req + neg + [m for m, _ in reg_spec])), a.seed, a.posterior)
    t = np.logical_and.reduce([pos[m] for m in req])
    for m in neg:
        t &= ~pos[m]
    region = np.ones(len(D), bool)
    for m, want in reg_spec:
        region &= pos[m] if want else ~pos[m]
    if not region.any():
        raise SystemExit(f"no {a.region} cells (the marker does not separate in this section)")
    xy = D[["x", "y"]].to_numpy(float)
    idx = np.flatnonzero(t)
    share = region_share(xy, region, a.radius, idx)
    bins = [float(x) for x in a.bins.split(",")]
    names = a.bin_names.split(",")
    b = pd.cut(share, bins, labels=names, include_lowest=True, right=False).astype(str)
    rx = CLASSES["T"][1]
    hit = {arm: np.array([bool(re.search(rx, s)) if s else False for s in D[f"{lv}_{arm}"].astype(str).to_numpy()[idx]])
           for arm in arms}
    inv = tiles(xy[idx], a.tile)
    T = int(inv.max()) + 1
    rng = np.random.default_rng(a.seed)
    W = rng.multinomial(T, np.full(T, 1.0 / T), size=a.block_draws).astype(float) if a.block_draws else np.ones((0, T))
    W = np.vstack([np.ones((1, T)), W])
    rows = []

    def add(bin_, arm, metric, v):
        bb = v[1:][np.isfinite(v[1:])]
        rows.append(dict(bin=bin_, arm=arm, metric=metric, estimate=float(v[0]),
                         lo=float(np.quantile(bb, 0.025)) if len(bb) else np.nan,
                         hi=float(np.quantile(bb, 0.975)) if len(bb) else np.nan))
    rec = {}
    for nm in names + ["all"]:
        m = np.ones(len(idx), bool) if nm == "all" else (b == nm)
        n = W @ np.bincount(inv[m], minlength=T)
        for arm in arms:
            k = W @ np.bincount(inv[m & hit[arm]], minlength=T)
            with np.errstate(invalid="ignore", divide="ignore"):
                rec[(nm, arm)] = k / n
            add(nm, arm, "recall", rec[(nm, arm)])
        for test, ref in (("wholeB", "nucB"), ("whole", "nucleus")):
            if (nm, test) in rec and (nm, ref) in rec:
                with np.errstate(invalid="ignore", divide="ignore"):
                    add(nm, test, f"loss_pct_vs_{ref}", 100 * (1 - rec[(nm, test)] / rec[(nm, ref)]))
    for test, ref in (("wholeB", "nucB"), ("whole", "nucleus")):
        if all((x, y) in rec for x in (names[0], names[-1]) for y in (test, ref)):
            with np.errstate(invalid="ignore", divide="ignore"):
                li = 100 * (1 - rec[(names[-1], test)] / rec[(names[-1], ref)])
                la = 100 * (1 - rec[(names[0], test)] / rec[(names[0], ref)])
            add(f"{names[-1]} minus {names[0]}", test, f"loss_pct_vs_{ref}", li - la)
    R = pd.DataFrame(rows)
    R.to_csv(out / "region.csv", index=False)
    C = pd.Series(b).value_counts().rename_axis("bin").reset_index(name="protein_T_cells")
    C.to_csv(out / "cells.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(
        script="protein_region.py", version=_cb_version(), sample=a.sample, level=lv, arms=arms, t_class=a.t_class,
        region=a.region, region_share_of_cells=float(region.mean()), protein_t_cells=int(t.sum()), radius=a.radius,
        bins=bins, bin_names=names, tile=a.tile, draws=a.block_draws, seed=a.seed, posterior=a.posterior), indent=1))
    pd.set_option("display.width", 160)
    print(C.to_string(index=False))
    show = R[R.metric.str.startswith("loss_pct")]
    print(show.round(2).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
