#!/usr/bin/env python
"""regional_sim.py -- does the regionally matched extra-change ratio separate changes that follow the touching cells
from changes that follow the region? (Methods, matched expected share).

Synthetic sections: cells placed uniformly, types clustered in patches (Voronoi cells around random centers, 30% of
cells of a random type). The reference labels (nucA, and the neighbors' nucleus labels) are the true types; the null
(nucB) changes 10% of labels to a random type; the test (wholeB) changes 20% of labels by one of four mechanisms:

  random     a type drawn from the section composition (changes ignore position)
  regional   the type of a random cell 30-100 um away, never a touching one
  contact    the type of one of the six nearest cells within 30 um (the contact definition of the test)
  mix        half of the changes regional, half contact
  density    random types, but changes concentrated where cells are dense (the contact-count confound)

and `compartment.compare_section` scores the extra-change ratio at contact, plain and matched on regional composition
and contact count, exactly as on real data.

  python regional_sim.py --out ~/claimbreak_results/regional_sim [--seeds 0,1,2,3,4 --cells 60000]
Writes OUT/regional_sim.csv (one row per mechanism and seed) and prints medians and ranges. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MECHANISMS = ("random", "regional", "contact", "mix", "density")


def section(seed: int, mech: str, n: int = 60000, side: float = 3000.0, K: int = 5, p_null: float = 0.10,
            p_test: float = 0.20):
    """One synthetic labels table in the layout of `compartment label` (L1_nucleus, L1_nucA, L1_nucB, L1_wholeB)."""
    rng = np.random.default_rng(seed)
    if mech == "density":                      # a dense half and a sparse half
        n2 = n // 8
        xy = np.vstack([rng.uniform([0, 0], [side / 2, side], (n - n2, 2)),
                        rng.uniform([side / 2, 0], [side, side], (n2, 2))])
    else:
        xy = rng.uniform(0, side, (n, 2))
    centres = rng.uniform(0, side, (int(500 * (side / 3000.0) ** 2), 2))
    truth = rng.integers(0, K, len(centres))[cKDTree(centres).query(xy)[1]]
    truth = np.where(rng.random(n) < 0.3, rng.integers(0, K, n), truth)
    comp = np.bincount(truth, minlength=K) / n
    tree = cKDTree(xy)
    d, nn = tree.query(xy, k=7)
    d, nn = d[:, 1:], nn[:, 1:]
    ring = near = None
    if mech in ("regional", "mix"):
        ring, near = tree.query_ball_point(xy, 100.0), tree.query_ball_point(xy, 30.0)

    def relabel(p, how):
        if how == "density":
            dense = (d <= 30).sum(1) >= 6
            p = np.where(dense, p * 1.5, p * 0.5)
        ch = rng.random(n) < p
        new = rng.choice(K, n, p=comp)
        for i in np.flatnonzero(ch):
            m = how if how != "mix" else ("regional" if rng.random() < 0.5 else "contact")
            if m == "regional":
                pool = np.setdiff1d(ring[i], near[i])
            elif m == "contact":
                pool = nn[i][d[i] <= 30]
            else:
                continue
            if len(pool):
                new[i] = truth[rng.choice(pool)]
        return np.where(ch & (new != truth), new, truth)
    names = np.array([f"t{k}" for k in range(K)])
    return pd.DataFrame({"sample": f"sim_{mech}_{seed}", "x": xy[:, 0], "y": xy[:, 1], "L1_nucleus": names[truth],
                         "L1_nucA": names[truth], "L1_nucB": names[relabel(np.full(n, p_null), "random")],
                         "L1_wholeB": names[relabel(np.full(n, p_test), mech)], "L1_nucA_margin": 1.0})


def score(D, **kw):
    """Plain and matched extra-change ratio at contact of one section (wholeB|nucA over nucB|nucA)."""
    from claimbreak.compartment import compare_section, extra_ratio
    _, S = compare_section(D, ["L1"], [("nucB", "nucA"), ("wholeB", "nucA")], **kw)
    S = pd.DataFrame(S).set_index("comparison")
    t, z = S.loc["wholeB|nucA"], S.loc["nucB|nucA"]
    er = lambda o, e: extra_ratio(t[o], t[e], t.change_rate, z[o], z[e], z.change_rate)
    return float(er("obs_contact", "exp_contact")), float(er("obs_contact_reg", "exp_contact_reg"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--cells", type=int, default=60000)
    ap.add_argument("--mechanisms", default=",".join(MECHANISMS))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mech in a.mechanisms.split(","):
        for s in [int(x) for x in a.seeds.split(",")]:
            plain, matched = score(section(s, mech, n=a.cells))
            rows.append(dict(mechanism=mech, seed=s, cells=a.cells, plain=plain, matched=matched))
            print(f"{mech} seed {s}: plain {plain:.3f}, matched {matched:.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "regional_sim.csv", index=False)
    S = R.groupby("mechanism")[["plain", "matched"]].agg(["median", "min", "max"]).round(3)
    with pd.option_context("display.width", 160):
        print("\n" + S.to_string())
    print(f"wrote {out}/regional_sim.csv")


if __name__ == "__main__":
    main()
