#!/usr/bin/env python
"""corr_readouts.py -- readouts C1-C6 from the labels
written by corr_import.py and the `claimbreak compartment contacts` runs on them.

Absorbed cells: nucleus label immune (--immune), whole-cell label a non-immune type that is among the cell's contact
neighbors' types (6 nearest cells within 30 um, neighbors typed by nucleus labels). Per donor and arm:
  recovery    share of absorbed cells whose label in the arm equals the nucleus label (cells unlabeled in the arm
              count as not recovered; the share labeled is reported)
  stability   share of cells with whole == nucleus whose label in the arm is still that label
  rm_nuc, rm_whole   (split, admix) median share of the cell's counts removed in the marker genes of its nucleus type and
              of its whole-cell type, absorbed cells; donors where rm_nuc > rm_whole
and, from the contacts runs (--contacts ARM=DIR,...), per100 of --pair (default T_NK|Tumor) for the nucleus labels,
the arm, and random relabeling of the arm at 10%.

  python corr_readouts.py --labels ~/claimbreak_results/bilous_corr_v1 --contacts whole=DIR,split=DIR,... --out DIR
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ARMS = ["rerun", "split", "splitsp", "shiftc", "shift", "rctd", "admix"]


def absorbed(D, lv, immune, k=6, r=30.0):
    nuc, who = D[f"{lv}_nucleus"].astype(str).to_numpy(), D[f"{lv}_whole"].astype(str).to_numpy()
    xy = D[["x", "y"]].to_numpy(float)
    d, idx = cKDTree(xy).query(xy, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    nb = np.where(d <= r, nuc[idx], "")
    touch = (nb == who[:, None]).any(1)
    return np.isin(nuc, immune) & (who != "") & ~np.isin(who, immune) & touch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--contacts", default="", help="ARM=DIR,... (contacts runs with --ref nucleus --test ARM)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="L2m")
    ap.add_argument("--immune", default="T_NK,Myeloid,B")
    ap.add_argument("--unit-pattern", default=r"L\d(?=_L)")
    ap.add_argument("--pair", default="T_NK|Tumour")
    a = ap.parse_args()
    lv, immune = a.level, a.immune.split(",")
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz")):
        D = pd.read_csv(f, keep_default_na=False, low_memory=False)
        for c in [c for c in D.columns if c.startswith("rm_")]:     # empty strings (no value) -> NaN
            D[c] = pd.to_numeric(D[c], errors="coerce")
        sample = f.name.split(".labels")[0]
        unit = re.search(a.unit_pattern, sample).group(0)
        D = D[(D[f"{lv}_nucleus"].astype(str) != "") & (D[f"{lv}_whole"].astype(str) != "")].reset_index(drop=True)
        ab = absorbed(D, lv, immune)
        agree = D[f"{lv}_nucleus"].astype(str) == D[f"{lv}_whole"].astype(str)
        for arm in ARMS:
            c = f"{lv}_{arm}"
            if c not in D:
                continue
            x = D[c].astype(str)
            r = dict(sample=sample, unit=unit, arm=arm, absorbed=int(ab.sum()),
                     recovered=int((x[ab] == D.loc[ab, f"{lv}_nucleus"].astype(str)).sum()),
                     absorbed_labelled=int((x[ab] != "").sum()),
                     agree_cells=int(agree.sum()), stable=int((x[agree] == D.loc[agree, f"{lv}_whole"].astype(str)).sum()))
            for m in ("nuc", "whole"):
                col = f"rm_{m}_{arm}"
                r[f"rm_{m}_median"] = float(np.nanmedian(D.loc[ab, col])) if col in D and ab.any() else np.nan
            rows.append(r)
    imp = Path(a.labels).expanduser() / "import_summary.csv"
    if imp.exists():                                         # carried along for the harvest
        pd.read_csv(imp).to_csv(out / "import_summary.csv", index=False)
    S = pd.DataFrame(rows)
    S.to_csv(out / "sections.csv", index=False)
    U = S.groupby(["unit", "arm"])[["absorbed", "recovered", "absorbed_labelled", "agree_cells", "stable"]].sum()
    U["recovery"] = U.recovered / U.absorbed.clip(lower=1)
    U["labelled_share"] = U.absorbed_labelled / U.absorbed.clip(lower=1)
    U["stability"] = U.stable / U.agree_cells.clip(lower=1)
    rm = S.groupby(["unit", "arm"])[["rm_nuc_median", "rm_whole_median"]].median()
    U = U.join(rm).reset_index()
    U.to_csv(out / "units.csv", index=False)
    pd.set_option("display.width", 220)
    print("per donor and arm (absorbed cells: nucleus immune, whole-cell label = a touching non-immune type):")
    print(U.round(3).to_string(index=False))

    P = None
    if a.contacts:
        recs = []
        for item in [x for x in a.contacts.split(",") if x]:
            arm, d = item.split("=", 1)
            C = pd.read_csv(Path(d).expanduser() / "units.csv")
            col = f"per100|{a.pair}"
            for _, g in C.groupby("unit"):
                v = g.set_index("labels")[col]
                recs.append(dict(unit=g.unit.iloc[0], arm=arm, nucleus=v.get("ref"), arm_value=v.get("test"),
                                 random_10=v.get("random_10")))
        P = pd.DataFrame(recs)
        P.to_csv(out / "per100.csv", index=False)
        print(f"\nper100 {a.pair} by donor (nucleus labels, arm, random 10% of the arm):")
        print(P.round(3).to_string(index=False))

    # verdicts
    def n_units(mask):
        return int(mask.sum()), int(len(mask))
    V = []
    W = U.set_index(["unit", "arm"])
    units = sorted(U.unit.unique())

    def per_unit(arm, fn):
        vals = [fn(W.loc[(u, arm)]) for u in units if (u, arm) in W.index]
        return np.array(vals, bool)

    if "split" in U.arm.values:
        m = per_unit("split", lambda r: r.recovery < 0.25)
        V.append(("C1a split recovery < 0.25", *n_units(m)))
    if P is not None and "split" in set(P.arm) and "whole" in set(P.arm):
        pw = P[P.arm == "whole"].set_index("unit").arm_value
        ps = P[P.arm == "split"].set_index("unit")
        ok = [u for u in ps.index if u in pw.index]
        m = np.array([(abs(ps.loc[u, "arm_value"] - pw[u]) <= 0.2 * pw[u]) and (ps.loc[u, "arm_value"] < ps.loc[u, "nucleus"])
                      for u in ok], bool)
        V.append(("C1b split per100 within 20% of whole and below nucleus", *n_units(m)))
    if "shift" in U.arm.values:
        V.append(("C2 shift recovery < 0.5", *n_units(per_unit("shift", lambda r: r.recovery < 0.5))))
    if "admix" in U.arm.values:
        ev = per_unit("admix", lambda r: np.isfinite(r.rm_nuc_median) and np.isfinite(r.rm_whole_median)
                      and (r.rm_nuc_median > 0 or r.rm_whole_median > 0))
        m = per_unit("admix", lambda r: r.rm_nuc_median > r.rm_whole_median)
        V.append(("C3a admix rm_nuc > rm_whole (donors where cellAdmix removed counts)", int(m[ev].sum()), int(ev.sum())))
        m = per_unit("admix", lambda r: r.recovery < 0.1)
        V.append(("C3b admix recovery < 0.1 (donors where cellAdmix removed counts)", int(m[ev].sum()), int(ev.sum())))
    if P is not None and "rctd" in set(P.arm):
        pr = P[P.arm == "rctd"]
        V.append(("C4 rctd per100 below nucleus", *n_units((pr.arm_value < pr.nucleus).to_numpy())))
    if P is not None:
        for arm in ("split", "admix", "shift"):
            pa = P[P.arm == arm]
            if len(pa):
                V.append((f"C5 {arm}: below nucleus and random 10% above the arm",
                          *n_units(((pa.arm_value < pa.nucleus) & (pa.random_10 > pa.arm_value)).to_numpy())))
    for arm in ("split", "admix"):
        if arm in U.arm.values:
            V.append((f"C6 {arm} stability >= 0.9", *n_units(per_unit(arm, lambda r: r.stability >= 0.9))))
    Vt = pd.DataFrame(V, columns=["readout", "donors_meeting", "donors"])
    Vt["held (>= 3 of 4)"] = (Vt.donors > 0) & (Vt.donors_meeting >= np.minimum(3, Vt.donors))
    Vt.to_csv(out / "verdicts.csv", index=False)
    print("\nverdicts:")
    print(Vt.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
