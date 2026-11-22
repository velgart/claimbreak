#!/usr/bin/env python
"""per100_check.py -- why one donor's T/NK per 100 tumor cells is far from the others (lung L3 gives 87.96 in the corrections table, L1/L2/L4 6-11).

For every labels folder given (e.g. the correction windows, bilous_corr_v1, and the full sections, bilous_v4) and every
lung section in it: cells, tumor and T/NK cells by each arm, T/NK cells touching tumor (contacts.knn_edges: 6
nearest within 30 um, symmetrised, as `compartment contacts`) and per100 = 100 x touching / tumor cells; then the same
summed per donor. With --windows, the window of each section (corr_export.py window.json: half-width, cells in the
window, cells in the section) is added. Reading: a window holding few tumor cells gives a large per100 from a small
denominator; per100 of the full section shows whether the donor itself is unusual.

  python per100_check.py --labels ~/claimbreak_work/ihc/bilous_corr_v1,~/claimbreak_work/ihc/bilous_v4 --arms nucleus,whole \
      [--windows /mnt/e/fibrosis/Bilous_corr/input] --out DIR
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from claimbreak.contacts import knn_edges


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="comma list of labels folders")
    ap.add_argument("--arms", default="nucleus,whole")
    ap.add_argument("--level", default="L2m")
    ap.add_argument("--immune", default="T_NK"); ap.add_argument("--partner", default="Tumour")
    ap.add_argument("--windows", default="", help="folder with <sample>/window.json (corr_export.py)")
    ap.add_argument("--unit-pattern", default=r"L\d(?=_L)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ld in [Path(x).expanduser() for x in a.labels.split(",") if x]:
        for f in sorted(ld.glob("*.labels.csv.gz")):
            s = f.name.replace(".labels.csv.gz", "")
            u = re.search(a.unit_pattern, s)
            cols = ["x", "y"] + [f"{a.level}_{x}" for x in a.arms.split(",")]
            L = pd.read_csv(f, usecols=lambda c: c in cols, keep_default_na=False)
            E = knn_edges(L[["x", "y"]].to_numpy(float))
            r = {"labels": ld.name, "sample": s, "unit": u.group(0) if u else s, "cells": len(L)}
            if a.windows:
                w = Path(a.windows).expanduser() / s / "window.json"
                if w.exists():
                    W = json.loads(w.read_text())
                    r.update({k: W.get(k) for k in ("cells_section", "cells_window", "half_width")})
            for arm in a.arms.split(","):
                c = f"{a.level}_{arm}"
                if c not in L:
                    continue
                lab = L[c].astype(str).to_numpy()
                imm, par = lab == a.immune, lab == a.partner
                touch = np.zeros(len(L), bool)
                for i, j in ((E[:, 0], E[:, 1]), (E[:, 1], E[:, 0])):
                    touch[i[par[j]]] = True
                r[f"{arm}_tumour"] = int(par.sum())
                r[f"{arm}_tnk"] = int(imm.sum())
                r[f"{arm}_tnk_touch"] = int((imm & touch).sum())
                r[f"{arm}_per100"] = 100 * (imm & touch).sum() / par.sum() if par.sum() else np.nan
                r[f"{arm}_tumour_share"] = float(par.mean())
            rows.append(r)
            print(f"{ld.name} {s}: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                                for k, v in r.items() if k not in ("labels", "sample")), flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(out / "per100_sections.csv", index=False)
    num = [c for c in S.columns if c.endswith(("_tumour", "_tnk", "_tnk_touch")) or c == "cells"]
    U = S.groupby(["labels", "unit"])[num].sum().reset_index()
    for arm in a.arms.split(","):
        if f"{arm}_tumour" in U:
            U[f"{arm}_per100"] = 100 * U[f"{arm}_tnk_touch"] / U[f"{arm}_tumour"].where(U[f"{arm}_tumour"] > 0)
    U.to_csv(out / "per100_units.csv", index=False)
    with pd.option_context("display.width", 220):
        print("\nper donor (sums over sections):\n" + U.round(2).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
