#!/usr/bin/env python
"""compartment_variants.py -- the same patient measured several ways (panel, segmentation stain, platform): label
effects of whole-cell counts side by side (read only).

IBD (Li et al. 2026, GSE312420): patients I0400, I0355, I0380, I0387 have serial sections run as Xenium 337 without
the segmentation stain, 337 with it, the Colon panel and 5K; the biology is (nearly) fixed and the measurement varies.
For every labels file (`claimbreak compartment label`, arms whole, nucleus, nucA, nucB, wholeB) the script computes,
per sample and type: whole / nucleus, wholeB / nucB (equal depth) and the null nucB / nucA; per sample: label change
rates (wholeB vs nucA, nucB vs nucA), median counts used and the share of counts outside the nucleus (from the label
file's n_whole / n_nucleus), and, with --summary, the extra-change ratio at contact (compartment summary sections.csv).
Rows are grouped by unit (--unit-pattern on the sample name) and --group (a column of the labels file, default panel).

  python compartment_variants.py --labels ~/claimbreak_data/IBD/labels/xenium_v1 --level L1 \
      --unit-pattern 'I0\\d{3}' --units I0400,I0355,I0380,I0387 --summary ~/claimbreak_results/ibd_v1/summary \
      --out ~/claimbreak_results/ibd_variants
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--level", default="L1")
    ap.add_argument("--group", default="panel")
    ap.add_argument("--unit-pattern", default=r"I0\d{3}")
    ap.add_argument("--units", default="", help="comma list of units to keep (default all)")
    ap.add_argument("--summary", default="", help="folder of `compartment summary` (sections.csv)")
    ap.add_argument("--focus", default="T_NK,Myeloid,B,Plasma")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    lv = a.level
    keep_units = set(u for u in a.units.split(",") if u)
    er = None
    if a.summary:
        f = Path(a.summary).expanduser() / "sections.csv"
        er = pd.read_csv(f).set_index("sample") if f.exists() else None
    T, S = [], []
    for f in sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz")):
        sample = f.name.replace(".labels.csv.gz", "")
        m = re.search(a.unit_pattern, sample)
        unit = m.group(0) if m else sample
        if keep_units and unit not in keep_units:
            continue
        L = pd.read_csv(f, keep_default_na=False, low_memory=False)
        need = [f"{lv}_{x}" for x in ("whole", "nucleus", "nucA", "nucB", "wholeB")]
        if not set(need) <= set(L.columns):
            print(f"{sample}: arms missing, skipped")
            continue
        grp = str(L[a.group].iloc[0]) if a.group in L else ""
        col = {x: L[f"{lv}_{x}"].astype(str) for x in ("whole", "nucleus", "nucA", "nucB", "wholeB")}
        cnt = {x: v[v != ""].value_counts() for x, v in col.items()}
        for t in sorted(set().union(*[set(c.index) for c in cnt.values()])):
            g = {x: int(cnt[x].get(t, 0)) for x in cnt}
            T.append(dict(sample=sample, unit=unit, group=grp, type=t, **{f"n_{x}": v for x, v in g.items()},
                          ratio_full=g["whole"] / g["nucleus"] if g["nucleus"] else np.nan,
                          ratio_dm=g["wholeB"] / g["nucB"] if g["nucB"] else np.nan,
                          ratio_null=g["nucB"] / g["nucA"] if g["nucA"] else np.nan))

        def chg(x, y):
            ok = (col[x] != "") & (col[y] != "")
            return float(np.mean(col[x][ok] != col[y][ok])) if ok.any() else np.nan

        nw = pd.to_numeric(L.get("n_whole"), errors="coerce") if "n_whole" in L else pd.Series(dtype=float)
        nn = pd.to_numeric(L.get("n_nucleus"), errors="coerce") if "n_nucleus" in L else pd.Series(dtype=float)
        row = dict(sample=sample, unit=unit, group=grp, cells=len(L), labelled_whole=float(np.mean(col["whole"] != "")),
                   labelled_nucB=float(np.mean(col["nucB"] != "")),
                   median_counts_whole=float(nw.median()) if len(nw) else np.nan,
                   outside_share=float(1 - nn.sum() / nw.sum()) if len(nw) and nw.sum() else np.nan,
                   change_test=chg("wholeB", "nucA"), change_null=chg("nucB", "nucA"))
        if er is not None and sample in er.index:
            row["extra_ratio_contact"] = float(er.loc[sample, "extra_ratio_contact"])
        S.append(row)
    T, S = pd.DataFrame(T), pd.DataFrame(S)
    if T.empty:
        raise SystemExit("no labels files matched")
    T.to_csv(out / "variants_types.csv", index=False)
    S.to_csv(out / "variants_sections.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    multi = S.groupby("unit").group.nunique()
    multi = multi[multi > 1].index
    print(f"units measured more than one way: {', '.join(multi) if len(multi) else 'none'}")
    sc = [c for c in ("cells", "median_counts_whole", "outside_share", "labelled_nucB", "change_test", "change_null",
                      "extra_ratio_contact") if c in S]
    print("\nper unit and group (medians over sections)\n"
          + S[S.unit.isin(multi)].groupby(["unit", "group"])[sc].median().round(3).to_string())
    for t in [x for x in a.focus.split(",") if x]:
        f = T[(T.type == t) & T.unit.isin(multi)]
        if len(f):
            print(f"\n{t}: wholeB / nucB (equal depth); rows unit, columns group\n"
                  + f.groupby(["unit", "group"]).ratio_dm.median().unstack("group").round(3).to_string())
    print(f"\nall units, {', '.join(a.focus.split(','))}: wholeB / nucB by group (median over units)\n"
          + T[T.type.isin(a.focus.split(","))].groupby(["type", "group", "unit"]).ratio_dm.median()
          .groupby(["type", "group"]).median().unstack("group").round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
