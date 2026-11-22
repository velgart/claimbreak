#!/usr/bin/env python
"""segseries_summary.py -- what the mask does to labels, on the same cells (reexpand.py series;
read only except OUT and --combine-out).

Input: LABELS_ROOT/<mask>/<sample>.labels.csv.gz from `claimbreak compartment label` run on every mask folder of
reexpand.py (arms whole, nucleus, nucA, nucB, wholeB), optionally LABELS_ROOT/<mask>/summary/sections.csv from
`claimbreak compartment compare` + `summary`, and H5AD_ROOT/<sample>.cells.csv.gz (counts per mask).

Per mask, sample and type: cells labeled the type from the mask's whole-cell counts against the nucleus
(ratio_full = whole / nucleus), at equal depth (ratio_dm = wholeB / nucB) and the counting-noise null (nucB / nucA);
per mask and sample: label change rates (wholeB vs nucA, the null nucB vs nucA), the share of the mask's counts
outside the nucleus (pooled over cells), and the extra-change ratio at contact when the summary exists.
The nucleus arms are the same counts in every mask, so differences between masks come from the mask alone.

With --unit-map CSV (columns cell_id, unit; optionally sample) or --unit-from-h5ad DIR (obs column --unit-col of
every .h5ad there, e.g. the kidney pieces): rows per mask, sample AND unit, so that sections holding several patients
(kidney regions) are summarized per patient. Cells without a unit keep the sample's.

With --combine-out DIR --samples S1,S2: one labels file per sample with the columns ihc_truth.py reads
(<level>_nucleus from r0, <level>_whole and <level>_wholeB from the platform mask, <level>_<mask> and
<level>_<mask>B for every other mask), so `ihc_truth.py truth` scores every mask against protein in one run.

  python segseries_summary.py --labels-root ~/claimbreak_work/segser/labels --h5ad-root ~/claimbreak_work/segser/h5ad --level L2m \
      --unit-pattern 'L\\d(?=_L)' --out ~/claimbreak_work/segser/summary [--combine-out ~/claimbreak_work/segser/ihc_labels \
      --samples GSM9509145_NSCLC_lung_L1_L1,GSM9509149_NSCLC_lung_L3_L3]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def radius_of(mask):
    m = re.fullmatch(r"r([\d.]+)", mask)
    return float(m.group(1)) if m else np.nan


def unit_of(sample, pat):
    m = re.search(pat, sample)
    return m.group(0) if m else sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--h5ad-root", default="")
    ap.add_argument("--level", default="L2m")
    ap.add_argument("--unit-pattern", default=r"L\d(?=_L)")
    ap.add_argument("--focus", default="T_NK", help="type printed in the summary tables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit-map", default="", help="CSV with cell_id, unit (and optionally sample)")
    ap.add_argument("--unit-from-h5ad", default="", help="folder of .h5ad files whose obs give each cell's unit")
    ap.add_argument("--unit-col", default="unit", help="obs column of --unit-from-h5ad")
    ap.add_argument("--combine-out", default="")
    ap.add_argument("--samples", default="")
    a = ap.parse_args()
    root = Path(a.labels_root).expanduser()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    lv = a.level
    masks = sorted([p.name for p in root.iterdir() if p.is_dir() and list(p.glob("*.labels.csv.gz"))],
                   key=lambda m: (np.isnan(radius_of(m)), radius_of(m)))
    if not masks:
        raise SystemExit(f"no mask folders with labels in {root}")
    print("masks:", ", ".join(masks), flush=True)
    umap = load_unit_map(a.unit_map, a.unit_from_h5ad, a.unit_col)
    T, S = [], []
    for mask in masks:
        summ = root / mask / "summary" / "sections.csv"
        er = pd.read_csv(summ).set_index("sample") if summ.exists() else None
        for f in sorted((root / mask).glob("*.labels.csv.gz")):
            sample = f.name.replace(".labels.csv.gz", "")
            L_all = pd.read_csv(f, keep_default_na=False, low_memory=False, dtype={"cell_id": str},
                                usecols=lambda c: c in ("cell_id", "panel") or (c.startswith(lv + "_") and c.count("_") == 1))
            panel = str(L_all["panel"].iloc[0]) if "panel" in L_all else ""
            for unit, L in split_units(L_all, sample, umap, unit_of(sample, a.unit_pattern)):
                summarise_one(T, S, L, mask, sample, unit, panel, lv, er, a)
    finish(T, S, masks, out, a, root, lv)


def load_unit_map(csv, h5ad_dir, col):
    """cell_id (and sample, if given) -> unit, or None."""
    if csv:
        M = pd.read_csv(csv, dtype=str)
        return M
    if h5ad_dir:
        import anndata as ad
        parts = []
        for f in sorted(Path(h5ad_dir).expanduser().glob("*.h5ad")):
            A = ad.read_h5ad(f, backed="r")
            if col in A.obs:
                parts.append(pd.DataFrame({"cell_id": A.obs_names.astype(str), "unit": A.obs[col].astype(str).to_numpy()}))
            A.file.close()
        if not parts:
            raise SystemExit(f"no .h5ad in {h5ad_dir} has obs column {col}")
        M = pd.concat(parts, ignore_index=True).drop_duplicates()
        dup = M.cell_id.duplicated(keep=False)
        if dup.any():
            print(f"  {int(dup.sum())} cell ids with more than one unit: left without a unit", flush=True)
            M = M[~dup]
        return M
    return None


def split_units(L, sample, umap, default):
    """[(unit, rows)]: one group per unit of the map, cells without one under the sample's default unit."""
    if umap is None:
        return [(default, L)]
    M = umap[umap["sample"] == sample] if "sample" in umap and (umap["sample"] == sample).any() else umap
    u = L["cell_id"].map(M.drop_duplicates("cell_id").set_index("cell_id")["unit"]).fillna(default)
    return [(k, L[u == k].reset_index(drop=True)) for k in sorted(u.unique())]


def summarise_one(T, S, L, mask, sample, unit, panel, lv, er, a):
    """Per-type and per-section rows of one mask, sample and unit."""
    col = {arm: L[f"{lv}_{arm}"].astype(str) for arm in ("whole", "nucleus", "nucA", "nucB", "wholeB")
           if f"{lv}_{arm}" in L}
    if not {"whole", "nucleus", "nucA", "nucB", "wholeB"} <= set(col):
        print(f"  {mask}/{sample}: arms missing, skipped", flush=True)
        return
    cnt = {arm: v[v != ""].value_counts() for arm, v in col.items()}
    types = sorted(set().union(*[set(c.index) for c in cnt.values()]))
    # each ratio on the cells labeled in both of its arms (before, each arm counted all cells it
    # labeled, so cells below the count threshold in one arm entered one side only); *_all keep the old values
    both = {pr: (col[pr[0]] != "") & (col[pr[1]] != "") for pr in
            (("whole", "nucleus"), ("wholeB", "nucB"), ("nucB", "nucA"))}
    cc = {pr: (col[pr[0]][m].value_counts(), col[pr[1]][m].value_counts()) for pr, m in both.items()}

    def rc(pr, t):
        x, y = cc[pr][0].get(t, 0), cc[pr][1].get(t, 0)
        return x / y if y else np.nan

    for t in types:
        g = {arm: int(cnt[arm].get(t, 0)) for arm in cnt}
        T.append(dict(mask=mask, radius=radius_of(mask), sample=sample, unit=unit,
                      panel=panel, type=t, **{f"n_{k}": v for k, v in g.items()},
                      ratio_full=rc(("whole", "nucleus"), t), ratio_dm=rc(("wholeB", "nucB"), t),
                      ratio_null=rc(("nucB", "nucA"), t),
                      ratio_full_all=g["whole"] / g["nucleus"] if g["nucleus"] else np.nan,
                      ratio_dm_all=g["wholeB"] / g["nucB"] if g["nucB"] else np.nan,
                      ratio_null_all=g["nucB"] / g["nucA"] if g["nucA"] else np.nan,
                      common_dm=int(both[("wholeB", "nucB")].sum())))

    def chg(x, y):
        ok = (col[x] != "") & (col[y] != "")
        return float(np.mean(col[x][ok] != col[y][ok])) if ok.any() else np.nan

    row = dict(mask=mask, radius=radius_of(mask), sample=sample, unit=unit,
               panel=panel, cells=len(L), labelled_whole=float(np.mean(col["whole"] != "")),
               change_test=chg("wholeB", "nucA"), change_null=chg("nucB", "nucA"),
               change_full=chg("whole", "nucleus"))
    if er is not None and sample in er.index:
        row["extra_ratio_contact"] = float(er.loc[sample, "extra_ratio_contact"])
        row["extra_ratio_far_only"] = float(er.loc[sample, "extra_ratio_far_only"])
    if a.h5ad_root:
        cf = Path(a.h5ad_root).expanduser() / f"{sample}.cells.csv.gz"
        if cf.exists():
            C = pd.read_csv(cf, usecols=lambda c: c in ("cell_id", "n_nucleus", f"n_{mask}"), dtype={"cell_id": str})
            if "cell_id" in C and "cell_id" in L:
                C = C[C.cell_id.isin(set(L.cell_id))]
            if f"n_{mask}" in C:
                tot = C[f"n_{mask}"].sum()
                row["outside_share"] = float((tot - C.n_nucleus.sum()) / tot) if tot else np.nan
                row["median_counts"] = float(C[f"n_{mask}"].median())
    S.append(row)


def finish(T, S, masks, out, a, root, lv):
    T, S = pd.DataFrame(T), pd.DataFrame(S)
    T.to_csv(out / "series_types.csv", index=False)
    S.to_csv(out / "series_sections.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    f = T[T.type == a.focus]
    if len(f):
        for v in ("ratio_dm", "ratio_full", "ratio_null"):
            print(f"\n{a.focus} {v} (median over units; rows mask, columns panel)\n"
                  + f.groupby(["mask", "panel", "unit"])[v].median().groupby(["mask", "panel"]).median()
                  .unstack("panel").reindex(masks).round(3).to_string())
    for v in ("outside_share", "change_test", "change_null", "extra_ratio_contact"):
        if v in S:
            print(f"\n{v} (median over sections; rows mask, columns panel)\n"
                  + S.groupby(["mask", "panel"])[v].median().unstack("panel").reindex(masks).round(3).to_string())
    if a.combine_out:
        co = Path(a.combine_out).expanduser()
        co.mkdir(parents=True, exist_ok=True)
        if "r0" not in masks or "platform" not in masks:
            raise SystemExit("--combine-out needs the r0 and platform masks")
        for sample in [s for s in a.samples.split(",") if s]:
            if not (root / "r0" / f"{sample}.labels.csv.gz").exists():
                print(f"no labels for {sample}; not combined", flush=True)
                continue
            base = pd.read_csv(root / "r0" / f"{sample}.labels.csv.gz", keep_default_na=False, low_memory=False,
                               usecols=["sample", "cell_id", "x", "y", f"{lv}_nucleus", f"{lv}_nucB"], dtype={"cell_id": str})
            for mask in masks:
                if mask == "r0":
                    continue
                X = pd.read_csv(root / mask / f"{sample}.labels.csv.gz", keep_default_na=False, low_memory=False,
                                usecols=["cell_id", f"{lv}_whole", f"{lv}_wholeB"], dtype={"cell_id": str})
                nm = "" if mask == "platform" else f"_{mask}"
                X = X.rename(columns={f"{lv}_whole": f"{lv}_whole{nm}", f"{lv}_wholeB": f"{lv}_wholeB{nm}"})
                if nm:
                    X = X.rename(columns={f"{lv}_whole{nm}": f"{lv}_{mask}", f"{lv}_wholeB{nm}": f"{lv}_{mask}B"})
                base = base.merge(X, on="cell_id", how="left")
            base.fillna("").to_csv(co / f"{sample}.labels.csv.gz", index=False, compression="gzip")
            print(f"wrote {co / (sample + '.labels.csv.gz')} ({base.shape[1] - 4} label columns)", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
