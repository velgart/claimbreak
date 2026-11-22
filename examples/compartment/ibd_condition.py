#!/usr/bin/env python
"""ibd_condition.py -- intraepithelial lymphocytes by condition, counted from nuclear, whole-cell and hybrid labels
(conditions from the sample titles, contrasts within strata): does the label choice change an IBD condition contrast?

A user who counts intraepithelial lymphocytes (IELs) labels the cells once and counts T/NK cells that sit inside the
epithelium. This script does that with each labeling a user could choose, from the same cells:
  nucleus   the label from the full nuclear counts (<level>_nucleus)
  whole     the label from the full whole-cell counts (<level>_whole)
  hybrid    the nuclear label where the nucleus holds >= --min-nuc counts (n_nucleus), else the whole-cell label
For each section and labeling: epithelial cells (--partner), T/NK cells (--immune), and for each T/NK cell the share
of epithelial cells among the other cells within --radius um, from the same labeling; IEL = share >= --inside, lamina
propria T = share < --away. Pooled per patient (unit), the readouts are
  iel_per100_epi   IELs per 100 epithelial cells (the histology convention)
  iel_share        IELs over all T/NK cells
  t_per100         T/NK cells per 100 cells
Conditions come from the GEO series matrix (--meta: a file or a folder searched for *series_matrix*; every
!Sample_characteristics_ch1 "key: value" row, matched to patients by --unit-pattern in any sample field), or from
--meta-csv (columns unit and one column per condition). Every characteristic with 2-4 levels and at least
--min-per-level patients in at least two levels is compared: per labeling the median readout per level, the ratio of
medians (each level against a reference: --ref-level, else the one level that reads as unaffected,
such as non-inflamed, control or healthy, else the first in sorted order), a Mann-Whitney P, and a 95% bootstrap
interval over patients (resampled within levels, the same draws for every labeling), plus the ratio of that ratio
between labelings (whole / nucleus, hybrid / nucleus): if it differs from one, the label choice changes the contrast.

  python ibd_condition.py --labels $D/IBD/labels/xenium_v1 --meta $D/IBD/meta --out $R/ibd_condition

Characteristics are assigned per section (the GEO accession in the section name, else the patient's samples), so a
patient with sections of two conditions counts in both. Writes sections.csv, sections_characteristics.csv,
patients.csv (with the characteristics), contrasts.csv, meta.json. Refuses to overwrite.
"""
import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_loss import region_share  # noqa: E402
from claimbreak.placement import unit_from  # noqa: E402


def _cb_version():
    try:
        import claimbreak
        return claimbreak.__version__
    except ImportError:
        return None

ARMS = ("nucleus", "whole", "hybrid")
READOUTS = ("iel_per100_epi", "iel_share", "t_per100")


def _open(p):
    return gzip.open(p, "rt", errors="replace") if str(p).endswith(".gz") else open(p, errors="replace")


def read_series_matrix(paths, title_fields=None):
    """per GEO sample: all !Sample_ fields joined (for matching) and the characteristics as a dict (values lower
    case); with title_fields, the sample title before '[' split on '_' gives further characteristics, when it has
    exactly that many parts (GSE312420: 'Ileum_CD_Inflamed [I0294 ITDF00A Xenium337]')"""
    out = []
    for p in paths:
        rows = {}
        with _open(p) as fh:
            for ln in fh:
                if not ln.startswith("!Sample_"):
                    continue
                parts = ln.rstrip("\n").split("\t")
                key = parts[0][1:]
                vals = [v.strip().strip('"') for v in parts[1:]]
                rows.setdefault(key, []).append(vals)
        if "Sample_geo_accession" not in rows:
            continue
        gsm = rows["Sample_geo_accession"][0]
        for j, g in enumerate(gsm):
            text = " ".join(v[j] for vals in rows.values() for v in vals if j < len(v))
            ch = {}
            for v in rows.get("Sample_characteristics_ch1", []):
                if j < len(v) and ":" in v[j]:
                    k, x = v[j].split(":", 1)
                    ch[k.strip().lower()] = x.strip().lower()
            title = rows.get("Sample_title", [[""] * len(gsm)])[0][j]
            if title_fields:
                parts = [x.strip() for x in title.split("[")[0].strip().split("_")]
                if len(parts) == len(title_fields):
                    for k, x in zip(title_fields, parts):
                        ch.setdefault(k, x.lower())
            out.append(dict(gsm=g, title=title, text=text, file=Path(p).name, **{f"ch:{k}": x for k, x in ch.items()}))
    return pd.DataFrame(out)


def section_characteristics(M, sections, units):
    """one row per section: the GEO sample whose accession appears in the section name, else every sample whose
    fields mention the section's patient; each characteristic's value if those samples agree, else 'mixed'"""
    rows = []
    chs = [c for c in M.columns if c.startswith("ch:")]
    for sec, u in zip(sections, units):
        q = M.iloc[0:0]
        if len(M):
            g = re.findall(r"GSM\d+", sec)
            q = M[M.gsm.isin(g)] if g else q
            if not len(q):
                q = M[M.text.str.contains(re.escape(u), regex=True)]
        r = {"section": sec, "unit": u, "geo_samples": ";".join(q.gsm) if len(q) else ""}
        for c in chs:
            v = q[c].dropna().unique() if len(q) else []
            r[c[3:]] = v[0] if len(v) == 1 else ("mixed" if len(v) > 1 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def pool(S, by):
    G = S.groupby(by + ["arm"])[["cells", "all_cells", "epi", "t", "iel", "lp"]].sum().reset_index()
    R = pd.DataFrame([readouts(r) for r in G.to_dict("records")], index=G.index)
    G = pd.concat([G, R], axis=1)
    P = G.pivot_table(index=by, columns="arm", values=list(READOUTS) + ["iel", "epi", "t"], aggfunc="first")
    P.columns = [f"{x}_{y}" for x, y in P.columns]
    return P.reset_index()


def section_counts(D, lv, immune, partner, radius, inside, away, min_nuc):
    xy = D[["x", "y"]].to_numpy(float)
    nuc = D[f"{lv}_nucleus"].astype(str).to_numpy()
    whole = D[f"{lv}_whole"].astype(str).to_numpy()
    nn = pd.to_numeric(D["n_nucleus"], errors="coerce").fillna(0).to_numpy() if "n_nucleus" in D else None
    labs = {"nucleus": nuc, "whole": whole}
    if nn is not None:
        labs["hybrid"] = np.where((nn >= min_nuc) & (nuc != ""), nuc, whole)
    out = {}
    for arm, L in labs.items():
        epi = L == partner
        t = np.flatnonzero(L == immune)
        sh = region_share(xy, epi, radius, t) if len(t) else np.array([])
        out[arm] = dict(cells=int((L != "").sum()), all_cells=len(L), epi=int(epi.sum()), t=int(len(t)),
                        iel=int(np.nansum(sh >= inside)), lp=int(np.nansum(sh < away)))
    return out


def readouts(r):
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(iel_per100_epi=100 * r["iel"] / r["epi"] if r["epi"] else np.nan,
                    iel_share=r["iel"] / r["t"] if r["t"] else np.nan,
                    t_per100=100 * r["t"] / r["all_cells"] if r["all_cells"] else np.nan)


def contrasts(S, SC, keys, min_per_level, draws, seed, ref_level=None):
    """per characteristic, over all sections and within each level of every other characteristic (stratum)"""
    out = []
    for key in keys:
        out += _contrast_key(S, SC, key, "all", min_per_level, draws, seed, ref_level)
        for k2 in [k for k in keys if k != key]:
            for v in sorted(SC[k2].dropna().astype(str).unique()):
                if v == "mixed":
                    continue
                out += _contrast_key(S, SC[SC[k2].astype(str) == v], key, f"{k2}={v}", min_per_level, draws, seed,
                                     ref_level)
    return pd.DataFrame(out)


def _contrast_key(S, SC, key, stratum, min_per_level, draws, seed, ref_level=None):
    """sections pooled per patient and level (a patient with sections in two levels counts in both), then levels
    compared"""
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(seed)
    rows = []
    for key in [key]:
        X = S.merge(SC[["section", key]], on="section")
        X = X[X[key].notna() & (X[key].astype(str) != "mixed")]
        if not len(X):
            continue
        Q = pool(X, ["unit", key])
        arms = [a for a in ARMS if f"iel_per100_epi_{a}" in Q]
        lev = Q[key].value_counts()
        lev = lev[lev >= min_per_level]
        if not 2 <= Q[key].nunique() <= 4 or len(lev) < 2:
            continue
        levels = sorted(lev.index)
        ref_l = ref_level
        if not (ref_l and ref_l in levels):
            # reference: a level that reads as the unaffected one (non-inflamed, control, healthy, ...), if any
            base = [x for x in levels if re.match(r"(?i)\s*(non|un|no\b|not\b|control|healthy|normal|uninv)", x)]
            ref_l = base[0] if len(base) == 1 else None
        if ref_l and ref_l in levels:
            levels = [ref_l] + [x for x in levels if x != ref_l]
        ref = levels[0]
        groups = {l: Q[Q[key] == l] for l in levels}
        # the same bootstrap draws (patients resampled within each level) for every labeling and readout
        idx = {l: rng.integers(0, len(g), (draws, len(g))) for l, g in groups.items()}
        for l in levels[1:]:
            for rd in READOUTS:
                boot = {}
                for arm in arms:
                    col = f"{rd}_{arm}"
                    a_, b_ = groups[ref][col].to_numpy(float), groups[l][col].to_numpy(float)
                    if np.isfinite(a_).sum() < 2 or np.isfinite(b_).sum() < 2:
                        continue
                    ma, mb = np.nanmedian(a_), np.nanmedian(b_)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        bb = np.nanmedian(b_[idx[l]], axis=1) / np.nanmedian(a_[idx[ref]], axis=1)
                    boot[arm] = bb
                    try:
                        pv = mannwhitneyu(a_[np.isfinite(a_)], b_[np.isfinite(b_)]).pvalue
                    except ValueError:
                        pv = np.nan
                    rows.append(dict(characteristic=key, stratum=stratum, level=l, reference=ref, n_level=len(groups[l]),
                                     n_reference=len(groups[ref]), readout=rd, arm=arm, median_level=mb,
                                     median_reference=ma, ratio=mb / ma if ma else np.nan,
                                     lo=np.nanquantile(bb, 0.025), hi=np.nanquantile(bb, 0.975), p_mannwhitney=pv))
                for arm in [x for x in ("whole", "hybrid") if x in boot and "nucleus" in boot]:
                    r_a = [r for r in rows if r["characteristic"] == key and r["level"] == l and r["readout"] == rd
                           and r["arm"] in ARMS]
                    ra = {r["arm"]: r["ratio"] for r in r_a}
                    with np.errstate(invalid="ignore", divide="ignore"):
                        bb = boot[arm] / boot["nucleus"]
                    rows.append(dict(characteristic=key, stratum=stratum, level=l, reference=ref, n_level=len(groups[l]),
                                     n_reference=len(groups[ref]), readout=rd, arm=f"{arm}/nucleus",
                                     ratio=ra[arm] / ra["nucleus"] if ra.get("nucleus") else np.nan,
                                     lo=np.nanquantile(bb, 0.025), hi=np.nanquantile(bb, 0.975)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=None, help="series matrix file or folder (searched for *series_matrix*)")
    ap.add_argument("--meta-csv", default=None, help="CSV with a column unit and one column per condition")
    ap.add_argument("--level", default="L1"); ap.add_argument("--immune", default="T_NK")
    ap.add_argument("--partner", default="Epithelial"); ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--inside", type=float, default=0.5); ap.add_argument("--away", type=float, default=0.2)
    ap.add_argument("--min-nuc", type=int, default=10, help="hybrid: nuclear label where n_nucleus >= this")
    ap.add_argument("--unit-pattern", default=r"I0\d{3}")
    ap.add_argument("--exclude", default=r"(withSegment|NoSegment|Colon|5k)",
                    help="regex on section names left out (default: the extra panels and masks of four patients, "
                         "so that every patient is counted on the same panel and masks); '' keeps all")
    ap.add_argument("--title-fields", default="site,disease,inflammation",
                    help="names for the '_'-separated parts of the GEO sample title before '[' ('' to skip)")
    ap.add_argument("--min-per-level", type=int, default=3); ap.add_argument("--ref-level", default=None)
    ap.add_argument("--draws", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    files = sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz"))
    if not files:
        raise SystemExit(f"no labels files in {a.labels}")
    lv = a.level
    srows, skipped = [], []
    for f in files:
        name = f.name.split(".labels")[0]
        if a.exclude and re.search(a.exclude, name, re.I):
            skipped.append(name)
            continue
        cols = pd.read_csv(f, nrows=0).columns
        need = [f"{lv}_nucleus", f"{lv}_whole", "x", "y"]
        if any(c not in cols for c in need):
            skipped.append(name)
            continue
        D = pd.read_csv(f, usecols=lambda c: c in set(need) | {"sample", "n_nucleus", "unit", "donor"},
                        keep_default_na=False, low_memory=False)
        sample = str(D["sample"].iloc[0]) if "sample" in D else name
        fb = D["unit"].iloc[0] if "unit" in D else (D["donor"].iloc[0] if "donor" in D else None)
        unit = unit_from(sample, a.unit_pattern, fb)
        C = section_counts(D, lv, a.immune, a.partner, a.radius, a.inside, a.away, a.min_nuc)
        for arm, r in C.items():
            srows.append(dict(section=sample, unit=unit, arm=arm, **r, **readouts(r)))
        print(f"{sample} ({unit}): " + "; ".join(f"{arm} IEL {r['iel']} / epi {r['epi']}" for arm, r in C.items()),
              flush=True)
    if not srows:
        raise SystemExit(f"no usable sections (skipped {len(skipped)})")
    out.mkdir(parents=True, exist_ok=True)
    S = pd.DataFrame(srows)
    S.to_csv(out / "sections.csv", index=False)
    P = pool(S, ["unit"])
    # conditions, per section
    if a.meta_csv:
        Mc = pd.read_csv(a.meta_csv, dtype=str)
        on = "section" if "section" in Mc else "unit"
        SC = S[["section", "unit"]].drop_duplicates().merge(Mc, on=on, how="left")
        source = a.meta_csv
    elif a.meta:
        mp = Path(a.meta).expanduser()
        paths = sorted(mp.glob("*series_matrix*")) if mp.is_dir() else [mp]
        M = read_series_matrix(paths, [x for x in a.title_fields.split(",") if x] or None)
        sec = S[["section", "unit"]].drop_duplicates()
        SC = section_characteristics(M, sec.section.tolist(), sec.unit.tolist())
        source = ", ".join(p.name for p in paths)
    else:
        SC, source = S[["section", "unit"]].drop_duplicates(), "none"
    SC.to_csv(out / "sections_characteristics.csv", index=False)
    keys = [c for c in SC.columns if c not in ("section", "unit", "geo_samples")]
    for k in keys:   # per patient, the values of its sections
        P[k] = P.unit.map(SC.groupby("unit")[k].agg(lambda v: ";".join(sorted(set(map(str, v.dropna()))))))
    P.to_csv(out / "patients.csv", index=False)
    C = contrasts(S, SC, keys, a.min_per_level, a.draws, a.seed, a.ref_level) if keys else pd.DataFrame()
    if len(C):
        C.to_csv(out / "contrasts.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(
        script="ibd_condition.py", version=_cb_version(), labels=str(a.labels), meta=source, level=lv, immune=a.immune,
        partner=a.partner, radius=a.radius, inside=a.inside, away=a.away, min_nuc=a.min_nuc,
        unit_pattern=a.unit_pattern, exclude=a.exclude, skipped=skipped, patients=int(len(P)),
        characteristics=keys, min_per_level=a.min_per_level, draws=a.draws, seed=a.seed), indent=1))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    show = ["unit"] + [f"iel_per100_epi_{x}" for x in ARMS if f"iel_per100_epi_{x}" in P] + keys
    print(f"\n{len(P)} patients; characteristics from {source}: {', '.join(keys) if keys else 'none found'}")
    print(P[show].round(2).to_string(index=False))
    if len(C):
        print("\ncondition contrasts (ratio of medians, level over reference; 95% bootstrap over patients):")
        print(C[C.readout == "iel_per100_epi"].drop(columns=["readout"]).round(3).to_string(index=False))
        print("\nthe same for the IEL share of T/NK cells:")
        print(C[C.readout == "iel_share"].drop(columns=["readout"]).round(3).to_string(index=False))
    else:
        print("\nno characteristic with at least two levels of", a.min_per_level, "patients; see patients.csv")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
