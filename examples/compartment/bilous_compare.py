"""Do label changes between count sets land on the neighbors' types? (GSE311609)

Input: OUT/<sample>.labels.csv.gz from bilous_label.py. For each section, level and comparison (test arm T against
reference arm R; default comparisons whole|nucleus, thinned|nucleus, cyto|nucleus, whole|thinned):
  cells: labeled in both arms (and R margin >= --min-margin); neighbors: the --k-contact nearest cells (ranks 1..k)
  within --radius um, and the far shell (ranks k+1..--k-far) within --radius-far um, typed by the --nbr-arm labels
  (default nucleus) of labeled neighbors.
  For a cell with R = a and a candidate type b != a, stratum = 'contact' (b among contact neighbors), 'far_only'
  (b among far-shell neighbors only) or 'absent'.
Outputs (OUT/compare_<tag>/):
  pairs.csv    per sample, level, comparison, a, b, stratum: cells n, changes to b, rate
  summary.csv  per sample, level, comparison: cells, change rate, among changed cells the share whose new type is a
               contact neighbor's type (observed) and the share expected if changes ignored the neighbors (the mean,
               over changed cells, of P(b among contact neighbors | R = a) in that section); ratio observed/expected;
               the same for far_only.
  The ratio is NOT 1 without contamination: where the reference-arm label is wrong by noise, an independent arm tends
  to restore the true type, which clusters with the neighbors (synthetic clean data: 1.4-1.9). Read every test against
  the null comparison (--null, default nucB|nucA: two independent halves of the nucleus counts), which has the same
  reference arm, the same depth and no compartment difference:
  did.csv      per sample, level, test sharing the null's reference arm (e.g. cytoB|nucA): ratio_test / ratio_null;
               and per pair (a, b): excess = [rate(contact) - rate(absent)]_test - [same]_null, with the cells n.
Usage: python bilous_compare.py --labels ~/claimbreak_data/Bilous_labels --out ~/claimbreak_data/Bilous_labels/compare_v1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--levels", default="L2m,L3m")
ap.add_argument("--comparisons", default="nucB|nucA,cytoB|nucA,wholeB|nucA,whole|nucleus,thinned|nucleus,cyto|nucleus,whole|thinned")
ap.add_argument("--null", default="nucB|nucA", help="null comparison; tests sharing its reference arm are read against it")
ap.add_argument("--nbr-arm", default="nucleus")
ap.add_argument("--k-contact", type=int, default=6)
ap.add_argument("--k-far", type=int, default=20)
ap.add_argument("--radius", type=float, default=30.0)
ap.add_argument("--radius-far", type=float, default=60.0)
ap.add_argument("--min-margin", type=float, default=0.0)
ap.add_argument("--common-cells", default="", help="comma list of arms: use only cells labelled in all of them (same cells in every comparison)")
ap.add_argument("--samples", default="", help="comma-separated substrings; default all")
a = ap.parse_args()

src = Path(a.labels).expanduser()
out = Path(a.out).expanduser()
if out.exists() and any(out.iterdir()):
    raise SystemExit(f"{out} exists and is not empty; choose a new --out")
out.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob("*.labels.csv.gz"))
if a.samples:
    keys = a.samples.split(",")
    files = [f for f in files if any(k in f.name for k in keys)]
levels = a.levels.split(",")
comps = [tuple(c.split("|")) for c in a.comparisons.split(",")]
pair_rows, summ_rows = [], []

for f in files:
    D = pd.read_csv(f, keep_default_na=False, low_memory=False)
    sample = D["sample"].iloc[0]
    xy = D[["x", "y"]].to_numpy(float)
    tree = cKDTree(xy)
    dist, idx = tree.query(xy, k=a.k_far + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]
    contact = (np.arange(a.k_far)[None, :] < a.k_contact) & (dist <= a.radius)
    far = (np.arange(a.k_far)[None, :] >= a.k_contact) & (dist <= a.radius_far)
    print(f"{sample}: {len(D):,} cells; median contact neighbours {np.median(contact.sum(1)):.0f}", flush=True)
    for lv in levels:
        nb = D[f"{lv}_{a.nbr_arm}"].to_numpy()
        types = np.array(sorted(set(nb) - {""}))
        tix = {t: i for i, t in enumerate(types)}
        nbi = np.array([tix.get(t, -1) for t in nb])
        K = len(types)
        # presence matrices: cells x types
        nbr_t = nbi[idx]
        C = np.zeros((len(D), K), bool)
        F = np.zeros((len(D), K), bool)
        for j in range(a.k_far):
            ok = (nbr_t[:, j] >= 0)
            rows = np.where(ok & contact[:, j])[0]
            C[rows, nbr_t[rows, j]] = True
            rows = np.where(ok & far[:, j])[0]
            F[rows, nbr_t[rows, j]] = True
        Fo = F & ~C
        for T, Rr in comps:
            if f"{lv}_{T}" not in D or f"{lv}_{Rr}" not in D:
                continue
            t, r = D[f"{lv}_{T}"].to_numpy(), D[f"{lv}_{Rr}"].to_numpy()
            mg = pd.to_numeric(D[f"{lv}_{Rr}_margin"], errors="coerce").to_numpy()
            el = (t != "") & (r != "") & (np.nan_to_num(mg, nan=-1) >= a.min_margin) & np.isin(r, types)
            for cx in [x for x in a.common_cells.split(",") if x]:
                el &= D[f"{lv}_{cx}"].to_numpy() != ""
            ri = np.array([tix.get(x, -1) for x in r])
            ti = np.array([tix.get(x, -1) for x in t])
            chg = el & (ti != ri) & (ti >= 0)
            # per pair and stratum
            for ai in range(K):
                ca = el & (ri == ai)
                na = int(ca.sum())
                if na == 0:
                    continue
                for bi in range(K):
                    if bi == ai:
                        continue
                    hit = ca & (ti == bi)
                    for sname, S in (("contact", C[:, bi]), ("far_only", Fo[:, bi]), ("absent", ~C[:, bi] & ~F[:, bi])):
                        m = ca & S
                        n = int(m.sum())
                        if n:
                            pair_rows.append({"sample": sample, "level": lv, "comparison": f"{T}|{Rr}",
                                              "a": types[ai], "b": types[bi], "stratum": sname, "n": n,
                                              "changes": int((hit & S).sum()), "rate": (hit & S).sum() / n})
            # summary: observed vs expected share of changes landing on a contact neighbor's type
            pC = np.zeros((K, K))
            pF = np.zeros((K, K))
            for ai in range(K):
                ca = el & (ri == ai)
                if ca.any():
                    pC[ai] = C[ca].mean(0)
                    pF[ai] = Fo[ca].mean(0)
            cr, ct = ri[chg], ti[chg]
            obsC = C[np.where(chg)[0], ct].mean() if chg.any() else np.nan
            expC = pC[cr, ct].mean() if chg.any() else np.nan
            obsF = Fo[np.where(chg)[0], ct].mean() if chg.any() else np.nan
            expF = pF[cr, ct].mean() if chg.any() else np.nan
            summ_rows.append({"sample": sample, "level": lv, "comparison": f"{T}|{Rr}", "cells": int(el.sum()),
                              "change_rate": chg.sum() / max(el.sum(), 1), "changed": int(chg.sum()),
                              "obs_contact": obsC, "exp_contact": expC, "ratio_contact": obsC / expC if expC else np.nan,
                              "obs_far_only": obsF, "exp_far_only": expF,
                              "ratio_far_only": obsF / expF if expF else np.nan})
            s = summ_rows[-1]
            print(f"  {lv} {T}|{Rr}: cells {s['cells']:,}, changed {s['change_rate']:.3f}; new type a contact "
                  f"neighbour's: {s['obs_contact']:.3f} vs {s['exp_contact']:.3f} expected (x{s['ratio_contact']:.2f}); "
                  f"far only x{s['ratio_far_only']:.2f}", flush=True)

P = pd.DataFrame(pair_rows)
P.to_csv(out / "pairs.csv", index=False)
S0 = pd.DataFrame(summ_rows)
did_rows = []
if len(S0) and a.null in set(S0.comparison):
    ref_arm = a.null.split("|")[1]
    for (smp, lv), g in S0.groupby(["sample", "level"]):
        nl = g[g.comparison == a.null]
        if nl.empty:
            continue
        nl = nl.iloc[0]
        for _, t in g[(g.comparison != a.null) & g.comparison.str.endswith("|" + ref_arm)].iterrows():
            did_rows.append({"sample": smp, "level": lv, "test": t.comparison, "null": a.null,
                             "change_test": t.change_rate, "change_null": nl.change_rate,
                             "ratio_test": t.ratio_contact, "ratio_null": nl.ratio_contact,
                             "ratio_of_ratios": t.ratio_contact / nl.ratio_contact})
    D2 = pd.DataFrame(did_rows)
    D2.to_csv(out / "did.csv", index=False)
    if len(P):
        W = P.pivot_table(index=["sample", "level", "comparison", "a", "b"], columns="stratum",
                          values=["rate", "n"], aggfunc="first")
        W.columns = [f"{x}_{y}" for x, y in W.columns]
        W = W.reset_index()
        W["contrast"] = W.get("rate_contact") - W.get("rate_absent")
        nul = W[W.comparison == a.null][["sample", "level", "a", "b", "contrast"]].rename(columns={"contrast": "c_null"})
        tst = W[(W.comparison != a.null) & W.comparison.str.endswith("|" + ref_arm)]
        PP = tst.merge(nul, on=["sample", "level", "a", "b"], how="inner")
        PP["excess"] = PP.contrast - PP.c_null
        PP.to_csv(out / "did_pairs.csv", index=False)
    if len(D2):
        print("ratio_test / ratio_null (median over sections):")
        print(D2.groupby(["level", "test"])[["change_test", "change_null", "ratio_test", "ratio_null",
                                             "ratio_of_ratios"]].median().round(3).to_string())
S = pd.DataFrame(summ_rows)
S.to_csv(out / "summary.csv", index=False)
print(f"wrote {out}/pairs.csv, summary.csv", flush=True)
if len(S):
    print(S.groupby(["level", "comparison"])[["change_rate", "ratio_contact", "ratio_far_only"]].median()
          .round(3).to_string())
