#!/usr/bin/env python
"""rpgn_glom.py --: a published spatial claim tested with nuclear labels.

Sultana et al., Nat Immunol 2025 (GSE294965; Xenium 480 genes, Baysor cells) report that adaptive immune cells are
nearly absent inside glomeruli of patients with rapidly progressive glomerulonephritis while T cells rise in the
100-um ring around them. With their own labels (processed_data.h5ad, celltype_l1) the median T/NK share per biopsy is
0.6-0.8% inside glomeruli in ANCA and SLE against 1.2% in controls, and 3-7% in the ring.

Readouts:
  direction   nuclear labels give a higher T/NK share inside glomeruli than whole-cell labels and their Baysor labels,
              and the difference is larger inside glomeruli than in the ring;
  flip        in most disease biopsies, nuclear T/NK inside glomeruli reaches at least half the ring share;
  no flip     nuclear T/NK inside glomeruli stays near the control level; the absence claim stands;
  between     nuclear labels raise the glomerular share but it stays far below the ring: direction holds, size not.

Inputs: claimbreak labels of the vendor (Xenium) cells, one <sample>.labels.csv.gz per slide (x, y in um; arms
L1_nucleus, L1_whole, L1_nucA, L1_nucB, L1_wholeB), and baysor_cells.parquet (written from the deposited h5ad: x, y,
Slide_ID, Biopsy_ID, Disease, celltype_l1, is_in_glom, is_in_periglom, glom_ID). Each vendor cell takes the region,
biopsy and disease of the nearest Baysor cell within --max-dist um (same micrometre frame; the median distance is
reported so a frame mismatch is visible). Units are biopsies.

  python rpgn_glom.py --labels W/labels --baysor D/derived/baysor_cells.parquet --slides RPGN_0011546=0011546,... \
      --out W/glom
writes by_biopsy.csv (biopsy x region x arm: cells, T/NK share), summary.csv (medians by disease and region),
ratio.csv (glomerulus over ring per biopsy and arm), match.csv (nearest-cell distances), meta.json.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

TNK_RX = r"(?i)(?:^|[^a-z])(?:t|nk|tnk|t_nk)(?:[^a-z]|$)"
BAYSOR_TNK = ["T", "NKC/T", "cycNKC/T"]
ARMS = ["nucleus", "whole", "nucA", "nucB", "wholeB"]


def regions(B):
    return np.where(B.is_in_glom.to_numpy(bool), "glom", np.where(B.is_in_periglom.to_numpy(bool), "periglom", "other"))


def assign(L, B, max_dist):
    """Region, biopsy and disease of the nearest Baysor cell within max_dist um."""
    tree = cKDTree(B[["x", "y"]].to_numpy(float))
    d, j = tree.query(L[["x", "y"]].to_numpy(float), k=1, distance_upper_bound=max_dist)
    ok = np.isfinite(d)
    jj = np.where(ok, j, 0)
    reg = regions(B)
    out = pd.DataFrame({"dist": d,
                        "region": np.where(ok, reg[jj], "unmatched"),
                        "biopsy": np.where(ok, B.Biopsy_ID.astype(str).to_numpy()[jj], ""),
                        "disease": np.where(ok, B.Disease.astype(str).to_numpy()[jj], ""),
                        "glom_ID": np.where(ok, B.glom_ID.astype(str).to_numpy()[jj], "")}, index=L.index)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--baysor", required=True)
    ap.add_argument("--slides", required=True, help="sample=Slide_ID pairs, comma-separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="L1")
    ap.add_argument("--max-dist", type=float, default=8.0)
    ap.add_argument("--min-cells", type=int, default=200, help="glomerular cells a biopsy needs to enter the medians")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; move it away")
    out.mkdir(parents=True, exist_ok=True)
    slides = dict(x.split("=", 1) for x in a.slides.split(",") if "=" in x)
    Ball = pd.read_parquet(Path(a.baysor).expanduser())
    Ball["Slide_ID"] = Ball.Slide_ID.astype(str)
    rows, match = [], []
    for sample, sid in slides.items():
        f = Path(a.labels).expanduser() / f"{sample}.labels.csv.gz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        L = pd.read_csv(f, keep_default_na=False, low_memory=False)
        B = Ball[Ball.Slide_ID == sid].reset_index(drop=True)
        if not len(B):
            raise SystemExit(f"no Baysor cells for slide {sid}")
        A = assign(L, B, a.max_dist)
        dd = A.dist[np.isfinite(A.dist)]
        match.append(dict(sample=sample, slide=sid, vendor_cells=len(L), baysor_cells=len(B),
                          matched=float(np.isfinite(A.dist).mean()), dist_median=float(dd.median()) if len(dd) else np.nan,
                          dist_p90=float(dd.quantile(0.9)) if len(dd) else np.nan))
        arms = [x for x in ARMS if f"{a.level}_{x}" in L]
        for arm in arms:
            lab = L[f"{a.level}_{arm}"].astype(str)
            tnk = lab.str.contains(TNK_RX, regex=True) & (lab != "")
            G = pd.DataFrame({"biopsy": A.biopsy, "disease": A.disease, "region": A.region, "tnk": tnk,
                              "lab": lab != ""})
            G = G[(G.region != "unmatched") & G.lab]
            for (b, d, r), g in G.groupby(["biopsy", "disease", "region"]):
                rows.append(dict(slide=sid, biopsy=b, disease=d, region=r, arm=arm, cells=len(g), tnk=float(g.tnk.mean())))
        rB = regions(B)
        Bt = pd.DataFrame({"biopsy": B.Biopsy_ID.astype(str), "disease": B.Disease.astype(str), "region": rB,
                           "tnk": B.celltype_l1.astype(str).isin(BAYSOR_TNK)})
        for (b, d, r), g in Bt.groupby(["biopsy", "disease", "region"]):
            rows.append(dict(slide=sid, biopsy=b, disease=d, region=r, arm="baysor_published", cells=len(g),
                             tnk=float(g.tnk.mean())))
    R = pd.DataFrame(rows)
    R.to_csv(out / "by_biopsy.csv", index=False)
    M = pd.DataFrame(match); M.to_csv(out / "match.csv", index=False)
    gl = R[(R.region == "glom") & (R.arm == "nucleus")]
    ok = set(gl[gl.cells >= a.min_cells].biopsy)
    Rk = R[R.biopsy.isin(ok)]
    S = Rk.groupby(["disease", "region", "arm"]).agg(biopsies=("biopsy", "nunique"), tnk_median=("tnk", "median"),
                                                     tnk_lo=("tnk", lambda s: s.quantile(0.25)),
                                                     tnk_hi=("tnk", lambda s: s.quantile(0.75))).reset_index()
    S.to_csv(out / "summary.csv", index=False)
    P = Rk.pivot_table(index=["biopsy", "disease", "arm"], columns="region", values="tnk").reset_index()
    if {"glom", "periglom"} <= set(P.columns):
        P["glom_over_ring"] = P["glom"] / P["periglom"].replace(0, np.nan)
    P.to_csv(out / "ratio.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(labels=str(a.labels), baysor=str(a.baysor), slides=slides,
                                                   level=a.level, max_dist=a.max_dist, min_cells=a.min_cells,
                                                   biopsies=sorted(ok)), indent=1))
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print("nearest Baysor cell:\n" + M.round(3).to_string(index=False))
        T = S.pivot_table(index=["disease", "region"], columns="arm", values="tnk_median").mul(100).round(2)
        print(f"\nT/NK share (%), median over biopsies with >= {a.min_cells} glomerular cells:\n" + T.to_string())
        if "glom_over_ring" in P:
            Q = P.groupby(["disease", "arm"]).glom_over_ring.median().unstack().round(2)
            print("\nglomerulus over ring, median per biopsy:\n" + Q.to_string())
            if "nucleus" in Q.columns:
                dis = P[(P.arm == "nucleus") & (P.disease != "Cntrl")]
                print(f"\ndisease biopsies with nuclear glomerulus/ring >= 0.5: {int((dis.glom_over_ring >= 0.5).sum())} "
                      f"of {int(dis.glom_over_ring.notna().sum())}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
