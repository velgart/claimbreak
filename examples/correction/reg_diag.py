#!/usr/bin/env python
"""reg_diag.py -- is an IHC -> Xenium nucleus registration right? Diagnostics that compute no IHC readout
(written after stage A, where R0's excess threshold 0.35 failed in dense lung tissue).

For each registration folder (ihc_truth.py register output) with its nuclei folder:
  1. matched share (mutual nearest neighbors) at radii 1, 2, 3, 4 um under the fitted transform, against the same
     under 40 random shifts (50-300 um, random directions): mean and maximum of the shifted shares, and a z-score;
  2. distribution of match distances (quantiles) for the transform and for shifted transforms;
  3. tiles of 500 um: matched share per tile and the median local offset of matches (non-rigid drift), so that a
     correct registration shows matches everywhere with small offsets, and a wrong one shows chance-level tiles;
  4. scale and rotation of the fitted affine against the image metadata scale.
Columns R0b_* (descriptive; ihc_wsl.sh uses R0b_match, the 2 um match test, to pick aligned
registrations): R0b valid if (i) z at 2 um >= 10 against the shifted transforms and the matched share at 2 um is at least twice
their mean; (ii) at least 80% of tiles with >= 200 IHC nuclei have a matched share above twice the shifted mean at
4 um; (iii) the fitted scale is within 1% of the metadata scale and the anisotropy below 1.02. A section with two
valid candidate components is ambiguous and not used.
Read only; writes OUT/<registration>.json, OUT/r0b.csv (one row per registration) and prints a table.

  python reg_diag.py --reg ~/claimbreak_work/ihc/reg_L3_c5,~/claimbreak_work/ihc/reg_L1_c1,~/claimbreak_work/ihc/reg_L1_c22 \
      --raw ~/claimbreak_data/Bilous/raw --out ~/claimbreak_work/ihc/reg_diag
"""
import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RADII = (1.0, 2.0, 3.0, 4.0)


def read_parquet_any(p):
    p = Path(p)
    if p.suffix == ".gz":
        with gzip.open(p, "rb") as fh:
            return pd.read_parquet(io.BytesIO(fh.read()))
    return pd.read_parquet(p)


def xenium_nuclei(raw, sample):
    for suf in ("", ".gz"):
        q = Path(raw).expanduser() / f"{sample}_nucleus_boundaries.parquet{suf}"
        if q.exists():
            B = read_parquet_any(q)
            return B.groupby(B["cell_id"].astype(str), sort=False)[["vertex_x", "vertex_y"]].mean().to_numpy(float)
    raise FileNotFoundError(f"{sample}_nucleus_boundaries.parquet(.gz) not in {raw}")


def mutual(P, Q, r):
    tp, tq = cKDTree(P), cKDTree(Q)
    dq, iq = tp.query(Q, distance_upper_bound=r)
    _, ip = tq.query(P, distance_upper_bound=r)
    ok = np.isfinite(dq)
    qi = np.flatnonzero(ok)
    pi = iq[ok]
    m = ip[pi] == qi
    return pi[m], qi[m], dq[ok][m]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reg", required=True, help="comma list of registration folders")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shifts", type=int, default=40)
    ap.add_argument("--tile-um", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for rdir in [Path(x).expanduser() for x in a.reg.split(",") if x]:
        reg = json.loads((rdir / "register.json").read_text())
        ndir = rdir.parent / f"nuc_c{reg['component']}"
        nmeta = json.loads((ndir / "nuclei.json").read_text())
        N = pd.read_csv(ndir / "nuclei.csv.gz", usecols=["x_px", "y_px"])
        M, t = np.array(reg["M"], float), np.array(reg["t"], float)
        Q1 = N[["x_px", "y_px"]].to_numpy(float) @ M.T + t
        P = xenium_nuclei(a.raw, reg["sample"])
        # restrict the Xenium nuclei to the IHC footprint (+ 20 um), so the shares are not diluted by tissue the
        # IHC crop does not cover; the same set is used for the shifted transforms
        lo, hi = Q1.min(0) - 20, Q1.max(0) + 20
        Pw = P[np.all((P >= lo) & (P <= hi), 1)]
        n_small = min(len(Pw), len(Q1))
        res = dict(registration=rdir.name, sample=reg["sample"], component=reg["component"], R0=reg.get("R0"),
                   n_xenium_in_footprint=int(len(Pw)), n_ihc=int(len(Q1)))
        # 1. matched share vs shifted
        ang = rng.uniform(0, 2 * np.pi, a.shifts)
        dist = rng.uniform(50, 300, a.shifts)
        shifts = np.c_[np.cos(ang) * dist, np.sin(ang) * dist]
        for r in RADII:
            real = len(mutual(Pw, Q1, r)[0]) / n_small
            sh = np.array([len(mutual(Pw, Q1 + s, r)[0]) / n_small for s in shifts])
            res[f"share_r{r:g}"] = real
            res[f"shift_mean_r{r:g}"] = float(sh.mean())
            res[f"shift_max_r{r:g}"] = float(sh.max())
            res[f"z_r{r:g}"] = float((real - sh.mean()) / max(sh.std(ddof=1), 1e-9))
            res[f"ratio_r{r:g}"] = float(real / max(sh.mean(), 1e-9))
        # 2. match distances
        pi, qi, d = mutual(Pw, Q1, 4.0)
        _, _, ds = mutual(Pw, Q1 + shifts[0], 4.0)
        res.update({f"dist_q{q}": float(np.quantile(d, q / 100)) if len(d) else np.nan for q in (25, 50, 75)})
        res.update({f"dist_shift_q{q}": float(np.quantile(ds, q / 100)) if len(ds) else np.nan for q in (25, 50, 75)})
        # 3. tiles: matched share per tile (among IHC nuclei of the tile), median offset of matches
        tile = np.floor((Q1 - lo) / a.tile_um).astype(int)
        key = tile[:, 0] * 100000 + tile[:, 1]
        matched = np.zeros(len(Q1), bool)
        matched[qi] = True
        off = np.full((len(Q1), 2), np.nan)
        off[qi] = Pw[pi] - Q1[qi]
        T = pd.DataFrame({"key": key, "m": matched, "dx": off[:, 0], "dy": off[:, 1]})
        g = T.groupby("key").agg(n=("m", "size"), share=("m", "mean"), dx=("dx", "median"), dy=("dy", "median"))
        g = g[g.n >= 200]
        chance = res["shift_mean_r4"]
        res.update(tiles=int(len(g)), tiles_above_2x_chance=int((g.share > 2 * chance).sum()),
                   tile_share_q10=float(g.share.quantile(0.1)) if len(g) else np.nan,
                   tile_share_median=float(g.share.median()) if len(g) else np.nan,
                   tile_offset_um_median=float(np.hypot(g.dx, g.dy).median()) if len(g) else np.nan,
                   tile_offset_um_q90=float(np.hypot(g.dx, g.dy).quantile(0.9)) if len(g) else np.nan)
        # 4. affine against the metadata scale
        U, S, Vt = np.linalg.svd(M)
        res.update(scale_um_per_px=float(np.sqrt(abs(np.linalg.det(M)))), scale_metadata=float(nmeta["um_per_px"]),
                   scale_ratio=float(np.sqrt(abs(np.linalg.det(M))) / nmeta["um_per_px"]),
                   anisotropy=float(S[0] / S[1]), rotation_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        c1 = res["z_r2"] >= 10 and res["ratio_r2"] >= 2
        c2 = res["tiles"] > 0 and res["tiles_above_2x_chance"] >= 0.8 * res["tiles"]
        c3 = abs(res["scale_ratio"] - 1) <= 0.01 and res["anisotropy"] < 1.02
        res.update(R0b_match=bool(c1), R0b_tiles=bool(c2), R0b_scale=bool(c3),
                   R0b="valid" if (c1 and c2 and c3) else "invalid")
        (out / f"{rdir.name}.json").write_text(json.dumps(res, indent=1))
        rows.append(res)
        print(f"{rdir.name}: done", flush=True)
    R = pd.DataFrame(rows)
    amb = R[R.R0b == "valid"].groupby("sample").size()
    R["R0b_used"] = [(r.R0b == "valid") and amb.get(r["sample"], 0) == 1 for _, r in R.iterrows()]
    R[["registration", "sample", "component", "R0", "R0b", "R0b_used", "R0b_match", "R0b_tiles", "R0b_scale"]].to_csv(
        out / "r0b.csv", index=False)
    D = R.set_index("registration").T
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(D.to_string())


if __name__ == "__main__":
    main()
