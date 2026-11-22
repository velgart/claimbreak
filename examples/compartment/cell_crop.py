#!/usr/bin/env python
"""cell_crop.py -- one real cell for Fig. 1a of the paper: a T cell whose nucleus is labeled T/NK
but whose whole segmented cell is labeled tumor, next to a tumor cell, with its nucleus and mask outlines and the
transcripts of T-cell and tumor markers.

Candidates are cells whose reference label (default the nucleus) is the immune type and whose test label (default the
whole cell) is the partner type, with a partner cell among their six nearest neighbors within 30 um. Each is scored
from the marker transcripts it holds: T-cell markers over its nucleus and partner markers outside it; the crop shows
the best one (or --cell). Only the marker genes are read from the transcripts file for the scoring, and only the
crop window for the picture, so the run takes a minute or two on a 5K section.

  python cell_crop.py --raw ~/claimbreak_data/Bilous/raw --sample <sample> \
      --labels ~/claimbreak_results/bilous_v4/<sample>.labels.csv.gz --out ~/claimbreak_results/fig1a_crop [--cell ID] [--half 15]

--raw holds the section's Xenium files, either GEO-style (<sample>_transcripts.parquet, <sample>_cell_boundaries.parquet,
<sample>_nucleus_boundaries.parquet, optionally .gz) or as a folder <raw>/<sample>/ with the instrument's names.
Writes OUT/candidates.csv (the ranked candidates; rerun with --cell to choose another), crop_transcripts.csv,
crop_polygons.csv, crop_cells.csv, crop_counts.csv, meta.json and crop.pdf/png (a preview; the paper's panel is drawn
by make_figures from the csv files). Points are rasterised and outlines kept as vectors, so the PDF stays small
(tens of kB). Refuses to overwrite OUT.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

T_GENES = ("CD3E,CD3D,CD3G,CD2,CD247,CD8A,CD8B,TRAC,TRBC1,TRBC2,CD7,LCK,ZAP70,ITK,CD6,CD5,CD27,IL7R,CXCR6,GZMA,GZMK,"
           "GZMB,GZMH,NKG7,PRF1,KLRD1,KLRB1,CD96,CD247,SH2D1A,THEMIS,TIGIT,PDCD1")
P_GENES = ("EPCAM,KRT7,KRT8,KRT18,KRT19,KRT5,KRT17,KRT6A,CEACAM5,CEACAM6,MUC1,MUC5B,NAPSA,SFTPB,SFTPC,SFTPA1,SFTA2,"
           "CLDN4,ELF3,CDH1,TACSTD2,AGR2,LCN2,MKI67,TOP2A")
COLS = ["cell_id", "feature_name", "overlaps_nucleus", "x_location", "y_location", "qv"]


def find(raw: Path, sample: str, stem: str):
    for p in (raw / f"{sample}_{stem}", raw / f"{sample}_{stem}.gz", raw / sample / stem, raw / sample / f"{stem}.gz",
              raw / sample / "outs" / stem, raw / sample / "outs" / f"{stem}.gz"):
        if p.exists():
            return p
    hits = sorted(raw.glob(f"{sample}*{stem}*"))
    return hits[0] if len(hits) == 1 else None


def read_parquet(path: Path, columns=None, filters=None):
    """Read with pyarrow filters where possible (plain parquet); a .gz file is read whole and filtered here."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            D = pd.read_parquet(io.BytesIO(fh.read()), columns=columns)
        for c, op, v in filters or []:
            if op == "in":
                D = D[D[c].isin(list(v))]
            elif op == ">=":
                D = D[D[c] >= v]
            elif op == "<=":
                D = D[D[c] <= v]
        return D
    return pd.read_parquet(path, columns=columns, filters=filters or None)


def as_str(s: pd.Series) -> pd.Series:
    if len(s) and isinstance(s.iloc[0], (bytes, bytearray)):
        return s.map(lambda b: b.decode())
    return s.astype(str)


def pick_columns(L: pd.DataFrame, ref: str, test: str):
    for r, t in ((ref, test), (ref.replace("_nucleus", "_nucB"), test.replace("_whole", "_wholeB"))):
        if r in L and t in L:
            return r, t
    raise KeyError(f"labels need {ref} and {test} (or the nucB/wholeB arms); have {list(L.columns)[:20]}")


def candidates(L, ref, test, immune, partner, k=6, r=30.0):
    from claimbreak.contacts import knn_edges
    E = knn_edges(L[["x", "y"]].to_numpy(float), k, r)
    rl = L[ref].astype(str).to_numpy()
    nb_partner = np.zeros(len(L), bool)
    for i, j in ((E[:, 0], E[:, 1]), (E[:, 1], E[:, 0])):
        nb_partner[i[rl[j] == partner]] = True
    c = (rl == immune) & (L[test].astype(str).to_numpy() == partner) & nb_partner
    return L.loc[c, "cell_id"].astype(str).tolist()


def score(Tx: pd.DataFrame, cells, tg, pg):
    q = Tx[Tx.cell_id.isin(set(cells))]
    q = q.assign(grp=np.where(q.feature_name.isin(tg), "T", np.where(q.feature_name.isin(pg), "P", "")),
                 nuc=q.overlaps_nucleus.astype(int) == 1)
    S = pd.DataFrame({"cell_id": list(cells)}).set_index("cell_id")
    for g in ("T", "P"):
        for n, lab in ((True, "nuc"), (False, "out")):
            S[f"{g}_{lab}"] = q[(q.grp == g) & (q.nuc == n)].groupby("cell_id").size()
    S = S.fillna(0).astype(int)
    S["score"] = np.minimum(S.T_nuc, S.P_out) + 0.1 * (S.T_nuc + S.P_out) - 0.5 * S.P_nuc - 0.5 * S.T_out
    return S.sort_values("score", ascending=False)


def draw_crop(ax, out: Path, half: float, scalebar: float = 5.0):
    """The picture: other cells' outlines gray, the target's mask black and its nucleus dashed, the partner's mask
    vermilion; transcripts of T-cell markers blue and of tumor markers vermilion (filled over a nucleus)."""
    P = pd.read_csv(out / "crop_polygons.csv")
    X = pd.read_csv(out / "crop_transcripts.csv")
    C = pd.read_csv(out / "crop_cells.csv").set_index("cell_id")
    col = {"target": "black", "partner": "#D55E00", "other": "#B0B0B0"}
    for (cid, kind), g in P.groupby(["cell_id", "kind"], sort=False):
        role = C.role.get(cid, "other")
        ax.fill(g.x, g.y, fc="none", ec=col[role], lw=1.1 if role == "target" else 0.6,
                ls=(0, (2, 1)) if kind == "nucleus" else "-", zorder=3 if role != "other" else 2)
    for grp, c in (("T", "#0072B2"), ("P", "#D55E00")):
        q = X[X.group == grp]
        ax.scatter(q.x, q.y, s=3.5, c=np.where(q.in_nucleus, c, "white"), edgecolors=c, linewidths=0.4,
                   rasterized=True, zorder=4)
    q = X[X.group == "other"]
    ax.scatter(q.x, q.y, s=0.6, c="#CFCFCF", linewidths=0, rasterized=True, zorder=1)
    ax.plot([half - 1 - scalebar, half - 1], [-half + 1.2] * 2, color="black", lw=1.2, solid_capstyle="butt")
    ax.text(half - 1 - scalebar / 2, -half + 1.7, f"{scalebar:g} µm", ha="center", va="bottom", fontsize=5.5)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True); ap.add_argument("--sample", required=True)
    ap.add_argument("--labels", required=True, help="the section's labels.csv.gz (cell_id, x, y, label columns)")
    ap.add_argument("--ref", default="L2m_nucleus"); ap.add_argument("--test", default="L2m_whole")
    ap.add_argument("--immune", default="T_NK"); ap.add_argument("--partner", default="Tumour")
    ap.add_argument("--t-genes", default=T_GENES); ap.add_argument("--p-genes", default=P_GENES)
    ap.add_argument("--cell", default="", help="cell_id to show (default: the best-scoring candidate)")
    ap.add_argument("--half", type=float, default=15.0, help="half-width of the crop (um)")
    ap.add_argument("--min-qv", type=float, default=20.0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, raw = Path(a.out).expanduser(), Path(a.raw).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    ftx, fcb, fnb = (find(raw, a.sample, s) for s in ("transcripts.parquet", "cell_boundaries.parquet",
                                                      "nucleus_boundaries.parquet"))
    if ftx is None or fcb is None:
        raise SystemExit(f"no transcripts or cell boundaries for {a.sample} in {raw}")
    L = pd.read_csv(a.labels, low_memory=False, keep_default_na=False)
    if "cell_id" not in L:
        raise SystemExit("the labels file has no cell_id column")
    L["cell_id"] = L.cell_id.astype(str)
    ref, test = pick_columns(L, a.ref, a.test)
    tg = [g for g in a.t_genes.split(",") if g]; pg = [g for g in a.p_genes.split(",") if g]
    cand = candidates(L, ref, test, a.immune, a.partner)
    print(f"{a.sample}: {len(L):,} cells; {len(cand):,} with {ref} = {a.immune}, {test} = {a.partner} and a "
          f"{a.partner} neighbour")
    Tx = read_parquet(ftx, COLS, [("feature_name", "in", tg + pg)])
    Tx["cell_id"] = as_str(Tx.cell_id); Tx["feature_name"] = as_str(Tx.feature_name)
    Tx = Tx[Tx.qv >= a.min_qv]
    have = sorted(set(Tx.feature_name))
    print(f"marker genes in the panel: T {[g for g in tg if g in have]}; {a.partner} {[g for g in pg if g in have]}")
    S = score(Tx, cand, set(tg), set(pg))
    out.mkdir(parents=True)
    S.head(a.top).to_csv(out / "candidates.csv")
    print(S.head(10).to_string())
    cell = a.cell or (S.index[0] if len(S) else "")
    if not cell:
        raise SystemExit("no candidate cell; try --ref/--test or another section")
    c = L[L.cell_id == cell]
    if not len(c):
        raise SystemExit(f"cell {cell} not in the labels")
    cx, cy, h = float(c.x.iloc[0]), float(c.y.iloc[0]), a.half
    win = [("x_location", ">=", cx - h), ("x_location", "<=", cx + h), ("y_location", ">=", cy - h),
           ("y_location", "<=", cy + h)]
    X = read_parquet(ftx, COLS, win)
    X["cell_id"] = as_str(X.cell_id); X["feature_name"] = as_str(X.feature_name)
    X = X[X.qv >= a.min_qv]
    X = pd.DataFrame({"x": X.x_location - cx, "y": X.y_location - cy, "gene": X.feature_name,
                      "group": np.where(X.feature_name.isin(tg), "T", np.where(X.feature_name.isin(pg), "P", "other")),
                      "in_nucleus": X.overlaps_nucleus.astype(int) == 1, "cell_id": X.cell_id,
                      "in_target": X.cell_id == cell})
    near = L[(L.x - cx).abs().le(h + 25) & (L.y - cy).abs().le(h + 25)].copy()
    pick = S.loc[cell] if cell in S.index else None
    # the partner shown: the touching partner-labeled neighbor holding most of the target's outside markers' type
    nb = near[(near[ref].astype(str) == a.partner) & (near.cell_id != cell)]
    d = np.hypot(nb.x - cx, nb.y - cy)
    partner_id = nb.cell_id.iloc[int(np.argmin(d))] if len(nb) else ""
    near["role"] = np.where(near.cell_id == cell, "target", np.where(near.cell_id == partner_id, "partner", "other"))
    ids = set(near.cell_id)
    polys = []
    for f, kind in ((fcb, "cell"), (fnb, "nucleus")):
        if f is None:
            continue
        B = read_parquet(f, ["cell_id", "vertex_x", "vertex_y"])
        B["cell_id"] = as_str(B.cell_id)
        B = B[B.cell_id.isin(ids)]
        polys.append(pd.DataFrame({"cell_id": B.cell_id, "kind": kind, "x": B.vertex_x - cx, "y": B.vertex_y - cy}))
    P = pd.concat(polys) if polys else pd.DataFrame(columns=["cell_id", "kind", "x", "y"])
    X.to_csv(out / "crop_transcripts.csv", index=False)
    P.to_csv(out / "crop_polygons.csv", index=False)
    near[["cell_id", "x", "y", ref, test, "role"]].rename(columns={ref: "ref_label", test: "test_label"}).to_csv(
        out / "crop_cells.csv", index=False)
    t = X[X.in_target]
    K = t.groupby(["in_nucleus", "group"]).size().unstack(fill_value=0)
    K.index = ["outside the nucleus" if not i else "nucleus" for i in K.index]
    K.to_csv(out / "crop_counts.csv")
    meta = dict(sample=a.sample, cell=cell, partner=partner_id, centre=[cx, cy], half=h, ref=ref, test=test,
                ref_label=str(c[ref].iloc[0]), test_label=str(c[test].iloc[0]), immune=a.immune, partner_type=a.partner,
                t_genes=[g for g in tg if g in have], p_genes=[g for g in pg if g in have], min_qv=a.min_qv,
                score=None if pick is None else {k: float(v) for k, v in pick.items()},
                transcripts=str(ftx), cell_boundaries=str(fcb), nucleus_boundaries=str(fnb) if fnb else None)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"cell {cell} ({ref} {meta['ref_label']}, {test} {meta['test_label']}); partner {partner_id}; "
          f"{len(X):,} transcripts in the window\n{K.to_string()}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    draw_crop(ax, out, h)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"crop.{ext}", bbox_inches="tight", dpi=600)
    print(f"wrote {out} (crop.pdf {(out / 'crop.pdf').stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
