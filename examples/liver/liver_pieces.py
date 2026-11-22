#!/usr/bin/env python
"""liver_pieces.py -- tissue pieces of the MASLD CosMx slides (Boesch et al. 2026, GSE312698): quality per piece and the
layout of the fields of view, to decide which pieces enter the compartment test and to map pieces to biopsies
. Reads only; writes into --out.

  qc      per piece (one compartment h5ad each): cells, FOVs, median counts whole and nucleus, nuclear share, median
          stain means (DAPI, PanCK, CD68, CD45, Membrane), extent in mm; `usable` = nuclear share >= --min-share and
          median nuclear count >= --min-nuc. Writes pieces_qc.csv.
  layout  per slide (metadata_file): pieces of the FOV centers at several link distances (x median FOV spacing), with
          FOVs and cells per piece; fov_layout.csv (FOV center in mm, piece at each distance, cells, median DAPI and
          nuclear share where a piece h5ad exists) and fov_layout.png (one panel per slide, FOVs colored by the piece
          at --fov-gap, numbered as the converter numbers them).

  filter  copies of the usable pieces (pieces_qc.csv) without the FOVs whose nuclear share is below
          --min-share (a FOV where the nucleus calls failed), into --out-h5ad; the originals are not changed.
          Writes filter_fovs.csv (FOVs kept and dropped per piece).

  python liver_pieces.py qc --h5ad '~/claimbreak_data/data/liver_mash/h5ad/*.h5ad' --out ~/claimbreak_results/liver_pieces
  python liver_pieces.py filter --h5ad ~/claimbreak_data/data/liver_mash/h5ad \
      --qc ~/claimbreak_results/liver_pieces/pieces_qc.csv --out-h5ad ~/claimbreak_data/data/liver_mash/h5ad_qc
  python liver_pieces.py layout --csv ~/claimbreak_data/data/liver_mash/csv --out ~/claimbreak_results/liver_pieces
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

STAINS = ["Mean.DAPI", "Mean.PanCK", "Mean.CD68", "Mean.CD45", "Mean.Membrane"]


def piece_labels(xy, gap):
    """Connected components of FOV centers linked when closer than gap x the median nearest spacing, numbered by the
    center's x then y (as compartment_to_h5ad.pieces)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if len(xy) < 2:
        return np.zeros(len(xy), int), float("nan")
    d, _ = cKDTree(xy).query(xy, k=2)
    step = float(np.median(d[:, 1]))
    pairs = cKDTree(xy).query_pairs(gap * step, output_type="ndarray")
    G = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    _, lab = connected_components(G, directed=False)
    cen = pd.DataFrame(xy, columns=["x", "y"]).groupby(lab).mean().sort_values(["x", "y"])
    order = {old: new for new, old in enumerate(cen.index)}
    return np.array([order[v] for v in lab]), step


def cmd_qc(a):
    import anndata as ad
    rows = []
    for f in sorted(glob.glob(os.path.expanduser(a.h5ad))):
        A = ad.read_h5ad(f, backed="r")
        o = A.obs
        name = os.path.basename(f)[:-5]
        r = dict(piece=name, slide=re.sub(r"_p\d+$", "", name), cells=len(o), fovs=o["fov"].nunique(),
                 median_whole=float(np.median(o.n_whole)), median_nucleus=float(np.median(o.n_nucleus)),
                 nuclear_share=float(o.n_nucleus.sum() / max(o.n_whole.sum(), 1)),
                 share_zero_nucleus=float((o.n_nucleus == 0).mean()),
                 x_mm=f"{o.x.min() / 1000:.1f}-{o.x.max() / 1000:.1f}", y_mm=f"{o.y.min() / 1000:.1f}-{o.y.max() / 1000:.1f}")
        for s in STAINS:
            if s in o:
                r[s.replace("Mean.", "median_")] = float(np.median(o[s]))
        rows.append(r)
        A.file.close()
    if not rows:
        raise SystemExit(f"no h5ad files match {a.h5ad}")
    T = pd.DataFrame(rows)
    T["usable"] = (T.nuclear_share >= a.min_share) & (T.median_nucleus >= a.min_nuc)
    os.makedirs(a.out, exist_ok=True)
    T.to_csv(os.path.join(a.out, "pieces_qc.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(T.round(3).to_string(index=False))
    u = T[T.usable]
    print(f"\nusable: {len(u)} of {len(T)} pieces, {u.cells.sum():,} of {T.cells.sum():,} cells "
          f"(nuclear share >= {a.min_share}, median nuclear count >= {a.min_nuc})")


def cmd_layout(a):
    import anndata as ad
    os.makedirs(a.out, exist_ok=True)
    gaps = [float(g) for g in a.gaps.split(",")]
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.csv), "*_metadata_file.csv*")))
    if not files:
        raise SystemExit(f"no *_metadata_file.csv(.gz) in {a.csv}")
    allF = []
    for f in files:
        slide = re.sub(r"_metadata_file\.csv(\.gz)?$", "", os.path.basename(f))
        cols = ["fov", "CenterX_global_px", "CenterY_global_px", "Mean.DAPI", "Area", "Area.um2"]
        m = pd.read_csv(f, usecols=lambda c: c in cols)
        px = float(np.sqrt(np.median(m["Area.um2"] / m["Area"]))) if {"Area", "Area.um2"} <= set(m) else a.px_um
        F = m.groupby("fov").agg(x=("CenterX_global_px", "median"), y=("CenterY_global_px", "median"),
                                 cells=("fov", "size"))
        if "Mean.DAPI" in m:
            F["median_DAPI"] = m.groupby("fov")["Mean.DAPI"].median()
        F[["x", "y"]] = F[["x", "y"]] * px / 1000                       # mm
        xy = F[["x", "y"]].to_numpy(float)
        print(f"== {slide}: {len(F)} FOVs, {int(F.cells.sum()):,} cells")
        for g in gaps:
            lab, step = piece_labels(xy, g)
            F[f"piece_gap{g:g}"] = lab
            P = F.groupby(lab).agg(fovs=("cells", "size"), cells=("cells", "sum"))
            desc = ", ".join(f"p{k}:{r.fovs}/{r.cells:,}" for k, r in P.iterrows())
            print(f"  gap {g:g} (link < {g * step:.2f} mm): {len(P)} pieces (FOVs/cells) {desc}")
        # nuclear share per FOV from the piece h5ad files, where they exist
        if a.h5ad:
            share = {}
            for h in sorted(glob.glob(os.path.join(os.path.expanduser(a.h5ad), f"{slide}_p*.h5ad"))):
                A = ad.read_h5ad(h, backed="r")
                o = A.obs.groupby("fov")[["n_nucleus", "n_whole"]].sum()
                share.update((o.n_nucleus / o.n_whole.clip(lower=1)).to_dict())
                A.file.close()
            F["nuclear_share"] = F.index.map(lambda v: share.get(v, np.nan)).astype(float)
        F.insert(0, "slide", slide)
        allF.append(F.reset_index())
    L = pd.concat(allF, ignore_index=True)
    L.to_csv(os.path.join(a.out, "fov_layout.csv"), index=False)
    print(f"wrote {os.path.join(a.out, 'fov_layout.csv')}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    key = f"piece_gap{a.fov_gap:g}"
    if key not in L:
        L[key] = np.concatenate([piece_labels(g[["x", "y"]].to_numpy(float), a.fov_gap)[0]
                                 for _, g in L.groupby("slide", sort=False)])
    sl = list(dict.fromkeys(L.slide))
    fig, axs = plt.subplots(1, len(sl), figsize=(5 * len(sl), 5), squeeze=False)
    for ax, s in zip(axs[0], sl):
        g = L[L.slide == s]
        cmap = plt.get_cmap("tab20")
        ax.scatter(g.x, g.y, c=[cmap(int(k) % 20) for k in g[key]], s=30, marker="s", edgecolors="none")
        for k, gg in g.groupby(key):
            ax.text(gg.x.mean(), gg.y.mean(), f"p{k}", ha="center", va="center", fontsize=9, weight="bold")
        if "nuclear_share" in g:
            lo = g[g.nuclear_share < a.min_share]
            ax.scatter(lo.x, lo.y, s=30, marker="x", c="k", linewidths=0.6, label=f"nuclear share < {a.min_share}")
            if len(lo):
                ax.legend(loc="lower right", fontsize=7)
        ax.set_title(f"{s} (pieces at gap {a.fov_gap:g})", fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fov_layout.png"), dpi=130)
    print(f"wrote {os.path.join(a.out, 'fov_layout.png')}")


def cmd_filter(a):
    import anndata as ad
    Q = pd.read_csv(a.qc)
    os.makedirs(a.out_h5ad, exist_ok=True)
    rows = []
    for p in Q[Q.usable].piece:
        src, dst = os.path.join(a.h5ad, f"{p}.h5ad"), os.path.join(a.out_h5ad, f"{p}.h5ad")
        if os.path.exists(dst):
            print(f"exists, skipped: {dst}")
            continue
        A = ad.read_h5ad(src)
        f = A.obs.groupby("fov", observed=True)[["n_nucleus", "n_whole"]].sum()
        sh = f.n_nucleus / f.n_whole.clip(lower=1)
        bad = set(sh.index[sh < a.min_share])
        keep = ~A.obs["fov"].isin(bad).to_numpy()
        for v, x in sh.items():
            rows.append(dict(piece=p, fov=v, nuclear_share=float(x), cells=int((A.obs["fov"] == v).sum()), kept=v not in bad))
        B = A[keep].copy()
        B.uns["fov_filter"] = f"FOVs with nuclear share < {a.min_share} dropped: {sorted(map(int, bad))}"
        B.write_h5ad(dst, compression="gzip")
        print(f"{p}: {len(sh) - len(bad)}/{len(sh)} FOVs, {B.n_obs:,}/{A.n_obs:,} cells kept -> {dst}", flush=True)
    if rows:
        F = pd.DataFrame(rows)
        F.to_csv(os.path.join(a.out_h5ad, "filter_fovs.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qc")
    q.add_argument("--h5ad", required=True, help="glob of piece h5ad files")
    q.add_argument("--out", required=True)
    q.add_argument("--min-share", type=float, default=0.25)
    q.add_argument("--min-nuc", type=float, default=10)
    q.set_defaults(fn=cmd_qc)
    f = sub.add_parser("filter")
    f.add_argument("--h5ad", required=True, help="folder with the piece h5ad files")
    f.add_argument("--qc", required=True, help="pieces_qc.csv from qc")
    f.add_argument("--out-h5ad", required=True)
    f.add_argument("--min-share", type=float, default=0.25)
    f.set_defaults(fn=cmd_filter)
    l = sub.add_parser("layout")
    l.add_argument("--csv", required=True, help="folder with <slide>_metadata_file.csv(.gz)")
    l.add_argument("--h5ad", default="", help="folder with the piece h5ad files (adds nuclear share per FOV)")
    l.add_argument("--out", required=True)
    l.add_argument("--gaps", default="1.1,1.3,1.6,2.5")
    l.add_argument("--fov-gap", type=float, default=1.6)
    l.add_argument("--min-share", type=float, default=0.25)
    l.add_argument("--px-um", type=float, default=0.12028)
    l.set_defaults(fn=cmd_layout)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
