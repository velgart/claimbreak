#!/usr/bin/env python
"""baysor_tiles.py -- a compartment .h5ad from a Baysor segmentation of a Xenium section, run tile by tile
(a second transcript-based segmentation next to Proseg).

Baysor (Petukhov et al., Nat Biotechnol 2022) assigns transcripts to cells from the transcripts themselves, optionally
with the platform's segmentation as a prior. A whole Xenium section is too large for it in memory, so the section is
cut into square tiles with an overlap margin, Baysor runs on each tile, and the tiles are merged:

  prep   transcripts.parquet -> TILES/<tile>/transcripts.csv (qv >= --min-qv, no control codewords) with
         transcript_id, x_location, y_location, z_location, feature_name, overlaps_nucleus and the Xenium cell as an
         integer prior (xcell; 0 = not in a Xenium cell); TILES/tiles.csv gives each tile's core and padded bounds.
         Tiles holding fewer than --min-tile-transcripts are skipped.
  run    (shell, see baysor_batch.sh) baysor run ... TILES/<tile>/transcripts.csv :xcell -o TILES/<tile>/out/
  merge  every TILES/<tile>/out/segmentation.csv -> OUT/<sample>.h5ad:
         1. join Baysor's rows to the tile's transcripts: by transcript_id when Baysor kept the column, else by
            molecule_id (row number of the input), else by row order; checked gene by gene (stops if they differ);
         2. Baysor cell -> the Xenium cell whose nuclear transcripts it holds most of (a nucleus claimed by two Baysor
            cells goes to the one holding more of it); cells without nuclear transcripts are dropped (reported); a
            cell is kept only in the tile whose core holds its Xenium nucleus centroid, so cells in the overlap are
            not counted twice;
         3. X = transcripts Baysor assigns to the cell (not noise); layers['nucleus'] = those of them with
            overlaps_nucleus == 1 whose Xenium cell is the cell's own; layers['cyto'] = X - nucleus. A transcript in
            an overlap assigned to kept cells of two tiles is counted once (first tile; reported).
         obs as compartment_to_h5ad.py (x, y = Xenium centroid; cell ids = Xenium cell ids, so protein and IHC matches
         carry over) plus baysor_cell, tile and share_own_nucleus; uns['baysor'] has the mapping and join statistics.

  python baysor_tiles.py prep  --raw RAW --sample S --tiles W/baysor/S [--tile-um 1000 --margin-um 30]
  python baysor_tiles.py merge --raw RAW --sample S --tiles W/baysor/S --out W/baysor/h5ad
RAW holds <S>_transcripts.parquet and <S>_cells.parquet (as compartment_to_h5ad.py; links are fine). The raw folder
is only read; nothing is overwritten.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL = re.compile(r"^(NegControl|NegPrb|Negative|BLANK|Blank|Unassigned|Intergenic|DeprecatedCodeword|antisense|"
                     r"UnassignedCodeword|SystemControl|FalseCode|falsecode)", re.I)


def find_raw(raw, sample, stem):
    for suf in ("", ".gz"):
        q = Path(raw) / f"{sample}_{stem}{suf}"
        if q.exists():
            return q
    return None


def read_parquet(p, columns=None):
    import pyarrow.parquet as pq
    if p.suffix == ".gz":
        import gzip
        import io
        with gzip.open(p, "rb") as f:
            return pq.read_table(io.BytesIO(f.read()), columns=columns).to_pandas()
    return pq.read_table(p, columns=columns).to_pandas()


def transcripts(raw, sample, min_qv):
    f = find_raw(raw, sample, "transcripts.parquet")
    if f is None:
        sys.exit(f"no {sample}_transcripts.parquet in {raw}")
    import pyarrow.parquet as pq
    cols = set(pq.ParquetFile(f).schema_arrow.names) if f.suffix != ".gz" else None
    need = ["transcript_id", "cell_id", "feature_name", "qv", "overlaps_nucleus", "x_location", "y_location",
            "z_location"]
    if cols is not None:
        miss = [c for c in need if c not in cols and c != "z_location"]
        if miss:
            sys.exit(f"transcripts lack {miss}")
        need = [c for c in need if c in cols]
    T = read_parquet(f, need)
    if "z_location" not in T:
        T["z_location"] = 0.0
    T["feature_name"] = T["feature_name"].astype(str)
    T["cell_id"] = T["cell_id"].astype(str)
    keep = (T["qv"].to_numpy() >= min_qv) & ~T["feature_name"].str.match(CONTROL).to_numpy()
    return T[keep].reset_index(drop=True), f


def cmd_prep(a):
    tiles = Path(a.tiles).expanduser().resolve()
    raw = Path(a.raw).expanduser().resolve()
    if tiles == raw or raw in tiles.parents:
        sys.exit("--tiles must be outside the raw folder")
    if tiles.exists() and any(tiles.iterdir()):
        sys.exit(f"{tiles} exists and is not empty")
    T, f = transcripts(raw, a.sample, a.min_qv)
    C = read_parquet(find_raw(raw, a.sample, "cells.parquet"), ["cell_id"])
    ci = pd.Index(C["cell_id"].astype(str))
    xc = ci.get_indexer(T["cell_id"])
    T["xcell"] = np.where(xc >= 0, xc + 1, 0).astype(np.int64)            # integer prior; 0 = no Xenium cell
    x, y = T.x_location.to_numpy(), T.y_location.to_numpy()
    x0, y0 = np.floor(x.min()), np.floor(y.min())
    nx, ny = int(np.ceil((x.max() - x0 + 1e-9) / a.tile_um)), int(np.ceil((y.max() - y0 + 1e-9) / a.tile_um))
    tiles.mkdir(parents=True, exist_ok=True)
    rows = []
    cols = ["transcript_id", "x_location", "y_location", "z_location", "feature_name", "overlaps_nucleus", "xcell"]
    for i in range(nx):
        for j in range(ny):
            cx0, cy0 = x0 + i * a.tile_um, y0 + j * a.tile_um
            cx1, cy1 = cx0 + a.tile_um, cy0 + a.tile_um
            m = (x >= cx0 - a.margin_um) & (x < cx1 + a.margin_um) & (y >= cy0 - a.margin_um) & (y < cy1 + a.margin_um)
            name = f"t{i:02d}_{j:02d}"
            n = int(m.sum())
            core = int(((x >= cx0) & (x < cx1) & (y >= cy0) & (y < cy1)).sum())
            run = n >= a.min_tile_transcripts
            rows.append(dict(tile=name, core_x0=cx0, core_x1=cx1, core_y0=cy0, core_y1=cy1, margin_um=a.margin_um,
                             transcripts=n, transcripts_core=core, run=run))
            if run:
                d = tiles / name
                d.mkdir()
                T.loc[m, cols].to_csv(d / "transcripts.csv", index=False)
    P = pd.DataFrame(rows)
    P.to_csv(tiles / "tiles.csv", index=False)
    (tiles / "prep.json").write_text(json.dumps(dict(sample=a.sample, raw=str(raw), transcripts_file=f.name,
                                                     min_qv=a.min_qv, tile_um=a.tile_um, margin_um=a.margin_um,
                                                     transcripts=int(len(T)), tiles=int(P.run.sum())), indent=1))
    print(f"{a.sample}: {len(T):,} transcripts in {int(P.run.sum())} tiles of {a.tile_um:g} um (+{a.margin_um:g} um "
          f"margin); median {int(P[P.run].transcripts.median()):,} transcripts per tile; skipped "
          f"{int((~P.run).sum())} near-empty tiles")


def _cell_ids(s):
    """Baysor's cell column as strings; noise and empty -> ''."""
    v = s.astype(str).str.strip()
    return v.where(~v.isin(["", "0", "nan", "NaN", "None"]), "")


def cmd_merge(a):
    import anndata as ad
    tiles, raw, out = (Path(p).expanduser().resolve() for p in (a.tiles, a.raw, a.out))
    if out == raw or raw in out.parents:
        sys.exit("--out must be outside the raw folder")
    dst = out / f"{a.sample}.h5ad"
    if dst.exists():
        sys.exit(f"{dst} exists; not overwritten")
    P = pd.read_csv(tiles / "tiles.csv")
    P = P[P.run]
    C = read_parquet(find_raw(raw, a.sample, "cells.parquet"))
    C["cell_id"] = C["cell_id"].astype(str)
    ci = pd.Index(C["cell_id"])
    cxy = C[["x_centroid", "y_centroid"]].to_numpy(float)
    parts, stats_t = [], []
    for t in P.itertuples():
        seg = tiles / t.tile / "out" / "segmentation.csv"
        if not seg.exists():
            sys.exit(f"no Baysor output for tile {t.tile}: {seg}")
        Ti = pd.read_csv(tiles / t.tile / "transcripts.csv", dtype={"feature_name": str})
        S = pd.read_csv(seg, low_memory=False)
        if "transcript_id" in S.columns:
            ri = pd.Index(Ti.transcript_id.astype("int64")).get_indexer(S.transcript_id.astype("int64"))
            how = "transcript_id"
        elif "molecule_id" in S.columns:
            ri = S.molecule_id.astype("int64").to_numpy() - 1
            how = "molecule_id"
        elif len(S) == len(Ti):
            ri = np.arange(len(S))
            how = "row order"
        else:
            sys.exit(f"tile {t.tile}: no transcript_id or molecule_id in segmentation.csv and {len(S):,} rows against "
                     f"{len(Ti):,} input transcripts; check the columns of {seg} (header)")
        if (ri < 0).any() or (ri >= len(Ti)).any():
            sys.exit(f"tile {t.tile}: {int(((ri < 0) | (ri >= len(Ti))).sum())} Baysor rows not in the tile input")
        gcol = next((c for c in ("feature_name", "gene") if c in S.columns), None)
        if gcol is not None:
            same = float((S[gcol].astype(str).to_numpy() == Ti.feature_name.to_numpy()[ri]).mean())
            if same < 0.999:
                sys.exit(f"tile {t.tile}: genes differ between Baysor's output and the input at the joined rows "
                         f"({same:.4f} agree); join {how}")
        ccol = next((c for c in ("cell", "assignment") if c in S.columns), None)
        if ccol is None:
            sys.exit(f"tile {t.tile}: no cell column in {list(S.columns)}")
        cell = _cell_ids(S[ccol])
        if "is_noise" in S.columns:
            cell = cell.where(~S.is_noise.astype(str).str.lower().isin(["true", "1"]), "")
        D = Ti.iloc[ri].reset_index(drop=True)
        D["bcell"] = (t.tile + ":" + cell).where(cell != "", "")
        D["tile"] = t.tile
        parts.append(D[D.bcell != ""])
        stats_t.append(dict(tile=t.tile, join=how, rows=int(len(S)), assigned=float((cell != "").mean())))
    D = pd.concat(parts, ignore_index=True)
    # Baysor cell -> Xenium nucleus by nuclear transcripts
    D["xc"] = D.xcell.to_numpy() - 1                                      # -1: not in a Xenium cell
    nuc = (D.overlaps_nucleus.to_numpy() == 1) & (D.xc.to_numpy() >= 0)
    pair = (pd.DataFrame({"b": D.bcell.to_numpy()[nuc], "x": D.xc.to_numpy()[nuc]}).value_counts().rename("n")
            .reset_index().sort_values(["n", "b"], ascending=[False, True]))
    best_b = pair.drop_duplicates("b")
    # keep a cell only in the tile whose core holds its Xenium nucleus centroid
    tiles_ = P.set_index("tile")
    bt = best_b.b.str.split(":", n=1).str[0].to_numpy()
    px, py = cxy[best_b.x.to_numpy(), 0], cxy[best_b.x.to_numpy(), 1]
    core = ((px >= tiles_.loc[bt, "core_x0"].to_numpy()) & (px < tiles_.loc[bt, "core_x1"].to_numpy())
            & (py >= tiles_.loc[bt, "core_y0"].to_numpy()) & (py < tiles_.loc[bt, "core_y1"].to_numpy()))
    in_core = best_b[core]
    best = in_core.drop_duplicates("x")                                   # one Baysor cell per Xenium nucleus
    b2x = dict(zip(best.b, best.x))
    n_b = int(D.bcell.nunique())
    D = D[D.bcell.isin(b2x)]
    dup = int(D.transcript_id.duplicated().sum())
    D = D.drop_duplicates("transcript_id", keep="first")
    order = np.array(sorted(b2x))
    row_of = pd.Series(np.arange(len(order)), index=order)
    xs = np.array([b2x[b] for b in order])
    T_all, _ = transcripts(raw, a.sample, a.min_qv)
    genes = np.array(sorted(T_all.feature_name.unique()))
    ni = ci.get_indexer(T_all.cell_id[T_all.overlaps_nucleus == 1])
    nuc_tot = np.bincount(ni[ni >= 0], minlength=len(ci))                 # nuclear transcripts per Xenium cell
    r = row_of.reindex(D.bcell).to_numpy().astype(np.int64)
    c = pd.Index(genes).get_indexer(D.feature_name)
    W = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(order), len(genes)))
    own = (D.overlaps_nucleus.to_numpy() == 1) & (D.xc.to_numpy() == xs[r])
    Nn = sp.csr_matrix((np.ones(int(own.sum()), np.float32), (r[own], c[own])), shape=W.shape)
    W.sum_duplicates()
    Nn.sum_duplicates()
    cell_ids = ci[xs].astype(str)
    cx = C.set_index("cell_id").loc[cell_ids]
    obs = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    obs["sample"], obs["donor"], obs["unit"], obs["panel"], obs["tissue"] = a.sample, a.sample, a.sample, a.panel, ""
    obs["section"] = f"{a.panel}:{a.sample}:baysor"
    obs["x"] = cx["x_centroid"].to_numpy(float)
    obs["y"] = cx["y_centroid"].to_numpy(float)
    for k in ("cell_area", "nucleus_area"):
        obs[k] = cx[k].to_numpy(float) if k in cx else np.nan
    obs["n_whole"] = np.asarray(W.sum(1)).ravel()
    obs["n_nucleus"] = np.asarray(Nn.sum(1)).ravel()
    obs["baysor_cell"] = order
    obs["tile"] = [b.split(":", 1)[0] for b in order]
    obs["share_own_nucleus"] = obs.n_nucleus.to_numpy() / np.maximum(nuc_tot[xs], 1)
    A = ad.AnnData(W, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    A.layers["nucleus"] = Nn
    A.layers["cyto"] = (W - Nn).tocsr()
    for k in ("sample", "donor", "unit", "panel", "section", "tissue"):
        A.obs[k] = A.obs[k].astype("category")
    stats = dict(tiles=stats_t, baysor_cells_with_counts=n_b, mapped_to_nucleus=int(len(best_b)),
                 kept_in_core=int(len(in_core)), kept=int(len(best)), xenium_cells=int(len(ci)),
                 transcripts_in_two_tiles=dup, cells=int(A.n_obs), genes=int(A.n_vars),
                 median_whole=float(np.median(obs.n_whole)), median_nucleus=float(np.median(obs.n_nucleus)),
                 share_in_nucleus=float(obs.n_nucleus.sum() / max(obs.n_whole.sum(), 1)),
                 median_share_own_nucleus=float(np.median(obs.share_own_nucleus)), min_qv=a.min_qv,
                 tiles_dir=str(tiles))
    A.uns["baysor"] = json.dumps(stats)
    out.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(dst, compression="gzip")
    print(f"Baysor cells with counts {n_b:,}; with a Xenium nucleus {len(best_b):,}; in their tile's core "
          f"{len(in_core):,}; kept (one per nucleus) {len(best):,} of {len(ci):,} Xenium cells; transcripts in two "
          f"tiles counted once: {dup:,}")
    print(f"wrote {dst}: {A.n_obs:,} cells x {A.n_vars:,} genes; median whole {stats['median_whole']:.0f}, nucleus "
          f"{stats['median_nucleus']:.0f}; share of its Xenium nucleus kept (median) {stats['median_share_own_nucleus']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    m = sub.add_parser("merge")
    for q in (p, m):
        q.add_argument("--raw", required=True)
        q.add_argument("--sample", required=True)
        q.add_argument("--tiles", required=True)
        q.add_argument("--min-qv", type=float, default=20.0)
    p.add_argument("--tile-um", type=float, default=1000.0)
    p.add_argument("--margin-um", type=float, default=30.0)
    p.add_argument("--min-tile-transcripts", type=int, default=5000)
    m.add_argument("--out", required=True)
    m.add_argument("--panel", default="Xenium")
    a = ap.parse_args()
    (cmd_prep if a.cmd == "prep" else cmd_merge)(a)


if __name__ == "__main__":
    main()
