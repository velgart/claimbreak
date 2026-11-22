#!/usr/bin/env python
"""reexpand.py -- the same cells, segmented again from their nuclei at several expansion distances, so that the mask
is the only thing that changes.

For one Xenium section (transcripts.parquet, cells.parquet and nucleus_boundaries.parquet; .gz accepted):
  nucleus     transcripts with overlaps_nucleus == 1 assigned to the cell (as layers['nucleus'] of bilous_to_h5ad.py)
  r<d>        nucleus + every other transcript (assigned to any cell or to none) whose nearest nucleus boundary is
              within d um, given to the cell owning that boundary (expansion by d um, nearest nucleus wins: the
              Voronoi-constrained nucleus expansion of Xenium's default segmentation, rebuilt for any d; r0 = nucleus)
  platform    the platform's own assignment (cell_id of the transcript): nucleus expansion on the Lung / Custom IO
              panels of GSE311609, which reaches about 15 um (0.96-0.97 of its transcripts outside nuclei are placed in
              the same cell by 15 um expansion; segseries results), multimodal stain segmentation on 5K and on
              most kidney cells
A cell with several nucleus polygons in nucleus_boundaries.parquet (column label_id) is densified polygon by polygon
(before, its polygons were joined into one ring, which put boundary points along the jump between them).
Cells: those in cells.parquet with a nucleus boundary. Transcripts: qv >= --min-qv, control codewords dropped (the
filters of bilous_to_h5ad.py). Boundary distances: nucleus polygons densified to --spacing um and searched with a k-d
tree (distance error <= spacing / 2).

Writes OUT/r<d>/<sample>.h5ad for every d in --radii and OUT/platform/<sample>.h5ad (X = counts of that mask,
layers nucleus and cyto, obs as bilous_to_h5ad.py plus `mask`), OUT/<sample>.cells.csv.gz (per cell: counts in the
nucleus, in every mask and from the platform) and OUT/<sample>.reexpand.json (QC: how the platform's transcripts
outside nuclei are placed by each expansion). The same sample name is used in every mask folder, so labels of
different masks can be set side by side cell by cell. Inputs are only read; refuses to overwrite.

  python reexpand.py --raw ~/claimbreak_data/Bilous/raw --sample GSM9509145_NSCLC_lung_L1_L1 --out ~/claimbreak_work/segser/h5ad \
      --tmp ~/claimbreak_work/segser/tmp [--radii 0,1,2,3,5,7,10,15] [--obs-from EXISTING.h5ad]
"""
import argparse
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

CONTROL = re.compile(r"^(NegControl|BLANK|Blank|Unassigned|Intergenic|DeprecatedCodeword|antisense|UnassignedCodeword)",
                     re.I)


def plain(p, tmp):
    """A seekable uncompressed copy in tmp for .gz inputs (parquet needs seeking)."""
    p = Path(p)
    if p.suffix != ".gz":
        return p, False
    q = Path(tmp) / p.name[:-3]
    with gzip.open(p, "rb") as fi, open(q, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 24)
    return q, True


def find(raw, sample, stem):
    for suf in ("", ".gz"):
        p = Path(raw) / f"{sample}_{stem}{suf}"
        if p.exists():
            return p
    return None


def densify(cell_idx, vx, vy, spacing):
    """Points every <= spacing um along each polygon (consecutive vertices with the same ring index, closed); returns
    the points and the ring index of each."""
    order = np.argsort(cell_idx, kind="stable")
    c, x, y = cell_idx[order], vx[order], vy[order]
    start = np.r_[True, c[1:] != c[:-1]]
    first = np.flatnonzero(start)
    nxt = np.arange(1, len(c) + 1)
    last = np.r_[first[1:], len(c)] - 1
    nxt[last] = first                                   # close each polygon
    x1, y1 = x[nxt], y[nxt]
    seg = np.hypot(x1 - x, y1 - y)
    n = np.maximum(np.ceil(seg / spacing).astype(int), 1)
    rep = np.repeat(np.arange(len(c)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    px = x[rep] + t * (x1[rep] - x[rep])
    py = y[rep] + t * (y1[rep] - y[rep])
    return np.c_[px, py], c[rep]


def sample_meta(sample):
    m = re.match(r"GSM\d+_(NSCLC|breast)_(5k|chuvio|lung|breast)_(\w+?)_(\w+)$", sample)
    panel = {"5k": "5K", "chuvio": "Custom IO", "lung": "Lung", "breast": "Breast"}.get(m.group(2), m.group(2)) if m else ""
    donor = m.group(3) if m else ""
    return panel, donor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--radii", default="0,1,2,3,5,7,10,15")
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--min-qv", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=5_000_000)
    ap.add_argument("--obs-from", default="", help="existing compartment .h5ad of this section: copy its obs columns "
                                                   "sample, donor, panel, section, unit, tissue for the same cells")
    a = ap.parse_args()
    raw, out, tmp = (Path(p).expanduser().resolve() for p in (a.raw, a.out, a.tmp))
    if raw == out or raw in out.parents or raw == tmp or raw in tmp.parents:
        sys.exit("--out and --tmp must be outside the raw data folder")
    radii = sorted({float(r) for r in a.radii.split(",") if r != ""})
    if radii[0] != 0:
        radii = [0.0] + radii
    tag = lambda r: f"r{r:g}"
    dsts = [out / tag(r) / f"{a.sample}.h5ad" for r in radii] + [out / "platform" / f"{a.sample}.h5ad"]
    if any(d.exists() for d in dsts):
        sys.exit(f"outputs for {a.sample} exist in {out}; not overwritten")
    tmp.mkdir(parents=True, exist_ok=True)
    ftr, fcells, fnuc = (find(raw, a.sample, s) for s in ("transcripts.parquet", "cells.parquet",
                                                            "nucleus_boundaries.parquet"))
    if ftr is None or fcells is None or fnuc is None:
        sys.exit(f"transcripts, cells or nucleus_boundaries parquet missing for {a.sample} in {raw}")

    # cells with a nucleus boundary
    p, rm = plain(fcells, tmp)
    cells = pd.read_parquet(p)
    if rm:
        p.unlink()
    cells["cell_id"] = cells["cell_id"].astype(str)
    p, rm = plain(fnuc, tmp)
    import pyarrow.parquet as pq
    bcols = ["cell_id", "vertex_x", "vertex_y"] + (["label_id"] if "label_id" in pq.ParquetFile(p).schema_arrow.names
                                                   else [])
    B = pd.read_parquet(p, columns=bcols)
    if rm:
        p.unlink()
    B["cell_id"] = B["cell_id"].astype(str)
    have = set(B["cell_id"].unique())
    cells = cells[cells["cell_id"].isin(have)].reset_index(drop=True)
    ci = pd.Index(cells["cell_id"])
    B = B[B["cell_id"].isin(ci)]
    # one ring per (cell, nucleus polygon); vertices stay in file order within a ring
    ring_key = B["cell_id"] + ("\x00" + B["label_id"].astype(str) if "label_id" in B else "")
    ring, ring_names = pd.factorize(ring_key, sort=False)
    ring_owner = ci.get_indexer(pd.Series(B["cell_id"].to_numpy()).groupby(ring).first().to_numpy())
    n_multi = int((pd.Series(ring_owner).value_counts() > 1).sum())
    if n_multi:
        print(f"{a.sample}: {n_multi:,} cells with more than one nucleus polygon (densified separately)", flush=True)
    pts, rr = densify(ring, B["vertex_x"].to_numpy(float), B["vertex_y"].to_numpy(float), a.spacing)
    owner = ring_owner[rr]
    tree = cKDTree(pts)
    print(f"{a.sample}: {len(ci):,} cells with a nucleus boundary; {len(pts):,} boundary points", flush=True)
    del B, pts

    # transcripts: per kept transcript (cell, gene, bin) for the nucleus/expansion masks and (cell, gene) for the platform
    p, rm = plain(ftr, tmp)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(p)
    need = ["cell_id", "feature_name", "qv", "overlaps_nucleus", "x_location", "y_location"]
    miss = set(need) - set(pf.schema_arrow.names)
    if miss:
        sys.exit(f"transcripts.parquet lacks {sorted(miss)}")
    rmax = radii[-1]
    redges = np.array(radii[1:])
    genes, gidx = [], {}
    E_c, E_g, E_b, P_c, P_g = [], [], [], [], []
    nb = len(radii) + 1
    qc = np.zeros((len(radii), 3), np.int64)            # per radius: platform outside-nucleus transcripts placed in
    n_tot = n_kept = 0                                  # the same cell, in another cell, in no cell
    for rb in pf.iter_batches(batch_size=a.batch, columns=need):
        d = rb.to_pandas()
        n_tot += len(d)
        f = d["feature_name"].astype(str)
        keep = (d["qv"].to_numpy() >= a.min_qv) & ~f.str.match(CONTROL).to_numpy()
        if not keep.any():
            continue
        d, f = d[keep], f[keep]
        n_kept += len(d)
        for g in pd.unique(f):
            if g not in gidx:
                gidx[g] = len(genes)
                genes.append(g)
        g = f.map(gidx).to_numpy(np.int32)
        pc = ci.get_indexer(d["cell_id"].astype(str))   # platform cell (-1: none or a cell without a nucleus)
        nuc = (d["overlaps_nucleus"].to_numpy() == 1) & (pc >= 0)
        b = np.full(len(d), nb, np.int16)               # nb = beyond the largest radius
        own = np.full(len(d), -1, np.int64)
        own[nuc], b[nuc] = pc[nuc], 0
        q = ~nuc
        dist, j = tree.query(np.c_[d["x_location"].to_numpy(float)[q], d["y_location"].to_numpy(float)[q]],
                             distance_upper_bound=rmax, workers=-1)
        ok = np.isfinite(dist)
        qi = np.flatnonzero(q)
        own[qi[ok]] = owner[j[ok]]
        b[qi[ok]] = 1 + np.searchsorted(redges, dist[ok], side="left")
        e = b <= len(radii) - 1
        E_c.append(own[e].astype(np.int32)); E_g.append(g[e]); E_b.append(b[e])
        pk = pc >= 0
        P_c.append(pc[pk].astype(np.int32)); P_g.append(g[pk])
        po = pk & ~nuc                                  # platform transcripts outside the nucleus
        for k in range(len(radii)):
            inr = po & (b <= k)
            qc[k, 0] += int((inr & (own == pc)).sum())
            qc[k, 1] += int((inr & (own != pc)).sum())
            qc[k, 2] += int((po & ~(b <= k)).sum())
        print(f"  {n_tot:,} transcripts read, {n_kept:,} kept", flush=True)
    if rm:
        p.unlink()
    E_c, E_g, E_b = np.concatenate(E_c), np.concatenate(E_g), np.concatenate(E_b)
    P_c, P_g = np.concatenate(P_c), np.concatenate(P_g)
    G = len(genes)
    order = np.argsort(genes)
    rank = np.empty(G, np.int32)
    rank[order] = np.arange(G)
    genes = np.array(genes)[order]
    E_g, P_g = rank[E_g], rank[P_g]

    def mat(c, g):
        return sp.coo_matrix((np.ones(len(c), np.float32), (c, g)), shape=(len(ci), G)).tocsr()

    N = mat(E_c[E_b == 0], E_g[E_b == 0])
    panel, donor = sample_meta(a.sample)
    obs = pd.DataFrame(index=pd.Index(ci.astype(str), name="cell_id"))
    obs["sample"], obs["donor"], obs["panel"] = a.sample, donor, panel
    obs["section"] = f"{panel}:{a.sample.split('_', 1)[1]}" if panel else a.sample
    if a.obs_from:
        import anndata as ad
        O = ad.read_h5ad(a.obs_from, backed="r").obs
        O.index = O.index.astype(str)
        for c in ("sample", "donor", "panel", "section", "unit", "tissue"):
            if c in O:
                obs[c] = O[c].astype(str).reindex(obs.index).fillna(str(O[c].iloc[0])).to_numpy()
    obs["x"] = cells["x_centroid"].to_numpy(float)
    obs["y"] = cells["y_centroid"].to_numpy(float)
    for c in ("cell_area", "nucleus_area"):
        obs[c] = cells[c].to_numpy(float) if c in cells else np.nan
    obs["n_nucleus"] = np.asarray(N.sum(1)).ravel()
    import anndata as ad
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    C = pd.DataFrame({"cell_id": ci.astype(str), "n_nucleus": obs["n_nucleus"].to_numpy()})
    masks = [(tag(r), E_b <= k) for k, r in enumerate(radii)] + [("platform", None)]
    for (name, m), dst in zip(masks, dsts):
        X = mat(E_c[m], E_g[m]) if m is not None else mat(P_c, P_g)
        o = obs.copy()
        o["mask"] = name
        o["n_whole"] = np.asarray(X.sum(1)).ravel()
        C[f"n_{name}"] = o["n_whole"].to_numpy()
        A = ad.AnnData(X=X, obs=o, var=var.copy())
        A.layers["nucleus"] = N.copy()
        cy = (X - N).tocsr()
        if cy.nnz and cy.data.min() < 0:
            sys.exit(f"{name}: nucleus counts exceed the mask's counts (bug)")
        A.layers["cyto"] = cy
        for c in ("sample", "donor", "panel", "section", "mask"):
            A.obs[c] = A.obs[c].astype("category")
        A.uns["source"] = json.dumps({"reexpand": vars(a), "mask": name})
        dst.parent.mkdir(parents=True, exist_ok=True)
        A.write_h5ad(dst, compression="gzip")
        print(f"  wrote {dst}: median counts {np.median(o.n_whole):.0f} (nucleus {np.median(o.n_nucleus):.0f})",
              flush=True)
    C.to_csv(out / f"{a.sample}.cells.csv.gz", index=False, compression="gzip")
    tot = qc.sum(1)
    Q = pd.DataFrame({"mask": [tag(r) for r in radii], "same_cell": qc[:, 0] / tot, "other_cell": qc[:, 1] / tot,
                      "no_cell": qc[:, 2] / tot})
    pl = C["n_platform"].to_numpy(float)
    Q["cell_total_corr_vs_platform"] = [float(np.corrcoef(C[f"n_{tag(r)}"], pl)[0, 1]) for r in radii]
    Q["median_count_ratio_vs_platform"] = [float(np.median(C[f"n_{tag(r)}"] / np.maximum(pl, 1))) for r in radii]
    info = dict(sample=a.sample, cells=int(len(ci)), transcripts_read=n_tot, transcripts_kept=n_kept,
                radii=radii, spacing=a.spacing, min_qv=a.min_qv, qc=Q.to_dict(orient="records"))
    (out / f"{a.sample}.reexpand.json").write_text(json.dumps(info, indent=1))
    print("platform transcripts outside nuclei, placed by each expansion (share of them):\n"
          + Q.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
