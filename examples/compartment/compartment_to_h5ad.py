#!/usr/bin/env python
"""compartment_to_h5ad.py -- one .h5ad per section with whole-cell and nucleus-only counts of the same cells, from
Xenium or CosMx transcript tables (generalises bilous_to_h5ad.py).

Output (same layout as bilous_to_h5ad.py, so bilous_label.py / compartment_label.py and the rest of the chain apply):
  X                  whole-cell counts (all transcripts the platform's segmentation assigned to the cell)
  layers['nucleus']  transcripts of that cell inside the nucleus  (Xenium: overlaps_nucleus == 1;
                                                                   CosMx: CellComp == "Nuclear")
  layers['cyto']     X - nucleus
  obs: sample, donor, panel, section, unit, tissue, x, y (cell centroid, um), cell_area, nucleus_area (um2; NaN where
       the platform gives none), n_whole, n_nucleus (+ CosMx protein means: Mean.PanCK, Mean.CD45, Mean.CD68, ...)
  var: gene names.  uns['source']: input files with size and SHA-256, filters.  uns['check']: agreement of the whole-cell
       totals with the platform's own count matrix (Xenium matrix.mtx/cell_feature_matrix.h5; CosMx exprMat_file).

Xenium (one sample per file prefix):
  --platform xenium --raw DIR --sample GSM9345477_Xenium_I0294I
  reads <sample>_transcripts.parquet(.gz), <sample>_cells.parquet(.gz) and, if present, the count matrix.
CosMx (one slide per file prefix; a slide can hold several tissue pieces):
  --platform cosmx --raw DIR --sample 262I275I3119
  2021 flat files (He et al. 2022 NSCLC; tx_file and metadata with fov + cell_ID and no cell id column; CellComp
  "0" = extracellular) are read too; give their pixel size, --px-um 0.18.
  reads <sample>_tx_file.csv(.gz), <sample>_metadata_file.csv(.gz) and, if present, <sample>_exprMat_file.csv(.gz).
  Tissue pieces on one slide are separated by connected components of the field-of-view centers (gap > --fov-gap x the
  median FOV spacing); each piece is written as its own section <sample>_p<k>, unit = slide (patients per piece are
  not in the deposit, so the hold-out unit is the slide).

The inputs are only read; .gz parquet is decompressed to a temporary copy in --tmp (deleted afterwards).
Refuses to overwrite an existing output file.

  python compartment_to_h5ad.py --platform xenium --raw ~/claimbreak_data/IBD/raw/xenium \
      --sample GSM9345477_Xenium_I0294I --out ~/claimbreak_data/IBD/h5ad --tmp ~/claimbreak_data/IBD/tmp
  python compartment_to_h5ad.py --platform cosmx --raw ~/claimbreak_data/IBD/raw/cosmx \
      --sample 262I275I3119 --out ~/claimbreak_data/IBD/h5ad --tmp ~/claimbreak_data/IBD/tmp
"""
import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL = re.compile(r"^(NegControl|NegPrb|Negative|BLANK|Blank|Unassigned|Intergenic|DeprecatedCodeword|antisense|"
                     r"UnassignedCodeword|SystemControl|FalseCode|falsecode)", re.I)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def find(raw, sample, stems):
    for stem in stems:
        for suf in ("", ".gz"):
            p = raw / f"{sample}{stem}{suf}"
            if p.exists():
                return p
    return None


def plain(p, tmp):
    """A seekable uncompressed copy in tmp for .gz parquet inputs."""
    if p.suffix != ".gz":
        return p, False
    q = tmp / p.name[:-3]
    with gzip.open(p, "rb") as fi, open(q, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 24)
    return q, True


class Counter:
    """Accumulates whole and nucleus counts (cells x genes) from batches of (cell index, gene name, in_nucleus)."""

    def __init__(self, n_cells):
        self.n, self.genes, self.gidx, self.W, self.N = n_cells, [], {}, None, None

    def add(self, r, g, nuc):
        for x in pd.unique(g):
            if x not in self.gidx:
                self.gidx[x] = len(self.genes)
                self.genes.append(x)
        c = pd.Series(g).map(self.gidx).to_numpy()
        shape = (self.n, 60000)
        w = sp.coo_matrix((np.ones(len(r), np.int32), (r, c)), shape=shape).tocsr()
        nn = sp.coo_matrix((np.ones(int(nuc.sum()), np.int32), (r[nuc], c[nuc])), shape=shape).tocsr()
        self.W = w if self.W is None else self.W + w
        self.N = nn if self.N is None else self.N + nn

    def result(self):
        G = len(self.genes)
        if G == 0:
            sys.exit("no transcripts kept")
        W, N = self.W[:, :G].tocsr(), self.N[:, :G].tocsr()
        order = np.argsort(self.genes)
        return np.array(self.genes)[order], W[:, order], N[:, order]


# ------------------------------------------------------------------------------------------------- Xenium
def xenium_meta(sample):
    """panel, donor, tissue from GSE312420-style names (GSM.._Xenium_[5k_|337_NoSegment_|337_withSegment_|Colon_]ID)."""
    s = sample.split("_", 1)[1] if sample.startswith("GSM") else sample
    s = re.sub(r"^Xenium_", "", s)
    panel = "337"
    for key, name in (("5k_", "5K"), ("337_NoSegment_", "337NS"), ("337_withSegment_", "337WS"), ("Colon_", "Colon")):
        if s.startswith(key):
            panel, s = name, s[len(key):]
            break
    m = re.match(r"(I\d{4})([IR]?)", s)
    donor = m.group(1) if m else s
    tissue = {"I": "ileum", "R": "rectum"}.get(m.group(2) if m else "", "rectum")
    return panel, donor, tissue


def xenium(a, raw, tmp):
    ftr = find(raw, a.sample, ["_transcripts.parquet"])
    fcells = find(raw, a.sample, ["_cells.parquet"])
    if ftr is None or fcells is None:
        sys.exit(f"transcripts or cells parquet missing for {a.sample} in {raw}")
    import pyarrow.parquet as pq
    pc, rm = plain(fcells, tmp)
    cells = pq.read_table(pc).to_pandas()
    if rm:
        pc.unlink()
    cells["cell_id"] = cells["cell_id"].astype(str)
    ci = pd.Index(cells["cell_id"])
    if not ci.is_unique:
        sys.exit("cell_id not unique in cells.parquet")
    pt, rm = plain(ftr, tmp)
    pf = pq.ParquetFile(pt)
    need = {"cell_id", "feature_name", "qv", "overlaps_nucleus"}
    if not need <= set(pf.schema_arrow.names):
        sys.exit(f"transcripts lack {sorted(need - set(pf.schema_arrow.names))}")
    C = Counter(len(ci))
    n_tot = n_kept = 0
    for rb in pf.iter_batches(batch_size=a.batch, columns=sorted(need)):
        d = rb.to_pandas()
        n_tot += len(d)
        f, cid = d["feature_name"].astype(str), d["cell_id"].astype(str)
        keep = (d["qv"].to_numpy() >= a.min_qv) & ~f.str.match(CONTROL).to_numpy() & cid.isin(ci).to_numpy()
        if keep.any():
            C.add(ci.get_indexer(cid[keep]), f[keep].to_numpy(), d["overlaps_nucleus"].to_numpy()[keep].astype(bool))
            n_kept += int(keep.sum())
        print(f"  {n_tot:,} transcripts read, {n_kept:,} kept", flush=True)
    if rm:
        pt.unlink()
    genes, W, N = C.result()
    panel, donor, tissue = xenium_meta(a.sample)
    obs = pd.DataFrame(index=pd.Index(ci, name="cell_id"))
    obs["x"] = cells["x_centroid"].to_numpy(float)
    obs["y"] = cells["y_centroid"].to_numpy(float)
    for c in ("cell_area", "nucleus_area"):
        obs[c] = cells[c].to_numpy(float) if c in cells else np.nan
    obs["sample"], obs["donor"], obs["panel"], obs["tissue"] = a.sample, donor, panel, tissue
    obs["section"] = f"{panel}:{a.sample}"
    obs["unit"] = donor
    files = [p for p in (ftr, fcells) if p is not None]
    check = xenium_check(raw, a.sample, obs, genes, W, tmp)
    return [(a.sample, obs, genes, W, N)], files, dict(min_qv=a.min_qv, read=n_tot, kept=n_kept), check


def xenium_check(raw, sample, obs, genes, W, tmp):
    fh5 = find(raw, sample, ["_cell_feature_matrix.h5"])
    fmtx = find(raw, sample, ["_matrix.mtx"])
    tot = None
    if fh5 is not None:
        import h5py
        p, rm = plain(fh5, tmp)
        with h5py.File(p, "r") as h:
            m = h["matrix"]
            bc = np.array(m["barcodes"]).astype(str)
            ft = np.array(m["features"]["feature_type"]).astype(str)
            nm = np.array(m["features"]["name"]).astype(str)
            M = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
        if rm:
            p.unlink()
        # older Xenium software files deprecated and control codewords under "Gene Expression": exclude them by name
        ge = (ft == "Gene Expression") & ~pd.Series(nm).str.match(CONTROL).to_numpy()
        tot = pd.Series(np.asarray(M[ge].sum(0)).ravel(), index=bc)
    elif fmtx is not None:
        from scipy.io import mmread
        fb, ff = find(raw, sample, ["_barcodes.tsv"]), find(raw, sample, ["_features.tsv"])
        if fb is None or ff is None:
            return {}
        M = sp.csc_matrix(mmread(gzip.open(fmtx) if fmtx.suffix == ".gz" else fmtx))
        bc = pd.read_csv(fb, sep="\t", header=None)[0].astype(str).to_numpy()
        F = pd.read_csv(ff, sep="\t", header=None)
        ge = (F[2].astype(str) == "Gene Expression").to_numpy() if F.shape[1] > 2 else np.ones(len(F), bool)
        tot = pd.Series(np.asarray(M[ge].sum(0)).ravel(), index=bc)
    if tot is None:
        return {}
    t = tot.reindex(obs.index).to_numpy()
    ok = np.isfinite(t)
    n = np.asarray(W.sum(1)).ravel()
    agree = float(np.mean(t[ok] == n[ok])) if ok.any() else float("nan")
    print(f"check vs platform matrix: {ok.sum():,} cells matched; totals equal for {agree:.4f}", flush=True)
    return {"cells_matched": int(ok.sum()), "share_cells_total_equal": agree}


# ------------------------------------------------------------------------------------------------- CosMx
def pieces(meta, gap):
    """Connected components of FOV centers; two FOVs are linked when closer than gap x the median nearest spacing."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    F = meta.groupby("fov")[["CenterX_global_px", "CenterY_global_px"]].median()
    xy = F.to_numpy(float)
    if len(xy) < 2:
        return pd.Series(0, index=F.index)
    d, _ = cKDTree(xy).query(xy, k=2)
    step = float(np.median(d[:, 1]))
    pairs = cKDTree(xy).query_pairs(gap * step, output_type="ndarray")
    G = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    _, lab = connected_components(G, directed=False)
    # order pieces by x then y of their center, so names are stable
    cen = pd.DataFrame(xy, columns=["x", "y"]).groupby(lab).mean().sort_values(["x", "y"])
    rank = {old: new for new, old in enumerate(cen.index)}
    return pd.Series([rank[v] for v in lab], index=F.index)


DEDUP_KEYS = ["fov", "x_local_px", "y_local_px", "x_global_px", "y_global_px", "z", "target"]
COMP = {"Nuclear": 1, "Cytoplasm": 2, "Membrane": 3}            # anything else (None, NaN) -> 0
COMP_NAMES = np.array(["None", "Nuclear", "Cytoplasm", "Membrane"])


def doubled(ftx, batch, rule, threshold=0.45):
    """First pass over a CosMx tx_file. Some deposited tx_files hold every transcript twice (same fov, position, z, gene
    and cell; the exprMat then has half the counts), sometimes with different CellComp calls on the two copies.
    A transcript is identified by a 64-bit hash of (fov, position, z, target). If the share of rows that repeat an
    earlier row is below `threshold`, the file is taken as not doubled (the few repeats are separate molecules on the
    same whole pixel) and nothing is dropped. Otherwise each group of n rows is n/2 molecules: the first ceil(n/2) rows
    are kept, and each kept row's compartment is set from it and its partner (the row n/2 later in the group) by `rule`:
      first   the earlier copy's call;   second  the later copy's call;
      both    Nuclear only if both copies say Nuclear (else the non-nuclear call);
      either  Nuclear if either copy says Nuclear (else the earlier call).
    Returns (keep mask, compartment codes in file order, info) or (None, None, info)."""
    cols = pd.read_csv(ftx, nrows=0).columns
    keys = [c for c in DEDUP_KEYS if c in cols]
    if "target" not in keys or not ({"x_local_px", "x_global_px"} & set(keys)):
        print(f"  no position columns, duplicate check skipped ({list(cols)})", flush=True)
        return None, None, {"dedup": "not possible"}
    hs, cs = [], []
    for d in pd.read_csv(ftx, usecols=keys + ["CellComp"], chunksize=batch, dtype={"CellComp": str, "target": str}):
        cs.append(d["CellComp"].map(COMP).fillna(0).to_numpy(np.int8))
        hs.append(pd.util.hash_pandas_object(d[keys], index=False).to_numpy())
    h, comp = np.concatenate(hs), np.concatenate(cs)
    order = np.argsort(h, kind="stable")                         # stable: file order within a group
    hsrt = h[order]
    start = np.r_[True, hsrt[1:] != hsrt[:-1]]
    gid = np.cumsum(start) - 1
    gstart = np.flatnonzero(start)
    gsize = np.diff(np.r_[gstart, len(h)])
    n = gsize[gid]                                               # group size, per sorted row
    k = np.arange(len(h)) - gstart[gid]                          # occurrence number within the group
    share = float(np.mean(k > 0))
    sizes = pd.Series(gsize).value_counts().sort_index()
    info = {"rows": int(len(h)), "share_repeated_rows": share, "group_sizes": {int(x): int(y) for x, y in sizes.items()},
            "keys": keys}
    print(f"  repeated rows (same {', '.join(keys)}): {share:.2%}; rows per transcript {info['group_sizes']}", flush=True)
    if share < threshold:
        info["dedup"] = "none (file not doubled)"
        return None, None, info
    half = n // 2
    idx = np.arange(len(h))
    keep_s = k < (n + 1) // 2
    partner = np.where(keep_s & (half > 0) & (k + half < n), idx + half, idx)
    own, other = comp[order], comp[order][partner]
    paired = keep_s & (partner != idx)
    X = pd.crosstab(pd.Series(COMP_NAMES[own[paired]], name="earlier"), pd.Series(COMP_NAMES[other[paired]], name="later"))
    print("  CellComp of the earlier (rows) x the later copy (columns):")
    print("    " + X.to_string().replace("\n", "\n    "), flush=True)
    if rule == "first":
        new = own
    elif rule == "second":
        new = other
    elif rule == "both":
        new = np.where((own == 1) & (other == 1), 1, np.where(own == 1, other, own))
    else:                                                        # either
        new = np.where((own == 1) | (other == 1), 1, own)
    keep = np.zeros(len(h), bool)
    keep[order[keep_s]] = True
    out = np.zeros(len(h), np.int8)
    out[order] = new
    diag = X.reindex(index=X.columns, fill_value=0).to_numpy()
    info.update(dedup=f"halved; compartment rule '{rule}'", rows_dropped=int((~keep).sum()),
                copies_crosstab={f"{r}|{c}": int(X.loc[r, c]) for r in X.index for c in X.columns},
                share_pairs_disagree=float(1 - np.trace(diag) / max(X.to_numpy().sum(), 1)))
    print(f"  file doubled: {info['rows_dropped']:,} rows dropped; copies disagree on CellComp in "
          f"{info['share_pairs_disagree']:.2%} of pairs; rule '{rule}'", flush=True)
    return keep, out, info


def cosmx(a, raw, tmp):
    ftx = find(raw, a.sample, ["_tx_file.csv"])
    fmeta = find(raw, a.sample, ["_metadata_file.csv"])
    if ftx is None or fmeta is None:
        sys.exit(f"tx_file or metadata_file missing for {a.sample} in {raw}")
    meta = pd.read_csv(fmeta)
    # 2021 flat files (He et al. 2022 NSCLC): no cell id column, the cell is (fov, cell_ID); build "fov_cellID"
    legacy = not ({"cell", "cell_id"} & set(meta.columns))
    if legacy:
        if not {"fov", "cell_ID"} <= set(meta.columns):
            sys.exit(f"metadata has neither a cell id nor fov + cell_ID: {list(meta.columns)[:12]}")
        meta["cell"] = meta["fov"].astype(str) + "_" + meta["cell_ID"].astype(str)
        if a.px_um == 0.12028:
            print("  WARNING 2021 flat files: pixel size taken as --px-um 0.12028; the 2021 NSCLC README gives 0.18 um",
                  flush=True)
    idc = "cell" if "cell" in meta else "cell_id"
    meta[idc] = meta[idc].astype(str)
    ci = pd.Index(meta[idc])
    if not ci.is_unique:
        sys.exit(f"{idc} not unique in metadata")
    # pixel size from Area (px) and Area.um2
    px = float(np.sqrt(np.median(meta["Area.um2"] / meta["Area"]))) if {"Area", "Area.um2"} <= set(meta) else a.px_um
    print(f"{a.sample}: {len(meta):,} cells, pixel {px:.5f} um", flush=True)
    C = Counter(len(ci))
    n_tot = n_kept = off = 0
    comps = {}
    first, newcomp, dinfo = (None, None, {"dedup": "off"}) if a.no_dedup else doubled(ftx, a.batch, a.dup_comp)
    txcols = ["fov", "cell_ID", "target", "CellComp"] if legacy else ["cell", "target", "CellComp"]
    for d in pd.read_csv(ftx, usecols=txcols, chunksize=a.batch,
                         dtype={"cell": str, "fov": str, "cell_ID": str, "target": str, "CellComp": str}):
        if legacy:                       # cell_ID 0 = not in a cell: not in the metadata, so dropped below
            d["cell"] = d["fov"] + "_" + d["cell_ID"]
        n_raw = len(d)
        if first is not None:
            m = first[off:off + n_raw]
            d = d[m].copy()
            d["CellComp"] = COMP_NAMES[newcomp[off:off + n_raw][m]]
        off += n_raw
        n_tot += n_raw
        d["CellComp"] = d["CellComp"].fillna("None").replace({"0": "None", "Extracellular": "None"})
        vc = d["CellComp"].value_counts()
        for k, v in vc.items():
            comps[k] = comps.get(k, 0) + int(v)
        f = d["target"].astype(str)
        keep = ~f.str.match(CONTROL).to_numpy() & d["cell"].isin(ci).to_numpy() & (d["CellComp"] != "None").to_numpy()
        if keep.any():
            C.add(ci.get_indexer(d["cell"][keep]), f[keep].to_numpy(), (d["CellComp"][keep] == "Nuclear").to_numpy())
            n_kept += int(keep.sum())
        print(f"  {n_tot:,} transcripts read, {n_kept:,} kept", flush=True)
    genes, W, N = C.result()
    obs_all = pd.DataFrame(index=pd.Index(ci, name="cell_id"))
    obs_all["x"] = meta["CenterX_global_px"].to_numpy(float) * px
    obs_all["y"] = meta["CenterY_global_px"].to_numpy(float) * px
    obs_all["cell_area"] = meta["Area.um2"].to_numpy(float) if "Area.um2" in meta else meta["Area"].to_numpy(float) * px ** 2
    obs_all["nucleus_area"] = np.nan
    for c in [c for c in meta.columns if c.startswith("Mean.")]:
        obs_all[c] = meta[c].to_numpy(float)
    obs_all["fov"] = meta["fov"].to_numpy()
    pc = pieces(meta, a.fov_gap)
    obs_all["piece"] = obs_all["fov"].map(pc).to_numpy()
    panel = "CosMx6k" if len(genes) > 3000 else "CosMx1k"      # by panel size (1k: ~1,000 genes; 6k: ~6,000)
    out = []
    for k in sorted(obs_all["piece"].unique()):
        m = (obs_all["piece"] == k).to_numpy()
        if m.sum() < a.min_piece_cells:
            print(f"  piece {k}: {m.sum():,} cells, skipped (< --min-piece-cells)", flush=True)
            continue
        name = f"{a.sample}_p{k}"
        obs = obs_all[m].copy()
        obs["sample"], obs["donor"], obs["panel"], obs["tissue"] = name, a.sample, panel, ""
        obs["section"] = f"{panel}:{name}"
        obs["unit"] = a.sample
        print(f"  piece {k}: {m.sum():,} cells, {obs.fov.nunique()} FOVs", flush=True)
        out.append((name, obs, genes, W[m], N[m]))
    check = {"CellComp_counts": comps}
    fexp = find(raw, a.sample, ["_exprMat_file.csv"])
    if fexp is not None and not a.no_check:
        E = pd.read_csv(fexp)
        cols = [c for c in E.columns if c not in ("fov", "cell_ID", "cell") and not c.startswith("Unnamed")
                and not CONTROL.match(c) and pd.api.types.is_numeric_dtype(E[c])]
        if "cell" in E:
            key = E["cell"].astype(str).to_numpy()
        else:                                   # map (fov, cell_ID) to the metadata's cell id
            mk = pd.Series(meta[idc].to_numpy(), index=pd.MultiIndex.from_arrays([meta["fov"], meta["cell_ID"]]))
            key = mk.reindex(pd.MultiIndex.from_arrays([E["fov"], E["cell_ID"]])).to_numpy()
        t = pd.Series(E[cols].sum(axis=1).to_numpy(), index=key).groupby(level=0).sum().reindex(ci).to_numpy()
        ok = np.isfinite(t)
        n = np.asarray(W.sum(1)).ravel()
        check.update(cells_matched=int(ok.sum()), share_cells_total_equal=float(np.mean(t[ok] == n[ok])))
        check["median_ratio_exprMat_to_tx"] = float(np.median(t[ok & (n > 0)] / n[ok & (n > 0)]))
        print(f"check vs exprMat: {ok.sum():,} cells; totals equal for {check['share_cells_total_equal']:.4f} "
              f"(median ratio exprMat/tx {check['median_ratio_exprMat_to_tx']:.3f})", flush=True)
    files = [p for p in (ftx, fmeta, fexp) if p is not None]
    return out, files, dict(read=n_tot, kept=n_kept, duplicates=dinfo, pixel_um=px,
                            fov_gap=a.fov_gap), check


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", choices=["xenium", "cosmx"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sample", required=True, help="file prefix (Xenium sample or CosMx slide)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--min-qv", type=float, default=20.0, help="Xenium only")
    ap.add_argument("--batch", type=int, default=5_000_000)
    ap.add_argument("--fov-gap", type=float, default=1.6, help="CosMx: FOV link distance, x median FOV spacing")
    ap.add_argument("--min-piece-cells", type=int, default=2000, help="CosMx: tissue pieces with fewer cells skipped")
    ap.add_argument("--px-um", type=float, default=0.12028, help="CosMx pixel size if the metadata has no Area.um2")
    ap.add_argument("--no-check", action="store_true", help="skip the comparison with the platform count matrix")
    ap.add_argument("--no-dedup", action="store_true",
                    help="CosMx: never halve a doubled tx_file (default: halved when >= 45%% of rows repeat)")
    ap.add_argument("--dup-comp", choices=["first", "second", "both", "either"], default="first",
                    help="CosMx doubled files: compartment when the two copies disagree (see doubled())")
    ap.add_argument("--no-sha", action="store_true", help="skip SHA-256 of the inputs (faster for large CosMx files)")
    a = ap.parse_args()
    raw, out, tmp = (Path(p).expanduser().resolve() for p in (a.raw, a.out, a.tmp))
    if out == raw or raw in out.parents or tmp == raw or raw in tmp.parents:
        sys.exit("--out and --tmp must be outside the raw data folder")
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    if a.platform == "xenium":
        dst = out / f"{a.sample}.h5ad"
        if dst.exists():
            sys.exit(f"{dst} exists; not overwritten")
    elif list(out.glob(f"{a.sample}_p*.h5ad")):
        sys.exit(f"{a.sample}_p*.h5ad exist in {out}; not overwritten")
    sections, files, info, check = (xenium if a.platform == "xenium" else cosmx)(a, raw, tmp)
    src = {p.name: {"bytes": p.stat().st_size, **({} if a.no_sha else {"sha256": sha(p)})} for p in files}
    import anndata as ad
    for name, obs, genes, W, N in sections:
        obs["n_whole"] = np.asarray(W.sum(1)).ravel()
        obs["n_nucleus"] = np.asarray(N.sum(1)).ravel()
        A = ad.AnnData(X=W.astype(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
        A.layers["nucleus"] = N.astype(np.float32)
        A.layers["cyto"] = (W - N).astype(np.float32)
        A.uns["source"] = json.dumps({"platform": a.platform, "files": src, "control_regex": CONTROL.pattern, **info})
        A.uns["check"] = json.dumps(check)
        for c in ("sample", "donor", "panel", "section", "unit", "tissue"):
            A.obs[c] = A.obs[c].astype(str).astype("category")
        dst = out / f"{name}.h5ad"
        A.write_h5ad(dst, compression="gzip")
        share = obs.n_nucleus.sum() / max(obs.n_whole.sum(), 1)
        print(f"wrote {dst}: {A.n_obs:,} cells x {A.n_vars:,} genes; median counts whole {np.median(obs.n_whole):.0f}, "
              f"nucleus {np.median(obs.n_nucleus):.0f} (share in nucleus {share:.3f})", flush=True)


if __name__ == "__main__":
    main()
