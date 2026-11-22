"""Platform output -> one compartment .h5ad per section: whole-cell counts in X, the nuclear counts of the same
cells in layers['nucleus'], the rest in layers['cyto'].

This is the Xenium part of examples/compartment/compartment_to_h5ad.py, packaged for `claimbreak kept-ratio`, and it
also reads a standard Xenium output folder (`outs/` as written by the instrument: transcripts.parquet, cells.parquet,
cell_feature_matrix.h5), not only GEO-style prefixed files (<sample>_transcripts.parquet). The counts it writes are
the same as the example script's for the same inputs (tests/test_kept_ratio.py checks this). CosMx: use the example
script, which handles the 2021 flat files, doubled tx_files and tissue pieces; its .h5ad files go to
`claimbreak kept-ratio --h5ad`.

The inputs are only read. A .gz parquet is decompressed to a temporary copy in `tmp` and deleted afterwards.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL = re.compile(r"^(NegControl|NegPrb|Negative|BLANK|Blank|Unassigned|Intergenic|DeprecatedCodeword|antisense|"
                     r"UnassignedCodeword|SystemControl|FalseCode|falsecode)", re.I)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def xenium_files(folder, sample: str = ""):
    """Paths of the transcripts, cells and count-matrix files of one Xenium section in `folder`: prefixed
    (<sample>_transcripts.parquet, as deposited on GEO) when `sample` is given, else the instrument's own names
    (transcripts.parquet, cells.parquet, cell_feature_matrix.h5), each optionally .gz."""
    folder = Path(folder).expanduser()

    def one(stem):
        names = [f"{sample}_{stem}"] if sample else [stem]
        for n in names:
            for suf in ("", ".gz"):
                p = folder / f"{n}{suf}"
                if p.exists():
                    return p
        return None
    return {"transcripts": one("transcripts.parquet"), "cells": one("cells.parquet"),
            "matrix": one("cell_feature_matrix.h5")}


def find_xenium_sections(root, depth: int = 3):
    """Xenium output folders under `root` (the folder itself, or subfolders down to `depth` levels), as (folder,
    sample prefix, name). A folder with transcripts.parquet is one section named after the folder (its parent's name
    for a folder called outs); a folder of GEO-style files holds one section per <prefix>_transcripts.parquet."""
    root = Path(root).expanduser()
    found = []
    dirs, frontier = [root], [root]
    for _ in range(depth):
        frontier = sorted(q for p in frontier for q in p.iterdir() if q.is_dir() and not q.name.startswith("."))
        dirs += frontier
    for d in dirs:
        if xenium_files(d)["transcripts"] is not None:
            name = d.name if d.name != "outs" else d.parent.name
            found.append((d, "", name))
            continue
        for f in sorted(d.glob("*_transcripts.parquet*")):
            pre = re.sub(r"_transcripts\.parquet(\.gz)?$", "", f.name)
            found.append((d, pre, pre))
    return found


def _plain(p: Path, tmp: Path):
    if p.suffix != ".gz":
        return p, False
    q = tmp / p.name[:-3]
    with gzip.open(p, "rb") as fi, open(q, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 24)
    return q, True


def _matrix_totals(fh5: Path, tmp: Path):
    import h5py
    p, rm = _plain(fh5, tmp)
    with h5py.File(p, "r") as h:
        m = h["matrix"]
        bc = np.array(m["barcodes"]).astype(str)
        ft = np.array(m["features"]["feature_type"]).astype(str)
        nm = np.array(m["features"]["name"]).astype(str)
        M = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
    if rm:
        p.unlink()
    ge = (ft == "Gene Expression") & ~pd.Series(nm).str.match(CONTROL).to_numpy()
    return pd.Series(np.asarray(M[ge].sum(0)).ravel(), index=bc)


def xenium_to_h5ad(folder, out, name: str = "", sample: str = "", tmp=None, min_qv: float = 20.0,
                   batch: int = 5_000_000, unit: str = "", panel: str = "Xenium", sha: bool = True,
                   verbose: bool = True) -> Path:
    """One Xenium section -> OUT/<name>.h5ad. Transcripts with qv >= min_qv, not control codewords and inside a cell
    are counted; overlaps_nucleus == 1 puts a transcript in layers['nucleus']. obs: x, y (centroid, um), cell_area,
    nucleus_area, sample, donor, unit, panel, section, tissue, n_whole, n_nucleus. uns['source'] holds the input
    files (size, SHA-256) and filters, uns['check'] the agreement of the whole-cell totals with the instrument's
    count matrix. Refuses to overwrite and to write inside the input folder."""
    import anndata as ad
    import pyarrow.parquet as pq
    say = print if verbose else (lambda *a, **k: None)
    folder, out = Path(folder).expanduser().resolve(), Path(out).expanduser().resolve()
    if out == folder or folder in out.parents:
        raise ValueError("the output folder must be outside the Xenium folder")
    name = name or sample or folder.name
    dst = out / f"{name}.h5ad"
    if dst.exists():
        raise FileExistsError(f"{dst} exists; not overwritten")
    F = xenium_files(folder, sample)
    if F["transcripts"] is None or F["cells"] is None:
        raise FileNotFoundError(f"no transcripts.parquet and cells.parquet in {folder}"
                                + (f" for prefix {sample}" if sample else ""))
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tmp).expanduser() if tmp else out / f".tmp_{name}"
    tmp.mkdir(parents=True, exist_ok=True)
    pc, rm = _plain(F["cells"], tmp)
    cells = pq.read_table(pc).to_pandas()
    if rm:
        pc.unlink()
    cells["cell_id"] = cells["cell_id"].astype(str)
    ci = pd.Index(cells["cell_id"])
    if not ci.is_unique:
        raise ValueError("cell_id not unique in cells.parquet")
    pt, rm = _plain(F["transcripts"], tmp)
    pf = pq.ParquetFile(pt)
    need = {"cell_id", "feature_name", "qv", "overlaps_nucleus"}
    if not need <= set(pf.schema_arrow.names):
        raise ValueError(f"transcripts lack {sorted(need - set(pf.schema_arrow.names))}")
    genes, gidx, Ws, Ns = [], {}, None, None
    n_tot = n_kept = 0
    for rb in pf.iter_batches(batch_size=batch, columns=sorted(need)):
        d = rb.to_pandas()
        n_tot += len(d)
        f, cid = d["feature_name"].astype(str), d["cell_id"].astype(str)
        keep = (d["qv"].to_numpy() >= min_qv) & ~f.str.match(CONTROL).to_numpy() & cid.isin(ci).to_numpy()
        if keep.any():
            g = f[keep].to_numpy()
            for x in pd.unique(g):
                if x not in gidx:
                    gidx[x] = len(genes)
                    genes.append(x)
            r = ci.get_indexer(cid[keep])
            c = pd.Series(g).map(gidx).to_numpy()
            nuc = d["overlaps_nucleus"].to_numpy()[keep].astype(bool)
            shape = (len(ci), 60000)
            w = sp.coo_matrix((np.ones(len(r), np.int32), (r, c)), shape=shape).tocsr()
            nn = sp.coo_matrix((np.ones(int(nuc.sum()), np.int32), (r[nuc], c[nuc])), shape=shape).tocsr()
            Ws = w if Ws is None else Ws + w
            Ns = nn if Ns is None else Ns + nn
            n_kept += int(keep.sum())
        say(f"  {n_tot:,} transcripts read, {n_kept:,} kept", flush=True)
    if rm:
        pt.unlink()
    if not genes:
        raise ValueError("no transcripts kept")
    G = len(genes)
    order = np.argsort(genes)
    genes = np.array(genes)[order]
    W, N = Ws[:, :G].tocsr()[:, order], Ns[:, :G].tocsr()[:, order]
    obs = pd.DataFrame(index=pd.Index(ci, name="cell_id"))
    obs["x"] = cells["x_centroid"].to_numpy(float)
    obs["y"] = cells["y_centroid"].to_numpy(float)
    for c in ("cell_area", "nucleus_area"):
        obs[c] = cells[c].to_numpy(float) if c in cells else np.nan
    obs["sample"], obs["donor"], obs["unit"], obs["panel"], obs["tissue"] = name, unit or name, unit or name, panel, ""
    obs["section"] = f"{panel}:{name}"
    obs["n_whole"] = np.asarray(W.sum(1)).ravel()
    obs["n_nucleus"] = np.asarray(N.sum(1)).ravel()
    check = {}
    if F["matrix"] is not None:
        tot = _matrix_totals(F["matrix"], tmp).reindex(obs.index).to_numpy()
        ok = np.isfinite(tot)
        check = {"cells_matched": int(ok.sum()),
                 "share_cells_total_equal": float(np.mean(tot[ok] == obs.n_whole.to_numpy()[ok])) if ok.any()
                 else float("nan")}
        say(f"  check vs cell_feature_matrix.h5: {ok.sum():,} cells; totals equal for "
            f"{check['share_cells_total_equal']:.4f}", flush=True)
    try:
        tmp.rmdir()
    except OSError:
        pass
    files = [p for p in F.values() if p is not None]
    src = {p.name: {"bytes": p.stat().st_size, **({"sha256": _sha(p)} if sha else {})} for p in files}
    A = ad.AnnData(X=W.astype(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    A.layers["nucleus"] = N.astype(np.float32)
    A.layers["cyto"] = (W - N).astype(np.float32)
    A.uns["source"] = json.dumps({"platform": "xenium", "folder": str(folder), "files": src,
                                  "control_regex": CONTROL.pattern, "min_qv": min_qv, "read": n_tot, "kept": n_kept})
    A.uns["check"] = json.dumps(check)
    for c in ("sample", "donor", "panel", "section", "unit", "tissue"):
        A.obs[c] = A.obs[c].astype(str).astype("category")
    A.write_h5ad(dst, compression="gzip")
    share = obs.n_nucleus.sum() / max(obs.n_whole.sum(), 1)
    say(f"  wrote {dst}: {A.n_obs:,} cells x {A.n_vars:,} genes; median counts whole {np.median(obs.n_whole):.0f}, "
        f"nucleus {np.median(obs.n_nucleus):.0f} (share in nucleus {share:.3f})", flush=True)
    return dst
