#!/usr/bin/env python
"""corr_export.py -- inputs for the correction test (SPLIT, cellAdmix) on GSE311609 lung sections
.

Per section: a central square window holding at most --max-cells cells (the same cells for every arm), and in
OUT/<sample>/
  counts.mtx, genes.txt, cells.csv   whole-cell counts (genes x cells, Matrix Market, integer), gene names, and per
                                     cell: cell_id, x, y, cell_area, nucleus_area, <level>_whole, <level>_nucleus (the
                                     claimbreak labels of the section)
  molecules.csv.gz                   (with --raw) transcripts of those cells for cellAdmix: mol_id, x, y, z, gene,
                                     cell, celltype (= <level>_whole, the analyst's label), overlaps_nucleus; same
                                     filters as bilous_to_h5ad.py (qv >= 20, gene probes), checked against the h5ad
  window.json                        the window, cell counts, molecule check
and once, OUT/ref/ (with --ref): counts.mtx (genes x cells), genes.txt, cells.csv (cell, type at --level, donor) for
the genes of the exported panels, read with the same settings as the labels (claimbreak Reference.load).

  python corr_export.py --h5ad-dir ~/claimbreak_data/Bilous_h5ad --labels ~/claimbreak_results/bilous_v4 \
      --raw ~/claimbreak_data/Bilous/raw --ref ~/claimbreak_data/Bilous/ref/<file>.h5ad \
      --samples lung_L1_L1,lung_L2_L2_1,lung_L2_L2_2,lung_L3_L3,lung_L4_L4 --out ~/claimbreak_data/Bilous_corr/input \
      --tmp ~/claimbreak_data/Bilous_tmp
Never overwrites (refuses an existing section folder).
"""
import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

CONTROL = re.compile(r"^(NegControl|BLANK|Blank|Unassigned|Intergenic|DeprecatedCodeword|antisense|UnassignedCodeword)",
                     re.I)


def window(xy, max_cells):
    """Largest square centerd on the median centroid holding at most max_cells cells: (mask, box)."""
    c = np.median(xy, 0)
    d = np.abs(xy - c).max(1)                      # Chebyshev distance to the center
    if len(d) <= max_cells:
        h = float(d.max()) + 1.0
    else:
        h = float(np.sort(d)[max_cells - 1])
    m = d <= h
    while m.sum() > max_cells:                     # ties at the edge
        h = np.nextafter(h, 0)
        m = d <= h
    return m, dict(cx=float(c[0]), cy=float(c[1]), half_width=float(h))


def plain(p, tmp):
    p = Path(p)
    if p.suffix != ".gz":
        return p, False
    q = Path(tmp) / p.name[:-3]
    with gzip.open(p, "rb") as fi, open(q, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 24)
    return q, True


def molecules(raw, sample, cell_ids, celltype, genes, tmp, min_qv=20.0, batch=5_000_000):
    import pyarrow.parquet as pq
    src = None
    for suf in ("", ".gz"):
        p = Path(raw) / f"{sample}_transcripts.parquet{suf}"
        if p.exists():
            src = p
            break
    if src is None:
        raise FileNotFoundError(f"{sample}_transcripts.parquet(.gz) not in {raw}")
    pt, rm = plain(src, tmp)
    pf = pq.ParquetFile(pt)
    cols = [c for c in ("cell_id", "feature_name", "qv", "overlaps_nucleus", "x_location", "y_location", "z_location")
            if c in pf.schema_arrow.names]
    keep_cells = pd.Index(cell_ids)
    gset = set(genes)
    parts = []
    for rb in pf.iter_batches(batch_size=batch, columns=cols):
        d = rb.to_pandas()
        f = d["feature_name"].astype(str)
        cid = d["cell_id"].astype(str)
        k = ((d["qv"].to_numpy() >= min_qv) & ~f.str.match(CONTROL).to_numpy() & f.isin(gset).to_numpy()
             & cid.isin(keep_cells).to_numpy())
        if k.any():
            parts.append(pd.DataFrame({"x": d["x_location"].to_numpy()[k], "y": d["y_location"].to_numpy()[k],
                                       "z": d["z_location"].to_numpy()[k] if "z_location" in d else 0.0,
                                       "gene": f.to_numpy()[k], "cell": cid.to_numpy()[k],
                                       "overlaps_nucleus": d["overlaps_nucleus"].to_numpy()[k].astype(int)}))
    if rm:
        pt.unlink()
    M = pd.concat(parts, ignore_index=True)
    M.insert(0, "mol_id", [f"m{i}" for i in range(len(M))])
    M["celltype"] = M["cell"].map(celltype)
    return M


def export_ref(ref_path, genes, out, level, ref_level_col, merge, donor):
    from claimbreak.compartment import Reference
    R = Reference.load(ref_path, {level: ref_level_col}, merge=merge, donor=donor)
    g, gidx, missing = R.match(sorted(genes))
    out.mkdir(parents=True)
    X = R.X[:, gidx].T.tocoo()
    X.data = np.round(X.data).astype(np.int64)
    sio.mmwrite(out / "counts.mtx", X, field="integer")
    (out / "genes.txt").write_text("\n".join(g) + "\n")
    pd.DataFrame({"cell": [f"r{i}" for i in range(R.X.shape[0])], "type": R.labels[level],
                  "donor": R.groups}).to_csv(out / "cells.csv", index=False)
    print(f"reference: {R.X.shape[0]:,} cells, {len(g)} genes ({len(missing)} panel genes missing); types "
          f"{pd.Series(R.labels[level]).value_counts().to_dict()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--samples", required=True, help="comma list of substrings of section names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="", help="folder of <sample>_transcripts.parquet(.gz); none = no molecules")
    ap.add_argument("--ref", default="")
    ap.add_argument("--ref-level-col", default="Level2")
    ap.add_argument("--ref-merge", default=r"^Tu_L\d=>Tumour")
    ap.add_argument("--ref-donor", default="donor_id")
    ap.add_argument("--level", default="L2m")
    ap.add_argument("--max-cells", type=int, default=30000)
    ap.add_argument("--tmp", default="")
    a = ap.parse_args()
    import anndata as ad
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.tmp).expanduser() if a.tmp else out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    keys = [k for k in a.samples.split(",") if k]
    files = [f for f in sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz")) if any(k in f.name for k in keys)]
    if not files:
        raise SystemExit(f"no labels files matching {keys}")
    all_genes = set()
    for f in files:
        sample = f.name.split(".labels")[0]
        dst = out / sample
        if dst.exists():
            raise SystemExit(f"{dst} exists; not overwritten")
        L = pd.read_csv(f, keep_default_na=False, low_memory=False,
                        usecols=lambda c: c in ("cell_id", "x", "y", f"{a.level}_whole", f"{a.level}_nucleus"))
        L["cell_id"] = L.cell_id.astype(str)
        A = ad.read_h5ad(Path(a.h5ad_dir).expanduser() / f"{sample}.h5ad")
        A.obs_names = A.obs_names.astype(str)
        L = L.set_index("cell_id").reindex(A.obs_names)
        m, box = window(A.obs[["x", "y"]].to_numpy(float), a.max_cells)
        B = A[m]
        dst.mkdir(parents=True)
        X = sp.csr_matrix(B.X).T.tocoo()
        X.data = np.round(X.data).astype(np.int64)
        sio.mmwrite(dst / "counts.mtx", X, field="integer")
        (dst / "genes.txt").write_text("\n".join(B.var_names.astype(str)) + "\n")
        C = pd.DataFrame({"cell_id": B.obs_names, "x": B.obs.x.to_numpy(), "y": B.obs.y.to_numpy(),
                          "cell_area": B.obs.get("cell_area", np.nan), "nucleus_area": B.obs.get("nucleus_area", np.nan),
                          f"{a.level}_whole": L.loc[B.obs_names, f"{a.level}_whole"].fillna("").to_numpy(),
                          f"{a.level}_nucleus": L.loc[B.obs_names, f"{a.level}_nucleus"].fillna("").to_numpy()})
        C.to_csv(dst / "cells.csv", index=False)
        info = dict(sample=sample, cells_section=int(A.n_obs), cells_window=int(B.n_obs), **box)
        all_genes |= set(B.var_names.astype(str))
        if a.raw:
            M = molecules(a.raw, sample, C.cell_id, dict(zip(C.cell_id, C[f"{a.level}_whole"])), B.var_names.astype(str),
                          tmp)
            per = M.groupby("cell").size().reindex(C.cell_id).fillna(0).to_numpy()
            tot = np.asarray(B.X.sum(1)).ravel()
            info["molecules"] = int(len(M))
            info["share_cells_molecules_equal_counts"] = float(np.mean(per == tot))
            info["molecules_unlabelled_cells"] = int((M.celltype == "").sum())
            M.to_csv(dst / "molecules.csv.gz", index=False, compression="gzip")
        (dst / "window.json").write_text(json.dumps(info, indent=1))
        print(f"{sample}: {B.n_obs:,} of {A.n_obs:,} cells in window (half-width {box['half_width']:.0f} um)"
              + (f"; {info['molecules']:,} molecules, per-cell totals equal to counts for "
                 f"{info['share_cells_molecules_equal_counts']:.4f} of cells" if a.raw else ""), flush=True)
    if a.ref:
        dst = out / "ref"
        if dst.exists():
            print(f"{dst} exists; reference not re-exported", flush=True)
        else:
            export_ref(a.ref, all_genes, dst, a.level, a.ref_level_col, a.ref_merge, a.ref_donor)
    if not a.tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
