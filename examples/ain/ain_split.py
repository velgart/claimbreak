#!/usr/bin/env python
"""ain_split.py -- split the converted Xenium regions of Baker et al. 2026 (acute interstitial nephritis, GSE336890)
into one compartment .h5ad per patient and tissue region, using the authors' per-cell metadata.

Each Xenium region (slide) holds sections of several patients. The authors' Seurat object (Zenodo 21793755,
xenium_AIN_for_deposit.rds; metadata exported with rds_meta.R to meta.csv.gz) names cells Region_<k>_<cell_id> and
gives base_patient_id, patient_category (AIN / ATI / REF), tissue_region (Cortex / Medulla) and their labels. Cells
not in the authors' object (their QC) are dropped and counted.

Output per piece <GSM>_<RegionNN>_<patient>_<tissue>.h5ad (same layout as compartment_to_h5ad.py):
  obs gains donor = unit = patient, section = piece, tissue = kidney_<tissue>, category, and the authors' columns
  author_label (final_annotations2), author_cd4, author_cd8, author_tls (in_cleaned_tls), author_tls_id,
  author_aggregate_class, author_segmentation (segmentation_method).
With --boundaries-raw, each piece also gets <piece>_cell_boundaries.parquet (that region's boundaries restricted to the
piece's cells) in --boundaries-out, for `claimbreak compartment contacts --contact boundary`.

  python ain_split.py --h5ad-dir ~/claimbreak_data/AIN/h5ad_regions --meta ~/Downloads/ain_meta/meta.csv.gz \
      --out ~/claimbreak_data/AIN/h5ad --boundaries-raw ~/claimbreak_data/AIN/raw \
      --boundaries-out ~/claimbreak_data/AIN/boundaries
"""
import argparse
import gzip
import io
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

AUTHOR = {"final_annotations2": "author_label", "final_cd4": "author_cd4", "final_cd8": "author_cd8",
          "in_cleaned_tls": "author_tls", "tls_id_corrected": "author_tls_id", "aggregate_class": "author_aggregate_class",
          "segmentation_method": "author_segmentation", "lineage": "author_lineage"}


def read_parquet_any(p):
    p = Path(p)
    if p.suffix == ".gz":
        with gzip.open(p, "rb") as fh:
            return pd.read_parquet(io.BytesIO(fh.read()))
    return pd.read_parquet(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad-dir", required=True, help="compartment_to_h5ad.py output for the regions")
    ap.add_argument("--meta", required=True, help="authors' per-cell metadata (rds_meta.R meta.csv.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boundaries-raw", default="", help="folder with <GSM>_<Region>_cell_boundaries.parquet(.gz)")
    ap.add_argument("--boundaries-out", default="")
    ap.add_argument("--min-cells", type=int, default=500, help="pieces with fewer cells are not written")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(a.meta, low_memory=False)
    m = M["cell"].astype(str).str.extract(r"^Region_(\d+)_(.+)$")
    M["region"], M["cell_id"] = pd.to_numeric(m[0]), m[1]
    print(f"authors' metadata: {len(M):,} cells, regions {sorted(M.region.dropna().unique().astype(int))}, "
          f"{M.base_patient_id.nunique()} patients", flush=True)
    bout = Path(a.boundaries_out).expanduser() if a.boundaries_out else None
    if bout:
        bout.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.h5ad_dir).expanduser().glob("*.h5ad")):
        mm = re.search(r"(GSM\d+)_Region(\d+)", f.name)
        if not mm:
            print(f"skip {f.name}: no GSM_RegionNN in the name", flush=True)
            continue
        gsm, reg = mm.group(1), int(mm.group(2))
        A = ad.read_h5ad(f)
        Mr = M[M.region == reg].set_index("cell_id")
        ids = A.obs_names.astype(str)
        keep = ids.isin(Mr.index)
        print(f"{f.name}: {A.n_obs:,} cells, {keep.sum():,} in the authors' object ({keep.mean():.3f}); "
              f"authors' cells of region {reg} not found here: {(~Mr.index.isin(ids)).sum():,}", flush=True)
        A = A[keep].copy()
        mr = Mr.loc[A.obs_names.astype(str)]
        A.obs["donor"] = A.obs["unit"] = mr["base_patient_id"].astype(str).to_numpy()
        A.obs["category"] = mr["patient_category"].astype(str).to_numpy()
        A.obs["tissue"] = ("kidney_" + mr["tissue_region"].astype(str).str.lower()).to_numpy()
        for src, dst in AUTHOR.items():
            if src in mr:
                A.obs[dst] = mr[src].astype(str).to_numpy()
        bnd = None
        if a.boundaries_raw and bout:
            hits = sorted(Path(a.boundaries_raw).expanduser().glob(f"{gsm}_Region{mm.group(2)}_cell_boundaries.parquet*"))
            if hits:
                bnd = read_parquet_any(hits[0])
                bnd["cell_id"] = bnd["cell_id"].astype(str)
            else:
                print(f"  no cell_boundaries for {gsm} region {reg}", flush=True)
        for (pat, tis), g in A.obs.groupby(["donor", "tissue"], observed=True):
            if len(g) < a.min_cells:
                print(f"  {pat} {tis}: {len(g)} cells, skipped", flush=True)
                continue
            piece = f"{gsm}_Region{mm.group(2)}_{pat}_{tis.split('_', 1)[1]}"
            dst = out / f"{piece}.h5ad"
            if dst.exists():
                print(f"  {dst.name} exists; skipped", flush=True)
                continue
            P = A[g.index].copy()
            P.obs["sample"] = P.obs["section"] = piece
            P.obs["panel"] = "Xenium Prime 5K + 100 kidney"
            P.uns["sample"] = piece
            P.write_h5ad(dst, compression="gzip")
            line = f"  wrote {dst.name}: {P.n_obs:,} cells, {g['category'].iloc[0]}"
            if bnd is not None:
                b = bnd[bnd.cell_id.isin(P.obs_names.astype(str))]
                b.to_parquet(bout / f"{piece}_cell_boundaries.parquet")
                line += f", {b.cell_id.nunique():,} boundaries"
            print(line, flush=True)


if __name__ == "__main__":
    main()
