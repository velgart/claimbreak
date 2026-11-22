#!/usr/bin/env python
"""ra_to_compartment.py -- RA synovium samples (xenium_to_h5ad.py output) -> compartment .h5ad files for
`claimbreak compartment label` and the rest of the chain.

Input: SAMPLES/<sample>.h5ad from examples/xenium/xenium_to_h5ad.py (Bhamidipati et al. 2026; X raw counts of all
cells, layers['nuclear'] = nucleus-overlapping transcripts, obs sample, patient, timepoint, x, y,
segmentation_method, cell_area, nucleus_area, passes_filter). The inputs are only read.
Output per sample OUT/<sample>.h5ad, the layout of compartment_to_h5ad.py:
  X                  whole-cell counts (float32), cells passing the authors' filter (> 50 genes) unless --all-cells
  layers['nucleus']  layers['nuclear'];  layers['cyto'] = X - nucleus (checked >= 0)
  obs                sample, donor = unit = patient, timepoint, panel ("5K+100"), section = sample, tissue
                     ("synovium"), x, y, cell_area, nucleus_area, segmentation_method, n_whole, n_nucleus
Refuses to overwrite. Prints one line per sample (cells, median counts whole / nucleus, segmentation shares).

  python ra_to_compartment.py --samples ~/claimbreak_data/data/RA_h5ad/samples \
      --out ~/claimbreak_data/data/RA_h5ad/compartment [--timepoint pre]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def convert(src: Path, dst: Path, all_cells: bool = False) -> dict:
    import anndata as ad
    A = ad.read_h5ad(src)
    if "nuclear" not in A.layers:
        raise ValueError(f"{src.name}: no layers['nuclear']")
    keep = np.ones(A.n_obs, bool)
    if not all_cells and "passes_filter" in A.obs:
        keep = A.obs["passes_filter"].astype(bool).to_numpy()
    A = A[keep].copy()
    X = sp.csr_matrix(A.X).astype(np.float32)
    N = sp.csr_matrix(A.layers["nuclear"]).astype(np.float32)
    C = (X - N).tocsr()
    C.eliminate_zeros()
    if C.nnz and C.data.min() < 0:
        raise ValueError(f"{src.name}: nuclear counts exceed whole-cell counts in {int((C.data < 0).sum())} entries")
    o = A.obs
    sample = str(o["sample"].iloc[0]) if "sample" in o else src.stem
    patient = o["patient"].astype(str).to_numpy() if "patient" in o else np.full(A.n_obs, sample)
    obs = pd.DataFrame(index=pd.Index(A.obs_names.astype(str), name="cell_id"))
    obs["sample"] = sample
    obs["donor"] = patient
    obs["unit"] = patient
    obs["timepoint"] = o["timepoint"].astype(str).to_numpy() if "timepoint" in o else ""
    obs["panel"] = "5K+100"
    obs["section"] = sample
    obs["tissue"] = "synovium"
    for c in ("x", "y", "cell_area", "nucleus_area"):
        obs[c] = o[c].to_numpy(float) if c in o else np.nan
    obs["segmentation_method"] = o["segmentation_method"].astype(str).to_numpy() if "segmentation_method" in o else ""
    obs["n_whole"] = np.asarray(X.sum(1)).ravel()
    obs["n_nucleus"] = np.asarray(N.sum(1)).ravel()
    for c in ("sample", "donor", "unit", "timepoint", "panel", "section", "tissue", "segmentation_method"):
        obs[c] = obs[c].astype("category")
    B = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(A.var_names.astype(str), name="gene")))
    B.layers["nucleus"] = N
    B.layers["cyto"] = C
    B.uns["source"] = str({"from": str(src), "all_cells": all_cells})
    dst.parent.mkdir(parents=True, exist_ok=True)
    B.write_h5ad(dst, compression="gzip")
    seg = obs["segmentation_method"].value_counts(normalize=True).round(3).to_dict()
    return dict(sample=sample, cells_in=int(len(keep)), cells_out=int(B.n_obs),
                median_whole=float(np.median(obs.n_whole)), median_nucleus=float(np.median(obs.n_nucleus)),
                nucleus_share=float(obs.n_nucleus.sum() / max(obs.n_whole.sum(), 1)), segmentation=seg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="folder of xenium_to_h5ad.py sample files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timepoint", default="pre", help="pre, post or all (file names <patient>_<timepoint>.h5ad)")
    ap.add_argument("--all-cells", action="store_true", help="keep cells failing the authors' > 50 genes filter")
    a = ap.parse_args()
    src, out = Path(a.samples).expanduser().resolve(), Path(a.out).expanduser().resolve()
    if out == src or src in out.parents:
        sys.exit("--out must be outside the input folder")
    files = sorted(src.glob("*.h5ad"))
    if a.timepoint != "all":
        files = [f for f in files if f.stem.endswith("_" + a.timepoint)]
    if not files:
        sys.exit(f"no sample files for timepoint {a.timepoint} in {src}")
    rows = []
    for f in files:
        dst = out / f.name
        if dst.exists():
            print(f"{dst.name} exists; skipped", flush=True)
            continue
        r = convert(f, dst, a.all_cells)
        rows.append(r)
        print(f"{r['sample']}: {r['cells_out']:,} of {r['cells_in']:,} cells; median counts whole {r['median_whole']:.0f}, "
              f"nucleus {r['median_nucleus']:.0f} (share {r['nucleus_share']:.2f}); segmentation {r['segmentation']}",
              flush=True)
    if rows:
        S = pd.DataFrame(rows)
        f = out / "conversion.csv"
        S.to_csv(f, mode="a", header=not f.exists(), index=False)
        print(f"wrote {len(rows)} files to {out}")


if __name__ == "__main__":
    main()
