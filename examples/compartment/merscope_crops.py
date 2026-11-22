#!/usr/bin/env python
"""merscope_crops.py -- test crops of a MERSCOPE DAPI plane for tuning nucleus segmentation. A full StarDist run takes hours; the crops let merscope_nuclei.py be tuned on the hard cases first.

Reads (never changes) the DAPI plane and the cell metadata; optionally a nucleus table of an earlier run. The plane is
divided into squares of --size px (full resolution); from a strided sample of each square (every 16th pixel) it
takes the share of unimaged pixels (0) and the median DAPI of imaged pixels; from the metadata the number of cells.
Squares with at least --min-cells cells and at most half unimaged are candidates; picked, without repeats:
  densest      most cells
  typical      cell count nearest the median
  bright       highest median DAPI
  unimaged     most cells among squares with 2-50% unimaged pixels (an unimaged area cutting into tissue)
  sparse       fewest cells (stroma, vessels)
  no_nucleus   highest share of cells without an owned nucleus in --nuclei-table (an earlier run), if given
Writes OUT/<name>_r<row>_c<col>.tif (uint16, zlib), OUT/<name>_r<row>_c<col>.cells.csv (EntityID, center_x,
center_y in um, shifted to the crop: the crop is a small mosaic with the same pixel size) and OUT/crops.json.

  python merscope_crops.py --img .../images/<prefix>_images_mosaic_DAPI_z3.tif --meta .../<prefix>_cell_metadata.csv \
      --out .../merscope_uterine_cd45ro/crops3000 [--nuclei-table .../nuclei_global_1776/MS_UTERINE_RO.nuclei.csv.gz]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=3000)
    ap.add_argument("--px", type=float, default=9.2594)
    ap.add_argument("--min-cells", type=int, default=150)
    ap.add_argument("--nuclei-table", default="")
    a = ap.parse_args()
    import tifffile
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    img = tifffile.memmap(str(Path(a.img).expanduser()), mode="r")
    H, W = img.shape
    S = a.size
    M = pd.read_csv(Path(a.meta).expanduser(), usecols=["EntityID", "center_x", "center_y"])
    r, c = (M.center_y.to_numpy() * a.px).astype(int), (M.center_x.to_numpy() * a.px).astype(int)
    ny, nx = H // S, W // S
    gi, gj = np.clip(r // S, 0, ny - 1), np.clip(c // S, 0, nx - 1)
    n = np.zeros((ny, nx), int)
    np.add.at(n, (gi, gj), 1)
    nonuc = np.full((ny, nx), np.nan)
    if a.nuclei_table:
        N = pd.read_csv(Path(a.nuclei_table).expanduser(), usecols=["owner"])
        has = np.isin(M.EntityID.to_numpy(), N.owner.to_numpy())
        tot = np.zeros((ny, nx))
        np.add.at(tot, (gi, gj), (~has).astype(float))
        nonuc = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
    zero, med = np.full((ny, nx), np.nan), np.full((ny, nx), np.nan)
    cand = np.argwhere(n >= a.min_cells)
    print(f"{ny} x {nx} squares of {S} px; {len(cand)} with >= {a.min_cells} cells; sampling DAPI ...", flush=True)
    for i, j in cand:
        x = np.asarray(img[i * S:(i + 1) * S:16, j * S:(j + 1) * S:16])
        zero[i, j] = float((x == 0).mean())
        v = x[x > 0]
        med[i, j] = float(np.median(v)) if v.size else np.nan
    picks, used = {}, set()

    def take(name, score, mask, best=max):
        ok = [tuple(x) for x in cand if mask[tuple(x)] and tuple(x) not in used and np.isfinite(score[tuple(x)])]
        if not ok:
            print(f"  {name}: no candidate", flush=True)
            return
        k = best(ok, key=lambda x: score[x])
        picks[name] = k
        used.add(k)

    allm = np.nan_to_num(zero, nan=1.0) <= 0.5          # mostly imaged squares only
    take("densest", n.astype(float), allm)
    typ = np.median(n[n >= a.min_cells])
    take("typical", -np.abs(n - typ).astype(float), allm)
    take("bright", med, allm)
    take("unimaged", n.astype(float), allm & (zero >= 0.02))
    take("sparse", -n.astype(float), allm)
    if a.nuclei_table:
        take("no_nucleus", np.where(n >= 300, nonuc, np.nan), allm)
    rows = []
    for name, (i, j) in picks.items():
        r0, c0 = i * S, j * S
        x = np.asarray(img[r0:r0 + S, c0:c0 + S])
        stem = f"{name}_r{r0}_c{c0}"
        tifffile.imwrite(out / f"{stem}.tif", x, compression="zlib")
        sel = (r >= r0) & (r < r0 + S) & (c >= c0) & (c < c0 + S)
        C = M[sel].copy()
        C["center_x"] -= c0 / a.px
        C["center_y"] -= r0 / a.px
        C.to_csv(out / f"{stem}.cells.csv", index=False)
        row = dict(name=name, row=int(r0), col=int(c0), size=S, cells=int(sel.sum()), unimaged_share=zero[i, j],
                   dapi_median=med[i, j], no_nucleus_share=None if np.isnan(nonuc[i, j]) else float(nonuc[i, j]))
        rows.append(row)
        print(f"  {name:10s} r{r0} c{c0}: {row['cells']} cells, unimaged {zero[i, j]:.3f}, DAPI median "
              f"{med[i, j]:.0f}" + (f", without nucleus {nonuc[i, j]:.2f}" if np.isfinite(nonuc[i, j]) else ""), flush=True)
    (out / "crops.json").write_text(json.dumps(dict(img=str(a.img), px=a.px, size=S, crops=rows), indent=1))
    print(f"wrote {len(rows)} crops to {out}")


if __name__ == "__main__":
    main()
