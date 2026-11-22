#!/usr/bin/env python
"""merscope_owners.py -- which cell owns each StarDist nucleus, by the platform's cell outlines.
merscope_nuclei.py gives each nucleus to the cell with the nearest center within 8 um; in packed
tissue that can be a neighbor, and where the platform drew no cell it still names one. Here a nucleus belongs to the
cell whose outline in the DAPI plane (ZIndex --z, default 3) contains the nucleus centroid, and to no cell otherwise
(as Xenium's nuclei lie inside their cells by construction).

Reads (never changes) <prefix>_cell_boundaries.parquet (EntityID, ZIndex, Geometry as WKB, um) and the nucleus table
<sample>.nuclei.csv.gz; writes <sample>.owners.csv.gz next to it (nucleus, owner, owner_center: the nearest-center owner of
merscope_nuclei.py) and <sample>.owners.json (agreement of the two rules). merscope_to_h5ad.py --nuclei uses the owners file when it
is present.

  python merscope_owners.py --boundaries .../download/<prefix>_cell_boundaries.parquet --nuclei .../nuclei
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boundaries", required=True)
    ap.add_argument("--nuclei", required=True, help="folder with <sample>.nuclei.csv.gz")
    ap.add_argument("--z", type=int, default=3)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    import shapely
    nd = Path(a.nuclei).expanduser()
    f_nt = next(iter(sorted(nd.glob("*.nuclei.csv.gz"))), None)
    if f_nt is None:
        sys.exit(f"no *.nuclei.csv.gz in {nd}")
    sample = f_nt.name.replace(".nuclei.csv.gz", "")
    dst = nd / f"{sample}.owners.csv.gz"
    if dst.exists():
        sys.exit(f"{dst} exists; not overwritten")
    t0 = time.time()
    B = pq.read_table(Path(a.boundaries).expanduser(), columns=["EntityID", "ZIndex", "Geometry"],
                      filters=[("ZIndex", "=", a.z)]).to_pandas()
    if B.empty:
        sys.exit(f"no outlines with ZIndex {a.z}")
    geo = shapely.from_wkb(B.Geometry.to_numpy())
    ok = ~shapely.is_empty(geo) & shapely.is_valid(geo)
    if (~ok).any():
        geo[~ok] = shapely.make_valid(geo[~ok])
    print(f"{len(B):,} cell outlines at ZIndex {a.z} ({int((~ok).sum()):,} repaired), {time.time() - t0:.0f} s", flush=True)
    N = pd.read_csv(f_nt)
    pts = shapely.points(N.x.to_numpy(), N.y.to_numpy())
    tree = shapely.STRtree(geo)
    pi, gi = tree.query(pts, predicate="within")
    owner = np.full(len(N), -1, np.int64)
    # a centroid inside two outlines (rare: outlines overlap at their edges) goes to the outline with the nearest center
    cen = shapely.centroid(geo)
    d = shapely.distance(pts[pi], cen[gi])
    order = np.lexsort((d, pi))
    pi, gi = pi[order], gi[order]
    first = np.unique(pi, return_index=True)[1]
    owner[pi[first]] = B.EntityID.to_numpy()[gi[first]]
    multi = int(len(pi) - len(first))
    O = pd.DataFrame({"nucleus": N.nucleus, "owner": owner, "owner_centre": N.owner})
    O.to_csv(dst, index=False)
    inside = owner >= 0
    same = inside & (owner == N.owner.to_numpy())
    per = pd.Series(owner[inside]).value_counts()
    summ = dict(sample=sample, z=a.z, outlines=int(len(B)), nuclei=int(len(N)), inside_a_cell=int(inside.sum()),
                inside_share=float(inside.mean()), in_two_outlines=multi,
                same_owner_as_centre_rule=float(same.sum() / max(inside.sum(), 1)),
                centre_rule_owned_but_outside_any_cell=int(((N.owner >= 0).to_numpy() & ~inside).sum()),
                cells_with_nucleus=int(per.size), cells_with_2plus=int((per > 1).sum()),
                seconds=round(time.time() - t0))
    (nd / f"{sample}.owners.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
