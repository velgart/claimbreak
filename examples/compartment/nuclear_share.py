#!/usr/bin/env python
"""nuclear_share.py --: is a gene a cell's own, or its neighbors'?

A gene a cell makes sits in its nucleus as often as the cell's own marker genes do; a gene picked up from neighbors
(segmentation spillover) sits outside the nucleus. For a gene g and a cell group c, pooled per unit (patient):
  f(g, c)   = nuclear counts of g in c / whole-cell counts of g in c
  f_own(c)  = the same for the group's own marker genes (--markers)
  R(g, c)   = f(g, c) / f_own(c)          about 1: the cell's own gene;  well below 1: neighbors' RNA
and the share of g that a group contributes, from whole-cell counts and corrected by R:
  share_whole(c)  = W(g,c) / sum_c' W(g,c')
  share_nuc(c)    = W(g,c) R(g,c) / sum_c' W(g,c') R(g,c')
Exposure: counts of g per cell of the group against the number of source cells (nuclear counts of g > 0, of another
group) among its k nearest neighbors within r um; a gene drawn from neighbors rises with them in the outside counts
and not in the nuclear counts. Pairs (optional, --pair-genes L,R): sender/receiver pairs within --pair-r um whose
sender has L and receiver has R, from whole-cell counts and from nuclear counts; share of pairs between two groups.
Niche (optional, --niche-col or --niche-type): the same in and out of a niche.

Claims R2 (Baker et al. 2026, AIN: CXCL9 in T cells) and R3 (He et al. 2022, CosMx lung: SPP1 in
macrophages).

  python nuclear_share.py --h5ad-dir DIR --label-obs author_label --groups "T=T_cell_CD8,T_cell_CD4;MAC=Macrophage" \
      --gene CXCL9 --markers "T=CD3E,CD3D,CD2,CD8A;MAC=CD68,C1QA,C1QB,CD163" --unit-col unit --cond-col category \
      --pair-genes CXCL9,CXCR3 --niche-col author_tls --out OUT
Labels can instead come from claimbreak labels files: --labels DIR --arm L2m_nucleus (joined on cell_id).
Inputs are read only. Writes nuclear_fraction.csv, share.csv, exposure.csv, pairs.csv, meta.json into --out.
"""
import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


def parse_map(s):
    out = {}
    for item in [x for x in s.split(";") if "=" in x]:
        k, v = item.split("=", 1)
        out[k.strip()] = [x.strip() for x in v.split(",") if x.strip()]
    return out


def col(M, j):
    x = M[:, j]
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()


def load(f, a, need):
    A = ad.read_h5ad(f)
    var = pd.Index(A.var_names.astype(str))
    genes = [g for g in need if g in var]
    idx = var.get_indexer(genes)
    W = {g: col(A.X, j) for g, j in zip(genes, idx)}
    N = {g: col(A.layers["nucleus"], j) for g, j in zip(genes, idx)}
    obs = A.obs.copy()
    obs.index = obs.index.astype(str)
    if a.label_obs:
        lab = obs[a.label_obs].astype(str).to_numpy()
    else:
        sample = f.name[:-len(".h5ad")]
        L = pd.read_csv(Path(a.labels).expanduser() / f"{sample}.labels.csv.gz", usecols=["cell_id", a.arm],
                        dtype={"cell_id": str}, keep_default_na=False).set_index("cell_id")[a.arm]
        lab = L.reindex(obs.index).fillna("").astype(str).to_numpy()
    return obs, lab, W, N, genes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad-dir", required=True)
    ap.add_argument("--pattern", default="*.h5ad")
    ap.add_argument("--label-obs", default="")
    ap.add_argument("--labels", default=""); ap.add_argument("--arm", default="")
    ap.add_argument("--groups", required=True, help="group=label,label;... (labels as in the label column)")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--markers", required=True, help="group=gene,gene;... the group's own genes")
    ap.add_argument("--unit-col", default="unit"); ap.add_argument("--cond-col", default="")
    ap.add_argument("--keep", default="", help="col=value[,value] filters on obs, ';'-separated")
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--r", type=float, default=30.0)
    ap.add_argument("--pair-genes", default="", help="ligand,receptor")
    ap.add_argument("--pair-r", type=float, default=20.0)
    ap.add_argument("--niche-col", default="", help="obs column, true-like values mark the niche")
    ap.add_argument("--niche-type", default="", help="label: niche = cells with >= --niche-min of this label among --niche-k nearest")
    ap.add_argument("--niche-k", type=int, default=50); ap.add_argument("--niche-min", type=float, default=0.5)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.label_obs and not (a.labels and a.arm):
        raise SystemExit("give --label-obs or --labels with --arm")
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; move it away")
    out.mkdir(parents=True, exist_ok=True)
    G = parse_map(a.groups); MK = parse_map(a.markers)
    lab2grp = {l: g for g, ls in G.items() for l in ls}
    pg = [x for x in a.pair_genes.split(",") if x]
    need = sorted({a.gene, *pg, *[m for ms in MK.values() for m in ms]})
    keep = [x.split("=", 1) for x in a.keep.split(";") if "=" in x]
    files = sorted(Path(a.h5ad_dir).expanduser().glob(a.pattern))
    if not files:
        raise SystemExit("no h5ad files")
    acc, expo, pairs, used = [], [], [], []
    for f in files:
        obs, lab, W, N, genes = load(f, a, need)
        ok = np.ones(len(obs), bool)
        for c, vals in keep:
            ok &= obs[c].astype(str).isin(vals.split(",")).to_numpy()
        if a.gene not in genes or not ok.any():
            continue
        used.append(f.name)
        if len(used) == 1:
            miss = [x for x in need if x not in genes]
            print(f"genes found: {', '.join(genes)}" + (f"; not in the panel: {', '.join(miss)}" if miss else ""), flush=True)
        grp = np.array([lab2grp.get(x, "") for x in lab], dtype=object)
        unit = obs[a.unit_col].astype(str).to_numpy()
        cond = obs[a.cond_col].astype(str).to_numpy() if a.cond_col else np.full(len(obs), "")
        xy = obs[["x", "y"]].to_numpy(float)
        tree = cKDTree(xy)
        # niche
        if a.niche_col:
            niche = obs[a.niche_col].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
        elif a.niche_type:
            _, jn = tree.query(xy, k=min(a.niche_k + 1, len(xy)))
            niche = (lab[jn[:, 1:]] == a.niche_type).mean(1) >= a.niche_min
        else:
            niche = np.zeros(len(obs), bool)
        # per unit x cond x niche x group: whole and nuclear counts of the gene and of the markers
        for gname in G:
            m = ok & (grp == gname)
            if not m.any():
                continue
            own = [x for x in MK.get(gname, []) if x in genes]
            for key in pd.DataFrame({"u": unit[m], "c": cond[m], "n": niche[m]}).drop_duplicates().itertuples(index=False):
                s = m & (unit == key.u) & (cond == key.c) & (niche == key.n)
                row = dict(file=f.name, unit=key.u, cond=key.c, niche=bool(key.n), group=gname, cells=int(s.sum()),
                           W_gene=float(W[a.gene][s].sum()), N_gene=float(N[a.gene][s].sum()),
                           W_own=float(sum(W[g][s].sum() for g in own)), N_own=float(sum(N[g][s].sum() for g in own)),
                           pos_whole=int((W[a.gene][s] > 0).sum()), pos_nuc=int((N[a.gene][s] > 0).sum()))
                acc.append(row)
        # exposure: source cells among k nearest within r (nuclear counts of the gene > 0, another group)
        src = N[a.gene] > 0
        d, j = tree.query(xy, k=a.k + 1, distance_upper_bound=a.r)
        j = j[:, 1:]; valid = j < len(xy)
        jj = np.where(valid, j, 0)
        for gname in G:
            m = ok & (grp == gname)
            if not m.any():
                continue
            other = (grp[jj] != gname) & valid
            nsrc = ((src[jj] & other).sum(1))[m]
            Wg, Ng = W[a.gene][m], N[a.gene][m]
            for b in sorted(set(np.minimum(nsrc, 3))):
                s = np.minimum(nsrc, 3) == b
                for u in np.unique(unit[m][s]):
                    t = s & (unit[m] == u)
                    expo.append(dict(file=f.name, unit=u, cond=cond[m][t][0], group=gname, sources=int(b), cells=int(t.sum()),
                                     whole_per_cell=float(Wg[t].mean()), nuc_per_cell=float(Ng[t].mean()),
                                     cyto_per_cell=float((Wg[t] - Ng[t]).mean())))
        # pairs: sender with ligand, receiver with receptor, within pair_r, sender != receiver
        if len(pg) == 2 and all(g in genes for g in pg):
            Lg, Rg = pg
            P = tree.query_pairs(a.pair_r, output_type="ndarray")
            P = np.vstack([P, P[:, ::-1]])
            P = P[ok[P[:, 0]] & ok[P[:, 1]]]
            for basis, Mx in (("whole", W), ("nucleus", N)):
                hit = (Mx[Lg][P[:, 0]] > 0) & (Mx[Rg][P[:, 1]] > 0)
                Q = pd.DataFrame({"unit": unit[P[hit, 0]], "cond": cond[P[hit, 0]], "send": grp[P[hit, 0]],
                                  "recv": grp[P[hit, 1]], "niche": niche[P[hit, 0]]})
                Q["send"] = Q.send.replace("", "other"); Q["recv"] = Q.recv.replace("", "other")
                for (u, c, sd, rv, nn), g in Q.groupby(["unit", "cond", "send", "recv", "niche"]):
                    pairs.append(dict(file=f.name, unit=u, cond=c, basis=basis, sender=sd, receiver=rv, niche=bool(nn),
                                      pairs=len(g)))
        print(f"{f.name}: {ok.sum():,} cells; groups " +
              ", ".join(f"{g} {(ok & (grp == g)).sum():,}" for g in G), flush=True)
    A = pd.DataFrame(acc)
    if not len(A):
        raise SystemExit("no cells of the groups with the gene")
    U = A.groupby(["unit", "cond", "group"])[["cells", "W_gene", "N_gene", "W_own", "N_own", "pos_whole", "pos_nuc"]].sum().reset_index()
    U["f_gene"] = U.N_gene / U.W_gene.replace(0, np.nan)
    U["f_own"] = U.N_own / U.W_own.replace(0, np.nan)
    U["R"] = U.f_gene / U.f_own
    U.to_csv(out / "nuclear_fraction.csv", index=False)
    S = []
    for (u, c), g in U.groupby(["unit", "cond"]):
        g = g[g.cells >= a.min_cells]
        if len(g) < 2:
            continue
        w = g.W_gene / g.W_gene.sum()
        corr = g.W_gene * g.R.fillna(0)
        n = corr / corr.sum() if corr.sum() > 0 else corr * np.nan
        for (_, r), sw, sn in zip(g.iterrows(), w, n):
            S.append(dict(unit=u, cond=c, group=r.group, share_whole=float(sw), share_nuc=float(sn), R=float(r.R)))
    S = pd.DataFrame(S); S.to_csv(out / "share.csv", index=False)
    E = pd.DataFrame(expo); E.to_csv(out / "exposure.csv", index=False)
    Pq = pd.DataFrame(pairs); Pq.to_csv(out / "pairs.csv", index=False)
    An = A.groupby(["cond", "niche", "group"])[["cells", "W_gene", "N_gene", "pos_whole", "pos_nuc"]].sum().reset_index()
    An.to_csv(out / "niche.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(vars(a), files=used), indent=1, default=str))
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(f"\nnuclear fraction of {a.gene} relative to the group's own genes (R; ~1 own, << 1 neighbours'), "
              "median over units:")
        print(U[U.cells >= a.min_cells].groupby(["cond", "group"])[["f_gene", "f_own", "R"]].median().round(3).to_string())
        if len(S):
            print(f"\nshare of {a.gene} by group, whole-cell vs nuclear-corrected, median over units:")
            print(S.groupby(["cond", "group"])[["share_whole", "share_nuc", "R"]].median().round(3).to_string())
            for c, g in S.groupby("cond"):
                for gname in G:
                    q = g[g.group == gname]
                    if len(q):
                        print(f"  {c} {gname}: units with share_nuc <= 0.5 x share_whole: "
                              f"{int((q.share_nuc <= 0.5 * q.share_whole).sum())} of {len(q)}")
        if len(E):
            E2 = E.assign(w=E.whole_per_cell * E.cells, n=E.nuc_per_cell * E.cells, o=E.cyto_per_cell * E.cells)
            q = E2.groupby(["cond", "group", "sources"])[["cells", "w", "n", "o"]].sum()
            q = pd.DataFrame({"cells": q.cells, "whole": q.w / q.cells, "nucleus": q.n / q.cells, "outside": q.o / q.cells})
            print(f"\n{a.gene} per cell by source neighbours (nuclear {a.gene} > 0, other group; 3 = 3 or more):")
            print(q.round(4).to_string())
        if len(Pq):
            T = Pq.groupby(["cond", "basis", "sender", "receiver"]).pairs.sum().reset_index()
            T["share"] = T.pairs / T.groupby(["cond", "basis"]).pairs.transform("sum")
            print(f"\npairs {pg[0]} sender -> {pg[1]} receiver within {a.pair_r} um, share by sender and receiver:")
            print(T.sort_values(["cond", "basis", "share"], ascending=[True, True, False]).groupby(["cond", "basis"]).head(6)
                  .round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
