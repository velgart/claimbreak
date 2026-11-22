#!/usr/bin/env python
"""region_loss.py -- where in the tissue the immune cells lose their labels (gut: intraepithelial against lamina propria T cells).

Question: in gut, is the T/NK label loss of whole-cell counts concentrated in T cells that sit inside the epithelium
(intraepithelial lymphocytes) rather than in the lamina propria? The same code places T cells in tumor nests against
stroma (lung), or in any partner-rich region against the rest.

For each cell labeled as the immune type by nuclear half A:
  region      the share of partner cells among all cells within --radius um (default 25), by the half-A labels of the
              OTHER cells. A cell inside an epithelial layer has mostly epithelial cells around it; one in the lamina
              propria has few. Half A of the neighbors is independent of the focal cell's half B and outside counts.
              Bins (--bins, default 0,0.2,0.5,1): e.g. "lamina propria" (< 0.2), "border" (0.2-0.5), "inside" (>= 0.5).
  touch       whether the cell touches >= 1 partner cell (6 nearest within 30 um, half-A labels of the neighbors).
  kept        whether half B (null) and whole-cell counts at the depth of half B (wholeB) keep the immune label; the kept
              ratio of the bin is kept(wholeB) / kept(half B), as Table 1 but per region.
Per unit (patient or donor) and pooled over units; with the mask type of each section (stain or nucleus expansion,
from the section name, --stain-pattern) so that tissue and segmentation can be told apart.

  python region_loss.py --labels DIR --out OUT --level L1 --immune T_NK --partner Epithelial --unit-pattern 'I0\\d{3}'

Checks of the "inside" calls and of which cells are lost:
  identity   per bin, for the immune calls of half A and, for comparison, for the partner cells: nuclear and whole-cell
             counts, the half-A margin and how often the half-A runner-up label is the partner (labels file), and with
             --h5ad DIR (<section>.h5ad from compartment_to_h5ad.py) cell and nucleus areas and the counts of
             --own-markers and --partner-markers per 1,000 nuclear counts. Immune calls inside the partner region
             that look like partner cells (size, partner markers in the nucleus) would be partner nuclei carrying the
             immune cells' RNA, for example from a cell above or below in the section, and their "loss" a correction.
  state      with --h5ad and --state-genes 'name=G1,G2;...': kept ratio of the immune calls by whether the nucleus
             holds >= 1 count of each gene set, and by quartile of nuclear counts (RNA content), per region bin.
  --partner nonimmune   every label that is not immune (immune labels: T, NK, B, plasma, myeloid, mast, ...).

Depth-matched state (depth.csv): "positive" (>= 1 nuclear count of a gene set) is more likely in a nucleus
with more counts, so the state comparison above mixes state with RNA depth. Here the immune calls of each unit are
cut into --depth-strata strata of nuclear counts (equal sizes, default 5); in each stratum the loss of positive and
negative cells is compared, and the differences are averaged with Mantel-Haenszel weights n_pos n_neg / n (strata
with >= 5 cells on each side). Per unit and region bin ('all' too): crude and depth-matched difference in loss
(positive minus negative, points); median over units, units with a positive difference, and a pooled estimate over
all unit-strata. The row feature 'inside_vs_away' does the same for the region contrast itself (last bin against the
first): is the region effect a depth effect?

Writes cells_by_bin.csv (unit, section, masks, bin: cells, kept_B, kept_W, kept_ratio, touch share, mean region),
units.csv (unit, bin: pooled over the unit's sections), summary.csv (bin, masks: median over units, units) and
meta.json. Refuses to overwrite.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from claimbreak.contacts import knn_edges

IMMUNE_RX = r"(?i)(^|[^a-z])(t|nk|tnk|t_nk|b|plasma|myeloid|macro|mono|mast|immune|lymph|dc|neutro)([^a-z]|$)"
from claimbreak.placement import unit_from


def _cb_version():
    try:
        import claimbreak
        return claimbreak.__version__
    except ImportError:
        return None


def region_share(xy, is_part, radius, idx=None):
    """share of partner cells among the other cells within radius (nan when there is none), for the cells idx"""
    idx = np.arange(len(xy)) if idx is None else idx
    q = xy[idx]
    n_all = cKDTree(xy).query_ball_point(q, r=radius, return_length=True) - 1
    n_par = (cKDTree(xy[is_part]).query_ball_point(q, r=radius, return_length=True) - is_part[idx]
             if is_part.any() else np.zeros(len(idx)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_all > 0, n_par / n_all, np.nan)


def touches(xy, is_part, k=6, r=30.0):
    E = knn_edges(xy, k, r)
    hit = np.zeros(len(xy), bool)
    hit[E[:, 0][is_part[E[:, 1]]]] = True
    hit[E[:, 1][is_part[E[:, 0]]]] = True
    return hit


def _h5_str(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x])


def read_h5(path, genes_wanted):
    """cell ids, obs areas, nuclear totals and nuclear counts per gene set from a compartment_to_h5ad.py file, read
    with h5py (only the needed parts). genes_wanted: {set name: [genes]}. Returns (ids, obs DataFrame, {set: counts},
    {set: genes found})."""
    import h5py
    try:
        from anndata.io import read_elem
    except ImportError:                     # anndata < 0.11
        from anndata.experimental import read_elem
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        ids = obs.index.astype(str).to_numpy()
        O = pd.DataFrame(index=ids)
        for c in ("cell_area", "nucleus_area"):
            if c in obs:
                O[c] = pd.to_numeric(obs[c], errors="coerce").to_numpy()
        genes = read_elem(f["var"]).index.astype(str).to_numpy()
        L = f["layers"]["nucleus"]
        if isinstance(L, h5py.Dataset):
            M = L[:]
            tot = M.sum(1)
            sums = {k: M[:, np.isin(genes, v)].sum(1) for k, v in genes_wanted.items()}
        else:
            enc = L.attrs.get("encoding-type", "csr_matrix")
            data, ind, ptr = L["data"][:], L["indices"][:], L["indptr"][:]
            if enc == "csc_matrix":
                col = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr)); row = ind
            else:
                row = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr)); col = ind
            n = len(ids)
            tot = np.bincount(row, weights=data, minlength=n)
            sums = {}
            for k, v in genes_wanted.items():
                m = np.isin(col, np.flatnonzero(np.isin(genes, v)))
                sums[k] = np.bincount(row[m], weights=data[m], minlength=n)
    O["n_nuc_h5"] = tot
    found = {k: sorted(set(v) & set(genes)) for k, v in genes_wanted.items()}
    return ids, O, sums, found


def section_table(D, lv, immune, partners, radius, bins, labels_bins, h5=None):
    A = D[f"{lv}_nucA"].astype(str).to_numpy()
    B = D[f"{lv}_nucB"].astype(str).to_numpy()
    W = D[f"{lv}_wholeB"].astype(str).to_numpy()
    xy = D[["x", "y"]].to_numpy(float)
    part = np.isin(A, partners)
    imm = np.flatnonzero(A == immune)
    reg = np.full(len(xy), np.nan)
    reg[imm] = region_share(xy, part, radius, imm)
    tch = touches(xy, part)
    focal = (A == immune) & np.isfinite(reg)
    b = pd.cut(reg[focal], bins, labels=labels_bins, include_lowest=True, right=False)
    T = pd.DataFrame({"bin": b.astype(str), "region": reg[focal], "touch": tch[focal],
                      "kept_B": B[focal] == immune, "kept_W": W[focal] == immune,
                      "labelled_B": B[focal] != "", "labelled_W": W[focal] != ""})
    # identity of the immune calls and, for comparison, of the partner cells
    P = pd.DataFrame({"bin": "partner cells", "region": np.nan, "touch": np.nan, "kept_B": np.nan, "kept_W": np.nan,
                      "labelled_B": np.nan, "labelled_W": np.nan}, index=np.flatnonzero(part))
    T.index = np.flatnonzero(focal)
    for c, nm in ((f"n_nucleus", "n_nucleus"), ("n_whole", "n_whole"), (f"{lv}_nucA_margin", "margin_A")):
        if c in D:
            v = pd.to_numeric(D[c], errors="coerce").to_numpy()
            T[nm], P[nm] = v[T.index], v[P.index]
    if f"{lv}_nucA_second" in D:
        sec = D[f"{lv}_nucA_second"].astype(str).to_numpy()
        T["second_is_partner"] = np.isin(sec[T.index], partners)
    if h5 is not None:
        ids, O, sums, _ = h5
        pos = pd.Series(np.arange(len(ids)), index=ids)
        cid = D["cell_id"].astype(str).to_numpy() if "cell_id" in D else None
        if cid is not None:
            k = pos.reindex(cid).to_numpy()
            ok = np.isfinite(k)
            kk = np.where(ok, k, 0).astype(int)
            for frame in (T, P):
                r = frame.index.to_numpy()
                good = ok[r]
                for c in ("cell_area", "nucleus_area"):
                    if c in O:
                        frame[c] = np.where(good, O[c].to_numpy()[kk[r]], np.nan)
                tot = np.where(good, O["n_nuc_h5"].to_numpy()[kk[r]], np.nan)
                for sname, v in sums.items():
                    cnt = np.where(good, v[kk[r]], np.nan)
                    frame[f"cnt_{sname}"] = cnt
                    with np.errstate(invalid="ignore", divide="ignore"):
                        frame[f"per1k_{sname}"] = 1000 * cnt / tot
                frame["n_nuc_h5"] = tot
    return T, P


def summarise(T, keys):
    g = T.groupby(keys, observed=True)
    S = g.agg(cells=("kept_B", "size"), kept_B=("kept_B", "mean"), kept_W=("kept_W", "mean"),
              touch=("touch", "mean"), region=("region", "mean")).reset_index()
    S["kept_ratio"] = S.kept_W / S.kept_B
    S["loss_pct"] = 100 * (1 - S.kept_ratio)
    return S


def identity_table(T, P, min_cells):
    """per unit and bin (and the partner cells), medians of the identity measures; plus a 'median' unit row"""
    X = pd.concat([T, P], ignore_index=True)
    cols = [c for c in X.columns if c in ("n_nucleus", "n_whole", "margin_A", "cell_area", "nucleus_area", "n_nuc_h5")
            or c.startswith("per1k_")]
    if not cols and "second_is_partner" not in X:
        return pd.DataFrame()
    agg = {c: (c, "median") for c in cols}
    agg["cells"] = ("bin", "size")
    if "second_is_partner" in X:
        X["second_is_partner"] = X["second_is_partner"].astype(float)
        agg["second_is_partner"] = ("second_is_partner", "mean")
    U = X.groupby(["unit", "bin"]).agg(**agg).reset_index()
    U = U[U.cells >= min_cells]
    M = U.groupby("bin").median(numeric_only=True).reset_index()
    M["unit"] = "median"
    M["units"] = U.groupby("bin").unit.nunique().reindex(M.bin).to_numpy()
    return pd.concat([U, M], ignore_index=True)


def state_table(T, min_cells):
    """kept ratio of the immune calls by state (>= 1 nuclear count of a gene set) and by quartile of nuclear counts,
    per unit and bin ('all' bins pooled too); plus 'median' rows over units"""
    feats = [c for c in T.columns if c.startswith("cnt_state_")]
    rows = []
    T = T.copy()
    if "n_nuc_h5" in T and T.n_nuc_h5.notna().any():
        T["rna_q"] = T.groupby("unit").n_nuc_h5.transform(
            lambda v: pd.qcut(v.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]) if v.notna().sum() >= 8
            else pd.Series(np.nan, index=v.index)).astype(str)
        feats_q = ["rna_q"]
    else:
        feats_q = []
    for bins_ in ("all", None):
        Tb = T.assign(bin="all") if bins_ == "all" else T
        for fcol in feats + feats_q:
            lev = Tb[fcol].astype(str) if fcol == "rna_q" else np.where(Tb[fcol] >= 1, "positive", "negative")
            G = Tb.assign(level=lev).groupby(["unit", "bin", "level"]).agg(
                cells=("kept_B", "size"), kept_B=("kept_B", "mean"), kept_W=("kept_W", "mean")).reset_index()
            G = G[(G.cells >= min_cells) & (G.level != "nan")]
            G["feature"] = fcol.replace("cnt_state_", "")
            rows.append(G)
    if not rows:
        return pd.DataFrame()
    S = pd.concat(rows, ignore_index=True)
    S["kept_ratio"] = S.kept_W / S.kept_B
    S["loss_pct"] = 100 * (1 - S.kept_ratio)
    M = S.groupby(["feature", "bin", "level"]).agg(units=("unit", "nunique"), cells=("cells", "sum"),
                                                   loss_pct=("loss_pct", "median")).reset_index()
    M["unit"] = "median"
    return pd.concat([S, M], ignore_index=True)


def depth_table(T, names, k=5, min_stratum=5, min_side=10):
    """state (and region) differences in loss within strata of nuclear counts; see the module docstring"""
    if "n_nuc_h5" not in T or not T.n_nuc_h5.notna().any():
        return pd.DataFrame()
    T = T[T.n_nuc_h5.notna()].copy()
    T["dstr"] = T.groupby("unit").n_nuc_h5.transform(
        lambda v: pd.qcut(v.rank(method="first"), k, labels=False) if len(v) >= 2 * k else pd.Series(0, index=v.index))
    jobs = []
    for f in [c for c in T.columns if c.startswith("cnt_state_")]:
        pos = (T[f] >= 1).to_numpy()
        for b in ["all"] + list(names):
            m = np.ones(len(T), bool) if b == "all" else (T.bin == b).to_numpy()
            jobs.append((b, f.replace("cnt_state_", ""), m, pos))
    jobs.append(("all", f"{names[-1]}_vs_{names[0]}", T.bin.isin([names[0], names[-1]]).to_numpy(),
                 (T.bin == names[-1]).to_numpy()))
    rows, pooled = [], {}

    def loss(x):
        return 100 * (1 - x.kept_W.sum() / x.kept_B.sum()) if x.kept_B.sum() > 0 else np.nan
    for b, feat, m, pos in jobs:
        X = T[m].assign(pos=pos[m])
        for u, Xu in X.groupby("unit"):
            num = den = 0.0
            npos = nneg = used = 0
            for _, Xs in Xu.groupby("dstr"):
                P_, N_ = Xs[Xs.pos], Xs[~Xs.pos]
                if len(P_) < min_stratum or len(N_) < min_stratum:
                    continue
                lp, ln = loss(P_), loss(N_)
                if not (np.isfinite(lp) and np.isfinite(ln)):
                    continue
                w = len(P_) * len(N_) / (len(P_) + len(N_))
                num += w * (lp - ln); den += w
                npos += len(P_); nneg += len(N_); used += 1
                acc = pooled.setdefault((b, feat), [0.0, 0.0])
                acc[0] += w * (lp - ln); acc[1] += w
            if npos < min_side or nneg < min_side or den == 0:
                continue
            rows.append(dict(unit=u, bin=b, feature=feat, strata=used, cells_pos=npos, cells_neg=nneg,
                             crude_diff=loss(Xu[Xu.pos]) - loss(Xu[~Xu.pos]), adjusted_diff=num / den))
    if not rows:
        return pd.DataFrame()
    U = pd.DataFrame(rows)
    M = U.groupby(["bin", "feature"]).agg(units=("unit", "nunique"), cells_pos=("cells_pos", "sum"),
                                          cells_neg=("cells_neg", "sum"), crude_diff=("crude_diff", "median"),
                                          adjusted_diff=("adjusted_diff", "median"),
                                          units_positive=("adjusted_diff", lambda v: int((v > 0).sum()))).reset_index()
    M["pooled_adjusted"] = [pooled[(b, f)][0] / pooled[(b, f)][1] if pooled.get((b, f), [0, 0])[1] else np.nan
                            for b, f in zip(M.bin, M.feature)]
    M["unit"] = "median"
    return pd.concat([U, M], ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="L1"); ap.add_argument("--immune", default="T_NK")
    ap.add_argument("--partner", required=True, help="partner label(s), comma separated (e.g. Epithelial)")
    ap.add_argument("--radius", type=float, default=25.0, help="um around each cell for the region share")
    ap.add_argument("--bins", default="0,0.2,0.5,1.0001")
    ap.add_argument("--bin-names", default="away,border,inside")
    ap.add_argument("--unit-pattern", default=r"I0\d{3}")
    ap.add_argument("--stain-pattern", default=r"(withSegment|Colon|5k)",
                    help="regex on the section name marking stain-segmented sections; others count as nucleus "
                         "expansion (as in the paper's numbers)")
    ap.add_argument("--min-cells", type=int, default=20, help="bins with fewer immune cells are left out of summaries")
    ap.add_argument("--h5ad", default=None, help="folder of <section>.h5ad (compartment_to_h5ad.py) for areas and "
                                                 "nuclear marker counts")
    ap.add_argument("--own-markers", default="", help="genes of the immune type, comma separated")
    ap.add_argument("--partner-markers", default="", help="genes of the partner type, comma separated")
    ap.add_argument("--state-genes", default="", help="'name=G1,G2;name2=G3,...': gene sets for the state analysis")
    ap.add_argument("--depth-strata", type=int, default=5, help="strata of nuclear counts for depth.csv")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    files = sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz"))
    if not files:
        raise SystemExit(f"no labels files in {a.labels}")
    bins = [float(x) for x in a.bins.split(",")]
    names = a.bin_names.split(",")
    if len(names) != len(bins) - 1:
        raise SystemExit("--bin-names needs one name per bin")
    partners = [x for x in a.partner.split(",") if x]
    lv = a.level
    sets = {}
    if a.own_markers:
        sets["own"] = [g for g in a.own_markers.split(",") if g]
    if a.partner_markers:
        sets["partner"] = [g for g in a.partner_markers.split(",") if g]
    for part_ in [x for x in a.state_genes.split(";") if x.strip()]:
        k, v = part_.split("=", 1)
        sets[f"state_{k.strip()}"] = [g.strip() for g in v.split(",") if g.strip()]
    found_all, no_h5 = {}, []
    rows, prows, skipped = [], [], []
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns
        need = [f"{lv}_nucA", f"{lv}_nucB", f"{lv}_wholeB", "x", "y"]
        if any(c not in cols for c in need):
            skipped.append(f.name)
            continue
        extra = {"sample", "unit", "donor", "cell_id", "n_nucleus", "n_whole", f"{lv}_nucA_margin", f"{lv}_nucA_second"}
        D = pd.read_csv(f, usecols=lambda c: c in set(need) | extra, keep_default_na=False, low_memory=False)
        if partners == ["nonimmune"]:
            labs = set(D[f"{lv}_nucA"].astype(str)) - {""}
            partners = sorted(x for x in labs if not re.search(IMMUNE_RX, x))
            print(f"partners (non-immune labels): {', '.join(partners)}")
        sample = str(D["sample"].iloc[0]) if "sample" in D else f.name.split(".labels")[0]
        fb = D["unit"].iloc[0] if "unit" in D else (D["donor"].iloc[0] if "donor" in D else None)
        unit = unit_from(sample, a.unit_pattern, fb)
        h5 = None
        if a.h5ad and sets:
            hp = Path(a.h5ad).expanduser() / f"{f.name.split('.labels')[0]}.h5ad"
            if hp.exists():
                h5 = read_h5(hp, sets)
                for k, v in h5[3].items():
                    found_all.setdefault(k, set()).update(v)
            else:
                no_h5.append(hp.name)
        T, P = section_table(D, lv, a.immune, partners, a.radius, bins, names, h5)
        for X in (T, P):
            X["section"], X["unit"] = sample, unit
            X["masks"] = "stain" if re.search(a.stain_pattern, sample, re.I) else "expansion"
        rows.append(T)
        prows.append(P)
        print(f"{sample} ({unit}, {T.masks.iloc[0] if len(T) else '-'}): {len(T):,} {a.immune} cells by half A; "
              + ", ".join(f"{n} {int((T.bin == n).sum())}" for n in names), flush=True)
    if not rows:
        raise SystemExit(f"no labels file has {lv}_nucA, {lv}_nucB, {lv}_wholeB (skipped {len(skipped)})")
    T = pd.concat(rows, ignore_index=True)
    Pc = pd.concat(prows, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    ident = identity_table(T, Pc, a.min_cells)
    ident.to_csv(out / "identity.csv", index=False)
    state = state_table(T, a.min_cells)
    if len(state):
        state.to_csv(out / "state.csv", index=False)
    depth = depth_table(T, names, a.depth_strata)
    if len(depth):
        depth.to_csv(out / "depth.csv", index=False)
    C = summarise(T, ["unit", "section", "masks", "bin"])
    C.to_csv(out / "cells_by_bin.csv", index=False)
    U = summarise(T, ["unit", "bin"])
    U.to_csv(out / "units.csv", index=False)
    Uk = U[U.cells >= a.min_cells]
    Um = summarise(T, ["unit", "masks", "bin"])
    Um = Um[Um.cells >= a.min_cells]
    S1 = Uk.groupby("bin").agg(units=("unit", "nunique"), cells=("cells", "sum"), loss_pct=("loss_pct", "median"),
                               kept_B=("kept_B", "median"), touch=("touch", "median")).reset_index()
    S1["masks"] = "all"
    S2 = Um.groupby(["masks", "bin"]).agg(units=("unit", "nunique"), cells=("cells", "sum"),
                                          loss_pct=("loss_pct", "median"), kept_B=("kept_B", "median"),
                                          touch=("touch", "median")).reset_index()
    S = pd.concat([S1, S2], ignore_index=True)
    S["bin"] = pd.Categorical(S.bin, names, ordered=True)
    S = S.sort_values(["masks", "bin"])
    S.to_csv(out / "summary.csv", index=False)
    # paired within units: loss inside minus away, per unit with both bins
    P = Uk.pivot_table(index="unit", columns="bin", values="loss_pct")
    paired = {}
    if {names[0], names[-1]} <= set(P.columns):
        d = (P[names[-1]] - P[names[0]]).dropna()
        paired = dict(units=int(len(d)), median_diff=float(d.median()) if len(d) else None,
                      units_higher_inside=int((d > 0).sum()))
    meta = dict(script="region_loss.py", version=_cb_version(), depth_strata=a.depth_strata, gene_sets={k: sorted(v) for k, v in found_all.items()},
                h5ad=a.h5ad, sections_without_h5ad=no_h5, labels=str(a.labels), level=lv, immune=a.immune,
                partners=partners, radius=a.radius, bins=bins, bin_names=names, unit_pattern=a.unit_pattern,
                stain_pattern=a.stain_pattern, min_cells=a.min_cells, sections=len(rows), skipped=skipped,
                paired_inside_minus_away=paired)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 160)
    print("\nmedian over units (bins with at least %d cells): loss of the label, whole-cell counts against half B (%%)"
          % a.min_cells)
    print(S.round(3).to_string(index=False))
    if paired:
        print(f"\nper unit, loss {names[-1]} minus {names[0]}: median {paired['median_diff']:+.1f} points, higher "
              f"{names[-1]} in {paired['units_higher_inside']} of {paired['units']} units")
    if len(ident):
        print("\nidentity of the immune calls by bin, against the partner cells (medians over units):")
        print(ident[ident.unit == "median"].drop(columns=["unit"]).round(3).to_string(index=False))
    if len(state):
        print("\nloss by state of the immune calls (medians over units; positive = >= 1 nuclear count of the set):")
        print(state[state.unit == "median"].drop(columns=["unit"]).round(3).to_string(index=False))
    if len(depth):
        print(f"\nloss positive minus negative (points), crude and within {a.depth_strata} strata of nuclear counts "
              "(medians over units; pooled over all unit-strata):")
        dm = depth[depth.unit == "median"].drop(columns=["unit", "strata"], errors="ignore")
        print(dm.round(2).to_string(index=False))
    if no_h5:
        print(f"\nno h5ad for {len(no_h5)} sections (identity from the labels file only), e.g. {no_h5[0]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
