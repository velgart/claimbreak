#!/usr/bin/env python
"""contact_contamination.py -- the neighbor's RNA in a cell, by contact, from whole-cell and from nucleus counts of
the same cells, with a positional check. The imaging counterpart of slidetags_tcr.py contamination,
with the same measure, so that whole-cell masks, nuclei in the same images, and isolated nuclei (Slide-GoTags) are
on one scale.

Per section (a labels file of the claimbreak pipeline and the section's .h5ad: X = whole-cell counts,
layers['nucleus'] = nuclear counts; obs_names = cell_id):
  cell types   from the labels of --arm (default nucA: nuclear half-A labels, which do not lose immune cells at
               contact), immune = --immune, partner = --partner
  markers      partner markers: the --n-markers genes most expressed in partner cells among genes detected in at
               least --min-detect of them and at least --min-ratio times higher (mean) in partner than in immune
               cells, chosen from NUCLEAR counts without positions; immune markers the same way, the other way round
               (the rule of slidetags_tcr.py contamination)
  load         partner-marker counts per 1,000 counts of each immune cell, from whole-cell and from nuclear counts
  contact      a partner cell among the --k nearest cells within --r um (claimbreak's contact)
Per section: load at contact / load away, for whole and nucleus; median over sections and a bootstrap over units
(--unit-pattern on the section name); distance bins to the nearest partner cell. Also the reverse (immune-marker load
of partner cells by contact with immune cells).

Positional check: --jitter 0,5,10,20 moves every cell by an isotropic Gaussian error of that RMS (um)
before contact and distance are computed, and --drop-frac removes that share of cells at random (the same cells at
every jitter level), so that the whole-cell excess can be read at the positional precision and capture rate of
Slide-GoTags. Counts, markers and loads are unchanged; only who counts as "at contact" changes.

Mislabeling checks. A partner cell labeled T/NK sits at contact by construction and carries partner RNA
in its nucleus, so a nuclear excess can come from labels rather than from the nucleus. The options below change which
cells are measured (the partner cells that define contact are always all cells with the partner label):
  --measure nucB     adds the load from nuclear half B (the half the labels did not see), rebuilt exactly as the
                     labeling split it (label genes of <sample>.gamma.csv, seed of <sample>.meta.json, the
                     binomial of claimbreak.compartment.count_sets; checked against n_nucA). Whole-cell and nuclear
                     loads are then computed on the label genes too, so the three are on the same genes;
  --measure nucB3    the same with nuclear thirds (labels --arm nucA3, --agree nucC3, load from third B)
  --min-margin X     measured cells need a label margin >= X nats ({level}_{arm}_margin)
  --agree ARMS       measured cells need the same label in these arms too (e.g. nucB, or nucA3,nucC3)
  --min-own K        measured cells need >= K nuclear counts of their own type's markers
  --protein FILE     cell types by protein instead of labels (Xenium Protein; --sample, --t-class, --partner-protein):
                     immune = CD3E+ (T) or CD3E+ CD45+ CD68- PanCK- (T_strict), partner e.g. CD45- or PANCK+, by
                     the mixture calls of xenium_protein.positive; no RNA label enters
  --tile SIDE        units are square tiles of SIDE um (one ratio per tile), for single sections
Every per-section (or per-tile) ratio is also given for 10%-trimmed means (*_ratio_trim), so that a few cells with
high load do not carry it.
Distribution: a minority of mislabeled partner cells gives a heavy upper tail at contact (the mean ratio
well above the quantile ratios at the middle of the distribution), while contamination of every cell shifts the
whole distribution. <kind>_tail.csv gives per section and measure: contact / away at the 50th, 75th and 90th
percentiles of load (NaN where the away percentile is 0; nuclear loads are often 0 at the median), the mean
ratio with the 5, 10 and 20% most loaded cells removed from contact and away alike (a minority with high load
at contact goes, a shift stays), the ratio of the shares of cells with any partner-marker
count, and the AUC (probability that a contact cell's load exceeds an away cell's, ties half). --from-cells DIR
computes these from the cells.csv.gz of an earlier run without reading counts. The AUC and
the quantile ratios read the shape; the trimmed ratios are unstable where most away cells have a load of 0 (the
denominator after trimming is then a few counts).

  python contact_contamination.py --labels LABELS_DIR --h5ad H5AD_DIR --level L2m --immune T_NK --partner Tumor \\
      --unit-pattern 'L\\d(?=_L)' --out OUT
"""
import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import trim_mean

sys.path.insert(0, str(Path(__file__).resolve().parent))


def read_counts(path):
    import anndata as ad
    import scipy.sparse as sp
    A = ad.read_h5ad(path)
    ids = A.obs_names.astype(str).to_numpy()
    return ids, A.var_names.astype(str).to_numpy(), sp.csr_matrix(A.X), sp.csr_matrix(A.layers["nucleus"])


def half_b(genes, W, N, meta_dir, stem, which="nucB"):
    """Whole, nucleus, the labeled nuclear part (nucA or nucA3) and the unseen one (nucB or nucB3) on the label
    genes, split as the labeling split them (count_sets: halves from seed + 1, thirds from seed + 2)."""
    from claimbreak.compartment import count_sets
    mf, gf = meta_dir / f"{stem}.meta.json", meta_dir / f"{stem}.gamma.csv"
    if not (mf.exists() and gf.exists()):
        return None, f"no {mf.name} or {gf.name}"
    m = json.loads(mf.read_text())
    lg = pd.read_csv(gf)["gene"].astype(str).tolist()
    gpos = pd.Series(np.arange(len(genes)), index=genes)
    miss = [g for g in lg if g not in gpos.index]
    if miss:
        return None, f"{len(miss)} label genes not in the h5ad"
    xidx = gpos[lg].to_numpy()
    Ws, Ns = W[:, xidx].astype(np.float32), N[:, xidx].astype(np.float32)
    seen = "nucA" if which == "nucB" else "nucA3"
    S = count_sets(Ws, Ns, None, [seen, which], int(m.get("seed", 0)))
    exact = int(m.get("markers", 0) or 0) == 0     # with marker genes, n_<arm> counts only the markers of level 1
    return (np.array(lg), Ws.tocsr(), Ns.tocsr(), S[seen].tocsr(), S[which].tocsr(), exact), ""


def protein_types(D, a):
    """immune and partner masks from protein (xenium_protein.positive), as protein_contacts.py."""
    from xenium_protein import CLASSES, STRICT, norm, positive
    cols = {norm(c): c for c in D.columns}
    req, neg = (CLASSES["T"][0], []) if a.t_class == "T" else (STRICT["T_strict"][0], STRICT["T_strict"][1])
    part = []
    for x in [y.strip() for y in a.partner_protein.split(",") if y.strip()]:
        if x[-1] not in "+-":
            raise SystemExit(f"--partner-protein {x}: end each marker with + or -")
        part.append((norm(x[:-1]), x[-1] == "+"))
    pos = {}
    for mk in sorted(set(req + neg + [p for p, _ in part])):
        if mk not in cols:
            raise SystemExit(f"protein marker {mk} missing")
        pos[mk] = positive(D[cols[mk]].to_numpy(), seed=a.seed, post=0.5)[0]
    imm = np.logical_and.reduce([pos[m] for m in req])
    for m in neg:
        imm &= ~pos[m]
    par = np.ones(len(D), bool)
    for m, want in part:
        par &= pos[m] if want else ~pos[m]
    return imm & ~par, par & ~imm


def pick(M, own, other, a):
    """Marker genes of the cells `own` against `other` from counts M (cells x genes)."""
    n_own, n_oth = max(int(own.sum()), 1), max(int(other.sum()), 1)
    mo = np.asarray(M[own].sum(0)).ravel() / n_own
    mt = np.asarray(M[other].sum(0)).ravel() / n_oth
    det = np.asarray((M[own] > 0).sum(0)).ravel() / n_own
    eps = 1e-3 * max(float(np.median(mo[mo > 0])) if (mo > 0).any() else 1.0, 1e-9)
    ratio = (mo + eps) / (mt + eps)
    cand = np.where((det >= a.min_detect) & (ratio >= a.min_ratio))[0]
    return cand[np.argsort(-mo[cand])[:a.n_markers]], ratio


def load_per1k(M, rows, genes_ix):
    tot = np.asarray(M[rows].sum(1)).ravel()
    m = np.asarray(M[rows][:, genes_ix].sum(1)).ravel() if len(genes_ix) else np.zeros(len(rows))
    return np.where(tot > 0, m / np.maximum(tot, 1e-12) * 1000, np.nan), tot


def tail_rows(X, measures, min_cells):
    """per section: quantile ratios, positive-share ratio and AUC of load at contact against away."""
    from scipy.stats import mannwhitneyu
    rows = []
    for (sec, unit), g in X.groupby(["section", "unit"]):
        c, f_ = g[g.contact], g[~g.contact]
        if len(c) < min_cells or len(f_) < min_cells:
            continue
        row = dict(section=sec, unit=unit, n_contact=len(c), n_away=len(f_))
        for m in measures:
            vc, vf = c[f"load_{m}"].dropna().to_numpy(), f_[f"load_{m}"].dropna().to_numpy()
            if len(vc) < min_cells or len(vf) < min_cells:
                continue
            row[f"{m}_mean_ratio"] = vc.mean() / vf.mean() if vf.mean() > 0 else np.nan
            for q in (50, 75, 90):
                qf = np.percentile(vf, q)
                row[f"{m}_q{q}_ratio"] = np.percentile(vc, q) / qf if qf > 0 else np.nan
            for k in (5, 10, 20):                 # the k% most loaded cells removed, at contact and away alike
                sc, sf = np.sort(vc)[:len(vc) - int(len(vc) * k / 100)], np.sort(vf)[:len(vf) - int(len(vf) * k / 100)]
                row[f"{m}_drop{k}_ratio"] = sc.mean() / sf.mean() if sf.mean() > 0 else np.nan
            pf = (vf > 0).mean()
            row[f"{m}_pos_ratio"] = (vc > 0).mean() / pf if pf > 0 else np.nan
            row[f"{m}_auc"] = mannwhitneyu(vc, vf).statistic / (len(vc) * len(vf))
        rows.append(row)
    return pd.DataFrame(rows)


def tail_summary(T, measures):
    out = {}
    for m in measures:
        out[m] = {k: (float(T[f"{m}_{k}"].replace([np.inf, -np.inf], np.nan).median())
                      if f"{m}_{k}" in T and T[f"{m}_{k}"].notna().any() else np.nan)
                  for k in ("mean_ratio", "drop5_ratio", "drop10_ratio", "drop20_ratio", "q50_ratio", "q75_ratio",
                            "q90_ratio", "pos_ratio", "auc")}
    return out


def print_tail(kind, S):
    print(f"  {kind} cells, contact / away by part of the distribution (median over sections):")
    for m, d in S.items():
        print(f"    {m:8s} mean {d['mean_ratio']:.2f}; top 5/10/20% removed "
              f"{d['drop5_ratio']:.2f}/{d['drop10_ratio']:.2f}/{d['drop20_ratio']:.2f}; q50 {d['q50_ratio']:.2f}, q75 {d['q75_ratio']:.2f}, "
              f"q90 {d['q90_ratio']:.2f}; any count {d['pos_ratio']:.2f}; AUC {d['auc']:.3f}")


def from_cells(src, out, a):
    """the distribution summaries from an earlier run's cells.csv.gz."""
    C = pd.read_csv(Path(src).expanduser() / "cells.csv.gz", dtype={"section": str, "unit": str})
    if "jitter" in C:
        C = C[C.jitter == C.jitter.min()]
    measures = [c[5:] for c in C.columns if c.startswith("load_")]
    out.mkdir(parents=True, exist_ok=True)
    summ = {"source": str(src), "measures": measures}
    for kind in ("immune", "partner"):
        T = tail_rows(C[C.kind == kind], measures, a.min_cells)
        T.to_csv(out / f"{kind}_tail.csv", index=False)
        summ[kind] = tail_summary(T, measures)
        print_tail(kind, summ[kind])
    (out / "tail_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(f"wrote {out}")


def unit_of(name, pat):
    m = re.search(pat, name) if pat else None
    return m.group(0) if m else name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=""); ap.add_argument("--h5ad", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--level", default="L2m")
    ap.add_argument("--arm", default="nucA", help="labels defining the cell types (default nuclear half A)")
    ap.add_argument("--immune", default="T_NK"); ap.add_argument("--partner", default="")
    ap.add_argument("--unit-pattern", default="")
    ap.add_argument("--n-markers", type=int, default=50); ap.add_argument("--min-detect", type=float, default=0.2)
    ap.add_argument("--min-ratio", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--r", type=float, default=30.0)
    ap.add_argument("--bins", default="15,30,60"); ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-cells", type=int, default=5, help="cells at contact and away for a section's ratio")
    ap.add_argument("--jitter", default="0", help="positional error levels, RMS um, comma-separated")
    ap.add_argument("--drop-frac", type=float, default=0.0, help="share of cells removed at random before contact")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure", default="", help="nucB (half B) or nucB3 (third B): add the load from the nuclear "
                    "part no label saw")
    ap.add_argument("--meta", default="", help="directory of <sample>.meta.json and .gamma.csv (default --labels)")
    ap.add_argument("--min-margin", type=float, default=0.0, help="measured cells: label margin >= this (nats)")
    ap.add_argument("--agree", default="", help="measured cells: same label in these arms, comma-separated")
    ap.add_argument("--min-own", type=int, default=0, help="measured cells: >= this many own-marker nuclear counts")
    ap.add_argument("--protein", default="", help="<sample>.protein.csv.gz: cell types by protein")
    ap.add_argument("--sample", default="", help="only this section (needed with --protein)")
    ap.add_argument("--t-class", default="T", choices=["T", "T_strict"])
    ap.add_argument("--partner-protein", default="CD45-")
    ap.add_argument("--tile", type=float, default=0.0, help="units = square tiles of this side (um)")
    ap.add_argument("--from-cells", default="", help="distribution summaries from DIR/cells.csv.gz only")
    a = ap.parse_args(argv)
    if not a.from_cells and not (a.labels and a.h5ad and a.partner):
        ap.error("--labels, --h5ad and --partner are needed (or --from-cells)")
    if a.from_cells:
        out = Path(a.out).expanduser()
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} exists and is not empty")
        return from_cells(a.from_cells, out, a)
    if a.measure not in ("", "nucB", "nucB3"):
        raise SystemExit("--measure: nucB or nucB3")
    a.seen = {"nucB": "nucA", "nucB3": "nucA3"}.get(a.measure, "")
    if a.protein and not a.sample:
        raise SystemExit("--protein needs --sample")
    if a.protein and (a.min_margin > 0 or a.agree):
        raise SystemExit("--min-margin and --agree act on labels; not with --protein")
    a.measures = ["whole", "nucleus"] + ([a.measure] if a.measure else [])
    agree = [x for x in a.agree.split(",") if x]
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    files = sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz"))
    if a.sample:
        files = [f for f in files if f.name.split(".labels")[0] == a.sample]
    meta_dir = Path(a.meta or a.labels).expanduser()
    jits = [float(x) for x in a.jitter.split(",")]
    if not 0 <= a.drop_frac < 1:
        raise SystemExit("--drop-frac must be in [0, 1)")
    col = f"{a.level}_{a.arm}"
    bins = [float(x) for x in a.bins.split(",")]
    edges = [0.0] + bins + [np.inf]
    bnames = [f"<{bins[0]:g}"] + [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])] + [f">{bins[-1]:g}"]
    cells, mrows, skipped = [], [], []
    for f in files:
        stem = f.name.split(".labels")[0]
        hp = Path(a.h5ad).expanduser() / f"{stem}.h5ad"
        cols = pd.read_csv(f, nrows=0).columns
        need = [] if a.protein else [col] + [f"{a.level}_{x}" for x in agree] + \
            ([f"{col}_margin"] if a.min_margin > 0 else [])
        lack = [c for c in need if c not in cols]
        if lack or not hp.exists():
            skipped.append(f"{stem} ({'no ' + ','.join(lack) if lack else 'no h5ad'})")
            continue
        ext = need + ([f"n_{a.seen}"] if a.measure and f"n_{a.seen}" in cols else [])
        L = pd.read_csv(f, usecols=["cell_id", "x", "y"] + ext, dtype={"cell_id": str}, keep_default_na=False)
        if a.protein:
            P = pd.read_csv(Path(a.protein).expanduser(), dtype={"cell_id": str})
            L = L.merge(P, on="cell_id", how="inner")
        ids, genes, W, N = read_counts(hp)
        if a.measure:
            hb, why = half_b(genes, W, N, meta_dir, stem, a.measure)
            if hb is None:
                skipped.append(f"{stem} ({why})")
                continue
            genes, W, N, NA, NB, exact = hb
        pos = pd.Series(np.arange(len(ids)), index=ids)
        r = pos.reindex(L.cell_id).to_numpy()
        ok = np.isfinite(r.astype(float))
        L, r = L[ok].reset_index(drop=True), r[ok].astype(int)
        if a.protein:
            imm, par = protein_types(L, a)
        else:
            lab = L[col].astype(str).to_numpy()
            imm, par = lab == a.immune, lab == a.partner
        if imm.sum() < 20 or par.sum() < 20:
            skipped.append(f"{stem} ({imm.sum()} {a.immune}, {par.sum()} {a.partner})")
            continue
        Nr, Wr = N[r], W[r]
        mats = {"whole": Wr, "nucleus": Nr}
        split_ok = np.nan
        if a.measure:
            mats[a.measure] = NB[r]
            if f"n_{a.seen}" in L:
                nA = np.asarray(NA[r].sum(1)).ravel()
                nL = pd.to_numeric(L[f"n_{a.seen}"], errors="coerce").to_numpy()
                # exact: n_<seen> = row sums of the rebuilt part; with marker genes only a bound (n <= row sums)
                split_ok = float(np.mean(np.abs(nA - nL) < 0.5)) if exact else float(np.mean(nL <= nA + 0.5))
                if not exact:
                    print(f"  {stem}: labels used marker genes; split checked as n_{a.seen} <= rebuilt counts "
                          f"({split_ok:.2%})", flush=True)
                if split_ok < 0.99:
                    skipped.append(f"{stem} (half A rebuilt for {split_ok:.1%} of cells only: not the labelling split)")
                    continue
            else:
                print(f"  {stem}: no n_{a.seen} in the labels; the {a.measure} split is not checked", flush=True)
        mk_p, rat_p = pick(Nr, par, imm, a)
        mk_i, rat_i = pick(Nr, imm, par, a)
        keep_m = np.ones(len(L), bool)                     # measured cells; contact uses all of par / imm
        if a.min_margin > 0:
            keep_m &= pd.to_numeric(L[f"{col}_margin"], errors="coerce").fillna(-np.inf).to_numpy() >= a.min_margin
        for x in agree:
            keep_m &= L[f"{a.level}_{x}"].astype(str).to_numpy() == L[col].astype(str).to_numpy()
        if a.min_own > 0:
            own_i = np.asarray(Nr[:, mk_i].sum(1)).ravel() if len(mk_i) else np.zeros(len(L))
            own_p = np.asarray(Nr[:, mk_p].sum(1)).ravel() if len(mk_p) else np.zeros(len(L))
            keep_m &= np.where(imm, own_i, own_p) >= a.min_own
        mrows.append(dict(section=stem, n_immune=int(imm.sum()), n_partner=int(par.sum()), n_genes=len(genes),
                          n_immune_measured=int((imm & keep_m).sum()), n_partner_measured=int((par & keep_m).sum()),
                          split_reproduced=split_ok,
                          n_partner_markers=len(mk_p), partner_markers=",".join(genes[mk_p][:15]),
                          n_immune_markers=len(mk_i), immune_markers=",".join(genes[mk_i][:15])))
        xy0 = L[["x", "y"]].to_numpy(float)
        rng_s = np.random.default_rng([a.seed, zlib.crc32(stem.encode())])
        keep = rng_s.random(len(xy0)) >= a.drop_frac if a.drop_frac > 0 else np.ones(len(xy0), bool)
        kx = np.where(keep)[0]
        for j in jits:
            xy = xy0[kx] + (rng_s.normal(0, j / np.sqrt(2), (len(kx), 2)) if j > 0 else 0)
            im_k, pa_k, km_k = imm[kx], par[kx], keep_m[kx]
            dk, ik = cKDTree(xy).query(xy, k=a.k + 1)
            dk, ik = dk[:, 1:], ik[:, 1:]
            for src, dst, mk, kind in ((im_k, pa_k, mk_p, "immune"), (pa_k, im_k, mk_i, "partner")):
                s_ix = np.where(src & km_k)[0]
                d_ix = np.where(dst)[0]
                if len(s_ix) == 0 or len(d_ix) == 0:
                    continue
                dn = cKDTree(xy[d_ix]).query(xy[s_ix], k=1)[0]
                contact = (dst[ik[s_ix]] & (dk[s_ix] <= a.r)).any(1)
                if a.tile > 0:                              # tiles on the unjittered positions
                    t_ = np.floor(xy0[kx[s_ix]] / a.tile).astype(int)
                    sec = np.array([f"{stem}_t{u}_{v}" for u, v in t_])
                    unit = sec
                else:
                    sec, unit = stem, unit_of(stem, a.unit_pattern)
                cols_ = dict(section=sec, unit=unit, kind=kind, jitter=j, d_um=dn, contact=contact)
                for m, M in mats.items():
                    cols_[f"load_{m}"], cols_[f"n_{m}"] = load_per1k(M, kx[s_ix], mk)
                cells.append(pd.DataFrame(cols_))
        print(f"{stem}: {imm.sum():,} {a.immune}, {par.sum():,} {a.partner}; markers {len(mk_p)} / {len(mk_i)} "
              f"of {len(genes)} genes", flush=True)
    if not cells:
        raise SystemExit("no section analysed: " + "; ".join(skipped))
    out.mkdir(parents=True, exist_ok=True)
    C = pd.concat(cells, ignore_index=True)
    C["bin"] = pd.cut(C.d_um, edges, labels=bnames, right=False)
    C.to_csv(out / "cells.csv.gz", index=False, compression="gzip")
    pd.DataFrame(mrows).to_csv(out / "markers.csv", index=False)
    summary = dict(skipped=skipped, k=a.k, r_um=a.r, arm=col, immune=a.immune, partner=a.partner,
                   n_markers=a.n_markers, min_ratio=a.min_ratio, min_detect=a.min_detect, jitter_um=jits,
                   drop_frac=a.drop_frac, measures=a.measures, min_margin=a.min_margin, agree=agree,
                   min_own=a.min_own, protein=a.protein, t_class=a.t_class if a.protein else "",
                   partner_protein=a.partner_protein if a.protein else "", tile_um=a.tile, by_jitter={})
    pd.set_option("display.width", 200)
    for j in jits:
        tag = "" if len(jits) == 1 else f"_j{j:g}"
        if len(jits) > 1 or j > 0 or a.drop_frac > 0:
            print(f"\n##### positional error {j:g} um RMS, {a.drop_frac:.0%} of cells dropped")
        S_j = summarise(C[C.jitter == j], a, out, tag)
        summary["by_jitter"][f"{j:g}"] = S_j
        if j == jits[0]:
            summary.update(S_j)
    if len(jits) > 1:
        print("\ncontact / away by positional error (median over sections):")
        rows = [dict(jitter_um=j, **{f"{kind}_{m}": summary["by_jitter"][f"{j:g}"][kind][m]["median_ratio"]
                                     for kind in ("immune", "partner") for m in ("whole", "nucleus")}) for j in jits]
        T = pd.DataFrame(rows)
        T.to_csv(out / "by_jitter.csv", index=False)
        print(T.round(2).to_string(index=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if skipped:
        print("\nskipped: " + "; ".join(skipped))
    print(f"wrote {out}")


def summarise(C, a, out, tag=""):
    rng = np.random.default_rng(0)
    summary = {}
    for kind, what in (("immune", f"{a.immune} cells, {a.partner}-marker load"),
                       ("partner", f"{a.partner} cells, {a.immune}-marker load")):
        X = C[C.kind == kind]
        agg = dict(n=("contact", "size"))
        for m in a.measures:
            agg[m] = (f"load_{m}", "mean")
        for m in a.measures:
            agg[f"n_{m}"] = (f"n_{m}", "median")
        byb = X.groupby("bin", observed=False).agg(**agg).reset_index()
        per = []
        for (sec, unit), g in X.groupby(["section", "unit"]):
            c, f_ = g[g.contact], g[~g.contact]
            if len(c) < a.min_cells or len(f_) < a.min_cells:
                continue
            row = dict(section=sec, unit=unit, n_contact=len(c), n_away=len(f_))
            for m in a.measures:
                vc, vf = c[f"load_{m}"].dropna(), f_[f"load_{m}"].dropna()
                lc, lf = vc.mean(), vf.mean()
                row[f"{m}_contact"], row[f"{m}_away"] = lc, lf
                row[f"{m}_ratio"] = lc / lf if lf > 0 else np.nan
                tc, tf = trim_mean(vc, 0.1), trim_mean(vf, 0.1)
                row[f"{m}_ratio_trim"] = tc / tf if tf > 0 else np.nan
            per.append(row)
        P = pd.DataFrame(per)
        S = {}
        for x, t in [(x, t) for t in ("", "_trim") for x in a.measures]:
            m = x + t
            v = P[f"{x}_ratio{t}"].replace([np.inf, -np.inf], np.nan).dropna() if len(P) else pd.Series(dtype=float)
            units = P.loc[v.index, "unit"].to_numpy() if len(v) else np.array([])
            uu = np.unique(units)
            bt = []
            for _ in range(a.boot if len(uu) >= 3 else 0):
                pick_u = rng.choice(uu, len(uu), replace=True)
                bt.append(np.median(np.concatenate([v.to_numpy()[units == u] for u in pick_u])))
            S[m] = dict(sections=int(len(v)), units=int(len(uu)), median_ratio=float(v.median()) if len(v) else np.nan,
                        ci_lo=float(np.percentile(bt, 2.5)) if bt else np.nan,
                        ci_hi=float(np.percentile(bt, 97.5)) if bt else np.nan, n_gt1=int((v > 1).sum()))
        T = tail_rows(X, a.measures, a.min_cells)
        T.to_csv(out / f"{kind}_tail{tag}.csv", index=False)
        S["tail"] = tail_summary(T, a.measures)
        summary[kind] = S
        byb.to_csv(out / f"{kind}_by_distance{tag}.csv", index=False)
        P.to_csv(out / f"{kind}_by_section{tag}.csv", index=False)
        print(f"\n{what} per 1,000 counts by distance (um) to the nearest "
              f"{a.partner if kind == 'immune' else a.immune} cell:\n" + byb.round(3).to_string(index=False))
        for m in [k for k in S if k != "tail"]:
            s_ = S[m]
            print(f"  {m:12s}: contact / away, median over {s_['sections']} {'tiles' if a.tile > 0 else 'sections'} ({s_['units']} units) "
                  f"{s_['median_ratio']:.2f} (95% {s_['ci_lo']:.2f}-{s_['ci_hi']:.2f}); > 1 in {s_['n_gt1']}")
        print_tail(kind, S["tail"])
    return summary


if __name__ == "__main__":
    main()
