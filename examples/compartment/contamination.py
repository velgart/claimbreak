#!/usr/bin/env python
"""contamination.py -- how many of a cell's whole-cell counts belong to other cell types, where they come from, and
how the label change depends on it (read only except OUT).

For compartment .h5ad files (X whole-cell, layers nucleus) and their labels (`claimbreak compartment label`, with the
nucleus, nucA, nucB, whole and wholeB arms), and the reference used for labeling:
  markers   genes whose highest mean reference proportion is in one type, at least 2**--min-log2 times the next type
            (Reference.markers, as `compartment label --markers`), so every marker count can be given a type
  per cell  type t = nucleus label (margin >= --min-margin). Foreign share of marker counts in the nucleus, in the
            cytoplasm (whole - nucleus) and in the whole cell: counts in markers of types other than t over all marker
            counts. Excess = foreign share whole - foreign share nucleus (what the mask adds).
            Contamination ratio rho = foreign marker counts / own marker counts in the whole cell.
            Neighbors: the 6 nearest cells within 30 um (types from nucleus labels); `touch_other` if any is of
            another type.
  outputs   OUT/cells.csv.gz (per cell: sample, unit, type, counts, foreign shares, rho, touch_other, flipped =
            whole label != nucleus label, flipped_dm = wholeB != nucA, null = nucB != nucA);
            OUT/by_type.csv (per sample and type: medians, split by touch_other);
            OUT/source.csv (per sample and type: share of foreign marker counts in the cytoplasm owned by each type,
            the excess over the cell's own profile, and each type's share among the cell's other-type neighbors,
            plain and weighted by the neighbors' cytoplasmic counts: foreign counts should follow the neighbors'
            spill, i.e. how many and how RNA-rich they are; the same weighted share among far neighbors,
            ranks k+1..20 within 60 um, and in the whole section, as nulls);
            OUT/flip_curve.csv (per sample, type and bin of log10 rho: cells, flip rates whole and depth-matched, null).
--split-typing: the type of a cell is called (same classifier, reference profiles, platform factor off) from
one binomial half of its nuclear counts and the nucleus foreign share is measured on the other half. Without it, the
type is the nucleus label, called from the same counts on which the nucleus foreign share is measured: a cell is typed
as the type whose markers dominate those counts, so the nucleus share is biased low and the nucleus -> cytoplasm step
high (selection on the measured counts), most at low depth.
the labeling's own settings apply to every typed count. The genes it excluded (meta.json `exclude_genes`,
e.g. CCL5 and EEF1G in synovium, 171 probes in CosMx lung) are dropped from the panel before markers are chosen, and
--split-typing calls the half-A type with the platform factor of the labels' nucleus arm (<sample>.gamma.csv) and the
labeling's floor. The platform factor is applied only with --use-label-gamma; --ignore-label-settings uses every
panel gene and the default floor, as if the labeling had no settings.
OUT/settings.json records what was applied.
--rho-split: rho and the label change are both read from the whole-cell counts, so a cell whose label
changed is a cell whose counts look foreign by construction. With --rho-split the whole-cell counts on the panel genes
are split binomially (p = 0.5, seed + 2): rho is measured on half C1, and half C2, thinned to the depth of nucB, is
labeled with the multinomial classifier and the platform factor of the labels' wholeB arm (<sample>.gamma.csv) and
compared with nucA: OUT/flip_curve_split.csv (flip rate of the C2 label against rho of C1, with the null nucB != nucA
in the same bins).
Also: OUT/pooled.csv, per sample, type and contact with another type, foreign shares pooled over cells (sum of
foreign marker counts over sum of marker counts, nucleus / cytoplasm / whole) and the share of cells with rho > 1;
OUT/pooled_units.csv the same per unit, with the homotypic baseline subtracted (cells touching another type minus cells
touching only their own type): the foreign share that contact adds. by_type.csv keeps the medians over cells.
The flip curve is the question for every tissue and platform: does the chance that a label changes depend on rho
the same way everywhere? The mask, preparation and biology enter through rho; the curve is what would transfer.

  python contamination.py --h5ad-dir H5AD --labels LABELS --ref REF.h5ad --ref-levels L2m=Level2 \
      --ref-merge '^Tu_L\\d=>Tumor' --level L2m --unit-pattern 'L\\d(?=_L)' --out ~/claimbreak_results/contam_lung
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


def unit_of(sample, pat):
    m = re.search(pat, sample)
    return m.group(0) if m else sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-levels", default="L2m=Level2")
    ap.add_argument("--ref-merge", default=r"^Tu_L\d=>Tumour")
    ap.add_argument("--ref-map", default="")
    ap.add_argument("--ref-raw", action="store_true")
    ap.add_argument("--level", default="L2m")
    ap.add_argument("--min-log2", type=float, default=2.0)
    ap.add_argument("--min-prop", type=float, default=1e-4)
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--r", type=float, default=30.0)
    ap.add_argument("--unit-pattern", default=r"L\d(?=_L)")
    ap.add_argument("--samples", default="", help="comma list (default: every labels file with an h5ad)")
    ap.add_argument("--max-cells", type=int, default=0, help="per-cell table: at most this many cells per sample")
    ap.add_argument("--split-typing", action="store_true",
                    help="type from a binomial half of the nuclear counts, nucleus foreign share on the other half")
    ap.add_argument("--rho-split", action="store_true",
                    help="rho from one binomial half of the whole-cell counts, the label change from the other")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-label-gamma", action="store_true",
                    help="split typing with the platform factor of the labels' nucleus arm")
    ap.add_argument("--ignore-label-settings", action="store_true",
                    help="ignore the labelling's excluded genes and platform factor")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import anndata as ad
    from claimbreak import compartment as cm
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    levels = dict(x.split("=", 1) for x in a.ref_levels.split(",") if x)
    ref = cm.Reference.load(a.ref, levels, merge=a.ref_merge, map_file=a.ref_map, raw=a.ref_raw)
    lv = a.level
    labdir, hdir = Path(a.labels).expanduser(), Path(a.h5ad_dir).expanduser()
    samples = [s for s in a.samples.split(",") if s] or sorted(
        f.name.replace(".labels.csv.gz", "") for f in labdir.glob("*.labels.csv.gz") if (hdir / f.name.replace(
            ".labels.csv.gz", ".h5ad")).exists())
    if not samples:
        raise SystemExit("no sample with both labels and an h5ad")
    cells_all, bt, src, curve, curve_s, pool_src = [], [], [], [], [], []
    rng = np.random.default_rng(0)
    import json
    applied = {}
    for sample in samples:
        A = ad.read_h5ad(hdir / f"{sample}.h5ad")
        L = pd.read_csv(labdir / f"{sample}.labels.csv.gz", keep_default_na=False, low_memory=False,
                        dtype={"cell_id": str})
        L = L.set_index("cell_id").reindex(A.obs_names.astype(str))
        # the labeling's excluded genes, floor and nucleus platform factor
        meta_f = labdir / f"{sample}.meta.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() and not a.ignore_label_settings else {}
        excl = set(meta.get("exclude_genes") or [])
        floor = float(meta.get("floor", 0.01)) if a.use_label_gamma else 0.01   # default floor without --use-label-gamma
        genes, gidx, _ = ref.match([g for g in A.var_names if g not in excl])
        gam_nuc = None
        gf_ = labdir / f"{sample}.gamma.csv"
        if a.use_label_gamma and not a.ignore_label_settings and gf_.exists():
            G0 = pd.read_csv(gf_)
            if f"{lv}_nucleus" in G0:
                gam_nuc = G0.set_index("gene")[f"{lv}_nucleus"].reindex(genes).to_numpy(float)
        applied[sample] = dict(excluded=sorted(excl & set(map(str, A.var_names))), floor=floor,
                               nucleus_platform_factor=gam_nuc is not None and bool(np.isfinite(gam_nuc).any()))
        types, mu = ref.profiles(lv, gidx)
        sel, per = ref.markers(lv, gidx, 10_000, a.min_prop, a.min_log2)
        owner = mu[:, sel].argmax(0)
        xi = np.array([A.var_names.get_loc(genes[j]) for j in sel], dtype=int)
        W = sp.csr_matrix(A.X)[:, xi]
        N = sp.csr_matrix(A.layers["nucleus"])[:, xi]
        K = len(types)
        Om = sp.csr_matrix((np.ones(len(owner)), (np.arange(len(owner)), owner)), shape=(len(owner), K))
        Wt = np.asarray((W @ Om).todense())                 # marker counts by owning type, whole cell
        Nt = np.asarray((N @ Om).todense())
        Ct = Wt - Nt
        if a.rho_split:
            # whole-cell counts of the panel genes split binomially: C1 for rho, C2 for the label change
            gi_all = np.array([A.var_names.get_loc(gname) for gname in genes], dtype=int)
            WP = sp.csr_matrix(A.X)[:, gi_all].astype(np.int64)
            W1 = WP.copy()
            W1.data = np.random.default_rng(a.seed + 2).binomial(W1.data, 0.5)
            W2 = (WP - W1).tocsr()
            W1.eliminate_zeros(); W2.eliminate_zeros()
            pos_all = {gname: j for j, gname in enumerate(genes)}
            Wt1 = np.asarray((W1[:, [pos_all[genes[j]] for j in sel]].astype(np.float64) @ Om).todense())
        t = L[f"{lv}_nucleus"].fillna("").astype(str).to_numpy()
        ti = pd.Index(types).get_indexer(t)
        if a.split_typing:
            # independent halves of the nuclear counts on the panel genes: type from A, nucleus share from B
            gi = np.array([A.var_names.get_loc(gname) for gname in genes], dtype=int)
            NP = sp.csr_matrix(A.layers["nucleus"])[:, gi].astype(np.int64)
            XB = NP.copy()
            XB.data = np.random.default_rng(a.seed + 1).binomial(XB.data, 0.5)
            XA = (NP - XB).tocsr()
            XA.eliminate_zeros()
            lab_a, _, _, _, _ = cm.classify(XA.astype(np.float32), mu, cm.LabelConfig(gamma_iter=0, floor=floor),
                                            fit_gamma=False, gamma0=gam_nuc)
            ti = np.where(lab_a >= 0, lab_a, -1)
            t = np.where(ti >= 0, np.array(types, dtype=object)[np.maximum(ti, 0)], "").astype(str)
            pos_ = {gname: j for j, gname in enumerate(genes)}
            NB = XB[:, [pos_[genes[j]] for j in sel]].astype(np.float64)
            Nt = np.asarray((NB @ Om).todense())         # nucleus marker counts by owning type: half B only
        ok = ti >= 0
        if a.min_margin > 0 and f"{lv}_nucleus_margin" in L:
            ok &= pd.to_numeric(L[f"{lv}_nucleus_margin"], errors="coerce").fillna(0).to_numpy() >= a.min_margin
        rows = np.flatnonzero(ok)
        own = lambda M: M[rows, ti[rows]]
        tot = lambda M: M[rows].sum(1)
        fw, fn, fc = (1 - own(M) / np.maximum(tot(M), 1) for M in (Wt, Nt, Ct))
        has = lambda M: tot(M) > 0
        rho = (tot(Wt) - own(Wt)) / np.maximum(own(Wt), 0.5)
        xy = A.obs[["x", "y"]].to_numpy(float)
        cyto_tot = np.asarray(sp.csr_matrix(A.X).sum(1)).ravel() - np.asarray(sp.csr_matrix(A.layers["nucleus"]).sum(1)).ravel()
        d, idx = cKDTree(xy).query(xy, k=a.k + 1)
        d, idx = d[:, 1:], idx[:, 1:]
        # far neighbors (ranks k+1..20 within 60 um) and the section, as nulls for the source correlation:
        # abundant RNA-rich types are frequent contact neighbors and frequent sources even if foreign counts were
        # tissue-wide ambient RNA; contact should predict the sources better than far neighbors or the section
        dF, idxF = cKDTree(xy).query(xy, k=21)
        dF, idxF = dF[:, a.k + 1:], idxF[:, a.k + 1:]
        nb_t = ti[idx]
        near = d <= a.r
        touch_other = ((nb_t != ti[:, None]) & (nb_t >= 0) & near).any(1)[rows]
        lab = lambda arm: L[f"{lv}_{arm}"].fillna("").astype(str).to_numpy()[rows] if f"{lv}_{arm}" in L else None
        w, wb, na, nb_ = lab("whole"), lab("wholeB"), lab("nucA"), lab("nucB")
        flip = np.where((w != "") & (t[rows] != ""), (w != t[rows]).astype(float), np.nan) if w is not None else np.nan
        flip_dm = (np.where((wb != "") & (na != ""), (wb != na).astype(float), np.nan)
                   if wb is not None and na is not None else np.full(len(rows), np.nan))
        null = (np.where((nb_ != "") & (na != ""), (nb_ != na).astype(float), np.nan)
                if nb_ is not None and na is not None else np.full(len(rows), np.nan))
        C = pd.DataFrame({"sample": sample, "unit": unit_of(sample, a.unit_pattern),
                          "panel": str(A.obs["panel"].iloc[0]) if "panel" in A.obs else "",
                          "cell_id": A.obs_names.astype(str)[rows], "type": t[rows],
                          "marker_whole": tot(Wt), "marker_nucleus": tot(Nt),
                          "foreign_whole": np.where(has(Wt), fw, np.nan), "foreign_nucleus": np.where(has(Nt), fn, np.nan),
                          "foreign_cyto": np.where(has(Ct), fc, np.nan), "rho": rho, "touch_other": touch_other,
                          "flipped": flip, "flipped_dm": flip_dm, "null": null})
        C["excess"] = C.foreign_whole - C.foreign_nucleus
        for nm, M in (("whole", Wt), ("nucleus", Nt), ("cyto", Ct)):
            C[f"marker_{nm}_n"] = tot(M)
            C[f"foreign_{nm}_n"] = tot(M) - own(M)
        if a.rho_split:
            C["rho_c1"] = (tot(Wt1) - own(Wt1)) / np.maximum(own(Wt1), 0.5)
            # the genes and platform factor of the labels' wholeB arm, where the labels have them
            use_g, gam = np.arange(len(genes)), None
            gf = labdir / f"{sample}.gamma.csv"
            if gf.exists():
                G_ = pd.read_csv(gf)
                keep_g = set(G_["gene"].astype(str))
                use_g = np.array([j for j, gname in enumerate(genes) if gname in keep_g], dtype=int)
                if f"{lv}_wholeB" in G_:
                    gam = G_.set_index("gene")[f"{lv}_wholeB"].reindex([genes[j] for j in use_g]).to_numpy(float)
            nB = pd.to_numeric(L["n_nucB"], errors="coerce").fillna(0).to_numpy() if "n_nucB" in L else \
                np.asarray(W2.sum(1)).ravel() * 0.25
            X2 = cm.thin(W2[:, use_g].astype(np.float32), nB, np.random.default_rng(a.seed + 3))
            mu2 = mu[:, use_g] / np.maximum(mu[:, use_g].sum(1, keepdims=True), 1e-12)
            lab2, _, _, _, _ = cm.classify(X2, mu2, cm.LabelConfig(gamma_iter=0, floor=floor), fit_gamma=False, gamma0=gam)
            l2 = np.where(lab2 >= 0, np.array(types, dtype=object)[np.maximum(lab2, 0)], "").astype(str)[rows]
            C["flipped_split"] = np.where((l2 != "") & (na != ""), (l2 != na).astype(float), np.nan) \
                if na is not None else np.nan
        print(f"{sample}: {len(C):,} cells typed; {len(sel)} markers ({', '.join(f'{k} {v}' for k, v in per.items())})",
              flush=True)
        for (tt, to), g in C.groupby(["type", "touch_other"]):
            bt.append(dict(sample=sample, unit=C.unit.iloc[0], panel=C.panel.iloc[0], type=tt, touch_other=bool(to),
                           cells=len(g), foreign_nucleus=g.foreign_nucleus.median(), foreign_cyto=g.foreign_cyto.median(),
                           foreign_whole=g.foreign_whole.median(), excess_mean=g.excess.mean(),
                           rho_median=g.rho.median(), flipped=g.flipped.mean(), flipped_dm=g.flipped_dm.mean(),
                           null=g["null"].mean()))
        # sources: foreign cytoplasmic marker counts by owning type vs neighbor type shares
        for k, tt in enumerate(types):
            m = rows[ti[rows] == k]
            if len(m) < 20:
                continue
            fc_t = Ct[m].sum(0)
            # expected cytoplasmic marker counts of each type if the cytoplasm held only the cell's own type
            # (markers are specific, not exclusive): own profile's share in each type's markers x cytoplasmic total
            pk = np.array([mu[k, sel][owner == j].sum() for j in range(K)])
            ex_t = np.clip(fc_t - Ct[m].sum() * pk / max(pk.sum(), 1e-12), 0, None)
            fc_t[k] = 0
            ex_t[k] = 0
            nb = nb_t[m][near[m]]
            nbw = cyto_tot[idx[m]][near[m]]
            keep_ = (nb >= 0) & (nb != k)
            nb, nbw = nb[keep_], nbw[keep_]
            nearF = dF[m] <= 60.0
            fb, fbw = ti[idxF[m]][nearF], cyto_tot[idxF[m]][nearF]
            kf = (fb >= 0) & (fb != k)
            fb, fbw = fb[kf], fbw[kf]
            sec_t = ti[rows]
            sec_w = cyto_tot[rows]
            ks = sec_t != k
            sec_t, sec_w = sec_t[ks], sec_w[ks]
            for j, uu in enumerate(types):
                if j == k:
                    continue
                src.append(dict(sample=sample, unit=C.unit.iloc[0], type=tt, source=uu,
                                share_of_foreign_cyto=float(fc_t[j] / max(fc_t.sum(), 1)),
                                share_of_excess_cyto=float(ex_t[j] / max(ex_t.sum(), 1e-9)),
                                share_of_other_neighbours=float(np.mean(nb == j)) if len(nb) else np.nan,
                                share_of_neighbour_cyto=float(nbw[nb == j].sum() / max(nbw.sum(), 1)) if len(nb) else np.nan,
                                share_of_far_cyto=float(fbw[fb == j].sum() / max(fbw.sum(), 1)) if len(fb) else np.nan,
                                share_of_section_cyto=float(sec_w[sec_t == j].sum() / max(sec_w.sum(), 1)),
                                source_markers=int((owner == j).sum())))
        bins = np.r_[-np.inf, np.arange(-2, 1.01, 0.25), np.inf]
        C["bin"] = pd.cut(np.log10(np.maximum(C.rho, 1e-3)), bins)
        for (tt, b), g in C.groupby(["type", "bin"], observed=True):
            curve.append(dict(sample=sample, unit=C.unit.iloc[0], panel=C.panel.iloc[0], type=tt, bin=str(b),
                              log10_rho_mid=float(np.log10(np.maximum(g.rho, 1e-3)).median()), cells=len(g),
                              flipped=g.flipped.mean(), flipped_dm=g.flipped_dm.mean(), null=g["null"].mean()))
        if a.rho_split:
            C["bin_s"] = pd.cut(np.log10(np.maximum(C.rho_c1, 1e-3)), bins)
            for (tt, b), g in C.groupby(["type", "bin_s"], observed=True):
                curve_s.append(dict(sample=sample, unit=C.unit.iloc[0], panel=C.panel.iloc[0], type=tt, bin=str(b),
                                    log10_rho_mid=float(np.log10(np.maximum(g.rho_c1, 1e-3)).median()), cells=len(g),
                                    flipped_split=g.flipped_split.mean(), null=g["null"].mean()))
            C = C.drop(columns="bin_s")
        pool_src.append(C[[c for c in C.columns if c.endswith("_n") or c in
                           ("sample", "unit", "type", "touch_other", "rho", "rho_c1")]])    # all cells, before --max-cells
        if a.max_cells and len(C) > a.max_cells:
            C = C.iloc[np.sort(rng.choice(len(C), a.max_cells, replace=False))]
        cells_all.append(C.drop(columns="bin"))
    (out / "settings.json").write_text(json.dumps(dict(ignore_label_settings=a.ignore_label_settings,
                                                       split_typing=a.split_typing, rho_split=a.rho_split,
                                                       samples=applied), indent=1))
    CA = pd.concat(cells_all)
    CA.to_csv(out / "cells.csv.gz", index=False, compression="gzip")
    # pooled foreign shares (sums over all typed cells, not only the --max-cells sample) and the homotypic
    # baseline
    CA = pd.concat(pool_src)
    agg = {f"{x}_{nm}_n": (f"{x}_{nm}_n", "sum") for x in ("foreign", "marker") for nm in ("whole", "nucleus", "cyto")}
    def pool(keys):
        g = CA.groupby(keys)
        P_ = g.agg(cells=("rho", "size"), rho_gt1=("rho", lambda v: float(np.mean(v > 1))), **agg).reset_index()
        for nm in ("whole", "nucleus", "cyto"):
            P_[f"pooled_foreign_{nm}"] = P_[f"foreign_{nm}_n"] / P_[f"marker_{nm}_n"].where(P_[f"marker_{nm}_n"] > 0)
        if "rho_c1" in CA:
            P_ = P_.merge(g.rho_c1.apply(lambda v: float(np.mean(v > 1))).rename("rho_c1_gt1").reset_index(), on=keys)
        return P_
    pool(["sample", "unit", "type", "touch_other"]).to_csv(out / "pooled.csv", index=False)
    PU = pool(["unit", "type", "touch_other"])
    val = [c for c in PU.columns if c.startswith(("pooled_foreign_", "rho_"))] + ["cells"]
    Wd = PU.pivot_table(index=["unit", "type"], columns="touch_other", values=val)
    Wd.columns = [f"{c}_{'touch' if t else 'own'}" for c, t in Wd.columns]
    for nm in ("whole", "nucleus", "cyto"):
        if f"pooled_foreign_{nm}_touch" in Wd and f"pooled_foreign_{nm}_own" in Wd:
            Wd[f"contact_adds_{nm}"] = Wd[f"pooled_foreign_{nm}_touch"] - Wd[f"pooled_foreign_{nm}_own"]
    Wd.reset_index().to_csv(out / "pooled_units.csv", index=False)
    if curve_s:
        pd.DataFrame(curve_s).to_csv(out / "flip_curve_split.csv", index=False)
    B, S, F = pd.DataFrame(bt), pd.DataFrame(src), pd.DataFrame(curve)
    B.to_csv(out / "by_type.csv", index=False)
    S.to_csv(out / "source.csv", index=False)
    F.to_csv(out / "flip_curve.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\nforeign share of marker counts (median over samples), by type and contact with another type\n"
          + B.groupby(["type", "touch_other"])[["cells", "foreign_nucleus", "foreign_cyto", "foreign_whole",
                                                 "rho_median", "flipped_dm", "null"]].median().round(3).to_string())
    if len(S):
        print("\ndo foreign cytoplasmic counts follow the neighbours? correlation over (sample, type, source)\n"
              f"  Spearman r, raw foreign counts {S[['share_of_foreign_cyto', 'share_of_other_neighbours']].corr('spearman').iloc[0, 1]:.3f}; "
              f"excess over the cell's own profile {S[['share_of_excess_cyto', 'share_of_other_neighbours']].corr('spearman').iloc[0, 1]:.3f}; "
              f"excess against neighbours weighted by their cytoplasmic counts "
              f"{S[['share_of_excess_cyto', 'share_of_neighbour_cyto']].corr('spearman').iloc[0, 1]:.3f}; "
              f"nulls: far neighbours (ranks {a.k + 1}-20) "
              f"{S[['share_of_excess_cyto', 'share_of_far_cyto']].corr('spearman').iloc[0, 1]:.3f}, section "
              f"{S[['share_of_excess_cyto', 'share_of_section_cyto']].corr('spearman').iloc[0, 1]:.3f}")
        # within (sample, type): does the contact share beat the section share as a predictor of the source shares?
        w = []
        for _, gg in S.groupby(["sample", "type"]):
            if len(gg) >= 3 and gg.share_of_excess_cyto.std() > 0:
                c = gg[["share_of_excess_cyto", "share_of_neighbour_cyto", "share_of_far_cyto",
                        "share_of_section_cyto"]].corr("spearman").iloc[0, 1:]
                w.append(c.to_numpy())
        if w:
            w = np.nanmedian(np.array(w, float), axis=0)
            print(f"  within sample and type (median Spearman r): contact {w[0]:.3f}, far {w[1]:.3f}, section {w[2]:.3f}")
    Wd = Wd.reset_index()
    show = [c for c in ("pooled_foreign_cyto_own", "pooled_foreign_cyto_touch", "contact_adds_cyto",
                        "pooled_foreign_whole_own", "pooled_foreign_whole_touch", "contact_adds_whole",
                        "rho_gt1_own", "rho_gt1_touch", "rho_c1_gt1_own", "rho_c1_gt1_touch") if c in Wd]
    print("\npooled over cells, median over units (own = touching only its own type; touch = touching another type)\n"
          + Wd.groupby("type")[show].median().round(3).to_string())
    if curve_s:
        Fs = pd.DataFrame(curve_s)
        print("\nflip rate of the C2 label against rho of C1 (median over samples)\n"
              + Fs.groupby(["type", "bin"], observed=True)[["cells", "flipped_split", "null"]].median().round(3)
              .to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
