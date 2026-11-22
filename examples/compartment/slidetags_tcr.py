#!/usr/bin/env python
"""slidetags_tcr.py -- negative control for the segmentation mechanism: are T cells with a recovered TCR labeled T
less often next to tumor cells when there is no segmentation?

Data: Slide-GoTags human tumors (Nagler et al., Nat Biotechnol 2026; Broad Single Cell Portal SCP3667). Nuclei are
isolated and sequenced one by one, then placed back in space, so no mask can collect a neighbor's RNA. In whole-cell
imaging data the T label is lost most where T cells touch tumor cells; here the prediction is that it is not.

Inputs (Single Cell Portal files, read only):
  scp_cluster_spatial.txt   NAME, X, Y, cell_type_original, TRA_cdr3, TRB_cdr3, genotype columns (row 2 = TYPE)
  scp_metadata.txt          NAME, biosample_id, ..., cell_type__ontology_label, Merged_Cluster_Name_scevan_broad

TCR+ nucleus: TRA_cdr3 or TRB_cdr3 recovered (T by sequence, independently of the expression label). For TCR+ nuclei
that are spatially mapped, per section and pooled:
  recall   share labeled T/NK by expression (--t-regex on --label-col)
  fate     what the others are labeled (tumor? another type?)
by distance to the nearest OTHER nucleus labeled tumor (--tumor-regex), in bins (--bins, in coordinate units times
--um-per-unit), and by contact (a tumor-labeled nucleus among the --k nearest mapped nuclei within --r um).
Near/far contrast pooled over sections with Mantel-Haenszel weights (odds ratio of the T label, near vs far) and a
bootstrap over sections. About 40% of nuclei are not mapped, so "nearest" is often the second nearest in reality:
this blurs a contact effect but cannot create one.

  python slidetags_tcr.py labels --spatial scp_cluster_spatial.txt --meta scp_metadata.txt
  python slidetags_tcr.py run    --spatial scp_cluster_spatial.txt --meta scp_metadata.txt --out OUT \\
         --t-regex '...' --tumor-regex '...'

`contamination`: the deposited TCR calls exist only for nuclei already labeled T cell, so the test above
is degenerate (recall 1 by construction). The same question is asked of the expression itself. Streaming the
Matrix Market file twice (it is several GB; no dense copy):
  pass 1  per section, the mean count of every gene in tumor-labeled and in T-labeled nuclei, and each
          nucleus's total; tumor marker genes per section = the --n-markers genes with the largest tumor/T ratio
          among genes detected in at least --min-detect of tumor nuclei and at most --max-detect-t of T nuclei
          (chosen without positions; ranking by the ratio does the selection, and the cap is loose, 0.5, so that
          contamination near tumor, which raises detection in some T nuclei, does not remove the genes that would
          show it)
  pass 2  per nucleus, counts of its section's tumor markers and of T marker genes (--t-genes)
For T-labeled nuclei: tumor-marker load per 1,000 counts by distance to the nearest tumor nucleus and contact vs
not, per section and pooled (ratio of means contact / no contact, median over sections, bootstrap over sections);
the reverse for tumor nuclei (T-marker load by distance to the nearest T nucleus). With whole-cell segmentation the
load rises at contact (a mask collects the neighbor's RNA); for isolated nuclei the prediction is a ratio near 1
(ambient RNA is not local). Also, with --genotype, mutant calls in non-tumor nuclei by distance to tumor
(genotype columns of the spatial file whose value contains 'mut').
  python slidetags_tcr.py contamination --spatial ... --meta ... --mtx matrix.mtx \\
         --features features.tsv --barcodes barcodes.tsv --out OUT \\
         --t-regex '^T-cell$' --tumor-regex '^Tumor$'
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EMPTY = {"", "nan", "none", "na", "null", "-"}


def read_scp(path):
    """Single Cell Portal table: header, then a TYPE row."""
    T = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    if len(T) and str(T.iloc[0, 0]).upper() == "TYPE":
        T = T.iloc[1:].reset_index(drop=True)
    return T


def load(spatial, meta, label_col, section_col):
    S = read_scp(spatial)
    M = read_scp(meta)
    keep = [c for c in M.columns if c == "NAME" or c not in S.columns]
    D = S.merge(M[keep], on="NAME", how="left")
    D["X"], D["Y"] = D["X"].astype(float), D["Y"].astype(float)
    if section_col not in D:
        D[section_col] = D["NAME"].str.rsplit("_", n=1).str[0]
    for c in ("TRA_cdr3", "TRB_cdr3"):
        if c not in D:
            D[c] = ""
    tcr = lambda s: ~s.astype(str).str.strip().str.lower().isin(EMPTY)
    D["tcr"] = tcr(D["TRA_cdr3"]) | tcr(D["TRB_cdr3"])
    if label_col not in D:
        raise SystemExit(f"no column {label_col}; columns: {', '.join(D.columns[:20])} ...")
    D["label"] = D[label_col].astype(str)
    D["section"] = D[section_col].astype(str)
    return D


def cmd_labels(a):
    D = load(a.spatial, a.meta, a.label_col, a.section_col)
    print(f"{len(D):,} mapped nuclei, {D.section.nunique()} sections; TCR+ {int(D.tcr.sum()):,}")
    for c in [a.label_col] + [x for x in ("cell_type_original", "Merged_Cluster_Name_scevan_broad",
                                          "cell_type__ontology_label") if x in D and x != a.label_col]:
        print(f"\n{c}:\n" + D.groupby(D[c].astype(str)).agg(n=("tcr", "size"), tcr_pos=("tcr", "sum"))
              .sort_values("n", ascending=False).to_string())
    R = D.groupby("section").agg(n=("tcr", "size"), tcr=("tcr", "sum"), xmin=("X", "min"), xmax=("X", "max"),
                                 ymin=("Y", "min"), ymax=("Y", "max"))
    print("\nper section (coordinate ranges, to check units):\n" + R.round(0).to_string())


def mh_or(tab):
    """Mantel-Haenszel odds ratio from rows (a, b, c, d) per stratum: a = near & T, b = near & not T,
    c = far & T, d = far & not T."""
    num = den = 0.0
    for a_, b_, c_, d_ in tab:
        n = a_ + b_ + c_ + d_
        if n == 0:
            continue
        num += a_ * d_ / n
        den += b_ * c_ / n
    return num / den if den > 0 else np.nan


def cmd_run(a):
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    D = load(a.spatial, a.meta, a.label_col, a.section_col)
    D["is_T"] = D.label.str.contains(a.t_regex, case=False, regex=True)
    D["is_tum"] = D.label.str.contains(a.tumor_regex, case=False, regex=True)
    bins = [float(x) for x in a.bins.split(",")]
    edges = [0.0] + bins + [np.inf]
    names = [f"<{bins[0]:g}"] + [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])] + [f">{bins[-1]:g}"]
    rows = []
    for sec, g in D.groupby("section"):
        xy = g[["X", "Y"]].to_numpy(float) * a.um_per_unit
        tum = g.is_tum.to_numpy()
        ip = np.where(g.tcr.to_numpy())[0]
        if len(ip) == 0 or tum.sum() < 2:
            continue
        tt = cKDTree(xy[tum])
        dist, idx = tt.query(xy[ip], k=2)
        tum_ix = np.where(tum)[0]
        self_hit = tum_ix[idx[:, 0]] == ip                      # a TCR+ nucleus labeled tumor: skip itself
        dnear = np.where(self_hit, dist[:, 1], dist[:, 0])
        dk, ik = cKDTree(xy).query(xy[ip], k=a.k + 1)
        dk, ik = dk[:, 1:], ik[:, 1:]
        contact = (tum[ik] & (dk <= a.r)).any(1)
        G = g.iloc[ip]
        rows.append(pd.DataFrame(dict(section=sec, name=G.NAME.to_numpy(), label=G.label.to_numpy(),
                                      is_T=G.is_T.to_numpy(), is_tum=G.is_tum.to_numpy(), d_tumor_um=dnear,
                                      contact=contact)))
    if not rows:
        raise SystemExit("no section with TCR+ nuclei and tumor-labelled nuclei; check --tumor-regex")
    T = pd.concat(rows, ignore_index=True)
    T["bin"] = pd.cut(T.d_tumor_um, edges, labels=names, right=False)
    T.to_csv(out / "tcr_nuclei.csv.gz", index=False, compression="gzip")
    byb = T.groupby(["bin"], observed=False).agg(n=("is_T", "size"), recall_T=("is_T", "mean"),
                                                  labelled_tumor=("is_tum", "mean")).reset_index()
    bys = T.groupby(["section", "contact"]).agg(n=("is_T", "size"), recall_T=("is_T", "mean"),
                                                labelled_tumor=("is_tum", "mean")).unstack("contact")
    fate = T.assign(where=np.where(T.contact, "contact", "no contact")).groupby(["where", "label"]).size() \
        .rename("n").reset_index()
    fate["share"] = fate.n / fate.groupby("where").n.transform("sum")
    tab = [(int((g.contact & g.is_T).sum()), int((g.contact & ~g.is_T).sum()),
            int((~g.contact & g.is_T).sum()), int((~g.contact & ~g.is_T).sum())) for _, g in T.groupby("section")]
    OR = mh_or(tab)
    rng = np.random.default_rng(0)
    secs = T.section.unique()
    boot = []
    for _ in range(a.boot):
        pick = rng.choice(len(secs), len(secs), replace=True)
        boot.append(mh_or([tab[i] for i in pick]))
    boot = np.array([b for b in boot if np.isfinite(b)])
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 20 else (np.nan, np.nan))
    near = T[T.contact]
    far = T[~T.contact]
    S = dict(n_tcr_mapped=int(len(T)), n_sections=int(T.section.nunique()),
             recall_contact=float(near.is_T.mean()) if len(near) else np.nan,
             recall_no_contact=float(far.is_T.mean()) if len(far) else np.nan,
             tumor_label_contact=float(near.is_tum.mean()) if len(near) else np.nan,
             tumor_label_no_contact=float(far.is_tum.mean()) if len(far) else np.nan,
             mh_or_T_contact_vs_not=OR, mh_or_lo=float(lo), mh_or_hi=float(hi),
             k=a.k, r_um=a.r, um_per_unit=a.um_per_unit, t_regex=a.t_regex, tumor_regex=a.tumor_regex,
             label_col=a.label_col)
    byb.to_csv(out / "recall_by_distance.csv", index=False)
    bys.to_csv(out / "recall_by_section.csv")
    fate.to_csv(out / "fate.csv", index=False)
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(f"TCR+ mapped nuclei: {len(T):,} in {T.section.nunique()} sections")
    print("\nT/NK label among TCR+ nuclei, by distance to the nearest tumor-labelled nucleus (um):\n"
          + byb.round(3).to_string(index=False))
    print(f"\ncontact (tumor among {a.k} nearest within {a.r:g} um): recall {S['recall_contact']:.3f} "
          f"(n={len(near)}) vs {S['recall_no_contact']:.3f} (n={len(far)}); labelled tumor "
          f"{S['tumor_label_contact']:.3f} vs {S['tumor_label_no_contact']:.3f}")
    print(f"Mantel-Haenszel odds ratio of the T label, contact vs not: {OR:.2f} (95% bootstrap over sections "
          f"{lo:.2f}-{hi:.2f}; <1 = T label lost at contact)")
    print("\nper section:\n" + bys.round(3).to_string())
    top = fate.sort_values("share", ascending=False).groupby("where").head(6)
    print("\nlabels of TCR+ nuclei (top 6 per group):\n" + top.round(3).to_string(index=False))
    print(f"wrote {out}")


T_GENES = "CD3E,CD3D,CD3G,CD2,CD247,TRAC,TRBC1,TRBC2,CD8A,CD8B,CD4,IL7R,LCK,ZAP70,CD96,THEMIS,BCL11B,ITK"
# markers are by default chosen from the data for both directions, ranked by abundance among genes at
# least --min-ratio times higher in their own group (ranking by the ratio alone would pick genes never seen
# in the other group and so give zero loads by construction).


def _mtx_header(path):
    with open(path) as fh:
        line = fh.readline()
        field = line.split()[3] if line.startswith("%%MatrixMarket") else "real"
        while line.startswith("%"):
            line = fh.readline()
        nr, nc, nnz = map(int, line.split())
        skip = 0
        fh.seek(0)
        for l in fh:
            skip += 1
            if not l.startswith("%"):
                break
    return nr, nc, nnz, skip, field


def _stream(path, skip, chunk):
    for c in pd.read_csv(path, sep=r"\s+", header=None, skiprows=skip, chunksize=chunk, names=["i", "j", "v"],
                         dtype={"i": np.int64, "j": np.int64, "v": np.float64}, engine="c"):
        yield c.i.to_numpy() - 1, c.j.to_numpy() - 1, c.v.to_numpy()


def _read_names(path):
    T = pd.read_csv(path, sep="\t", header=None, dtype=str)
    return T.iloc[:, 1].to_numpy() if T.shape[1] > 1 else T.iloc[:, 0].to_numpy()


def cmd_contamination(a):
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    D = load(a.spatial, a.meta, a.label_col, a.section_col)
    D["is_T"] = D.label.str.contains(a.t_regex, case=False, regex=True)
    D["is_tum"] = D.label.str.contains(a.tumor_regex, case=False, regex=True)
    genes = _read_names(a.features)
    bc = pd.read_csv(a.barcodes, sep="\t", header=None, dtype=str).iloc[:, 0].to_numpy()
    nr, nc, nnz, skip, field = _mtx_header(a.mtx)
    if nr == len(genes) and nc == len(bc):
        gi_col, ci_col = 0, 1                         # genes x cells (10x)
    elif nc == len(genes) and nr == len(bc):
        gi_col, ci_col = 1, 0
    else:
        raise SystemExit(f"matrix {nr} x {nc} matches neither {len(genes)} features nor {len(bc)} barcodes")
    pos = pd.Series(np.arange(len(D)), index=D.NAME.to_numpy())
    cell_row = pos.reindex(bc).to_numpy()             # matrix column -> row of D (nan: not mapped)
    if np.isfinite(cell_row.astype(float)).sum() == 0:
        raise SystemExit(f"no barcode of the matrix matches a NAME of the spatial file (barcodes e.g. {bc[:3]}; "
                         f"names e.g. {D.NAME.to_numpy()[:3]})")
    mapped = np.isfinite(cell_row.astype(float))
    crow = np.where(mapped, np.nan_to_num(cell_row.astype(float), nan=-1), -1).astype(np.int64)
    secs = np.unique(D.section)
    sec_ix = pd.Series(np.arange(len(secs)), index=secs)
    d_sec = sec_ix[D.section.to_numpy()].to_numpy()
    grp = np.where(D.is_tum, 0, np.where(D.is_T, 1, 2))
    G, S = len(genes), len(secs)
    print(f"{nnz:,} entries; {mapped.sum():,} of {len(bc):,} barcodes mapped; field {field}", flush=True)
    # pass 1: totals, and per (section, group, gene) sums and detections of the raw values
    tot = np.zeros(len(D))
    sums = np.zeros(S * 2 * G)
    det = np.zeros(S * 2 * G)
    vals_int = True
    for gi, ci, v in _stream(a.mtx, skip, a.chunk):
        g_, c_ = (gi, ci) if gi_col == 0 else (ci, gi)
        r = crow[c_]
        ok = r >= 0
        g_, r, v = g_[ok], r[ok], v[ok]
        if vals_int and np.any(np.abs(v - np.round(v)) > 1e-6):
            vals_int = False
        tot += np.bincount(r, weights=v, minlength=len(D))
        k = grp[r]
        m = k < 2
        ix = (d_sec[r[m]] * 2 + k[m]) * G + g_[m]
        sums += np.bincount(ix, weights=v[m], minlength=S * 2 * G)
        det += np.bincount(ix, weights=(v[m] > 0).astype(float), minlength=S * 2 * G)
    if not vals_int:
        print("WARNING: non-integer values (normalized matrix?); loads are per 1,000 of the summed values", flush=True)
    sums, det = sums.reshape(S, 2, G), det.reshape(S, 2, G)
    n_sg = np.array([[((d_sec == s) & (grp == k)).sum() for k in (0, 1)] for s in range(S)], float)
    mean = sums / np.maximum(n_sg[:, :, None], 1)          # raw counts per nucleus (depths are similar enough)
    fdet = det / np.maximum(n_sg[:, :, None], 1)
    tgenes = set(x.strip().upper() for x in a.t_genes.split(",") if x.strip())
    is_tg = np.array([str(x).upper() in tgenes for x in genes])
    mark = np.zeros((G, S), bool)                      # tumor markers per section
    markT = np.zeros((G, S), bool)                     # T markers per section
    rows = []

    def pick(s, i, j, exclude):
        """Markers of group i against group j in section s: detected in >= min_detect of i, at most max_detect of
        j, mean ratio i/j >= min_ratio; ranked by mean in i (abundance, default) or by the ratio."""
        eps = 1e-3 * max(float(np.median(mean[s, i][mean[s, i] > 0])) if (mean[s, i] > 0).any() else 1.0, 1e-9)
        ratio = (mean[s, i] + eps) / (mean[s, j] + eps)
        ok = (fdet[s, i] >= a.min_detect) & (fdet[s, j] <= a.max_detect_t) & (ratio >= a.min_ratio) & ~exclude
        cand = np.where(ok)[0]
        key = mean[s, i][cand] if a.marker_rank == "abundance" else ratio[cand]
        return cand[np.argsort(-key)[:a.n_markers]], ratio

    for s in range(S):
        if n_sg[s, 0] < 20 or n_sg[s, 1] < 20:
            continue
        top, ratio = pick(s, 0, 1, is_tg)
        mark[top, s] = True
        if a.t_markers == "data":
            topT, _ = pick(s, 1, 0, np.zeros(G, bool))
        else:
            topT = np.where(is_tg)[0]
        markT[topT, s] = True
        rows.append(dict(section=secs[s], n_tumor=int(n_sg[s, 0]), n_T=int(n_sg[s, 1]), n_markers=len(top),
                         median_ratio=float(np.median(ratio[top])) if len(top) else np.nan,
                         tumor_det_in_T=float(fdet[s, 1][top].mean()) if len(top) else np.nan,
                         markers=",".join(map(str, genes[top][:15])), n_T_markers=len(topT),
                         T_markers=",".join(map(str, genes[topT][:15]))))
    pd.DataFrame(rows).to_csv(out / "markers.csv", index=False)
    # pass 2: per-nucleus marker loads
    tum_load = np.zeros(len(D))
    t_load = np.zeros(len(D))
    for gi, ci, v in _stream(a.mtx, skip, a.chunk):
        g_, c_ = (gi, ci) if gi_col == 0 else (ci, gi)
        r = crow[c_]
        ok = r >= 0
        g_, r, v = g_[ok], r[ok], v[ok]
        mt = mark[g_, d_sec[r]]
        tum_load += np.bincount(r[mt], weights=v[mt], minlength=len(D))
        mT = markT[g_, d_sec[r]]
        t_load += np.bincount(r[mT], weights=v[mT], minlength=len(D))
    D["total"] = tot
    D["tumor_marker_per1k"] = np.where(tot > 0, tum_load / np.maximum(tot, 1e-12) * 1000, np.nan)
    D["T_marker_per1k"] = np.where(tot > 0, t_load / np.maximum(tot, 1e-12) * 1000, np.nan)
    D["in_matrix"] = np.isin(np.arange(len(D)), crow[crow >= 0])

    def distances(src, dst, sec_df):
        xy = sec_df[["X", "Y"]].to_numpy(float) * a.um_per_unit
        d_ix = np.where(dst)[0]
        s_ix = np.where(src)[0]
        if len(d_ix) < 2 or len(s_ix) == 0:
            return None
        dist, idx = cKDTree(xy[d_ix]).query(xy[s_ix], k=2)
        self_hit = d_ix[idx[:, 0]] == s_ix
        dn = np.where(self_hit, dist[:, 1], dist[:, 0])
        dk, ik = cKDTree(xy).query(xy[s_ix], k=a.k + 1)
        contact = (dst[ik[:, 1:]] & (dk[:, 1:] <= a.r)).any(1)
        return s_ix, dn, contact

    bins = [float(x) for x in a.bins.split(",")]
    edges = [0.0] + bins + [np.inf]
    names = [f"<{bins[0]:g}"] + [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])] + [f">{bins[-1]:g}"]
    summary = {}
    for what, src_col, dst_col, lcol in (("T nuclei, tumor-marker load", "is_T", "is_tum", "tumor_marker_per1k"),
                                         ("tumor nuclei, T-marker load", "is_tum", "is_T", "T_marker_per1k")):
        recs = []
        for sec, g in D.groupby("section"):
            if not mark[:, sec_ix[sec]].any():
                continue
            res = distances(g[src_col].to_numpy() & g.in_matrix.to_numpy(), g[dst_col].to_numpy(), g)
            if res is None:
                continue
            s_ix, dn, contact = res
            recs.append(pd.DataFrame(dict(section=sec, d_um=dn, contact=contact, load=g[lcol].to_numpy()[s_ix],
                                          total=g.total.to_numpy()[s_ix])))
        if not recs:
            continue
        R = pd.concat(recs, ignore_index=True)
        R["bin"] = pd.cut(R.d_um, edges, labels=names, right=False)
        tag = "T_tumorload" if src_col == "is_T" else "tumor_Tload"
        R.to_csv(out / f"{tag}_nuclei.csv.gz", index=False, compression="gzip")
        byb = R.groupby("bin", observed=False).agg(n=("load", "size"), mean_load=("load", "mean"),
                                                   median_total=("total", "median")).reset_index()
        per = R.groupby("section").apply(lambda x: pd.Series(dict(
            n_contact=int(x.contact.sum()), n_far=int((~x.contact).sum()),
            load_contact=x.load[x.contact].mean(), load_far=x.load[~x.contact].mean())), include_groups=False)
        per["ratio"] = per.load_contact / per.load_far
        okr = per[(per.n_contact >= 5) & (per.n_far >= 5) & np.isfinite(per.ratio)]
        rng = np.random.default_rng(0)
        bt = [np.median(okr.ratio.to_numpy()[rng.integers(0, len(okr), len(okr))]) for _ in range(a.boot)] \
            if len(okr) >= 3 else []
        summary[tag] = dict(sections=int(len(okr)), median_ratio_contact_far=float(okr.ratio.median()),
                            ci_lo=float(np.percentile(bt, 2.5)) if bt else np.nan,
                            ci_hi=float(np.percentile(bt, 97.5)) if bt else np.nan,
                            n_sections_ratio_gt1=int((okr.ratio > 1).sum()))
        byb.to_csv(out / f"{tag}_by_distance.csv", index=False)
        per.to_csv(out / f"{tag}_by_section.csv")
        print(f"\n{what} by distance (um) to the nearest {'tumor' if src_col == 'is_T' else 'T'} nucleus:\n"
              + byb.round(3).to_string(index=False))
        print(f"per section (contact = {'tumor' if src_col == 'is_T' else 'T'} among {a.k} nearest within "
              f"{a.r:g} um):\n" + per.round(3).to_string())
        s_ = summary[tag]
        print(f"median ratio contact / no contact over {s_['sections']} sections: {s_['median_ratio_contact_far']:.2f} "
              f"(95% {s_['ci_lo']:.2f}-{s_['ci_hi']:.2f}); ratio > 1 in {s_['n_sections_ratio_gt1']}", flush=True)
    if a.genotype:
        gcols = [c for c in pd.read_csv(a.spatial, sep="\t", nrows=0).columns
                 if c not in ("NAME", "X", "Y", "cell_type_original", "TRA_cdr3", "TRB_cdr3")]
        S2 = read_scp(a.spatial).set_index("NAME")[gcols]
        mut = S2.apply(lambda c: c.astype(str).str.contains("mut", case=False)).any(axis=1)
        D["mutant"] = mut.reindex(D.NAME).fillna(False).to_numpy()
        recs = []
        for sec, g in D.groupby("section"):
            res = distances(~g.is_tum.to_numpy(), g.is_tum.to_numpy(), g)
            if res is None:
                continue
            s_ix, dn, contact = res
            recs.append(pd.DataFrame(dict(section=sec, contact=contact, mutant=g.mutant.to_numpy()[s_ix],
                                          label=g.label.to_numpy()[s_ix])))
        M = pd.concat(recs, ignore_index=True)
        tab = M.groupby(["contact"]).agg(n=("mutant", "size"), mutant=("mutant", "mean"))
        tab.to_csv(out / "mutant_in_nontumor.csv")
        print("\nnon-tumor nuclei with a mutant call, contact with tumor vs not:\n" + tab.round(4).to_string())
        summary["mutant_in_nontumor"] = tab.reset_index().to_dict("records")
    summary.update(n_markers=a.n_markers, min_detect=a.min_detect, max_detect_t=a.max_detect_t, k=a.k, r_um=a.r,
                   integer_counts=bool(vals_int))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("labels", "run", "contamination"):
        p = sub.add_parser(name)
        p.add_argument("--spatial", required=True)
        p.add_argument("--meta", required=True)
        p.add_argument("--label-col", default="cell_type_original")
        p.add_argument("--section-col", default="biosample_id")
        if name in ("run", "contamination"):
            p.add_argument("--out", required=True)
            p.add_argument("--t-regex", required=True, help="labels counted as T/NK (case-insensitive regex)")
            p.add_argument("--tumor-regex", required=True, help="labels counted as tumor")
            p.add_argument("--um-per-unit", type=float, default=1.0, help="coordinate units to um")
            p.add_argument("--bins", default="15,30,60", help="distance bin edges (um)")
            p.add_argument("--k", type=int, default=6)
            p.add_argument("--r", type=float, default=30.0)
            p.add_argument("--boot", type=int, default=1000)
        if name == "contamination":
            p.add_argument("--mtx", required=True); p.add_argument("--features", required=True)
            p.add_argument("--barcodes", required=True)
            p.add_argument("--n-markers", type=int, default=50)
            p.add_argument("--min-detect", type=float, default=0.2, help="tumor markers: detected in >= this of tumor")
            p.add_argument("--max-detect-t", type=float, default=0.5, help="... and in <= this of T nuclei")
            p.add_argument("--t-genes", default=T_GENES, help="T genes (excluded from tumor markers; T markers "
                                                             "with --t-markers list)")
            p.add_argument("--t-markers", choices=["data", "list"], default="data",
                           help="T markers chosen like the tumor markers (data, default) or the --t-genes list")
            p.add_argument("--marker-rank", choices=["abundance", "ratio"], default="abundance",
                           help="rank candidate markers by expression in their own group (default) or by the ratio")
            p.add_argument("--min-ratio", type=float, default=10.0, help="markers: mean ratio own/other group")
            p.add_argument("--genotype", action="store_true", help="also mutant calls in non-tumor nuclei")
            p.add_argument("--chunk", type=int, default=20_000_000)
    a = ap.parse_args()
    {"labels": cmd_labels, "run": cmd_run, "contamination": cmd_contamination}[a.cmd](a)


if __name__ == "__main__":
    main()
