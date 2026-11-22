"""Compare claims with the baseline and with planted truth; summarise break points."""
from __future__ import annotations

import numpy as np
import pandas as pd


def versus_baseline(base: pd.DataFrame, run: pd.DataFrame) -> dict:
    """Counts of kept / lost / flipped / spurious claims relative to the baseline decisions."""
    m = base[["claim", "decision"]].merge(run[["claim", "decision"]], on="claim", how="outer",
                                          suffixes=("_base", "_run")).fillna(0)
    b, r = m.decision_base.to_numpy(), m.decision_run.to_numpy()
    pos = b != 0
    kept = int(((b == r) & pos).sum())
    lost = int((pos & (r == 0)).sum())
    flipped = int((pos & (r != 0) & (r != b)).sum())
    spurious = int(((b == 0) & (r != 0)).sum())
    n_pos = int(pos.sum())
    n_run = int((r != 0).sum())
    union = n_pos + n_run - kept
    return {"n_base": n_pos, "kept": kept, "lost": lost, "flipped": flipped, "spurious": spurious,
            "n_run": n_run,
            "survival": kept / n_pos if n_pos else np.nan,
            "jaccard": kept / union if union else np.nan,
            "precision": kept / n_run if n_run else np.nan,
            "sign_flip_rate": flipped / n_pos if n_pos else np.nan,
            "spurious_per_base": spurious / n_pos if n_pos else np.nan}


def planted_truth(planted: list, question: str, types) -> set:
    """Claims implied by planted signals, as {(claim, sign)}; currently for proximity_de spike-ins."""
    out = set()
    if question not in ("proximity_de", "proximity_de_joint"):
        return out
    for rec in planted:
        if rec.get("kind") == "spike":
            for g in rec["genes"]:
                out.add((f"{rec['receiver']}|{rec['sender']}|{g}", rec["sign"]))
    return out


def truth_detail(run: pd.DataFrame, planted: list) -> dict:
    """Niche-DE calls against each planted spike-in separately (spikes numbered 1, 2, ... in config order):
    power_<i> (planted genes called up in the receiver with the true sender), misattrib_up_<i> / misattrib_down_<i>
    (planted genes called in the receiver with another neighbor type), and their totals."""
    spikes = [r for r in planted if r.get("kind") == "spike"]
    if not spikes:
        return {}
    c = run[run.decision != 0]
    parts = c.claim.str.split("|", expand=True) if len(c) else pd.DataFrame(columns=[0, 1, 2])
    d = c.decision.to_numpy()
    out = {"misattrib_up": 0, "misattrib_down": 0}
    for i, r in enumerate(spikes, 1):
        inrec = (parts[0] == r["receiver"]).to_numpy() & parts[2].isin(r["genes"]).to_numpy() if len(c) else \
            np.zeros(0, bool)
        right = (parts[1] == r["sender"]).to_numpy() if len(c) else np.zeros(0, bool)
        sign = r.get("sign", 1)
        out[f"power_{i}"] = float((inrec & right & (d == sign)).sum()) / max(len(r["genes"]), 1)
        out[f"misattrib_up_{i}"] = int((inrec & ~right & (d > 0)).sum())
        out[f"misattrib_down_{i}"] = int((inrec & ~right & (d < 0)).sum())
        out["misattrib_up"] += out[f"misattrib_up_{i}"]
        out["misattrib_down"] += out[f"misattrib_down_{i}"]
    return out


def on_universe(run: pd.DataFrame, universe) -> pd.DataFrame:
    """The run's claims reindexed to a common claim set: claims the run could not estimate get p = 1, stat 0,
    decision 0 (so every corrector is scored on the same negatives and positives as the naive model)."""
    u = pd.Index(list(universe))
    r = run.drop_duplicates("claim").set_index("claim").reindex(u)
    r["p"] = r["p"].fillna(1.0)
    for c in ("stat", "effect"):
        if c in r:
            r[c] = r[c].fillna(0.0)
    r["decision"] = r["decision"].fillna(0).astype(int)
    r["estimable"] = run.drop_duplicates("claim").set_index("claim").reindex(u)["p"].notna().to_numpy()
    return r.reset_index().rename(columns={"index": "claim"})


def claim_classes(claims, planted: list) -> pd.DataFrame:
    """Claim classes for niche-DE claims "A|B|gene" on a null tissue.

    cls: "planted" (planted gene, its receiver, the true sender), "misattr" (planted gene, its receiver, another
    neighbor type) or "null_claim" (every other claim; not "null", which CSV readers read as missing): on the
    permuted tissue, with size factors from non-planted genes,
    its reference-label coefficient is 0). Flags: nonrecv (index type receives no planting: the 0.5 null set,
    kept as a sensitivity class), recv_nonplanted (non-planted gene in a receiver: diagnostic class for any
    remaining normalisation artifact). pair: "receiver|sender" for planted and misattributed claims (bootstrap
    cluster)."""
    spikes = [r for r in planted if r.get("kind") == "spike"]
    ids = pd.Series(np.asarray(claims, dtype=object))
    parts = ids.str.split("|", expand=True)
    A, B, g = parts[0].to_numpy(), parts[1].to_numpy(), parts[2].to_numpy()
    recv_genes = {}
    for r in spikes:
        for gg in r["genes"]:
            recv_genes[(r["receiver"], gg)] = r["sender"]
    receivers = {r["receiver"] for r in spikes}
    snd = np.array([recv_genes.get((a, gg)) for a, gg in zip(A, g)], dtype=object)
    in_recv_pl = np.array([x is not None for x in snd])
    cls = np.where(in_recv_pl, np.where(snd == B, "planted", "misattr"), "null_claim")
    isrecv = np.isin(A, list(receivers))
    pair = np.where(in_recv_pl, pd.Series(A).astype(str) + "|" + pd.Series(snd).astype(str), "")
    return pd.DataFrame({"claim": ids.to_numpy(), "cls": cls, "index": A, "neighbour": B, "gene": g,
                         "nonrecv": (cls == "null_claim") & ~isrecv,
                         "recv_nonplanted": (cls == "null_claim") & isrecv & ~np.isin(g, [x for r in spikes for x in r["genes"]]),
                         "pair": pair})


def _se_df(m: pd.DataFrame, df_default: int):
    """Standard errors and degrees of freedom of a claims table: the stored columns or, for older
    tables, se = |stat| / t(p)."""
    from scipy import stats as st
    if "se" in m and "df" in m:
        return m.se.to_numpy(float), np.where(np.isfinite(m.df.to_numpy(float)), m.df.to_numpy(float), df_default)
    dfv = np.full(len(m), float(max(df_default, 1)))
    tval = st.t.isf(np.clip(m.p.to_numpy(), 1e-300, 1) / 2, dfv)
    return np.abs(m.stat.to_numpy()) / np.maximum(tval, 1e-12), dfv


def coverage(run: pd.DataFrame, base: pd.DataFrame, planted: list, df: int, level: float = 0.95) -> dict:
    """Coverage of the reference-label estimate (the target a label-error correction aims at) by the run's
    interval, stat +/- t_{df} * se with se = |stat| / t(p), over all estimable claims and over claims whose index
    type is a planted receiver (where misattributed effects live). Also misattr_typeI_05: among claims in planted
    receivers on planted genes with a non-sender neighbor, the share with p < 0.05 whose interval excludes the
    reference estimate (wrong attribution not explained by the reference-label analysis itself)."""
    from scipy import stats as st
    spikes = [r for r in planted if r.get("kind") == "spike"]
    cols = [c for c in ("claim", "stat", "p", "se", "df") if c in run]
    bcols = ["claim", "stat"] + (["stat_avg"] if "stat_avg" in base else [])
    m = run[cols].dropna(subset=["stat", "p"]).merge(base[bcols], on="claim", suffixes=("", "_ref"))
    if m.empty:
        return {}
    se, dfv = _se_df(m, df)
    tcrit = st.t.ppf(0.5 + level / 2, np.maximum(dfv, 1))
    cov = np.abs(m.stat.to_numpy() - m.stat_ref.to_numpy()) <= tcrit * se + 1e-15
    if "stat_avg" in m:                       # averaged target and standardised bias
        tgt = m.stat_avg.to_numpy()
        cov_avg = np.abs(m.stat.to_numpy() - tgt) <= tcrit * se + 1e-15
        sbias = np.sign(tgt) * (m.stat.to_numpy() - tgt) / np.maximum(se, 1e-300)
    else:
        cov_avg = sbias = None
    parts = m.claim.str.split("|", expand=True)
    out = {"cover_all": float(cov.mean())}
    if spikes:
        recv = parts[0].isin({r["receiver"] for r in spikes}).to_numpy()
        if recv.any():
            out["cover_recv"] = float(cov[recv].mean())
        pg = np.zeros(len(m), bool)
        for r in spikes:
            pg |= (parts[0] == r["receiver"]).to_numpy() & parts[2].isin(r["genes"]).to_numpy() & \
                  (parts[1] != r["sender"]).to_numpy()
        senders = {(r["receiver"], g): r["sender"] for r in spikes for g in r["genes"]}
        if pg.any():
            out["misattr_typeI_05"] = float(((m.p.to_numpy() < 0.05) & ~cov)[pg].mean())
            out["n_misattr"] = int(pg.sum())
            out["cover_misattr"] = float(cov[pg].mean())
        pl = np.zeros(len(m), bool)
        for r in spikes:
            pl |= (parts[0] == r["receiver"]).to_numpy() & parts[2].isin(r["genes"]).to_numpy() & \
                  (parts[1] == r["sender"]).to_numpy()
        if pl.any():
            out["cover_planted"] = float(cov[pl].mean())                 # the claims that carry the effects
            out["n_planted_est"] = int(pl.sum())
            if cov_avg is not None:
                out["cover_planted_avg"] = float(cov_avg[pl].mean())
                out["sbias_planted_median"] = float(np.nanmedian(sbias[pl]))
        if pg.any() and cov_avg is not None:
            out["cover_misattr_avg"] = float(cov_avg[pg].mean())
        nul = ~(pg | pl)                                                   # null class: every claim except planted genes in their receivers
        if nul.any():
            out["cover_null"] = float(cov[nul].mean())
            if cov_avg is not None:
                out["cover_null_avg"] = float(cov_avg[nul].mean())
    return out


def threshold_free(run: pd.DataFrame, base: pd.DataFrame, planted: list, null: bool,
                   thresholds=(1e-3, 1e-5)) -> dict:
    """Scoring that does not depend on a model- or run-specific threshold.

    auprc, pauroc_01: average precision and McClish-standardised partial AUROC (FPR <= 0.01) of the signed score
        sign(stat) * -log10(p) over all claims, positives = planted claims (in their planted direction).
    power_p<k>, fp_p<k>: planted claims called in the right direction / other claims called, at raw p < 10^-k.
    Null calibration (only on a null tissue), over claims whose index type receives no planting, whose true
        coefficient is 0 by construction: null_typeI_05, null_typeI_001 (share with p below 0.05 / 0.001; 1 minus
        the coverage of the 95% / 99.9% interval), null_rmse (root mean square effect), n_null.
    planted_bias: median over planted claims of stat / stat with reference labels (1 = unbiased)."""
    spikes = [r for r in planted if r.get("kind") == "spike"]
    if not spikes or "p" not in run:
        return {}
    pos = {f"{r['receiver']}|{r['sender']}|{g}": r.get("sign", 1) for r in spikes for g in r["genes"]}
    p = run.p.fillna(1.0).clip(lower=1e-300).to_numpy()
    st = run.stat.fillna(0).to_numpy()
    cl = run.claim.to_numpy()
    tsign = np.array([pos.get(c, 0) for c in cl])
    y = tsign != 0
    score = np.where(y, tsign, 1) * np.sign(st) * -np.log10(p)        # planted: score in the planted direction
    score = np.where(y, score, np.abs(score))                         # others: any direction counts as a call
    out = {}
    if y.any() and (~y).any():
        from sklearn.metrics import average_precision_score, roc_auc_score
        out["auprc"] = float(average_precision_score(y, score))
        out["pauroc_01"] = float(roc_auc_score(y, score, max_fpr=0.01))
    for thr in thresholds:
        k = int(round(-np.log10(thr)))
        called = p < thr
        out[f"power_p{k}"] = float((called & y & (np.sign(st) == tsign)).sum() / max(y.sum(), 1))
        out[f"fp_p{k}"] = int((called & ~(y & (np.sign(st) == tsign))).sum())
    if null:
        # null = every claim except planted genes in their receivers; the 0.5 definition
        # (index types without planting) is kept as null_typeI_05_nonrecv, and non-planted genes in receivers are
        # reported separately (recv_nonplanted_typeI_05) so that a normalisation artifact stays visible.
        C = claim_classes(cl, planted)
        est = run.p.notna().to_numpy().copy()             # validity metrics on estimable claims only
        if "estimable" in run:
            est &= run.estimable.to_numpy(bool)
        C["nonrecv"] &= est
        C["recv_nonplanted"] &= est
        nm = (C.cls == "null_claim").to_numpy() & est
        for key, mask in (("", nm), ("_nonrecv", C.nonrecv.to_numpy()), ("recv_nonplanted", C.recv_nonplanted.to_numpy())):
            if not mask.any():
                continue
            name = "null_typeI_05" + key if key != "recv_nonplanted" else "recv_nonplanted_typeI_05"
            out[name] = float((p[mask] < 0.05).mean())
        if nm.any():
            out["null_typeI_001"] = float((p[nm] < 0.001).mean())
            eff = run.effect.to_numpy()[nm] if "effect" in run else st[nm]
            out["null_rmse"] = float(np.sqrt(np.nanmean(eff ** 2)))
            out["n_null"] = int(nm.sum())
    if "stat" in base:
        bm = base.set_index("claim").stat
        pc = [c for c in cl[y] if c in bm.index]
        if pc:
            r = run.set_index("claim").stat.reindex(pc).to_numpy() / bm.reindex(pc).to_numpy()
            out["planted_bias"] = float(np.nanmedian(r))
    return out


def fixed_threshold(run: pd.DataFrame, base: pd.DataFrame, planted: list, null: bool,
                    min_effect: float = 0.0) -> dict:
    """Rescore a run at the baseline's p-value threshold (the largest raw p among baseline calls), so that
    Benjamini-Hochberg cannot become more lenient when a perturbation floods the run with calls.
    Returns p_thr_base, p_thr_run (largest raw p among the run's own calls), n_fixed, fp_fixed, fdr_fixed,
    power_fixed and power_fixed_<i> per spike (suffix _fixed on truth_detail keys)."""
    if "p" not in run or "p" not in base:
        return {}
    bc = base[base.decision != 0]
    thr = float(bc.p.max()) if len(bc) else 0.0
    rc = run[run.decision != 0]
    out = {"p_thr_base": thr, "p_thr_run": float(rc.p.max()) if len(rc) else 0.0}
    eff = run.effect.to_numpy() if "effect" in run else np.ones(len(run))
    dec = np.where((run.p.to_numpy() <= thr) & (np.abs(eff) >= min_effect), np.sign(run.stat.to_numpy()), 0)
    r2 = run.assign(decision=dec)
    out["n_fixed"] = int((dec != 0).sum())
    truth = {(f"{r['receiver']}|{r['sender']}|{g}", r.get("sign", 1)) for r in planted if r.get("kind") == "spike"
             for g in r["genes"]}
    vt = versus_truth(r2, truth, null)
    out.update({f"{k}_fixed": v for k, v in vt.items() if k in ("fp", "fdr", "power")})
    out.update({f"{k}_fixed": v for k, v in truth_detail(r2, planted).items() if k.startswith("power_")})
    return out


def versus_truth(run: pd.DataFrame, truth: set, null: bool) -> dict:
    """FDR and power against planted truth. With a null tissue and no spike-ins every call is false."""
    called = {(c, int(d)) for c, d in zip(run.claim, run.decision) if d != 0}
    if not truth and not null:
        return {}
    tp = len(called & truth)
    fp = len(called - truth)
    fn = len(truth - called)
    return {"tp": tp, "fp": fp, "fn": fn,
            "fdr": fp / (tp + fp) if (tp + fp) else 0.0,
            "power": tp / (tp + fn) if (tp + fn) else np.nan}


def break_points(decisions: pd.DataFrame, base: pd.DataFrame, native_rate: float | None = None,
                 threshold: float = 0.5, descending: bool = False) -> pd.DataFrame:
    """Per baseline claim and perturbation spec: the first level, in order of increasing severity, at which the
    claim survives in fewer than `threshold` of replicates (inf: never). Severity rises with the level for error
    rates (descending=False) and falls with it for depths (recall, thinning: descending=True).
    decisions: long table with spec, level, rep, question, claim, decision (only rows with a non-zero decision or
    a non-zero baseline decision need to be present). Baseline support and effect are carried over.
    """
    cols = ["question", "claim", "decision"] + [c for c in ("support", "effect") if c in base]
    bpos = base[base.decision != 0][cols].rename(columns={"decision": "d0"})
    rows = []
    for (spec, q), g in decisions.groupby(["spec", "question"]):
        levels = np.sort(g.level.unique())
        if descending:
            levels = levels[::-1]
        reps = g.groupby("level").rep.nunique()
        bq = bpos[bpos.question == q]
        if bq.empty:
            continue
        piv = g.merge(bq, on=["question", "claim"])
        piv = piv[piv.decision == piv.d0].groupby(["claim", "level"]).size().unstack(fill_value=0)
        piv = piv.reindex(index=bq.claim, columns=levels, fill_value=0)
        frac = piv / reps.reindex(levels).to_numpy()[None, :]
        broken = frac < threshold
        B = broken.to_numpy()
        first = np.where(B.any(1), levels[np.argmax(B, 1)], np.inf)
        df = pd.DataFrame({"spec": spec, "question": q, "claim": bq.claim.to_numpy(), "break_level": first})
        for c in ("support", "effect"):
            if c in bq:
                df[f"base_{c}"] = bq[c].to_numpy()
        if native_rate:
            df["fragility"] = df.break_level / native_rate
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def break_summary(bp: pd.DataFrame, flip_rate: float | None = None, n_bins: int = 4) -> pd.DataFrame:
    """Break points stratified by claim support (quartiles within each spec x question), so that rarity is not
    read as fragility. Columns: n claims, support range, share ever broken, share broken at or below the flip
    rate (label-error arms), median finite break level."""
    rows = []
    if bp.empty or "spec" not in bp:
        return pd.DataFrame(rows)
    for (spec, q), g in bp.groupby(["spec", "question"]):
        g = g.copy()
        if "base_support" in g and g.base_support.notna().sum() >= n_bins:
            g["bin"] = pd.qcut(g.base_support.rank(method="first"), n_bins, labels=False) + 1
        else:
            g["bin"] = 0
        for b, h in g.groupby("bin"):
            fin = h.break_level[np.isfinite(h.break_level)]
            r = {"spec": spec, "question": q, "support_bin": int(b), "n": len(h),
                 "support_min": h.base_support.min() if "base_support" in h else np.nan,
                 "support_max": h.base_support.max() if "base_support" in h else np.nan,
                 "broken": float(np.isfinite(h.break_level).mean()),
                 "median_break": float(fin.median()) if len(fin) else np.inf}
            if flip_rate and "fragility" in h and h.fragility.notna().any():
                r["broken_within_flip"] = float((h.fragility <= 1).mean())
            rows.append(r)
    return pd.DataFrame(rows)


def explain(decisions: pd.DataFrame, base: pd.DataFrame, question: str, spec: str | None = None,
            level: float | None = None) -> pd.DataFrame:
    """Which claims changed: per claim (and spec, level), the share of replicates in which it was gained (absent
    at baseline, present in the run), lost, or flipped, with baseline and run effect and support.

    decisions: the decisions_<spec>.csv.gz tables; base: baseline_claims.csv.
    """
    d = decisions[decisions.question == question]
    if d.empty:
        return pd.DataFrame(columns=["spec", "level", "claim", "gained", "lost", "flipped", "sign", "base_effect",
                                     "run_effect", "base_support"])
    if spec is not None:
        d = d[d.spec == spec]
    if level is not None:
        d = d[np.isclose(d.level, level)]
    b = base[base.question == question].set_index("claim")
    reps = d.groupby(["spec", "level"]).rep.nunique().rename("n_reps")
    d = d.join(b.decision.rename("d0"), on="claim")
    d["d0"] = d.d0.fillna(0).astype(int)
    d["gained"] = (d.d0 == 0) & (d.decision != 0)
    d["flipped"] = (d.d0 != 0) & (d.decision != 0) & (d.decision != d.d0)
    g = d.groupby(["spec", "level", "claim"]).agg(gained=("gained", "sum"), flipped=("flipped", "sum"),
                                                 present=("decision", lambda x: (x != 0).sum()),
                                                 sign=("decision", lambda x: int(np.sign(x.sum()))),
                                                 run_effect=("effect", "mean"), run_support=("support", "mean"))
    g = g.join(reps, on=["spec", "level"]).reset_index()
    base_pos = b.index[b.decision != 0]
    # lost: baseline claims absent (decision 0 or not recorded) in a replicate
    lost_rows = []
    for (sp, lv), n in reps.items():
        kept = d[(d.spec == sp) & np.isclose(d.level, lv) & (d.d0 != 0) & (d.decision == d.d0)]
        k = kept.groupby("claim").rep.nunique()
        for c in base_pos:
            lost_rows.append({"spec": sp, "level": lv, "claim": c, "lost": n - int(k.get(c, 0))})
    L = pd.DataFrame(lost_rows)
    out = g.merge(L, on=["spec", "level", "claim"], how="outer") if len(L) else g
    out = out.join(b[["decision", "effect", "support"]].rename(
        columns={"decision": "base_decision", "effect": "base_effect", "support": "base_support"}), on="claim")
    out["n_reps"] = out.groupby(["spec", "level"]).n_reps.transform("max")
    for c in ("gained", "flipped", "lost"):
        if c not in out:
            out[c] = 0.0
        out[c] = out[c].fillna(0) / out.n_reps
    out = out[(out.gained > 0) | (out.flipped > 0) | (out.lost > 0)]
    return out.sort_values(["spec", "level", "gained", "lost"], ascending=[True, True, False, False])
