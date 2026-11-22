"""Do label errors change the conclusions an analyst draws, and would the field's stress test notice?

For every section, kNN neighborhood enrichment (k = 10, fixed graph from the cell positions) is computed from two
label sets of the same cells: a reference (default `nucleus`, labels from the counts over the nucleus) and the
analyst's labels (default `whole`, whole-cell counts). For each ordered pair (a, b) the readouts are log2 observed
over expected (as `compartment consequence`) and a permutation z-score (labels permuted within the section, as in
squidpy's nhood_enrichment), and the call: depleted, neither or enriched, at |z| >= z_thr (`call_z`) and, in
addition, |log2 O/E| >= min_effect (`call`; with large sections nearly every z is significant).

The analyst's call is compared with the reference call:
  manufactured  reference neither, analyst depleted or enriched
  exaggerated   same call, |log2 O/E| larger for the analyst
  attenuated    same call, smaller
  lost          reference depleted or enriched, analyst neither
  reversed      opposite calls
  neither       both neither

The field's stress test is then run on the analyst's labels, as published robustness tests do: a fraction eps of
cells (0.05, 0.10, 0.15) is reassigned at random (`uniform`: to any other type with equal probability;
`composition`: in proportion to the other types' frequencies), `reps` times; the claim counts as robust at eps if
the call is unchanged in at least `robust` (0.9) of the draws. The mean shift of log2 O/E under random relabeling
says in which direction the test believes error acts (toward zero: the observed effect is conservative).

A claim is "certified wrongly" when the analyst's call is manufactured or exaggerated relative to the reference and
the random stress test calls it robust while moving it toward zero: the test says error can only weaken the claim,
whereas the errors that produced it strengthened it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .placement import unit_from

CALL = {-1: "depleted", 0: "neither", 1: "enriched"}


def edge_index(xy, k):
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = idx[:, 1:]
    return np.repeat(np.arange(len(xy)), k), idx.ravel(), idx


def log2oe(codes, src, dst, K, k):
    n = len(codes)
    M = np.bincount(codes[src] * K + codes[dst], minlength=K * K).reshape(K, K).astype(float)
    na = np.bincount(codes, minlength=K).astype(float)
    exp = k * na[:, None] * (na[None, :] - np.eye(K)) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(M / exp), M


def enrichment(codes, src, dst, K, k, nperm, rng):
    """log2 O/E and permutation z of the edge counts (labels permuted within the section)."""
    L, M = log2oe(codes, src, dst, K, k)
    P = np.empty((nperm, K * K))
    for p in range(nperm):
        c = rng.permutation(codes)
        P[p] = np.bincount(c[src] * K + c[dst], minlength=K * K)
    sd = P.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ((M.ravel() - P.mean(0)) / np.where(sd > 0, sd, np.nan)).reshape(K, K)
    return L, z


def call_of(L, z, z_thr, min_effect):
    c = np.where(z >= z_thr, 1, np.where(z <= -z_thr, -1, 0))
    if min_effect > 0:
        c = np.where(np.abs(np.nan_to_num(L)) >= min_effect, c, 0)
    return np.where(np.isfinite(z), c, 0)


def status_of(c_ref, c_test, L_ref, L_test):
    if c_ref == 0 and c_test == 0:
        return "neither"
    if c_ref == 0:
        return "manufactured"
    if c_test == 0:
        return "lost"
    if c_ref != c_test:
        return "reversed"
    return "exaggerated" if abs(L_test) > abs(L_ref) else "attenuated"


def random_relabel(codes, eps, K, rng, mode="uniform", freq=None):
    out = codes.copy()
    m = int(round(eps * len(codes)))
    if m == 0:
        return out
    idx = rng.choice(len(codes), m, replace=False)
    if mode == "uniform":
        new = (codes[idx] + rng.integers(1, K, m)) % K
    else:
        p = np.asarray(freq, float)
        new = np.empty(m, np.int64)
        for a in np.unique(codes[idx]):
            w = idx[codes[idx] == a]
            q = p.copy()
            q[a] = 0
            new[np.isin(idx, w)] = rng.choice(K, len(w), p=q / q.sum())
    out[idx] = new
    return out


def calls(labels_dir, out, level: str, ref_arm: str = "nucleus", test_arm: str = "whole", immune=(), partners=(),
          k: int = 10, nperm: int = 100, nperm_stress: int = 50, z_thr: float = 2.0, min_effect: float = 0.2,
          eps=(0.05, 0.10, 0.15), reps: int = 10, mode: str = "uniform", robust: float = 0.9,
          min_cells: int = 200, unit_pattern: str = r"I0\d{3}", use_sections=None, seed: int = 20261014,
          verbose: bool = True):
    """Write sections.csv (one row per section and ordered pair), units.csv (medians per unit) and summary.csv
    (per pair over units) into OUT; returns the three tables."""
    say = print if verbose else (lambda *a, **k: None)
    out = Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    files = sorted(Path(labels_dir).expanduser().glob("*.labels.csv.gz"))
    if use_sections is not None:
        p = Path(use_sections).expanduser()
        if p.suffix == ".csv":
            t = pd.read_csv(p)
            names = set(t.loc[t["used"].astype(bool), "sample"].astype(str)) if "used" in t else set(t["sample"])
        else:
            names = {ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
        files = [f for f in files if f.name.split(".labels")[0] in names]
    if not files:
        raise FileNotFoundError(f"no labels files to use in {labels_dir}")
    immune, partners = list(immune), list(partners)
    rng = np.random.default_rng(seed)
    cr, ct = f"{level}_{ref_arm}", f"{level}_{test_arm}"
    rows = []
    started = datetime.now(timezone.utc).isoformat()
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns
        miss = [c for c in (cr, ct, "x", "y") if c not in cols]
        if miss:
            raise KeyError(f"{f.name}: missing columns {miss}")
        use = ["sample", "x", "y", cr, ct] + [c for c in ("unit", "donor") if c in cols]
        D = pd.read_csv(f, usecols=lambda c: c in use, keep_default_na=False, low_memory=False)
        D = D[(D[cr].astype(str) != "") & (D[ct].astype(str) != "")].reset_index(drop=True)
        sample = str(D["sample"].iloc[0]) if "sample" in D else f.name.split(".labels")[0]
        fb = D["unit"].iloc[0] if "unit" in D else (D["donor"].iloc[0] if "donor" in D else None)
        unit = unit_from(sample, unit_pattern, fb)
        types = np.array(sorted(set(D[cr].astype(str)) | set(D[ct].astype(str))))
        K = len(types)
        code = {t: i for i, t in enumerate(types)}
        R = D[cr].astype(str).map(code).to_numpy()
        T = D[ct].astype(str).map(code).to_numpy()
        src, dst, _ = edge_index(D[["x", "y"]].to_numpy(float), k)
        Lr, zr = enrichment(R, src, dst, K, k, nperm, rng)
        Lt, zt = enrichment(T, src, dst, K, k, nperm, rng)
        nr, nt = np.bincount(R, minlength=K), np.bincount(T, minlength=K)
        cr_z, ct_z = call_of(Lr, zr, z_thr, 0), call_of(Lt, zt, z_thr, 0)
        cr_e, ct_e = call_of(Lr, zr, z_thr, min_effect), call_of(Lt, zt, z_thr, min_effect)
        freq = nt / nt.sum()
        stress = {}
        for e in eps:
            same_z, same_e, dL = np.zeros((K, K)), np.zeros((K, K)), np.zeros((K, K))
            for _ in range(reps):
                Ts = random_relabel(T, e, K, rng, mode, freq)
                Ls, zs = enrichment(Ts, src, dst, K, k, nperm_stress, rng)
                same_z += call_of(Ls, zs, z_thr, 0) == ct_z
                same_e += call_of(Ls, zs, z_thr, min_effect) == ct_e
                dL += np.nan_to_num(Ls - Lt)
            stress[e] = (same_z / reps, same_e / reps, dL / reps)
        keep = (nr >= min_cells) & (nt >= min_cells)
        pairs = [(i, p, "immune-partner") for i in immune for p in partners if i in code and p in code]
        pairs += [(t, t, "homotypic") for t in types]
        if not immune:
            pairs = [(a, b, "homotypic" if a == b else "other") for a in types for b in types]
        for ta, tb, group in pairs:
            i, j = code[ta], code[tb]
            if not (keep[i] and keep[j]):
                continue
            r = dict(section=sample, unit=unit, a=ta, b=tb, group=group, n_a_ref=int(nr[i]), n_b_ref=int(nr[j]),
                     L_ref=Lr[i, j], z_ref=zr[i, j], L_test=Lt[i, j], z_test=zt[i, j],
                     call_ref_z=CALL[int(cr_z[i, j])], call_test_z=CALL[int(ct_z[i, j])],
                     call_ref=CALL[int(cr_e[i, j])], call_test=CALL[int(ct_e[i, j])],
                     status_z=status_of(cr_z[i, j], ct_z[i, j], Lr[i, j], Lt[i, j]),
                     status=status_of(cr_e[i, j], ct_e[i, j], Lr[i, j], Lt[i, j]),
                     disagreement=float(np.mean(R != T)))
            for e, (sz, se, d) in stress.items():
                tag = f"{int(round(100 * e))}"
                r[f"persist_z_{tag}"] = sz[i, j]
                r[f"persist_{tag}"] = se[i, j]
                r[f"dL_random_{tag}"] = d[i, j]
            emax = f"{int(round(100 * max(eps)))}"
            toward0 = np.sign(r[f"dL_random_{emax}"]) == -np.sign(r["L_test"])
            r["random_toward_zero"] = bool(toward0)
            r["random_robust"] = bool(r[f"persist_{emax}"] >= robust) and r["call_test"] != "neither"
            r["certified_wrongly"] = r["status"] in ("manufactured", "exaggerated") and r["random_robust"] and \
                bool(toward0)
            rows.append(r)
        say(f"{sample} ({unit}): {len(D):,} cells, disagreement {np.mean(R != T):.3f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    S = pd.DataFrame(rows)
    S.to_csv(out / "sections.csv", index=False)
    num = [c for c in S.columns if c.startswith(("L_", "z_", "persist", "dL_"))]
    U = S.groupby(["a", "b", "group", "unit"])[num].median().reset_index()
    U["sections"] = S.groupby(["a", "b", "group", "unit"]).size().to_numpy()
    # unit-level calls and status from the unit medians
    for side in ("ref", "test"):
        U[f"call_{side}"] = [CALL[int(c)] for c in call_of(U[f"L_{side}"].to_numpy(), U[f"z_{side}"].to_numpy(),
                                                           z_thr, min_effect)]
    inv = {v: k_ for k_, v in CALL.items()}
    U["status"] = [status_of(inv[a], inv[b], x, y) for a, b, x, y in zip(U.call_ref, U.call_test, U.L_ref, U.L_test)]
    emax = f"{int(round(100 * max(eps)))}"
    U["random_toward_zero"] = np.sign(U[f"dL_random_{emax}"]) == -np.sign(U["L_test"])
    U["random_robust"] = (U[f"persist_{emax}"] >= robust) & (U.call_test != "neither")
    U["certified_wrongly"] = U.status.isin(["manufactured", "exaggerated"]) & U.random_robust & U.random_toward_zero
    U.to_csv(out / "units.csv", index=False)
    g = U.groupby(["a", "b", "group"])
    summ = g[["L_ref", "L_test", f"dL_random_{emax}", f"persist_{emax}"]].median()
    summ["units"] = g.size()
    for st in ("manufactured", "exaggerated", "attenuated", "lost", "reversed", "neither"):
        summ[st] = g["status"].apply(lambda v, st=st: int((v == st).sum()))
    summ["certified_wrongly"] = g["certified_wrongly"].sum().astype(int)
    summ["random_toward_zero"] = g["random_toward_zero"].sum().astype(int)
    summ = summ.reset_index().sort_values(["group", "a", "b"])
    summ.to_csv(out / "summary.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(
        started=started, finished=datetime.now(timezone.utc).isoformat(), labels=str(labels_dir), level=level,
        ref_arm=ref_arm, test_arm=test_arm, immune=immune, partners=partners, k=k, nperm=nperm,
        nperm_stress=nperm_stress, z_thr=z_thr, min_effect=min_effect, eps=list(eps), reps=reps, mode=mode,
        robust=robust, min_cells=min_cells, unit_pattern=unit_pattern, seed=seed,
        sections=sorted(S.section.unique().tolist()) if len(S) else []), indent=1))
    return S, U, summ
