"""Error kernel per dataset.

The fragility map writes every claim as theta_obs = a * theta_true + beta, with beta predicted from two measured
kernels of the dataset. This module measures them; it makes no claims and has no decision rules.

Label kernel (label_kernel; from the destinations table of `compartment exposure`, with its kept counts):
for focal type a and partner b, by the share of a's border held by b (bins 0, (0,0.25], ..., (0.75,1]):
  kept ratio  kappa_ab(s) = kept_test / kept_null on the same cells (test wholeB, null nucB, focal nucA)
  K_ab        = kappa_ab(s > 0) / kappa_ab(0): the pair-resolved loss of a-labels where a touches b, which sets
                beta of contact and compartment claims (beta_2 ~ log K_ab)
  q_ab        excess a -> b rate (test minus null) against the share of b: slope (per unit share) and intercept

Expression kernel (spill_kernel; from the counts, X = whole cell and layers['nucleus']): cells typed by one label arm
(default the full-nucleus labels); neighbors are the k nearest within r um (default 6 within 30 um, the contact used
throughout). Sections are pooled per unit and panel. For receiving type A, partner P and gene
g, cells of A are binned by their number of P neighbors (0, 1, 2, 3+), and per bin the mean whole, nuclear and
outside-nucleus counts of g give cell-weighted slopes per P neighbor (s_W, s_N, s_O). With the producer's level
p = mean whole counts of g in P cells (P = the type with the highest mean):
  sigma   = s_W / p          share of a producer's g that appears in a touching receiver, per contact
  nu_b    = s_N / s_W        nuclear share of the borrowed RNA
  nu_own  = nuclear fraction of A's own markers (top genes by specificity of the type means)
  R_b     = nu_b / nu_own    the R of a gene that is wholly borrowed; f_b = (1 - R) / (1 - R_b) is then the borrowed
                             fraction of any gene with nuclear ratio R in A (class 7 of the map)
A row is a calibration pair (decoy = True) when A does not make g: A cells with no P neighbor hold at most `eps`
of the producer's level (rho <= eps). Only decoy rows estimate R_b cleanly; the others mix own and borrowed.
Also written: the R matrix of every gene in every type (r_matrix.csv; R = (N/W of g in A) / nu_own(A)).

Outputs of spill_kernel: spill_pairs.csv.gz (unit, gene, producer, receiver, per-bin cells and means, slopes, sigma,
nu_b, R_b, rho, decoy), decoy_pairs.csv.gz (its decoy rows), r_matrix.csv.gz (rows with >= 20 counts),
own_markers.csv, sections.csv, meta.json. kernel_summary pools them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .placement import unit_from

BINS = 4                      # 0, 1, 2, 3+ partner neighbors
SHARE_MID = {"0": 0.0, "(0,0.25]": 0.125, "(0.25,0.5]": 0.375, "(0.5,0.75]": 0.625, "(0.75,1]": 0.875}


# --------------------------------------------------------------------------- expression kernel

def _section_sums(A, lab, k: int, r: float, types):
    """Per section: sums over strata of whole and nuclear counts (strata x genes, dense float64) and per-stratum cells
    and summed partner counts. Strata: (receiver a, partner p, bin) for a != p, then ("iso", a), then ("all", a)."""
    n = A.n_obs
    T = len(types)
    tix = {t: i for i, t in enumerate(types)}
    code = np.array([tix.get(x, -1) for x in lab])
    xy = A.obs[["x", "y"]].to_numpy(float)
    d, j = cKDTree(xy).query(xy, k=min(k + 1, n), distance_upper_bound=r)
    j = j[:, 1:]
    valid = j < n
    nb = np.where(valid, code[np.where(valid, j, 0)], -1)
    M = np.zeros((n, T), np.int32)                         # partner counts per type
    for t in range(T):
        M[:, t] = (nb == t).sum(1)
    het = np.where(code >= 0, M.sum(1) - M[np.arange(n), np.maximum(code, 0)], 0)
    n_pair = T * T * BINS
    rows, cols, msum = [], [], np.zeros(n_pair + 2 * T)
    lab_ok = np.where(code >= 0)[0]
    for p in range(T):
        c = lab_ok[code[lab_ok] != p]
        b = np.minimum(M[c, p], BINS - 1)
        sid = (code[c] * T + p) * BINS + b
        rows.append(c); cols.append(sid)
        np.add.at(msum, sid, M[c, p])
    iso = lab_ok[het[lab_ok] == 0]
    rows.append(iso); cols.append(n_pair + code[iso])
    rows.append(lab_ok); cols.append(n_pair + T + code[lab_ok])
    rr, cc = np.concatenate(rows), np.concatenate(cols)
    Ind = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n_pair + 2 * T))
    cells = np.asarray(Ind.sum(0)).ravel()
    Xw = sp.csr_matrix(A.X).astype(np.float64)
    Xn = sp.csr_matrix(A.layers["nucleus"]).astype(np.float64)
    SW = np.asarray((Ind.T @ Xw).todense())
    SN = np.asarray((Ind.T @ Xn).todense())
    return SW, SN, cells, msum, int((code >= 0).sum()), float(np.mean(M.sum(1)))


def _wslope(x, y, w):
    """Weighted least-squares slope and intercept; (nan, nan) when undetermined."""
    x, y, w = (np.asarray(v, float) for v in (x, y, w))
    ok = (w > 0) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(x[ok]) == 0:
        return np.nan, np.nan
    x, y, w = x[ok], y[ok], w[ok]
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    sxx = np.sum(w * (x - xm) ** 2)
    b = np.sum(w * (x - xm) * (y - ym)) / sxx
    return float(b), float(ym - b * xm)


def own_markers(iso_mean, types, genes, n_markers: int = 10, min_mean: float = 0.5):
    """Top genes per type by specificity of the type means: log2((mean in t + 0.01) / (max over other types +
    0.01)), genes with mean >= min_mean in t. Returns {type: [gene indices]}."""
    out = {}
    for t in range(len(types)):
        other = np.delete(iso_mean, t, axis=0)
        mx = other.max(0) if len(other) else np.zeros(iso_mean.shape[1])
        spec = np.log2((iso_mean[t] + 0.01) / (mx + 0.01))
        spec[iso_mean[t] < min_mean] = -np.inf
        idx = np.argsort(-spec)[:n_markers]
        out[types[t]] = [int(i) for i in idx if np.isfinite(spec[i]) and spec[i] > 1]
    return out


def _unit_tables(unit, acc, types, genes, n_markers, eps, min_cells, min_counts, min_prod):
    SW, SN, cells, msum = acc["SW"], acc["SN"], acc["cells"], acc["msum"]
    T, G = len(types), len(genes)
    n_pair = T * T * BINS
    iso_c, all_c = cells[n_pair:n_pair + T], cells[n_pair + T:]
    with np.errstate(invalid="ignore", divide="ignore"):
        iso_W = SW[n_pair:n_pair + T] / iso_c[:, None]
        iso_N = SN[n_pair:n_pair + T] / iso_c[:, None]
    iso_W = np.where(iso_c[:, None] >= min_cells, iso_W, np.nan)
    iso_N = np.where(iso_c[:, None] >= min_cells, iso_N, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        all_W = np.where(all_c[:, None] >= min_cells, SW[n_pair + T:] / all_c[:, None], np.nan)
        all_N = np.where(all_c[:, None] >= min_cells, SN[n_pair + T:] / all_c[:, None], np.nan)
    # markers and producers from all cells of a type (small types are rarely isolated)
    mk = own_markers(np.nan_to_num(all_W), types, genes, n_markers)
    nu_own = {}
    for t, idx in mk.items():
        ti = types.index(t)
        w, nn = SW[n_pair + T + ti, idx].sum(), SN[n_pair + T + ti, idx].sum()
        nu_own[t] = nn / w if w > 0 else np.nan
    mrows = [{"unit": unit, "type": t, "markers": ",".join(genes[i] for i in idx), "nu_own": nu_own[t],
              "iso_cells": int(iso_c[types.index(t)]), "cells": int(all_c[types.index(t)])} for t, idx in mk.items()]
    # R matrix
    allW, allN = SW[n_pair + T:], SN[n_pair + T:]
    tot = allW.sum(0)
    keepg = np.where(tot >= min_counts)[0]
    prod = np.full(G, -1)
    ok_all = np.isfinite(all_W).any(0)
    prod[ok_all] = np.nanargmax(np.where(np.isfinite(all_W[:, ok_all]), all_W[:, ok_all], -1), axis=0)
    rm = []
    for ti, t in enumerate(types):
        if all_c[ti] < min_cells:
            continue
        w, nn = allW[ti, keepg], allN[ti, keepg]
        with np.errstate(invalid="ignore", divide="ignore"):
            R = (nn / w) / nu_own[t]
            rho = np.where(prod[keepg] >= 0, all_W[ti, keepg] / all_W[np.maximum(prod[keepg], 0), keepg], np.nan)
        rm.append(pd.DataFrame({"unit": unit, "gene": np.array(genes)[keepg], "type": t, "cells": int(all_c[ti]),
                                "W": w, "N": nn, "R": R, "mean": all_W[ti, keepg], "iso_mean": iso_W[ti, keepg],
                                "producer": np.array(types + [""])[prod[keepg]], "rho": rho}))
    rmat = pd.concat(rm, ignore_index=True) if rm else pd.DataFrame()
    rmat = rmat[rmat.W >= 20] if len(rmat) else rmat
    # spill pairs: g with its producer p, every receiver a != p
    prow = []
    gsel = [g for g in keepg if prod[g] >= 0 and np.isfinite(all_W[prod[g], g]) and all_W[prod[g], g] >= min_prod]
    for g in gsel:
        p = prod[g]
        pW = all_W[p, g]
        for a in range(T):
            if a == p or not np.isfinite(nu_own.get(types[a], np.nan)):
                continue
            sid = (a * T + p) * BINS + np.arange(BINS)
            c = cells[sid]
            if c[1:].sum() < min_cells or c[0] < min_cells:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                mbar = np.where(c > 0, msum[sid] / c, np.nan)
                w_ = SW[sid, g] / c
                n_ = SN[sid, g] / c
            sW, iW = _wslope(mbar, w_, c)
            sN, iN = _wslope(mbar, n_, c)
            rho = w_[0] / pW            # receiver cells with no producer neighbor: its own level
            nu_b = sN / sW if sW > 0 else np.nan
            prow.append({"unit": unit, "gene": genes[g], "producer": types[p], "receiver": types[a],
                         **{f"cells{b}": int(c[b]) for b in range(BINS)},
                         **{f"whole{b}": float(w_[b]) for b in range(BINS)},
                         **{f"nuc{b}": float(n_[b]) for b in range(BINS)},
                         "m3": float(mbar[3]) if c[3] > 0 else np.nan,
                         "prod_whole": float(pW), "prod_nuc": float(all_N[p, g]),
                         "recv_iso_whole": float(iso_W[a, g]) if np.isfinite(iso_W[a, g]) else np.nan,
                         "rho": rho, "slope_whole": sW, "slope_nuc": sN, "slope_out": sW - sN if np.isfinite(sW) else np.nan,
                         "icpt_whole": iW, "sigma": sW / pW, "nu_b": nu_b, "nu_own": nu_own[types[a]],
                         "R_b": nu_b / nu_own[types[a]] if np.isfinite(nu_b) else np.nan,
                         "nu_prod": float(all_N[p, g] / pW) if pW > 0 else np.nan,
                         "decoy": bool(np.isfinite(rho) and rho <= eps)})
    return pd.DataFrame(prow), rmat, pd.DataFrame(mrows)


def spill_kernel(h5ad_dir, labels_dir, out, level: str = "L1", label_arm: str = "nucleus",
                 unit_pattern: str = r"I0\d{3}", k: int = 6, r: float = 30.0, samples: str = "",
                 n_markers: int = 10, eps: float = 0.02, min_cells: int = 30, min_counts: float = 50,
                 min_prod: float = 0.3, verbose: bool = True):
    """All labels files of a folder with their .h5ad (same stem) -> OUT tables (see module docstring)."""
    import anndata as ad
    hdir, ldir, out = Path(h5ad_dir).expanduser(), Path(labels_dir).expanduser(), Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; choose a new output folder")
    files = sorted(ldir.glob("*.labels.csv.gz"))
    if samples:
        keys = [s for s in samples.split(",") if s]
        files = [f for f in files if any(x in f.name for x in keys)]
    files = [f for f in files if (hdir / f"{f.name.split('.labels')[0]}.h5ad").exists()]
    if not files:
        raise FileNotFoundError(f"no labels files in {ldir} with an .h5ad in {hdir}")
    col = f"{level}_{label_arm}"
    # sections are pooled per unit and panel (lung and gut image one donor on several panels); the panel is
    # named by its gene count, with a suffix when two panels of the same size differ
    import hashlib
    groups, pnames = {}, {}
    for f in files:
        sample = f.name.split(".labels")[0]
        B = ad.read_h5ad(hdir / f"{sample}.h5ad", backed="r")
        g = tuple(B.var_names.astype(str))
        B.file.close()
        h = hashlib.md5("\n".join(g).encode()).hexdigest()
        if h not in pnames:
            same = [v for v in pnames.values() if v.split("_")[0] == f"p{len(g)}"]
            pnames[h] = f"p{len(g)}" + (f"_{len(same) + 1}" if same else "")
        groups.setdefault((unit_from(sample, unit_pattern), pnames[h]), {"genes": list(g), "files": []})["files"].append(f)
    D0 = pd.read_csv(files[0], keep_default_na=False, low_memory=False, usecols=lambda c: c in ("cell_id", col))
    if col not in D0:
        raise ValueError(f"{files[0].name}: no column {col}")
    types = set()
    for f in files:
        types |= set(pd.read_csv(f, keep_default_na=False, usecols=[col])[col].astype(str)) - {""}
    types = sorted(types)
    if verbose:
        print(f"{len(files)} sections in {len(groups)} unit x panel groups; panels "
              + ", ".join(sorted(set(pnames.values()))) + f"; {len(types)} types", flush=True)
    P, Rm, Mk, secs = [], [], [], []
    for (unit, panel), grp in groups.items():
        genes, acc = grp["genes"], None
        for f in grp["files"]:
            sample = f.name.split(".labels")[0]
            A = ad.read_h5ad(hdir / f"{sample}.h5ad")
            if "nucleus" not in A.layers:
                raise ValueError(f"{sample}.h5ad has no 'nucleus' layer")
            if list(A.var_names.astype(str)) != genes:
                raise ValueError(f"{sample}: genes differ from its panel group")      # cannot happen (grouped by hash)
            D = pd.read_csv(f, keep_default_na=False, low_memory=False, usecols=["cell_id", col], dtype={"cell_id": str})
            lab = D.set_index("cell_id")[col].reindex(A.obs_names.astype(str)).fillna("").astype(str).to_numpy()
            SW, SN, cells, msum, nl, mnb = _section_sums(A, lab, k, r, types)
            if acc is None:
                acc = {"SW": SW, "SN": SN, "cells": cells, "msum": msum}
            else:
                for key, v in (("SW", SW), ("SN", SN), ("cells", cells), ("msum", msum)):
                    acc[key] = acc[key] + v
            secs.append({"sample": sample, "unit": unit, "panel": panel, "genes": len(genes), "cells": int(A.n_obs),
                         "labelled": nl, "mean_neighbours": mnb})
            if verbose:
                print(f"{sample} ({unit}, {panel}): {A.n_obs:,} cells, {nl:,} labelled, "
                      f"{mnb:.2f} neighbours within {r:g} um (k {k})", flush=True)
            del A
        p_, r_, m_ = _unit_tables(unit, acc, types, genes, n_markers, eps, min_cells, min_counts, min_prod)
        for x in (p_, r_, m_):
            if len(x):
                x.insert(1, "panel", panel)
        P.append(p_); Rm.append(r_); Mk.append(m_)
        if verbose and len(p_):
            dz = p_[p_.decoy & p_.sigma.notna()]
            print(f"  {unit} {panel}: {len(p_):,} gene x receiver rows, {len(dz):,} calibration (decoy) rows; "
                  f"decoy median sigma {dz.sigma.median():.4f}, R_b {dz.R_b.median():.2f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    res = {"spill_pairs": pd.concat(P, ignore_index=True), "r_matrix": pd.concat(Rm, ignore_index=True),
           "own_markers": pd.concat(Mk, ignore_index=True), "sections": pd.DataFrame(secs)}
    for key, v in res.items():
        v.to_csv(out / (f"{key}.csv.gz" if key in ("spill_pairs", "r_matrix") else f"{key}.csv"), index=False)
    sp_ = res["spill_pairs"]
    sp_[sp_.decoy].to_csv(out / "decoy_pairs.csv.gz", index=False)     # what the harvest carries
    meta = dict(h5ad_dir=str(hdir), labels_dir=str(ldir), level=level, label_arm=label_arm, unit_pattern=unit_pattern,
                k=k, r=r, samples=samples, n_markers=n_markers, eps=eps, min_cells=min_cells, min_counts=min_counts,
                min_prod=min_prod, types=types, panels=sorted(set(pnames.values())),
                written=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return res


# --------------------------------------------------------------------------- label kernel

def label_kernel(exposure_dir, out=None, min_cells: int = 20):
    """From destinations.csv of `compartment exposure` (with kept counts) -> label_bins.csv (unit, a, b, bin:
    kept ratio, excess a -> b) and label_fit.csv (unit, a, b: kappa at 0, kappa touching, K, q slope and intercept).
    """
    src = Path(exposure_dir).expanduser()
    out = Path(out).expanduser() if out else src
    D = pd.read_csv(src / "destinations.csv", keep_default_na=False)
    if "kept_test" not in D:
        raise ValueError(f"{src}/destinations.csv has no kept counts; rerun the exposure")
    U = D.groupby(["unit", "level", "a", "b", "bin"])[["cells", "to_b_test", "to_b_null", "kept_test",
                                                       "kept_null"]].sum().reset_index()
    U["share_mid"] = U["bin"].map(SHARE_MID)
    U["kappa"] = U.kept_test / U.kept_null.replace(0, np.nan)
    U["excess"] = (U.to_b_test - U.to_b_null) / U.cells
    out.mkdir(parents=True, exist_ok=True)
    U.to_csv(out / "label_bins.csv", index=False)
    fits = []
    for (u, lv, a, b), g in U.groupby(["unit", "level", "a", "b"]):
        z, t = g[g["bin"] == "0"], g[(g["bin"] != "0") & g.share_mid.notna()]
        if z.kept_null.sum() < min_cells or t.kept_null.sum() < min_cells:
            continue
        k0 = z.kept_test.sum() / z.kept_null.sum()
        kt = t.kept_test.sum() / t.kept_null.sum()
        gg = g[(g.cells >= min_cells) & g.share_mid.notna()]
        slope, icpt = _wslope(gg.share_mid, gg.excess, gg.cells)
        fits.append({"unit": u, "level": lv, "a": a, "b": b, "cells_0": int(z.cells.sum()), "cells_touch": int(t.cells.sum()),
                     "kappa_0": k0, "kappa_touch": kt, "K": kt / k0 if k0 > 0 else np.nan,
                     "q_slope": slope, "q_icpt": icpt,
                     "touch_share": float(t.cells.sum() / g.cells.sum())})
    F = pd.DataFrame(fits)
    F.to_csv(out / "label_fit.csv", index=False)
    return U, F


# --------------------------------------------------------------------------- summary

def _wmed(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return np.nan
    o = np.argsort(x[ok])
    cw = np.cumsum(w[ok][o])
    return float(x[ok][o][np.searchsorted(cw, cw[-1] / 2)])


def _ratio_boot(g, num, den, draws: int, rng, by: str = "gene", q=(0.025, 0.975)):
    """Interval of sum(num) / sum(den) over rows of g, resampling whole `by` groups (genes) with replacement:
    rows of one gene share its counts, so genes, not rows, are the independent draws."""
    if draws <= 0 or not len(g):
        return np.nan, np.nan
    codes, uniq = pd.factorize(g[by])
    a = np.bincount(codes, weights=np.nan_to_num(num), minlength=len(uniq))
    b = np.bincount(codes, weights=np.nan_to_num(den), minlength=len(uniq))
    if len(uniq) < 2:
        return np.nan, np.nan
    C = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=draws)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (C @ a) / (C @ b)
    r = r[np.isfinite(r) & ((C @ b) > 0)]
    if not len(r):
        return np.nan, np.nan
    lo, hi = np.quantile(r, q)
    return float(lo), float(hi)


def kernel_summary(spill_dir=None, label_dir=None, out=None, dataset: str = "", min_contact_cells: int = 50,
                   boot: int = 1000, seed: int = 0):
    """Pool per dataset. Expression: over calibration (decoy) rows with at least min_contact_cells receiving cells
    touching the producer: sigma (median weighted by touching cells), nu_b and R_b (ratios of weighted slope sums),
    per unit, overall and by receiver type.
    Labels: per pair, the median over units of K and q_slope. Writes kernel_expr.csv, kernel_expr_types.csv,
    kernel_labels.csv into OUT; returns them.
    nu_b and R_b get 95% intervals (nu_b_lo/hi, R_b_lo/hi) from `boot` draws resampling genes, with the
    number of genes behind each row; a dataset of one unit (the renal carcinomas) has no other source of
    uncertainty. boot=0 skips them."""
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    if spill_dir:
        sd = Path(spill_dir).expanduser()
        P = pd.read_csv(sd / "decoy_pairs.csv.gz") if (sd / "decoy_pairs.csv.gz").exists() else pd.read_csv(sd / "spill_pairs.csv.gz")
        P["touch"] = P[[f"cells{b}" for b in range(1, BINS)]].sum(axis=1)
        dz = P[P.decoy & (P.touch >= min_contact_cells) & P.sigma.notna()]

        def pool(g):
            # sigma: weighted median over rows (all slopes, so noise does not bias it up); nu_b and R_b: ratios of
            # touch-weighted sums of the slopes (stable where single slopes are small), and medians of rows with s_W > 0
            w = g.touch.to_numpy(float)
            sW, sN = g.slope_whole.to_numpy(float), g.slope_nuc.to_numpy(float)
            pos = g[g.slope_whole > 0]
            nb = float(np.nansum(w * sN) / np.nansum(w * sW)) if np.nansum(w * sW) > 0 else np.nan
            rb = float(np.nansum(w * sN / g.nu_own.to_numpy(float)) / np.nansum(w * sW)) if np.nansum(w * sW) > 0 else np.nan
            rng = np.random.default_rng(seed)
            with np.errstate(invalid="ignore", divide="ignore"):
                nb_lo, nb_hi = _ratio_boot(g, w * sN, w * sW, boot, rng)
                rb_lo, rb_hi = _ratio_boot(g, w * sN / g.nu_own.to_numpy(float), w * sW, boot, rng)
            return {"rows": len(g), "genes": g.gene.nunique(), "sigma": _wmed(g.sigma, g.touch),
                    "nu_b_lo": nb_lo, "nu_b_hi": nb_hi, "R_b_lo": rb_lo, "R_b_hi": rb_hi,
                    "sigma_iqr_lo": float(g.sigma.quantile(0.25)), "sigma_iqr_hi": float(g.sigma.quantile(0.75)),
                    "nu_b": nb, "R_b": rb, "nu_b_med": _wmed(pos.nu_b, pos.touch), "R_b_med": _wmed(pos.R_b, pos.touch),
                    "nu_own_med": float(g.nu_own.median())}
        if "panel" not in dz:
            dz = dz.assign(panel="")
        rows = [{"dataset": dataset, "unit": u, "panel": pn, **pool(g)} for (u, pn), g in dz.groupby(["unit", "panel"])]
        if len(dz):
            rows.append({"dataset": dataset, "unit": "ALL", "panel": "", **pool(dz)})
        E = pd.DataFrame(rows)
        E.to_csv(out / "kernel_expr.csv", index=False)
        Tt = pd.DataFrame([{"receiver": rc, **pool(g)} for rc, g in dz.groupby("receiver")])
        if len(Tt):
            Tt.insert(0, "dataset", dataset)
        Tt.to_csv(out / "kernel_expr_types.csv", index=False)
        res["expr"], res["expr_types"] = E, Tt
    if label_dir:
        F = pd.read_csv(Path(label_dir).expanduser() / "label_fit.csv")
        L = F.groupby(["level", "a", "b"]).agg(units=("unit", "nunique"), K=("K", "median"), K_lo=("K", "min"),
                                               K_hi=("K", "max"), q_slope=("q_slope", "median"),
                                               q_icpt=("q_icpt", "median"), touch_share=("touch_share", "median"),
                                               cells_touch=("cells_touch", "sum")).reset_index()
        L.insert(0, "dataset", dataset)
        L.to_csv(out / "kernel_labels.csv", index=False)
        res["labels"] = L
    return res
