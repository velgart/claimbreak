"""Label loss against physical exposure to other cell types (analysis A and B).

Question: does a cell keep its label from whole-cell counts when no cell of another type touches it? If the loss of
T/NK labels comes from neighbors' transcripts, a T/NK cell surrounded only by its own type (or by nothing) keeps its
label as often as the counting-noise null does, and the loss grows with the number of touching cells of other types.
If it comes from the difference between nuclear and cytoplasmic RNA of the cell itself, the loss is the same at every
exposure.

Per cell, the focal type a is the label of one nuclear half (default nucA). Among cells of focal type a that are
labeled in the test (default wholeB) and in the null (default nucB), the test keeps a cell if it also calls it a,
and so does the null. The ratio kept_test / kept_null (same cells) is the kept ratio of the accompanying paper,
resolved by exposure. Neighbors are typed from their full nuclear counts (default arm nucleus) and found by
physical gaps between nuclei (claimbreak.geometry): for each gap g, m = nuclei within g and k = those of another
type. Two continuous views at the contact gap: the heterotypic border share and the RNA-weighted exposure.

Destinations (analysis B): for each pair (a, b), cells of focal type a binned by the share of their border held by
type-b neighbors; the rate of a -> b in the test and in the null. If a -> b changes come from b's transcripts, the
excess rate (test minus null) rises from about zero at share 0; an intercept above zero is a change toward b that
does not need b next to the cell.

Outputs (exposure_files): strata.csv, destinations.csv, dest_region.csv, notes.txt; exposure_summary adds
units.csv, dose.csv, tissue.csv, dest_units.csv, dest_fit.csv, dest_region_units.csv, dest_region_fit.csv.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as geo

BORDER_EDGES = [0.0, 1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9, 1.0 + 1e-9]
BORDER_NAMES = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1)", "1"]
EXPO_EDGES = [0.0, 1e-9, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf]
EXPO_NAMES = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,1]", "(1,2]", "(2,4]", ">4"]
SHARE_EDGES = [0.0, 1e-9, 0.25, 0.5, 0.75, 1.0 + 1e-9]
SHARE_NAMES = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1]"]
OUT_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9]
OUT_NAMES = ["o[0,0.2]", "o(0.2,0.4]", "o(0.4,0.6]", "o(0.6,0.8]", "o(0.8,1]"]
REGION_EDGES = [0.0, 1e-9, 0.1, 0.25, 0.5, 1.0 + 1e-9]
REGION_NAMES = ["0", "(0,0.1]", "(0.1,0.25]", "(0.25,0.5]", "(0.5,1]"]
REGION_MID = dict(zip(REGION_NAMES, [0.0, 0.05, 0.175, 0.375, 0.75]))


def _bin(x, edges, names, missing="none"):
    x = np.asarray(x, float)
    out = np.full(len(x), missing, dtype=object)
    ok = np.isfinite(x)
    k = np.searchsorted(edges, x[ok], side="left")
    # value 0 falls in the first bin; values in (e_{k-1}, e_k] in bin k-1
    k = np.where(x[ok] <= edges[0], 1, k)
    out[ok] = np.array(names, dtype=object)[np.clip(k - 1, 0, len(names) - 1)]
    return out


def unit_column(D: pd.DataFrame, pattern: str):
    """Per-cell unit: the pattern on the sample name first (as in `compartment summary`; the labels' own
    `unit` column can name a measurement rather than the patient, e.g. gut '337_Colon_I0355', CosMx 'Lung5_Rep1'),
    then the labels' `unit` column where present and non-empty, then the sample name."""
    sample = str(D["sample"].iloc[0])
    m = re.search(pattern, sample) if pattern else None
    if m:
        u = m.group("unit") if "unit" in m.groupdict() else m.group(0)
        return np.full(len(D), u, dtype=object)
    if "unit" in D:
        return D["unit"].astype(str).replace({"": sample, "nan": sample}).to_numpy()
    return np.full(len(D), sample, dtype=object)


def exposure_section(D: pd.DataFrame, levels, nucleus_area=None, focal: str = "nucA", test: str = "wholeB",
                     null: str = "nucB", nbr: str = "nucleus", gaps=(2.0, 4.0, 6.0), contact_gap: float = 2.0,
                     mcap: int = 8, bins: int = 72, unit_pattern: str = "", min_margin: float = 0.0,
                     regions=(10.0, 20.0, 30.0), boundaries=None, tol: float = 1.0):
    """One labels table (as written by `compartment label`). nucleus_area: per-cell nucleus area (um^2) aligned with
    D, or None. Returns (strata rows, destination rows, regional destination rows, notes).

    scheme "outside" (share of the cell's counts outside the nucleus, crossed with the heterotypic border
    share: h0, h(0,0.5], h(0.5,1], hnone); regional destinations: for cells with no type-b neighbor at the border,
    the a -> b rate against the share of type-b cells among all typed cells whose centroid lies within R um (R in
    `regions`); a rate rising with the regional share and starting near zero means RNA reaching the cell from b-cells
    that do not touch it by nucleus geometry (processes of elongated cells, diffuse RNA); a rate flat above zero means
    a change toward b that needs no b nearby (confusability, or what the cell's own compartments carry).

    boundaries, the section's segmentation polygons (cell_id, vertex_x, vertex_y, as Xenium
    cell_boundaries.parquet): border shares (schemes border, exposure, outside; destinations) from shared perimeter,
    the points of a cell's polygon within tol um of another cell's polygon (contacts.boundary_shares), instead of
    nuclear disks; the nucleus-gap and regional schemes are unchanged."""
    n = len(D)
    sample = str(D["sample"].iloc[0])
    unit = unit_column(D, unit_pattern)
    xy = D[["x", "y"]].to_numpy(float)
    r, note = geo.nucleus_radius(nucleus_area if nucleus_area is not None else np.full(n, np.nan))
    gmax = max(max(gaps), contact_gap)
    pi, pj, pgap, pd_ = geo.gap_pairs(xy, r, gmax)
    if boundaries is not None:
        from .contacts import boundary_shares
        ip, jp, sp_, covered = boundary_shares(boundaries, D["cell_id"].astype(str).to_numpy(), tol)
        border_source = f"polygons (tol {tol:g} um; {np.mean(covered > 0):.3f} of cells touch another)"
    else:
        ip, jp, sp_, covered = geo.border_shares(n, xy, r, pi, pj, pgap, pd_, contact_gap, bins)
        border_source = f"nuclear disks (gap {contact_gap:g} um)"
    # RNA content from nuclear counts (whole-cell counts grow with the mask and with contamination)
    mcol = "n_nucleus" if "n_nucleus" in D else ("n_whole" if "n_whole" in D else None)
    mass = pd.to_numeric(D[mcol], errors="coerce").fillna(0).to_numpy(float) if mcol else np.ones(n)
    nw = pd.to_numeric(D["n_whole"], errors="coerce").to_numpy(float) if "n_whole" in D else np.full(n, np.nan)
    nn = pd.to_numeric(D["n_nucleus"], errors="coerce").to_numpy(float) if "n_nucleus" in D else np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        outside = np.where(nw > 0, (nw - nn) / nw, np.nan)
    notes = {"sample": sample, "cells": n, "border": border_source, "radius_note": note,
             "median_radius": float(np.median(r)),
             "pairs_within_gmax": int(len(pi) // 2),
             "median_contacts": float(np.median(geo.counts_within(n, pi, pgap, contact_gap))),
             "covered_border_median": float(np.median(covered))}
    srows, drows, rrows = [], [], []
    for lv in levels:
        cols = [f"{lv}_{x}" for x in (focal, test, null, nbr)]
        if not all(c in D for c in cols):
            notes[f"{lv}_missing"] = [c for c in cols if c not in D]
            continue
        F, T, N, B = (D[c].astype(str).replace("nan", "").to_numpy() for c in cols)
        types = np.array(sorted((set(F) | set(T) | set(N) | set(B)) - {""}))
        tix = {t: i for i, t in enumerate(types)}
        code = lambda v: np.array([tix.get(x, -1) for x in v])    # noqa: E731
        fc, tc, nc, bc = code(F), code(T), code(N), code(B)
        el = (fc >= 0) & (tc >= 0) & (nc >= 0)
        if min_margin > 0 and f"{lv}_{focal}_margin" in D:
            el &= pd.to_numeric(D[f"{lv}_{focal}_margin"], errors="coerce").fillna(-1).to_numpy() >= min_margin
        frac, expo = geo.heterotypic(n, ip, jp, sp_, fc, bc, mass)
        base = pd.DataFrame({"unit": unit, "type": np.where(fc >= 0, types[np.maximum(fc, 0)], ""),
                             "kt": (tc == fc), "kn": (nc == fc)})[el]
        cols_out = ["unit", "type"]

        def emit(df, scheme, g=np.nan, extra=()):
            agg = df.groupby(cols_out + list(extra), observed=True).agg(
                cells=("kt", "size"), kept_test=("kt", "sum"), kept_null=("kn", "sum")).reset_index()
            for _, row in agg.iterrows():
                rr = {"sample": sample, "unit": row["unit"], "level": lv, "type": row["type"], "scheme": scheme,
                      "g": g, "m": row.get("m", np.nan), "k": row.get("k", np.nan), "bin": row.get("bin", ""),
                      "cells": int(row.cells), "kept_test": int(row.kept_test), "kept_null": int(row.kept_null)}
                srows.append(rr)

        emit(base, "all")
        for g in gaps:
            within = pgap <= g
            m_all = np.bincount(pi[within], minlength=n)
            het = within & (bc[pj] >= 0) & (fc[pi] >= 0) & (bc[pj] != fc[pi])
            k_het = np.bincount(pi[het], minlength=n)
            df = base.assign(m=np.minimum(m_all, mcap)[el], k=np.minimum(k_het, mcap)[el])
            emit(df, "gap", g, ("m", "k"))
        emit(base.assign(bin=_bin(frac, BORDER_EDGES, BORDER_NAMES)[el]), "border", contact_gap, ("bin",))
        emit(base.assign(bin=_bin(expo, EXPO_EDGES, EXPO_NAMES)[el]), "exposure", contact_gap, ("bin",))
        hb = np.where(~np.isfinite(frac), "hnone", np.where(frac <= 0, "h0", np.where(frac <= 0.5, "h(0,0.5]",
                                                                                       "h(0.5,1]")))
        ob = _bin(outside, OUT_EDGES, OUT_NAMES)
        emit(base.assign(bin=(ob.astype(str) + "|" + hb.astype(str))[el]), "outside", contact_gap, ("bin",))

        # destinations: a -> b against the share of the border held by b
        for bi, b in enumerate(types):
            shb = geo.type_share(n, ip, jp, sp_, bc, bi)
            sb = _bin(shb, SHARE_EDGES, SHARE_NAMES)
            for ai, a in enumerate(types):
                if ai == bi:
                    continue
                ca = el & (fc == ai)
                if not ca.any():
                    continue
                # kept_test / kept_null (the cell keeps its own type) by the share of type b, for the
                # pair-resolved kept ratio of the fragility map (claimbreak.kernel.label_kernel)
                df = pd.DataFrame({"unit": unit[ca], "bin": sb[ca], "tb": tc[ca] == bi, "nb": nc[ca] == bi,
                                   "ka": tc[ca] == ai, "kna": nc[ca] == ai})
                agg = df.groupby(["unit", "bin"]).agg(cells=("tb", "size"), to_b_test=("tb", "sum"),
                                                      to_b_null=("nb", "sum"), kept_test=("ka", "sum"),
                                                      kept_null=("kna", "sum")).reset_index()
                for _, row in agg.iterrows():
                    drows.append({"sample": sample, "unit": row["unit"], "level": lv, "a": a, "b": b,
                                  "bin": row["bin"], "cells": int(row.cells), "to_b_test": int(row.to_b_test),
                                  "to_b_null": int(row.to_b_null), "kept_test": int(row.kept_test),
                                  "kept_null": int(row.kept_null)})
            # regional destinations: cells without b at the border, by the regional share of b
            if regions:
                for R, shr in region_shares(xy, bc, len(types), regions, bi).items():
                    rb = _bin(shr, REGION_EDGES, REGION_NAMES)
                    for ai, a in enumerate(types):
                        if ai == bi:
                            continue
                        ca = el & (fc == ai) & (shb <= 0)
                        if not ca.any():
                            continue
                        df = pd.DataFrame({"unit": unit[ca], "bin": rb[ca], "tb": tc[ca] == bi, "nb": nc[ca] == bi})
                        agg = df.groupby(["unit", "bin"]).agg(cells=("tb", "size"), to_b_test=("tb", "sum"),
                                                              to_b_null=("nb", "sum")).reset_index()
                        for _, row in agg.iterrows():
                            rrows.append({"sample": sample, "unit": row["unit"], "level": lv, "a": a, "b": b,
                                          "radius": R, "bin": row["bin"], "cells": int(row.cells),
                                          "to_b_test": int(row.to_b_test), "to_b_null": int(row.to_b_null)})
    return pd.DataFrame(srows), pd.DataFrame(drows), pd.DataFrame(rrows), notes


def region_shares(xy, codes, K: int, radii, b: int):
    """Share of type-b cells among typed cells (codes >= 0) with centroid within R of each cell (the cell itself
    excluded), for each R in radii. Returns {R: array}."""
    from scipy.spatial import cKDTree
    codes = np.asarray(codes)
    typed = codes >= 0
    t_all = cKDTree(xy[typed])
    is_b = codes == b
    t_b = cKDTree(xy[is_b]) if is_b.any() else None
    out = {}
    for R in radii:
        tot = t_all.query_ball_point(xy, R, return_length=True) - typed.astype(int)
        nb = (t_b.query_ball_point(xy, R, return_length=True) - is_b.astype(int)) if t_b is not None \
            else np.zeros(len(xy), int)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[float(R)] = np.where(tot > 0, nb / np.maximum(tot, 1), np.nan)
    return out


def read_areas(h5ad_dir, sample: str, cell_ids):
    """Nucleus areas of a sample's cells from <h5ad_dir>/<sample>.h5ad (obs nucleus_area), aligned with cell_ids;
    None if the file or the column is missing."""
    if not h5ad_dir:
        return None, "no --h5ad-dir: default radius"
    f = Path(h5ad_dir).expanduser() / f"{sample}.h5ad"
    if not f.exists():
        return None, f"{f.name} not found: default radius"
    import anndata as ad
    A = ad.read_h5ad(f, backed="r")
    if "nucleus_area" not in A.obs:
        return None, f"{f.name}: no nucleus_area in obs: default radius"
    s = pd.Series(pd.to_numeric(A.obs["nucleus_area"], errors="coerce").to_numpy(float), index=A.obs_names.astype(str))
    A.file.close()
    a = s.reindex(pd.Index(np.asarray(cell_ids).astype(str))).to_numpy(float)
    return a, f"{f.name}: nucleus areas for {np.isfinite(a).mean():.3f} of cells"


def exposure_files(labels_dir, out, levels, h5ad_dir: str = "", samples: str = "", verbose: bool = True,
                   boundaries: str = "", **kw):
    """Every labels file of a folder -> OUT/strata.csv, destinations.csv, notes.txt (refuses a non-empty OUT).
    boundaries, a folder of segmentation polygons (<sample>_cell_boundaries.parquet, as for `compartment
    contacts --contact boundary`): border shares from shared perimeter; a section without a polygon file stops the
    run rather than silently falling back to nuclear disks."""
    src, out = Path(labels_dir).expanduser(), Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; choose a new output folder")
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.labels.csv.gz"))
    if samples:
        keys = samples.split(",")
        files = [f for f in files if any(k in f.name for k in keys)]
    S, Dd, Rr, lines = [], [], [], []
    for f in files:
        D = pd.read_csv(f, keep_default_na=False, low_memory=False, dtype={"cell_id": str})
        sample = str(D["sample"].iloc[0])
        area, anote = read_areas(h5ad_dir, f.name.replace(".labels.csv.gz", ""), D["cell_id"])
        bnd = None
        if boundaries:
            from .contacts import find_boundaries, read_parquet_any
            bp = find_boundaries(boundaries, f.name.replace(".labels.csv.gz", ""))
            if bp is None:
                raise FileNotFoundError(f"no cell_boundaries.parquet for {sample} in {boundaries}")
            bnd = read_parquet_any(bp)
        s, d, rg, notes = exposure_section(D, levels, area, boundaries=bnd, **kw)
        notes["areas"] = anote
        S.append(s)
        Dd.append(d)
        Rr.append(rg)
        line = "; ".join(f"{k} {v}" for k, v in notes.items() if v != "")
        lines.append(line)
        if verbose:
            print(line, flush=True)
    S = pd.concat(S, ignore_index=True) if S else pd.DataFrame()
    Dd = pd.concat(Dd, ignore_index=True) if Dd else pd.DataFrame()
    Rr = pd.concat(Rr, ignore_index=True) if Rr else pd.DataFrame()
    S.to_csv(out / "strata.csv", index=False)
    Dd.to_csv(out / "destinations.csv", index=False)
    Rr.to_csv(out / "dest_region.csv", index=False)
    (out / "notes.txt").write_text("\n".join(lines) + "\n")
    return S, Dd


def _wfit(x, y, w):
    """Weighted line y = b0 + b1 x on finite rows (regional bins of cells without typed neighbors within the
    radius have no midpoint and must not reach the solver). Returns (b0, b1) or None with fewer than two distinct x."""
    x, y, w = (np.asarray(v, float) for v in (x, y, w))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 2 or len(np.unique(x)) < 2:
        return None
    try:
        b = np.linalg.lstsq(np.c_[np.ones_like(x), x] * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return float(b[0]), float(b[1])


def _ratio(t, nn):
    t, nn = np.asarray(t, float), np.asarray(nn, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nn > 0, t / nn, np.nan)


def exposure_summary(exposure_dir, out=None, focus: str = r"(?i)^t.?nk$|^t$|t_?cell", min_kept: int = 20,
                     verbose: bool = True):
    """Pool strata per unit (sums over sections; ratio = kept_test / kept_null on the same cells) and summarise.
    Writes units.csv (per unit, type, scheme, g, m, k, bin), dose.csv (per unit, type, g: k pooled over m, and k
    within each m), tissue.csv (medians over units with at least min_kept null-kept cells), dest_units.csv and
    dest_fit.csv (per unit and pair: excess rate a -> b by border-share bin, and a weighted line through it)."""
    src = Path(exposure_dir).expanduser()
    out = Path(out).expanduser() if out else src
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(src / "strata.csv", keep_default_na=False, na_values=[""])
    key = ["unit", "level", "type", "scheme", "g", "m", "k", "bin"]
    S["bin"] = S["bin"].fillna("")
    U = S.groupby(key, dropna=False)[["cells", "kept_test", "kept_null"]].sum().reset_index()
    U["ratio"] = _ratio(U.kept_test, U.kept_null)
    U.to_csv(out / "units.csv", index=False)

    G = U[U.scheme == "gap"]
    crude = G.groupby(["unit", "level", "type", "g", "k"])[["cells", "kept_test", "kept_null"]].sum().reset_index()
    crude["m"] = "all"
    within = G.copy()
    within["m"] = within["m"].astype(int).astype(str)
    dose = pd.concat([crude, within[crude.columns]], ignore_index=True)
    dose["ratio"] = _ratio(dose.kept_test, dose.kept_null)
    dose.to_csv(out / "dose.csv", index=False)

    rows = []
    allu = U[U.scheme == "all"].set_index(["unit", "level", "type"])["ratio"]
    for (lv, t, sch, g, m, k, b), grp in pd.concat([
            U[U.scheme.isin(["border", "exposure"])].assign(m="", k=""),
            dose.assign(scheme="gap", bin="")], ignore_index=True).groupby(
            ["level", "type", "scheme", "g", "m", "k", "bin"], dropna=False):
        ok = grp[grp.kept_null >= min_kept]
        rows.append({"level": lv, "type": t, "scheme": sch, "g": g, "m": m, "k": k, "bin": b,
                     "units": len(ok), "median_ratio": ok.ratio.median(), "units_below_1": int((ok.ratio < 1).sum()),
                     "cells": int(grp.cells.sum()), "pooled_ratio": _ratio(grp.kept_test.sum(), grp.kept_null.sum())[()]})
    Tt = pd.DataFrame(rows)
    Tt.to_csv(out / "tissue.csv", index=False)

    dfile = src / "destinations.csv"
    if dfile.exists() and dfile.stat().st_size > 1:
        Dd = pd.read_csv(dfile, keep_default_na=False)
        du = Dd.groupby(["unit", "level", "a", "b", "bin"])[["cells", "to_b_test", "to_b_null"]].sum().reset_index()
        du["rate_test"] = du.to_b_test / du.cells
        du["rate_null"] = du.to_b_null / du.cells
        du["excess"] = du.rate_test - du.rate_null
        mid = dict(zip(SHARE_NAMES, [0.0, 0.125, 0.375, 0.625, 0.875]))
        du["share_mid"] = du["bin"].map(mid)
        du.to_csv(out / "dest_units.csv", index=False)
        fits = []
        for (u, lv, a, b), grp in du.groupby(["unit", "level", "a", "b"]):
            grp = grp[(grp.cells >= 20) & grp.share_mid.notna()]
            if len(grp) < 2 or grp.share_mid.nunique() < 2:
                continue
            w = grp.cells.to_numpy(float)
            beta = _wfit(grp.share_mid, grp.excess, w)
            if beta is None:
                continue
            fits.append({"unit": u, "level": lv, "a": a, "b": b, "intercept": beta[0], "slope": beta[1],
                         "excess_at_0": float(grp.loc[grp.share_mid == 0, "excess"].iloc[0]) if (grp.share_mid == 0).any()
                         else np.nan, "cells": int(w.sum())})
        pd.DataFrame(fits).to_csv(out / "dest_fit.csv", index=False)

    rfile = src / "dest_region.csv"
    if rfile.exists() and rfile.stat().st_size > 1:
        Rg = pd.read_csv(rfile, dtype={"bin": str})
        ru = Rg.groupby(["unit", "level", "a", "b", "radius", "bin"])[["cells", "to_b_test", "to_b_null"]].sum().reset_index()
        ru["excess"] = (ru.to_b_test - ru.to_b_null) / ru.cells
        ru["share_mid"] = ru["bin"].map(REGION_MID)
        ru.to_csv(out / "dest_region_units.csv", index=False)
        fits = []
        for (u, lv, a, b, R), grp in ru.groupby(["unit", "level", "a", "b", "radius"]):
            grp = grp[(grp.cells >= 20) & grp.share_mid.notna()]
            if len(grp) < 2 or grp.share_mid.nunique() < 2:
                continue
            w = grp.cells.to_numpy(float)
            beta = _wfit(grp.share_mid, grp.excess, w)
            if beta is None:
                continue
            fits.append({"unit": u, "level": lv, "a": a, "b": b, "radius": R, "intercept": beta[0], "slope": beta[1],
                         "cells": int(w.sum())})
        pd.DataFrame(fits).to_csv(out / "dest_region_fit.csv", index=False)
    if verbose and len(Tt):
        import re as _re
        rx = _re.compile(focus)
        foc = Tt[Tt["type"].astype(str).map(lambda v: bool(rx.search(v)))]
        if len(foc):
            pd.set_option("display.width", 200, "display.max_rows", 200)
            for lv, f in foc.groupby("level"):
                print(f"\n{lv}: focal type(s) {sorted(set(f['type']))}; kept ratio (test / null on the same cells), "
                      f"median over units with >= {min_kept} null-kept cells")
                gp = f[(f.scheme == "gap") & (f.m == "all")]
                if len(gp):
                    print("  by heterotypic nuclei within gap g (rows k, columns g):\n" + gp.pivot_table(
                        index="k", columns="g", values="median_ratio").round(3).to_string())
                bd = f[f.scheme == "border"]
                if len(bd):
                    print("  by heterotypic border share:\n" + bd.set_index("bin").reindex(BORDER_NAMES + ["none"])[
                        ["units", "median_ratio", "units_below_1", "cells"]].dropna(how="all").round(3).to_string())
                ex = f[f.scheme == "exposure"]
                if len(ex):
                    print("  by RNA-weighted exposure:\n" + ex.set_index("bin").reindex(EXPO_NAMES + ["none"])[
                        ["units", "median_ratio", "units_below_1", "cells"]].dropna(how="all").round(3).to_string())
    return U, dose, Tt
