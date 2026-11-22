"""Contact-scale readouts of immune infiltration from two label sets, and the field's random stress test on them.

kNN colocalization (k = 10) dilutes what happens at contact: a T cell relabeled as the tumor cell it touches
changes one of ten neighbors. The readouts here are the ones that published analyses compute at contact scale:

  infiltration   share of type-a cells (immune) with at least one type-b cell (partner) among their contacts
  per100         type-a cells touching type b per 100 type-b cells (in gut with a = T/NK, b = epithelial: the
                 intraepithelial-lymphocyte count of histology, here in two dimensions)
  mixing         Keren et al. (Cell 2018): immune-partner contacts over immune-immune contacts (undirected contact
                 edges); class cold (< 250 immune cells), compartmentalized (score < 0.22) or mixed; per section and
                 per unit (patient or donor), the unit from summed contact counts of its sections

Contacts: `knn` (default), the 6 nearest cells within 30 um, symmetrised; or `boundary`, cells whose segmentation
boundaries come within `tol` um of each other (boundaries densified to points every `tol` um; Xenium
cell_boundaries.parquet, joined on cell_id; needs pyarrow),
the analogue of Keren's touching cells.

Each readout is computed from the reference labels (default `nucleus`) and the analyst's labels (default `whole`),
and under the field's stress test on the analyst's labels (eps of cells reassigned at random, reps draws). Written
before any run: whole-cell labeling turns immune cells that touch a larger partner cell into that
cell, so infiltration, per100 and the mixing score are expected lower from the analyst's labels than from the
reference (and the mixing class to move toward compartmentalized); random relabeling is expected to move them the
other way (toward mixing).
"""
from __future__ import annotations

import gzip
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .calls import random_relabel
from .placement import unit_from

KEREN_COLD, KEREN_CUT = 250, 0.22


# --------------------------------------------------------------------------- contact graphs

def knn_edges(xy, k: int = 6, r: float = 30.0):
    """Undirected contact edges (i < j): j among the k nearest of i within r, or i among those of j."""
    d, idx = cKDTree(xy).query(xy, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    i = np.repeat(np.arange(len(xy)), k)
    j = idx.ravel()
    m = d.ravel() <= r
    a, b = np.minimum(i[m], j[m]), np.maximum(i[m], j[m])
    e = np.unique(a.astype(np.int64) * len(xy) + b)
    return np.column_stack([e // len(xy), e % len(xy)])


def read_parquet_any(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            return pd.read_parquet(io.BytesIO(fh.read()))
    return pd.read_parquet(path)


def find_boundaries(bdir, sample):
    bdir = Path(bdir).expanduser()
    for p in (bdir / f"{sample}_cell_boundaries.parquet", bdir / f"{sample}_cell_boundaries.parquet.gz",
              bdir / sample / "cell_boundaries.parquet", bdir / sample / "cell_boundaries.parquet.gz"):
        if p.exists():
            return p
    hits = sorted(bdir.glob(f"{sample}*cell_boundaries.parquet*"))
    return hits[0] if len(hits) == 1 else None


def densify(bnd: pd.DataFrame, owner: np.ndarray, step: float):
    """Points every <= step um along each cell's boundary polygon (vertices in file order, closed)."""
    x, y = bnd["vertex_x"].to_numpy(float), bnd["vertex_y"].to_numpy(float)
    nxt = np.r_[np.arange(1, len(x)), 0]
    last = np.r_[owner[1:] != owner[:-1], True]               # last vertex of each cell: close to its first one
    first = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    nxt[last] = first[np.cumsum(np.r_[True, owner[1:] != owner[:-1]])[last] - 1]
    seg = np.hypot(x[nxt] - x, y[nxt] - y)
    m = np.maximum(np.ceil(seg / step).astype(int), 1)
    rep = np.repeat(np.arange(len(x)), m)
    t = (np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)) / np.repeat(m, m)
    return x[rep] + t * (x[nxt][rep] - x[rep]), y[rep] + t * (y[nxt][rep] - y[rep]), owner[rep]


def boundary_edges(bnd: pd.DataFrame, cell_ids, tol: float = 1.0):
    """Undirected edges between cells (indices into cell_ids) whose boundaries come within tol um: boundaries are
    densified to points every tol um, and points of different cells within tol are contacts."""
    pos = pd.Index(pd.Series(cell_ids).astype(str))
    b = bnd[bnd["cell_id"].astype(str).isin(pos)]
    owner = pos.get_indexer(b["cell_id"].astype(str))
    px, py, po = densify(b, owner, tol)
    P = np.column_stack([px, py])
    d, j = cKDTree(P).query(P, k=8, distance_upper_bound=tol)
    i = np.repeat(np.arange(len(P)), 8)
    j, d = j.ravel(), d.ravel()
    ok = np.isfinite(d)
    a, c = po[i[ok]], po[j[ok]]
    m = a != c
    lo, hi = np.minimum(a[m], c[m]), np.maximum(a[m], c[m])
    n = len(pos)
    e = np.unique(lo.astype(np.int64) * n + hi)
    return np.column_stack([e // n, e % n])


def boundary_shares(bnd: pd.DataFrame, cell_ids, tol: float = 1.0, chunk: int = 2_000_000):
    """Shared perimeter from segmentation polygons: each cell's boundary is
    densified to points every tol um; a point is held by the nearest point of another cell within tol um. Returns the
    arrays of geometry.border_shares (pairs i_p, j_p into cell_ids, share_p = points of i held by j / points of i)
    and the covered share per cell (points held by any other cell / points); cells without a polygon have no pairs
    and covered 0."""
    pos = pd.Index(pd.Series(cell_ids).astype(str))
    n = len(pos)
    b = bnd[bnd["cell_id"].astype(str).isin(pos)]
    if not len(b):
        return np.array([], int), np.array([], int), np.array([], float), np.zeros(n)
    owner = pos.get_indexer(b["cell_id"].astype(str))
    px, py, po = densify(b, owner, tol)
    P = np.column_stack([px, py])
    tree = cKDTree(P)
    total = np.bincount(po, minlength=n).astype(float)
    hi_owner = np.full(len(P), -1, np.int64)
    k = 8
    for lo in range(0, len(P), chunk):
        sl = slice(lo, min(len(P), lo + chunk))
        d, j = tree.query(P[sl], k=k, distance_upper_bound=tol)
        own = po[sl][:, None]
        jj = np.where(j < len(P), j, 0)
        foreign = (j < len(P)) & (po[jj] != own)          # columns are sorted by distance: the first foreign is nearest
        has = foreign.any(1)
        first = foreign.argmax(1)
        hi_owner[sl][has] = po[jj[np.flatnonzero(has), first[has]]]
    held = hi_owner >= 0
    covered = np.bincount(po[held], minlength=n) / np.maximum(total, 1)
    key = po[held].astype(np.int64) * n + hi_owner[held]
    uk, cnt = np.unique(key, return_counts=True)
    i_p, j_p = (uk // n).astype(int), (uk % n).astype(int)
    return i_p, j_p, cnt / total[i_p], covered


# --------------------------------------------------------------------------- readouts

def counts(codes, edges, K, imm, part):
    """Sufficient counts for every readout: per (a, b) cells of type a touching >= 1 b (touch), cells per type,
    immune-partner and immune-immune edges, immune cells."""
    a, b = codes[edges[:, 0]], codes[edges[:, 1]]
    n = len(codes)
    # does cell i touch type t: mark both directions
    T = np.zeros((n, K), bool)
    T[edges[:, 0], b] = True
    T[edges[:, 1], a] = True
    touch = np.zeros((K, K))
    for t in range(K):
        touch[t] = T[codes == t].sum(0)
    is_imm, is_part = np.isin(codes, imm), np.isin(codes, part)
    ia, ib = is_imm[edges[:, 0]], is_imm[edges[:, 1]]
    pa, pb = is_part[edges[:, 0]], is_part[edges[:, 1]]
    return dict(touch=touch, n=np.bincount(codes, minlength=K).astype(float),
                e_ip=float(((ia & pb) | (ib & pa)).sum()), e_ii=float((ia & ib).sum()), n_imm=float(is_imm.sum()))


def counts_fixed(codes, ref, edges, K, imm, part):
    """as counts, but with partners fixed at the reference labels: rows (and immune cells) by `codes`, the
    touched type and the partner counts by `ref`. per100|a|b is then cells labeled a in `codes` touching >= 1 cell
    labeled b in `ref`, per 100 cells labeled b in `ref`, so only the immune side of the label changes varies."""
    a_ref, b_ref = ref[edges[:, 0]], ref[edges[:, 1]]
    n = len(codes)
    T = np.zeros((n, K), bool)
    T[edges[:, 0], b_ref] = True
    T[edges[:, 1], a_ref] = True
    touch = np.zeros((K, K))
    for t in range(K):
        touch[t] = T[codes == t].sum(0)
    is_imm, is_part = np.isin(codes, imm), np.isin(ref, part)
    ia, ib = is_imm[edges[:, 0]], is_imm[edges[:, 1]]
    pa, pb = is_part[edges[:, 0]], is_part[edges[:, 1]]
    return dict(touch=touch, n=np.bincount(codes, minlength=K).astype(float),
                n_part=np.bincount(ref, minlength=K).astype(float),
                e_ip=float(((ia & pb) | (ib & pa)).sum()), e_ii=float((ia & ib).sum()), n_imm=float(is_imm.sum()))


def readouts(c, pairs):
    """From summed counts: infiltration and per100 per (a, b) pair, mixing score and class."""
    out = {}
    for a, b in pairs:
        out[f"infil|{a.name}|{b.name}"] = c["touch"][a.i, b.i] / c["n"][a.i] if c["n"][a.i] else np.nan
        npart = c.get("n_part", c["n"])
        out[f"per100|{a.name}|{b.name}"] = 100 * c["touch"][a.i, b.i] / npart[b.i] if npart[b.i] else np.nan
    out["mixing"] = c["e_ip"] / c["e_ii"] if c["e_ii"] else np.nan
    out["n_immune"] = c["n_imm"]
    out["mixing_class"] = keren_class(out["mixing"], out["n_immune"])
    return out


def keren_class(score, n_imm, cold=KEREN_COLD, cut=KEREN_CUT):
    if n_imm < cold:
        return "cold"
    if not np.isfinite(score):
        return "cold"
    return "compartmentalized" if score < cut else "mixed"


class _T:
    def __init__(self, name, i):
        self.name, self.i = name, i


def _sum(cs):
    out = dict(touch=sum(c["touch"] for c in cs), n=sum(c["n"] for c in cs), e_ip=sum(c["e_ip"] for c in cs),
               e_ii=sum(c["e_ii"] for c in cs), n_imm=sum(c["n_imm"] for c in cs))
    if all("n_part" in c for c in cs):
        out["n_part"] = sum(c["n_part"] for c in cs)
    return out


def _remap(c, types, all_types):
    """Counts on a section's own type codes -> counts on the union of types."""
    K = len(all_types)
    ix = [all_types.index(t) for t in types]
    touch = np.zeros((K, K))
    touch[np.ix_(ix, ix)] = c["touch"]
    n = np.zeros(K)
    n[ix] = c["n"]
    out = dict(c, touch=touch, n=n)
    if "n_part" in c:
        npart = np.zeros(K)
        npart[ix] = c["n_part"]
        out["n_part"] = npart
    return out


def confusion_relabel(ref, test, K, rng):
    """Each cell's reference label replaced by a draw from the section's confusion row P(test | ref): the same expected
    number of changes of every kind (a -> b) as between ref and test, but placed at random among the type-a cells."""
    C = np.zeros((K, K))
    np.add.at(C, (ref, test), 1)
    out = ref.copy()
    for a in range(K):
        idx = np.flatnonzero(ref == a)
        if len(idx) and C[a].sum():
            out[idx] = rng.choice(K, len(idx), p=C[a] / C[a].sum())
    return out


def contacts(labels_dir, out, level: str, immune, partners, ref_arm: str = "nucleus", test_arm: str = "whole",
             mixing_immune=None, mixing_partner=None, contact: str = "knn", k: int = 6, r: float = 30.0,
             boundaries=None, tol: float = 1.0, eps=(0.05, 0.10, 0.15), reps: int = 10, mode: str = "uniform",
             robust: float = 0.9, unit_pattern: str = r"I0\d{3}", use_sections=None, seed: int = 20261015,
             confusion: bool = False, fix_partners: bool = False, verbose: bool = True):
    """Write sections.csv and units.csv (one row per section or unit and label set: ref, test, and the mean over
    random draws at each eps; with `confusion`, also "confusion": the reference labels relabeled at random with the
    section's own confusion P(test label | ref label), the mean over `reps` draws -- as many changes of each kind as the
    test labels have, placed without regard to position; with `fix_partners`, also "test_fixed" and,
    with `confusion`, "confusion_fixed": the same label sets with the partner cells fixed at the reference labels, so
    that only the immune side changes), classes.csv (mixing class per section and unit: reference, analyst, share of random
    draws keeping the analyst's class) and meta.json into OUT."""
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
    if contact == "boundary" and not boundaries:
        raise ValueError("contact='boundary' needs the folder of cell_boundaries.parquet files (boundaries)")
    immune, partners = list(immune), list(partners)
    mix_imm = list(mixing_immune or immune)
    mix_part = list(mixing_partner or partners[:1])
    rng = np.random.default_rng(seed)
    cr, ct = f"{level}_{ref_arm}", f"{level}_{test_arm}"
    started = datetime.now(timezone.utc).isoformat()
    per = []           # (sample, unit, types, {"ref": c, "test": c, (eps, r): c})
    all_types = set()
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns
        need = [cr, ct, "x", "y"] + (["cell_id"] if contact == "boundary" else [])
        miss = [c for c in need if c not in cols]
        if miss:
            raise KeyError(f"{f.name}: missing columns {miss}")
        use = set(need) | {"sample", "unit", "donor"}
        D = pd.read_csv(f, usecols=lambda c: c in use, keep_default_na=False, low_memory=False)
        D = D[(D[cr].astype(str) != "") & (D[ct].astype(str) != "")].reset_index(drop=True)
        sample = str(D["sample"].iloc[0]) if "sample" in D else f.name.split(".labels")[0]
        fb = D["unit"].iloc[0] if "unit" in D else (D["donor"].iloc[0] if "donor" in D else None)
        unit = unit_from(sample, unit_pattern, fb)
        types = sorted(set(D[cr].astype(str)) | set(D[ct].astype(str)))
        code = {t: i for i, t in enumerate(types)}
        K = len(types)
        R = D[cr].astype(str).map(code).to_numpy()
        T = D[ct].astype(str).map(code).to_numpy()
        if contact == "boundary":
            bp = find_boundaries(boundaries, sample)
            if bp is None:
                raise FileNotFoundError(f"no cell_boundaries.parquet for {sample} in {boundaries}")
            E = boundary_edges(read_parquet_any(bp), D["cell_id"].astype(str).to_numpy(), tol)
        else:
            E = knn_edges(D[["x", "y"]].to_numpy(float), k, r)
        imm = [code[t] for t in mix_imm if t in code]
        part = [code[t] for t in mix_part if t in code]
        cs = {"ref": counts(R, E, K, imm, part), "test": counts(T, E, K, imm, part)}
        if fix_partners:
            cs["test_fixed"] = counts_fixed(T, R, E, K, imm, part)
        freq = np.bincount(T, minlength=K) / len(T)
        for e in eps:
            for rep in range(reps):
                cs[(e, rep)] = counts(random_relabel(T, e, K, rng, mode, freq), E, K, imm, part)
        if confusion:                       # own generator, so the random_<eps> draws do not change
            crng = np.random.default_rng([seed, len(per)])
            for rep in range(reps):
                Cf = confusion_relabel(R, T, K, crng)
                cs[("conf", rep)] = counts(Cf, E, K, imm, part)
                if fix_partners:
                    cs[("conf_fixed", rep)] = counts_fixed(Cf, R, E, K, imm, part)
        per.append((sample, unit, types, cs))
        all_types |= set(types)
        say(f"{sample} ({unit}): {len(D):,} cells, {len(E):,} contacts ({len(E) / len(D):.2f} per cell), "
            f"disagreement {np.mean(R != T):.3f}", flush=True)

    all_types = sorted(all_types)
    ti = {t: _T(t, i) for i, t in enumerate(all_types)}
    pairs = [(ti[a], ti[b]) for a in immune for b in partners if a in ti and b in ti]
    keys = ["ref", "test"] + [(e, rep) for e in eps for rep in range(reps)]
    if confusion:
        keys += [("conf", rep) for rep in range(reps)]
    if fix_partners:
        keys += ["test_fixed"] + ([("conf_fixed", rep) for rep in range(reps)] if confusion else [])
    srows, crows = [], []

    def emit(level_name, name, unit, cmap):
        R = {kk: readouts(cmap[kk], pairs) for kk in keys}
        base = dict(level=level_name, name=name, unit=unit)
        num = [x for x in R["ref"] if x != "mixing_class"]
        rows = [dict(base, labels="ref", **{x: R["ref"][x] for x in num}),
                dict(base, labels="test", **{x: R["test"][x] for x in num})]
        for e in eps:
            draws = [R[(e, rep)] for rep in range(reps)]
            rows.append(dict(base, labels=f"random_{int(round(100 * e))}",
                             **{x: float(np.nanmean([d[x] for d in draws])) for x in num}))
        if confusion:
            draws = [R[("conf", rep)] for rep in range(reps)]
            rows.append(dict(base, labels="confusion", **{x: float(np.nanmean([d[x] for d in draws])) for x in num}))
        if fix_partners:
            rows.append(dict(base, labels="test_fixed", **{x: R["test_fixed"][x] for x in num}))
            if confusion:
                draws = [R[("conf_fixed", rep)] for rep in range(reps)]
                rows.append(dict(base, labels="confusion_fixed",
                                 **{x: float(np.nanmean([d[x] for d in draws])) for x in num}))
        crow = dict(base, mixing_ref=R["ref"]["mixing"], mixing_test=R["test"]["mixing"],
                    class_ref=R["ref"]["mixing_class"], class_test=R["test"]["mixing_class"])
        for e in eps:
            draws = [R[(e, rep)] for rep in range(reps)]
            tag = int(round(100 * e))
            crow[f"mixing_random_{tag}"] = float(np.nanmean([d["mixing"] for d in draws]))
            crow[f"keep_class_{tag}"] = float(np.mean([d["mixing_class"] == R["test"]["mixing_class"] for d in draws]))
        tag = int(round(100 * max(eps)))
        crow["class_changed"] = crow["class_ref"] != crow["class_test"]
        crow["certified_wrongly"] = bool(crow["class_changed"] and crow[f"keep_class_{tag}"] >= robust)
        return rows, crow

    for sample, unit, types, cs in per:
        cmap = {kk: _remap(v, types, all_types) for kk, v in cs.items()}
        rr, cc = emit("section", sample, unit, cmap)
        srows += rr
        crows.append(cc)
    urows = []
    for u in sorted({p[1] for p in per}):
        mine = [p for p in per if p[1] == u]
        cmap = {kk: _sum([_remap(p[3][kk], p[2], all_types) for p in mine]) for kk in keys}
        rr, cc = emit("unit", u, u, cmap)
        urows += rr
        cc["sections"] = len(mine)
        crows.append(cc)
    out.mkdir(parents=True, exist_ok=True)
    S, U, C = pd.DataFrame(srows), pd.DataFrame(urows), pd.DataFrame(crows)
    S.to_csv(out / "sections.csv", index=False)
    U.to_csv(out / "units.csv", index=False)
    C.to_csv(out / "classes.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(
        started=started, finished=datetime.now(timezone.utc).isoformat(), labels=str(labels_dir), level=level,
        ref_arm=ref_arm, test_arm=test_arm, immune=immune, partners=partners, mixing_immune=mix_imm,
        mixing_partner=mix_part, contact=contact, k=k, r=r, boundaries=str(boundaries) if boundaries else None,
        tol=tol, eps=list(eps), reps=reps, mode=mode, robust=robust, keren_cold=KEREN_COLD, keren_cut=KEREN_CUT,
        unit_pattern=unit_pattern, seed=seed, confusion=confusion, fix_partners=fix_partners,
        sections=[p[0] for p in per]), indent=1))
    return S, U, C


def compare_units(U: pd.DataFrame):
    """Per readout: units, median ref, test and random values, units with test below ref, units where random moves
    the test value up (the opposite direction)."""
    num = [c for c in U.columns if c.startswith(("infil|", "per100|")) or c == "mixing"]
    rows = []
    rnd = sorted(l for l in U["labels"].unique() if l.startswith("random_"))
    W = U.pivot_table(index="unit", columns="labels", values=num)
    for x in num:
        if (x, "ref") not in W or (x, "test") not in W:
            continue
        ok = W[(x, "ref")].notna() & W[(x, "test")].notna()
        r = dict(readout=x, units=int(ok.sum()), ref=W[(x, "ref")][ok].median(), test=W[(x, "test")][ok].median(),
                 test_below_ref=int((W[(x, "test")][ok] < W[(x, "ref")][ok]).sum()))
        for l in rnd:
            r[l] = W[(x, l)][ok].median()
            r[f"{l}_above_test"] = int((W[(x, l)][ok] > W[(x, "test")][ok]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- immune aggregates

def immune_components(codes, edges, imm, min_size: int = 50):
    """Connected components of immune cells on the contact graph (edges between two immune cells only): number of
    components with at least min_size immune cells, immune cells in them, and the largest component."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    is_imm = np.isin(codes, imm)
    m = is_imm[edges[:, 0]] & is_imm[edges[:, 1]]
    n = len(codes)
    G = coo_matrix((np.ones(int(m.sum())), (edges[m, 0], edges[m, 1])), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    sizes = np.bincount(lab[is_imm]) if is_imm.any() else np.zeros(0, int)
    big = sizes[sizes >= min_size]
    return dict(n_immune=int(is_imm.sum()), aggregates=int(len(big)), cells_in_aggregates=int(big.sum()),
                share_in_aggregates=float(big.sum() / max(is_imm.sum(), 1)), largest=int(sizes.max()) if len(sizes) else 0)


def aggregates(labels_dir, out, level: str, immune, ref_arm: str = "nucleus", test_arms=("whole",), k: int = 6,
               r: float = 30.0, min_size: int = 50, eps=(0.05, 0.10, 0.15), reps: int = 10, mode: str = "uniform",
               unit_pattern: str = r"I0\d{3}", use_sections=None, seed: int = 20261016, verbose: bool = True):
    """Immune aggregates (connected immune components of >= min_size cells on the kNN contact graph) per section from
    the reference arm and each test arm, and under random relabeling of the first test arm; per unit summed over
    sections. Writes sections.csv, units.csv, meta.json into OUT."""
    say = print if verbose else (lambda *a, **k: None)
    out = Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    files = sorted(Path(labels_dir).expanduser().glob("*.labels.csv.gz"))
    if use_sections is not None:
        p = Path(use_sections).expanduser()
        t = pd.read_csv(p)
        names = set(t.loc[t["used"].astype(bool), "sample"].astype(str)) if "used" in t else set(t["sample"])
        files = [f for f in files if f.name.split(".labels")[0] in names]
    if not files:
        raise FileNotFoundError(f"no labels files to use in {labels_dir}")
    rng = np.random.default_rng(seed)
    arms = [ref_arm] + list(test_arms)
    rows = []
    for f in files:
        cols = [f"{level}_{a}" for a in arms]
        D = pd.read_csv(f, usecols=lambda c: c in set(cols) | {"sample", "x", "y", "unit", "donor"},
                        keep_default_na=False, low_memory=False)
        D = D[(D[cols].astype(str) != "").all(axis=1)].reset_index(drop=True)
        sample = str(D["sample"].iloc[0]) if "sample" in D else f.name.split(".labels")[0]
        fb = D["unit"].iloc[0] if "unit" in D else (D["donor"].iloc[0] if "donor" in D else None)
        unit = unit_from(sample, unit_pattern, fb)
        types = sorted(set(np.concatenate([D[c].astype(str).to_numpy() for c in cols])))
        code = {t: i for i, t in enumerate(types)}
        imm = [code[t] for t in immune if t in code]
        E = knn_edges(D[["x", "y"]].to_numpy(float), k, r)
        for a, c in zip(arms, cols):
            C = D[c].astype(str).map(code).to_numpy()
            rows.append(dict(section=sample, unit=unit, labels=a, **immune_components(C, E, imm, min_size)))
        T = D[cols[1]].astype(str).map(code).to_numpy()
        freq = np.bincount(T, minlength=len(types)) / len(T)
        for e in eps:
            draws = [immune_components(random_relabel(T, e, len(types), rng, mode, freq), E, imm, min_size)
                     for _ in range(reps)]
            rows.append(dict(section=sample, unit=unit, labels=f"random_{int(round(100 * e))}",
                             **{x: float(np.mean([d[x] for d in draws])) for x in draws[0]}))
        say(f"{sample} ({unit}): " + ", ".join(f"{r['labels']} {r['aggregates']:.0f}" for r in rows[-(len(arms) + len(eps)):]),
            flush=True)
    out.mkdir(parents=True, exist_ok=True)
    S = pd.DataFrame(rows)
    S.to_csv(out / "sections.csv", index=False)
    U = S.groupby(["unit", "labels"])[["n_immune", "aggregates", "cells_in_aggregates"]].sum().reset_index()
    U["share_in_aggregates"] = U.cells_in_aggregates / U.n_immune.clip(lower=1)
    U.to_csv(out / "units.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(labels=str(labels_dir), level=level, immune=list(immune),
        ref_arm=ref_arm, test_arms=list(test_arms), k=k, r=r, min_size=min_size, eps=list(eps), reps=reps, mode=mode,
        unit_pattern=unit_pattern, seed=seed, sections=sorted(S.section.unique().tolist())), indent=1))
    return S, U
