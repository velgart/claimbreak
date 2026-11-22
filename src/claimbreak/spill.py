"""Gene-level readouts of a section from whole-cell and nucleus-only counts of the same cells.

Published spatial analyses call a cell positive for a gene when it has one or more counts (SCT-normalized > 0) and read
three kinds of statistics from that call: which cell types express the gene (and what share of its transcripts each
type holds), how positivity in one cell depends on its neighbors (a 'paracrine' gradient), and how many
ligand-positive cells lie near receptor-positive cells ('predicted interactions'). Transcripts from a neighbor that
are assigned to the cell by segmentation enter all three in the same direction as real paracrine signalling: they
make a cell positive for its neighbor's genes, more so the more such neighbors it touches.

The same statistics are computed here from four count arms of the same cells:

  whole, nucleus   all counts of the cell, and those inside its nucleus
  nucB, wholeB     a binomial half of the nuclear counts, and the rest of the cell (whole minus the other half)
                   thinned to the same per-cell depth (the depth-matched pair of the compartment test)

Transcripts that belong to a neighbor lie outside the nucleus, so a readout produced by them is larger from wholeB than
from nucB; a readout of the cell's own expression is not (apart from the gene's own nuclear retention, which the
producing cell type shows: compare the recipient's wholeB / nucB ratio with the producer's).

Readouts (per section; units and groups are medians of sections):
  positivity      per arm and cell type: cells, share positive for `gene`, mean counts, share of the gene's counts
  gradient        recipients (cells not of `source_types`, optionally only of `recipients`): share positive for
                  `gene` by the number of source cells within `radius` um (bins 0, 1, ..., last bin and more), source
                  cells being those of `source_types` positive for `source_gene` in `source_arm`; and by whether a
                  source cell is a contact (among the k nearest within r um)
  interactions    with ligand and receptor: ordered pairs (ligand-positive sender, receptor-positive receiver, distinct
                  cells) within `lr_radius` um, by sender and receiver type, per arm
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .compartment import count_sets
from .placement import unit_from

ARMS = ("whole", "nucleus", "nucB", "wholeB")


def _group(unit, pattern):
    if not pattern:
        return ""
    m = re.search(pattern, str(unit))
    if not m:
        return ""
    return m.group("g") if "g" in m.groupdict() else m.group(0)


def _bins(n, last):
    return np.minimum(n, last)


def section_tables(A, lab, gene, source_types=(), source_gene=None, source_arm="whole", recipients=(),
                   radius=50.0, last_bin=4, k=6, r=30.0, arms=ARMS, ligand=None, receptor=None, lr_radius=20.0,
                   seed=0):
    """One section: A (AnnData with X = whole counts, layers['nucleus']) and lab (cell types aligned with A.obs_names,
    "" = unlabeled). Returns dict of DataFrames: positivity, gradient, interactions (or None)."""
    need = [g for g in dict.fromkeys([gene, source_gene, ligand, receptor]) if g]
    missing = [g for g in need if g not in A.var_names]
    if missing:
        raise ValueError(f"genes not on the panel: {', '.join(missing)}")
    arms = list(dict.fromkeys(list(arms) + ([source_arm] if source_gene else [])))
    Xw = sp.csr_matrix(A.X).astype(np.float32)
    Xn = sp.csr_matrix(A.layers["nucleus"]).astype(np.float32)
    sets = count_sets(Xw, Xn, None, arms, seed)
    gi = {g: A.var_names.get_loc(g) for g in need}
    col = {arm: {g: np.asarray(M[:, gi[g]].todense()).ravel() for g in need} for arm, M in sets.items()}
    lab = np.asarray(lab).astype(str)
    xy = A.obs[["x", "y"]].to_numpy(float)
    ok = lab != ""
    out = {}

    rows = []
    for arm in arms:
        v = col[arm][gene]
        tot = v[ok].sum()
        for t in sorted(set(lab[ok])):
            m = lab == t
            rows.append({"arm": arm, "type": t, "cells": int(m.sum()), "pos": float(np.mean(v[m] > 0)),
                         "mean": float(v[m].mean()), "share_counts": float(v[m].sum() / tot) if tot > 0 else np.nan})
    out["positivity"] = pd.DataFrame(rows)

    out["gradient"] = None
    if source_types:
        src = np.isin(lab, list(source_types))
        if source_gene:
            src &= col[source_arm][source_gene] > 0
        rec = ok & ~np.isin(lab, list(source_types))
        if recipients:
            rec &= np.isin(lab, list(recipients))
        if src.sum() and rec.sum():
            n_src = cKDTree(xy[src]).query_ball_point(xy[rec], radius, return_length=True)
            d, idx = cKDTree(xy).query(xy[rec], k=k + 1)
            touch = ((d[:, 1:] <= r) & src[idx[:, 1:]]).any(1)
            b = _bins(np.asarray(n_src), last_bin)
            ctx = np.where(touch, "contact", np.where(n_src > 0, "near_only", "none"))
            rows = []
            for arm in arms:
                pos = col[arm][gene][rec] > 0
                for kb in range(last_bin + 1):
                    m = b == kb
                    rows.append({"arm": arm, "stratum": f"bin{kb}", "cells": int(m.sum()),
                                 "pos": float(pos[m].mean()) if m.any() else np.nan})
                for c in ("none", "near_only", "contact"):
                    m = ctx == c
                    rows.append({"arm": arm, "stratum": c, "cells": int(m.sum()),
                                 "pos": float(pos[m].mean()) if m.any() else np.nan})
            out["gradient"] = pd.DataFrame(rows)
            out["n_source"] = int(src.sum())

    out["interactions"] = None
    if ligand and receptor:
        rows = []
        for arm in arms:
            L = np.where(ok & (col[arm][ligand] > 0))[0]
            R = np.where(ok & (col[arm][receptor] > 0))[0]
            if len(L) and len(R):
                D = cKDTree(xy[L]).sparse_distance_matrix(cKDTree(xy[R]), lr_radius, output_type="coo_matrix")
                s, q = L[D.row], R[D.col]
                keep = s != q
                pr = pd.DataFrame({"sender": lab[s[keep]], "receiver": lab[q[keep]]})
                c = pr.value_counts().rename("n").reset_index()
            else:
                c = pd.DataFrame(columns=["sender", "receiver", "n"])
            c.insert(0, "arm", arm)
            c["ligand_pos"], c["receptor_pos"] = len(L), len(R)
            rows.append(c)
        out["interactions"] = pd.concat(rows, ignore_index=True)
    return out


def readouts(G: pd.DataFrame, last_bin: int):
    """Per section and arm: positivity with no source cell near, with `last_bin`+ near, their ratio and difference;
    the same for contact against none."""
    W = G.pivot_table(index=["sample", "unit", "group", "arm"], columns="stratum", values="pos").reset_index()
    for c in [f"bin{i}" for i in range(last_bin + 1)] + ["none", "near_only", "contact"]:
        if c not in W:
            W[c] = np.nan                                 # a stratum with no cells in any section
    top = f"bin{last_bin}"
    W["rr_bins"] = W[top] / W["bin0"]
    W["diff_bins"] = W[top] - W["bin0"]
    W["rr_contact"] = W["contact"] / W["none"]
    W["diff_contact"] = W["contact"] - W["none"]
    return W


def spill(h5ad_dir, labels_dir, out, gene: str, level: str = "L1", label_arm: str = "nucleus", source_types=(),
          source_gene=None, source_arm: str = "whole", recipients=(), radius: float = 50.0, last_bin: int = 4,
          k: int = 6, r: float = 30.0, arms=ARMS, ligand=None, receptor=None, lr_radius: float = 20.0,
          unit_pattern: str = r"I0\d{3}", group_pattern: str = "", samples: str = "", seed: int = 20261017,
          verbose: bool = True):
    """All labels files of a folder with their .h5ad (same stem) -> OUT/positivity.csv, gradient.csv,
    gradient_readouts.csv, interactions.csv (with ligand and receptor), meta.json. Returns the tables."""
    hdir, ldir, out = Path(h5ad_dir).expanduser(), Path(labels_dir).expanduser(), Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; choose a new output folder")
    import anndata as ad
    files = sorted(ldir.glob("*.labels.csv.gz"))
    if samples:
        keys = [s for s in samples.split(",") if s]
        files = [f for f in files if any(x in f.name for x in keys)]
    if not files:
        raise FileNotFoundError(f"no labels files in {ldir}" + (f" matching {samples}" if samples else ""))
    col = f"{level}_{label_arm}"
    P, G, I, used = [], [], [], []
    for f in files:
        sample = f.name.split(".labels")[0]
        h = hdir / f"{sample}.h5ad"
        if not h.exists():
            if verbose:
                print(f"{sample}: no {h.name} in {hdir}; skipped", flush=True)
            continue
        D = pd.read_csv(f, keep_default_na=False, low_memory=False, usecols=lambda c: c in ("cell_id", col))
        if col not in D:
            raise ValueError(f"{f.name}: no column {col}")
        A = ad.read_h5ad(h)
        lab = D.set_index(D.cell_id.astype(str))[col].reindex(A.obs_names.astype(str)).fillna("").to_numpy()
        types = set(lab) - {""}
        for t in list(source_types) + list(recipients):
            if t not in types and verbose:
                print(f"  {sample}: type {t} absent", flush=True)
        unit = unit_from(sample, unit_pattern)
        grp = _group(unit, group_pattern)
        T = section_tables(A, lab, gene, source_types, source_gene, source_arm, recipients, radius, last_bin, k, r,
                           arms, ligand, receptor, lr_radius, seed)
        for key, store in (("positivity", P), ("gradient", G), ("interactions", I)):
            if T.get(key) is not None:
                x = T[key]
                x.insert(0, "group", grp); x.insert(0, "unit", unit); x.insert(0, "sample", sample)
                store.append(x)
        used.append({"sample": sample, "unit": unit, "group": grp, "cells": int(A.n_obs),
                     "labelled": int((lab != "").sum()), "source_cells": T.get("n_source")})
        if verbose:
            p = T["positivity"]
            msg = ", ".join(f"{a} {p[p.arm == a].pipe(lambda z: (z.pos * z.cells).sum() / z.cells.sum()):.3f}"
                            for a in arms)
            print(f"{sample} ({unit}): cells {A.n_obs:,}; {gene} positive {msg}"
                  + (f"; source cells {T['n_source']:,}" if T.get("n_source") else ""), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    res = {"sections": pd.DataFrame(used)}
    res["sections"].to_csv(out / "sections.csv", index=False)
    res["positivity"] = pd.concat(P, ignore_index=True)
    res["positivity"].to_csv(out / "positivity.csv", index=False)
    if G:
        res["gradient"] = pd.concat(G, ignore_index=True)
        res["gradient"].to_csv(out / "gradient.csv", index=False)
        res["readouts"] = readouts(res["gradient"], last_bin)
        res["readouts"].to_csv(out / "gradient_readouts.csv", index=False)
    if I:
        res["interactions"] = pd.concat(I, ignore_index=True)
        res["interactions"].to_csv(out / "interactions.csv", index=False)
    meta = dict(gene=gene, level=level, label_arm=label_arm, source_types=list(source_types), source_gene=source_gene,
                source_arm=source_arm, recipients=list(recipients), radius=radius, last_bin=last_bin, k=k, r=r,
                arms=list(arms), ligand=ligand, receptor=receptor, lr_radius=lr_radius, unit_pattern=unit_pattern,
                group_pattern=group_pattern, samples=samples, seed=seed,
                written=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return res


def unit_view(res, recipients=(), source_types=()):
    """Printable per-unit tables: positivity of the gene by type and arm; gradient readouts by arm; homotypic share and
    interactions per 1000 cells by arm."""
    V = {}
    P = res["positivity"]
    pu = P.groupby(["unit", "group", "type", "arm"]).pos.median().unstack("arm")
    V["positivity"] = pu
    sc = P.groupby(["unit", "group", "type", "arm"]).share_counts.median().unstack("arm")
    V["share_counts"] = sc
    if "readouts" in res:
        R = res["readouts"]
        V["gradient"] = R.groupby(["unit", "group", "arm"])[["bin0", "rr_bins", "diff_bins", "none", "contact",
                                                              "rr_contact"]].median().unstack("arm")
    if "interactions" in res:
        I = res["interactions"]
        cells = res["sections"].set_index("sample").cells
        tot = I.groupby(["sample", "unit", "group", "arm"]).n.sum().rename("n").reset_index()
        tot["per1000"] = tot.n / tot["sample"].map(cells) * 1000
        V["interactions_per1000"] = tot.groupby(["unit", "group", "arm"]).per1000.median().unstack("arm")
        homo = I[(I.sender == I.receiver)].groupby(["sample", "unit", "group", "arm", "sender"]).n.sum()
        share = (homo / I.groupby(["sample", "unit", "group", "arm"]).n.sum()).rename("share").reset_index()
        V["homotypic_share"] = share.groupby(["unit", "group", "sender", "arm"]).share.median().unstack("arm")
    return V
