#!/usr/bin/env python
"""vendor_calls.py -- the analyses a typical user runs, from whole-cell, nuclear, hybrid and protein labels of the
same cells, per field of view: which calls change? (vendor data with
protein truth.)

Per section, cells are coded from each arm's labels into four classes by regular expressions: T (T/NK labels), I
(other immune), P (the partner: tumor or epithelium), O (the rest). Arms:
  whole, nucleus      the labels file's {level}_whole and {level}_nucleus
  hybrid              nucleus where it has a label, whole otherwise (the paper's hybrid)
  protein             (Xenium Protein, --protein) the nucleus arm with T replaced by protein: T = CD3E+ CD45+ CD68-
                      PanCK- cells (xenium_protein.positive), cells labeled T but not protein T become O; with
                      --protein-partner PANCK+ the partner too is PanCK+ by protein

The section is cut into square fields of --fov um (default 800, about a MIBI field of Keren et al. 2018), and in each
field with enough cells, for the pair (T, P):
  nhood     squidpy-style neighborhood enrichment: kNN graph (--k-nhood 6, squidpy's default for generic
            coordinates), log2 observed / expected T-P edges and a permutation z (labels permuted in the field,
            --nperm); call enriched / depleted / neither at |z| >= 2 and |log2 O/E| >= 0.2 (claimbreak.calls)
  per100    T cells touching >= 1 P per 100 P (contacts: 6 nearest within 30 um, claimbreak.contacts.knn_edges)
  infil     share of T cells touching >= 1 P
  mixing    Keren mixing score, immune (T + I) to P contacts over immune-immune contacts, and its class: cold (< 250
            immune cells), compartmentalized (< 0.22) or mixed

A flip is a field whose call (nhood) or class (mixing) differs between two arms. flips.csv counts them per pair of
arms; fields.csv has every readout; summary.json the medians. `--check` prints the label vocabulary of each section
and the class sizes the regexes give, and writes nothing, so the regexes can be chosen first.

  python vendor_calls.py --labels DIR --level L1 --sample XP_CCRCC --partner '(?i)^(PT|epithelial|tumou?r)' \
         [--protein FILE --protein-partner PANCK+] --out DIR [--check] [--max-fovs 40]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
from claimbreak.calls import call_of, edge_index, enrichment  # noqa: E402
from claimbreak.contacts import KEREN_COLD, KEREN_CUT, keren_class, knn_edges  # noqa: E402

T_RE = r"(?i)(^|[^a-z])(t|nk|tnk|t_nk|t[ _-]?cells?|cd[48]\+?[ _-]?t?)([^a-z]|$)"
CODES = {"T": 0, "I": 1, "P": 2, "O": 3}


def code_labels(lab, t_re, i_re, p_re):
    lab = pd.Series(lab).astype(str)
    c = np.full(len(lab), CODES["O"])
    c[lab.str.contains(p_re, regex=True).to_numpy()] = CODES["P"]
    c[lab.str.contains(i_re, regex=True).to_numpy()] = CODES["I"]
    c[lab.str.contains(t_re, regex=True).to_numpy()] = CODES["T"]
    c[(lab == "").to_numpy()] = -1                                   # unlabeled in this arm
    return c


def protein_arm(D, a, nuc_codes):
    from xenium_protein import STRICT, norm, positive
    cols = {norm(c): c for c in D.columns}
    req, neg = STRICT["T_strict"][0], STRICT["T_strict"][1]
    pos = {}
    want = set(req + neg)
    if a.protein_partner:
        want.add(norm(a.protein_partner[:-1]))
    for m in sorted(want):
        if m not in cols:
            raise SystemExit(f"protein marker {m} missing from {a.protein}")
        pos[m] = positive(pd.to_numeric(D[cols[m]], errors="coerce").fillna(0).to_numpy(), seed=a.seed, post=0.5)[0]
    t = np.logical_and.reduce([pos[m] for m in req])
    for m in neg:
        t &= ~pos[m]
    c = nuc_codes.copy()
    c[(c == CODES["T"]) & ~t] = CODES["O"]
    c[t] = CODES["T"]
    if a.protein_partner:
        p = pos[norm(a.protein_partner[:-1])]
        p = p if a.protein_partner.endswith("+") else ~p
        c[(c == CODES["P"]) & ~p] = CODES["O"]
        c[p & ~t] = CODES["P"]
    return c


def field_readouts(codes, xy, a, rng):
    ok = codes >= 0
    codes, xy = codes[ok], xy[ok]
    out = dict(n=len(codes), n_T=int((codes == 0).sum()), n_P=int((codes == 2).sum()),
               n_imm=int(((codes == 0) | (codes == 1)).sum()))
    if out["n_T"] < a.min_t or out["n_P"] < a.min_p:
        return out
    src, dst, _ = edge_index(xy, a.k_nhood)
    L, z = enrichment(codes, src, dst, 4, a.k_nhood, a.nperm, rng)
    l2, zz = float(L[0, 2]), float(z[0, 2])
    out.update(log2oe=l2, z=zz, nhood=int(call_of(np.array([l2]), np.array([zz]), 2.0, 0.2)[0]))
    e = knn_edges(xy, 6, 30.0)
    ca, cb = codes[e[:, 0]], codes[e[:, 1]]
    tp = ((ca == 0) & (cb == 2)) | ((ca == 2) & (cb == 0))
    touch = np.zeros(len(codes), bool)
    touch[e[tp & (ca == 0), 0]] = True
    touch[e[tp & (cb == 0), 1]] = True
    nt = int(touch[codes == 0].sum())
    out.update(per100=100 * nt / out["n_P"], infil=nt / out["n_T"])
    imm_a, imm_b = (ca == 0) | (ca == 1), (cb == 0) | (cb == 1)
    e_ip = int(((imm_a & (cb == 2)) | (imm_b & (ca == 2))).sum())
    e_ii = int((imm_a & imm_b).sum())
    out["mixing"] = e_ip / e_ii if e_ii else np.nan
    out["mixing_class"] = keren_class(out["mixing"], out["n_imm"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True); ap.add_argument("--out", default="")
    ap.add_argument("--level", default="L1"); ap.add_argument("--sample", default="", help="only these sections (comma)")
    ap.add_argument("--t-regex", default=T_RE)
    ap.add_argument("--immune-regex", default=r"(?i)(^|[^a-z])(b|b[ _-]?cells?|plasma\w*|myeloid|macrophages?|mono\w*|"
                                              r"dcs?|mast\w*|neutro\w*|immune|lympho\w*|mye\w*)([^a-z]|$)")
    ap.add_argument("--partner", required=True, help="regex of the partner labels (tumor or epithelium)")
    ap.add_argument("--protein", default="", help="<sample>.protein.csv.gz (Xenium Protein): adds the protein arm")
    ap.add_argument("--protein-partner", default="", help="e.g. PANCK+: partner by protein too")
    ap.add_argument("--fov", type=float, default=800.0); ap.add_argument("--min-t", type=int, default=20)
    ap.add_argument("--min-p", type=int, default=50); ap.add_argument("--k-nhood", type=int, default=6)
    ap.add_argument("--nperm", type=int, default=200); ap.add_argument("--max-fovs", type=int, default=0,
                                                                         help="per section, at most this many fields")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    files = sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz"))
    if a.sample:
        keep = set(a.sample.split(","))
        files = [f for f in files if f.name.split(".labels")[0] in keep]
    if not files:
        raise SystemExit("no labels files")
    cw, cn = f"{a.level}_whole", f"{a.level}_nucleus"
    if a.check:
        for f in files:
            L = pd.read_csv(f, usecols=lambda c: c in (cw, cn), keep_default_na=False)
            print(f"\n== {f.name}: {len(L):,} cells")
            for c in (cn, cw):
                if c not in L:
                    print(f"  no {c}"); continue
                vc = L[c].astype(str).value_counts()
                print(f"  {c}: " + ", ".join(f"{k or '(none)'} {v:,}" for k, v in vc.head(25).items()))
                cc = pd.Series(code_labels(L[c], a.t_regex, a.immune_regex, a.partner)).map(
                    {-1: "unlabelled", 0: "T", 1: "I", 2: "P", 3: "O"}).value_counts()
                print(f"    classes: " + ", ".join(f"{k} {v:,}" for k, v in cc.items()))
                for cls, code in (("T", 0), ("I", 1), ("P", 2)):
                    names = sorted(set(L[c].astype(str)[code_labels(L[c], a.t_regex, a.immune_regex, a.partner) == code]))
                    print(f"    {cls} <- {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")
        return
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for f in files:
        stem = f.name.split(".labels")[0]
        D = pd.read_csv(f, keep_default_na=False, dtype={"cell_id": str}, low_memory=False)
        if cw not in D or cn not in D:
            print(f"  {stem}: no {cw} or {cn}; skipped"); continue
        if a.protein:
            P = pd.read_csv(Path(a.protein).expanduser(), dtype={"cell_id": str})
            D = D.merge(P, on="cell_id", how="inner")
        xy = D[["x", "y"]].to_numpy(float)
        nucl, whol = D[cn].astype(str).to_numpy(), D[cw].astype(str).to_numpy()
        arms = {"whole": code_labels(whol, a.t_regex, a.immune_regex, a.partner),
                "nucleus": code_labels(nucl, a.t_regex, a.immune_regex, a.partner),
                "hybrid": code_labels(np.where(nucl != "", nucl, whol), a.t_regex, a.immune_regex, a.partner)}
        if a.protein:
            arms["protein"] = protein_arm(D, a, arms["nucleus"])
        fx = np.floor((xy - xy.min(0)) / a.fov).astype(int)
        fid = pd.Series([f"{u}_{v}" for u, v in fx])
        fovs = fid.value_counts()
        fovs = fovs[fovs >= 200].index.tolist()
        if a.max_fovs and len(fovs) > a.max_fovs:
            fovs = list(rng.choice(fovs, a.max_fovs, replace=False))
        print(f"  {stem}: {len(D):,} cells, {len(fovs)} fields of {a.fov:g} um", flush=True)
        for fv in fovs:
            m = (fid == fv).to_numpy()
            for arm, c in arms.items():
                r = field_readouts(c[m], xy[m], a, rng)
                rows.append(dict(section=stem, fov=fv, arm=arm, **r))
    F = pd.DataFrame(rows)
    F.to_csv(out / "fields.csv", index=False)
    armlist = [x for x in ("whole", "nucleus", "hybrid", "protein") if x in set(F.arm)]
    vals = [c for c in ("nhood", "mixing_class", "per100", "log2oe") if c in F]      # a readout no field reached is absent
    if not vals:
        raise SystemExit(f"no field reached --min-t {a.min_t} T cells and --min-p {a.min_p} partner cells; "
                         "lower them, or raise --max-fovs")
    W = F.pivot_table(index=["section", "fov"], columns="arm", values=vals, aggfunc="first")
    flips = []
    for i, x in enumerate(armlist):
        for y in armlist[i + 1:]:
            for (sec), g in W.groupby(level=0):
                for what in ("nhood", "mixing_class"):
                    if (what, x) not in g or (what, y) not in g:
                        continue
                    v = g[[(what, x), (what, y)]].dropna()
                    if what == "mixing_class":
                        v = v[(v[(what, x)] != "cold") | (v[(what, y)] != "cold")]
                    n_flip = int((v[(what, x)] != v[(what, y)]).sum())
                    tr = v.groupby([v[(what, x)].astype(str), v[(what, y)].astype(str)]).size()
                    flips.append(dict(section=sec, readout=what, arm_a=x, arm_b=y, fields=len(v), flips=n_flip,
                                      transitions="; ".join(f"{p}->{q}: {k}" for (p, q), k in tr.items() if p != q)))
    Fl = pd.DataFrame(flips)
    Fl.to_csv(out / "flips.csv", index=False)
    med = F.groupby(["section", "arm"])[["per100", "infil", "log2oe", "mixing"]].median().reset_index()
    med.to_csv(out / "medians.csv", index=False)
    summ = dict(args=vars(a), fields=int(F[["section", "fov"]].drop_duplicates().shape[0]),
                keren=dict(cold=KEREN_COLD, cut=KEREN_CUT))
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 90):
        print("\nmedian over fields (per100: T touching partner per 100 partners; log2oe: T-partner enrichment):")
        print(med.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nflips (fields whose call or class differs between two arms):")
        print(Fl.to_string(index=False) if len(Fl) else "  none")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
