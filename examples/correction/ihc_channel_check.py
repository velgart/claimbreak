#!/usr/bin/env python
"""ihc_channel_check.py -- which protein does each deposited Opal channel of the lung IHC carry?

Why: GSE311609 deposits one OME-TIFF with the channels DAPI, Opal 690, Opal 620, Opal 520 and Sample AF. The authors'
panel description lists a six-plex: CD31 Opal 480, Integrin alphaV Opal 520, FASL Opal 570,
CD105 Opal 620, panCK Opal 690, CD8 Opal 780. Read with the deposited names, the channel used as CD8 (Opal 520)
would be Integrin alphaV, and CD8 would not be in the file. This script reads each channel against the Xenium
transcripts of the same cells, so the answer does not depend on the channel names.

Inputs: the registrations written by `ihc_truth.py register` (matches.csv.gz: one row per IHC nucleus matched to a
Xenium nucleus, with the ring and nucleus intensity of every channel) and the section .h5ad files written by
`bilous_to_h5ad.py` (X = whole-cell counts, layers['nucleus'] = nuclear counts, obs x, y; obs_names = cell_id).
No labels are used.

Per section, per channel, the autofluorescence residual of the ring intensity (2-5 um around the nucleus) and the
positive calls of ihc_truth.py (same functions, same defaults). Then, for whole-cell and for nuclear counts:

  classes.csv   share of channel-positive cells in cell classes defined from transcripts only (each cell in at most
                one class): CD8 T (CD3D/CD3E/CD2/CD247 >= 1, CD8A/CD8B >= 1, CD4 = 0), CD4 T (T >= 1, CD4 >= 1,
                CD8 = 0), NK-like (NKG7/KLRD1/GNLY >= 1, no T, no CD8), B, myeloid, endothelial, fibroblast,
                epithelial (keratins/EPCAM >= 3). Thresholds in CLASSES; cells meeting none or several are "other".
                The channel's own positive share over all matched cells is printed alongside.
  targets.csv   for each channel and each panel target (CD8: CD8A+CD8B; Integrin alphaV: ITGAV; CD105: ENG;
                panCK: keratins; CD31: PECAM1; FASL: FASLG), the AUC of the residual for cells with >= 1 count of the
                target against cells with none, computed within quintiles of total counts and averaged (weights =
                cells), so that bigger cells with more counts and more stain do not make every gene look specific.
                Also the Spearman correlation of the residual with the cell's own target count per 1,000 and with the
                mean of its neighbors (up to 6 Xenium cells within 15 um): a stain that belongs to the cell follows
                its own transcripts more than its neighbors'.
  tcells.csv    within T cells only: AUC of the residual for CD8 T against CD4 T cells (depth quintiles), with the
                positive share of each. Integrin alphaV, CD105 or panCK would not separate CD8 from CD4 T cells.
  ome_channels.txt   (with --image) every <Channel> element of the OME-XML with all attributes, filter, light-source
                and detector elements, annotations, and every marker or fluorophore word found in the header.

  python ihc_channel_check.py --matches REG_L3,REG_L1 --h5ad H5AD_DIR --out OUT [--image IHC.ome.tif]
"""
import argparse
import importlib.util
import warnings
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

T_GENES = ["CD3D", "CD3E", "CD2", "CD247"]
SETS = {
    "T": T_GENES,
    "CD8": ["CD8A", "CD8B"],
    "CD4": ["CD4"],
    "NK": ["NKG7", "KLRD1", "GNLY"],
    "B": ["MS4A1", "CD79A", "CD19"],
    "myeloid": ["CD68", "CD163", "LYZ", "C1QA", "CD14"],
    "endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "keratin": ["KRT8", "KRT18", "KRT19", "KRT7", "KRT5", "KRT17", "EPCAM"],
    "ITGAV": ["ITGAV"],
    "ENG": ["ENG"],
    "PECAM1": ["PECAM1"],
    "FASLG": ["FASLG"],
}
# panel targets of the authors' six-plex -> transcript sets
TARGETS = {"CD8 (CD8A+CD8B)": "CD8", "Integrin aV (ITGAV)": "ITGAV", "CD105 (ENG)": "ENG",
           "panCK (keratins)": "keratin", "CD31 (PECAM1)": "PECAM1", "FASL (FASLG)": "FASLG"}


def _classes(S):
    """Cell classes from transcript set sums S[set] (counts); a cell meeting several definitions is 'other'."""
    T, C8, C4, epi = S["T"], S["CD8"], S["CD4"], S["keratin"]
    d = {
        "CD8 T": (T >= 1) & (C8 >= 1) & (C4 == 0) & (epi == 0),
        "CD4 T": (T >= 1) & (C4 >= 1) & (C8 == 0) & (epi == 0),
        "NK-like": (S["NK"] >= 1) & (T == 0) & (C8 == 0) & (epi == 0),
        "B": (S["B"] >= 1) & (T == 0) & (epi == 0),
        "myeloid": (S["myeloid"] >= 2) & (T == 0) & (epi == 0),
        "endothelial": (S["endothelial"] >= 2) & (T == 0) & (epi == 0),
        "fibroblast": (S["fibroblast"] >= 3) & (T == 0) & (epi == 0),
        "epithelial": (epi >= 3) & (T == 0),
    }
    M = np.column_stack(list(d.values()))
    one = M.sum(1) == 1
    cls = np.full(len(T), "other", dtype=object)
    for j, k in enumerate(d):
        cls[one & M[:, j]] = k
    return cls


def _truth_mod():
    p = Path(__file__).resolve().parent / "ihc_truth.py"
    spec = importlib.util.spec_from_file_location("ihc_truth", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def auc(x, y):
    """P(x > y) + 0.5 P(x = y) by ranks; nan when either group is empty."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    r = rankdata(np.r_[x, y])
    return float((r[: len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def auc_strat(v, pos, depth, q=5, min_n=10):
    """AUC of v for pos against ~pos within depth quantiles, weighted by cells; (auc, n_pos, n_neg)."""
    edges = np.unique(np.quantile(depth, np.linspace(0, 1, q + 1)))
    b = np.clip(np.searchsorted(edges, depth, side="right") - 1, 0, max(len(edges) - 2, 0))
    num = den = 0.0
    for k in np.unique(b):
        s = b == k
        a_, n1, n0 = auc(v[s & pos], v[s & ~pos]), int((s & pos).sum()), int((s & ~pos).sum())
        if n1 >= min_n and n0 >= min_n and np.isfinite(a_):
            w = n1 + n0
            num += w * a_
            den += w
    return (num / den if den else np.nan), int(pos.sum()), int((~pos).sum())


def load_h5ad(h5dir, sample, genes):
    import anndata as ad
    hits = sorted(Path(h5dir).expanduser().glob(f"{sample}*.h5ad"))
    if not hits:
        raise SystemExit(f"no {sample}*.h5ad in {h5dir}")
    A = ad.read_h5ad(hits[0])
    have = [g for g in genes if g in A.var_names]
    tot = {"whole": np.asarray(A.X.sum(1)).ravel()}
    has_nuc = "nucleus" in A.layers
    if has_nuc:
        tot["nucleus"] = np.asarray(A.layers["nucleus"].sum(1)).ravel()
    B = A[:, have]
    lay = {"whole": B.X}
    if has_nuc:
        lay["nucleus"] = B.layers["nucleus"]
    lay = {k: (v.toarray() if hasattr(v, "toarray") else np.asarray(v)).astype(np.float32) for k, v in lay.items()}
    xy = A.obs[["x", "y"]].to_numpy(float)
    return dict(path=str(hits[0]), cell_id=A.obs_names.astype(str).to_numpy(), genes=have, counts=lay, totals=tot,
                xy=xy, missing=[g for g in genes if g not in A.var_names])


def ome_dump(path, out):
    import tifffile
    lines = []
    with tifffile.TiffFile(path) as t:
        x = t.ome_metadata or ""
        p0 = t.pages[0]
        for tag in ("Software", "Artist", "DateTime", "Make", "Model"):
            if tag in p0.tags:
                lines.append(f"tiff tag {tag}: {p0.tags[tag].value}")
        lines.append(f"series 0 shape {t.series[0].shape} axes {t.series[0].axes}")
        desc = [pg.description for pg in t.pages[:12] if getattr(pg, "description", "")]
    for i, ch in enumerate(re.findall(r"<Channel\b[^>]*/?>", x)):
        attrs = dict(re.findall(r'([A-Za-z:]+)="([^"]*)"', ch))
        lines.append(f"channel {i}: {attrs}")
    for el in ("Filter", "FilterSet", "Dichroic", "LightSource", "Laser", "LightEmittingDiode", "Arc", "Detector",
               "Objective", "Instrument", "LightPath", "EmissionFilterRef", "ExcitationFilterRef"):
        for m in re.findall(rf"<{el}\b[^>]*/?>", x):
            lines.append(f"{el}: {m[:300]}")
    for el in ("MapAnnotation", "XMLAnnotation", "CommentAnnotation"):
        for m in re.findall(rf"<{el}\b.*?</{el}>", x, re.S):
            lines.append(f"{el}: {m[:3000]}")
    words = r"(?i)\b(opal\s*\d{3}|cd8a?|cd105|endoglin|cd31|pecam|fasl|cd178|integrin|itgav|alpha\s*v|pan-?ck|" \
            r"pan-?cytokeratin|cytokeratin|dapi|autofluorescence|sample af|unmix\w*|vectra|phenoimager|polaris|" \
            r"inform|akoya|\d{3}\s*nm)\b"
    alltxt = x + "\n".join(d for d in desc if d and d != x)
    hits = sorted(set(m.strip() for m in re.findall(words, alltxt)))
    lines.append("marker / fluorophore / instrument words in the header: " + (", ".join(hits) if hits else "none"))
    for i, d in enumerate(desc):
        if d and d != x and not d.lstrip().startswith("<?xml"):
            lines.append(f"page {i} ImageDescription (first 1500 chars): {d[:1500]}")
    lines.append(f"OME-XML length {len(x):,} characters")
    (out / "ome_channels.txt").write_text("\n".join(lines) + "\n")
    (out / "ome_metadata.xml").write_text(x)
    return lines


def neighbour_mean(xy_all, val_all, xy_q, self_idx, k=6, r=15.0):
    """Mean of val over up to k other cells within r um of each query cell (nan when none)."""
    d, idx = cKDTree(xy_all).query(xy_q, k=k + 1, distance_upper_bound=r)
    ok = np.isfinite(d) & (idx != self_idx[:, None]) & (idx < len(xy_all))
    idx = np.where(ok, idx, 0)
    v = np.where(ok, val_all[idx], 0.0)
    n = ok.sum(1)
    return np.where(n > 0, v.sum(1) / np.maximum(n, 1), np.nan)


def run(a):
    warnings.filterwarnings("ignore", message="An input array is constant")
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    tm = _truth_mod()
    genes = sorted({g for v in SETS.values() for g in v})
    meta = dict(max_match_um=a.max_match_um, no_af=bool(a.no_af), nb_k=a.nb_k, nb_r=a.nb_r, sections=[])
    if a.image:
        print("OME metadata of the deposited image:")
        for ln in ome_dump(a.image, out):
            print("  " + ln[:400])
        print()
    crows, trows, krows, chrows = [], [], [], []
    for mdir in [Path(x).expanduser() for x in a.matches.split(",") if x]:
        reg = json.loads((mdir / "register.json").read_text())
        sample = reg["sample"]
        Mt = pd.read_csv(mdir / "matches.csv.gz", dtype={"cell_id": str})
        n0 = len(Mt)
        Mt = Mt[Mt["dist_um"] <= a.max_match_um].reset_index(drop=True)
        H = load_h5ad(a.h5ad, sample, genes)
        pos_of = pd.Series(np.arange(len(H["cell_id"])), index=H["cell_id"])
        ii = Mt.cell_id.map(pos_of)
        keep = ii.notna().to_numpy()
        Mt, ii = Mt[keep].reset_index(drop=True), ii[keep].astype(int).to_numpy()
        print(f"{sample}: {len(Mt):,} matched cells within {a.max_match_um} um (of {n0:,} matches); h5ad {H['path']}")
        if H["missing"]:
            print(f"  genes not in the panel: {', '.join(H['missing'])}")
        meta["sections"].append(dict(sample=sample, matches=str(mdir), h5ad=H["path"], cells=len(Mt),
                                     missing_genes=H["missing"]))
        opal = [c for c in Mt.columns if c.startswith("Opal") and c.endswith("_ring")]
        af = Mt["Sample_AF_ring"].to_numpy(float) if "Sample_AF_ring" in Mt else None
        res, pos = {}, {}
        for c in opal:
            v = Mt[c].to_numpy(float)
            r, b = tm.af_residual(v, af) if (af is not None and not a.no_af) else (v, 0.0)
            p, dsep = tm.positive_resid(r)
            name = c[:-5].replace("_", " ")
            res[name], pos[name] = r, p
            chrows.append(dict(sample=sample, channel=name, af_b=b, ashman_d=dsep, positive_share=float(p.mean()),
                               positive_n=int(p.sum())))
        gi = {g: j for j, g in enumerate(H["genes"])}
        for layer, C in H["counts"].items():
            Ssum_all = {k: C[:, [gi[g] for g in v if g in gi]].sum(1) if any(g in gi for g in v)
                        else np.zeros(len(C), np.float32) for k, v in SETS.items()}
            S = {k: v[ii] for k, v in Ssum_all.items()}
            depth_all = H["totals"][layer]
            depth = depth_all[ii]
            cls = _classes(S)
            for ch in res:
                for k in ["all"] + sorted(set(cls)):
                    s = np.ones(len(cls), bool) if k == "all" else cls == k
                    crows.append(dict(sample=sample, counts=layer, channel=ch, cell_class=k, cells=int(s.sum()),
                                      positive_share=float(pos[ch][s].mean()) if s.any() else np.nan,
                                      median_residual=float(np.median(res[ch][s])) if s.any() else np.nan))
                tc = (cls == "CD8 T") | (cls == "CD4 T")
                au, n8, n4 = auc_strat(res[ch][tc], (cls == "CD8 T")[tc], depth[tc])
                krows.append(dict(sample=sample, counts=layer, channel=ch, auc_cd8_vs_cd4=au, cd8_t=n8, cd4_t=n4,
                                  pos_share_cd8_t=float(pos[ch][cls == "CD8 T"].mean()) if n8 else np.nan,
                                  pos_share_cd4_t=float(pos[ch][cls == "CD4 T"].mean()) if n4 else np.nan))
                for tname, sk in TARGETS.items():
                    if not any(g in gi for g in SETS[sk]):
                        continue
                    own = S[sk]
                    au, n1, n0_ = auc_strat(res[ch], own >= 1, depth)
                    per_all = Ssum_all[sk] / np.maximum(depth_all, 1) * 1000
                    nb = neighbour_mean(H["xy"], per_all, H["xy"][ii], ii, a.nb_k, a.nb_r)
                    ok = np.isfinite(nb)
                    rho_own = spearmanr(res[ch], per_all[ii]).correlation
                    rho_nb = spearmanr(res[ch][ok], nb[ok]).correlation if ok.sum() > 10 else np.nan
                    trows.append(dict(sample=sample, counts=layer, channel=ch, target=tname, auc=au,
                                      cells_with=n1, cells_without=n0_, rho_own=rho_own, rho_neighbours=rho_nb))
    Ch, Cl, Tg, Tc = map(pd.DataFrame, (chrows, crows, trows, krows))
    Ch.to_csv(out / "channels.csv", index=False)
    Cl.to_csv(out / "classes.csv", index=False)
    Tg.to_csv(out / "targets.csv", index=False)
    Tc.to_csv(out / "tcells.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\nchannels (autofluorescence residual of the ring; positive calls as in ihc_truth.py truth):\n"
          + Ch.round(3).to_string(index=False))
    for layer in Cl["counts"].unique():
        c = Cl[Cl["counts"] == layer]
        print(f"\n[{layer} counts] positive share of each channel by transcript class (cells per class "
              "below)\n" + c.pivot_table(index=["sample", "channel"], columns="cell_class", values="positive_share")
              .round(3).to_string())
        n = c.drop_duplicates(["sample", "cell_class"]).pivot_table(index="sample", columns="cell_class", values="cells")
        print("cells per class:\n" + n.astype(int).to_string())
        t = Tg[Tg["counts"] == layer]
        print(f"\n[{layer} counts] AUC of each channel for cells with the panel target's transcripts (within depth "
              "quintiles; 0.5 = no relation)\n"
              + t.pivot_table(index=["sample", "channel"], columns="target", values="auc").round(3).to_string())
        print(f"\n[{layer} counts] Spearman rho of the channel with the cell's own target per 1,000 / with its "
              "neighbours' mean\n"
              + t.assign(own_nb=t.rho_own.round(3).astype(str) + " / " + t.rho_neighbours.round(3).astype(str))
              .pivot_table(index=["sample", "channel"], columns="target", values="own_nb", aggfunc="first").to_string())
        k = Tc[Tc["counts"] == layer]
        print(f"\n[{layer} counts] T cells only: CD8 T against CD4 T\n"
              + k.drop(columns="counts").round(3).to_string(index=False))
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", required=True, help="comma list of registration folders (ihc_truth.py register)")
    ap.add_argument("--h5ad", required=True, help="folder of <sample>.h5ad (bilous_to_h5ad.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default="", help="the deposited OME-TIFF, for its channel metadata")
    ap.add_argument("--max-match-um", type=float, default=2.0)
    ap.add_argument("--no-af", action="store_true", help="no autofluorescence subtraction (sensitivity)")
    ap.add_argument("--nb-k", type=int, default=6)
    ap.add_argument("--nb-r", type=float, default=15.0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
