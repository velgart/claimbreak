#!/usr/bin/env python
"""ihc_truth.py -- protein identity (post-Xenium multiplex IHC, CD8 and pan-cytokeratin) as ground truth for the
labels of the same cells.

GSE311609 (Bilous et al. 2026) deposits one OME-TIFF per slide (channels DAPI, Opal 690, Opal 620, Opal 520, Sample AF;
0.496 um / px; pyramid) holding several Xenium sections after the run, without alignment files. Steps:

  overview   low-resolution view: tissue components (bounding boxes in full-resolution pixels, area), a composite PNG
             with component numbers, per-channel PNGs, and density images of Xenium sections at the same scale to
             tell which component is which section
  nuclei     nuclei of one component from DAPI at full resolution (smoothed local maxima, minimum distance 3 um);
             per nucleus the mean of every channel in a disk of 2 um (nucleus) and a ring of 2-5 um (membrane and
             cytoplasm, where CD8 and pan-cytokeratin stain)
  register   IHC nuclei (pixels) onto Xenium nucleus centroids (um; from nucleus_boundaries.parquet) with the
             registration of examples/analysis/protein_truth.py (all orientations, rotations, scales; density
             cross-correlation; mutual nearest neighbors; affine refit); matches within --max-dist um; validity rule
             R0 of protein_truth (matched share of the smaller set >= 0.5 and excess over a 100 um shift >= 0.35)
  truth      protein classes (two-component mixture per Opal channel on the ring intensity; which channel is CD8 and
             which pan-cytokeratin is decided by the rule of assign_roles) and the readouts T2-T5 for every
             label arm present in the labels files

  python ihc_truth.py overview --image IHC.ome.tif --out DIR [--xenium H5AD,...]
  python ihc_truth.py nuclei   --image IHC.ome.tif --components DIR/components.csv --component 2 --out DIR2
  python ihc_truth.py register --nuclei DIR2 --raw RAW --sample GSM9509145_NSCLC_lung_L1_L1 --out DIR3
  python ihc_truth.py truth    --matches DIR3[,DIR3b] --labels LABELS[,LABELS2] --out DIR4
  python ihc_truth.py stains   --nuclei DIR2[,DIR2b] --out DIR5     (label-free channel roles; L6 usability)
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CH = ["DAPI", "Opal 690", "Opal 620", "Opal 520", "Sample AF"]


def _pt():
    p = Path(__file__).resolve().parents[1] / "analysis" / "protein_truth.py"
    spec = importlib.util.spec_from_file_location("protein_truth", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fresh(out):
    out = Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    return out


def image_info(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        s = t.series[0]
        x = t.ome_metadata or ""
        names = []
        import re
        names = re.findall(r'<Channel[^>]*Name="([^"]+)"', x) or CH[: s.shape[0]]
        px = re.findall(r'PhysicalSizeX="([^"]+)"', x)
        um = float(px[0]) if px else 0.4962
        return dict(shape=tuple(s.shape), axes=s.axes, levels=[tuple(l.shape) for l in s.levels], channels=names,
                    um_per_px=um)


def read_level(path, level):
    import tifffile
    with tifffile.TiffFile(path) as t:
        return t.series[0].levels[level].asarray()


def read_crop(path, y0, y1, x0, x1):
    """(C, y, x) crop of the full-resolution level (zarr view when available, else the whole channel)."""
    import tifffile
    try:
        import zarr
        store = tifffile.imread(path, aszarr=True, level=0)
        z = zarr.open(store, mode="r")
        a = np.asarray(z[:, y0:y1, x0:x1])
        store.close()
        return a
    except Exception:
        with tifffile.TiffFile(path) as t:
            return t.series[0].levels[0].asarray()[:, y0:y1, x0:x1]


def to_png(a, path):
    from PIL import Image
    a = np.asarray(a, float)
    hi = np.percentile(a, 99.5) if a.size else 1
    Image.fromarray(np.clip(a / max(hi, 1e-9) * 255, 0, 255).astype(np.uint8)).save(path)


# --------------------------------------------------------------------------- overview

def cmd_overview(a):
    from skimage import filters, measure, morphology
    out = _fresh(a.out)
    info = image_info(a.image)
    L = read_level(a.image, a.level)                         # (C, y, x)
    f = info["shape"][1] / L.shape[1]                         # full-res px per level px
    um = info["um_per_px"] * f
    dapi = filters.gaussian(L[0].astype(float), 2)
    thr = filters.threshold_otsu(dapi)
    m = morphology.remove_small_objects(morphology.closing(dapi > thr, morphology.disk(4)), 64)
    m = morphology.remove_small_holes(m, 10_000)
    lab = measure.label(morphology.dilation(m, morphology.disk(15)))      # sections are well separated
    rows = []
    for r in measure.regionprops(lab):
        area_mm2 = r.area * um * um / 1e6
        if area_mm2 < a.min_area_mm2:
            continue
        y0, x0, y1, x1 = r.bbox
        rows.append(dict(component=r.label, area_mm2=round(area_mm2, 2), y0=int(y0 * f), x0=int(x0 * f),
                         y1=int(y1 * f), x1=int(x1 * f), width_um=round((x1 - x0) * um), height_um=round((y1 - y0) * um),
                         **{f"mean_{n}": float(L[i][lab == r.label].mean()) for i, n in enumerate(info["channels"])}))
    C = pd.DataFrame(rows)
    C.to_csv(out / "components.csv", index=False)
    for i, n in enumerate(info["channels"]):
        to_png(L[i], out / f"channel_{i}_{n.replace(' ', '_')}.png")
    from PIL import Image, ImageDraw
    rgb = np.zeros(L.shape[1:] + (3,), float)
    for i, w in ((0, (0, 0, 1)), (1, (1, 0, 0)), (2, (0, 1, 0)), (3, (1, 0, 1))):
        if i < L.shape[0]:
            ch = L[i].astype(float)
            ch = np.clip(ch / max(np.percentile(ch, 99.5), 1), 0, 1)
            rgb += ch[..., None] * np.array(w)
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for r in rows:
        d.rectangle([r["x0"] / f, r["y0"] / f, r["x1"] / f, r["y1"] / f], outline=(255, 255, 0), width=3)
        d.text((r["x0"] / f + 10, r["y0"] / f + 10), str(r["component"]), fill=(255, 255, 0))
    im.save(out / "composite.png")
    for h in [x for x in (a.xenium or "").split(",") if x]:
        import anndata as ad
        A = ad.read_h5ad(h, backed="r")
        xy = A.obs[["x", "y"]].to_numpy(float)
        H, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[max(1, int(np.ptp(xy[:, 1]) / um)),
                                                          max(1, int(np.ptp(xy[:, 0]) / um))])
        to_png(H, out / f"xenium_{Path(h).stem}.png")
        print(f"{Path(h).stem}: {len(xy):,} cells, {np.ptp(xy[:, 0]):.0f} x {np.ptp(xy[:, 1]):.0f} um", flush=True)
    (out / "image.json").write_text(json.dumps({**info, "level": a.level, "level_um_per_px": um}, indent=1))
    print(json.dumps(info))
    print(C.to_string(index=False))
    print(f"wrote {out}")


# --------------------------------------------------------------------------- nuclei

def disk_offsets(r_in, r_out):
    rr = int(np.ceil(r_out))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    d = np.hypot(yy, xx)
    m = (d >= r_in) & (d <= r_out) if r_in > 0 else d <= r_out
    return yy[m], xx[m]


def sample_means(img, ys, xs, oy, ox, chunk=20000):
    H, W = img.shape
    out = np.empty(len(ys), np.float32)
    for s in range(0, len(ys), chunk):
        Y = np.clip(ys[s:s + chunk, None] + oy[None, :], 0, H - 1)
        X = np.clip(xs[s:s + chunk, None] + ox[None, :], 0, W - 1)
        out[s:s + chunk] = img[Y, X].mean(1)
    return out


def cmd_nuclei(a):
    from skimage import feature, filters
    out = _fresh(a.out)
    info = image_info(a.image)
    C = pd.read_csv(a.components).set_index("component").loc[int(a.component)]
    um = info["um_per_px"]
    pad = int(50 / um)
    y0, x0 = max(0, int(C.y0) - pad), max(0, int(C.x0) - pad)
    y1, x1 = min(info["shape"][1], int(C.y1) + pad), min(info["shape"][2], int(C.x1) + pad)
    I = read_crop(a.image, y0, y1, x0, x1)
    dapi = filters.gaussian(I[0].astype(np.float32), a.sigma_px, preserve_range=True)
    tissue = filters.gaussian(I[0].astype(np.float32), 8, preserve_range=True)
    thr = filters.threshold_otsu(tissue[::4, ::4])
    pk = feature.peak_local_max(dapi, min_distance=max(1, int(round(a.min_dist_um / um))),
                                threshold_abs=float(np.percentile(dapi[tissue > thr], a.peak_pct)),
                                exclude_border=False)
    pk = pk[tissue[pk[:, 0], pk[:, 1]] > thr]
    ys, xs = pk[:, 0], pk[:, 1]
    oy_n, ox_n = disk_offsets(0, a.nuc_um / um)
    oy_r, ox_r = disk_offsets(a.nuc_um / um, a.ring_um / um)
    D = pd.DataFrame({"x_px": xs + x0, "y_px": ys + y0, "dapi_peak": dapi[ys, xs]})
    for i, n in enumerate(info["channels"]):
        k = n.replace(" ", "_")
        D[f"{k}_nuc"] = sample_means(I[i], ys, xs, oy_n, ox_n)
        D[f"{k}_ring"] = sample_means(I[i], ys, xs, oy_r, ox_r)
    D.to_csv(out / "nuclei.csv.gz", index=False, compression="gzip")
    meta = dict(component=int(a.component), crop=[y0, y1, x0, x1], um_per_px=um, n=int(len(D)), sigma_px=a.sigma_px,
                min_dist_um=a.min_dist_um, peak_pct=a.peak_pct, nuc_um=a.nuc_um, ring_um=a.ring_um,
                density_per_mm2=float(len(D) / ((y1 - y0) * (x1 - x0) * um * um / 1e6)))
    (out / "nuclei.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


# --------------------------------------------------------------------------- register

def xenium_nuclei(raw, sample):
    from claimbreak.contacts import read_parquet_any
    p = None
    for suf in ("", ".gz"):
        q = Path(raw) / f"{sample}_nucleus_boundaries.parquet{suf}"
        if q.exists():
            p = q
            break
    if p is None:
        raise FileNotFoundError(f"{sample}_nucleus_boundaries.parquet(.gz) not in {raw}")
    B = read_parquet_any(p)
    B["cell_id"] = B["cell_id"].astype(str)
    g = B.groupby("cell_id", sort=False)[["vertex_x", "vertex_y"]].mean()
    return g.index.to_numpy(), g.to_numpy(float)


def cmd_register(a):
    pt = _pt()
    out = _fresh(a.out)
    N = pd.read_csv(Path(a.nuclei).expanduser() / "nuclei.csv.gz")
    nmeta = json.loads((Path(a.nuclei).expanduser() / "nuclei.json").read_text())
    cid, P = xenium_nuclei(a.raw, a.sample)
    Q = N[["x_px", "y_px"]].to_numpy(float)
    um = nmeta["um_per_px"]
    if a.fix_scale:          # image scale from the OME metadata; only small factors around it are searched
        M, t, info, (pi, qi, d) = pt.register(P, Q, max_dist=a.max_dist, scales=np.arange(0.98, 1.021, 0.01),
                                              bin_um=20.0, fine_bin_um=10.0, s0=um)
    else:
        M, t, info, (pi, qi, d) = pt.register(P, Q, max_dist=a.max_dist, scales=np.arange(0.9, 1.11, 0.02),
                                              bin_um=20.0, fine_bin_um=10.0)
    info["fix_scale"] = bool(a.fix_scale)
    info["um_per_px_image"] = um
    info["sample"] = a.sample
    info["component"] = nmeta["component"]
    Mt = pd.DataFrame({"cell_id": cid[pi], "dist_um": d, "x_xen": P[pi, 0], "y_xen": P[pi, 1]})
    Mt = pd.concat([Mt, N.iloc[qi].reset_index(drop=True)], axis=1)
    Mt.to_csv(out / "matches.csv.gz", index=False, compression="gzip")
    info["M"], info["t"] = np.asarray(M).tolist(), np.asarray(t).tolist()
    (out / "register.json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps({k: info[k] for k in ("sample", "component", "n_cosmx", "n_cellscape", "n_matched",
                                           "matched_share_small", "chance_share_small", "excess_share",
                                           "median_residual_um", "um_per_px", "R0")}, default=float))
    print("(n_cosmx = Xenium nuclei, n_cellscape = IHC nuclei: names from protein_truth.py)")


# --------------------------------------------------------------------------- stains

def cmd_stains(a):
    """Label-free description of each Opal channel on the nuclei of one or more components (no Xenium, no labels), so
    that channel roles can be read from the stain itself and a section without Xenium data (L6) can be judged usable.
    Per channel: Ashman D of the autofluorescence-residual mixture (positive_resid; D >= 2 = separated), positive
    share, ring over nucleus intensity of positive cells (membrane and cytoplasmic stains > 1), and the neighbor
    enrichment of positive cells (share of positive cells among the 6 nearest nuclei within 20 um of a positive cell,
    over the positive share: epithelial sheets cluster strongly, scattered lymphocytes weakly).
    Rule: among separated channels, panCK = the channel with the largest
    positive share; CD8 = among the other separated channels, the one with the smallest positive share whose positive
    cells are less clustered than panCK's. Reported next to the label-based roles; it does not replace them."""
    out = _fresh(a.out)
    rows, roles = [], []
    for d in [Path(x).expanduser() for x in a.nuclei.split(",") if x]:
        N = pd.read_csv(d / "nuclei.csv.gz")
        meta = json.loads((d / "nuclei.json").read_text())
        um = meta["um_per_px"]
        xy = N[["x_px", "y_px"]].to_numpy(float) * um
        dist, idx = cKDTree(xy).query(xy, k=7)
        dist, idx = dist[:, 1:], idx[:, 1:]
        near = dist <= 20.0
        af = N["Sample_AF_ring"].to_numpy(float) if "Sample_AF_ring" in N else None
        opal = [c[:-5] for c in N.columns if c.startswith("Opal") and c.endswith("_ring")]
        for ch in opal:
            v = N[f"{ch}_ring"].to_numpy(float)
            r, b = af_residual(v, af) if af is not None and not a.no_af else (v, 0.0)
            pos, dsep = positive_resid(r)
            share = float(pos.mean())
            nb_pos = (pos[idx] & near).sum(1) / np.maximum(near.sum(1), 1)
            enrich = float(nb_pos[pos].mean() / share) if pos.any() and share > 0 else np.nan
            vn = N[f"{ch}_nuc"].to_numpy(float)
            ring_nuc = float(np.median(v[pos] / np.maximum(vn[pos], 1e-6))) if pos.any() else np.nan
            rows.append(dict(component=meta["component"], nuclei=len(N), channel=ch, af_b=b, ashman_d=dsep,
                             separated=dsep >= 2, positive_share=share, neighbour_enrichment=enrich,
                             ring_over_nucleus=ring_nuc))
        R = pd.DataFrame([x for x in rows if x["component"] == meta["component"]])
        sep = R[R.separated]
        ck = sep.sort_values("positive_share").channel.iloc[-1] if len(sep) else ""
        rest = sep[(sep.channel != ck)]
        if ck and len(rest):
            e_ck = float(sep.loc[sep.channel == ck, "neighbour_enrichment"].iloc[0])
            rest = rest[rest.neighbour_enrichment < e_ck] if np.isfinite(e_ck) else rest
        cd8 = rest.sort_values("positive_share").channel.iloc[0] if len(rest) else ""
        roles.append(dict(component=meta["component"], nuclei=len(N), panCK=ck, CD8=cd8,
                          usable=bool(ck and cd8), separated_channels=int(len(sep))))
    C, Rl = pd.DataFrame(rows), pd.DataFrame(roles)
    C.to_csv(out / "channels.csv", index=False)
    Rl.to_csv(out / "roles.csv", index=False)
    with pd.option_context("display.width", 200):
        print(C.round(3).to_string(index=False))
        print("\nroles from the stain alone (rule in the docstring):\n" + Rl.to_string(index=False))
    print(f"wrote {out}")


# --------------------------------------------------------------------------- truth

IMMUNE = ("T_NK", "Myeloid", "B")
EPITHELIAL = ("Tumour", "Epithelia")          # label types that pan-cytokeratin stains (T5 sensitivity partner)


def af_residual(v, af):
    """Opal ring intensity minus autofluorescence bleed: v - b * AF, b = median of v / AF over cells with AF > 0.
    Not clipped: clipping at 0 put half of all cells at exactly 0 and the mixture then called them all
    positive (synthetic check: 50% positive, false-positive rate 0.46)."""
    ok = af > 0
    b = float(np.median(v[ok] / af[ok])) if ok.any() else 0.0
    return v - b * af, b


def positive_resid(r, seed=0, min_ashman=2.0):
    """Positive cells from an autofluorescence residual: Gaussian mixtures (2 or 3 components, by BIC) on
    arcsinh((r - c) / s), with the noise center c and scale s taken from the lower half of the cells (c = their
    median, s = 2 x 1.4826 x their MAD), so that abundant positive cells (pan-cytokeratin in tumor) do not inflate
    the scale; positive = the highest-mean component (posterior > 0.5) when it is separated from the next one
    (Ashman's D >= min_ashman), else none. Returns (positive, D)."""
    from sklearn.mixture import GaussianMixture
    r = np.asarray(r, float)
    low = r[r <= np.median(r)]
    c = float(np.median(low))
    s = 2 * 1.4826 * float(np.median(np.abs(low - c)))
    x = np.arcsinh((r - c) / max(s, 1e-9)).reshape(-1, 1)
    best = None
    for k in (2, 3):
        g = GaussianMixture(k, random_state=seed, n_init=3).fit(x)
        mu, var = g.means_.ravel(), g.covariances_.ravel()
        o = np.argsort(mu)
        hi, nx = o[-1], o[-2]
        d = float(np.sqrt(2) * abs(mu[hi] - mu[nx]) / np.sqrt(var[hi] + var[nx]))
        bic = g.bic(x)
        if best is None or bic < best[0]:
            best = (bic, g, hi, d)
    _, g, hi, d = best
    pos = (g.predict_proba(x)[:, hi] > 0.5) if d >= min_ashman else np.zeros(len(x), bool)
    return pos, d


def assign_roles(pos, opal, labels):
    """Channel roles: pan-cytokeratin = the largest positive-share ratio in
    label-Tumor over label-immune cells; CD8 = among the rest, the largest ratio in label-T_NK over label-Tumor."""
    def ratio(c, a_, b_):
        pa, pb = pos[c][np.isin(labels, a_)].mean(), pos[c][np.isin(labels, b_)].mean()
        return (pa + 1e-3) / (pb + 1e-3)
    ck = max(opal, key=lambda c: ratio(c, ["Tumour"], list(IMMUNE)))
    cd8 = max([c for c in opal if c != ck], key=lambda c: ratio(c, ["T_NK"], ["Tumour"]))
    return ck, cd8, ratio


def cmd_truth(a):
    pt = _pt()
    out = _fresh(a.out)
    lv = a.level
    mats = [Path(x).expanduser() for x in a.matches.split(",") if x]
    labs = [Path(x).expanduser() for x in a.labels.split(",") if x]
    rows, chan_rows, per_rows, t6_rows, sec_rows, fate_rows, tum_rows = [], [], [], [], [], [], []
    for mdir in mats:
        reg = json.loads((mdir / "register.json").read_text())
        sample = reg["sample"]
        Mt = pd.read_csv(mdir / "matches.csv.gz", dtype={"cell_id": str})
        # matches farther than --max-match-um are dropped (at 4 um in dense tissue about
        # a third of mutual nearest neighbors are chance pairings, which give a cell another cell's protein)
        n0 = len(Mt)
        Mt = Mt[Mt["dist_um"] <= a.max_match_um].reset_index(drop=True)
        print(f"{sample}: {len(Mt):,} of {n0:,} matches within {a.max_match_um} um (R0 {reg.get('R0')})", flush=True)
        L = None
        for ld in labs:
            f = ld / f"{sample}.labels.csv.gz"
            if f.exists():
                X = pd.read_csv(f, keep_default_na=False, low_memory=False, dtype={"cell_id": str})
                L = X if L is None else L.merge(X[["cell_id"] + [c for c in X.columns if c not in L.columns]],
                                                on="cell_id", how="left")
        if L is None:
            raise SystemExit(f"no labels for {sample} in {labs}")
        L = L.fillna("")
        # hybrid arms A+B (A's label where A labeled the cell, else B's), as xenium_protein.py truth --hybrid
        for h in [x for x in (getattr(a, "hybrid", "") or "").split(",") if x]:
            first, second = h.split(":")
            if f"{lv}_{first}" in L and f"{lv}_{second}" in L:
                c1 = L[f"{lv}_{first}"].astype(str)
                L[f"{lv}_{first}+{second}"] = np.where(c1 != "", c1, L[f"{lv}_{second}"].astype(str))
                print(f"  hybrid arm {first}+{second}: {first} where labelled ({(c1 != '').mean():.3f} of cells), "
                      f"else {second}", flush=True)
            else:
                print(f"  --hybrid {h}: no columns {lv}_{first} and {lv}_{second} in the labels of {sample}", flush=True)
        arms = [c[len(lv) + 1:] for c in L.columns if c.startswith(f"{lv}_") and not c.endswith(("_second", "_margin", "_psi"))]
        # protein classes on matched cells: autofluorescence residual + mixture per Opal channel (ring intensity)
        opal = [c for c in Mt.columns if c.startswith("Opal") and c.endswith("_ring")]
        af = Mt["Sample_AF_ring"].to_numpy(float) if "Sample_AF_ring" in Mt else None
        pos = {}
        for c in opal:
            v = Mt[c].to_numpy(float)
            if af is not None and not a.no_af:
                r, b = af_residual(v, af)
                p, dsep = positive_resid(r)
                method = "af_residual_mixture"
            else:
                b = 0.0
                p, dsep = pt.marker_positive(v, return_d=True)
                method = "log_mixture"
            pos[c] = p
            chan_rows.append(dict(sample=sample, channel=c, method=method, af_b=b, ashman_d=dsep,
                                  positive_share=float(p.mean())))
        lab_of = L.set_index("cell_id")
        nl = Mt.cell_id.map(lab_of[f"{lv}_nucleus"]).fillna("").to_numpy()
        wl = Mt.cell_id.map(lab_of[f"{lv}_whole"]).fillna("").to_numpy() if f"{lv}_whole" in L else nl
        # channel roles from nucleus labels and from
        # whole-cell labels; and the section is evaluable only when both give the same roles
        ck, cd8, ratio = assign_roles(pos, opal, nl)
        ck_w, cd8_w, _ = assign_roles(pos, opal, wl)
        roles_agree = (ck == ck_w) and (cd8 == cd8_w)
        dmap = {r["channel"]: r["ashman_d"] for r in chan_rows if r["sample"] == sample}
        separated = dmap[ck] >= 2 and dmap[cd8] >= 2
        for r in chan_rows:
            if r["sample"] == sample:
                r["role"] = "panCK" if r["channel"] == ck else ("CD8" if r["channel"] == cd8 else "")
                r["role_from_whole"] = "panCK" if r["channel"] == ck_w else ("CD8" if r["channel"] == cd8_w else "")
                r["ratio_tumour_vs_immune"] = ratio(r["channel"], ["Tumour"], list(IMMUNE))
                r["ratio_tnk_vs_tumour"] = ratio(r["channel"], ["T_NK"], ["Tumour"])
        evaluable = bool(roles_agree and separated)
        sec_rows.append(dict(sample=sample, matched=len(Mt), panCK=ck, CD8=cd8, roles_agree=roles_agree,
                             channels_separated=separated, evaluable=evaluable))
        # protein classes: CD8 (CD8+ panCK-, as written), CD8any (CD8+ regardless of panCK; sensitivity: a CD8 cell's
        # ring overlaps a touching tumor cell's membrane), panCK (panCK+ CD8-)
        pcls = np.where(pos[cd8] & ~pos[ck], "CD8", np.where(pos[ck] & ~pos[cd8], "panCK", "other"))
        prot = pd.Series(pcls, index=Mt.cell_id.to_numpy())
        cd8any = pd.Series(pos[cd8], index=Mt.cell_id.to_numpy())
        xy = L[["x", "y"]].to_numpy(float)
        d_, idx = cKDTree(xy).query(xy, k=a.k + 1)
        d_, idx = d_[:, 1:], idx[:, 1:]
        pc = L.cell_id.map(prot).fillna("").to_numpy()
        anyc = L.cell_id.map(cd8any).fillna(False).to_numpy(bool)
        matched = pc != ""
        touch_ck = ((pc[idx] == "panCK") & (d_ <= a.r)).any(1)
        classes = {"CD8": pc == "CD8", "CD8any": matched & anyc}
        # CD8iso = CD8any cells with no other CD8any cell within --iso-um (the 2-5 um ring of a cell next to a
        # CD8 cell can pick up that cell's membrane stain; isolated CD8+ cells cannot owe their call to a neighbor)
        cpos = np.flatnonzero(classes["CD8any"])
        if len(cpos) > 1:
            dn, _ = cKDTree(xy[cpos]).query(xy[cpos], k=2)
            iso = np.zeros(len(L), bool)
            iso[cpos[dn[:, 1] > a.iso_um]] = True
            classes["CD8iso"] = iso
        ckpos = L.cell_id.map(pd.Series(pos[ck], index=Mt.cell_id.to_numpy())).fillna(False).to_numpy(bool)
        fate_classes = dict(classes, panCK=pc == "panCK", panCKany=matched & ckpos)
        for arm in arms:
            x = L[f"{lv}_{arm}"].astype(str).to_numpy()
            for cname, cm in classes.items():
                for stratum, m in (("touching_panCK", cm & touch_ck), ("not_touching", cm & ~touch_ck), ("all", cm)):
                    ok = m & (x != "")
                    rows.append(dict(sample=sample, arm=arm, protein_class=cname, stratum=stratum, cd8_cells=int(m.sum()),
                                     labelled=int(ok.sum()),
                                     called_T_NK=float(np.mean(x[ok] == "T_NK")) if ok.any() else np.nan,
                                     # unlabeled cells counted as misses (all protein cells of the class)
                                     called_T_NK_all=float(np.mean(x[m] == "T_NK")) if m.any() else np.nan,
                                     called_tumour=float(np.mean(x[ok] == "Tumour")) if ok.any() else np.nan,
                                     evaluable=evaluable))
            # what the protein classes are labeled (full label distribution), and how many cells of each
            # label are panCK+ / CD8+ by protein
            for cname, cm in fate_classes.items():
                strata = (("touching_panCK", cm & touch_ck), ("not_touching", cm & ~touch_ck), ("all", cm))
                for stratum, m in strata:
                    ok = m & (x != "")
                    if not ok.any():
                        continue
                    vc = pd.Series(x[ok]).value_counts()
                    for lab_, n_ in vc.items():
                        fate_rows.append(dict(sample=sample, arm=arm, protein_class=cname, stratum=stratum, label=lab_,
                                              n=int(n_), share=float(n_ / ok.sum()), evaluable=evaluable))
            okm = matched & (x != "")
            for lab_ in pd.unique(x[okm]):
                m = okm & (x == lab_)
                tum_rows.append(dict(sample=sample, arm=arm, label=lab_, n=int(m.sum()),
                                     panCK_pos=float(ckpos[m].mean()), CD8_pos=float(anyc[m].mean()),
                                     touching_panCK=float(touch_ck[m].mean()), evaluable=evaluable))
            # T6: on matched cells labeled in this arm, recall of protein CD8 cells as T_NK and the share
            # of T_NK-labeled cells that are protein CD8 (descriptive: T_NK also holds CD4 T and NK cells)
            ok = matched & (x != "")
            for cname, cm in classes.items():
                t6_rows.append(dict(sample=sample, arm=arm, protein_class=cname, matched_labelled=int(ok.sum()),
                                    protein_cd8=int((cm & ok).sum()), labelled_T_NK=int(((x == "T_NK") & ok).sum()),
                                    recall=float(np.mean(x[cm & ok] == "T_NK")) if (cm & ok).any() else np.nan,
                                    # matched protein CD8 cells unlabeled in the arm counted as misses
                                    recall_all=(float(np.mean(x[cm & matched] == "T_NK"))
                                                if (cm & matched).any() else np.nan),
                                    labelled_share=(float(ok[cm & matched].mean()) if (cm & matched).any()
                                                    else np.nan),
                                    cd8_share_of_T_NK=float(np.mean(cm[(x == "T_NK") & ok])) if ((x == "T_NK") & ok).any() else np.nan,
                                    T_NK_per_100_cd8=(100 * float(((x == "T_NK") & ok).sum()) / int((cm & ok).sum())
                                                      if (cm & ok).any() else np.nan),
                                    evaluable=evaluable))
            # per100 among matched cells (T5): labels with partner Tumor or Tumor + Epithelia
            # (sensitivity: pan-cytokeratin also stains normal epithelium)
            # infil (T5b): share of T_NK cells touching the partner; scale-free, so it can be set against
            # the protein share of CD8 cells touching panCK+ (protein CD8 is a subset of T_NK, so per100 is not)
            for pname, ptypes in (("Tumour", ("Tumour",)), ("Tumour+Epithelia", EPITHELIAL)):
                isb = np.isin(x, ptypes) & ok
                tt = (np.isin(x[idx], ptypes) & (d_ <= a.r) & ok[idx]).any(1)
                tn = (x == "T_NK") & ok
                per_rows.append(dict(sample=sample, arm=arm, partner=pname, n_partner=int(isb.sum()),
                                     per100_labels=100 * float((tn & tt).sum()) / max(int(isb.sum()), 1),
                                     infil=float((tn & tt).sum() / tn.sum()) if tn.any() else np.nan,
                                     evaluable=evaluable))
        n_ck = int((pc == "panCK").sum())
        for cname, cm in classes.items():
            per_rows.append(dict(sample=sample, arm=f"protein_{cname}", partner="panCK", n_partner=n_ck,
                                 per100_labels=100 * float((cm & touch_ck).sum()) / max(n_ck, 1),
                                 infil=float((cm & touch_ck).sum() / cm.sum()) if cm.any() else np.nan,
                                 evaluable=evaluable))
        print(f"{sample}: {len(Mt):,} matched cells; panCK = {ck}, CD8 = {cd8} (from whole-cell labels: {ck_w}, "
              f"{cd8_w}; agree {roles_agree}); separated {separated}; protein CD8 {int(classes['CD8'].sum()):,}, "
              f"CD8any {int(classes['CD8any'].sum()):,} (touching panCK {int((classes['CD8'] & touch_ck).sum()):,}), "
              f"panCK {n_ck:,}; evaluable {evaluable}", flush=True)
    if not rows:
        raise SystemExit("no section to evaluate")
    R, Ch, Pp, T6, Sx = (pd.DataFrame(x) for x in (rows, chan_rows, per_rows, t6_rows, sec_rows))
    F, U = pd.DataFrame(fate_rows), pd.DataFrame(tum_rows)
    F.to_csv(out / "fate.csv", index=False)
    U.to_csv(out / "label_protein.csv", index=False)
    R.to_csv(out / "cd8_calls.csv", index=False)
    Ch.to_csv(out / "channels.csv", index=False)
    Pp.to_csv(out / "per100.csv", index=False)
    T6.to_csv(out / "t6_counts.csv", index=False)
    Sx.to_csv(out / "sections.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(matches=[str(m) for m in mats], labels=[str(l) for l in labs],
                                                   level=lv, k=a.k, r=a.r, no_af=bool(a.no_af),
                                                   max_match_um=a.max_match_um, iso_um=a.iso_um), indent=1))
    pd.set_option("display.width", 220)
    print("\nsections:\n" + Sx.to_string(index=False))
    print("\nchannels:\n" + Ch.round(3).to_string(index=False))
    print("\nprotein CD8 cells: share called T_NK by each label arm (T2, T3)\n"
          + R.pivot_table(index=["sample", "protein_class", "arm"], columns="stratum", values="called_T_NK").round(3).to_string())
    print("\nT6: recall of protein CD8 cells as T_NK, and T_NK labels per 100 protein CD8 cells (matched cells)\n"
          + T6.pivot_table(index=["sample", "protein_class", "arm"],
                           values=["recall", "recall_all", "labelled_share", "T_NK_per_100_cd8"]).round(3).to_string())
    print("\nper100 (matched cells; protein rows = CD8 touching panCK+ per 100 panCK+) (T5) and infiltration share (T5b)\n"
          + Pp.pivot_table(index=["partner", "arm"], columns="sample", values=["per100_labels", "infil"]).round(3).to_string())
    for cname in ("CD8any", "panCKany"):
        f = F[(F.protein_class == cname) & (F.stratum == "all")]
        if len(f):
            top = f.groupby("label").n.sum().sort_values(ascending=False).index[:7]
            print(f"\nlabels of protein {cname} cells (share per arm; fate.csv has all labels and strata)\n"
                  + f[f.label.isin(top)].pivot_table(index=["sample", "arm"], columns="label", values="share")
                  .fillna(0).round(3).to_string())
    if len(U):
        print("\nshare panCK+ (protein) among cells of each label (label_protein.csv also has CD8+)\n"
              + U.pivot_table(index=["sample", "label"], columns="arm", values="panCK_pos").round(3).to_string())
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overview")
    o.add_argument("--image", required=True); o.add_argument("--out", required=True)
    o.add_argument("--level", type=int, default=3); o.add_argument("--min-area-mm2", type=float, default=1.0)
    o.add_argument("--xenium", default="", help="comma list of Xenium .h5ad to render at the same scale")
    n = sub.add_parser("nuclei")
    n.add_argument("--image", required=True); n.add_argument("--components", required=True)
    n.add_argument("--component", required=True); n.add_argument("--out", required=True)
    n.add_argument("--sigma-px", type=float, default=1.5); n.add_argument("--min-dist-um", type=float, default=3.0)
    n.add_argument("--peak-pct", type=float, default=50.0, help="peak threshold: percentile of smoothed DAPI in tissue")
    n.add_argument("--nuc-um", type=float, default=2.0); n.add_argument("--ring-um", type=float, default=5.0)
    r = sub.add_parser("register")
    r.add_argument("--nuclei", required=True); r.add_argument("--raw", required=True)
    r.add_argument("--sample", required=True); r.add_argument("--out", required=True)
    r.add_argument("--max-dist", type=float, default=4.0)
    r.add_argument("--fix-scale", action="store_true",
                   help="take the image scale from the OME metadata (nuclei.json) and search only +-2%% around it")
    st = sub.add_parser("stains", help="label-free channel description and roles on nuclei of one or more components")
    st.add_argument("--nuclei", required=True, help="comma list of nuclei folders (written by `nuclei`)")
    st.add_argument("--out", required=True); st.add_argument("--no-af", action="store_true")
    t = sub.add_parser("truth")
    t.add_argument("--matches", required=True); t.add_argument("--labels", required=True)
    t.add_argument("--out", required=True); t.add_argument("--level", default="L2m")
    t.add_argument("--k", type=int, default=6); t.add_argument("--r", type=float, default=30.0)
    t.add_argument("--allow-invalid", action="store_true", help="ignored (kept for older command lines)")
    t.add_argument("--max-match-um", type=float, default=2.0, help="drop IHC-Xenium matches farther than this")
    t.add_argument("--no-af", action="store_true", help="no autofluorescence subtraction (sensitivity)")
    t.add_argument("--iso-um", type=float, default=8.0, help="CD8iso: no other CD8+ cell within this distance")
    t.add_argument("--hybrid", default="", help="A:B pairs: add the arm A+B, A's label where A labelled the "
                                                "cell, else B's (e.g. nucleus:whole)")
    a = ap.parse_args()
    {"overview": cmd_overview, "nuclei": cmd_nuclei, "register": cmd_register, "stains": cmd_stains,
     "truth": cmd_truth}[a.cmd](a)


if __name__ == "__main__":
    main()
