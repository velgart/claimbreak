"""IHC truth chain (examples/correction/ihc_truth.py): autofluorescence residual and the truth step."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("ihc_truth", ROOT / "examples" / "correction" / "ihc_truth.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _channel(rng, n, share, afk):
    af = rng.lognormal(5, 0.5, n)
    pos = rng.random(n) < share
    v = afk * af * rng.lognormal(0, 0.15, n) + rng.lognormal(3, 0.5, n) + np.where(pos, rng.lognormal(6.5, 0.4, n), 0)
    return v, af, pos


def test_af_residual_does_not_call_half_positive():
    m = _mod()
    rng = np.random.default_rng(0)
    v, af, pos = _channel(rng, 6000, 0.08, 0.8)
    r, b = m.af_residual(v, af)
    assert np.mean(r == 0) < 0.01                       # guards against clipping half of the cells to exactly 0
    p, d = m.positive_resid(r)
    assert d >= 2 and p[pos].mean() > 0.9 and p[~pos].mean() < 0.02


def test_positive_resid_none_when_not_separable():
    m = _mod()
    rng = np.random.default_rng(1)
    r = rng.normal(0, 1, 5000)
    p, d = m.positive_resid(r)
    assert p.sum() == 0 or d >= 2


def test_truth_end_to_end(tmp_path):
    m = _mod()
    rng = np.random.default_rng(3)
    n = 5000
    x, y = rng.uniform(0, 1200, n), rng.uniform(0, 600, n)
    t = np.where(x < 550, "Tumour", "Fibro_muscle").astype(object)
    u = rng.random(n)
    t[u < 0.12] = "T_NK"
    t[(u >= 0.12) & (u < 0.2)] = "Myeloid"
    xy = np.c_[x, y]
    d, idx = cKDTree(xy).query(xy, k=7)
    touch = ((t[idx[:, 1:]] == "Tumour") & (d[:, 1:] <= 30)).any(1)
    whole = t.copy()
    whole[(t == "T_NK") & touch & (rng.random(n) < 0.5)] = "Tumour"
    cid = np.array([f"c{i}" for i in range(n)])
    lab = tmp_path / "labels"
    lab.mkdir()
    pd.DataFrame({"cell_id": cid, "x": x, "y": y, "L2m_nucleus": t, "L2m_whole": whole}).to_csv(
        lab / "S_L1_L1.labels.csv.gz", index=False)
    af = rng.lognormal(5, 0.5, n)
    ck = np.where(t == "Tumour", rng.lognormal(7, 0.4, n), rng.lognormal(3, 0.5, n))
    ck = ck + np.where((t == "T_NK") & touch, rng.lognormal(6, 0.5, n), 0)
    cd8 = 0.9 * af * rng.lognormal(0, 0.15, n) + rng.lognormal(3, 0.5, n) + np.where(
        (t == "T_NK") & (rng.random(n) < 0.7), rng.lognormal(6.3, 0.4, n), 0)
    reg = tmp_path / "reg"
    reg.mkdir()
    pd.DataFrame({"cell_id": cid, "dist_um": 0.5, "Opal_690_ring": ck, "Opal_620_ring": 0.3 * af + rng.lognormal(4, 0.6, n),
                  "Opal_520_ring": cd8, "Sample_AF_ring": af}).to_csv(reg / "matches.csv.gz", index=False)
    (reg / "register.json").write_text(json.dumps({"sample": "S_L1_L1", "R0": "valid"}))
    out = tmp_path / "out"
    m.cmd_truth(SimpleNamespace(matches=str(reg), labels=str(lab), out=str(out), level="L2m", k=6, r=30.0,
                                allow_invalid=False, no_af=False, max_match_um=2.0, iso_um=8.0))
    S = pd.read_csv(out / "sections.csv")
    assert bool(S.evaluable.iloc[0]) and S.panCK.iloc[0] == "Opal_690_ring" and S.CD8.iloc[0] == "Opal_520_ring"
    T6 = pd.read_csv(out / "t6_counts.csv").set_index(["protein_class", "arm"])
    assert T6.loc[("CD8any", "nucleus"), "recall"] > T6.loc[("CD8any", "whole"), "recall"]
    R = pd.read_csv(out / "cd8_calls.csv")
    tc = R[(R.protein_class == "CD8any") & (R.stratum == "touching_panCK")].set_index("arm")
    assert tc.loc["nucleus", "called_T_NK"] > tc.loc["whole", "called_T_NK"] + 0.2


def test_truth_hybrid_and_recall_over_all_cells(tmp_path):
    """with a third of cells unlabeled in the nucleus, recall_all of the nucleus falls below its recall over
    labeled cells, and the hybrid arm nucleus+whole labels every cell."""
    m = _mod()
    rng = np.random.default_rng(5)
    n = 4000
    x, y = rng.uniform(0, 1000, n), rng.uniform(0, 600, n)
    t = np.where(rng.random(n) < 0.15, "T_NK", np.where(x < 500, "Tumour", "Fibro_muscle")).astype(object)
    nuc = t.copy()
    nuc[rng.random(n) < 1 / 3] = ""
    cid = np.array([f"c{i}" for i in range(n)])
    lab = tmp_path / "labels"
    lab.mkdir()
    pd.DataFrame({"cell_id": cid, "x": x, "y": y, "L2m_nucleus": nuc, "L2m_whole": t}).to_csv(
        lab / "S_L1_L1.labels.csv.gz", index=False)
    af = rng.lognormal(5, 0.5, n)
    ck = np.where(t == "Tumour", rng.lognormal(7, 0.4, n), rng.lognormal(3, 0.5, n))
    cd8 = 0.9 * af * rng.lognormal(0, 0.15, n) + rng.lognormal(3, 0.5, n) + np.where(
        (t == "T_NK") & (rng.random(n) < 0.8), rng.lognormal(6.3, 0.4, n), 0)
    reg = tmp_path / "reg"
    reg.mkdir()
    pd.DataFrame({"cell_id": cid, "dist_um": 0.5, "Opal_690_ring": ck, "Opal_620_ring": 0.3 * af + rng.lognormal(4, 0.6, n),
                  "Opal_520_ring": cd8, "Sample_AF_ring": af}).to_csv(reg / "matches.csv.gz", index=False)
    (reg / "register.json").write_text(json.dumps({"sample": "S_L1_L1", "R0": "valid"}))
    out = tmp_path / "out"
    m.cmd_truth(SimpleNamespace(matches=str(reg), labels=str(lab), out=str(out), level="L2m", k=6, r=30.0,
                                allow_invalid=False, no_af=False, max_match_um=2.0, iso_um=8.0, hybrid="nucleus:whole"))
    T6 = pd.read_csv(out / "t6_counts.csv").set_index(["protein_class", "arm"])
    nu, hy = T6.loc[("CD8any", "nucleus")], T6.loc[("CD8any", "nucleus+whole")]
    assert 0.6 < nu.labelled_share < 0.72 and nu.recall_all < nu.recall - 0.2
    assert hy.labelled_share > 0.99 and hy.recall_all > nu.recall_all + 0.2


def test_reg_diag_r0b(tmp_path):
    import subprocess
    import sys
    rng = np.random.default_rng(2)
    P = rng.uniform(0, [1500, 1000], (15000, 2))
    um, th = 0.4962, np.deg2rad(30)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    keep = rng.random(len(P)) < 0.7
    Q = ((P[keep] + rng.normal(0, 1.2, (keep.sum(), 2))) @ R) / um
    (tmp_path / "raw").mkdir()
    pd.DataFrame({"cell_id": np.arange(len(P)).astype(str), "vertex_x": P[:, 0], "vertex_y": P[:, 1]}).to_parquet(
        tmp_path / "raw" / "S_nucleus_boundaries.parquet")
    (tmp_path / "nuc_c5").mkdir()
    pd.DataFrame({"x_px": Q[:, 0], "y_px": Q[:, 1]}).to_csv(tmp_path / "nuc_c5" / "nuclei.csv.gz", index=False)
    (tmp_path / "nuc_c5" / "nuclei.json").write_text(json.dumps({"um_per_px": um}))
    Rb = np.array([[np.cos(th + 0.05), -np.sin(th + 0.05)], [np.sin(th + 0.05), np.cos(th + 0.05)]])
    for name, M, t in (("reg_good", um * R, [0, 0]), ("reg_bad", um * Rb, [40, -25])):
        (tmp_path / name).mkdir()
        (tmp_path / name / "register.json").write_text(json.dumps(
            {"sample": "S", "component": 5, "R0": "invalid", "M": np.asarray(M).tolist(), "t": t}))
    subprocess.run([sys.executable, str(ROOT / "examples" / "correction" / "reg_diag.py"), "--reg",
                    f"{tmp_path / 'reg_good'},{tmp_path / 'reg_bad'}", "--raw", str(tmp_path / "raw"), "--out",
                    str(tmp_path / "diag"), "--shifts", "15"], check=True, capture_output=True)
    V = pd.read_csv(tmp_path / "diag" / "r0b.csv").set_index("registration")
    assert V.loc["reg_good", "R0b"] == "valid" and V.loc["reg_bad", "R0b"] == "invalid"


def test_truth_fate_outputs(tmp_path):
    test_truth_end_to_end(tmp_path)
    F = pd.read_csv(tmp_path / "out" / "fate.csv")
    U = pd.read_csv(tmp_path / "out" / "label_protein.csv")
    f = F[(F.protein_class == "CD8any") & (F.stratum == "touching_panCK")]
    # the synthetic whole-cell arm turns half the touching T_NK cells into Tumor
    assert f[(f.arm == "whole") & (f.label == "Tumour")].share.sum() > f[(f.arm == "nucleus") & (f.label == "Tumour")].share.sum() + 0.1
    g = F[F.stratum == "all"].groupby(["protein_class", "arm"]).share.sum()
    assert np.allclose(g.to_numpy(), 1.0)
    t = U[(U.arm == "nucleus")].set_index("label")
    assert t.loc["Tumour", "panCK_pos"] > t.loc["T_NK", "panCK_pos"]
    assert "CD8iso" in set(pd.read_csv(tmp_path / "out" / "t6_counts.csv").protein_class)


def test_stains_roles_from_the_stain_alone(tmp_path):
    """a tumor sheet (pan-cytokeratin, clustered, ~40%), scattered CD8 cells (~4%) and a noise channel:
    the label-free rule names panCK and CD8 without labels or Xenium data."""
    m = _mod()
    rng = np.random.default_rng(3)
    n = 6000
    xy = rng.uniform(0, 1200, (n, 2)) / 0.5            # pixels at 0.5 um / px
    tum = xy[:, 0] * 0.5 < 480
    cd8 = ~tum & (rng.random(n) < 0.07)
    D = pd.DataFrame({"x_px": xy[:, 0], "y_px": xy[:, 1], "dapi_peak": 100.0})
    af = rng.lognormal(5, 0.5, n)
    D["Sample_AF_nuc"] = af; D["Sample_AF_ring"] = af
    for ch, pos in (("Opal_690", tum), ("Opal_520", cd8), ("Opal_620", np.zeros(n, bool))):
        ring = 0.8 * af * rng.lognormal(0, 0.15, n) + rng.lognormal(3, 0.5, n) + np.where(pos, rng.lognormal(6.5, 0.4, n), 0)
        D[f"{ch}_ring"] = ring
        D[f"{ch}_nuc"] = ring * np.where(pos, 0.4, 1.0)
    nd = tmp_path / "nuc_c9"
    nd.mkdir()
    D.to_csv(nd / "nuclei.csv.gz", index=False, compression="gzip")
    (nd / "nuclei.json").write_text(json.dumps({"component": 9, "um_per_px": 0.5}))
    m.cmd_stains(SimpleNamespace(nuclei=str(nd), out=str(tmp_path / "st"), no_af=False))
    R = pd.read_csv(tmp_path / "st" / "roles.csv")
    assert R.panCK.iloc[0] == "Opal_690" and R.CD8.iloc[0] == "Opal_520" and bool(R.usable.iloc[0])
    C = pd.read_csv(tmp_path / "st" / "channels.csv").set_index("channel")
    assert C.loc["Opal_690", "neighbour_enrichment"] > C.loc["Opal_520", "neighbour_enrichment"]
    assert C.loc["Opal_520", "ring_over_nucleus"] > 1.5
