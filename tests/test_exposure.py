"""claimbreak.geometry and claimbreak.exposure: physical neighborhoods, and label loss against exposure.

Two synthetic scenarios on the same tissue: labels lost because of touching cells of another type (the loss grows
with heterotypic contacts and is absent without them) and labels lost intrinsically (the same loss at every
exposure). The exposure analysis must tell them apart."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from claimbreak import exposure as xp
from claimbreak import geometry as geo
from claimbreak.cli import main


def test_gap_pairs_and_border_share():
    # a center cell with six touching neighbors on a hexagon, and one far-away cell
    ang = np.arange(6) * np.pi / 3
    xy = np.vstack([[0, 0], np.c_[7 * np.cos(ang), 7 * np.sin(ang)], [100, 100]])
    r = np.full(len(xy), 3.0)
    i, j, gap, d = geo.gap_pairs(xy, r, 2.0)
    assert set(j[i == 0]) == {1, 2, 3, 4, 5, 6}
    assert np.allclose(gap[i == 0], 1.0)
    assert 7 not in set(i) | set(j)
    ip, jp, sh, cov = geo.border_shares(len(xy), xy, r, i, j, gap, d, 2.0)
    assert cov[0] == pytest.approx(sum(sh[ip == 0]))
    assert 0.6 < cov[0] <= 1.0                       # six disks of radius 3 at distance 7 cover most of the circle
    assert cov[7] == 0
    focal = np.array([0, 1, 1, 0, 0, 0, 0, 0])       # neighbors 1 and 2 of another type
    frac, expo = geo.heterotypic(len(xy), ip, jp, sh, focal, focal, mass=np.r_[100, 300, 300, 100, 100, 100, 100, 100])
    assert frac[0] == pytest.approx(2 / 6, abs=0.02)
    assert expo[0] == pytest.approx(3 * cov[0] * 2 / 6, rel=0.05)
    assert np.isnan(frac[7])
    r2, note = geo.nucleus_radius([np.nan, np.nan])
    assert np.allclose(r2, geo.DEFAULT_RADIUS) and "default" in note


def _tissue(scenario, seed=3, n_side=70, q=0.25, p_int=0.3):
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    xy = np.c_[gx.ravel(), gy.ravel()] * 7.5 + rng.uniform(-0.4, 0.4, (n_side * n_side, 2))
    n = len(xy)
    truth = np.where(rng.random(n) < 0.3, "T_NK", "Tumour")
    for c in rng.uniform(0, n_side * 7.5, (25, 2)):            # T aggregates: cells with no other type nearby
        truth[np.linalg.norm(xy - c, axis=1) < 22] = "T_NK"
    area = np.where(truth == "T_NK", np.pi * 2.6 ** 2, np.pi * 3.4 ** 2)
    r = np.sqrt(area / np.pi)
    i, j, gap, _ = geo.gap_pairs(xy, r, 2.0)
    het = truth[i] != truth[j]
    k = np.bincount(i[het], minlength=n)

    def noisy(lab, p=0.02):
        flip = rng.random(n) < p
        return np.where(flip, np.where(lab == "T_NK", "Tumour", "T_NK"), lab)

    nucA, nucB = noisy(truth), noisy(truth)
    if scenario == "neighbour":
        lose = rng.random(n) < 1 - (1 - q) ** k
    elif scenario == "process":                      # lost to tumor within 20 um, touching or not
        sh = xp.region_shares(xy, (truth == "Tumour").astype(int), 2, (20.0,), 1)[20.0]
        lose = rng.random(n) < 0.6 * sh
    else:
        lose = rng.random(n) < p_int
    wholeB = np.where((truth == "T_NK") & lose, "Tumour", noisy(truth))
    D = pd.DataFrame({"sample": f"S_{scenario}", "cell_id": [f"c{x}" for x in range(n)], "x": xy[:, 0],
                      "y": xy[:, 1], "n_whole": np.where(truth == "T_NK", 100, 300),
                      "n_nucleus": np.where(truth == "T_NK", 30, 90),
                      "unit": f"U_{scenario}", "L2m_nucA": nucA, "L2m_nucB": nucB, "L2m_wholeB": wholeB,
                      "L2m_nucleus": truth})
    return D, area


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("exposure")
    lab, h5 = root / "labels", root / "h5ad"
    lab.mkdir(), h5.mkdir()
    for sc in ("neighbour", "intrinsic", "process"):
        D, area = _tissue(sc, n_side=110)
        D.to_csv(lab / f"S_{sc}.labels.csv.gz", index=False, compression="gzip")
        A = ad.AnnData(sp.csr_matrix((len(D), 1), dtype=np.float32),
                       obs=pd.DataFrame({"nucleus_area": area}, index=D.cell_id.to_numpy()))
        A.write_h5ad(h5 / f"S_{sc}.h5ad")
    out = root / "out"
    assert main(["compartment", "exposure", "--labels", str(lab), "--h5ad-dir", str(h5), "--out", str(out),
                 "--levels", "L2m", "--unit-pattern", ""]) in (None, 0)
    return out


def _dose(out, unit):
    d = pd.read_csv(out / "dose.csv")
    d = d[(d.unit == unit) & (d.type == "T_NK") & (d.g == 2.0) & (d.m == "all")]
    pooled = d.assign(kk=np.minimum(d.k, 3)).groupby("kk")[["kept_test", "kept_null"]].sum()
    return (pooled.kept_test / pooled.kept_null).to_dict()


def test_neighbour_loss_grows_with_contacts(runs):
    r = _dose(runs, "U_neighbour")
    assert 0.9 < r[0] < 1.1                         # no heterotypic contact: kept as often as the null
    assert r[3] < 0.6 and r[3] < r[1] < r[0]        # loss grows with heterotypic contacts


def test_intrinsic_loss_is_flat(runs):
    r = _dose(runs, "U_intrinsic")
    assert r[0] < 0.8                               # lost even without contact
    assert abs(r[3] - r[0]) < 0.15                  # and about as much at high contact


def test_border_and_destinations(runs):
    t = pd.read_csv(runs / "units.csv", dtype={"bin": str})
    b = t[(t.unit == "U_neighbour") & (t.type == "T_NK") & (t.scheme == "border")].set_index("bin")
    assert b.loc["0", "ratio"] > 0.9 and b.loc["1", "ratio"] < 0.6
    e = t[(t.unit == "U_neighbour") & (t.type == "T_NK") & (t.scheme == "exposure")].set_index("bin")
    assert e.loc["0", "ratio"] > 0.9 and e.loc["(0.5,1]", "ratio"] > e.loc["(1,2]", "ratio"] > e.loc["(2,4]", "ratio"]
    e = t[(t.unit == "U_intrinsic") & (t.type == "T_NK") & (t.scheme == "exposure")].set_index("bin")
    assert abs(e.loc["0", "ratio"] - e.loc["(1,2]", "ratio"]) < 0.1
    f = pd.read_csv(runs / "dest_fit.csv")
    nb = f[(f.unit == "U_neighbour") & (f.a == "T_NK") & (f.b == "Tumour")].iloc[0]
    it = f[(f.unit == "U_intrinsic") & (f.a == "T_NK") & (f.b == "Tumour")].iloc[0]
    assert abs(nb.intercept) < 0.06 and nb.slope > 0.3     # changes toward tumor need tumor at the border
    assert it.intercept > 0.2                              # intrinsic loss goes to tumor without contact
    notes = (runs / "notes.txt").read_text()
    assert "nucleus areas for 1.000 of cells" in notes


def test_segseries_per_unit(tmp_path):
    """segseries_summary.py --unit-map: a section holding two patients gives one row per patient."""
    import subprocess
    import sys
    from pathlib import Path
    ex = Path(__file__).resolve().parents[1] / "examples" / "compartment" / "segseries_summary.py"
    rng = np.random.default_rng(0)
    n = 400
    unit = np.where(np.arange(n) < 200, "P1", "P2")
    for mask, p_loss in (("r1", 0.1), ("r5", 0.4)):
        d = tmp_path / "labels" / mask
        d.mkdir(parents=True)
        nuc = np.where(rng.random(n) < 0.4, "T_NK", "Epi")
        loss = (nuc == "T_NK") & (rng.random(n) < np.where(unit == "P1", p_loss, p_loss / 2))
        pd.DataFrame({"cell_id": [f"c{i}" for i in range(n)], "panel": "kid", "L2m_whole": np.where(loss, "Epi", nuc),
                      "L2m_nucleus": nuc, "L2m_nucA": nuc, "L2m_nucB": nuc,
                      "L2m_wholeB": np.where(loss, "Epi", nuc)}).to_csv(d / "REG1.labels.csv.gz", index=False)
    pd.DataFrame({"cell_id": [f"c{i}" for i in range(n)], "unit": unit}).to_csv(tmp_path / "map.csv", index=False)
    out = tmp_path / "summ"
    subprocess.run([sys.executable, str(ex), "--labels-root", str(tmp_path / "labels"), "--level", "L2m",
                    "--unit-map", str(tmp_path / "map.csv"), "--out", str(out)], check=True, capture_output=True)
    T = pd.read_csv(out / "series_types.csv")
    t = T[T.type == "T_NK"].set_index(["mask", "unit"]).ratio_dm
    assert set(T.unit) == {"P1", "P2"}
    assert t[("r5", "P1")] < t[("r5", "P2")] < 1 and t[("r1", "P1")] > t[("r5", "P1")]


def test_regional_destinations(runs):
    """Cells without tumor at the border: no change toward tumor if loss needs contact; a rate rising with the
    regional tumor share from about zero if tumor RNA reaches from nearby cells; flat above zero if intrinsic."""
    f = pd.read_csv(runs / "dest_region_fit.csv")
    f = f[(f.a == "T_NK") & (f.b == "Tumour") & (f.radius == 20.0)].set_index("unit")
    # contact loss: cells without tumor at the border change little (a tumor nucleus within the gap can be shadowed
    # by a closer neighbor in the angular border, so not exactly zero)
    assert abs(f.loc["U_neighbour", "intercept"]) < 0.06 and abs(f.loc["U_neighbour", "slope"]) < 0.15
    assert f.loc["U_process", "slope"] > 0.3 and abs(f.loc["U_process", "intercept"]) < 0.06
    assert f.loc["U_intrinsic", "intercept"] > 0.2 and f.loc["U_intrinsic", "slope"] < 0.15
    u = pd.read_csv(runs / "units.csv", dtype={"bin": str})
    o = u[(u.unit == "U_neighbour") & (u.type == "T_NK") & (u.scheme == "outside")]
    assert len(o) and o["bin"].str.contains("|", regex=False).all()


def test_unit_pattern_first():
    D = pd.DataFrame({"sample": ["GSM1_Xenium_337_Colon_I0355"] * 2, "unit": ["337_Colon_I0355", ""]})
    assert list(xp.unit_column(D, r"I0\d{3}")) == ["I0355", "I0355"]
    assert list(xp.unit_column(D, r"nomatch")) == ["337_Colon_I0355", "GSM1_Xenium_337_Colon_I0355"]


def test_summary_with_unbinned_regions(tmp_path):
    """a regional bin without a midpoint ('none': no typed cell within the radius) must not stop the summary."""
    d = tmp_path / "x"
    d.mkdir()
    pd.DataFrame([{"sample": "s", "unit": "u", "level": "L", "type": "T", "scheme": "all", "g": np.nan, "m": np.nan,
                   "k": np.nan, "bin": "", "cells": 100, "kept_test": 50, "kept_null": 90}]).to_csv(d / "strata.csv",
                                                                                                   index=False)
    rows = [{"sample": "s", "unit": "u", "level": "L", "a": "T", "b": "Tu", "radius": 20.0, "bin": b, "cells": 100,
             "to_b_test": t, "to_b_null": 2} for b, t in (("none", 30), ("0", 2), ("(0.1,0.25]", 12), ("(0.5,1]", 40))]
    pd.DataFrame(rows).to_csv(d / "dest_region.csv", index=False)
    xp.exposure_summary(d, verbose=False)
    f = pd.read_csv(d / "dest_region_fit.csv")
    assert len(f) == 1 and f.slope.iloc[0] > 0.3 and abs(f.intercept.iloc[0]) < 0.05
    assert xp._wfit([0, np.nan], [1, 2], [1, 1]) is None


def _squares(nx=4, ny=3, side=8.0, gap=0.0, ids=None):
    """Square cell polygons on a grid (Xenium cell_boundaries layout: one row per vertex, polygons closed)."""
    rows = []
    for c in range(nx * ny):
        x0, y0 = (c % nx) * (side + gap), (c // nx) * (side + gap)
        cid = ids[c] if ids is not None else f"c{c}"
        for vx, vy in ((x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side), (x0, y0)):
            rows.append((cid, vx, vy))
    return pd.DataFrame(rows, columns=["cell_id", "vertex_x", "vertex_y"])


def test_boundary_shares_on_a_grid():
    """shared perimeter from polygons. On a 4 x 3 grid of touching squares an interior cell shares a quarter
    of its perimeter with each of its four neighbors, a corner cell half its perimeter with two; an isolated
    square and a cell without a polygon touch nothing."""
    from claimbreak.contacts import boundary_shares
    B = pd.concat([_squares(), _squares(1, 1, ids=["far"]).assign(vertex_x=lambda d: d.vertex_x + 500)])
    ids = [f"c{c}" for c in range(12)] + ["far", "nopoly"]
    ip, jp, sh, cov = boundary_shares(B, ids, tol=1.0)
    s = pd.DataFrame({"i": ip, "j": jp, "s": sh})
    inner = s[s.i == 5].set_index("j").s                    # c5: row 1, column 1 -> edge neighbors c1, c4, c6, c9
    assert np.allclose(inner.reindex([1, 4, 6, 9]), 0.25, atol=0.06)
    assert inner.drop([1, 4, 6, 9], errors="ignore").max() < 0.1     # diagonals: a corner point (ties at junctions)
    assert cov[5] == pytest.approx(1.0, abs=0.05)
    corner = s[s.i == 0]
    assert set(corner[corner.s > 0.1].j) == {1, 4} and cov[0] == pytest.approx(0.5, abs=0.08)
    assert cov[12] == 0 and cov[13] == 0 and not (s.i >= 12).any()
    assert np.allclose(s.groupby("i").s.sum().to_numpy(), cov[s.groupby("i").s.sum().index], atol=1e-9)
    ip2, jp2, sh2, cov2 = boundary_shares(_squares(gap=3.0), [f"c{c}" for c in range(12)], tol=1.0)
    assert len(ip2) == 0 and cov2.max() == 0                # 3 um apart: no shared border at 1 um


def test_exposure_with_polygons(tmp_path):
    """`compartment exposure --boundaries` takes border shares from polygons (notes say so), and a T/NK cell
    whose whole perimeter is tumor lands in border bin 1."""
    side, nx, ny = 8.0, 5, 5
    ids = [f"c{c}" for c in range(nx * ny)]
    cx = np.array([(c % nx) * side + side / 2 for c in range(nx * ny)])
    cy = np.array([(c // nx) * side + side / 2 for c in range(nx * ny)])
    lab = np.array(["Tumour"] * (nx * ny), dtype=object)
    lab[12] = "T_NK"                                         # center cell, surrounded by tumor
    D = pd.DataFrame({"sample": "S1", "cell_id": ids, "x": cx, "y": cy, "n_whole": 60, "n_nucleus": 30,
                      "L1_nucA": lab, "L1_nucB": lab, "L1_wholeB": lab, "L1_nucleus": lab})
    (tmp_path / "lab").mkdir()
    D.to_csv(tmp_path / "lab" / "S1.labels.csv.gz", index=False, compression="gzip")
    (tmp_path / "bnd").mkdir()
    _squares(nx, ny, side).to_parquet(tmp_path / "bnd" / "S1_cell_boundaries.parquet")
    S, _ = xp.exposure_files(tmp_path / "lab", tmp_path / "out", ["L1"], boundaries=str(tmp_path / "bnd"),
                             regions=(), unit_pattern="S1", verbose=False)
    assert "polygons" in (tmp_path / "out" / "notes.txt").read_text()
    b = S[(S.scheme == "border") & (S.type == "T_NK")]
    assert list(b["bin"]) == ["1"] and int(b.cells.iloc[0]) == 1
    with pytest.raises(FileNotFoundError):
        xp.exposure_files(tmp_path / "lab", tmp_path / "out2", ["L1"], boundaries=str(tmp_path / "lab"),
                          regions=(), verbose=False)
