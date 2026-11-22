"""examples/compartment/reexpand.py: nucleus expansion rebuilt at several distances on a synthetic Xenium section."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ad = pytest.importorskip("anndata")
pytest.importorskip("pyarrow")


def _section(d, rng):
    # two cells 12 um apart (nucleus radius 3 um); transcripts at known distances from the nuclei
    cid = ["a", "b"]
    cx, cy = np.array([0.0, 12.0]), np.array([0.0, 0.0])
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    B = pd.DataFrame({"cell_id": np.repeat(cid, 12), "vertex_x": np.r_[3 * np.cos(ang), 12 + 3 * np.cos(ang)],
                      "vertex_y": np.r_[3 * np.sin(ang), 3 * np.sin(ang)]})
    B.to_parquet(d / "S_nucleus_boundaries.parquet")
    pd.DataFrame({"cell_id": cid, "x_centroid": cx, "y_centroid": cy, "cell_area": [100.0, 100.0],
                  "nucleus_area": [28.0, 28.0]}).to_parquet(d / "S_cells.parquet")
    rows = []
    # cell a: 10 nuclear transcripts of g1; g2 at 1.5 um outside its nucleus (x = -4.5); g3 at 4 um (x = -7)
    rows += [("a", "g1", 0.0, 0.0, 1)] * 10
    rows += [("a", "g2", -4.5, 0.0, 0)] * 4
    rows += [("UNASSIGNED", "g3", -7.0, 0.0, 0)] * 3
    # between the cells at x = 6.5 (3.5 um from a's nucleus edge, 2.5 um from b's): platform says a, nearest is b
    rows += [("a", "g4", 6.5, 0.0, 0)] * 5
    rows += [("b", "g1", 12.0, 0.0, 1)] * 7
    rows += [("b", "NegControlProbe_1", 12.0, 0.0, 1)] * 2
    T = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "overlaps_nucleus"])
    T["qv"] = 30.0
    T.to_parquet(d / "S_transcripts.parquet")


def test_reexpand_masks(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _section(raw, np.random.default_rng(0))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "compartment" / "reexpand.py"), "--raw", str(raw),
                        "--sample", "S", "--out", str(out), "--tmp", str(tmp_path / "tmp"), "--radii", "0,2,3,5",
                        "--spacing", "0.2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    C = pd.read_csv(out / "S.cells.csv.gz").set_index("cell_id")
    assert C.loc["a", "n_nucleus"] == 10 and C.loc["b", "n_nucleus"] == 7          # control probe dropped
    assert C.loc["a", "n_r0"] == 10
    assert C.loc["a", "n_r2"] == 14                       # + g2 at 1.5 um
    assert C.loc["a", "n_r3"] == 14 and C.loc["b", "n_r3"] == 12     # g4 at 2.5 um from b goes to b
    assert C.loc["a", "n_r5"] == 17                       # + unassigned g3 at 4 um
    assert C.loc["a", "n_platform"] == 19 and C.loc["b", "n_platform"] == 7
    A = ad.read_h5ad(out / "r3" / "S.h5ad")
    assert list(A.var_names) == ["g1", "g2", "g3", "g4"]
    assert float(A.layers["nucleus"].sum()) == 17 and float(A.layers["cyto"].sum()) == float(A.X.sum()) - 17
    q = json.loads((out / "S.reexpand.json").read_text())["qc"]
    r5 = [x for x in q if x["mask"] == "r5"][0]
    assert abs(r5["same_cell"] - 4 / 9) < 1e-9 and abs(r5["other_cell"] - 5 / 9) < 1e-9
    r2 = subprocess.run([sys.executable, str(ROOT / "examples" / "compartment" / "reexpand.py"), "--raw", str(raw),
                         "--sample", "S", "--out", str(out), "--tmp", str(tmp_path / "tmp")], capture_output=True, text=True)
    assert r2.returncode != 0                              # refuses to overwrite


def test_segseries_summary_and_combine(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    types = np.array(["T_NK", "Tumour", "Myeloid"])
    true = types[rng.integers(0, 3, n)]
    root = tmp_path / "labels"
    for mask, p in (("r0", 0.0), ("r5", 0.3), ("platform", 0.35)):
        d = root / mask
        d.mkdir(parents=True)
        def noisy(q):
            x = true.copy()
            flip = (true == "T_NK") & (rng.random(n) < q)
            x[flip] = "Tumour"
            return x
        L = pd.DataFrame({"sample": "S_lung_L1_L1", "cell_id": [f"c{i}" for i in range(n)], "x": rng.random(n),
                          "y": rng.random(n), "panel": "Lung", "L2m_nucleus": true, "L2m_nucA": noisy(0.02),
                          "L2m_nucB": noisy(0.02), "L2m_whole": noisy(p), "L2m_wholeB": noisy(p),
                          "L2m_whole_second": "Myeloid", "L2m_whole_margin": 1.0})
        L.to_csv(d / "S_lung_L1_L1.labels.csv.gz", index=False, compression="gzip")
    out = tmp_path / "summ"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "compartment" / "segseries_summary.py"),
                        "--labels-root", str(root), "--out", str(out), "--combine-out", str(tmp_path / "comb"),
                        "--samples", "S_lung_L1_L1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    T = pd.read_csv(out / "series_types.csv")
    t = T[T.type == "T_NK"].set_index("mask")
    assert t.loc["r0", "ratio_dm"] > 0.9 and t.loc["r5", "ratio_dm"] < 0.8 and t.loc["platform", "ratio_dm"] < 0.8
    C = pd.read_csv(tmp_path / "comb" / "S_lung_L1_L1.labels.csv.gz")
    assert {"L2m_nucleus", "L2m_whole", "L2m_wholeB", "L2m_r5", "L2m_r5B", "x", "y"} <= set(C.columns)
    assert "L2m_r0" not in C.columns


def test_segseries_ratios_on_common_cells(tmp_path):
    """a ratio uses the cells labeled in both of its arms. Here wholeB leaves 40% of cells unlabeled (too few
    counts) and changes no label: the common-cell ratio is 1, the old all-cell count ratio about 0.6."""
    rng = np.random.default_rng(2)
    n = 500
    true = np.array(["T_NK", "Tumour"])[rng.integers(0, 2, n)]
    root = tmp_path / "labels" / "r1"
    root.mkdir(parents=True)
    wb = true.copy()
    wb[rng.random(n) < 0.4] = ""
    L = pd.DataFrame({"sample": "S_lung_L1_L1", "cell_id": [f"c{i}" for i in range(n)], "x": rng.random(n),
                      "y": rng.random(n), "panel": "Lung", "L2m_nucleus": true, "L2m_nucA": true, "L2m_nucB": true,
                      "L2m_whole": true, "L2m_wholeB": wb})
    L.to_csv(root / "S_lung_L1_L1.labels.csv.gz", index=False, compression="gzip")
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "compartment" / "segseries_summary.py"),
                        "--labels-root", str(tmp_path / "labels"), "--out", str(tmp_path / "summ")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    t = pd.read_csv(tmp_path / "summ" / "series_types.csv").set_index("type").loc["T_NK"]
    assert t.ratio_dm == 1.0 and t.ratio_dm_all < 0.75 and t.ratio_null == 1.0


def test_reexpand_two_nuclei_in_one_cell(tmp_path):
    """a cell with two nucleus polygons (label_id) is densified ring by ring. Before, the two rings were
    joined into one, and the jump between them put boundary points in the gap, so a transcript in the gap was given
    to that cell at small radii."""
    raw = tmp_path / "raw"
    raw.mkdir()
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    B = pd.DataFrame({"cell_id": ["a"] * 24, "label_id": [1] * 12 + [2] * 12,
                      "vertex_x": np.r_[3 * np.cos(ang), 20 + 3 * np.cos(ang)], "vertex_y": np.r_[3 * np.sin(ang), 3 * np.sin(ang)]})
    B.to_parquet(raw / "S_nucleus_boundaries.parquet")
    pd.DataFrame({"cell_id": ["a"], "x_centroid": [10.0], "y_centroid": [0.0], "cell_area": [300.0],
                  "nucleus_area": [56.0]}).to_parquet(raw / "S_cells.parquet")
    rows = [("a", "g1", 0.0, 0.0, 1)] * 5 + [("a", "g1", 20.0, 0.0, 1)] * 5 + [("UNASSIGNED", "g2", 10.0, 0.0, 0)] * 3
    T = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "overlaps_nucleus"])
    T["qv"] = 30.0
    T.to_parquet(raw / "S_transcripts.parquet")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "compartment" / "reexpand.py"), "--raw", str(raw),
                        "--sample", "S", "--out", str(out), "--tmp", str(tmp_path / "tmp"), "--radii", "0,2,10",
                        "--spacing", "0.2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    C = pd.read_csv(out / "S.cells.csv.gz").set_index("cell_id")
    assert C.loc["a", "n_nucleus"] == 10
    assert C.loc["a", "n_r2"] == 10                       # the gap transcripts are 7 um from both nuclei
    assert C.loc["a", "n_r10"] == 13
