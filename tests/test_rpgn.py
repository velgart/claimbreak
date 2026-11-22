"""rpgn_glom.py assigns vendor cells to the nearest Baysor cell's region and compares T/NK inside glomeruli
with the ring per biopsy and arm."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "compartment" / "rpgn_glom.py"


def _section(tmp, rng):
    # Baysor cells on a grid: a glomerulus (r < 60 um) in a ring (60-160 um) in other tissue, two biopsies
    xs, ys = np.meshgrid(np.arange(0, 400, 10.0), np.arange(0, 400, 10.0))
    x, y = xs.ravel(), ys.ravel()
    r = np.hypot(x - 200, y - 200)
    glom, ring = r < 60, (r >= 60) & (r < 160)
    biopsy = np.where(x < 200, "S - A", "S - B")
    tnk_true = rng.random(len(x)) < np.where(glom, 0.2, 0.2)              # same true T share everywhere
    published = np.where(tnk_true & ~glom, "T", np.where(glom, "POD", "PT"))   # published labels lose T in glomeruli
    B = pd.DataFrame({"x": x, "y": y, "Slide_ID": "0001", "Biopsy_ID": biopsy, "Disease": "ANCA",
                      "celltype_l1": published, "is_in_glom": glom, "is_in_periglom": ring,
                      "glom_ID": np.where(glom, "g1", "Other")})
    B.to_parquet(tmp / "baysor.parquet")
    # vendor cells 2 um away from each Baysor cell
    nuc = np.where(tnk_true, "T_NK", np.where(glom, "Podocyte", "PT"))
    whole = np.where(glom, "Podocyte", nuc)                                    # whole-cell labels lose T in glomeruli
    L = pd.DataFrame({"sample": "RPGN_0001", "cell_id": [f"c{i}" for i in range(len(x))], "x": x + 2, "y": y,
                      "L1_nucleus": nuc, "L1_whole": whole, "L1_nucB": nuc, "L1_nucA": nuc, "L1_wholeB": whole})
    (tmp / "lab").mkdir()
    L.to_csv(tmp / "lab" / "RPGN_0001.labels.csv.gz", index=False, compression="gzip")


def test_glom_vs_ring(tmp_path):
    _section(tmp_path, np.random.default_rng(3))
    r = subprocess.run([sys.executable, str(SCRIPT), "--labels", str(tmp_path / "lab"), "--baysor",
                        str(tmp_path / "baysor.parquet"), "--slides", "RPGN_0001=0001", "--out", str(tmp_path / "g"),
                        "--min-cells", "20"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    M = pd.read_csv(tmp_path / "g" / "match.csv").iloc[0]
    assert M.matched == 1.0 and abs(M.dist_median - 2) < 1e-6
    P = pd.read_csv(tmp_path / "g" / "ratio.csv").set_index(["biopsy", "arm"])
    for b in ("S - A", "S - B"):
        assert P.loc[(b, "nucleus"), "glom_over_ring"] > 0.5            # nuclear labels see the T cells inside
        assert P.loc[(b, "whole"), "glom"] == 0 and P.loc[(b, "baysor_published"), "glom"] == 0
    # refuses to overwrite
    r2 = subprocess.run([sys.executable, str(SCRIPT), "--labels", str(tmp_path / "lab"), "--baysor",
                         str(tmp_path / "baysor.parquet"), "--slides", "RPGN_0001=0001", "--out", str(tmp_path / "g")],
                        capture_output=True, text=True)
    assert r2.returncode != 0
