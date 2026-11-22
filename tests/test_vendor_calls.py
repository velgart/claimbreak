"""vendor_calls.py, the typical user's analyses from whole-cell, nuclear, hybrid and protein labels."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "compartment" / "vendor_calls.py"


def _section(d, rng, n=8000):
    xy = rng.uniform(0, 1600, (n, 2))
    truth = np.where(xy[:, 0] < 800, "Tumor", "Fibroblast").astype(object)
    truth[rng.choice(n, 900, replace=False)] = "T_NK"
    truth[rng.choice(n, 300, replace=False)] = "Myeloid"
    nuc, whole = truth.copy(), truth.copy()
    inside = (truth == "T_NK") & (xy[:, 0] < 800)
    whole[inside & (rng.random(n) < 0.6)] = "Tumor"                  # whole-cell labels lose T cells in tumor
    nuc[rng.random(n) < 0.1] = ""
    ids = [f"c{i}" for i in range(n)]
    pd.DataFrame(dict(cell_id=ids, x=xy[:, 0], y=xy[:, 1], L1_nucleus=nuc, L1_whole=whole)).to_csv(
        d / "lab" / "S1.labels.csv.gz", index=False, compression="gzip")
    t = truth == "T_NK"
    hi = lambda m: np.where(m, np.exp(rng.normal(5, 0.3, n)), np.where(rng.random(n) < 0.3, np.exp(rng.normal(1, 0.3, n)), 0.0))
    pd.DataFrame(dict(cell_id=ids, CD3E=hi(t), CD45=hi(t | (truth == "Myeloid")), CD68=hi(truth == "Myeloid"),
                      PanCK=hi(truth == "Tumor"))).to_csv(d / "prot.csv.gz", index=False, compression="gzip")


def test_vendor_calls(tmp_path):
    (tmp_path / "lab").mkdir()
    _section(tmp_path, np.random.default_rng(2))
    base = [sys.executable, str(SCRIPT), "--labels", str(tmp_path / "lab"), "--partner", "(?i)^tumou?r$"]
    r = subprocess.run(base + ["--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "T <- T_NK" in r.stdout and "P <- Tumor" in r.stdout and "I <- Myeloid" in r.stdout
    r = subprocess.run(base + ["--out", str(tmp_path / "o"), "--nperm", "30", "--protein", str(tmp_path / "prot.csv.gz"),
                               "--protein-partner", "PANCK+"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    M = pd.read_csv(tmp_path / "o" / "medians.csv").set_index("arm")
    assert set(M.index) == {"whole", "nucleus", "hybrid", "protein"}
    assert M.loc["whole", "per100"] < 0.7 * M.loc["nucleus", "per100"]
    assert abs(M.loc["protein", "per100"] / M.loc["nucleus", "per100"] - 1) < 0.25
    F = pd.read_csv(tmp_path / "o" / "flips.csv")
    assert {"nhood", "mixing_class"} <= set(F.readout) and (F.fields >= 1).all()
    assert json.loads((tmp_path / "o" / "summary.json").read_text())["fields"] >= 4
    r = subprocess.run(base + ["--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode != 0                                          # refuses to overwrite


def test_vendor_calls_no_qualifying_field(tmp_path):
    """No field reaches --min-t: a clear message instead of a KeyError (found by the release smoke test)."""
    (tmp_path / "lab").mkdir()
    _section(tmp_path, np.random.default_rng(2))
    r = subprocess.run([sys.executable, str(SCRIPT), "--labels", str(tmp_path / "lab"), "--partner", "(?i)^tumou?r$",
                        "--out", str(tmp_path / "o"), "--nperm", "10", "--min-t", "100000"], capture_output=True, text=True)
    assert r.returncode != 0 and "no field reached --min-t" in (r.stderr + r.stdout)
