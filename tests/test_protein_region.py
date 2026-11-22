"""examples/compartment/protein_region.py: label loss of protein T cells by region, regions typed by protein."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_protein_contacts import _section  # noqa: E402


def _mod():
    spec = importlib.util.spec_from_file_location("protein_region", ROOT / "examples" / "compartment" / "protein_region.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_protein_region(tmp_path):
    m = _mod()
    root = _section(tmp_path)
    L = pd.read_csv(root / "lab" / "XP_T.labels.csv.gz", keep_default_na=False)
    L["L1_nucB"], L["L1_wholeB"] = L.L1_nucleus, L.L1_whole
    L.to_csv(root / "lab" / "XP_T.labels.csv.gz", index=False, compression="gzip")
    m.main(["--labels", str(root / "lab"), "--protein", str(root / "XP_T.protein.csv.gz"), "--sample", "XP_T",
            "--out", str(tmp_path / "o"), "--region", "PANCK+", "--block-draws", "50"])
    R = pd.read_csv(tmp_path / "o" / "region.csv").set_index(["bin", "arm", "metric"])
    inside = R.loc[("inside", "wholeB", "loss_pct_vs_nucB"), "estimate"]
    away = R.loc[("away", "wholeB", "loss_pct_vs_nucB"), "estimate"]
    assert inside > away + 20                                   # whole-cell counts lose T cells inside tumor
    d = R.loc[("inside minus away", "wholeB", "loss_pct_vs_nucB")]
    assert d.lo > 0
    assert json.loads((tmp_path / "o" / "meta.json").read_text())["region"] == "PANCK+"
