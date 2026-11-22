"""examples/compartment/cell_crop.py: the Fig. 1a crop picks a T cell labeled tumor next to a tumor cell."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "compartment" / "cell_crop.py"


def _square(cid, x, y, h):
    return pd.DataFrame({"cell_id": cid, "vertex_x": [x - h, x + h, x + h, x - h, x - h],
                         "vertex_y": [y - h, y - h, y + h, y + h, y - h]})


def test_cell_crop(tmp_path):
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"; raw.mkdir()
    # a row of cells 10 um apart: tumor, T (whole cell called tumor), tumor, and a T cell called T, far away
    cells = pd.DataFrame({"cell_id": ["a", "t1", "b", "t2"], "x": [100.0, 110, 120, 300], "y": [100.0, 100, 100, 100],
                          "L2m_nucleus": ["Tumour", "T_NK", "Tumour", "T_NK"],
                          "L2m_whole": ["Tumour", "Tumour", "Tumour", "T_NK"]})
    cells["sample"] = "s1"
    cells.to_csv(tmp_path / "s1.labels.csv.gz", index=False)
    tx = []
    for cid, x, y in cells[["cell_id", "x", "y"]].itertuples(index=False):
        tumour, t = cid in ("a", "b"), cid.startswith("t")
        for _ in range(20):           # nuclear transcripts within 2 um, cytoplasmic 2.5-4.5 um from the center
            for nuc in (True, False):
                r = rng.uniform(0, 2) if nuc else rng.uniform(2.5, 4.5); th = rng.uniform(0, 2 * np.pi)
                if t and nuc:
                    g = "CD3E"
                elif tumour or (cid == "t1" and not nuc):
                    g = "EPCAM"
                else:
                    g = "ACTB"
                tx.append((cid, g, int(nuc), x + r * np.cos(th), y + r * np.sin(th), 30.0))
    pd.DataFrame(tx, columns=["cell_id", "feature_name", "overlaps_nucleus", "x_location", "y_location", "qv"]).to_parquet(
        raw / "s1_transcripts.parquet")
    pd.concat([_square(c, x, y, 4.8) for c, x, y in cells[["cell_id", "x", "y"]].itertuples(index=False)]).to_parquet(
        raw / "s1_cell_boundaries.parquet")
    pd.concat([_square(c, x, y, 2.0) for c, x, y in cells[["cell_id", "x", "y"]].itertuples(index=False)]).to_parquet(
        raw / "s1_nucleus_boundaries.parquet")
    out = tmp_path / "crop"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"), env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, str(SCRIPT), "--raw", str(raw), "--sample", "s1", "--labels",
                        str(tmp_path / "s1.labels.csv.gz"), "--out", str(out), "--half", "12"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    m = json.loads((out / "meta.json").read_text())
    assert m["cell"] == "t1" and m["partner"] in ("a", "b")
    K = pd.read_csv(out / "crop_counts.csv", index_col=0)
    assert K.loc["nucleus", "T"] == 20 and K.loc["outside the nucleus", "P"] == 20
    C = pd.read_csv(out / "crop_cells.csv")
    assert set(C.role) == {"target", "partner", "other"} and "t2" not in set(C.cell_id)
    assert (out / "crop.pdf").stat().st_size < 200_000
