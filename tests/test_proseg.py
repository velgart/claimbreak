"""examples/compartment/proseg_to_h5ad.py: Proseg assignments joined to Xenium transcripts, cells mapped to
Xenium nuclei, nucleus layer = own nuclear transcripts kept by the Proseg cell."""
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
EX = Path(__file__).resolve().parents[1] / "examples" / "compartment" / "proseg_to_h5ad.py"
S = "GSM1_NSCLC_lung_L1_L1"


def _xenium(raw: Path, seed=0):
    """Three Xenium cells c0, c1, c2 with nuclear and cytoplasmic transcripts, one control probe, low-qv transcripts and
    unassigned transcripts. Proseg (below) keeps c0, moves 3 of c1's cytoplasmic transcripts to c0, takes 2 of c2's
    nuclear transcripts into c0, and makes one extra cell from unassigned transcripts only."""
    rng = np.random.default_rng(seed)
    rows = []
    tid = 1000
    for c, (nn, nc) in {"c0": (10, 8), "c1": (6, 9), "c2": (12, 4)}.items():
        for k in range(nn + nc):
            rows.append(dict(transcript_id=tid, cell_id=c, overlaps_nucleus=int(k < nn),
                             feature_name=f"G{rng.integers(0, 5)}", qv=30.0))
            tid += 1
    for k in range(5):
        rows.append(dict(transcript_id=tid, cell_id="UNASSIGNED", overlaps_nucleus=0, feature_name="G1", qv=30.0)); tid += 1
    rows.append(dict(transcript_id=tid, cell_id="c0", overlaps_nucleus=1, feature_name="NegControlProbe_1", qv=30.0)); tid += 1
    rows.append(dict(transcript_id=tid, cell_id="c1", overlaps_nucleus=1, feature_name="G2", qv=5.0)); tid += 1
    X = pd.DataFrame(rows)
    raw.mkdir(parents=True)
    X.to_parquet(raw / f"{S}_transcripts.parquet")
    pd.DataFrame({"cell_id": ["c0", "c1", "c2"], "x_centroid": [1.0, 2.0, 3.0], "y_centroid": [0.0, 0.0, 0.0],
                  "cell_area": [100.0] * 3, "nucleus_area": [30.0] * 3}).to_parquet(raw / f"{S}_cells.parquet")
    return X


def _proseg(X, d: Path, with_id=True):
    """Proseg cells 0 (c0), 1 (c1), 2 (c2), 3 (unassigned only); rows are the qv >= 20 transcripts in file order."""
    Y = X[X.qv >= 20].reset_index(drop=True)
    cell = Y.cell_id.map({"c0": 0, "c1": 1, "c2": 2, "UNASSIGNED": 3}).astype(float).to_numpy().copy()
    c1_cyto = np.flatnonzero((Y.cell_id == "c1") & (Y.overlaps_nucleus == 0))[:3]
    c2_nuc = np.flatnonzero((Y.cell_id == "c2") & (Y.overlaps_nucleus == 1))[:2]
    cell[c1_cyto] = 0
    cell[c2_nuc] = 0
    cell[np.flatnonzero(Y.cell_id == "c2")[-1]] = np.nan             # background
    d.mkdir(parents=True)
    T = pd.DataFrame({"gene": Y.feature_name, "x": 0.0, "y": 0.0, "assignment": cell})
    if with_id:
        T.insert(0, "transcript_id", Y.transcript_id)
    T.to_csv(d / "transcript-metadata.csv.gz", index=False)
    pd.DataFrame({"cell": [0, 1, 2, 3], "original_cell_id": ["c0", "c1", "c2", "UNASSIGNED"],
                  "centroid_x": [1, 2, 3, 4], "centroid_y": 0}).to_csv(d / "cell-metadata.csv.gz", index=False)
    return Y, cell


@pytest.mark.parametrize("with_id", [True, False])
def test_convert(tmp_path, with_id):
    X = _xenium(tmp_path / "raw")
    Y, cell = _proseg(X, tmp_path / "ps", with_id)
    r = subprocess.run([sys.executable, str(EX), "convert", "--proseg", str(tmp_path / "ps"), "--raw", str(tmp_path / "raw"),
                        "--sample", S, "--out", str(tmp_path / "out"), "--tmp", str(tmp_path / "tmp")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:] + r.stdout[-1500:]
    A = ad.read_h5ad(tmp_path / "out" / f"{S}.h5ad")
    assert list(A.obs_names) == ["c0", "c1", "c2"]                       # cell 3 has no nucleus: dropped
    assert A.uns["proseg"]["join"] == ("transcript_id" if with_id else "row order")
    assert A.uns["proseg"]["original_id_agreement"] == 1.0
    genes = ~Y.feature_name.str.startswith("NegControl")
    for i, c in enumerate(["c0", "c1", "c2"]):
        mine = (cell == i) & genes.to_numpy()
        assert A.obs.loc[c, "n_whole"] == mine.sum()
        own_nuc = mine & (Y.cell_id == c).to_numpy() & (Y.overlaps_nucleus == 1).to_numpy()
        assert A.obs.loc[c, "n_nucleus"] == own_nuc.sum()
    # c0 holds 2 of c2's nuclear transcripts: in its whole-cell counts, not in its nucleus layer
    assert A.obs.loc["c0", "n_whole"] == 10 + 8 + 3 + 2
    assert A.obs.loc["c0", "n_nucleus"] == 10
    assert A.obs.loc["c2", "n_nucleus"] == 10 and abs(A.obs.loc["c2", "share_own_nucleus"] - 10 / 12) < 1e-9
    assert abs(A.layers["cyto"] - (A.X - A.layers["nucleus"])).sum() == 0
    assert (A.layers["cyto"].toarray() >= 0).all()


def test_probe_and_misaligned_rows_stop(tmp_path):
    X = _xenium(tmp_path / "raw")
    _proseg(X, tmp_path / "ps", with_id=False)
    r = subprocess.run([sys.executable, str(EX), "probe", "--proseg", str(tmp_path / "ps")], capture_output=True, text=True)
    assert r.returncode == 0 and "assignment" in r.stdout
    T = pd.read_csv(tmp_path / "ps" / "transcript-metadata.csv.gz")
    T.iloc[::-1].reset_index(drop=True).to_csv(tmp_path / "ps" / "transcript-metadata.csv.gz", index=False)
    r = subprocess.run([sys.executable, str(EX), "convert", "--proseg", str(tmp_path / "ps"), "--raw", str(tmp_path / "raw"),
                        "--sample", S, "--out", str(tmp_path / "out"), "--tmp", str(tmp_path / "tmp")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "do not match" in (r.stdout + r.stderr)
