"""examples/compartment/baysor_tiles.py: tiles, merge, and the mapping of Baysor cells to Xenium nuclei, on a
fabricated section and a fabricated Baysor run."""
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("pyarrow")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kept_ratio import _xenium_outs  # noqa: E402

EX = Path(__file__).resolve().parents[1] / "examples" / "compartment"


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2000:]
    return r.stdout


@pytest.fixture(scope="module")
def geo(tmp_path_factory):
    root, outs = _xenium_outs(tmp_path_factory.mktemp("bay"), n=900)
    g = root / "geo"
    g.mkdir()
    for f in ("transcripts.parquet", "cells.parquet"):
        (g / f"S0_{f}").write_bytes((outs[0] / f).read_bytes())
    return root, g


def _fake_baysor(tiles: Path, how: str, split_cell=None):
    """Baysor 'run' on every tile: the Xenium segmentation itself (cell = prior), noise where there is no prior.
    how: transcript_id | molecule_id; split_cell: a Xenium cell index whose cytoplasm becomes a separate Baysor cell."""
    P = pd.read_csv(tiles / "tiles.csv")
    for k, t in enumerate(P[P.run].itertuples()):
        T = pd.read_csv(tiles / t.tile / "transcripts.csv")
        cell = np.where(T.xcell > 0, "CR" + T.xcell.astype(str), "")
        if split_cell is not None:
            m = (T.xcell == split_cell + 1) & (T.overlaps_nucleus == 0)
            cell = np.where(m, "CRsplit", cell)
        S = pd.DataFrame({"x": T.x_location, "y": T.y_location, "gene": T.feature_name, "cell": cell,
                          "is_noise": cell == ""})
        if how == "transcript_id" or k % 2 == 0:
            S.insert(0, "transcript_id", T.transcript_id)
        else:
            S.insert(0, "molecule_id", np.arange(1, len(T) + 1))
        (tiles / t.tile / "out").mkdir()
        S.to_csv(tiles / t.tile / "out" / "segmentation.csv", index=False)


def test_identity_segmentation_reproduces_the_xenium_counts(geo, tmp_path):
    root, g = geo
    tiles = tmp_path / "tiles"
    _run(EX / "baysor_tiles.py", "prep", "--raw", g, "--sample", "S0", "--tiles", tiles, "--tile-um", "400",
         "--margin-um", "30", "--min-tile-transcripts", "10")
    P = pd.read_csv(tiles / "tiles.csv")
    assert P.run.sum() >= 4
    T = pd.read_csv(tiles / P[P.run].tile.iloc[0] / "transcripts.csv")
    assert {"transcript_id", "x_location", "y_location", "z_location", "feature_name", "overlaps_nucleus",
            "xcell"} <= set(T.columns)
    assert not T.feature_name.str.startswith("NegControl").any()
    _fake_baysor(tiles, "mixed")
    _run(EX / "baysor_tiles.py", "merge", "--raw", g, "--sample", "S0", "--tiles", tiles, "--out", tmp_path / "h5ad")
    _run(EX / "compartment_to_h5ad.py", "--platform", "xenium", "--raw", g, "--sample", "S0", "--out",
         tmp_path / "ref", "--tmp", tmp_path / "tmp")
    B, X = ad.read_h5ad(tmp_path / "h5ad" / "S0.h5ad"), ad.read_h5ad(tmp_path / "ref" / "S0.h5ad")
    X = X[X.obs.n_nucleus > 0]                                    # cells without a nucleus cannot be mapped
    assert sorted(B.obs_names) == sorted(X.obs_names)
    B = B[X.obs_names]
    assert list(B.var_names) == list(X.var_names)
    for L in (None, "nucleus", "cyto"):
        a = sp.csr_matrix(B.X if L is None else B.layers[L])
        b = sp.csr_matrix(X.X if L is None else X.layers[L])
        assert (a != b).nnz == 0, L
    assert np.allclose(B.obs.share_own_nucleus, 1)
    assert B.obs.tile.nunique() == P.run.sum() or B.obs.tile.nunique() >= 4


def test_split_cell_keeps_its_nucleus_and_refuses_overwrite(geo, tmp_path):
    root, g = geo
    tiles = tmp_path / "tiles"
    _run(EX / "baysor_tiles.py", "prep", "--raw", g, "--sample", "S0", "--tiles", tiles, "--tile-um", "400",
         "--min-tile-transcripts", "10")
    _fake_baysor(tiles, "transcript_id", split_cell=5)
    out = _run(EX / "baysor_tiles.py", "merge", "--raw", g, "--sample", "S0", "--tiles", tiles, "--out",
               tmp_path / "h5ad")
    assert "kept (one per nucleus)" in out
    B = ad.read_h5ad(tmp_path / "h5ad" / "S0.h5ad")
    c5 = B[B.obs_names == "c00005"]
    assert c5.n_obs == 1 and float(c5.layers["cyto"].sum()) == 0          # its cytoplasm went to another cell
    assert float(c5.X.sum()) == float(c5.layers["nucleus"].sum()) > 0
    r = subprocess.run([sys.executable, str(EX / "baysor_tiles.py"), "merge", "--raw", str(g), "--sample", "S0",
                        "--tiles", str(tiles), "--out", str(tmp_path / "h5ad")], capture_output=True, text=True)
    assert r.returncode != 0 and "not overwritten" in (r.stdout + r.stderr)
    r = subprocess.run([sys.executable, str(EX / "baysor_tiles.py"), "prep", "--raw", str(g), "--sample", "S0",
                        "--tiles", str(tiles)], capture_output=True, text=True)
    assert r.returncode != 0 and "not empty" in (r.stdout + r.stderr)
