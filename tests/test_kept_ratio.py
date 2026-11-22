"""`claimbreak kept-ratio` end to end on fabricated Xenium output folders, and claimbreak.ingest against the
example converter."""
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("pyarrow")
from claimbreak import ingest  # noqa: E402
from claimbreak.cli import main  # noqa: E402

EX = Path(__file__).resolve().parents[1] / "examples" / "compartment"


def _xenium_outs(root: Path, seed=11, n=1500, G=60):
    """A reference (Level2: T, Myeloid, Fibro, Tumor) and two Xenium output folders whose cytoplasm carries the
    transcripts of the nearest neighbor; T cells are small (few counts), tumor cells large."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    types = ["T", "Myeloid", "Fibro", "Tumour"]
    prof = {t: rng.dirichlet(np.full(G, 0.6)) for t in types}
    genes = [f"g{i:02d}" for i in range(G)]
    lab = np.repeat(types, 200)
    X = np.vstack([rng.multinomial(rng.integers(200, 500), prof[t]) for t in lab])
    obs = pd.DataFrame({"Level2": lab, "donor_id": np.tile(list("abcd"), len(lab) // 4)},
                       index=[f"r{i}" for i in range(len(lab))])
    ad.AnnData(sp.csr_matrix(X.astype(np.float32)), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(root / "ref.h5ad")
    size = {"T": (24, 40), "Myeloid": (40, 80), "Fibro": (40, 80), "Tumour": (70, 140)}
    outs = []
    for s in range(2):
        xy = rng.uniform(0, 1200, (n, 2))
        ct = rng.choice(types, n, p=[0.25, 0.2, 0.25, 0.3])
        _, nn = cKDTree(xy).query(xy, k=2)
        rows = []
        for i in range(n):
            lo, hi = size[ct[i]]
            nuc = rng.multinomial(rng.integers(lo, hi), prof[ct[i]])
            mix = 0.5 * prof[ct[i]] + 0.5 * prof[ct[nn[i, 1]]]
            cyt = rng.multinomial(rng.integers(lo, hi) * 2, mix)
            for g in np.flatnonzero(nuc):
                rows += [(f"c{i:05d}", genes[g], 1, 30.0)] * int(nuc[g])
            for g in np.flatnonzero(cyt):
                rows += [(f"c{i:05d}", genes[g], 0, 30.0)] * int(cyt[g])
        T = pd.DataFrame(rows, columns=["cell_id", "feature_name", "overlaps_nucleus", "qv"])
        ii = T.cell_id.str[1:].astype(int).to_numpy()
        T["x_location"] = xy[ii, 0] + rng.normal(0, 2, len(T))
        T["y_location"] = xy[ii, 1] + rng.normal(0, 2, len(T))
        T["z_location"] = rng.uniform(0, 10, len(T))
        T = pd.concat([T, pd.DataFrame({"cell_id": ["UNASSIGNED"] * 20, "feature_name": "g01", "overlaps_nucleus": 0,
                                        "qv": 35.0}),
                       pd.DataFrame({"cell_id": ["c00000"] * 5, "feature_name": "NegControlProbe_1",
                                     "overlaps_nucleus": 1, "qv": 35.0}),
                       pd.DataFrame({"cell_id": ["c00001"] * 7, "feature_name": "g02", "overlaps_nucleus": 1,
                                     "qv": 5.0})], ignore_index=True)
        T[["x_location", "y_location", "z_location"]] = T[["x_location", "y_location", "z_location"]].fillna(600.0)
        T.insert(0, "transcript_id", np.arange(len(T), dtype=np.int64) + 10 ** 9)
        C = pd.DataFrame({"cell_id": [f"c{i:05d}" for i in range(n)], "x_centroid": xy[:, 0], "y_centroid": xy[:, 1],
                          "cell_area": rng.uniform(20, 80, n), "nucleus_area": rng.uniform(10, 30, n)})
        d = root / f"run{s}" / "outs"
        d.mkdir(parents=True)
        T.to_parquet(d / "transcripts.parquet")
        C.to_parquet(d / "cells.parquet")
        outs.append(d)
    return root, outs


@pytest.fixture(scope="module")
def xen(tmp_path_factory):
    return _xenium_outs(tmp_path_factory.mktemp("xen"))


def test_find_sections(xen):
    root, outs = xen
    found = ingest.find_xenium_sections(root / "run0")
    assert [(f, p, n) for f, p, n in found] == [(outs[0], "", "run0")]
    assert {n for _, _, n in ingest.find_xenium_sections(root)} == {"run0", "run1"}      # outs two levels down
    assert ingest.find_xenium_sections(root, depth=0) == []


def test_ingest_matches_example_converter(xen, tmp_path):
    root, outs = xen
    # the example script reads GEO-style prefixed files: make a prefixed copy of section 0
    geo = tmp_path / "geo"
    geo.mkdir()
    for f in ("transcripts.parquet", "cells.parquet"):
        (geo / f"S0_{f}").write_bytes((outs[0] / f).read_bytes())
    r = subprocess.run([sys.executable, str(EX / "compartment_to_h5ad.py"), "--platform", "xenium", "--raw", str(geo),
                        "--sample", "S0", "--out", str(tmp_path / "ex"), "--tmp", str(tmp_path / "tmp")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    p = ingest.xenium_to_h5ad(outs[0], tmp_path / "pkg", name="S0", verbose=False)
    a, b = ad.read_h5ad(tmp_path / "ex" / "S0.h5ad"), ad.read_h5ad(p)
    assert list(a.var_names) == list(b.var_names) and list(a.obs_names) == list(b.obs_names)
    for L in (None, "nucleus", "cyto"):
        x = a.X if L is None else a.layers[L]
        y = b.X if L is None else b.layers[L]
        assert (sp.csr_matrix(x) != sp.csr_matrix(y)).nnz == 0
    assert "NegControlProbe_1" not in b.var_names
    assert b.obs.n_whole.iloc[1] == a.obs.n_whole.iloc[1]                       # the qv 5 transcripts dropped
    with pytest.raises(FileExistsError):
        ingest.xenium_to_h5ad(outs[0], tmp_path / "pkg", name="S0", verbose=False)
    with pytest.raises(ValueError):
        ingest.xenium_to_h5ad(outs[0], outs[0] / "inside", verbose=False)       # never writes into the input


def test_kept_ratio_end_to_end(xen, tmp_path):
    root, outs = xen
    out = tmp_path / "kr"
    rc = main(["kept-ratio", "--xenium", str(root / "run0"), str(root / "run1"), "--ref", str(root / "ref.h5ad"),
               "--ref-level", "Level2", "--out", str(out), "--block-draws", "50", "--min-null-cells", "200",
               "--tile", "300", "--quiet"])
    assert rc == 0
    for f in ("report.md", "kept_ratio.csv", "sections.csv", "settings.json", "audit/support.csv",
              "summary/units_kept.csv", "labels/run0.labels.csv.gz", "h5ad/run1.h5ad"):
        assert (out / f).exists(), f
    sec = pd.read_csv(out / "sections.csv")
    assert set(sec["sample"]) == {"run0", "run1"} and sec.halves_agreement.between(0, 1).all()
    K = pd.read_csv(out / "kept_ratio.csv")
    assert {"unit", "type", "kept_ratio", "kept_ratio_lo", "kept_ratio_hi", "cells_null", "enough_cells"} <= set(K)
    assert {"null_kept", "stable"} <= set(K) and K.null_kept.between(0, 1).all()
    assert (K.stable == (K.null_kept >= 0.5)).all()
    t = K[K.type == "T"]
    assert len(t) == 2 and (t.kept_ratio < 1).all()                              # small cells lose labels
    big = K[K.type == "Tumour"]
    assert (big.kept_ratio > t.kept_ratio.max()).all()                           # large cells lose fewer
    text = (out / "report.md").read_text()
    assert "## How many labels do whole-cell counts lose?" in text and "| run0 | T |" in text
    assert "Null keeps:" in text and "a reference that covers the tissue" not in text   # no audit warning here
    assert json.loads((out / "settings.json").read_text())["ref_level"] == "Level2"
    # a second run into the same folder is refused, the first is left alone
    assert main(["kept-ratio", "--xenium", str(root / "run0"), "--ref", str(root / "ref.h5ad"), "--ref-level",
                 "Level2", "--out", str(out), "--quiet"]) == 1
    # the same chain from the h5ad files gives the same kept ratios
    out2 = tmp_path / "kr2"
    assert main(["kept-ratio", "--h5ad", str(out / "h5ad" / "run0.h5ad"), str(out / "h5ad" / "run1.h5ad"), "--ref",
                 str(root / "ref.h5ad"), "--ref-level", "Level2", "--out", str(out2), "--block-draws", "50",
                 "--min-null-cells", "200", "--tile", "300", "--quiet"]) == 0
    pd.testing.assert_frame_equal(K, pd.read_csv(out2 / "kept_ratio.csv"))


def test_kept_ratio_needs_one_source(xen, tmp_path):
    root, _ = xen
    with pytest.raises(SystemExit):
        main(["kept-ratio", "--ref", str(root / "ref.h5ad"), "--ref-level", "Level2", "--out", str(tmp_path / "x")])
    assert main(["kept-ratio", "--xenium", str(tmp_path), "--ref", str(root / "ref.h5ad"), "--ref-level", "Level2",
                 "--out", str(tmp_path / "y"), "--quiet"]) == 1                 # no section in that folder
