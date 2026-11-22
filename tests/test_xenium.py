"""Xenium raw-output loader (examples/xenium/xenium_to_h5ad.py) on a fabricated archive."""
import gzip
import importlib.util
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

pytest.importorskip("pyarrow")
SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "xenium" / "xenium_to_h5ad.py"


def _fake_archive(tmp, name="xenium_output_patient3_pre_treatment", n_cells=200, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(80)]
    feats = [(f"ENSG{i}", g, "Gene Expression") for i, g in enumerate(genes)] + \
            [("NC1", "NegControlProbe_1", "Negative Control Probe"), ("U1", "UnassignedCodeword_1", "Unassigned Codeword")]
    cells = [f"c{i:04d}-1" for i in range(n_cells)]
    rows = []
    for ci, c in enumerate(cells):
        n = int(rng.integers(5, 150))
        for _ in range(n):
            g = int(rng.integers(0, len(genes)))
            rows.append((c, g, int(rng.random() < 0.4), 30 if rng.random() < 0.95 else 10))
    T = pd.DataFrame(rows, columns=["cell_id", "g", "overlaps_nucleus", "qv"])
    T["feature_name"] = [genes[g] for g in T.g]
    extra = pd.DataFrame({"cell_id": "UNASSIGNED", "feature_name": "G1", "overlaps_nucleus": 0, "qv": 35}, index=range(50))
    T = pd.concat([T.drop(columns="g"), extra], ignore_index=True)
    good = T[(T.qv >= 20) & (T.cell_id != "UNASSIGNED")]
    gi = {g: i for i, g in enumerate(genes)}
    ci = {c: i for i, c in enumerate(cells)}
    M = sp.coo_matrix((np.ones(len(good)), ([gi[g] for g in good.feature_name], [ci[c] for c in good.cell_id])),
                      shape=(len(feats), n_cells)).tocsr()
    M.sum_duplicates()
    C = pd.DataFrame({"cell_id": cells, "x_centroid": rng.uniform(0, 500, n_cells),
                      "y_centroid": rng.uniform(0, 500, n_cells),
                      "transcript_counts": np.asarray(M.sum(0)).ravel().astype(int),
                      "cell_area": rng.uniform(20, 80, n_cells), "nucleus_area": rng.uniform(10, 30, n_cells),
                      "segmentation_method": rng.choice(["Segmented by interior stain (18S)",
                                                         "Segmented by nucleus expansion of 5.0µm"], n_cells)})
    d = tmp / "build" / name
    (d / "cell_feature_matrix").mkdir(parents=True)
    with gzip.open(d / "cell_feature_matrix" / "matrix.mtx.gz", "wb") as fh:
        scipy.io.mmwrite(fh, M.astype(int))
    pd.Series(cells).to_csv(d / "cell_feature_matrix" / "barcodes.tsv.gz", index=False, header=False)
    pd.DataFrame(feats).to_csv(d / "cell_feature_matrix" / "features.tsv.gz", sep="\t", index=False, header=False)
    C.to_parquet(d / "cells.parquet")
    T.to_parquet(d / "transcripts.parquet")
    (d / "morphology_focus").mkdir()
    (d / "morphology_focus" / "morphology_focus_0000.ome.tif").write_bytes(b"x" * 1000)
    arch = tmp / "data" / f"{name}.tar.gz"
    arch.parent.mkdir(exist_ok=True)
    with tarfile.open(arch, "w:gz") as tf:
        tf.add(d, arcname=name)
    return arch, M, good


def test_xenium_loader_survey_and_combine(tmp_path):
    import anndata as ad
    arch, M, good = _fake_archive(tmp_path)
    _fake_archive(tmp_path, name="xenium_output_patient3_post_treatment", seed=1)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "data"), "--out", str(out), "--jobs", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    A = ad.read_h5ad(out / "samples" / "p03_pre.h5ad")
    assert A.n_vars == 80 and A.n_obs == 200                       # control features dropped
    assert np.array_equal(A.X.toarray(), M[:80].T.toarray())      # counts preserved
    assert A.uns["xenium"]["tx_match_share"] == 1.0                # qv >= 20 and assignment reproduce the matrix
    nuc = good[good.overlaps_nucleus == 1].groupby("cell_id").size()
    assert np.array_equal(A.obs.nuclear_total.to_numpy(), nuc.reindex([c.split(":")[1] for c in A.obs_names],
                                                                      fill_value=0).to_numpy())
    assert (A.layers["nuclear"].toarray() <= A.X.toarray()).all()
    assert (A.obs.passes_filter == (A.obs.n_genes > 50)).all()
    S = pd.read_csv(out / "survey.csv")
    assert set(S["sample"]) == {"p03_pre", "p03_post"} and (S.tx_match_share == 1.0).all()
    # rerun skips finished samples; combine pre only, with labels
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "data"), "--out", str(out)], capture_output=True,
                       text=True)
    assert r.returncode == 0 and "'seconds': 0" in r.stdout
    L = pd.DataFrame({"sample": "p03_pre", "cell_id": A.obs.cell_id.astype(str).to_numpy()[:150], "celltype": "F"})
    L.to_csv(tmp_path / "labels.csv", index=False)
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "data"), "--out", str(out), "--combine", "ra_pre",
                        "--timepoint", "pre", "--labels", str(tmp_path / "labels.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    B = ad.read_h5ad(out / "ra_pre.h5ad")
    assert B.n_obs == 200 and set(B.obs["sample"].astype(str)) == {"p03_pre"}
    assert B.obs.celltype.notna().sum() == 150
    # coverage over the combined samples only, both sides; stop flags; column clashes refused
    Ap = ad.read_h5ad(out / "samples" / "p03_post.h5ad")
    L2 = pd.concat([L, pd.DataFrame({"sample": "p03_post", "cell_id": Ap.obs.cell_id.astype(str).to_numpy(),
                                     "celltype": "G"})])
    L2.to_csv(tmp_path / "labels2.csv", index=False)
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "data"), "--out", str(out), "--combine", "ra_pre2",
                        "--timepoint", "pre", "--labels", str(tmp_path / "labels2.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    C = pd.read_csv(out / "ra_pre2_label_coverage.csv")
    assert C["sample"].tolist() == ["p03_pre"] and C.cells.tolist() == [200] and C.cells_labelled.tolist() == [150]
    assert C.label_rows.tolist() == [150] and C.label_rows_matched.tolist() == [150]
    assert "200 rows of other samples ignored" in r.stdout and "STOP" in r.stdout and "0.2500 unlabelled" in r.stdout
    B2 = ad.read_h5ad(out / "ra_pre2.h5ad")
    assert B2.uns["xenium_combined"]["label_coverage"]["share_unmatched"] == 0.0
    L.assign(timepoint="x").to_csv(tmp_path / "labels3.csv", index=False)
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "data"), "--out", str(out), "--combine", "ra_pre3",
                        "--timepoint", "pre", "--labels", str(tmp_path / "labels3.csv")], capture_output=True, text=True)
    assert r.returncode != 0 and "share names with obs columns" in r.stderr
