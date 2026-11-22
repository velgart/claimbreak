"""examples/ra: RA sample files -> compartment layout; synovium coarse mapping."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
ad = pytest.importorskip("anndata")


def _mod(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ra_to_compartment(tmp_path):
    src = tmp_path / "samples"
    src.mkdir()
    rng = np.random.default_rng(0)
    for tp in ("pre", "post"):
        n, g = 50, 8
        X = rng.poisson(3, (n, g))
        N = rng.binomial(X, 0.4)
        o = pd.DataFrame({"sample": f"p01_{tp}", "patient": "p01", "timepoint": tp, "x": rng.random(n),
                          "y": rng.random(n), "segmentation_method": "Segmented by interior stain",
                          "cell_area": 50.0, "nucleus_area": 20.0, "passes_filter": np.arange(n) >= 5},
                         index=[f"c{i}" for i in range(n)])
        A = ad.AnnData(sp.csr_matrix(X.astype(np.int32)), obs=o, var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
        A.layers["nuclear"] = sp.csr_matrix(N.astype(np.int32))
        A.write_h5ad(src / f"p01_{tp}.h5ad")
    out = tmp_path / "comp"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "ra" / "ra_to_compartment.py"), "--samples", str(src),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert [f.name for f in out.glob("*.h5ad")] == ["p01_pre.h5ad"]
    B = ad.read_h5ad(out / "p01_pre.h5ad")
    assert B.n_obs == 45 and {"nucleus", "cyto"} <= set(B.layers)
    assert np.allclose((B.layers["nucleus"] + B.layers["cyto"]).toarray(), B.X.toarray())
    assert str(B.obs["unit"].iloc[0]) == "p01" and str(B.obs["panel"].iloc[0]) == "5K+100"
    assert np.allclose(B.obs.n_whole, np.asarray(B.X.sum(1)).ravel())


def test_synovium_coarse():
    m = _mod("examples/ra/synovium_reference.py")
    assert m.coarse("type A synoviocyte") == "Myeloid"
    assert m.coarse("fibroblast of synovial tissue") == "Fibroblast"
    assert m.coarse("synovial macrophage") == "Myeloid"
    assert m.coarse("CD8-positive, alpha-beta T cell") == "T_NK"
    assert m.coarse("plasma cell") == "Plasma"
    assert m.coarse("hematopoietic stem cell") == ""


def test_gse200815_reference_cluster_and_build(tmp_path):
    pytest.importorskip("scanpy")
    pytest.importorskip("igraph")
    import gzip
    import tarfile
    import scipy.io
    m = _mod("examples/ra/gse200815_reference.py")
    rng = np.random.default_rng(0)
    markers = sorted({g for v in m.LINEAGE.values() for g in v})
    genes = markers + [f"G{i}" for i in range(150)] + ["MT-CO1"]
    gi = {g: i for i, g in enumerate(genes)}
    files = tmp_path / "files"
    files.mkdir()
    for name in ("GSM1_RA001", "GSM2_RA002", "GSM3_PsA003"):
        cols = []
        for i in range(640):
            t = list(m.LINEAGE)[i % len(m.LINEAGE)]
            lam = np.full(len(genes), 0.5)
            lam[[gi[g] for g in m.LINEAGE[t]]] += 12
            cols.append(rng.poisson(lam))
        cols += [rng.poisson(np.full(len(genes), 0.01)) for _ in range(800)]
        M = sp.csc_matrix(np.array(cols).T)
        with gzip.open(files / f"{name}_matrix.mtx.gz", "wb") as f:
            scipy.io.mmwrite(f, M)
        pd.DataFrame({0: [f"E{i}" for i in range(len(genes))], 1: genes, 2: "Gene Expression"}).to_csv(
            files / f"{name}_features.tsv.gz", sep="\t", header=False, index=False)
        pd.Series([f"B{i}-1" for i in range(M.shape[1])]).to_csv(files / f"{name}_barcodes.tsv.gz", sep="\t",
                                                                  header=False, index=False)
    tar = tmp_path / "RAW.tar"
    with tarfile.open(tar, "w") as tf:
        for f in sorted(files.iterdir()):
            tf.add(f, arcname=f.name)
    work = tmp_path / "work"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "ra" / "gse200815_reference.py"), "cluster",
                        "--tar", str(tar), "--work", str(work), "--min-umi", "100", "--min-genes", "40",
                        "--resolution", "0.5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    T = pd.read_csv(work / "clusters.csv", keep_default_na=False)
    assert set(T.proposed) == set(m.LINEAGE)                      # every lineage found, RA samples only
    A = ad.read_h5ad(work / "cells.h5ad")
    assert set(A.obs["diagnosis"]) == {"RA"} and 1255 <= A.n_obs < 1280          # top 1% UMIs cut per sample
    out = tmp_path / "ref.h5ad"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "ra" / "gse200815_reference.py"), "build",
                        "--work", str(work), "--out", str(out), "--per-type", "100"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    R = ad.read_h5ad(out)
    assert set(R.obs.coarse) == set(m.LINEAGE) and R.obs.coarse.value_counts().max() <= 100
    assert {"donor_id", "coarse", "amp_state"} <= set(R.obs.columns) and "feature_name" in R.var
    # a reviewed "B|Plasma" cluster is split cell by cell; synthetic truth is i % 8 (LINEAGE order)
    Tm = T.copy()
    Tm["cluster"] = Tm.cluster.astype(str)
    A.obs["cluster"] = A.obs.cluster.astype(str)
    bp = Tm.proposed.isin(["B", "Plasma"])
    A.obs["truth"] = [list(m.LINEAGE)[int(b.split(":B")[1].split("-")[0]) % len(m.LINEAGE)] for b in A.obs_names]
    first = str(Tm.cluster[bp].iloc[0])
    A.obs.loc[A.obs.cluster.astype(str).isin(Tm.cluster[bp].astype(str)), "cluster"] = first
    A.write_h5ad(work / "cells.h5ad")
    Tm.loc[bp, "coarse"] = "B|Plasma"
    Tm.loc[bp, "cluster"] = first
    Tm = Tm.drop_duplicates("cluster")
    Tm.to_csv(work / "clusters_merged.csv", index=False)
    out2 = tmp_path / "ref2.h5ad"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "ra" / "gse200815_reference.py"), "build",
                        "--work", str(work), "--clusters", str(work / "clusters_merged.csv"), "--out", str(out2),
                        "--per-type", "1000"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "split: B" in r.stdout
    R2 = ad.read_h5ad(out2)
    sub = A.obs.loc[[c for c in R2.obs_names if R2.obs.coarse[c] in ("B", "Plasma")]]
    assert (sub.truth.to_numpy() == R2.obs.coarse[sub.index].astype(str).to_numpy()).mean() > 0.95
    assert set(R2.obs.coarse) == set(m.LINEAGE)
