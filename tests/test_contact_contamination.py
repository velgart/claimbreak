"""contact_contamination.py: the neighbor's RNA by contact, whole-cell against nucleus counts."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "compartment" / "contact_contamination.py"


def _section(tmp, name, rng, h5="h5", labdir="lab"):
    ad = pytest.importorskip("anndata")
    import scipy.sparse as sp
    n = 5000
    xy = rng.uniform(0, 1500, (n, 2))
    lab = np.where(xy[:, 0] < 750, "Tumour", "Fibro").astype(object)
    lab[rng.choice(n, 700, replace=False)] = "T_NK"
    G = 60
    nuc = np.full((n, G), 0.2)
    nuc[:, :20] = 0.02
    nuc[lab == "Tumour", :20] = 3.0
    nuc[lab == "T_NK", 20:30] = 3.0
    cyto = nuc * 0.8
    inside = (lab == "T_NK") & (xy[:, 0] < 730)                  # touching tumor: the mask collects its RNA
    cyto[inside, :20] += 1.0
    N = rng.poisson(nuc)
    W = N + rng.poisson(cyto)
    ids = np.array([f"{name}_{i}" for i in range(n)])
    A = ad.AnnData(X=sp.csr_matrix(W.astype(np.float32)), obs=pd.DataFrame(index=ids),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(G)]))
    A.layers["nucleus"] = sp.csr_matrix(N.astype(np.float32))
    A.write_h5ad(tmp / h5 / f"{name}.h5ad")
    pd.DataFrame(dict(cell_id=ids, x=xy[:, 0], y=xy[:, 1], L2m_nucA=lab)).to_csv(
        tmp / labdir / f"{name}.labels.csv.gz", index=False, compression="gzip")


def test_whole_cell_contamination_found_nucleus_flat(tmp_path):
    (tmp_path / "h5").mkdir(); (tmp_path / "lab").mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        _section(tmp_path, f"S_L{i}_L{i}", rng)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), "--labels", str(tmp_path / "lab"), "--h5ad", str(tmp_path / "h5"),
                        "--partner", "Tumour", "--unit-pattern", r"L\d(?=_L)", "--n-markers", "20", "--min-ratio", "5",
                        "--boot", "200", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    S = json.loads((out / "summary.json").read_text())
    assert S["immune"]["whole"]["median_ratio"] > 2
    assert 0.8 < S["immune"]["nucleus"]["median_ratio"] < 1.25
    assert S["immune"]["whole"]["units"] == 3


def test_positional_error_dilutes_contact(tmp_path):
    """--jitter moves cells before contact is computed, --drop-frac removes cells; the whole-cell excess at
    contact shrinks with the positional error and the summary keeps the first level at the top."""
    (tmp_path / "h5").mkdir(); (tmp_path / "lab").mkdir()
    rng = np.random.default_rng(1)
    for i in range(3):
        _section(tmp_path, f"S_L{i}_L{i}", rng)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), "--labels", str(tmp_path / "lab"), "--h5ad", str(tmp_path / "h5"),
                        "--partner", "Tumour", "--unit-pattern", r"L\d(?=_L)", "--n-markers", "20", "--min-ratio", "5",
                        "--boot", "100", "--jitter", "0,60", "--drop-frac", "0.4", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    S = json.loads((out / "summary.json").read_text())
    j0, j60 = S["by_jitter"]["0"]["immune"]["whole"], S["by_jitter"]["60"]["immune"]["whole"]
    assert S["immune"]["whole"]["median_ratio"] == j0["median_ratio"]
    assert j0["median_ratio"] > 2 and j60["median_ratio"] < j0["median_ratio"]
    assert (out / "by_jitter.csv").exists() and (out / "immune_by_section_j60.csv").exists()
    C = pd.read_csv(out / "cells.csv.gz")
    assert set(C.jitter) == {0, 60}
    n_t = (C.kind == "immune").sum() / 2 / 3                   # T/NK cells kept per section and level
    assert 0.5 * 700 < n_t < 0.7 * 700


def _mislabelled(tmp, name, rng, seed=7):
    """no nuclear contamination, but tumor cells at the tumor edge labeled T_NK with low margin.
    Writes meta.json, gamma.csv and n_nucA as the labeling does (count_sets split of the label genes)."""
    ad = pytest.importorskip("anndata")
    import scipy.sparse as sp
    from claimbreak.compartment import count_sets
    n = 5000
    xy = rng.uniform(0, 1500, (n, 2))
    truth = np.where(xy[:, 0] < 750, "Tumour", "Fibro").astype(object)
    truth[rng.choice(n, 700, replace=False)] = "T_NK"
    G = 60
    nuc = np.full((n, G), 0.2)
    nuc[:, :20] = 0.02
    nuc[truth == "Tumour", :20] = 3.0
    nuc[truth == "T_NK", 20:30] = 3.0
    N = rng.poisson(nuc)
    W = N + rng.poisson(nuc * 0.8)
    lab = truth.copy()
    margin = rng.uniform(30, 80, n)
    edge = np.where((truth == "Tumour") & (xy[:, 0] > 700))[0]
    wrong = rng.choice(edge, len(edge) // 3, replace=False)        # tumor cells next to the stroma, called T_NK
    lab[wrong] = "T_NK"
    margin[wrong] = rng.uniform(0, 10, len(wrong))
    ids = np.array([f"{name}_{i}" for i in range(n)])
    genes = [f"g{j}" for j in range(G)]
    A = ad.AnnData(X=sp.csr_matrix(W.astype(np.float32)), obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=genes))
    A.layers["nucleus"] = sp.csr_matrix(N.astype(np.float32))
    A.write_h5ad(tmp / "h5" / f"{name}.h5ad")
    lg = genes[5:]                                                  # label genes: a subset, in a different order
    lg = lg[::-1]
    xi = [genes.index(g) for g in lg]                               # as label_section: csr, then the label genes
    S = count_sets(A.X[:, xi].astype(np.float32), A.layers["nucleus"][:, xi].astype(np.float32), None,
                   ["nucA", "nucA3"], seed)
    (tmp / "lab" / f"{name}.meta.json").write_text(json.dumps(dict(seed=seed, arms=["nucA", "nucB"])))
    pd.DataFrame(dict(gene=lg)).to_csv(tmp / "lab" / f"{name}.gamma.csv", index=False)
    pd.DataFrame(dict(cell_id=ids, x=xy[:, 0], y=xy[:, 1], L2m_nucA=lab, L2m_nucA_margin=margin,
                      L2m_nucB=np.where(np.isin(np.arange(n), wrong[::2]), "Tumour", lab),
                      n_nucA=np.asarray(S["nucA"].sum(1)).ravel().astype(int),
                      n_nucA3=np.asarray(S["nucA3"].sum(1)).ravel().astype(int))).to_csv(
        tmp / "lab" / f"{name}.labels.csv.gz", index=False, compression="gzip")
    cd3 = np.where(truth == "T_NK", np.exp(rng.normal(5, 0.3, n)),
                   np.where(rng.random(n) < 0.3, np.exp(rng.normal(1, 0.3, n)), 0.0))
    cd45 = np.where(truth == "T_NK", np.exp(rng.normal(5, 0.3, n)), np.where(rng.random(n) < 0.3,
                                                                              np.exp(rng.normal(1, 0.3, n)), 0.0))
    pck = np.where(truth == "Tumour", np.exp(rng.normal(5, 0.3, n)), np.where(rng.random(n) < 0.3,
                                                                              np.exp(rng.normal(1, 0.3, n)), 0.0))
    pd.DataFrame(dict(cell_id=ids, CD3E=cd3, CD45=cd45, PanCK=pck)).to_csv(tmp / f"{name}.protein.csv.gz", index=False,
                                                                compression="gzip")
    return wrong


def _run(tmp, out, *extra):
    r = subprocess.run([sys.executable, str(SCRIPT), "--labels", str(tmp / "lab"), "--h5ad", str(tmp / "h5"),
                        "--partner", "Tumour", "--unit-pattern", r"L\d(?=_L)", "--n-markers", "20", "--min-ratio", "5",
                        "--boot", "100", "--out", str(tmp / out), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads((tmp / out / "summary.json").read_text())


def test_mislabelled_partners_and_checks(tmp_path):
    """Mislabeled tumor cells make a nuclear excess; confident calls, agreement with an arm that relabels them,
    and protein types remove it; half B is rebuilt exactly (n_nucA) and loads are on the label genes."""
    (tmp_path / "h5").mkdir(); (tmp_path / "lab").mkdir()
    rng = np.random.default_rng(3)
    for i in range(3):
        _mislabelled(tmp_path, f"S_L{i}_L{i}", rng)
    base = _run(tmp_path, "base")
    assert base["immune"]["nucleus"]["median_ratio"] > 1.5
    conf = _run(tmp_path, "conf", "--min-margin", "20")
    assert 0.8 < conf["immune"]["nucleus"]["median_ratio"] < 1.25
    hb = _run(tmp_path, "hb", "--measure", "nucB", "--min-margin", "20")
    assert hb["measures"] == ["whole", "nucleus", "nucB"]
    assert 0.8 < hb["immune"]["nucB"]["median_ratio"] < 1.3 and "nucB_trim" in hb["immune"]
    M = pd.read_csv(tmp_path / "hb" / "markers.csv")
    assert (M.split_reproduced == 1).all() and (M.n_genes == 55).all()
    assert (M.n_immune_measured < M.n_immune).all()
    P = pd.read_csv(tmp_path / "hb" / "immune_by_section.csv")
    assert {"nucB_ratio", "nucB_ratio_trim", "whole_ratio_trim"} <= set(P.columns)
    h3 = _run(tmp_path, "h3", "--measure", "nucB3", "--min-margin", "20")
    assert (pd.read_csv(tmp_path / "h3" / "markers.csv").split_reproduced == 1).all()
    assert 0.75 < h3["immune"]["nucB3"]["median_ratio"] < 1.35
    ag = _run(tmp_path, "ag", "--agree", "nucB")
    assert ag["immune"]["nucleus"]["median_ratio"] < base["immune"]["nucleus"]["median_ratio"]
    pr = _run(tmp_path, "pr", "--protein", str(tmp_path / "S_L0_L0.protein.csv.gz"), "--sample", "S_L0_L0",
              "--partner-protein", "PANCK+", "--tile", "300")
    assert pr["immune"]["nucleus"]["units"] >= 3
    assert 0.75 < pr["immune"]["nucleus"]["median_ratio"] < 1.3


def test_split_mismatch_skipped(tmp_path):
    """a meta.json seed that is not the labeling's: half A does not reproduce n_nucA, the section is skipped."""
    (tmp_path / "h5").mkdir(); (tmp_path / "lab").mkdir()
    rng = np.random.default_rng(4)
    for i in range(2):
        _mislabelled(tmp_path, f"S_L{i}_L{i}", rng)
    (tmp_path / "lab" / "S_L1_L1.meta.json").write_text(json.dumps(dict(seed=99)))
    S = _run(tmp_path, "o", "--measure", "nucB")
    assert any("S_L1_L1" in x and "not the labelling split" in x for x in S["skipped"])
    assert S["immune"]["nucB"]["sections"] == 1


def test_tail_separates_mislabelling_from_contamination(tmp_path):
    """mislabeled minority: mean ratio high, median-quantile ratio and AUC near null; contamination of every T/NK
    cell at contact (whole cell in _section): the whole distribution moves. --from-cells reads an earlier run."""
    (tmp_path / "h5").mkdir(); (tmp_path / "lab").mkdir()
    rng = np.random.default_rng(5)
    for i in range(3):
        _mislabelled(tmp_path, f"S_L{i}_L{i}", rng)
    S = _run(tmp_path, "mis")
    t = S["immune"]["tail"]["nucleus"]
    assert t["mean_ratio"] > 1.5 and t["auc"] < 0.6 and t["q90_ratio"] > 3 * t["q75_ratio"]
    T = pd.read_csv(tmp_path / "mis" / "immune_tail.csv")
    assert {"nucleus_q90_ratio", "whole_auc", "nucleus_pos_ratio"} <= set(T.columns)
    r = subprocess.run([sys.executable, str(SCRIPT), "--from-cells", str(tmp_path / "mis"), "--out",
                        str(tmp_path / "fc")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    F = json.loads((tmp_path / "fc" / "tail_summary.json").read_text())
    assert F["immune"]["nucleus"]["mean_ratio"] == pytest.approx(t["mean_ratio"])
    (tmp_path / "h5b").mkdir(); (tmp_path / "labb").mkdir()
    for i in range(3):
        _section(tmp_path, f"S_L{i}_L{i}", rng, h5="h5b", labdir="labb")
    r = subprocess.run([sys.executable, str(SCRIPT), "--labels", str(tmp_path / "labb"), "--h5ad", str(tmp_path / "h5b"),
                        "--partner", "Tumour", "--unit-pattern", r"L\d(?=_L)", "--n-markers", "20", "--min-ratio", "5",
                        "--boot", "50", "--out", str(tmp_path / "con")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    w = json.loads((tmp_path / "con" / "summary.json").read_text())["immune"]["tail"]["whole"]
    assert w["drop20_ratio"] > 1.5 and w["auc"] > 0.8 and w["q90_ratio"] < 3 * w["q75_ratio"]
