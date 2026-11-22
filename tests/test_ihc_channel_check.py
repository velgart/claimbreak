"""Channel identity check of the lung IHC against transcripts (examples/correction/ihc_channel_check.py)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    p = ROOT / "examples" / "correction" / "ihc_channel_check.py"
    spec = importlib.util.spec_from_file_location("ihc_channel_check", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_auc_strat_ignores_depth_confounding():
    m = _mod()
    rng = np.random.default_rng(0)
    n = 20000
    depth = rng.lognormal(5, 0.6, n)
    v = np.log(depth) + rng.normal(0, 0.3, n)             # stain follows cell size only
    g = rng.random(n) < (depth / depth.max())             # gene detected more often in big cells
    assert m.auc(v[g], v[~g]) > 0.6                       # raw AUC is fooled by depth
    a, _, _ = m.auc_strat(v, g, depth, q=10)
    assert abs(a - 0.5) < 0.06


def test_channel_check_end_to_end(tmp_path):
    ad = pytest.importorskip("anndata")
    import scipy.sparse as sp
    m = _mod()
    rng = np.random.default_rng(1)
    n = 6000
    kind = rng.choice(["cd8", "cd4", "fib", "endo", "epi"], n, p=[0.1, 0.1, 0.25, 0.15, 0.4])
    genes = ["CD3E", "CD3D", "CD8A", "CD8B", "CD4", "ITGAV", "ENG", "PECAM1", "VWF", "COL1A1", "COL1A2", "DCN",
             "KRT8", "KRT18", "KRT19", "FASLG", "GAPDH"]
    gi = {g: j for j, g in enumerate(genes)}
    lam = np.full((n, len(genes)), 0.02)
    lam[:, gi["GAPDH"]] = 20
    for k, gs in {"cd8": ["CD3E", "CD3D", "CD8A", "CD8B"], "cd4": ["CD3E", "CD3D", "CD4"],
                  "fib": ["COL1A1", "COL1A2", "DCN", "ITGAV"], "endo": ["PECAM1", "VWF", "ENG", "ITGAV"],
                  "epi": ["KRT8", "KRT18", "KRT19"]}.items():
        for g in gs:
            lam[kind == k, gi[g]] = 3.0
    X = rng.poisson(lam).astype(np.float32)
    xy = np.c_[rng.uniform(0, 3000, n), rng.uniform(0, 3000, n)]
    cid = np.array([f"c{i}" for i in range(n)])
    A = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}, index=cid),
                   var=pd.DataFrame(index=genes))
    A.layers["nucleus"] = sp.csr_matrix(np.floor(X * 0.5))
    h5 = tmp_path / "h5"
    h5.mkdir()
    A.write_h5ad(h5 / "GSM1_lung_L1_L1.h5ad")
    af = rng.lognormal(5, 0.4, n)

    def stain(on):
        return 0.5 * af + rng.lognormal(3, 0.4, n) + np.where(on, rng.lognormal(6.5, 0.3, n), 0)
    reg = tmp_path / "reg"
    reg.mkdir()
    pd.DataFrame({"cell_id": cid, "dist_um": rng.uniform(0, 1.5, n),
                  "Opal_520_ring": stain(kind == "cd8"),                       # behaves as CD8
                  "Opal_620_ring": stain(np.isin(kind, ["fib", "endo"])),       # behaves as integrin alphaV
                  "Opal_690_ring": stain(kind == "epi"),
                  "Sample_AF_ring": af}).to_csv(reg / "matches.csv.gz", index=False, compression="gzip")
    (reg / "register.json").write_text(json.dumps({"sample": "GSM1_lung_L1_L1"}))
    out = tmp_path / "out"
    m.run(SimpleNamespace(matches=str(reg), h5ad=str(h5), out=str(out), image="", max_match_um=2.0, no_af=False,
                          nb_k=6, nb_r=15.0))
    T = pd.read_csv(out / "targets.csv").query("counts == 'whole'").set_index(["channel", "target"]).auc
    assert T["Opal 520", "CD8 (CD8A+CD8B)"] > 0.8 and T["Opal 520", "Integrin aV (ITGAV)"] < 0.5
    assert T["Opal 620", "Integrin aV (ITGAV)"] > 0.8 and T["Opal 620", "CD8 (CD8A+CD8B)"] < 0.5
    assert T["Opal 690", "panCK (keratins)"] > 0.8
    K = pd.read_csv(out / "tcells.csv").query("counts == 'whole'").set_index("channel")
    assert K.loc["Opal 520", "auc_cd8_vs_cd4"] > 0.9 and abs(K.loc["Opal 620", "auc_cd8_vs_cd4"] - 0.5) < 0.1
    C = pd.read_csv(out / "classes.csv").query("counts == 'whole'").set_index(["channel", "cell_class"])
    assert C.loc[("Opal 520", "CD8 T"), "positive_share"] > 0.8
    assert C.loc[("Opal 520", "fibroblast"), "positive_share"] < 0.05
    assert C.loc[("Opal 620", "endothelial"), "positive_share"] > 0.8
