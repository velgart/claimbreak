"""examples/liver: Census liver reference cell-type mapping, and piece QC / FOV layout of CosMx slides."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / "liver" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_liver_coarse_mapping():
    m = _mod("liver_reference")
    want = {"hepatocyte": "Hepatocyte", "periportal region hepatocyte": "Hepatocyte", "Kupffer cell": "Myeloid",
            "endothelial cell of hepatic sinusoid": "Endothelial", "hepatic stellate cell": "Stellate_Fib",
            "intrahepatic cholangiocyte": "Cholangiocyte", "hepatic pit cell": "T_NK", "plasma cell": "Plasma",
            "vascular associated smooth muscle cell": "Stellate_Fib", "mast cell": "", "epithelial cell": "",
            "hematopoietic stem cell": "", "CD8-positive, alpha-beta T cell": "T_NK", "neutrophil": "Myeloid"}
    assert {k: m.coarse(k) for k in want} == want


def _slide(root, slide, blocks, seed=0):
    """metadata_file with FOV grids at the given offsets (mm); 20 cells per FOV."""
    rng = np.random.default_rng(seed)
    rows, fov = [], 0
    step_px = 0.51 / 0.12028 * 1000                     # 0.51 mm FOVs
    for (ox, oy, nx, ny) in blocks:
        for i in range(nx):
            for j in range(ny):
                fov += 1
                cx, cy = ox / 0.12028 * 1000 + i * step_px, oy / 0.12028 * 1000 + j * step_px
                for _ in range(20):
                    rows.append(dict(fov=fov, CenterX_global_px=cx + rng.uniform(-100, 100),
                                     CenterY_global_px=cy + rng.uniform(-100, 100), Area=5000,
                                     **{"Area.um2": 5000 * 0.12028 ** 2, "Mean.DAPI": rng.uniform(20, 60)}))
    pd.DataFrame(rows).to_csv(root / f"{slide}_metadata_file.csv.gz", index=False)
    return pd.DataFrame(rows)


def test_layout_separates_biopsies(tmp_path, capsys):
    m = _mod("liver_pieces")
    # two biopsies 0.7 mm apart (FOVs 0.51 mm) (touching FOV grids would merge at gap 1.6 but not at 1.1), one far away
    _slide(tmp_path, "S1", [(0, 0, 3, 4), (1.02 + 0.70, 0, 3, 4), (10, 10, 2, 2)])
    m.cmd_layout(SimpleNamespace(csv=str(tmp_path), h5ad="", out=str(tmp_path / "o"), gaps="1.1,1.6,2.5",
                                 fov_gap=1.6, min_share=0.25, px_um=0.12028))
    L = pd.read_csv(tmp_path / "o" / "fov_layout.csv")
    assert L["piece_gap1.1"].nunique() == 3 and L["piece_gap1.6"].nunique() == 2
    assert len(L) == 3 * 4 * 2 + 4 and (L.cells == 20).all()
    assert (tmp_path / "o" / "fov_layout.png").exists() or pytest.importorskip("matplotlib") is None


def test_qc_flags_pieces_without_nuclear_counts(tmp_path):
    ad = pytest.importorskip("anndata")
    m = _mod("liver_pieces")
    rng = np.random.default_rng(1)
    for name, nshare in (("S1_p0", 0.4), ("S1_p1", 0.02)):
        n = 3000
        W = rng.poisson(100, n)
        N = rng.binomial(W, nshare)
        obs = pd.DataFrame({"fov": rng.integers(1, 10, n), "n_whole": W, "n_nucleus": N, "x": rng.uniform(0, 5000, n),
                            "y": rng.uniform(0, 5000, n), "Mean.DAPI": rng.uniform(10, 50, n)},
                           index=[f"{name}_{i}" for i in range(n)])
        ad.AnnData(X=np.zeros((n, 2), np.float32), obs=obs).write_h5ad(tmp_path / f"{name}.h5ad")
    m.cmd_qc(SimpleNamespace(h5ad=str(tmp_path / "*.h5ad"), out=str(tmp_path / "o"), min_share=0.25, min_nuc=10))
    T = pd.read_csv(tmp_path / "o" / "pieces_qc.csv").set_index("piece")
    assert T.loc["S1_p0", "usable"] and not T.loc["S1_p1", "usable"]
    assert abs(T.loc["S1_p0", "nuclear_share"] - 0.4) < 0.01 and T.loc["S1_p0", "slide"] == "S1"


def test_filter_drops_fovs_without_nuclear_counts(tmp_path):
    ad = pytest.importorskip("anndata")
    m = _mod("liver_pieces")
    rng = np.random.default_rng(2)
    n = 2000
    fov = rng.integers(1, 6, n)
    W = rng.poisson(100, n)
    N = np.where(fov == 3, 0, rng.binomial(W, 0.4))              # FOV 3: nucleus calls failed
    obs = pd.DataFrame({"fov": fov, "n_whole": W, "n_nucleus": N, "x": 0.0, "y": 0.0}, index=[f"c{i}" for i in range(n)])
    (tmp_path / "h").mkdir()
    ad.AnnData(X=np.zeros((n, 2), np.float32), obs=obs).write_h5ad(tmp_path / "h" / "S_p0.h5ad")
    pd.DataFrame({"piece": ["S_p0", "S_p1"], "usable": [True, False]}).to_csv(tmp_path / "qc.csv", index=False)
    m.cmd_filter(SimpleNamespace(h5ad=str(tmp_path / "h"), qc=str(tmp_path / "qc.csv"), out_h5ad=str(tmp_path / "q"),
                                 min_share=0.25))
    B = ad.read_h5ad(tmp_path / "q" / "S_p0.h5ad")
    assert B.n_obs == (fov != 3).sum() and not (B.obs.fov == 3).any()
    F = pd.read_csv(tmp_path / "q" / "filter_fovs.csv")
    assert F.set_index("fov").loc[3, "kept"] == False and F.kept.sum() == 4


def test_marker_call_is_reference_free():
    """canonical-marker call relative to the tissue average."""
    ad = pytest.importorskip("anndata")
    m = _mod("liver_label_check")
    rng = np.random.default_rng(3)
    genes = ["ALB", "APOA1", "PECAM1", "VWF", "CD68", "C1QA", "CD3E", "NKG7"] + [f"g{i}" for i in range(20)]
    truth = rng.choice(["Hepatocyte", "Endothelial", "Myeloid", "T_NK"], 3000, p=[0.6, 0.2, 0.15, 0.05])
    hot = {"Hepatocyte": [0, 1], "Endothelial": [2, 3], "Myeloid": [4, 5], "T_NK": [6, 7]}
    X = rng.poisson(1.0, (3000, len(genes))).astype(np.float32)
    for i, t in enumerate(truth):
        X[i, hot[t]] += rng.poisson(15, 2)
        X[i, [0, 1]] += rng.poisson(3, 2)                    # hepatocyte RNA everywhere (spill-over)
    call, used = m.marker_call(ad.AnnData(X=X, var=pd.DataFrame(index=genes)))
    assert set(used) == {"Hepatocyte", "Endothelial", "Myeloid", "T_NK"}
    ok = call != ""
    assert ok.mean() > 0.95 and (call[ok] == truth[ok]).mean() > 0.9
