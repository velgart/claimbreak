"""examples/compartment/region_loss.py: immune label loss by region (e.g. intraepithelial against lamina
propria T cells)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("region_loss", ROOT / "examples" / "compartment" / "region_loss.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _gut(root, name, seed):
    """an epithelial band (x < 300) with intraepithelial T cells and a lamina propria (x >= 300) with T cells; the
    whole-cell counts lose half of the intraepithelial T labels and a tenth of the others"""
    rng = np.random.default_rng(seed)
    n = 6000
    xy = rng.uniform(0, 800, (n, 2))
    epi = xy[:, 0] < 300
    lab = np.where(epi, np.where(rng.random(n) < 0.12, "T_NK", "Epithelial"),
                   rng.choice(["T_NK", "Fibroblast", "Myeloid"], n, p=[0.3, 0.4, 0.3]))
    whole = lab.copy()
    lose = (lab == "T_NK") & (rng.random(n) < np.where(epi, 0.5, 0.1))
    whole[lose] = np.where(epi[lose], "Epithelial", "Fibroblast")
    L = pd.DataFrame({"sample": name, "x": xy[:, 0], "y": xy[:, 1], "L1_nucA": lab, "L1_nucB": lab, "L1_wholeB": whole})
    L.to_csv(root / f"{name}.labels.csv.gz", index=False, compression="gzip")


def test_region_loss(tmp_path):
    m = _mod()
    lab = tmp_path / "lab"; lab.mkdir()
    _gut(lab, "I0301_withSegment_a", 1)
    _gut(lab, "I0302_noseg_a", 2)
    m.main(["--labels", str(lab), "--out", str(tmp_path / "o"), "--partner", "Epithelial", "--unit-pattern", r"I0\d{3}"])
    S = pd.read_csv(tmp_path / "o" / "summary.csv")
    a = S[S.masks == "all"].set_index("bin")
    assert a.loc["inside", "loss_pct"] == pytest.approx(50, abs=6)
    assert a.loc["away", "loss_pct"] == pytest.approx(10, abs=4)
    assert set(S.masks) == {"all", "stain", "expansion"}
    meta = json.loads((tmp_path / "o" / "meta.json").read_text())
    assert meta["paired_inside_minus_away"]["units_higher_inside"] == 2
    with pytest.raises(SystemExit):
        m.main(["--labels", str(lab), "--out", str(tmp_path / "o"), "--partner", "Epithelial"])


def test_region_share():
    m = _mod()
    xy = np.array([[0, 0], [1, 0], [2, 0], [100, 0]], float)
    part = np.array([False, True, True, False])
    r = m.region_share(xy, part, 5.0)
    assert r[0] == pytest.approx(1.0) and r[1] == pytest.approx(0.5) and np.isnan(r[3])


def test_identity_and_state_with_h5ad(tmp_path):
    """identity of the immune calls (areas, nuclear marker counts) and loss by state, from an h5ad"""
    ad = pytest.importorskip("anndata")
    import scipy.sparse as sp
    m = _mod()
    lab = tmp_path / "lab"; lab.mkdir()
    h5 = tmp_path / "h5"; h5.mkdir()
    _gut(lab, "I0301_withSegment_a", 3)
    L = pd.read_csv(lab / "I0301_withSegment_a.labels.csv.gz", keep_default_na=False)
    n = len(L)
    rng = np.random.default_rng(0)
    L["cell_id"] = [f"c{i}" for i in range(n)]
    L["n_nucleus"] = rng.poisson(40, n)
    t = (L.L1_nucA == "T_NK").to_numpy()
    # cytotoxic T cells (GZMB in the nucleus) never lose their label; the others lose it as before
    cyto = t & (rng.random(n) < 0.5)
    L.loc[cyto, "L1_wholeB"] = "T_NK"
    L.to_csv(lab / "I0301_withSegment_a.labels.csv.gz", index=False, compression="gzip")
    genes = ["CD3E", "EPCAM", "GZMB", "OTHER"]
    N = np.zeros((n, 4))
    N[:, 0] = np.where(t, 5, 0); N[:, 1] = np.where(L.L1_nucA == "Epithelial", 8, 0); N[:, 2] = np.where(cyto, 2, 0)
    N[:, 3] = 30
    A = ad.AnnData(X=sp.csr_matrix(N), obs=pd.DataFrame({"cell_area": np.where(t, 40.0, 120.0)},
                                                        index=L.cell_id.to_numpy()),
                   var=pd.DataFrame(index=genes))
    A.layers["nucleus"] = sp.csr_matrix(N)
    A.write_h5ad(h5 / "I0301_withSegment_a.h5ad")
    m.main(["--labels", str(lab), "--out", str(tmp_path / "o"), "--partner", "Epithelial", "--h5ad", str(h5),
            "--own-markers", "CD3E,CD3D", "--partner-markers", "EPCAM", "--state-genes", "cytotox=GZMB,PRF1",
            "--unit-pattern", r"I0\d{3}"])
    I = pd.read_csv(tmp_path / "o" / "identity.csv")
    med = I[I.unit == "median"].set_index("bin")
    assert med.loc["partner cells", "per1k_partner"] > med.loc["inside", "per1k_partner"]
    assert med.loc["inside", "cell_area"] == pytest.approx(40) and med.loc["partner cells", "cell_area"] == pytest.approx(120)
    St = pd.read_csv(tmp_path / "o" / "state.csv")
    s = St[(St.unit == "median") & (St.feature == "cytotox") & (St.bin == "all")].set_index("level")
    assert s.loc["positive", "loss_pct"] == pytest.approx(0, abs=1e-9) and s.loc["negative", "loss_pct"] > 10
    meta = json.loads((tmp_path / "o" / "meta.json").read_text())
    assert meta["gene_sets"]["own"] == ["CD3E"]
    Dp = pd.read_csv(tmp_path / "o" / "depth.csv")               
    d = Dp[(Dp.unit == "median") & (Dp.feature == "cytotox") & (Dp.bin == "all")].iloc[0]
    assert d.adjusted_diff < -10 and meta["depth_strata"] == 5


def test_depth_matched_state():
    """a state that only marks deeper nuclei shows a crude loss difference but none within depth strata; a
    state that changes the loss keeps its difference"""
    m = _mod()
    rng = np.random.default_rng(5)
    rows = []
    for u in ("u1", "u2", "u3"):
        n = 4000
        depth = rng.lognormal(3.5, 0.6, n)
        rank = pd.Series(depth).rank(pct=True).to_numpy()
        p_lose = 0.6 - 0.5 * rank                                 # deeper nuclei keep their label
        marker = rng.random(n) < np.clip(rank, 0.05, 0.95)        # positive mostly in deep nuclei; no own effect
        real = rng.random(n) < 0.5                                # a state that adds 20 points of loss
        p2 = np.clip(p_lose + 0.2 * real, 0, 1)
        lost = rng.random(n) < p2
        rows.append(pd.DataFrame({"unit": u, "bin": rng.choice(["away", "inside"], n), "n_nuc_h5": depth,
                                  "kept_B": True, "kept_W": ~lost, "cnt_state_marker": marker.astype(float),
                                  "cnt_state_real": real.astype(float)}))
    T = pd.concat(rows, ignore_index=True)
    D = m.depth_table(T, ["away", "inside"], k=10)
    med = D[(D.unit == "median") & (D.bin == "all")].set_index("feature")
    assert med.loc["marker", "crude_diff"] < -10                  # crude: positive cells look protected
    assert abs(med.loc["marker", "adjusted_diff"]) < 4            # matched on depth: no difference
    assert med.loc["real", "adjusted_diff"] == pytest.approx(20, abs=5)
    assert med.loc["real", "units_positive"] == 3
    assert abs(med.loc["inside_vs_away", "adjusted_diff"]) < 5    # the region was drawn at random here
