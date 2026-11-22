"""claimbreak.placement: same excess label changes, placed differently; units held out."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from claimbreak import placement as pl
from claimbreak.cli import main

TYPES = ["Epithelial", "T_NK", "Myeloid", "Fibroblast"]


def _noisy(t, rate, rng):
    out = t.copy()
    m = rng.random(len(t)) < rate
    out[m] = rng.integers(0, len(TYPES), m.sum())
    return out


def _labels(root: Path, units=4, per_unit=2, n=2500, seed=1):
    """Sections in which whole-cell labels take the type of a touching cell when much of the cell's counts lie
    outside the nucleus (the mechanism of the accompanying paper), on top of independent counting noise."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for u in range(units):
        for s in range(per_unit):
            xy = rng.uniform(0, 600, (n, 2))
            epi = ((xy[:, 0] // 150) % 2 == 0)
            p = np.where(epi[:, None], [0.65, 0.2, 0.1, 0.05], [0.05, 0.3, 0.3, 0.35])
            truth = np.array([rng.choice(4, p=q) for q in p])
            N = _noisy(truth, 0.05, rng)
            B, A = _noisy(truth, 0.12, rng), _noisy(truth, 0.12, rng)
            out_share = rng.beta(2, 3, n)
            W = _noisy(truth, 0.12, rng)
            _, nn = cKDTree(xy).query(xy, k=4)
            nb = N[nn[:, 1]]
            flip = (nb != W) & (rng.random(n) < 0.6 / (1 + np.exp(-(out_share - 0.5) * 10)))
            W = np.where(flip, nb, W)
            margin = np.where(B == truth, rng.gamma(3, 1.5, n), rng.gamma(1, 0.7, n))
            second = np.where(B != truth, truth, (B + rng.integers(1, 4, n)) % 4)
            n_whole = rng.integers(60, 300, n)
            n_nuc = np.round(n_whole * (1 - out_share)).astype(int)
            name = f"GSM{u}{s}_I0{100 + u}_sec{s}"
            T = np.array(TYPES)
            D = pd.DataFrame({"sample": name, "cell_id": [f"c{i}" for i in range(n)], "x": xy[:, 0], "y": xy[:, 1],
                              "n_whole": n_whole, "n_nucleus": n_nuc, "cell_area": rng.uniform(40, 120, n),
                              "nucleus_area": rng.uniform(15, 40, n), "L1_nucleus": T[N], "L1_nucB": T[B],
                              "L1_nucA": T[A], "L1_wholeB": T[W], "L1_nucB_second": T[second],
                              "L1_nucB_margin": margin.round(3), "n_nucB": n_nuc // 2})
            D.loc[:5, "L1_nucA"] = ""                     # unlabeled cells are dropped
            D.to_csv(root / f"{name}.labels.csv.gz", index=False, compression="gzip")
    return root


@pytest.fixture(scope="module")
def labels(tmp_path_factory):
    return _labels(tmp_path_factory.mktemp("placement") / "labels")


def test_units_and_patterns():
    assert pl.unit_from("GSM1_I0284_x", r"I0\d{3}") == "I0284"
    assert pl.unit_from("bilous_5k_L3_L2", r"_(?P<unit>L\d)_L\d") == "L3"
    assert pl.unit_from("GSM9_Region01_AIN_02_D_cortex", r"(AIN|ATI|REF)_\d+_D") == "AIN_02_D"   # plain group ignored
    assert pl.unit_from("other", r"I0\d{3}", "D7") == "D7"


def test_placement_runs_and_model_placement_beats_random(labels, tmp_path):
    r = pl.placement(labels, tmp_path / "out", "L1", ["T_NK", "Myeloid"], ["Epithelial", "Fibroblast"],
                     draws=4, min_cells=50, save_models=True, verbose=False)
    out = tmp_path / "out"
    for f in ("coloc_sections.csv", "coloc_units.csv", "coloc_summary.csv", "sign_match.csv", "sign_match_units.csv",
              "flips.csv", "models.json", "meta.json", "flipmodel_full.json", "flipmodel_no_nucleus.json"):
        assert (out / f).exists(), f
    assert set(r["arms"]) >= {"ambiguity", "oracle", "random", "kernel", "kernel_count", "transfer_default",
                              "transfer_no_nucleus", "louo_full", "louo_no_nucleus"}
    assert r["units"].unit.nunique() == 4
    m = r["models"]
    assert m["louo"]["louo_full"]["auc_median"] > 0.6
    beta = pd.DataFrame({u: v["beta"] for u, v in m["louo"]["louo_full"]["by_unit"].items()}).median(axis=1)
    assert beta["outside_nucleus_share"] > 0 and beta["b_contact"] > 0
    f = r["flips"]
    assert (f.contact_share_louo_full > f.contact_share_random).all()
    assert (f.contact_share_oracle > f.contact_share_random).all()
    # changes placed by the fitted model move colocalization closer to the real change than random ones
    T = r["sections"]
    err = lambda a: (T[a] - T.real).abs().median()
    assert err("louo_full") < err("random") and err("oracle") < err("random")
    with pytest.raises(FileExistsError):
        pl.placement(labels, out, "L1", ["T_NK"], ["Epithelial"], verbose=False)


def test_section_filter_and_saved_model_transfer(labels, tmp_path):
    first = pl.placement(labels, tmp_path / "fit", "L1", ["T_NK"], ["Epithelial"], transfer=(), louo=("full",),
                         kernels=False, draws=2, min_cells=50, save_models=True, verbose=False)
    assert first["models"]["transfer"] == {}
    used = sorted(p.name.split(".labels")[0] for p in labels.glob("*.labels.csv.gz"))[:4]
    pd.DataFrame({"sample": used + ["skip_me"], "used": [True] * 4 + [False]}).to_csv(tmp_path / "s.csv", index=False)
    rc = main(["compartment", "placement", "--labels", str(labels), "--out", str(tmp_path / "cli"), "--immune", "T_NK",
               "--partners", "Epithelial", "--use-sections", str(tmp_path / "s.csv"), "--louo", "", "--transfer", "",
               "--model", f"synth={tmp_path / 'fit' / 'flipmodel_full.json'}", "--draws", "2", "--min-cells", "50",
               "--no-kernels"])
    assert rc == 0
    meta = json.loads((tmp_path / "cli" / "meta.json").read_text())
    assert sorted(s["name"] for s in meta["sections"]) == used
    S = pd.read_csv(tmp_path / "cli" / "coloc_sections.csv")
    assert "transfer_synth" in S and "louo_full" not in S and "kernel" not in S


def test_missing_type_is_reported(labels, tmp_path):
    with pytest.raises(ValueError, match="not in the labels"):
        pl.placement(labels, tmp_path / "x", "L1", ["Plasma"], ["Epithelial"], verbose=False)
