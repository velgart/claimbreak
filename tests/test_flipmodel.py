import json

import numpy as np
import pandas as pd
import pytest

from claimbreak import flipmodel as fm
from claimbreak.perturb import uniform_confusion
from claimbreak.stress import stress_test
from claimbreak.tissue import take

from conftest import synthetic


@pytest.fixture(scope="module")
def small():
    return synthetic(n_sec=3, n=1500, seed=3)


def test_margin_second():
    S = np.array([[0.9, 0.5, 0.1], [0.2, 0.3, 0.8], [0.4, np.nan, 0.6]])
    m, s = fm.margin_second(S, np.array([0, 0, 2]))
    assert np.allclose(m, [0.4, -0.6, 0.2])
    assert list(s) == [1, 2, 0]


def test_neighbourhood_counts(small):
    nb = fm.Neighbourhood.build(small, k=20)
    assert (nb.contact.sum(1) == 6).all() and (nb.far.sum(1) == 14).all() and (nb.k20.sum(1) == 20).all()
    nb2 = fm.Neighbourhood.build(small, k=20, r_contact=0.0)
    assert nb2.contact.sum() == 0


def test_default_model_loads():
    m = fm.FlipModel.default()
    assert set(m.beta) == set(fm.ALL_FEATURES)
    assert m.beta["margin"] < 0 and m.beta["outside_nucleus_share"] > 0 and m.beta["b_contact"] > 0


def test_counts_shared_and_rate(small):
    rng = np.random.default_rng(0)
    C = uniform_confusion(small.types)
    n = fm.draw_counts(small.labels, 0.1, C, rng)
    assert np.trace(n) == 0
    assert abs(n.sum() / small.n - 0.1) < 0.02
    L1, i1 = fm.place(small.labels, n, small.types, np.random.default_rng(1))
    L2, i2 = fm.place(small.labels, n, small.types, np.random.default_rng(2))
    T = list(small.types)
    for L in (L1, L2):        # identical pair counts whatever the placement
        got = pd.crosstab(pd.Categorical(small.labels, T), pd.Categorical(L, T), dropna=False).to_numpy().copy()
        np.fill_diagonal(got, 0)
        assert (got == n).all()
    assert i1["shortfall"] == 0 and abs(i1["realised_eps"] - n.sum() / small.n) < 1e-9


def test_model_placement_goes_to_contact(small):
    rng = np.random.default_rng(4)
    nb = fm.Neighbourhood.build(small, k=20)
    codes = small.codes()[0]
    margin = rng.normal(size=small.n)
    second = (codes + 1) % len(small.types)
    D = fm.Design(nb, fm.cell_features(small, margin=margin), second, ["b_contact"])
    model = fm.FlipModel(beta={"b_contact": 3.0})
    n = fm.draw_counts(small.labels, 0.05, uniform_confusion(small.types), rng)
    _, ir = fm.place(small.labels, n, small.types, rng, design=D)
    _, im = fm.place(small.labels, n, small.types, rng, design=D, model=model)
    assert im["contact_share"] > ir["contact_share"] + 0.2


def test_fit_recovers_contact_and_margin(small):
    rng = np.random.default_rng(5)
    nb = fm.Neighbourhood.build(small, k=20)
    codes = small.codes()[0]
    T = len(small.types)
    margin = rng.normal(size=small.n)
    second = (codes + 1) % T
    # test labeling: a cell takes the type of a contact neighbor with odds rising with contacts, lower margin
    test = small.labels.copy()
    for i in range(small.n):
        others = [b for b in range(T) if b != codes[i] and nb.contact[i, b] > 0]
        if not others:
            continue
        b = max(others, key=lambda j: nb.contact[i, j])
        eta = -4 + 0.8 * nb.contact[i, b] - 1.0 * margin[i]
        if rng.random() < 1 / (1 + np.exp(-eta)):
            test[i] = small.types[b]
    D = fm.Design(nb, fm.cell_features(small, margin=margin), second, ["b_contact", "margin", "log_counts"])
    model, info = fm.fit(small, test, D, rng=rng)
    assert model.beta["b_contact"] > 0.3 and model.beta["margin"] < -0.3
    assert abs(model.beta["log_counts"]) < 0.2
    assert info["auc_leave_one_unit_out"] > 0.7


def test_missing_features_dropped(small):
    nb = fm.Neighbourhood.build(small, k=20)
    D = fm.Design(nb, fm.cell_features(small), None, list(fm.ALL_FEATURES))
    assert "outside_nucleus_share" not in D.features and "second_is_b" not in D.features
    assert "b_contact" in D.features and "log_counts" in D.features


def test_stress_test_runs(small):
    t = take(small, np.flatnonzero(small.section != "s2"))
    nuc = np.asarray(t.counts.sum(1)).ravel() * 0.5
    tab, info = stress_test(t, eps=(0.1,), reps=1, confusion="uniform", nucleus_counts=nuc, log=lambda *a: None)
    assert {"claim", "eps", "ref_effect", "delta_random", "delta_flipmodel", "kept_random", "kept_flipmodel",
            "reading"} <= set(tab.columns)
    assert "outside_nucleus_share" in info["features_used"]
    assert any(c.split("|")[0] == c.split("|")[1] for c in tab.claim)          # homotypic pairs included
    arms = {r["arm"] for r in info["realised_eps"]}
    assert arms == {"random", "flipmodel"}
    json.dumps(info, default=str)


def test_cli_stress(tmp_path, small):
    import anndata as ad
    import scipy.sparse as sp
    from claimbreak.cli import main
    t = take(small, np.flatnonzero(small.section != "s2"))
    A = ad.AnnData(X=sp.csr_matrix(t.counts),
                   obs=pd.DataFrame({"x": t.xy[:, 0], "y": t.xy[:, 1], "label": t.labels, "section": t.section,
                                     "cell_area": 100.0, "nucleus_area": 40.0},
                                    index=[f"c{i}" for i in range(t.n)]))
    A.layers["nucleus"] = sp.csr_matrix(t.counts.multiply(0.5).floor())
    f = tmp_path / "d.h5ad"
    A.write_h5ad(f)
    out = tmp_path / "out"
    rc = main(["stress", str(f), "--label-key", "label", "--section-key", "section", "--eps", "0.1", "--reps", "1",
               "--confusion", "uniform", "--nucleus-layer", "nucleus", "--cell-area-key", "cell_area",
               "--nucleus-area-key", "nucleus_area", "--out", str(out)])
    assert rc == 0
    assert (out / "claims.csv").exists() and (out / "report.md").exists()
    info = json.loads((out / "info.json").read_text())
    assert "nucleus_cell_ratio" in info["features_used"]
