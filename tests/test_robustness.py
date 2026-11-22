"""robustness options for the compartment test (nuclear thirds and the --agree filter, contact by
touching boundaries) and the fitted class prior."""
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

from claimbreak import compartment as cm
from claimbreak.cli import main

from test_compartment import _make


# --------------------------------------------------------------------------- nuclear thirds

def test_thirds_partition_the_nucleus_and_leave_other_arms_unchanged():
    rng = np.random.default_rng(0)
    Xn = sp.csr_matrix(rng.poisson(3, (300, 40)).astype(np.float32))
    Xc = sp.csr_matrix(rng.poisson(4, (300, 40)).astype(np.float32))
    Xw = (Xn + Xc).tocsr()
    base = cm.count_sets(Xw, Xn, Xc, ["nucA", "nucB", "wholeB", "cytoB"], seed=4)
    both = cm.count_sets(Xw, Xn, Xc, ["nucA3", "nucB3", "nucC3", "wholeB3", "cytoB3", "nucA", "nucB", "wholeB", "cytoB"],
                         seed=4)
    for k in base:          # thirds use their own random stream: existing arms are identical whatever the order
        assert (base[k] != both[k]).nnz == 0, k
    S = (both["nucA3"] + both["nucB3"] + both["nucC3"] - Xn).tocsr()
    assert abs(S).sum() == 0
    rest = (Xw - both["nucA3"] - both["nucC3"]).toarray()
    assert (both["wholeB3"].toarray() <= rest).all()                    # neither nucA3 nor nucC3 counts
    nB3 = np.asarray(both["nucB3"].sum(1)).ravel()
    w3 = np.asarray(both["wholeB3"].sum(1)).ravel()
    assert abs(np.median(w3 - nB3)) <= 2                                  # depth-matched to nucB3
    shares = [both[k].sum() / Xn.sum() for k in ("nucA3", "nucB3", "nucC3")]
    assert np.allclose(shares, 1 / 3, atol=0.02)


def test_label_with_thirds_records_agreement(tmp_path):
    d = _make(tmp_path)
    out = tmp_path / "lab3"
    assert main(["compartment", "label", "--ref", str(d / "ref.h5ad"), "--inputs", str(d / "sec0.h5ad"),
                 "--out", str(out), "--arms", "nucA3,nucB3,nucC3,wholeB3,cytoB3,nucleus", "--levels", "L2m",
                 "--ref-levels", "L2m=Level2"]) == 0
    m = json.loads((out / "sec0.meta.json").read_text())
    assert "L2m:nucC3==nucA3" in m["agreement"] and "L2m:wholeB3==nucA3" in m["agreement"]


# --------------------------------------------------------------------------- the correction confound and --agree

def _confound_table(n=6000, seed=1, err_nuc=0.35, err_test=0.08, contaminate=0.0):
    """Three types in spatial patches. Nuclear thirds A, B, C err at err_nuc (to a random other type); the test arm
    errs less (err_test, also at random) -- more informative counts, no contamination -- so it corrects A's errors
    more often than B does, and corrections land on the (clustered) true type. With contaminate > 0 the test also
    takes the type of a touching neighbor of another type with that probability."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1500, (n, 2))
    patch = (np.floor(xy[:, 0] / 60) + 3 * np.floor(xy[:, 1] / 60)).astype(int) % 3
    types = np.array(["Tumour", "Fibro", "T_NK"])
    true = np.where(rng.random(n) < 0.8, patch, rng.integers(0, 3, n))

    def noisy(e):
        lab = true.copy()
        w = rng.random(n) < e
        lab[w] = (true[w] + rng.integers(1, 3, w.sum())) % 3
        return lab
    A, B, C, T = noisy(err_nuc), noisy(err_nuc), noisy(err_nuc), noisy(err_test)
    if contaminate:
        _, nn = cKDTree(xy).query(xy, k=7)
        for i in np.flatnonzero(rng.random(n) < contaminate):
            other = [j for j in nn[i, 1:] if true[j] != true[i]]
            if other:
                T[i] = true[other[0]]
    D = pd.DataFrame({"sample": "S_U1", "cell_id": [f"c{i}" for i in range(n)], "x": xy[:, 0], "y": xy[:, 1],
                      "L1_nucleus": types[true]})
    for nm, v in (("nucA3", A), ("nucB3", B), ("nucC3", C), ("wholeB3", T)):
        D[f"L1_{nm}"] = types[v]
        D[f"L1_{nm}_margin"] = 1.0
    return D


def _ratios(D, agree=""):
    _, S = cm.compare_section(D, ["L1"], [("nucB3", "nucA3"), ("wholeB3", "nucA3")], agree=agree)
    S = pd.DataFrame(S).set_index("comparison")
    t, z = S.loc["wholeB3|nucA3"], S.loc["nucB3|nucA3"]
    o_t, e_t, n_t = t.obs_contact, t.exp_contact, t.changed
    o_0, e_0, n_0 = z.obs_contact, z.exp_contact, z.changed * t.cells / z.cells
    extra = (o_t * n_t - o_0 * n_0) / (e_t * n_t - e_0 * n_0) if n_t > n_0 else np.nan
    return t.ratio_contact, z.ratio_contact, extra, int(t.cells)


def test_agree_filter_removes_the_correction_channel():
    # the more accurate test arm changes FEWER labels than the noisy null here, so read the contact ratios directly
    gap, gap_a = [], []
    for seed in range(3):
        D = _confound_table(seed=seed, err_nuc=0.15, err_test=0.075)
        rt, rn, _, cells = _ratios(D)
        rt_a, rn_a, _, cells_a = _ratios(D, agree="nucC3")
        assert cells_a < 0.8 * cells
        gap.append(rt - rn)
        gap_a.append(rt_a - rn_a)
    # corrections of nucA land on the clustered true type: the test looks contact-directed without contamination
    assert np.mean(gap) > 0.08
    # on confirmed labels most of that channel is gone (it shrinks with the accuracy of the thirds)
    assert np.mean(gap_a) < 0.5 * np.mean(gap)


def test_agree_filter_keeps_a_real_contact_effect():
    for err in (0.15, 0.3):
        D = _confound_table(err_nuc=err, err_test=err / 2, contaminate=0.25)
        rt_a, rn_a, _, _ = _ratios(D, agree="nucC3")
        assert rt_a > 1.3 and rt_a - rn_a > 0.3


def test_agree_with_the_reference_arm_itself_changes_nothing():
    D = _confound_table(n=1500)
    a = pd.DataFrame(cm.compare_section(D, ["L1"], [("wholeB3", "nucA3")])[1])
    b = pd.DataFrame(cm.compare_section(D, ["L1"], [("wholeB3", "nucA3")], agree="nucA3")[1])
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(KeyError):
        cm.compare_section(D, ["L1"], [("wholeB3", "nucA3")], agree="nope")


# --------------------------------------------------------------------------- contact by touching boundaries

def _grid(root, side=40, seed=2):
    """Square cells of 10 um; the test arm takes the type of an edge-sharing neighbor; a kNN contact of 6 cells
    also counts diagonal and second-ring cells, boundary contact (tol 1 um) counts the 8 cells sharing an edge or a
    corner."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    x, y = gx.ravel() * 10.0 + 5, gy.ravel() * 10.0 + 5
    n = len(x)
    types = np.array(["Tumour", "Fibro", "T_NK"])
    true = rng.integers(0, 3, n)
    ids = np.array([f"c{i}" for i in range(n)])
    A, B = true.copy(), true.copy()
    for lab in (A, B):
        w = rng.random(n) < 0.05
        lab[w] = (lab[w] + 1) % 3
    T = true.copy()
    for i in np.flatnonzero(rng.random(n) < 0.2):
        nb = [j for j in (i - 1, i + 1, i - side, i + side) if 0 <= j < n and true[j] != true[i]]
        if nb:
            T[i] = true[nb[0]]
    D = pd.DataFrame({"sample": "S_U1", "cell_id": ids, "x": x, "y": y, "L1_nucleus": types[true],
                      "L1_nucA": types[A], "L1_nucB": types[B], "L1_wholeB": types[T],
                      "L1_nucA_margin": 1.0})
    D.to_csv(root / "S_U1.labels.csv.gz", index=False, compression="gzip")
    vx = np.column_stack([x - 5, x + 5, x + 5, x - 5]).ravel()
    vy = np.column_stack([y - 5, y - 5, y + 5, y + 5]).ravel()
    bd = root / "bnd"
    bd.mkdir()
    pd.DataFrame({"cell_id": np.repeat(ids, 4), "vertex_x": vx, "vertex_y": vy}).to_parquet(
        bd / "S_U1_cell_boundaries.parquet")
    return root


def test_compare_with_boundary_contact(tmp_path):
    pytest.importorskip("pyarrow")
    root = _grid(tmp_path / "lab")
    for c in ("knn", "boundary"):
        args = ["compartment", "compare", "--labels", str(root), "--out", str(tmp_path / c), "--levels", "L1",
                "--comparisons", "nucB|nucA,wholeB|nucA", "--contact", c]
        if c == "boundary":
            args += ["--boundaries", str(root / "bnd")]
        assert main(args) == 0
    E = pd.read_csv(tmp_path / "boundary" / "boundary_contacts.csv")
    assert E.touching_median.iloc[0] == 8 and E.no_contact_share.iloc[0] == 0
    st = json.loads((tmp_path / "boundary" / "settings.json").read_text())
    assert st["contact"] == "boundary" and st["tol"] == 1.0
    S = {c: pd.read_csv(tmp_path / c / "summary.csv").set_index("comparison") for c in ("knn", "boundary")}
    # every test change goes to an edge-sharing neighbor: touching boundaries catch all of them
    assert S["boundary"].loc["wholeB|nucA", "obs_contact"] > 0.97
    assert S["boundary"].loc["wholeB|nucA", "ratio_contact"] > 1.0
    # far-only neighbors exclude touching cells, so no change can be far-only and contact at once
    assert S["boundary"].loc["wholeB|nucA", "obs_far_only"] < S["boundary"].loc["wholeB|nucA", "obs_contact"]
    assert main(["compartment", "compare", "--labels", str(root), "--out", str(tmp_path / "x"), "--levels", "L1",
                 "--contact", "boundary"]) == 1                      # boundaries folder required


# --------------------------------------------------------------------------- fitted class prior

def test_uniform_prior_is_the_default_and_unchanged():
    rng = np.random.default_rng(3)
    mu = rng.dirichlet(np.ones(30), 4)
    X = sp.csr_matrix(np.vstack([rng.multinomial(25, mu[k]) for k in rng.integers(0, 4, 500)]).astype(np.float32))
    a = cm.classify(X, mu, cm.LabelConfig())
    b = cm.classify(X, mu, cm.LabelConfig(prior="uniform"))
    for u, v in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(u, v)
    with pytest.raises(ValueError):
        cm.classify(X, mu, cm.LabelConfig(prior="natural"))


def test_fitted_prior_recovers_skewed_composition():
    rng = np.random.default_rng(4)
    mu = rng.dirichlet(np.full(40, 0.6), 3)
    share = np.array([0.8, 0.15, 0.05])
    k = rng.choice(3, 3000, p=share)
    X = sp.csr_matrix(np.vstack([rng.multinomial(12, mu[t]) for t in k]).astype(np.float32))
    cfg = cm.LabelConfig(prior="fit", gamma_iter=0)
    lab_f = cm.classify(X, mu, cfg, fit_gamma=False)[0]
    pi = cm.classify.last_prior
    lab_u = cm.classify(X, mu, cm.LabelConfig(gamma_iter=0), fit_gamma=False)[0]
    ok = lab_f >= 0
    assert np.allclose(pi, share, atol=0.06)
    assert (lab_f[ok] == k[ok]).mean() >= (lab_u[ok] == k[ok]).mean()      # Bayes-optimal under the true shares
    assert (lab_f[ok] == 2).mean() < (lab_u[ok] == 2).mean()               # the rare class is called less often


def test_cli_prior_recorded(tmp_path):
    d = _make(tmp_path)
    out = tmp_path / "labp"
    assert main(["compartment", "label", "--ref", str(d / "ref.h5ad"), "--inputs", str(d / "sec0.h5ad"),
                 "--out", str(out), "--arms", "nucA,nucB,wholeB", "--levels", "L2m", "--ref-levels", "L2m=Level2",
                 "--prior", "fit", "--floor", "0.05"]) == 0
    m = json.loads((out / "sec0.meta.json").read_text())
    assert m["prior"] == "fit" and m["floor"] == 0.05
    assert abs(sum(m["prior_fitted"]["L2m_wholeB"].values()) - 1) < 1e-3
