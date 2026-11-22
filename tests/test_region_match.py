"""regionally matched expectation of the extra-change ratio, safer units, count clipping, coloc self-count,
and the ground-truth simulation of the kept ratio."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

from claimbreak import compartment as cm


def test_regional_shares_match_brute_force():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 600, (3000, 2))
    code = rng.integers(0, 3, len(xy))
    code[:50] = -1                                         # unlabeled cells are not counted
    share, tot = cm.regional_shares(xy, code, 3, (30.0, 100.0), px=1.0)
    tree = cKDTree(xy)
    for i in rng.choice(len(xy), 40, replace=False):
        d = np.hypot(*(xy - xy[i]).T)
        m = (d > 30) & (d <= 100) & (code >= 0)
        assert abs(tot[i] - m.sum()) <= max(3, 0.05 * m.sum())
        if m.sum() > 20:
            ref = np.bincount(code[m], minlength=3) / m.sum()
            assert np.abs(share[i] - ref).max() < 0.05


def _section(seed, mech, n=30000, L=2200.0, K=5):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, L, (n, 2))
    centres = rng.uniform(0, L, (300, 2))
    truth = rng.integers(0, K, len(centres))[cKDTree(centres).query(xy)[1]]
    truth = np.where(rng.random(n) < 0.3, rng.integers(0, K, n), truth)
    comp = np.bincount(truth, minlength=K) / n
    tree = cKDTree(xy)
    ring, near = tree.query_ball_point(xy, 100.0), tree.query_ball_point(xy, 30.0)
    d, nn = tree.query(xy, k=7)
    d, nn = d[:, 1:], nn[:, 1:]

    def relabel(p, how):
        ch = rng.random(n) < p
        new = rng.choice(K, n, p=comp)
        for i in np.flatnonzero(ch):
            if how == "regional":
                pool = np.setdiff1d(ring[i], near[i])
            elif how == "contact":
                pool = nn[i][d[i] <= 30]
            else:
                continue
            if len(pool):
                new[i] = truth[rng.choice(pool)]
        return np.where(ch & (new != truth), new, truth)
    names = np.array([f"t{k}" for k in range(K)])
    return pd.DataFrame({"sample": "sim", "x": xy[:, 0], "y": xy[:, 1], "L1_nucleus": names[truth],
                         "L1_nucA": names[truth], "L1_nucB": names[relabel(0.10, "random")],
                         "L1_wholeB": names[relabel(0.20, mech)], "L1_nucA_margin": 1.0})


def _ratios(D, **kw):
    _, S = cm.compare_section(D, ["L1"], [("nucB", "nucA"), ("wholeB", "nucA")], **kw)
    S = pd.DataFrame(S).set_index("comparison")
    t, z = S.loc["wholeB|nucA"], S.loc["nucB|nucA"]
    er = lambda o, e: cm.extra_ratio(t[o], t[e], t.change_rate, z[o], z[e], z.change_rate)
    return er("obs_contact", "exp_contact"), er("obs_contact_reg", "exp_contact_reg")


def test_matched_expectation_separates_regional_from_contact_changes():
    plain, matched = _ratios(_section(0, "regional"))
    assert plain > 1.5                                     # the plain ratio cannot tell region from contact
    assert 0.9 < matched < 1.15
    plain_c, matched_c = _ratios(_section(1, "contact"))
    assert matched_c > 2.5 and plain_c > matched_c


def test_region_off_keeps_old_columns():
    _, S = cm.compare_section(_section(2, "regional", n=5000, L=900.0), ["L1"], [("nucB", "nucA")], region=None)
    assert "exp_contact_reg" not in S[0] and "exp_contact" in S[0]


def test_summary_reports_matched_ratio(tmp_path):
    D = _section(3, "regional")
    D.to_csv(tmp_path / "s1.labels.csv.gz", index=False, compression="gzip")
    cm.compare(tmp_path, tmp_path / "cmp", ["L1"], "nucB|nucA,wholeB|nucA", verbose=False)
    r = cm.summarise(tmp_path / "cmp", tmp_path / "sum", test="wholeB|nucA", null="nucB|nucA", level="L1",
                     min_null_cells=100, draws=200, block_draws=100)
    U = r["units"]
    assert {"pooled_contact_reg", "block_contact_reg", "block_contact_reg_lo"} <= set(U.columns)
    assert U.pooled_contact.iloc[0] > 1.4 and abs(U.pooled_contact_reg.iloc[0] - 1) < 0.15
    assert "extra_ratio_contact_reg" in r["sections"]


def test_unit_default_is_the_section():
    assert cm.unit_of("GSM1_I0355_a") == "GSM1_I0355_a"
    assert cm.unit_of("GSM1_I0355_a", r"I0\d{3}") == "I0355"
    assert "own unit" in cm.unit_note(["a", "b"])
    assert "does not match 1 of 2" in cm.unit_note(["x_I0355", "other"], r"I0\d{3}")
    assert cm.unit_note(["x_I0355"], r"I0\d{3}") == ""


def test_count_sets_clip_a_nucleus_that_is_not_a_subset():
    rng = np.random.default_rng(0)
    Xn = sp.csr_matrix(rng.poisson(1.0, (200, 30)).astype(np.float32))
    Xw = sp.csr_matrix(np.maximum(Xn.toarray() - 1, 0) + rng.poisson(1.0, (200, 30)))   # not a superset
    out = cm.count_sets(Xw, Xn, None, ["nucA", "nucB", "wholeB", "wBall", "wB50", "wholeB3"], seed=1)
    for k in ("wholeB", "wBall", "wB50", "wholeB3"):
        assert out[k].nnz == 0 or out[k].data.min() >= 0


def test_coloc_random_labels_score_zero_for_same_type():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, (20000, 2))
    lab = rng.integers(0, 3, len(xy))
    _, idx = cKDTree(xy).query(xy, k=11)
    M, _ = cm.coloc(lab, idx[:, 1:], 3)
    assert np.abs(np.diag(M)).max() < 0.02


def _load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kept_ratio_recovers_planted_loss_and_flags_false_positives():
    sim = _load(Path(__file__).parents[1] / "examples" / "compartment" / "kept_ratio_sim.py")
    clean = pd.DataFrame(sim.run_one(0, 0.6, 0.15, False, 30000, 150, 0.15)).set_index("type")
    assert clean.loc["T", "true_recall_ratio"] < 0.85                # the planted contamination costs T labels
    assert abs(clean.loc["T", "difference"]) < 0.04                  # and the kept ratio recovers it
    fp = pd.DataFrame(sim.run_one(0, 0.0, 0.0, True, 30000, 150, 0.15)).set_index("type")
    assert fp.loc["T", "kept_ratio"] < fp.loc["T", "true_recall_ratio"] - 0.05   # systematic nuclear false calls


def test_regional_sim_script_separates_mechanisms():
    """examples/compartment/regional_sim.py, small sections: regional changes score about one matched,
    contact-driven changes stay well above one."""
    sim = _load(Path(__file__).parents[1] / "examples" / "compartment" / "regional_sim.py")
    p_reg, m_reg = sim.score(sim.section(0, "regional", n=20000, side=1700.0))
    p_con, m_con = sim.score(sim.section(1, "contact", n=20000, side=1700.0))
    assert p_reg > 1.4 and abs(m_reg - 1) < 0.15
    assert m_con > 2.0
