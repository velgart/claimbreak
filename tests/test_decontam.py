"""claimbreak.decontam: neighbor-aware labeling on a synthetic tissue with known truth.

Small T cells (little RNA) sit in tumor (much RNA); the counts outside each nucleus hold, besides the cell's own RNA,
RNA of its touching neighbors in proportion to the border they hold and to their RNA. The plain classifier on the
outside counts loses T cells next to tumor; the neighbor-aware arms must recover them without losing tumor cells."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from claimbreak import compartment as cm
from claimbreak import decontam as dc
from claimbreak import geometry as geo

TYPES = np.array(["Myeloid", "T_NK", "Tumour"])


def _tissue(seed=5, n_side=60, catch=0.7, alpha=0.25, scale=1.0, mismatch=0.0):
    rng = np.random.default_rng(seed)
    G = 80
    mu = rng.dirichlet(np.full(G, alpha), size=3)
    # cells' true profiles differ from the reference: a gene factor (platform) and per-cell noise around it
    mu_c = mu * np.exp(rng.normal(0, mismatch, G))[None, :] if mismatch else mu
    mu_c = mu_c / mu_c.sum(1, keepdims=True)
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    xy = np.c_[gx.ravel(), gy.ravel()] * 7.5 + rng.uniform(-0.4, 0.4, (n_side * n_side, 2))
    n = len(xy)
    t = np.where(rng.random(n) < 0.25, 1, 2)
    t[rng.random(n) < 0.08] = 0
    for c in rng.uniform(0, n_side * 7.5, (12, 2)):                  # T aggregates
        t[np.linalg.norm(xy - c, axis=1) < 20] = 1
    area = np.where(t == 1, np.pi * 2.6 ** 2, np.pi * 3.4 ** 2)
    rna = scale * np.where(t == 2, 300, np.where(t == 1, 80, 200)).astype(float)
    r = np.sqrt(area / np.pi)
    i, j, gap, d = geo.gap_pairs(xy, r, 10.0)
    ip, jp, sh, cov = geo.border_shares(n, xy, r, i, j, gap, d, 10.0)
    src = sp.csr_matrix((sh * rna[jp], (ip, t[jp])), shape=(n, 3)).toarray()   # neighbors' RNA at the border, by type
    src = src / np.maximum(src.sum(1, keepdims=True), 1e-12)
    # foreign share of the outside counts: catchment grows with the neighbors' RNA relative to the cell's own
    rel = (sp.csr_matrix((sh * rna[jp], (ip, np.zeros(len(ip), int))), shape=(n, 1)).toarray().ravel()) / rna
    psi = catch * rel / (1 + rel)
    Xn, Xo = [], []
    for k in range(n):
        nn = rng.poisson(0.3 * rna[k])
        no = rng.poisson(0.7 * rna[k] / max(1 - psi[k], 0.05))
        p_own = mu_c[t[k]]
        if mismatch:
            p_own = p_own * np.exp(rng.normal(0, mismatch, G)); p_own = p_own / p_own.sum()
        Xn.append(rng.multinomial(nn, p_own))
        own = rng.binomial(no, 1 - psi[k])
        Xo.append(rng.multinomial(own, p_own) + rng.multinomial(no - own, src[k] @ mu_c if src[k].sum() else p_own))
    Xn, Xo = sp.csr_matrix(np.array(Xn, np.float32)), sp.csr_matrix(np.array(Xo, np.float32))
    genes = [f"g{x}" for x in range(G)]
    A = ad.AnnData(X=(Xn + Xo).tocsr(), layers={"nucleus": Xn, "cyto": Xo},
                   obs=pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "n_whole": np.asarray((Xn + Xo).sum(1)).ravel(),
                                     "n_nucleus": np.asarray(Xn.sum(1)).ravel(), "nucleus_area": area,
                                     "panel": "P"}, index=[f"c{x}" for x in range(n)]),
                   var=pd.DataFrame(index=genes))
    ref = cm.Reference(sp.csr_matrix(np.ones((3, G))), genes, {"L": TYPES}, np.array(list("abc")))
    ref._mu[("L", tuple(range(G)))] = (TYPES, mu)
    het = geo.heterotypic(n, ip, jp, sh, t, t)[0]
    return A, ref, TYPES[t], het, psi


@pytest.fixture(scope="module")
def labelled():
    A, ref, truth, het, psi = _tissue()
    res, gam, info = cm.label_section(A, ref, ["L"], ["whole", "nucleus", "cyto", "nucA", "nucB", "cytoB",
                                                   "wholeM", "cytoM", "wholeMX"], cm.LabelConfig(gamma_iter=0))
    return res, truth, het, psi, info


def _acc(res, arm, truth, m):
    lab = res[f"L_{arm}"].to_numpy()
    ok = m & (lab != "")
    return np.mean(lab[ok] == truth[ok])


def test_columns_and_psi(labelled):
    res, truth, het, psi, info = labelled
    for arm in ("wholeM", "cytoM", "wholeMX"):
        assert {f"L_{arm}", f"L_{arm}_second", f"L_{arm}_margin", f"L_{arm}_psi"} <= set(res.columns)
    assert "border_covered" in res and "L:cytoM==nucA" in info["agreement"]
    # estimated foreign share rises with the true one (correctly labeled T cells: tumor into tumor is not identifiable)
    ok = (truth == "T_NK") & (res["L_cytoM"].to_numpy() == truth) & np.isfinite(res["L_cytoM_psi"].to_numpy(float))
    assert np.corrcoef(res["L_cytoM_psi"].to_numpy(float)[ok], psi[ok])[0, 1] > 0.4


def test_t_cells_next_to_tumour_recovered(labelled):
    res, truth, het, psi, _ = labelled
    tb = (truth == "T_NK") & (het > 0.6)
    plain_c, dec_c = _acc(res, "cytoB", truth, tb), _acc(res, "cytoM", truth, tb)
    plain_w, dec_w, dec_x = _acc(res, "whole", truth, tb), _acc(res, "wholeM", truth, tb), _acc(res, "wholeMX", truth, tb)
    assert plain_c < 0.6 and dec_c > plain_c + 0.25
    assert dec_w > plain_w + 0.1 and dec_w > 0.85
    assert dec_x > plain_w + 0.1


def test_no_loss_elsewhere(labelled):
    res, truth, het, psi, _ = labelled
    for m in (truth == "Tumour", (truth == "T_NK") & (het == 0), truth == "Myeloid"):
        assert _acc(res, "wholeM", truth, m) >= _acc(res, "whole", truth, m) - 0.03
        assert _acc(res, "cytoM", truth, m) >= _acc(res, "cytoB", truth, m) - 0.05
    assert _acc(res, "wholeM", truth, np.ones(len(truth), bool)) > 0.93


def test_weights_follow_border():
    xy = np.array([[0, 0], [7, 0], [-7, 0], [200, 200]], float)
    W, E, cov, _ = dc.contamination_weights(xy, np.full(4, np.pi * 9), np.array([0, 1, 1, 0]),
                                            np.array([50, 300, 300, 50]), 2, dc.DecontamConfig(w_region=0, w_ambient=0.01))
    assert W[0, 1] > 0.95 and np.allclose(W.sum(1), 1)
    assert W[3, 1] == pytest.approx(600 / 700, abs=0.01)      # isolated cell: ambient composition only
    assert E[0] > 10 * E[1] and E[0] > 10 * E[3]               # small cell between two large ones: most exposed
    assert np.all(np.diff(dc.psi_of(np.array([0, 0.5, 1, 5, 50]), 1.0, 0.9)) >= 0) and dc.psi_of(np.array([50.0]), 1.0, 0.9)[0] == 0.9


def test_m_arms_need_multinomial(labelled):
    A, ref, *_ = _tissue(n_side=10)
    with pytest.raises(ValueError):
        cm.label_section(A, ref, ["L"], ["cytoM"], cm.LabelConfig(labeller="celltypist"))


def test_fitted_psi_beats_free_psi_at_low_depth():
    """A free foreign share per candidate lets a wrong type absorb the cell's own counts as 'contamination'; at low
    depth that costs tumor cells and T cells inside T aggregates. psi fixed by exposure (one constant) does not."""
    A, ref, truth, het, _ = _tissue(alpha=1.0, scale=0.2)
    arms = ["whole", "nucleus", "cytoB", "wholeM", "cytoM"]
    fit, _, info = cm.label_section(A, ref, ["L"], arms, cm.LabelConfig(gamma_iter=0, min_counts=5))
    free, _, _ = cm.label_section(A, ref, ["L"], arms, cm.LabelConfig(gamma_iter=0, min_counts=5, decon_psi_mode="profile"))
    assert 0 < info["decon_kappa"]["L_wholeM"] < dc.KAPPAS[-1]
    tum, tb = truth == "Tumour", (truth == "T_NK") & (het > 0.6)
    assert _acc(fit, "cytoM", truth, tum) > _acc(free, "cytoM", truth, tum) + 0.1
    assert _acc(fit, "wholeM", truth, tb) > _acc(fit, "whole", truth, tb) + 0.15
    assert _acc(fit, "wholeM", truth, tum) >= _acc(fit, "whole", truth, tum) - 0.01


def test_fit_options_and_outside_index():
    s = dc.outside_index([0.2, 0.5, 0.8], 1.0)
    assert s[1] == pytest.approx(1.0) and s[0] < 1 < s[2]
    assert np.allclose(dc.outside_index([0.2, 0.5, 0.8], 0.0), 1)
    A, ref, truth, het, _ = _tissue(n_side=30, mismatch=0.5)
    for on in ("max", "auto", "nucB"):
        res, _, info = cm.label_section(A, ref, ["L"], ["nucleus", "nucA", "nucB", "cytoB", "wholeM", "cytoM"],
                                        cm.LabelConfig(gamma_iter=2, decon_fit_on=on))
        assert info["decon_fit_on"]["L_cytoM"] == ("nucB" if on in ("auto", "nucB") else "max")
        assert info["decon_beta"]["L_cytoM"] in (0.0, 0.5, 1.0, 2.0)
        assert _acc(res, "wholeM", truth, truth == "T_NK") > 0.9
    with pytest.raises(ValueError):
        cm.label_section(A, ref, ["L"], ["cytoM"], cm.LabelConfig(decon_fit_on="wholeB"))


def test_wballm_fair_arm():
    A, ref, truth, het, _ = _tissue(alpha=1.0, scale=0.4, mismatch=0.5)
    res, _, info = cm.label_section(A, ref, ["L"], ["nucA", "nucB", "nucleus", "wBall", "wBallM"],
                                    cm.LabelConfig(gamma_iter=2, min_counts=5))
    assert "L:wBallM==nucA" in info["agreement"]
    assert (res["n_wBallM"] == res["n_wBall"]).all()            # the same counts as the plain arm
    tb = (truth == "T_NK") & (het > 0.6)
    assert _acc(res, "wBallM", truth, tb) > _acc(res, "wBall", truth, tb) + 0.05
