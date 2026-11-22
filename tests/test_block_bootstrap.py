"""Block bootstrap over tiles: the tile sums reproduce the pooled extra-change ratio, and when label changes
come in spatial patches the block interval is wider than the parametric one, which treats cells as independent."""
import numpy as np
import pandas as pd

from claimbreak import compartment as cm


def _section(sample, seed, patchy):
    rng = np.random.default_rng(seed)
    n_side = 90
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    xy = np.c_[gx.ravel(), gy.ravel()] * 10.0 + rng.uniform(-1, 1, (n_side * n_side, 2))
    n = len(xy)
    truth = np.where(rng.random(n) < 0.3, "T_NK", "Tumour")
    _, idx = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(xy).query(xy, k=7)
    het = (truth[idx[:, 1:]] != truth[:, None]).any(1)
    # loss of T/NK to tumor at contact; with patchy=True only inside some 200 um patches, strongly
    if patchy:
        cx, cy = (xy[:, 0] // 200).astype(int), (xy[:, 1] // 200).astype(int)
        hot = ((cx * 7 + cy * 3) % 5 == 0)
        p = np.where(hot, 0.8, 0.0)
    else:
        p = np.full(n, 0.16)
    lose = (truth == "T_NK") & het & (rng.random(n) < p)

    def noisy(lab, q=0.03):
        f = rng.random(n) < q
        return np.where(f, np.where(lab == "T_NK", "Tumour", "T_NK"), lab)
    D = pd.DataFrame({"sample": sample, "cell_id": [f"c{i}" for i in range(n)], "x": xy[:, 0], "y": xy[:, 1],
                      "L_nucleus": truth, "L_nucA": noisy(truth), "L_nucB": noisy(truth),
                      "L_wholeB": np.where(lose, "Tumour", noisy(truth))})
    D["L_nucA_margin"] = 5.0
    return D


def _run(tmp_path, patchy):
    lab = tmp_path / f"lab_{patchy}"
    lab.mkdir()
    for k, s in enumerate(("U1_a", "U1_b", "U2_a")):
        _section(s, 10 + k, patchy).to_csv(lab / f"{s}.labels.csv.gz", index=False, compression="gzip")
    cm.compare(lab, tmp_path / f"cmp_{patchy}", ["L"], "nucB|nucA,wholeB|nucA", verbose=False)
    return cm.summarise(tmp_path / f"cmp_{patchy}", tmp_path / f"sum_{patchy}", unit_pattern=r"U\d", draws=500,
                        block_draws=300, min_null_cells=100)


def test_tile_sums_reproduce_pooled_ratio(tmp_path):
    r = _run(tmp_path, False)
    U = r["units"]
    assert {"block_contact", "block_contact_lo", "block_contact_hi", "tiles"} <= set(U.columns)
    assert np.allclose(U.block_contact, U.pooled_contact, rtol=1e-6)
    K = r["units_kept"]
    assert set(K.type) == {"T_NK"} and (K.kept_ratio < 1).all() and (K.kept_ratio_lo <= K.kept_ratio).all()


def test_patchy_changes_widen_the_interval(tmp_path):
    """The same mean loss (0.16 at contact), uniform or in 200 um patches: the block interval of the T/NK kept ratio
    is much wider for patches; the extra-change ratio interval is at least as wide as the parametric one."""
    flat, patch = _run(tmp_path, False), _run(tmp_path, True)
    wk = lambda r: (r["units_kept"].kept_ratio_hi - r["units_kept"].kept_ratio_lo).to_numpy()
    assert (wk(patch) > 2 * wk(flat)).all(), (wk(flat), wk(patch))
    U = patch["units"]
    assert ((U.block_contact_hi - U.block_contact_lo) > (U.pooled_contact_hi - U.pooled_contact_lo)).all()
