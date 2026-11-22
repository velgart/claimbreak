"""claimbreak.contacts: contact-scale infiltration and the Keren mixing score, reference against analyst labels."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from claimbreak import contacts as ct
from claimbreak.cli import main


def _grid_sections(root, units=3, side=60, seed=5):
    """Square cells on a 10 um grid (boundaries = squares). Tumor fills the left half, T cells are scattered over
    the whole section (well mixed in the reference); the analyst's labels turn T cells that touch tumor into
    tumor, as whole-cell counts do."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    bdir = root / "bnd"
    bdir.mkdir()
    for u in range(units):
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        x, y = gx.ravel() * 10.0 + 5, gy.ravel() * 10.0 + 5
        n = len(x)
        left = x < side * 5
        ref = np.where(left, "Tumour", "Fibroblast").astype(object)
        tcell = rng.random(n) < 0.25
        ref[tcell] = "T_NK"
        ref = ref.astype(str)
        _, nn = cKDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=5)
        touch = (ref[nn[:, 1:]] == "Tumour").any(1)
        test = ref.copy()
        test[(ref == "T_NK") & touch & (rng.random(n) < 0.6)] = "Tumour"
        name = f"GSM{u}_I0{300 + u}"
        ids = [f"c{u}_{i}" for i in range(n)]
        pd.DataFrame({"sample": name, "cell_id": ids, "x": x, "y": y, "L1_nucleus": ref, "L1_whole": test}).to_csv(
            root / f"{name}.labels.csv.gz", index=False, compression="gzip")
        vx = np.column_stack([x - 5, x + 5, x + 5, x - 5]).ravel()
        vy = np.column_stack([y - 5, y - 5, y + 5, y + 5]).ravel()
        try:
            pd.DataFrame({"cell_id": np.repeat(ids, 4), "vertex_x": vx, "vertex_y": vy}).to_parquet(
                bdir / f"{name}_cell_boundaries.parquet")
        except ImportError:
            pass
    return root


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    return _grid_sections(tmp_path_factory.mktemp("contacts") / "labels")


def test_edges():
    xy = np.array([[0, 0], [10, 0], [20, 0], [100, 0]], float)
    e = ct.knn_edges(xy, k=1, r=15)
    assert {tuple(x) for x in e} == {(0, 1), (1, 2)}
    bnd = pd.DataFrame({"cell_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                        "vertex_x": [0, 10, 10, 0, 10, 20, 20, 10, 50, 60, 60, 50],
                        "vertex_y": [0, 0, 10, 10] * 3})
    e = ct.boundary_edges(bnd, ["a", "b", "c"], tol=0.5)
    assert {tuple(x) for x in e} == {(0, 1)}
    # long shared edge with vertices that do not coincide: a's right edge has vertices only at its corners,
    # b's left edge is shifted by 0.3 um and has a vertex in the middle
    bnd2 = pd.DataFrame({"cell_id": ["a"] * 4 + ["b"] * 5,
                         "vertex_x": [0, 40, 40, 0, 40.3, 80, 80, 40.3, 40.3],
                         "vertex_y": [0, 0, 40, 40, 3, 3, 37, 37, 20]})
    assert {tuple(x) for x in ct.boundary_edges(bnd2, ["a", "b"], tol=1.0)} == {(0, 1)}
    assert len(ct.boundary_edges(bnd2, ["a", "b"], tol=0.2)) == 0
    assert ct.keren_class(0.5, 100) == "cold" and ct.keren_class(0.1, 300) == "compartmentalized"
    assert ct.keren_class(0.3, 300) == "mixed"


@pytest.mark.parametrize("contact", ["knn", pytest.param("boundary", marks=pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None, reason="pyarrow needed for parquet"))])
def test_infiltration_falls_and_random_raises_mixing(grid, tmp_path, contact):
    S, U, C = ct.contacts(grid, tmp_path / contact, "L1", ["T_NK"], ["Tumour"], contact=contact,
                          boundaries=grid / "bnd" if contact == "boundary" else None, k=8, r=15.0, tol=0.5,
                          eps=(0.05, 0.15), reps=3, verbose=False)
    cmp_ = ct.compare_units(U).set_index("readout")
    assert cmp_.loc["infil|T_NK|Tumour", "test_below_ref"] == 3
    assert cmp_.loc["per100|T_NK|Tumour", "test_below_ref"] == 3
    assert cmp_.loc["mixing", "test_below_ref"] == 3
    assert cmp_.loc["mixing", "random_15_above_test"] == 3
    w = U.pivot_table(index="unit", columns="labels", values="mixing")
    assert (w["random_15"] > w["random_5"]).all()
    cu = C[C.level == "unit"]
    assert len(cu) == 3 and set(cu.class_ref) <= {"mixed", "compartmentalized", "cold"}
    for f in ("sections.csv", "units.csv", "classes.csv", "meta.json"):
        assert (tmp_path / contact / f).exists()


def test_cli_and_refusals(grid, tmp_path):
    rc = main(["compartment", "contacts", "--labels", str(grid), "--out", str(tmp_path / "c"), "--immune", "T_NK",
               "--partners", "Tumour", "--eps", "0.1", "--reps", "2", "--k", "8", "--r", "15"])
    assert rc == 0 and (tmp_path / "c" / "compare_units.csv").exists()
    with pytest.raises(FileExistsError):
        ct.contacts(grid, tmp_path / "c", "L1", ["T_NK"], ["Tumour"], verbose=False)
    with pytest.raises(ValueError):
        ct.contacts(grid, tmp_path / "d", "L1", ["T_NK"], ["Tumour"], contact="boundary", verbose=False)


def test_aggregates(grid, tmp_path):
    S, U = ct.aggregates(grid, tmp_path / "agg", "L1", ["T_NK"], k=8, r=15.0, min_size=5, eps=(0.1,), reps=2,
                         unit_pattern=r"I0\d{3}", verbose=False)
    w = U.pivot_table(index="unit", columns="labels", values="share_in_aggregates")
    assert set(w.columns) == {"nucleus", "whole", "random_10"} and len(w) == 3
    E = ct.knn_edges(np.array([[0, 0], [1, 0], [2, 0], [50, 0]], float), k=2, r=1.5)
    r = ct.immune_components(np.array([0, 0, 0, 0]), E, [0], min_size=3)
    assert r["aggregates"] == 1 and r["cells_in_aggregates"] == 3 and r["largest"] == 3


def test_confusion_relabel_matches_the_real_changes():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 3, 20000)
    test = ref.copy()
    test[(ref == 0) & (rng.random(20000) < 0.4)] = 1          # 40% of type 0 become type 1, nothing else
    out = ct.confusion_relabel(ref, test, 3, np.random.default_rng(1))
    assert set(np.unique(out[ref == 2])) == {2} and set(np.unique(out[ref == 1])) == {1}
    assert abs((out[ref == 0] == 1).mean() - 0.4) < 0.02


def test_contacts_confusion_row(grid, tmp_path):
    kw = dict(eps=(0.1,), reps=3, k=8, r=15.0, mode="composition", verbose=False)
    _, U0, _ = ct.contacts(grid, tmp_path / "a", "L1", ["T_NK"], ["Tumour"], **kw)
    _, U1, _ = ct.contacts(grid, tmp_path / "b", "L1", ["T_NK"], ["Tumour"], confusion=True, **kw)
    col = "per100|T_NK|Tumour"
    w0 = U0.pivot_table(index="unit", columns="labels", values=col)
    w1 = U1.pivot_table(index="unit", columns="labels", values=col)
    assert set(w1.columns) == {"ref", "test", "random_10", "confusion"}
    pd.testing.assert_frame_equal(w0, w1[w0.columns])              # the other rows are unchanged
    # the analyst loses T cells only where they touch tumor; the same number placed at random loses fewer contacts
    assert (w1["test"] < w1["confusion"]).all() and (w1["confusion"] < w1["ref"]).all()


def test_counts_fixed_partners():
    """partners fixed at the reference labels; only the immune side of the changes varies."""
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 300, (3000, 2))
    E = ct.knn_edges(xy, 6, 30.0)
    ref = rng.integers(0, 3, 3000)                       # 0 immune, 1 partner, 2 other
    same = ct.counts_fixed(ref, ref, E, 3, [0], [1])
    plain = ct.counts(ref, E, 3, [0], [1])
    assert np.allclose(same["touch"], plain["touch"]) and np.allclose(same["n_part"], plain["n"])
    test = ref.copy()
    test[(ref == 1) & (rng.random(3000) < 0.5)] = 2      # only partner labels change
    pairs = [(ct._T("I", 0), ct._T("P", 1))]
    r_ref = ct.readouts(ct.counts(ref, E, 3, [0], [1]), pairs)["per100|I|P"]
    r_fix = ct.readouts(ct.counts_fixed(test, ref, E, 3, [0], [1]), pairs)["per100|I|P"]
    r_test = ct.readouts(ct.counts(test, E, 3, [0], [1]), pairs)["per100|I|P"]
    assert r_fix == pytest.approx(r_ref) and r_test != pytest.approx(r_ref)


def test_contacts_fixed_rows(grid, tmp_path):
    kw = dict(eps=(0.1,), reps=3, k=8, r=15.0, mode="composition", verbose=False, confusion=True)
    _, U0, _ = ct.contacts(grid, tmp_path / "a", "L1", ["T_NK"], ["Tumour"], **kw)
    _, U1, _ = ct.contacts(grid, tmp_path / "b", "L1", ["T_NK"], ["Tumour"], fix_partners=True, **kw)
    col = "per100|T_NK|Tumour"
    w0 = U0.pivot_table(index="unit", columns="labels", values=col)
    w1 = U1.pivot_table(index="unit", columns="labels", values=col)
    assert {"test_fixed", "confusion_fixed"} <= set(w1.columns)
    pd.testing.assert_frame_equal(w0, w1[w0.columns])              # the other rows are unchanged
    assert (w1["test_fixed"] < w1["confusion_fixed"]).all()       # T cells lost at tumor contact
