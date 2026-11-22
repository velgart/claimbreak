"""examples/compartment/xenium_protein.py: protein extraction from a 10x cell_feature_matrix.h5 and recall of
protein classes by label arms, split by PanCK+ contact."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")
ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("xenium_protein", ROOT / "examples" / "compartment" / "xenium_protein.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _h5(path, cells, genes, proteins, G, Pv):
    """10x-style cell_feature_matrix.h5 (features x cells, CSC over cells)."""
    import scipy.sparse as sp
    X = sp.csc_matrix(np.vstack([G.T, Pv.T]))
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["data"], m["indices"], m["indptr"] = X.data.astype(np.int32), X.indices, X.indptr
        m["shape"] = np.array(X.shape)
        m["barcodes"] = np.array([c.encode() for c in cells])
        ft = m.create_group("features")
        ft["name"] = np.array([n.encode() for n in genes + proteins])
        ft["feature_type"] = np.array([b"Gene Expression"] * len(genes) + [b"Protein Expression"] * len(proteins))


def _section(root, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1200, (n, 2))
    true = np.where(xy[:, 0] < 500, "Tumour", rng.choice(["T", "B", "Mye", "Stroma"], n))
    true[(xy[:, 0] < 500) & (rng.random(n) < 0.15)] = "T"
    cells = [f"c{i}" for i in range(n)]
    prot = ["CD3E", "CD8A", "CD4", "CD20", "CD68", "PanCK", "PD-1"]
    sig = {"CD3E": true == "T", "CD8A": (true == "T") & (rng.random(n) < 0.5), "CD4": (true == "T") & (rng.random(n) < 0.5),
           "CD20": true == "B", "CD68": true == "Mye", "PanCK": true == "Tumour", "PD-1": np.zeros(n, bool)}
    Pv = np.column_stack([np.where(sig[p], rng.lognormal(5, 0.4, n), rng.lognormal(0.5, 0.8, n) * (rng.random(n) < 0.4))
                          for p in prot]).round().astype(int)
    G = rng.poisson(2, (n, 5))
    _h5(root / "cell_feature_matrix.h5", cells, [f"g{i}" for i in range(5)], prot, G, Pv)
    # labels: nucleus arm correct; whole arm loses T cells that touch tumor (they become PT)
    from scipy.spatial import cKDTree
    _, nn = cKDTree(xy).query(xy, k=7)
    touch_tum = (true[nn[:, 1:]] == "Tumour").any(1)
    lab_map = {"Tumour": "PT", "T": "T_NK", "B": "B", "Mye": "Myeloid", "Stroma": "Stromal"}
    nuc = np.array([lab_map[t] for t in true])
    whole = nuc.copy()
    whole[(true == "T") & touch_tum & (rng.random(n) < 0.6)] = "PT"
    L = pd.DataFrame({"sample": "XP_T", "cell_id": cells, "x": xy[:, 0], "y": xy[:, 1], "L1_nucleus": nuc,
                      "L1_whole": whole})
    (root / "lab").mkdir()
    L.to_csv(root / "lab" / "XP_T.labels.csv.gz", index=False, compression="gzip")
    return root


def test_extract_and_truth(tmp_path):
    m = _mod()
    root = _section(tmp_path)
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_T", out=str(tmp_path / "prot")))
    P = pd.read_csv(tmp_path / "prot" / "XP_T.protein.csv.gz")
    assert list(P.columns) == ["cell_id", "CD3E", "CD8A", "CD4", "CD20", "CD68", "PanCK", "PD-1"] and len(P) == 4000
    assert json.loads((tmp_path / "prot" / "XP_T.protein_markers.json").read_text())["cells"] == 4000
    m.cmd_truth(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_T.protein.csv.gz"),
                                sample="XP_T", out=str(tmp_path / "tr"), level="L1", arms="nucleus,whole", k=6, r=30.0,
                                seed=0, partners="PanCK,CD68"))
    M = pd.read_csv(tmp_path / "tr" / "markers.csv").set_index("marker")
    assert (M.loc[["CD3E", "CD20", "CD68", "PANCK"], "ashman_d"] >= 2).all()
    R = pd.read_csv(tmp_path / "tr" / "recall.csv")
    r = R[(R.protein_class == "T") & (R.variant == "all")].pivot_table(index="stratum", columns="arm", values="recall")
    assert r.loc["all", "nucleus"] > 0.95
    assert r.loc["panck_contact", "whole"] < r.loc["no_panck_contact", "whole"] - 0.3     # loss sits at PanCK+ contact
    assert abs(r.loc["no_panck_contact", "whole"] - r.loc["no_panck_contact", "nucleus"]) < 0.05
    rb = R[(R.protein_class == "B") & (R.variant == "all") & (R.stratum == "all")].set_index("arm").recall
    assert rb["nucleus"] > 0.95 and rb["whole"] > 0.95
    F = pd.read_csv(tmp_path / "tr" / "fate.csv")
    assert F[(F.protein_class == "T") & (F.variant == "missed") & (F.arm == "whole")].sort_values("share").label.iloc[-1] == "PT"
    Pr = pd.read_csv(tmp_path / "tr" / "precision.csv").set_index(["protein_class", "arm"]).precision
    assert Pr[("T", "nucleus")] > 0.9


def test_extract_refuses_matrix_without_protein(tmp_path):
    m = _mod()
    _h5(tmp_path / "cell_feature_matrix.h5", ["a", "b"], ["g0"], [], np.array([[1], [2]]), np.zeros((2, 0)))
    with pytest.raises(SystemExit):
        m.cmd_extract(SimpleNamespace(outs=str(tmp_path), sample="S", out=str(tmp_path / "o")))


def test_positivity_ignores_low_nonzero_background():
    """most negatives read 0 but many read small non-zero values; 8% true positives must not become 50%."""
    m = _mod()
    rng = np.random.default_rng(5)
    n = 20000
    truth = rng.random(n) < 0.08
    v = np.where(truth, rng.lognormal(5.5, 0.5, n), np.where(rng.random(n) < 0.5, rng.lognormal(1.5, 0.8, n), 0)).round()
    pos, d, thr = m.positive(v)
    assert d >= 2 and abs(pos.mean() - 0.08) < 0.02
    assert pos[truth].mean() > 0.9 and pos[~truth].mean() < 0.02
    assert thr > 20


def test_cd68_partner_strata(tmp_path):
    """contact strata for every separating partner marker (here PanCK and CD68)."""
    m = _mod()
    root = _section(tmp_path)
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_T", out=str(tmp_path / "prot")))
    m.cmd_truth(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_T.protein.csv.gz"),
                                sample="XP_T", out=str(tmp_path / "tr"), level="L1", arms="nucleus,whole", k=6, r=30.0,
                                seed=0, partners="PanCK,CD68"))
    R = pd.read_csv(tmp_path / "tr" / "recall.csv")
    assert {"panck_contact", "no_panck_contact", "cd68_contact", "no_cd68_contact"} <= set(R.stratum)
    meta = json.loads((tmp_path / "tr" / "meta.json").read_text())
    assert set(meta["contact_share"]) == {"PANCK", "CD68"}


def test_block_bootstrap_of_protein_loss(tmp_path):
    """recall_ci.csv; the synthetic loss at PanCK+ contact has an interval above the loss without contact."""
    m = _mod()
    root = _section(tmp_path)
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_T", out=str(tmp_path / "prot")))
    m.cmd_truth(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_T.protein.csv.gz"),
                                sample="XP_T", out=str(tmp_path / "tr"), level="L1", arms="nucleus,whole", k=6, r=30.0,
                                seed=0, partners="PanCK", tile=200.0, block_draws=200))
    C = pd.read_csv(tmp_path / "tr" / "recall_ci.csv")
    d = C[(C.protein_class == "T") & (C.metric == "loss_whole_contact_minus_no") & (C.stratum == "panck_contact")].iloc[0]
    assert d.lo > 0.2 and d.lo <= d.estimate <= d.hi
    R = pd.read_csv(tmp_path / "tr" / "recall.csv")
    r = R[(R.protein_class == "T") & (R.variant == "all") & (R.arm == "nucleus") & (R.stratum == "all")].recall.iloc[0]
    e = C[(C.protein_class == "T") & (C.metric == "recall_nucleus") & (C.stratum == "all")].estimate.iloc[0]
    assert abs(r - e) < 1e-9                                   # point estimate = the unresampled recall


def test_remedy_pairs_and_gap_closed(tmp_path):
    """--pairs and --gaps; a model arm that recovers half of the T cells the base arm lost closes about half of
    the gap, with a block interval; arms not in the labels are dropped from pairs and gaps."""
    m = _mod()
    root = _section(tmp_path)
    f = root / "lab" / "XP_T.labels.csv.gz"
    L = pd.read_csv(f)
    rng = np.random.default_rng(3)
    L["L1_nucB"] = L["L1_nucleus"]
    L["L1_wBall"] = L["L1_whole"]
    fix = (L.L1_whole != L.L1_nucleus) & (rng.random(len(L)) < 0.5)
    L["L1_wBallM"] = np.where(fix, L.L1_nucleus, L.L1_whole)
    L.to_csv(f, index=False, compression="gzip")
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_T", out=str(tmp_path / "prot")))
    m.cmd_truth(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_T.protein.csv.gz"),
                                sample="XP_T", out=str(tmp_path / "tr"), level="L1",
                                arms="nucleus,whole,nucB,wBall,wBallM", k=6, r=30.0, seed=0, partners="PanCK",
                                tile=200.0, block_draws=200, pairs="nucB:wBall,nucB:wBallM,nucleus:wholeM",
                                gaps="nucB:wBall:wBallM,nucleus:whole:wholeM"))
    C = pd.read_csv(tmp_path / "tr" / "recall_ci.csv")
    g = C[(C.protein_class == "T") & (C.metric == "gap_closed_wBallM_wBall") & (C.stratum == "all")].iloc[0]
    assert 0.35 < g.estimate < 0.65 and g.lo < g.estimate < g.hi
    assert {"loss_wBall_vs_nucB", "loss_wBallM_vs_nucB"} <= set(C.metric)
    assert not C.metric.str.contains("wholeM").any()          # arm absent: pair and gap dropped
    meta = json.loads((tmp_path / "tr" / "meta.json").read_text())
    assert meta["gaps"] == [["nucB", "wBall", "wBallM"]]


def test_gaps_need_triples(tmp_path):
    m = _mod()
    root = _section(tmp_path)
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_T", out=str(tmp_path / "prot")))
    with pytest.raises(SystemExit):
        m.cmd_truth(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_T.protein.csv.gz"),
                                    sample="XP_T", out=str(tmp_path / "tr"), level="L1", arms="nucleus,whole", k=6,
                                    r=30.0, seed=0, partners="PanCK", tile=200.0, block_draws=50,
                                    pairs="nucleus:whole", gaps="nucleus:whole"))


def _grid_section(root, nx=60, ny=60, side=8.0, seed=7):
    """Touching square cells (polygons in <sample>_cell_boundaries.parquet), tumor on the left, immune on the right
    with a mixed band; whole-cell labels lose a T cell with probability 0.6 x its tumor share of the perimeter."""
    rng = np.random.default_rng(seed)
    n = nx * ny
    col, row = np.arange(n) % nx, np.arange(n) // nx
    xy = np.c_[col * side + side / 2, row * side + side / 2]
    p_tum = np.clip((nx / 2 - col) / 12 + 0.5, 0, 1)
    true = np.where(rng.random(n) < p_tum, "Tumour", rng.choice(["T", "T", "B", "Mye"], n))
    cells = [f"c{i}" for i in range(n)]
    prot = ["CD3E", "CD8A", "CD4", "CD20", "CD68", "PanCK", "CD45"]
    imm = np.isin(true, ["T", "B", "Mye"])
    sig = {"CD3E": true == "T", "CD8A": (true == "T") & (rng.random(n) < 0.5), "CD4": (true == "T") & (rng.random(n) < 0.5),
           "CD20": true == "B", "CD68": true == "Mye", "PanCK": true == "Tumour", "CD45": imm}
    Pv = np.column_stack([np.where(sig[p], rng.lognormal(5, 0.4, n), rng.lognormal(0.5, 0.8, n) * (rng.random(n) < 0.4))
                          for p in prot]).round().astype(int)
    _h5(root / "cell_feature_matrix.h5", cells, ["g0"], prot, rng.poisson(2, (n, 1)), Pv)
    rows = []
    for i in range(n):
        x0, y0 = col[i] * side, row[i] * side
        for vx, vy in ((x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side), (x0, y0)):
            rows.append((cells[i], vx, vy))
    (root / "bnd").mkdir()
    pd.DataFrame(rows, columns=["cell_id", "vertex_x", "vertex_y"]).to_parquet(root / "bnd" / "XP_G_cell_boundaries.parquet")
    idx = np.arange(n).reshape(ny, nx)
    tum_share = np.zeros(n)
    for i in range(n):
        r, c = divmod(i, nx)
        nb = [idx[rr, cc] for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= rr < ny and 0 <= cc < nx]
        tum_share[i] = np.mean([true[j] == "Tumour" for j in nb])
    lab_map = {"Tumour": "PT", "T": "T_NK", "B": "B", "Mye": "Myeloid"}
    nuc = np.array([lab_map[t] for t in true])
    whole = nuc.copy()
    whole[(true == "T") & (rng.random(n) < 0.6 * tum_share)] = "PT"
    (root / "lab").mkdir()
    pd.DataFrame({"sample": "XP_G", "cell_id": cells, "x": xy[:, 0], "y": xy[:, 1], "L1_nucB": nuc,
                  "L1_wholeB": whole}).to_csv(root / "lab" / "XP_G.labels.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"cell_id": cells, "group": [{"T": "Cycling T Cells", "B": "B Cells", "Mye": "Macrophages",
                                               "Tumour": "Tumour cells"}[t] for t in true]}).to_csv(root / "groups.csv", index=False)
    return root


def test_border_loss_rises_with_non_immune_share(tmp_path):
    """`border` bins protein T cells by the CD45-negative share of their polygon perimeter; the synthetic loss
    (0.6 x tumor share) rises across bins and the slope is near 0.6."""
    m = _mod()
    root = _grid_section(tmp_path)
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_G", out=str(tmp_path / "prot")))
    m.cmd_border(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_G.protein.csv.gz"),
                                 boundaries=str(root / "bnd"), sample="XP_G", out=str(tmp_path / "bo"), level="L1",
                                 immune_marker="CD45", tol=1.0, classes="T,B", tile=100.0, block_draws=100,
                                 min_cells=20, seed=0))
    C = pd.read_csv(tmp_path / "bo" / "loss_by_share.csv")
    t = C[(C.protein_class == "T") & (C.metric == "loss_wholeB_vs_nucB")].set_index("stratum")
    assert t.loc["nonimm_0", "estimate"] < 0.05 < t.loc["nonimm_(0.5,0.75]", "estimate"]
    F = pd.read_csv(tmp_path / "bo" / "slope.csv").set_index("protein_class")
    assert 0.4 < F.loc["T", "slope"] < 0.8
    b = C[(C.protein_class == "B") & (C.metric == "loss_wholeB_vs_nucB") & (C.stratum == "all")].estimate.iloc[0]
    assert abs(b) < 0.02                                   # B cells are not lost in the synthetic tissue


def test_groups_scored_against_protein(tmp_path):
    """`groups` scores a published labeling: T groups recall the protein T cells, the map's regexes pick the
    right groups."""
    m = _mod()
    root = _grid_section(tmp_path)
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_G", out=str(tmp_path / "prot")))
    m.cmd_groups(SimpleNamespace(groups=str(root / "groups.csv"), protein=str(tmp_path / "prot" / "XP_G.protein.csv.gz"),
                                 out=str(tmp_path / "gr"), map=m.GROUP_MAP, seed=0))
    R = pd.read_csv(tmp_path / "gr" / "recall_precision.csv").set_index("protein_class")
    assert R.loc["T", "groups"] == "Cycling T Cells" and R.loc["T", "recall"] > 0.95 and R.loc["T", "precision"] > 0.9
    assert R.loc["B", "groups"] == "B Cells" and R.loc["Myeloid", "groups"] == "Macrophages"
    S = pd.read_csv(tmp_path / "gr" / "group_marker_shares.csv", index_col=0)
    assert S.loc["Tumour cells", "T"] < 0.05


def test_groups_border_scores_published_labels_by_share(tmp_path):
    """`groups-border` scores a published labeling and the arms on the same cells by the non-immune share of
    the outline; published labels made like wholeB (T lost with probability 0.6 x tumor share) fall with the share,
    nucleus-like ones do not; ungrouped cells are left out of recall but kept as neighbors."""
    m = _mod()
    root = _grid_section(tmp_path)
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_G", out=str(tmp_path / "prot")))
    L = pd.read_csv(root / "lab" / "XP_G.labels.csv.gz")
    name = {"PT": "Tumour cells", "T_NK": "Cycling T Cells", "B": "B Cells", "Myeloid": "Macrophages"}
    G = pd.DataFrame({"cell_id": L.cell_id, "group": L.L1_wholeB.map(name)})
    G = G.iloc[::2]                                          # half the cells grouped
    G.to_csv(tmp_path / "pub.csv", index=False)
    m.cmd_groups_border(SimpleNamespace(groups=str(tmp_path / "pub.csv"), labels=str(root / "lab"),
                                        protein=str(tmp_path / "prot" / "XP_G.protein.csv.gz"),
                                        boundaries=str(root / "bnd"), sample="XP_G", out=str(tmp_path / "gb"),
                                        level="L1", map=m.GROUP_MAP, arms="nucB,wholeB", pairs="nucB:pub,nucB:wholeB",
                                        immune_marker="CD45", tol=1.0, classes="T,B", tile=100.0, block_draws=50,
                                        min_cells=10, seed=0))
    C = pd.read_csv(tmp_path / "gb" / "recall_by_share.csv")
    t = C[(C.protein_class == "T")].set_index(["metric", "stratum"])
    assert t.loc[("recall_pub", "outline_0"), "estimate"] > 0.95
    assert t.loc[("loss_pub_vs_nucB", "all"), "estimate"] > 0.03
    assert t.loc[("recall_pub", "all"), "cells"] < 0.6 * (L.L1_nucB == "T_NK").sum()   # grouped half only
    F = pd.read_csv(tmp_path / "gb" / "slope.csv").set_index(["protein_class", "metric", "base"])
    assert F.loc[("T", "recall_pub", "outline"), "slope"] < -0.3
    assert abs(F.loc[("T", "recall_nucB", "outline"), "slope"]) < 0.1


def test_border_outline_base(tmp_path):
    """`border --base outline` bins by the share of the whole outline; in the closed grid every interior cell
    is fully touched, so both bases agree there and the loss still rises."""
    m = _mod()
    root = _grid_section(tmp_path)
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_G", out=str(tmp_path / "prot")))
    m.cmd_border(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_G.protein.csv.gz"),
                                 boundaries=str(root / "bnd"), sample="XP_G", out=str(tmp_path / "bo"), level="L1",
                                 immune_marker="CD45", tol=1.0, classes="T", tile=100.0, block_draws=50,
                                 min_cells=20, seed=0, base="outline"))
    F = pd.read_csv(tmp_path / "bo" / "slope.csv").set_index("protein_class")
    assert 0.3 < F.loc["T", "slope"] < 0.8
    assert json.loads((tmp_path / "bo" / "meta.json").read_text())["base"] == "outline"


def test_strict_classes_drop_spilled_protein(tmp_path):
    """CD3E read on tumor cells next to T cells (protein from a neighbor) enters the loose T class but not
    T_strict (CD3E+ CD45+ CD68- PanCK-); class_overlap.csv shows it; a stricter posterior calls fewer cells."""
    m = _mod()
    root = _section(tmp_path)
    rng = np.random.default_rng(3)
    L = pd.read_csv(root / "lab" / "XP_T.labels.csv.gz")
    n = len(L)
    tum = L.L1_nucleus.to_numpy() == "PT"
    t = L.L1_nucleus.to_numpy() == "T_NK"
    imm = np.isin(L.L1_nucleus.to_numpy(), ["T_NK", "B", "Myeloid"])
    spill = tum & (rng.random(n) < 0.3)
    hi, lo = rng.lognormal(5, 0.4, n), rng.lognormal(0.5, 0.8, n) * (rng.random(n) < 0.4)
    P = pd.DataFrame({"cell_id": L.cell_id,
                      "CD3E": np.where(t | spill, rng.lognormal(5, 0.4, n), lo),
                      "CD8A": np.where(t & (rng.random(n) < 0.5), hi, lo),
                      "CD4": np.where(t & (rng.random(n) < 0.5), hi, lo),
                      "CD20": np.where(L.L1_nucleus == "B", hi, lo),
                      "CD68": np.where(L.L1_nucleus == "Myeloid", hi, lo),
                      "PanCK": np.where(tum, hi, lo),
                      "CD45": np.where(imm, rng.lognormal(5, 0.4, n), lo)}).round()
    P.to_csv(tmp_path / "p.csv.gz", index=False, compression="gzip")
    common = dict(labels=str(root / "lab"), protein=str(tmp_path / "p.csv.gz"), sample="XP_T", level="L1",
                  arms="nucleus,whole", k=6, r=30.0, seed=0, partners="PanCK,CD68", block_draws=0)
    m.cmd_truth(SimpleNamespace(out=str(tmp_path / "loose"), **common))
    m.cmd_truth(SimpleNamespace(out=str(tmp_path / "strict"), strict=True, posterior=0.9, **common))
    assert not (tmp_path / "loose" / "class_overlap.csv").read_text().count("T_strict")
    O = pd.read_csv(tmp_path / "strict" / "class_overlap.csv").set_index("protein_class")
    assert O.loc["T", "share_PANCK+"] > 0.2 and O.loc["T_strict", "share_PANCK+"] == 0
    assert O.loc["T_strict", "share_CD45+"] == 1 and O.loc["T_strict", "cells"] < O.loc["T", "cells"]
    R = pd.read_csv(tmp_path / "strict" / "recall.csv")
    r = R[(R.variant == "all") & (R.stratum == "all") & (R.arm == "nucleus")].set_index("protein_class").recall
    assert r["T_strict"] > 0.95 > r["T"]                         # the spilled tumor cells were the misses
    Ml = pd.read_csv(tmp_path / "loose" / "markers.csv").set_index("marker").positive_share
    Ms = pd.read_csv(tmp_path / "strict" / "markers.csv").set_index("marker").positive_share
    assert (Ms <= Ml + 1e-9).all()


def test_recall_counts_unlabelled_cells_as_misses(tmp_path):
    """recall_all scores every protein cell of the class (unlabeled in the arm = missed) and `labeled` gives
    the labeled share; the block bootstrap writes recallall_ and lossall_ metrics beside recall_ and loss_."""
    m = _mod()
    root = _section(tmp_path)
    f = root / "lab" / "XP_T.labels.csv.gz"
    L = pd.read_csv(f, keep_default_na=False)
    L.loc[L.index[::4], "L1_nucleus"] = ""                 # a quarter of cells too shallow for the nuclear arm
    L.to_csv(f, index=False, compression="gzip")
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_T", out=str(tmp_path / "prot")))
    m.cmd_truth(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_T.protein.csv.gz"),
                                sample="XP_T", out=str(tmp_path / "tr"), level="L1", arms="nucleus,whole", k=6, r=30.0,
                                seed=0, partners="PanCK,CD68", block_draws=50))
    R = pd.read_csv(tmp_path / "tr" / "recall.csv")
    q = R[(R.protein_class == "T") & (R.variant == "all") & (R.stratum == "all")].set_index("arm")
    assert 0.7 < q.loc["nucleus", "labelled"] < 0.8 and q.loc["whole", "labelled"] > 0.99
    assert q.loc["nucleus", "recall_all"] < q.loc["nucleus", "recall"] - 0.15
    CI = pd.read_csv(tmp_path / "tr" / "recall_ci.csv")
    assert {"recallall_nucleus", "labelled_nucleus", "lossall_whole_vs_nucleus"} <= set(CI.metric)


def test_border_slope_within_regional_bins(tmp_path):
    """`border --region` also fits the slope within bins of the regional non-immune share
    (slope_regional.csv); slope.csv keeps its columns."""
    m = _mod()
    root = _grid_section(tmp_path)
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_G", out=str(tmp_path / "prot")))
    m.cmd_border(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_G.protein.csv.gz"),
                                 boundaries=str(root / "bnd"), sample="XP_G", out=str(tmp_path / "bo"), level="L1",
                                 immune_marker="CD45", tol=1.0, classes="T", tile=100.0, block_draws=50,
                                 min_cells=10, seed=0, region="30,100", region_bins=2))
    F = pd.read_csv(tmp_path / "bo" / "slope.csv")
    assert list(F.columns) == ["protein_class", "slope", "bins"]
    FR = pd.read_csv(tmp_path / "bo" / "slope_regional.csv")
    assert {"all", "0", "1"} <= set(FR.region_bin.astype(str))
    assert json.loads((tmp_path / "bo" / "meta.json").read_text())["region"] == "30,100"


def test_hybrid_arm_falls_back_to_whole_cell_labels(tmp_path):
    """--hybrid nucleus:whole labels every cell: the nuclear label where there is one, else the whole-cell
    label; with a quarter of cells unlabeled in the nucleus, its all-cell recall beats the nucleus alone."""
    m = _mod()
    root = _section(tmp_path)
    f = root / "lab" / "XP_T.labels.csv.gz"
    L = pd.read_csv(f, keep_default_na=False)
    L.loc[L.index[::4], "L1_nucleus"] = ""
    L.to_csv(f, index=False, compression="gzip")
    m.cmd_extract(SimpleNamespace(outs=str(root), sample="XP_T", out=str(tmp_path / "prot")))
    m.cmd_truth(SimpleNamespace(labels=str(root / "lab"), protein=str(tmp_path / "prot" / "XP_T.protein.csv.gz"),
                                sample="XP_T", out=str(tmp_path / "tr"), level="L1", arms="nucleus,whole", k=6, r=30.0,
                                seed=0, partners="PanCK,CD68", block_draws=20, hybrid="nucleus:whole"))
    R = pd.read_csv(tmp_path / "tr" / "recall.csv")
    q = R[(R.protein_class == "T") & (R.variant == "all") & (R.stratum == "all")].set_index("arm")
    assert q.loc["nucleus+whole", "labelled"] > 0.99
    assert q.loc["nucleus+whole", "recall_all"] > q.loc["nucleus", "recall_all"] + 0.1
    Pr = pd.read_csv(tmp_path / "tr" / "precision.csv")
    assert "nucleus+whole" in set(Pr.arm)
