import numpy as np
import pandas as pd
import pytest

from claimbreak import perturb, plant, split, thin, questions as Q, compare
from claimbreak.labeling import NearestCentroid, native_confusion
from claimbreak.tissue import check_output_dir


def test_split_conserves(tissue):
    A, B = split(tissue, 0.5, np.random.default_rng(0))
    assert np.allclose((A.counts + B.counts).toarray(), tissue.counts.toarray())
    assert A.graph is tissue.graph


def test_thin_fraction(tissue):
    t = thin(tissue, 0.25, np.random.default_rng(1))
    r = t.counts.sum() / tissue.counts.sum()
    assert abs(r - 0.25) < 0.01


def test_label_errors_rate(tissue):
    C = perturb.uniform_confusion(tissue.types)
    for mode in ("random", "boundary"):
        w = perturb.spatial_weights(tissue, mode)
        lab, info = perturb.label_errors(tissue, 0.1, C, np.random.default_rng(2), weights=w)
        assert abs(info["realised_eps"] - 0.1) < 0.02, (mode, info)
        assert set(lab) <= set(tissue.types)


def test_boundary_concentrates(tissue):
    C = perturb.uniform_confusion(tissue.types)
    w = perturb.spatial_weights(tissue, "boundary", gamma=2)
    lab, _ = perturb.label_errors(tissue, 0.1, C, np.random.default_rng(3), weights=w)
    het = perturb.heterogeneity(tissue)
    err = lab != tissue.labels
    assert err[het > np.median(het)].mean() > 1.5 * err[het <= np.median(het)].mean()


def test_lineage_confusion(tissue):
    C = perturb.lineage_confusion(tissue.types, lambda x: x[0])
    lab, _ = perturb.label_errors(tissue, 0.2, C, np.random.default_rng(4), types=["A", "A2", "B", "B2"])
    ch = lab != tissue.labels
    assert all(a[0] == b[0] for a, b in zip(tissue.labels[ch], lab[ch]))


def test_contamination_conserves_and_moves(tissue):
    t, info = perturb.contaminate(tissue, 0.2, np.random.default_rng(5), lam=10.0)
    assert t.counts.sum() == tissue.counts.sum()
    assert abs(info["moved_fraction"] - 0.2) < 0.01
    # A-type marker genes (30-39 are shared by targets; 0-9 Epi markers) appear in neighbors of Epi cells
    epi_mark = t.counts[:, :10].sum(1).A1 / t.counts.sum(1).A1
    epi0 = tissue.counts[:, :10].sum(1).A1 / tissue.counts.sum(1).A1
    nonepi = tissue.labels != "Epi"
    assert epi_mark[nonepi].mean() > epi0[nonepi].mean()


def test_null_keeps_composition(tissue):
    t, _ = plant.null(tissue, np.random.default_rng(6))
    assert np.allclose(np.sort(t.counts.sum(1).A1), np.sort(tissue.counts.sum(1).A1))
    for ty in tissue.types:  # each cell keeps a profile of its own type
        m = tissue.labels == ty
        assert np.isclose(t.counts[m].sum(), tissue.counts[m].sum())


def test_spike_raises_near_senders(tissue):
    t, rec = plant.spike(tissue, "Inf", "A", 5, effect=1.5, lam=15, rng=np.random.default_rng(7))
    g = [int(x[1:]) for x in rec["genes"]]
    cells = np.flatnonzero(tissue.labels == "A")
    d = plant.distance_to_type(tissue, cells, "Inf")
    near, far = cells[d < 10], cells[d > 60]
    ratio = lambda X, c: X[c][:, g].sum() / X[c].sum()
    assert ratio(t.counts, near) > 1.5 * ratio(tissue.counts, near)
    assert ratio(t.counts, far) < 1.2 * ratio(tissue.counts, far)


def test_labeller_accuracy(tissue):
    lab = NearestCentroid(tissue).predict(tissue.counts, tissue.section)
    assert (lab == tissue.labels).mean() > 0.8
    C, rate = native_confusion(tissue, 0.5, np.random.default_rng(8))
    assert 0 < rate < 0.5 and np.allclose(C.to_numpy().sum(1), 1)


def test_output_guard(tmp_path):
    data = tmp_path / "data" / "x.h5ad"
    data.parent.mkdir()
    with pytest.raises(ValueError):
        check_output_dir(tmp_path / "data" / "out", data)
    check_output_dir(tmp_path / "results", data)


def test_questions_run(tissue):
    for name in ["composition", "colocalization", "niches"]:
        q = Q.make(name); q.prepare(tissue)
        c = q.run(tissue, tissue.labels)
        assert {"claim", "stat", "p", "decision"} <= set(c.columns)
        assert compare.versus_baseline(c, c)["survival"] in (1.0,) or c.decision.eq(0).all()


def test_proximity_de_null_and_spike(tissue):
    rng = np.random.default_rng(9)
    nul, _ = plant.null(tissue, rng)
    sp_, rec = plant.spike(nul, "Inf", "A", 5, effect=1.5, lam=25, rng=rng)
    truth = compare.planted_truth(sp_.meta["planted"], "proximity_de", sp_.types)
    q = Q.ProximityDE(n_genes=60, index_types=["A", "B"], neighbour_types=["Inf", "Epi"], aggregate="pooled")
    q.prepare(sp_)
    v = compare.versus_truth(q.run(sp_, sp_.labels), truth, null=True)
    assert v["power"] >= 0.8 and v["fdr"] <= 0.2, v
    # unit-level test: planted genes carry the right sign even when underpowered
    qu = Q.ProximityDE(n_genes=60, index_types=["A"], neighbour_types=["Inf"])
    qu.prepare(sp_)
    cu = qu.run(sp_, sp_.labels).set_index("claim")
    assert all(cu.at[c, "stat"] > 0 for c, _ in truth)


def test_versus_baseline_overlap():
    b = pd.DataFrame({"claim": list("abcd"), "decision": [1, 1, 0, 0]})
    r = pd.DataFrame({"claim": list("abcd"), "decision": [1, 0, 1, 1]})
    v = compare.versus_baseline(b, r)
    assert v["kept"] == 1 and v["spurious"] == 2 and v["n_run"] == 3
    assert np.isclose(v["jaccard"], 1 / 4) and np.isclose(v["precision"], 1 / 3)


def test_break_points_direction():
    base = pd.DataFrame({"question": "q", "claim": ["x"], "decision": [1], "support": [10]})
    rows = []
    for lev, d in [(1.0, 1), (0.5, 1), (0.25, 0)]:              # depth arm: breaks only at the lowest depth
        rows.append({"spec": "s", "question": "q", "claim": "x", "level": lev, "rep": 0, "decision": d})
    d = pd.DataFrame(rows)
    assert compare.break_points(d, base, descending=True).break_level.iloc[0] == 0.25
    rows = [{"spec": "s", "question": "q", "claim": "x", "level": lev, "rep": 0, "decision": dd}
            for lev, dd in [(0.02, 1), (0.1, 0), (0.2, 0)]]      # error arm: breaks at 0.1
    bp = compare.break_points(pd.DataFrame(rows), base, native_rate=0.05)
    assert bp.break_level.iloc[0] == 0.1 and np.isclose(bp.fragility.iloc[0], 2.0)
    assert bp.base_support.iloc[0] == 10


def test_colocalization_rules(tissue):
    q = Q.Colocalization(n_perm=20); q.prepare(tissue)
    c = q.run(tissue, tissue.labels)
    assert not any(a == b for a, b in c.claim.str.split("|"))           # self-pairs excluded
    pos = c[c.decision > 0]
    assert (2 ** pos.effect >= 1.5 - 1e-9).all() and (pos.stat > 2).all()
    q2 = Q.Colocalization(n_perm=20, min_ratio=1.0, include_self=True); q2.prepare(tissue)
    assert (q2.run(tissue, tissue.labels).decision != 0).sum() >= (c.decision != 0).sum()


def test_niches_coenrichment_and_seed(tissue):
    q = Q.Niches(n_niches=6, n_init=3); q.prepare(tissue)
    c0, c1 = q.run(tissue, tissue.labels, rep=0), q.run(tissue, tissue.labels, rep=1)
    co = c0[c0.claim.str.startswith("co|")]
    assert len(co) == len(tissue.types) * (len(tissue.types) - 1) // 2
    assert set(c0.claim) == set(c1.claim)


def test_native_flip_below_disagreement(tissue):
    from claimbreak.labeling import native_flip
    A, _ = split(tissue, 0.5, np.random.default_rng(0))
    f = native_flip(A, np.random.default_rng(1), n_resample=2)
    _, dis = native_confusion(A, 1.0, np.random.default_rng(2))
    assert 0 < f < 1 and f <= dis + 0.05


def test_type_rates(tissue):
    C = perturb.uniform_confusion(tissue.types)
    mult = {t: (4.0 if t in ("A", "A2") else 1.0) for t in tissue.types}
    w = perturb.spatial_weights(tissue, "random", type_mult=mult)
    lab, info = perturb.label_errors(tissue, 0.1, C, np.random.default_rng(10), weights=w)
    assert abs(info["realised_eps"] - 0.1) < 0.02
    err = pd.Series(lab != tissue.labels).groupby(tissue.labels).mean()
    assert err[["A", "A2"]].mean() > 3 * err[["Epi", "Inf", "Endo"]].mean()


def test_native_flip_by_type(tissue):
    from claimbreak.labeling import native_flip
    f, per = native_flip(tissue, np.random.default_rng(3), n_resample=1, by_type=True)
    assert set(per) == set(tissue.types) and 0 <= min(per.values()) <= f <= max(per.values())


def test_niche_consensus(tissue):
    q = Q.Niches(n_niches=6, n_seeds=5); q.prepare(tissue)
    c = q.run(tissue, tissue.labels, rep=0)
    assert c.stat.between(0, 1).all()
    pos = c[c.decision > 0]
    assert (pos.stat >= 0.7).all() and "unstable" in c
    assert not (c.unstable & (c.decision != 0)).any()


def test_break_summary():
    bp = pd.DataFrame({"spec": "s", "question": "q", "claim": [f"c{i}" for i in range(8)],
                       "break_level": [0.02, np.inf, 0.1, np.inf, 0.02, np.inf, np.inf, np.inf],
                       "base_support": [10, 20, 30, 40, 1000, 2000, 3000, 4000],
                       "fragility": [0.2, np.inf, 1.0, np.inf, 0.2, np.inf, np.inf, np.inf]})
    s = compare.break_summary(bp, 0.1)
    assert len(s) == 4 and s.n.sum() == 8
    assert s.broken.iloc[0] >= s.broken.iloc[-1]


def test_label_errors_by_condition(tissue):
    T = list(tissue.types)
    def only(src, dst):
        M = pd.DataFrame(0.0, index=T, columns=T)
        for t in T:
            M.loc[t, dst if t == src else [x for x in T if x != t][0]] = 1.0
        return M
    C = {"ctrl": only("A", "A2"), "dis": only("A", "B")}
    lab, info = perturb.label_errors(tissue, 0.3, C, np.random.default_rng(11), types=["A"])
    ch = lab != tissue.labels
    assert set(lab[ch & (tissue.condition == "ctrl")]) == {"A2"}
    assert set(lab[ch & (tissue.condition == "dis")]) == {"B"}
    assert "realised_eps_ctrl" in info


def test_native_confusion_mask_fallback(tissue):
    m = tissue.condition == "ctrl"
    pooled, _ = native_confusion(tissue, 0.5, np.random.default_rng(12), min_errors=1)
    Cm, dm = native_confusion(tissue, 0.5, np.random.default_rng(13), mask=m, fallback=pooled, min_errors=10**9)
    assert np.allclose(Cm.to_numpy(), pooled.to_numpy())          # every row falls back
    Cm2, _ = native_confusion(tissue, 0.5, np.random.default_rng(13), mask=m, types=tissue.types)
    assert list(Cm2.index) == list(tissue.types) and 0 < dm < 1


def test_joint_niche_de_reduces_misattribution(tissue):
    nul, _ = plant.null(tissue, np.random.default_rng(9))
    sp_, _ = plant.spike(nul, "Inf", "A", 8, effect=1.5, lam=15, rng=np.random.default_rng(1))
    truth = compare.planted_truth(sp_.meta["planted"], "proximity_de_joint", sp_.types)
    genes = {g for rec in sp_.meta["planted"] for g in rec.get("genes", [])}
    res = {}
    for q in [Q.ProximityDE(n_genes=40, aggregate="pooled"),
              Q.ProximityDEJoint(n_genes=40, aggregate="pooled", sigma=20)]:
        q.prepare(sp_)
        c = q.run(sp_, sp_.labels)
        called = c[c.decision != 0].claim.str.split("|", expand=True)
        mis = int(((called[0] == "A") & (called[1] != "Inf") & called[2].isin(genes)).sum()) if len(called) else 0
        res[q.name] = (compare.versus_truth(c, truth, True), mis)
    assert res["proximity_de_joint"][0]["power"] >= 0.6
    assert res["proximity_de_joint"][1] <= res["proximity_de"][1]


def test_spike_marker_rules(tissue):
    n = plant.candidate_counts(tissue, "Inf", "A")
    assert n["none"] >= max(n["all"], n["pair"])
    t, rec = plant.spike(tissue, "Inf", "A", 5, effect=1.0, lam=15, rng=np.random.default_rng(3),
                         marker_rule="pair")
    assert rec["marker_rule"] == "pair"
    M = {ty: np.asarray(tissue.counts[tissue.labels == ty].mean(0)).ravel() for ty in tissue.types}
    gi = [int(g[1:]) for g in rec["genes"]]
    r = (M["Inf"][gi] + 1e-3) / (M["A"][gi] + 1e-3)
    assert ((r > 0.5) & (r < 2)).all()           # no sender markers
    with pytest.raises(ValueError, match="candidates by rule"):
        plant.spike(tissue, "Inf", "A", 10_000, effect=1.0, lam=15, rng=np.random.default_rng(3))


def test_marginal_kernel_equals_joint_with_one_neighbour(tissue):
    """With one neighbor type and the same kernel features, marginal and joint fits coincide; with all types,
    the joint model reports collinearity diagnostics."""
    kw = dict(n_genes=30, sigma=20, k=30, neighbour_types=["Inf"], index_types=["A"])
    a, b = Q.ProximityDE(**kw), Q.ProximityDEJoint(**kw)
    a.prepare(tissue); b.prepare(tissue)
    ca, cb = a.run(tissue, tissue.labels), b.run(tissue, tissue.labels)
    assert np.allclose(ca.stat.to_numpy(), cb.stat.to_numpy(), rtol=1e-4, atol=1e-8)
    assert np.allclose(ca.effect.to_numpy(), cb.effect.to_numpy(), rtol=1e-4, atol=1e-8)
    j = Q.ProximityDEJoint(n_genes=30, sigma=20, index_types=["A"])
    j.prepare(tissue); j.run(tissue, tissue.labels)
    D = j.diagnostics
    assert set(D.neighbour) == set(tissue.types) and (D.vif >= 0.99).all()


def _noisy(tissue, eps=0.15, seed=1):
    from claimbreak import perturb
    C = perturb.uniform_confusion(tissue.types)
    lab, _ = perturb.label_errors(tissue, eps, C, np.random.default_rng(seed))
    return lab


def _cctx(tissue):
    return {"types": tissue.types, "native_C": None, "native_type_flip": {}, "native_flip": 0.0,
            "native_by_cond": {}}


def test_simex_and_ppi_identities(tissue):
    """SIMEX with an oracle matrix returns the naive claims at zero error; PPI with every cell revealed returns
    the analysis on the true labels, whatever the observed labels."""
    from claimbreak import correct as CR
    q = Q.ProximityDE(n_genes=20, index_types=["A"])
    q.prepare(tissue)
    naive = q.run(tissue, tissue.labels)
    s = CR.make({"name": "simex", "pi": "oracle", "B": 1}).apply(q, tissue, tissue.labels, naive, _cctx(tissue),
                                                                  np.random.default_rng(0))
    assert np.allclose(s.stat, naive.stat) and (s.decision == naive.decision).all()
    lab = _noisy(tissue)
    noisy = q.run(tissue, lab)
    p = CR.make({"name": "ppi", "fraction": 1.0, "patch": 1e9}).apply(q, tissue, lab, noisy, _cctx(tissue),
                                                                     np.random.default_rng(0))
    m = naive.merge(p, on="claim", suffixes=("_t", "_p"))
    assert len(m) == len(naive) and np.allclose(m.stat_t, m.stat_p, atol=1e-10)


def test_simex_removes_false_calls(tissue):
    """The synthetic tissue has no neighborhood effects: label error creates calls, and SIMEX with an oracle
    matrix removes most of them. (Whether by removing bias or by widening variance is measured on planted truth.)"""
    from claimbreak import correct as CR
    q = Q.ProximityDE(n_genes=20, index_types=["A", "B"])
    q.prepare(tissue)
    lab = _noisy(tissue, 0.2)
    noisy = q.run(tissue, lab)
    s = CR.make({"name": "simex", "pi": "oracle", "B": 2}).apply(q, tissue, lab, noisy, _cctx(tissue),
                                                                  np.random.default_rng(0))
    assert (noisy.decision != 0).sum() >= 20
    assert (s.decision != 0).sum() <= 0.2 * (noisy.decision != 0).sum()


def test_twode_removes_neighbour_marker_calls(tissue):
    """A gene planted in A near Inf that is an Inf marker is removed by 2DE; non-marker planted genes are kept."""
    from claimbreak import correct as CR
    tt, rs = plant.spike(tissue, "Inf", "A", 6, effect=2.0, lam=20, rng=np.random.default_rng(5),
                         marker_rule="sender")
    tt, rp = plant.spike(tt, "Inf", "A", 6, effect=2.0, lam=20, rng=np.random.default_rng(6), marker_rule="pair")
    q = Q.ProximityDE(n_genes=100, index_types=["A"], neighbour_types=["Inf"], aggregate="pooled")
    q.prepare(tt)
    cl = q.run(tt, tt.labels)
    out = CR.make({"name": "twode"}).apply(q, tt, tt.labels, cl, _cctx(tt), np.random.default_rng(0))
    called = lambda d, genes: d[(d.decision > 0) & d.claim.isin([f"A|Inf|{g}" for g in genes])].claim.nunique()
    assert called(cl, rs["genes"]) >= 3
    assert called(out, rs["genes"]) == 0                       # sender markers removed (power cost)
    assert called(out, rp["genes"]) >= 1                       # non-marker genes kept (p_g sits near tau = 0.9)


def test_margin_errors_fall_on_ambiguous_cells(tissue):
    from claimbreak import perturb
    from claimbreak.labeling import NearestCentroid, centroid_scores
    lab = NearestCentroid(tissue)
    sc = centroid_scores(lab, tissue.counts, tissue.section)
    P, q = perturb.margin_probabilities(sc, tissue.labels, tissue.types, 0.1)
    assert abs(q.mean() - 0.1) < 0.01                                  # temperature calibrated to the flip rate
    new, info = perturb.margin_errors(tissue.labels, P, q, tissue.types, 0.1, np.random.default_rng(0))
    flipped = new != tissue.labels
    assert abs(info["realised_eps"] - 0.1) < 0.02
    assert q[flipped].mean() > 1.5 * q.mean()                          # errors concentrate on ambiguous cells


def test_patch_errors_are_clustered(tissue):
    from claimbreak import perturb
    C = perturb.uniform_confusion(tissue.types)
    new, info = perturb.patch_errors(tissue, 0.1, C, np.random.default_rng(0), size=100, within=0.5)
    assert 0.05 < info["realised_eps"] < 0.15
    flipped = new != tissue.labels
    idx, _ = tissue.graph.knn(10)
    near = flipped[idx].mean(1)
    assert near[flipped].mean() > 3 * flipped.mean()                  # flipped cells sit among flipped cells


def test_threshold_free_metrics():
    from claimbreak import compare
    planted = [{"kind": "spike", "sender": "B", "receiver": "A", "genes": ["g1", "g2"], "sign": 1}]
    claims = ["A|B|g1", "A|B|g2", "A|B|g3", "C|B|g1", "C|B|g2", "C|B|g3"]
    run = pd.DataFrame({"claim": claims, "stat": [1, 1, 0.1, 0.1, -0.1, 0.1], "p": [1e-6, 1e-4, .5, .5, .02, .9],
                        "effect": [1, 1, .1, .1, -.1, .1], "decision": [1, 1, 0, 0, 0, 0]})
    m = compare.threshold_free(run, run, planted, True)
    assert m["auprc"] == 1.0 and m["power_p3"] == 1.0 and m["power_p5"] == 0.5 and m["fp_p3"] == 0
    # null class: every claim except planted genes in the receiver (A|B|g3 is null too); the narrow definition kept
    assert m["n_null"] == 4 and abs(m["null_typeI_05"] - 1 / 4) < 1e-9 and m["planted_bias"] == 1.0
    assert abs(m["null_typeI_05_nonrecv"] - 1 / 3) < 1e-9 and m["recv_nonplanted_typeI_05"] == 0.0
    C = compare.claim_classes(claims, planted)
    assert list(C.cls) == ["planted", "planted", "null_claim", "null_claim", "null_claim", "null_claim"] and C.recv_nonplanted.sum() == 1


def test_ppi_index_only_and_joint_correctors(tissue):
    from claimbreak import correct as CR
    lab = _noisy(tissue, 0.15)
    q = Q.ProximityDE(n_genes=20, index_types=["A"])
    q.prepare(tissue)
    # index-only PPI with every cell revealed = own labels true, neighbor labels observed
    p = CR.make({"name": "ppi", "fraction": 1.0, "reveal": "index"}).apply(q, tissue, lab, q.run(tissue, lab),
                                                                          _cctx(tissue), np.random.default_rng(0))
    sl, sc = q.slopes(tissue, tissue.labels, nb_labels=lab)
    ref = q.claims_from(sl, sc, n_units_total=len(pd.unique(tissue.unit)))
    m = ref.merge(p, on="claim")
    assert len(m) == len(ref) and np.allclose(m.stat_x, m.stat_y, atol=1e-7)
    # joint model: slopes() reproduces run() under the units test; oracle SIMEX is the identity at zero error
    j = Q.ProximityDEJoint(n_genes=20, index_types=["A"], sigma=20, k=30)
    j.prepare(tissue)
    jr = j.run(tissue, tissue.labels)
    sl, sc = j.slopes(tissue, tissue.labels)
    jc = j.claims_from(sl, sc, n_units_total=len(pd.unique(tissue.unit)))
    mm = jr.merge(jc, on="claim")
    assert len(mm) == len(jr) and np.allclose(mm.stat_x, mm.stat_y) and np.allclose(mm.p_x, mm.p_y)
    s = CR.make({"name": "simex", "pi": "oracle", "B": 1}).apply(j, tissue, tissue.labels, jr, _cctx(tissue),
                                                                  np.random.default_rng(0))
    assert np.allclose(s.stat, jr.stat)


def test_native_simex_rate_options(tissue):
    from claimbreak import correct as CR
    c = CR.make({"name": "simex", "pi": "native", "rate": "disagreement", "rate_scale": 1.25, "then": {"name": "twode"}})
    assert c.label == "simex_native_dis_x1.25+twode"
    ctx = {"types": tissue.types, "native_C": perturb_conf(tissue), "native_type_flip": {}, "native_flip": 0.1,
           "native_disagreement": 0.2, "native_by_cond": {}}
    P1, _ = CR.make({"name": "simex", "pi": "native"}).matrices(tissue, tissue.labels, ctx)
    P2, _ = c.matrices(tissue, tissue.labels, ctx)
    d1 = 1 - np.diag(list(P1.values())[0]); d2 = 1 - np.diag(list(P2.values())[0])
    assert np.allclose(d2, np.minimum(d1 * 2 * 1.25, 0.95))


def perturb_conf(tissue):
    from claimbreak import perturb
    return perturb.uniform_confusion(tissue.types)


def test_margin_errors_reach_target_rate(tissue):
    from claimbreak import perturb
    from claimbreak.labeling import NearestCentroid, centroid_scores
    sc = centroid_scores(NearestCentroid(tissue), tissue.counts, tissue.section)
    P, q = perturb.margin_probabilities(sc, tissue.labels, tissue.types, 0.1)
    for eps in (0.2, 0.3):
        _, info = perturb.margin_errors(tissue.labels, P, q, tissue.types, eps, np.random.default_rng(1))
        assert abs(info["realised_eps"] - eps) < 0.02


def test_ppi_classical_and_noisy_validation(tissue):
    from claimbreak import correct as CR, perturb
    from claimbreak.labeling import NearestCentroid, centroid_scores
    lab = _noisy(tissue, 0.15)
    q = Q.ProximityDE(n_genes=20, index_types=["A"])
    q.prepare(tissue)
    ctx = _cctx(tissue)
    sc = centroid_scores(NearestCentroid(tissue), tissue.counts, tissue.section)
    Pq = perturb.margin_probabilities(sc, tissue.labels, tissue.types, 0.1)
    ctx["margin"] = lambda: Pq
    c = CR.make({"name": "ppi", "fraction": 1.0, "reveal": "index", "mode": "classical"})
    assert c.label == "cls_idx_100"
    out = c.apply(q, tissue, lab, q.run(tissue, lab), ctx, np.random.default_rng(0))
    sl, s2 = q.slopes(tissue, tissue.labels, nb_labels=lab)
    ref = q.claims_from(sl, s2, n_units_total=len(pd.unique(tissue.unit)))
    m = ref.merge(out, on="claim")
    assert len(m) == len(ref) and np.allclose(m.stat_x, m.stat_y)       # all validated: analysis on validator labels
    v = CR.make({"name": "ppi", "fraction": 0.01, "reveal": "index", "validation_error": 0.03})
    assert v.label == "ppi_idx_1_ve3"
    vl = v.validation_labels(tissue, ctx, np.random.default_rng(0))
    assert abs((vl != tissue.labels).mean() - 0.03) < 0.01


def test_coverage_and_universe():
    from claimbreak import compare
    planted = [{"kind": "spike", "sender": "B", "receiver": "A", "genes": ["g1"], "sign": 1}]
    base = pd.DataFrame({"claim": ["A|B|g1", "A|C|g1", "D|B|g2"], "stat": [1.0, 0.2, 0.0]})
    run = pd.DataFrame({"claim": ["A|B|g1", "A|C|g1"], "stat": [1.0, 1.0], "p": [1e-6, 1e-4],
                        "decision": [1, 1], "effect": [1.0, 1.0]})
    c = compare.coverage(run, base, planted, df=11)
    assert c["cover_all"] == 0.5 and c["misattr_typeI_05"] == 1.0 and c["n_misattr"] == 1
    u = compare.on_universe(run, base.claim)
    assert len(u) == 3 and u.set_index("claim").loc["D|B|g2", "p"] == 1.0 and u.estimable.sum() == 2


def test_spike_set_plants_disjoint_pairs(tissue):
    t, rec = plant.spike_set(tissue, n_pairs=6, genes=3, effect=1.0, lam=20, rng=np.random.default_rng(0),
                             min_cells=500, max_per_receiver=2)
    sp = [r for r in t.meta["planted"] if r["kind"] == "spike"]
    assert len(sp) == 6 and len({(r["sender"], r["receiver"]) for r in sp}) == 6
    for r_ in {r["receiver"] for r in sp}:
        g = [g for r in sp if r["receiver"] == r_ for g in r["genes"]]
        assert len(g) == len(set(g))                                   # disjoint genes per receiver


def test_ppi_sampling_designs(tissue):
    from claimbreak import correct as CR, perturb
    from claimbreak.labeling import NearestCentroid, centroid_scores
    lab = _noisy(tissue, 0.15)
    ctx = _cctx(tissue)
    sc = centroid_scores(NearestCentroid(tissue), tissue.counts, tissue.section)
    Pq = perturb.margin_probabilities(sc, tissue.labels, tissue.types, 0.1)
    ctx["margin"] = lambda: Pq
    ctx["sample_seed"] = 7
    for samp in ("uniform", "stratified", "uncertainty"):
        c = CR.make({"name": "ppi", "reveal": "index", "fraction": 0.1, "sampling": samp})
        pi = c.inclusion(tissue, lab, ctx)
        assert abs(pi.mean() - 0.1) < 1e-3 and (pi > 0).all() and (pi <= 1).all()
    unc = CR.make({"name": "ppi", "reveal": "index", "fraction": 0.1, "sampling": "uncertainty"})
    pi = unc.inclusion(tissue, lab, ctx)
    wrong = lab != tissue.labels
    assert pi[wrong].mean() > 1.5 * pi[~wrong].mean()                  # wrong labels oversampled
    # nested samples across budgets on the common stream
    r1 = CR.make({"name": "ppi", "reveal": "index", "fraction": 0.05}).revealed(tissue, None, lab, ctx)
    r2 = CR.make({"name": "ppi", "reveal": "index", "fraction": 0.10}).revealed(tissue, None, lab, ctx)
    assert (r1 <= r2).all()
    # weighted PPI with everything revealed (pi = 1) equals the analysis on validator labels
    q = Q.ProximityDE(n_genes=20, index_types=["A"])
    q.prepare(tissue)
    full = CR.make({"name": "ppi", "reveal": "index", "fraction": 1.0, "sampling": "uncertainty"})
    out = full.apply(q, tissue, lab, q.run(tissue, lab), ctx, np.random.default_rng(0))
    sl, s2 = q.slopes(tissue, tissue.labels, nb_labels=lab)
    ref = q.claims_from(sl, s2, n_units_total=len(pd.unique(tissue.unit)))
    m = ref.merge(out, on="claim")
    assert np.allclose(m.stat_x, m.stat_y)
    # copy validator: with copy 1.0 and everything revealed, PPI returns the naive claims
    cp = CR.make({"name": "ppi", "reveal": "index", "fraction": 1.0, "validation_copy": 1.0})
    assert cp.label == "ppi_idx_100_cp100"
    o2 = cp.apply(q, tissue, lab, q.run(tissue, lab), ctx, np.random.default_rng(0))
    m2 = q.run(tissue, lab).merge(o2, on="claim")
    assert np.allclose(m2.stat_x, m2.stat_y)


def test_simex_margin_runs_and_is_identity_without_error(tissue):
    from claimbreak import correct as CR, perturb
    from claimbreak.labeling import NearestCentroid, centroid_scores
    ctx = _cctx(tissue)
    sc = centroid_scores(NearestCentroid(tissue), tissue.counts, tissue.section)
    Pq = perturb.margin_probabilities(sc, tissue.labels, tissue.types, 0.1)
    ctx["margin"] = lambda: Pq
    q = Q.ProximityDE(n_genes=20, index_types=["A"])
    q.prepare(tissue)
    c = CR.make({"name": "simex", "pi": "oracle", "mechanism": "margin", "B": 1})
    assert c.label == "simex_oracle_margin"
    naive = q.run(tissue, tissue.labels)
    s = c.apply(q, tissue, tissue.labels, naive, ctx, np.random.default_rng(0))
    assert np.allclose(s.stat, naive.stat)
    lab = _noisy(tissue, 0.15)
    s2 = c.apply(q, tissue, lab, q.run(tissue, lab), ctx, np.random.default_rng(0))
    assert len(s2) > 0 and s2.p.notna().all()



def test_additive_ppi_is_unbiased_under_uniform_and_targeted_sampling(tissue):
    """Over repeated validation draws, the additive PPI estimate of a planted slope is centerd on the slope with
    reference own labels (index-only target), for uniform and uncertainty-targeted (IPW) sampling, while the naive
    estimate is attenuated."""
    from claimbreak import correct as CR, perturb
    from claimbreak.labeling import NearestCentroid, centroid_scores
    tt, rec = plant.spike(tissue, "Inf", "A", 5, effect=2.0, lam=20, rng=np.random.default_rng(5))
    sc = centroid_scores(NearestCentroid(tt), tt.counts, tt.section)
    Pq = perturb.margin_probabilities(sc, tt.labels, tt.types, 0.1)
    lab, _ = perturb.margin_errors(tt.labels, *Pq, tt.types, 0.25, np.random.default_rng(2))
    q = Q.ProximityDE(n_genes=20, index_types=["A"], neighbour_types=["Inf"])
    q.prepare(tt)
    sl, sc2 = q.slopes(tt, tt.labels, nb_labels=lab)
    target = q.claims_from(sl, sc2, n_units_total=len(pd.unique(tt.unit))).set_index("claim").stat
    naive = q.run(tt, lab).set_index("claim").stat
    ids = [f"A|Inf|{g}" for g in rec["genes"]]
    ctx = _cctx(tt)
    ctx["margin"] = lambda: Pq
    for samp in ("uniform", "uncertainty"):
        est = []
        for d in range(12):
            ctx["sample_seed"] = 100 + d
            c = CR.make({"name": "ppi", "reveal": "index", "fraction": 0.3, "sampling": samp, "floor": 0.5,
                         "min_cells": 5})
            est.append(c.apply(q, tt, lab, q.run(tt, lab), ctx, np.random.default_rng(d)).set_index("claim").stat)
        m = pd.concat(est, axis=1).loc[ids].mean(1)
        assert np.abs(m - target[ids]).mean() < 0.5 * np.abs(naive[ids] - target[ids]).mean(), samp


def test_additive_ppi_special_cases(tissue):
    from claimbreak import correct as CR
    lab = _noisy(tissue, 0.15)
    q = Q.ProximityDE(n_genes=20, index_types=["A"])
    q.prepare(tissue)
    ctx = _cctx(tissue)
    ctx["sample_seed"] = 3
    naive = q.run(tissue, lab)
    # no error: with lam = 1 the rectifier is exactly zero and PPI equals naive; with tuned lam (< 1, about
    # 1 / (1 + n / N)) the estimate blends two unbiased sums and stays close to naive
    ref = q.run(tissue, tissue.labels)
    for lam, tol in ((1, 1e-7), ("tuned", None)):
        c = CR.make({"name": "ppi", "reveal": "index", "fraction": 0.5, "lam": lam, "min_cells": 5})
        o = c.apply(q, tissue, tissue.labels, ref, ctx, np.random.default_rng(0))
        m = ref.merge(o, on="claim")
        if tol:
            assert np.allclose(m.stat_x, m.stat_y, atol=tol)
        else:
            assert np.corrcoef(m.stat_x, m.stat_y)[0, 1] > 0.9 and 0.4 < o.ppi_lambda.iloc[0] < 1
    assert CR.make({"name": "ppi", "reveal": "index", "fraction": 0.5, "lam": 1}).label == "ppi_idx_50_lam1"
    assert CR.make({"name": "ppi", "reveal": "index", "fraction": 0.5, "estimator": "ratio"}).label == "ppi_idx_50_ratio"


def test_claims_from_drops_missing_units_per_gene(tissue):
    """A gene not estimable in some units is tested over the remaining units (not with zeros); with fewer than 3
    units left it is not testable (p NaN, no call)."""
    from scipy import stats as st
    q = Q.ProximityDE(n_genes=20, index_types=["A"], neighbour_types=["Inf"])
    q.prepare(tissue)
    sl, sc = q.slopes(tissue, tissue.labels)
    (key, (units, S, pooled)), = sl.items()
    S2 = S.copy()
    S2[0, 0] = np.nan                                   # gene 0 missing in unit 0
    S2[:-2, 1] = np.nan                                 # gene 1 left with 2 units
    out = q.claims_from({key: (units, S2, None)}, sc, n_units_total=len(units))
    ref = st.ttest_1samp(S[1:, 0], 0.0)
    assert np.isclose(out.p.iloc[0], ref.pvalue) and np.isclose(out.stat.iloc[0], S[1:, 0].mean())
    assert np.isnan(out.p.iloc[1]) and out.decision.iloc[1] == 0
    full = q.claims_from({key: (units, S, None)}, sc, n_units_total=len(units))
    assert np.allclose(full.p.iloc[2:], out.p.iloc[2:])  # other genes unchanged
