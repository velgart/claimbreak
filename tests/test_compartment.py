"""claimbreak.compartment reproduces the example scripts exactly, and its new options behave as stated."""
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from claimbreak import compartment as cm
from claimbreak.cli import main

EX = Path(__file__).resolve().parents[1] / "examples" / "compartment"
ARMS = "whole,nucleus,thinned,cyto,nucA,nucB,cytoB,wholeB,wB50,wBall,nB50"


def _make(root: Path, seed=7):
    """Reference (4 donors, Level2 with donor tumor types, Level3) and two sections whose cytoplasm carries
    transcripts of the touching neighbors."""
    rng = np.random.default_rng(seed)
    G = 80
    l3 = ["T_a", "T_b", "Mye", "Fib", "Tu_L1", "Tu_L2"]
    l2 = {"T_a": "T", "T_b": "T", "Mye": "Myeloid", "Fib": "Fibro", "Tu_L1": "Tu_L1", "Tu_L2": "Tu_L2"}
    prof = {t: rng.dirichlet(np.full(G, 1.5)) for t in l3}
    prof["Tu_L2"] = 0.7 * prof["Tu_L1"] + 0.3 * prof["Tu_L2"]
    lab = np.repeat(l3, 160)
    X = np.vstack([rng.multinomial(rng.integers(150, 400), prof[t]) for t in lab])
    obs = pd.DataFrame({"Level3": lab, "Level2": [l2[t] for t in lab], "donor_id": np.tile(list("abcd"), len(lab) // 4)},
                       index=[f"r{i}" for i in range(len(lab))])
    var = pd.DataFrame({"feature_name": [f"g{i}" for i in range(G)]}, index=[f"E{i}" for i in range(G)])
    ad.AnnData(sp.csr_matrix(X.astype(np.float32)), obs=obs, var=var).write_h5ad(root / "ref.h5ad")
    for s in range(2):
        n = 1200
        xy = rng.uniform(0, 400, (n, 2))
        zone = np.digitize(xy[:, 0], [130, 260])
        ct = np.array([rng.choice(["Tu_L1", "Fib", "Mye"]) if z == 0 else rng.choice(["T_a", "T_b", "Fib"])
                       if z == 1 else rng.choice(["Tu_L1", "T_a", "Mye"]) for z in zone])
        from scipy.spatial import cKDTree
        _, nn = cKDTree(xy).query(xy, k=4)
        nuc = np.vstack([rng.multinomial(rng.integers(12, 40), prof[t]) for t in ct])
        cyto = np.vstack([rng.multinomial(rng.integers(20, 80), 0.6 * prof[ct[i]] + 0.4 * prof[ct[nn[i, 1]]])
                          for i in range(n)])
        W = nuc + cyto
        genes = [f"g{i}" for i in range(G - 3)] + ["notinref1", "notinref2", "notinref3"]
        o = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "n_whole": W.sum(1), "n_nucleus": nuc.sum(1),
                          "donor": f"D{s}", "panel": "synth panel", "section": f"S{s}"},
                         index=[f"c{s}_{i}" for i in range(n)])
        A = ad.AnnData(sp.csr_matrix(W.astype(np.float32)), obs=o, var=pd.DataFrame(index=genes))
        A.layers["nucleus"] = sp.csr_matrix(nuc.astype(np.float32))
        A.layers["cyto"] = sp.csr_matrix(cyto.astype(np.float32))
        A.write_h5ad(root / f"sec{s}.h5ad")
    return root


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return _make(tmp_path_factory.mktemp("compartment"))


def _script(name, *args):
    r = subprocess.run([sys.executable, str(EX / name), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def _same_tables(d1: Path, d2: Path, pattern: str):
    f1 = sorted(p.name for p in d1.glob(pattern))
    assert f1 and f1 == sorted(p.name for p in d2.glob(pattern)), (f1, list(d2.glob(pattern)))
    for f in f1:
        a = pd.read_csv(d1 / f, keep_default_na=False, low_memory=False)
        b = pd.read_csv(d2 / f, keep_default_na=False, low_memory=False)
        b = b[[c for c in a.columns if c in b.columns]] if set(a.columns) <= set(b.columns) else b
        # the package may add columns (compare: the regionally matched expectation); the script's columns
        # must match exactly
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=f)


@pytest.fixture(scope="module")
def labelled(data):
    common = ["--ref", data / "ref.h5ad", "--inputs", data / "sec0.h5ad", data / "sec1.h5ad", "--arms", ARMS,
              "--ref-cv", "--cv-folds", "2", "--cv-per-type", "60", "--cv-depths", "20,60", "--seed", "3"]
    _script("compartment_label.py", *common, "--out", data / "lab_script")
    assert main(["compartment", "label", *map(str, common), "--out", str(data / "lab_pkg")]) == 0
    return data / "lab_script", data / "lab_pkg"


def test_label_matches_script(labelled):
    s, p = labelled
    _same_tables(s, p, "*.labels.csv.gz")
    _same_tables(s, p, "*.gamma.csv")
    _same_tables(s, p, "ref_cv_*.csv")
    lab = pd.read_csv(p / "sec0.labels.csv.gz", keep_default_na=False)
    assert set(lab["L2m_whole"]) - {""} <= {"T", "Myeloid", "Fibro", "Tumour"}      # Tu_L1/Tu_L2 merged


def test_label_meta_records_composition_and_settings(labelled):
    import json
    m = json.loads((labelled[1] / "sec0.meta.json").read_text())
    assert m["genes_used"] == 77 and len(m["genes_missing"]) == 3
    assert m["gamma_fit"] == "hard" and m["gamma_clip"] == [0.05, 20.0] and m["seed"] == 3
    assert abs(sum(m["composition"]["L2m_whole"].values()) - 1) < 1e-6
    assert "L2m:nucB==nucA" in m["agreement"]


def test_compare_matches_script(labelled):
    s, p = labelled
    args = ["--levels", "L2m,L3m", "--comparisons", "nucB|nucA,cytoB|nucA,wholeB|nucA,wB50|nucA,whole|nucleus",
            "--common-cells", "nucB,wholeB"]
    _script("bilous_compare.py", "--labels", s, "--out", s / "cmp", *args)
    assert main(["compartment", "compare", "--labels", str(p), "--out", str(p / "cmp"), *args]) == 0
    for f in ("pairs.csv", "summary.csv", "did.csv", "did_pairs.csv"):
        _same_tables(s / "cmp", p / "cmp", f)
    S = pd.read_csv(p / "cmp" / "summary.csv")
    assert S.loc[S.comparison == "wholeB|nucA", "ratio_contact"].notna().all()


def test_consequence_matches_script(labelled):
    s, p = labelled
    args = ["--level", "L2m", "--ref", "nucB", "--test", "wholeB", "--draws", "5", "--min-cells", "50", "--seed", "1"]
    _script("bilous_consequence.py", "--labels", s, "--out", s / "csq", *args)
    assert main(["compartment", "consequence", "--labels", str(p), "--out", str(p / "csq"), *args]) == 0
    _same_tables(s / "csq", p / "csq", "*.csv")


def test_refuses_to_overwrite(labelled):
    p = labelled[1]
    assert main(["compartment", "compare", "--labels", str(p), "--out", str(p / "cmp"), "--levels", "L2m"]) == 1
    with pytest.raises(FileExistsError):
        cm.consequence(p, p / "csq", "L2m")


def test_classifier_options():
    rng = np.random.default_rng(0)
    mu = rng.dirichlet(np.full(30, 0.3), size=3)
    lab = rng.integers(0, 3, 600)
    gamma_true = np.exp(rng.normal(0, 0.5, 30))
    q = mu * gamma_true
    q /= q.sum(1, keepdims=True)
    X = sp.csr_matrix(np.vstack([rng.multinomial(40, q[k]) for k in lab]).astype(np.float32))
    base = cm.classify(X, mu)
    l0, _, _, _, g0 = cm.classify(X, mu, cm.LabelConfig(gamma_iter=0))
    assert np.all(g0 == 1)
    ls, _, _, _, gs = cm.classify(X, mu, cm.LabelConfig(gamma_fit="soft"))
    ld, _, _, _, gd = cm.classify(X, mu, cm.LabelConfig(gamma_damp=0.5, gamma_clip=(0.2, 5)))
    assert gd.min() >= 0.2 - 1e-12 and gd.max() <= 5 + 1e-12
    for L in (base[0], l0, ls, ld):
        assert np.mean(L == lab) > 0.8
    with pytest.raises(ValueError):
        cm.classify(X, mu, cm.LabelConfig(gamma_fit="other"))


def test_collapse_warning(data):
    ref = cm.Reference.load(data / "ref.h5ad", {"L2m": "Level2"}, merge=r"^Tu_L\d=>Tumour", verbose=False)
    A = ad.read_h5ad(data / "sec0.h5ad")
    _, _, info = cm.label_section(A, ref, ["L2m"], ["whole", "nucleus"], cm.LabelConfig(collapse_warn=0.99))
    assert info["warnings"] == []
    _, _, info = cm.label_section(A, ref, ["L2m"], ["whole"], cm.LabelConfig(collapse_warn=0.05))
    assert info["warnings"] and "collapse" in info["warnings"][0]


def _collapse_case(boost, seed=5):
    """Three real types with 15 weak markers each; a 'Broad' type whose profile is the mean of the three (ambient)
    with five probe genes raised `boost`-fold; tissue cells of the real types in which those probes are counted 30
    times too often (off-target probes)."""
    rng = np.random.default_rng(seed)
    G = 60
    real = []
    for k in range(3):
        p = np.full(G, 0.2)
        p[k * 15:(k + 1) * 15] += 1.0
        real.append(p / p.sum())
    probes = np.arange(G - 5, G)
    broad = np.mean(real, 0)
    broad[probes] *= boost
    types = np.array(["a", "b", "c", "Broad"])
    mu = np.vstack(real + [broad / broad.sum()])
    lab = rng.integers(0, 3, 900)
    q = mu[lab].copy()
    q[:, probes] *= 30
    q /= q.sum(1, keepdims=True)
    X = sp.csr_matrix(np.vstack([rng.multinomial(60, qi) for qi in q]).astype(np.float32))
    genes = [f"g{i}" for i in range(G)]
    ref = cm.Reference(sp.csr_matrix(np.ones((4, G))), genes, {"L": types}, np.array(list("wxyz")))
    ref._mu[("L", tuple(range(G)))] = (types, mu)
    return ref, mu, X, lab, [genes[j] for j in probes]


def _label(ref, mu, X, genes, cfg):
    keep = np.array([g not in set(cfg.exclude_genes) for g in genes])
    gidx = np.arange(len(genes))[keep]
    if cfg.markers:
        sel, _ = ref.markers("L", np.arange(len(genes)), cfg.markers, cfg.marker_min, cfg.marker_min_log2)
        sel = np.array([j for j in sel if keep[j]])
    else:
        sel = gidx
    m = mu[:, sel] / mu[:, sel].sum(1, keepdims=True)
    return cm.classify(X[:, sel], m, cfg)[0]


@pytest.mark.parametrize("boost", [1.5, 4.0])
def test_off_target_probes_collapse_labels_and_the_remedies(boost):
    ref, mu, X, lab, probes = _collapse_case(boost)
    genes = list(ref.symbols)
    base = cm.LabelConfig(gamma_iter=0)
    assert np.mean(_label(ref, mu, X, genes, base) == 3) > 0.6                        # collapse onto Broad
    mk = _label(ref, mu, X, genes, cm.LabelConfig(gamma_iter=0, markers=10))
    if boost < 2:          # probes below the 2-fold specificity bar are not markers: markers alone suffice
        assert np.mean(mk == 3) < 0.1 and np.mean(mk == lab) > 0.8
    else:                  # probes clearly 'specific' to Broad become its markers: markers alone fail
        assert np.mean(mk == 3) > 0.5
    both = _label(ref, mu, X, genes, cm.LabelConfig(gamma_iter=0, markers=10, exclude_genes=tuple(probes)))
    assert np.mean(both == 3) < 0.1 and np.mean(both == lab) > 0.8


def test_label_with_markers_and_exclusions(data):
    ref = cm.Reference.load(data / "ref.h5ad", {"L2m": "Level2"}, merge=r"^Tu_L\d=>Tumour", verbose=False)
    A = ad.read_h5ad(data / "sec0.h5ad")
    cfg = cm.LabelConfig(markers=8, exclude_genes=("g0", "g1"))
    res, gam, info = cm.label_section(A, ref, ["L2m"], ["whole", "nucA", "nucB"], cfg)
    assert info["genes_excluded"] == ["g0", "g1"] and "g0" not in set(gam.gene)
    mk = info["markers"]["L2m"]
    assert mk["n_genes"] == sum(mk["per_type"].values()) and all(v <= 8 for v in mk["per_type"].values())
    assert gam["L2m_whole"].isna().sum() == len(gam) - mk["n_genes"]       # gamma only on the genes used
    assert (res["L2m_whole"] != "").mean() > 0.5
    cv, conf, K = ref.cross_validate(panel_genes_idx(ref, A, cfg), "p", ["L2m"], cfg, folds=2, per_type=50,
                                     depths=(30,))
    assert len(cv) and cv.accuracy.mean() > 0.5


def panel_genes_idx(ref, A, cfg):
    return cm.panel_genes(ref, A.var_names, cfg)[1]


def test_summary(labelled, tmp_path):
    p = labelled[1]
    assert main(["compartment", "summary", "--compare", str(p / "cmp"), "--consequence", str(p / "csq"),
                 "--out", str(tmp_path / "sum"), "--level", "L2m", "--min-null-cells", "10", "--min-stratum", "5",
                 "--unit-pattern", "sec\\d"]) == 0
    S = pd.read_csv(p / "cmp" / "summary.csv")
    sec = pd.read_csv(tmp_path / "sum" / "sections.csv")
    r = sec.iloc[0]
    t = S[(S["sample"] == r["sample"]) & (S.level == "L2m") & (S.comparison == "wholeB|nucA")].iloc[0]
    n0 = S[(S["sample"] == r["sample"]) & (S.level == "L2m") & (S.comparison == "nucB|nucA")].iloc[0]
    want = (t.obs_contact * t.change_rate - n0.obs_contact * n0.change_rate) / \
           (t.exp_contact * t.change_rate - n0.exp_contact * n0.change_rate)
    assert np.isclose(r.extra_ratio_contact, want)
    assert set(sec.unit) == {"sec0", "sec1"}
    pr = pd.read_csv(tmp_path / "sum" / "pairs.csv")
    assert {"excess_median", "reverse_excess", "sections_pos"} <= set(pr.columns) and len(pr)
    cp = pd.read_csv(tmp_path / "sum" / "coloc_pairs.csv")
    assert {"d_test", "d_type", "units_test_below"} <= set(cp.columns)


def test_summary_other_reference_arm(labelled, tmp_path):
    """A test with another reference arm than the null (whole|nucleus against nucB|nucA): pairs are the test's own
    contrast, not an empty table."""
    p = labelled[1]
    assert main(["compartment", "summary", "--compare", str(p / "cmp"), "--out", str(tmp_path / "s2"), "--level", "L2m",
                 "--test", "whole|nucleus", "--min-null-cells", "10", "--min-stratum", "5", "--unit-pattern", "sec\\d"]) == 0
    pr = pd.read_csv(tmp_path / "s2" / "pairs.csv")
    assert len(pr) and pr.null_contrast.isna().all()
    sec = pd.read_csv(tmp_path / "s2" / "sections.csv")
    assert "test_changes_more" in sec


def test_fixed_platform_factor_reproduces_labels(data):
    """Labeling with the whole arm's platform factor from a run reproduces that run's labels without refitting."""
    ref = cm.Reference.load(data / "ref.h5ad", {"L2m": "Level2"}, merge=r"^Tu_L\d=>Tumour", verbose=False)
    A = ad.read_h5ad(data / "sec0.h5ad")
    res, gam, _ = cm.label_section(A, ref, ["L2m"], ["whole"])
    g = dict(zip(gam.gene, gam["L2m_whole"]))
    res2, gam2, _ = cm.label_section(A, ref, ["L2m"], ["whole"], gamma_fixed={"L2m_whole": g})
    assert (res2["L2m_whole"] == res["L2m_whole"]).mean() > 0.999
    assert np.allclose(gam2["L2m_whole"], gam["L2m_whole"], atol=1e-3)


def test_summary_pooled_unit_ratio(labelled, tmp_path):
    """per unit, the extra-change ratio pooled over sections (sum of numerators over sum of denominators, in
    cells) with a bootstrap interval; sections with few extra changes are flagged, not dropped silently."""
    p = labelled[1]
    assert main(["compartment", "summary", "--compare", str(p / "cmp"), "--out", str(tmp_path / "sp"), "--level", "L2m",
                 "--min-null-cells", "10", "--min-stratum", "5", "--unit-pattern", "sec\\d", "--draws", "400",
                 "--min-extra", "5"]) == 0
    S = pd.read_csv(p / "cmp" / "summary.csv")
    sec = pd.read_csv(tmp_path / "sp" / "sections.csv")
    U = pd.read_csv(tmp_path / "sp" / "units.csv")
    assert {"extra_changes", "extra_ok"} <= set(sec.columns)
    assert {"pooled_contact", "pooled_contact_lo", "pooled_contact_hi", "evaluable"} <= set(U.columns)
    for _, u in U.iterrows():
        g = sec[(sec.unit == u.unit) & sec.used]
        num = den = 0.0
        for smp in g["sample"]:
            t = S[(S["sample"] == smp) & (S.level == "L2m") & (S.comparison == "wholeB|nucA")].iloc[0]
            n0 = S[(S["sample"] == smp) & (S.level == "L2m") & (S.comparison == "nucB|nucA")].iloc[0]
            n = min(t.cells, n0.cells)
            num += (t.obs_contact * t.change_rate - n0.obs_contact * n0.change_rate) * n
            den += (t.exp_contact * t.change_rate - n0.exp_contact * n0.change_rate) * n
        assert np.isclose(u.pooled_contact, num / den)
        if np.isfinite(u.pooled_contact_lo):
            assert u.pooled_contact_lo <= u.pooled_contact <= u.pooled_contact_hi
    # one section per unit here: the pooled ratio equals the section's ratio
    m = sec[sec.used].groupby("unit").extra_ratio_contact.first()
    assert np.allclose(U.set_index("unit").pooled_contact.loc[m.index], m)
    # a high threshold marks every section and unit as not evaluable
    assert main(["compartment", "summary", "--compare", str(p / "cmp"), "--out", str(tmp_path / "sp2"), "--level",
                 "L2m", "--min-null-cells", "10", "--min-stratum", "5", "--unit-pattern", "sec\\d", "--draws", "0",
                 "--min-extra", "1000000"]) == 0
    U2 = pd.read_csv(tmp_path / "sp2" / "units.csv")
    assert not U2.evaluable.any() and not pd.read_csv(tmp_path / "sp2" / "sections.csv").extra_ok.any()


def test_summary_unstable_denominator_not_evaluable(labelled, tmp_path):
    """a unit whose denominator is not stable (here: forced by a huge min-extra / 4 threshold without draws)
    is not evaluable even with many extra changes; with draws, evaluable units have a finite interval."""
    p = labelled[1]
    assert main(["compartment", "summary", "--compare", str(p / "cmp"), "--out", str(tmp_path / "u1"), "--level", "L2m",
                 "--min-null-cells", "10", "--min-stratum", "5", "--unit-pattern", "sec\\d", "--draws", "300",
                 "--min-extra", "1"]) == 0
    U = pd.read_csv(tmp_path / "u1" / "units.csv")
    assert {"expected_extra_contact", "ci_excludes_1"} <= set(U.columns)
    ev = U[U.evaluable]
    assert np.isfinite(ev.pooled_contact_lo).all() and np.isfinite(ev.pooled_contact_hi).all()
    assert (ev.ci_excludes_1 == (ev.pooled_contact_lo > 1)).all()
    # without draws, the denominator must reach min_extra / 4 expected extra contact changes
    big = int(4 * U.expected_extra_contact.max()) + 4
    assert main(["compartment", "summary", "--compare", str(p / "cmp"), "--out", str(tmp_path / "u2"), "--level", "L2m",
                 "--min-null-cells", "10", "--min-stratum", "5", "--unit-pattern", "sec\\d", "--draws", "0",
                 "--min-extra", str(big)]) == 0
    assert not pd.read_csv(tmp_path / "u2" / "units.csv").evaluable.any()


# --------------------------------------------------------------------------- second labeler

def test_celltypist_labeller_agrees_with_multinomial(data):
    pytest.importorskip("celltypist")
    ref = cm.Reference.load(data / "ref.h5ad", {"L2m": "Level2"}, merge=r"^Tu_L\d=>Tumour", verbose=False)
    A = ad.read_h5ad(data / "sec0.h5ad")
    arms = ["whole", "nucleus", "nucA", "nucB", "wholeB"]
    ct = cm.LabelConfig(labeller="celltypist", ct_per_type=150, ct_train_depths=(20, 60), ct_max_iter=300)
    r_ct, g_ct, i_ct = cm.label_section(A, ref, ["L2m"], arms, ct, seed=3)
    r_mn, _, _ = cm.label_section(A, ref, ["L2m"], arms, cm.LabelConfig(), seed=3)
    assert g_ct.drop(columns="gene").isna().all().all()                     # no platform factor
    for arm in arms:
        ok = r_ct[f"L2m_{arm}"] != ""
        assert (ok == (r_mn[f"L2m_{arm}"] != "")).all()                    # same min-counts rule
        assert ok.mean() > 0.5
        assert (r_ct.loc[ok, f"L2m_{arm}"] == r_mn.loc[ok, f"L2m_{arm}"]).mean() > 0.7, arm
        m = r_ct.loc[ok, f"L2m_{arm}_margin"]
        assert (m >= 0).all() and (r_ct.loc[ok, f"L2m_{arm}_second"] != r_ct.loc[ok, f"L2m_{arm}"]).all()
    assert set(r_ct["L2m_whole"]) - {""} <= {"T", "Myeloid", "Fibro", "Tumour"}
    assert "L2m:nucB==nucA" in i_ct["agreement"]
    # a model is trained once per level and gene set, and labeling is reproducible
    r2, _, _ = cm.label_section(A, ref, ["L2m"], arms, ct, seed=3)
    assert len(ref._ct) == 1
    pd.testing.assert_frame_equal(r_ct, r2)


def test_celltypist_markers_and_cross_validation(data):
    pytest.importorskip("celltypist")
    ref = cm.Reference.load(data / "ref.h5ad", {"L2m": "Level2"}, merge=r"^Tu_L\d=>Tumour", verbose=False)
    A = ad.read_h5ad(data / "sec1.h5ad")
    cfg = cm.LabelConfig(labeller="celltypist", markers=8, ct_per_type=100, ct_train_depths=(30,))
    res, _, info = cm.label_section(A, ref, ["L2m"], ["whole", "nucA", "nucB"], cfg)
    assert info["markers"]["L2m"]["n_genes"] > 0 and (res["L2m_whole"] != "").mean() > 0.5
    cv, conf, K = ref.cross_validate(panel_genes_idx(ref, A, cfg), "p", ["L2m"], cfg, folds=2, per_type=50,
                                     depths=(30,))
    assert K == 2 and len(cv) and cv.accuracy.mean() > 0.6


def test_celltypist_maps_features_by_name():
    """Training drops genes without counts; prediction must use the model's features by name, not by position."""
    pytest.importorskip("celltypist")
    rng = np.random.default_rng(0)
    G = 30
    prof = {t: rng.dirichlet(np.full(G, 1.0)) for t in "ab"}
    for t in prof:                                       # gene 0 never seen in training
        prof[t][0] = 0
        prof[t] /= prof[t].sum()
    lab = np.repeat(list("ab"), 200)
    M = sp.csr_matrix(np.vstack([rng.multinomial(100, prof[t]) for t in lab]).astype(np.float32))
    genes = [f"g{i}" for i in range(G)]
    cfg = cm.LabelConfig(labeller="celltypist", ct_train_depths=(), ct_per_type=0)
    model = cm.ct_fit(M, lab, genes, cfg)
    assert "g0" not in list(model.features) and len(model.features) == G - 1
    Q = sp.csr_matrix(np.vstack([rng.multinomial(60, prof[t]) for t in lab]).astype(np.float32))
    types, li, _, _, n = cm.ct_classify(Q, genes, model, cfg)
    assert (types[li] == lab).mean() > 0.95
    # shuffled query columns give the same labels when the names follow them
    perm = rng.permutation(G)
    types2, li2, _, _, _ = cm.ct_classify(Q[:, perm], [genes[j] for j in perm], model, cfg)
    assert (types2[li2] == types[li]).all()


def test_cli_celltypist_label_then_compare(data, tmp_path):
    pytest.importorskip("celltypist")
    out = tmp_path / "lab_ct"
    assert main(["compartment", "label", "--ref", str(data / "ref.h5ad"), "--inputs", str(data / "sec0.h5ad"),
                 "--out", str(out), "--ref-levels", "L2m=Level2", "--ref-merge", r"^Tu_L\d=>Tumour",
                 "--arms", "nucA,nucB,wholeB", "--labeller", "celltypist", "--ct-per-type", "100",
                 "--ct-train-depths", "20,60", "--seed", "3"]) == 0
    import json
    m = json.loads((out / "sec0.meta.json").read_text())
    assert m["labeller"] == "celltypist" and m["ct_train_depths"] == [20, 60] and m["ct_per_type"] == 100
    assert main(["compartment", "compare", "--labels", str(out), "--out", str(tmp_path / "cmp"), "--levels", "L2m",
                 "--comparisons", "nucB|nucA,wholeB|nucA", "--nbr-arm", "nucA"]) == 0
    S = pd.read_csv(tmp_path / "cmp" / "summary.csv")
    assert set(S.comparison) == {"nucB|nucA", "wholeB|nucA"}
