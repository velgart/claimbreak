"""Label audit: an over-detected gene that reads as one type's marker is flagged, the calls resting on it are
reported, its exclusion restores the labels, and the audit never changes the labels themselves."""
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from claimbreak import compartment as cm
from claimbreak.cli import main

TYPES = ["T", "Fib", "Mye"]


def _make(root, hot: bool, seed=11):
    """Reference T/Fib/Mye on 60 genes; two sections. With hot=True, fibroblast nuclei carry a gene (g0) that the
    reference holds mostly in T cells, at ~20% of their nuclear counts (as CCL5 in synovium)."""
    rng = np.random.default_rng(seed)
    G = 60
    genes = [f"g{i}" for i in range(G)]
    base = rng.dirichlet(np.full(G, 2.0))
    prof = {}
    for k, t in enumerate(TYPES):
        p = base.copy()
        p[1 + 8 * k: 9 + 8 * k] *= 12                    # 8 markers per type (g1-8 T, g9-16 Fib, g17-24 Mye)
        prof[t] = p / p.sum()
    for t in TYPES:                                       # g0: a T gene in the reference, as CCL5 (1% in T, ~33x)
        prof[t][0] = 0.0
        prof[t] *= (1 - (0.01 if t == "T" else 0.0003)) / prof[t].sum()
        prof[t][0] = 0.01 if t == "T" else 0.0003
    lab = np.repeat(TYPES, 200)
    X = np.vstack([rng.multinomial(rng.integers(300, 800), prof[t]) for t in lab])
    obs = pd.DataFrame({"coarse": lab, "donor_id": np.tile(list("abcd"), len(lab) // 4)},
                       index=[f"r{i}" for i in range(len(lab))])
    ad.AnnData(sp.csr_matrix(X.astype(np.float32)), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(root / "ref.h5ad")
    for s in range(2):
        n = 900
        ct = rng.choice(TYPES, n, p=[0.2, 0.6, 0.2])
        nuc, cyt = [], []
        for t in ct:
            pn = prof[t].copy()
            if hot and t == "Fib":
                pn[0] = 0.25 * pn.sum() / 0.75 + pn[0]    # g0 ~ 20-25% of the nuclear counts
                pn /= pn.sum()
            nuc.append(rng.multinomial(rng.integers(15, 60), pn))
            cyt.append(rng.multinomial(rng.integers(40, 120), prof[t]))
        nuc, cyt = np.array(nuc), np.array(cyt)
        W = nuc + cyt
        o = pd.DataFrame({"x": rng.uniform(0, 300, n), "y": rng.uniform(0, 300, n), "n_whole": W.sum(1),
                          "n_nucleus": nuc.sum(1), "panel": "p", "truth": ct}, index=[f"c{s}_{i}" for i in range(n)])
        A = ad.AnnData(sp.csr_matrix(W.astype(np.float32)), obs=o, var=pd.DataFrame(index=genes))
        A.layers["nucleus"] = sp.csr_matrix(nuc.astype(np.float32))
        A.layers["cyto"] = sp.csr_matrix(cyt.astype(np.float32))
        A.write_h5ad(root / f"sec{s}.h5ad")
    return root


@pytest.fixture(scope="module")
def hot(tmp_path_factory):
    return _make(tmp_path_factory.mktemp("audit_hot"), hot=True)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return _make(tmp_path_factory.mktemp("audit_clean"), hot=False)


def _ref(root):
    return cm.Reference.load(root / "ref.h5ad", {"L1": "coarse"}, verbose=False)


def _fib_called_T(res, A):
    f = A.obs["truth"].to_numpy() == "Fib"
    v = res["L1_nucleus"].to_numpy()
    return float((v[f & (v != "")] == "T").mean())


def test_overdetected_gene_flagged_with_driver_and_support(hot):
    A = ad.read_h5ad(hot / "sec0.h5ad")
    res, _, info = cm.label_section(A, _ref(hot), ["L1"], ["whole", "nucleus"], cm.LabelConfig(gamma_iter=0))
    assert _fib_called_T(res, A) > 0.1                               # the failure: fibroblasts read as T
    au = info["audit"]["L1_nucleus"]
    assert au["overdetected"][0]["gene"] == "g0" and au["overdetected"][0]["type_of_max"] == "T"
    assert au["drivers"]["T"]["top"][0][0] == "g0"                  # top driver, diluted by true T cells
    assert au["decided"]["g0"]["type"] == "T" and au["decided"]["g0"]["share"] > 0.2
    assert au["support"]["T"] < au["support"]["Fib"]                 # T calls without T markers
    assert any("audit: g0 is" in w for w in info["warnings"])
    assert any("exist only through over-detected g0" in w for w in info["warnings"])
    # the cells decided by g0 are the misread fibroblasts
    v = res["L1_nucleus"].to_numpy()
    assert (A.obs["truth"].to_numpy()[v == "T"] == "Fib").mean() > 0.2


def test_exclusion_restores_labels_and_silences_audit(hot):
    A = ad.read_h5ad(hot / "sec0.h5ad")
    res, _, info = cm.label_section(A, _ref(hot), ["L1"], ["nucleus"],
                                    cm.LabelConfig(gamma_iter=0, exclude_genes=("g0",)))
    assert _fib_called_T(res, A) < 0.1
    assert not info["audit"]["L1_nucleus"]["overdetected"]
    assert not any("exist only through" in w for w in info["warnings"])


def test_clean_data_raise_no_audit_warning(clean):
    A = ad.read_h5ad(clean / "sec0.h5ad")
    _, _, info = cm.label_section(A, _ref(clean), ["L1"], ["whole", "nucleus"], cm.LabelConfig())
    assert not [w for w in info["warnings"] if "audit" in w], info["warnings"]
    assert all(v > 0.5 for v in info["audit"]["L1_nucleus"]["support"].values())


def test_audit_never_changes_labels(hot):
    A = ad.read_h5ad(hot / "sec1.h5ad")
    ref = _ref(hot)
    r1, g1, i1 = cm.label_section(A, ref, ["L1"], ["whole", "nucleus", "nucA", "nucB"], cm.LabelConfig(), seed=2)
    r0, g0, i0 = cm.label_section(A, ref, ["L1"], ["whole", "nucleus", "nucA", "nucB"], cm.LabelConfig(audit=False),
                                  seed=2)
    pd.testing.assert_frame_equal(r1, r0)
    pd.testing.assert_frame_equal(g1, g0)
    assert "audit" not in i0 and set(i1["audit"]) == {"L1_whole", "L1_nucleus"}


def test_markers_arm_and_celltypist_free_paths(hot):
    A = ad.read_h5ad(hot / "sec0.h5ad")
    _, _, info = cm.label_section(A, _ref(hot), ["L1"], ["nucleus"], cm.LabelConfig(markers=10))
    assert "L1_nucleus" in info["audit"]                             # audit on the marker genes only
    d, rows, warn = cm.label_audit(sp.csr_matrix(A.layers["nucleus"]), _ref(hot).profiles("L1", np.arange(60))[1],
                                   TYPES, list(A.var_names), np.zeros(A.n_obs, int))
    assert d["drivers"] == {} and "overdetected" in set(rows.check)   # no runner-up: no driver check


def test_cli_label_audit_relabel(hot, tmp_path):
    common = ["--ref", str(hot / "ref.h5ad"), "--ref-levels", "L1=coarse", "--inputs", str(hot / "sec0.h5ad"),
              str(hot / "sec1.h5ad"), "--arms", "whole,nucleus,nucA,nucB", "--gamma-iter", "0"]
    assert main(["compartment", "label", *common, "--out", str(tmp_path / "lab")]) == 0
    assert (tmp_path / "lab" / "sec0.audit.csv").exists()
    m = json.loads((tmp_path / "lab" / "sec0.meta.json").read_text())
    assert m["label_audit"]["L1_nucleus"]["overdetected"][0]["gene"] == "g0" and m["audit_log2"] == 1.0
    assert main(["compartment", "audit", "--labels", str(tmp_path / "lab"), "--out", str(tmp_path / "au")]) == 0
    ex = (tmp_path / "au" / "exclude_overdetected.txt").read_text().split()
    assert ex == ["g0"]
    assert main(["compartment", "label", *common, "--exclude-genes", str(tmp_path / "au" / "exclude_overdetected.txt"),
                 "--out", str(tmp_path / "lab2")]) == 0
    A = ad.read_h5ad(hot / "sec0.h5ad")
    r1 = pd.read_csv(tmp_path / "lab" / "sec0.labels.csv.gz", keep_default_na=False).set_index("cell_id")
    r2 = pd.read_csv(tmp_path / "lab2" / "sec0.labels.csv.gz", keep_default_na=False).set_index("cell_id")
    assert _fib_called_T(r2.reindex(A.obs_names), A) < 0.1 < _fib_called_T(r1.reindex(A.obs_names), A)
    m2 = json.loads((tmp_path / "lab2" / "sec0.meta.json").read_text())
    assert not m2["label_audit"]["L1_nucleus"]["overdetected"]


def test_audit_summary_needs_audited_labels(tmp_path):
    with pytest.raises(ValueError, match="no \\*.audit.csv"):
        cm.audit_summary(tmp_path)


def test_relabel_excluding_script(hot, tmp_path):
    """relabel a labels folder with extra exclusions, settings from its own meta.json."""
    import subprocess, sys
    from pathlib import Path
    ex = Path(__file__).resolve().parents[1] / "examples" / "compartment" / "relabel_excluding.py"
    common = ["--ref", str(hot / "ref.h5ad"), "--ref-levels", "L1=coarse", "--inputs", str(hot / "sec0.h5ad"),
              str(hot / "sec1.h5ad"), "--arms", "whole,nucleus,nucA,nucB", "--gamma-iter", "0", "--seed", "4"]
    assert main(["compartment", "label", *common, "--out", str(tmp_path / "lab")]) == 0
    r = subprocess.run([sys.executable, str(ex), "--from", str(tmp_path / "lab"), "--h5ad-dir", str(hot),
                        "--out", str(tmp_path / "lab2"), "--exclude", "g0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    m = json.loads((tmp_path / "lab2" / "sec0.meta.json").read_text())
    assert m["exclude_genes"] == ["g0"] and m["seed"] == 4 and m["gamma_iter"] == 0
    assert m["arms"] == ["whole", "nucleus", "nucA", "nucB"]
    assert not m["label_audit"]["L1_nucleus"]["overdetected"]
    assert (tmp_path / "lab2_audit" / "exclude_overdetected.txt").read_text().split() == []
    A = ad.read_h5ad(hot / "sec0.h5ad")
    r2 = pd.read_csv(tmp_path / "lab2" / "sec0.labels.csv.gz", keep_default_na=False).set_index("cell_id")
    assert _fib_called_T(r2.reindex(A.obs_names), A) < 0.1
    r3 = subprocess.run([sys.executable, str(ex), "--from", str(tmp_path / "lab"), "--h5ad-dir", str(hot),
                         "--out", str(tmp_path / "lab2"), "--exclude", "g0"], capture_output=True, text=True)
    assert r3.returncode != 0 and "not empty" in r3.stderr                # never overwrites


def test_audit_existing_reproduces_label_time_audit(hot, tmp_path):
    """labels written without the audit are audited afterwards from their stored labels,
    runner-ups and gamma, with the same result as the audit at labeling time, without relabeling."""
    common = ["--ref", str(hot / "ref.h5ad"), "--ref-levels", "L1=coarse", "--inputs", str(hot / "sec0.h5ad"),
              str(hot / "sec1.h5ad"), "--arms", "whole,nucleus,nucA,nucB", "--seed", "3"]
    assert main(["compartment", "label", *common, "--out", str(tmp_path / "with")]) == 0
    assert main(["compartment", "label", *common, "--no-audit", "--out", str(tmp_path / "old")]) == 0
    assert not list((tmp_path / "old").glob("*.audit.csv"))
    assert main(["compartment", "audit", "--labels", str(tmp_path / "old"), "--out", str(tmp_path / "re"),
                 "--recompute", str(hot), "--arms", "nucleus,whole,nucB"]) == 0
    for s in ("sec0", "sec1"):
        a = pd.read_csv(tmp_path / "with" / f"{s}.audit.csv", keep_default_na=False)
        b = pd.read_csv(tmp_path / "re" / f"{s}.audit.csv", keep_default_na=False)
        b = b[b.arm.isin(["nucleus", "whole"])]
        key = ["level", "arm", "check", "type", "gene"]
        m = a.merge(b, on=key, suffixes=("_a", "_b"), how="outer", indicator=True)
        assert (m["_merge"] == "both").all(), m[m["_merge"] != "both"]
        assert np.allclose(m.value_a, m.value_b, atol=1e-3)          # gamma stored to 4 decimals
    R = pd.read_csv(tmp_path / "re" / "reproduce.csv")
    assert (R.label_is_max > 0.999).all() and (R.share_within_tol > 0.99).all()
    assert set(R.arm) == {"nucleus", "whole", "nucB"}                # a split arm, rebuilt with the seed
    assert (tmp_path / "re" / "exclude_overdetected.txt").read_text().split() == ["g0"]
    with pytest.raises(ValueError, match="not empty"):
        cm.audit_existing(tmp_path / "old", hot, tmp_path / "re")


def test_audit_existing_flags_a_wrong_reference(hot, tmp_path):
    """Reconstruction against another reference is caught by the margin check, not silently audited."""
    assert main(["compartment", "label", "--ref", str(hot / "ref.h5ad"), "--ref-levels", "L1=coarse", "--inputs",
                 str(hot / "sec0.h5ad"), "--arms", "whole,nucleus", "--no-audit", "--out", str(tmp_path / "old")]) == 0
    W = ad.read_h5ad(hot / "ref.h5ad")                    # same genes and types, T and Fib profiles swapped
    W.obs["coarse"] = W.obs["coarse"].astype(str).map({"T": "Fib", "Fib": "T", "Mye": "Mye"})
    W.write_h5ad(tmp_path / "wrong_ref.h5ad")
    R = cm.audit_existing(tmp_path / "old", hot, tmp_path / "re", reference=str(tmp_path / "wrong_ref.h5ad"),
                          verbose=False)
    assert (R.label_is_max < 0.99).any()


def test_support_judged_against_depth(hot):
    """shallow true members miss their markers by chance; support is compared with the support expected at
    each call's depth, so shallow clean labels raise no support warning while the misread fibroblasts still do."""
    A = ad.read_h5ad(hot / "sec0.h5ad")
    ref = _ref(hot)
    types, mu = ref.profiles("L1", np.arange(60))
    rng = np.random.default_rng(0)
    kT = list(types).index("T")
    n = 400                                      # true T cells with 3-6 counts: few carry one of the 8 T markers
    X = np.vstack([rng.multinomial(rng.integers(3, 7), mu[kT]) for _ in range(n)])
    lab = np.full(n, kT)
    cfg = cm.LabelConfig(min_counts=3, audit_min_support=0.999)          # would warn without the depth check
    d, rows, warn = cm.label_audit(sp.csr_matrix(X.astype(np.float32)), mu, types, list(A.var_names), lab, cfg=cfg)
    assert d["support"]["T"] < 0.999 and abs(d["support"]["T"] / d["support_expected"]["T"] - 1) < 0.1
    assert not [w for w in warn if "marker transcript" in w]
    assert "support_expected" in set(rows.check)
    # the 8 planted markers per type (real markers are this specific; the toy's 20 include shared base genes)
    res, _, info = cm.label_section(A, ref, ["L1"], ["nucleus"], cm.LabelConfig(gamma_iter=0, audit_markers=8))
    au = info["audit"]["L1_nucleus"]


def _specific(rng, G=300):
    """Two types on G genes with 10 specific markers each (5% of the profile per marker in the own type, 0.05% in
    the other), so markers discriminate as on a large panel."""
    base = rng.dirichlet(np.full(G, 2.0)) * 0.5
    mu = np.vstack([base.copy(), base.copy()])
    mu[0, 1:11] = 0.05; mu[1, 1:11] = 0.0005               # T markers g1-g10
    mu[1, 11:21] = 0.05; mu[0, 11:21] = 0.0005             # Fib markers g11-g20
    mu[:, 0] = [0.01, 0.0003]                              # g0: a T gene in the reference (as CCL5)
    return mu / mu.sum(1, keepdims=True), [f"g{i}" for i in range(G)]


def _decided(rng, mu, genes, hot_type):
    """Cells of `hot_type` (0 = T, 1 = Fib) whose profile carries g0 (or g1 for T) far above the reference."""
    X = []
    n_hot = 750 if hot_type == 1 else 1400                 # genuine case: mostly shallow T cells, so g1 is over-detected
    for i in range(1500):
        t = hot_type if i < n_hot else 1 - hot_type
        p = mu[t].copy()
        if i < n_hot:
            j = 1 if hot_type == 0 else 0                  # T: its own marker g1 read 8x; Fib: g0 at 25%
            p[j] = p[j] * 8 if hot_type == 0 else 0.33
            p /= p.sum()
        X.append(rng.multinomial(rng.integers(3, 9) if (hot_type == 0 and i < n_hot) else rng.integers(15, 40), p))
    Xs = sp.csr_matrix(np.array(X, dtype=np.float32))
    cfg = cm.LabelConfig(gamma_iter=0, audit_decided_share=0.0, audit_markers=10, audit_min_share=0.002, min_counts=3)
    lab, sec, _, _, gam = cm.classify(Xs, mu, cfg, fit_gamma=False)
    return cm.label_audit(Xs, mu, np.array(["T", "Fib"]), genes, lab, sec, gam, cfg)


def test_decided_calls_separate_misread_from_genuine():
    """calls that exist only through an over-detected gene are placed between true members (0) and their
    runner-up type (1) by the runner-up's markers at their depth: misread fibroblasts (g0 as CCL5) sit near 1,
    shallow true T cells whose own marker is detected 8x as often (as TRAC in gut) near 0."""
    rng = np.random.default_rng(5)
    mu, genes = _specific(rng)
    d, _, warn = _decided(rng, mu, genes, hot_type=1)
    assert d["decided"]["g0"]["misread"] > 0.7
    assert any("runner-up marker score" in w for w in warn if "through over-detected g0" in w)
    d, _, warn = _decided(rng, mu, genes, hot_type=0)
    assert d["decided"]["g1"]["misread"] < 0.3
    assert not any("likely misread" in w or "consistent with genuine" in w for w in warn)   # no verdict
