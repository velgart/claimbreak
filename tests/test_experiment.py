import numpy as np
import pandas as pd

from claimbreak import run


def cfg(out):
    return {"out": str(out), "seed": 1, "reps": 2,
            "plant": [{"kind": "spike", "sender": "Inf", "receiver": "A", "genes": 5, "effect": 1.5, "lam": 25}],
            "questions": ["composition", {"name": "colocalization", "n_perm": 20},
                          {"name": "niches", "n_niches": 6},
                          {"name": "proximity_de", "n_genes": 40, "index_types": ["A", "B"],
                           "neighbour_types": ["Inf", "Epi"], "aggregate": "pooled"}],
            "perturbations": [
                {"kind": "label_errors", "name": "err_random", "confusion": "native", "levels": [0.05, 0.2]},
                {"kind": "label_errors", "name": "err_boundary_ctx", "confusion": "native", "spatial": "boundary",
                 "context_bias": 2.0, "levels": [0.2]},
                {"kind": "recall", "levels": [0.5]},
                {"kind": "contamination", "name": "contam_recall", "lam": 10, "labels": "recall", "levels": [0.2]}]}


def test_run_and_resume(tissue, tmp_path):
    res = run(cfg(tmp_path), tissue=tissue, verbose=False)
    S = res["summary"]
    assert set(S.spec) == {"err_random", "err_boundary_ctx", "recall_2", "contam_recall"}
    s = S[S.question == "colocalization"].groupby(["spec", "level"]).survival.mean()
    assert s[("err_random", 0.05)] >= s[("err_random", 0.2)]
    assert (tmp_path / "breakpoints.csv").exists()
    n = len(S)
    res2 = run(cfg(tmp_path), tissue=tissue, verbose=False)      # resumes: nothing new to do
    assert len(res2["summary"]) == n
    pde = S[(S.question == "proximity_de") & (S.spec == "err_random")]
    assert "power" in pde and "marker_share_up" in pde


def test_placeholder_and_inspect(tissue, tmp_path):
    import anndata as ad, numpy as np, pandas as pd, pytest, yaml
    from claimbreak.experiment import load_config, ConfigError
    from claimbreak.tissue import describe_file
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump({"out": "x", "data": {"path": "a", "label_key": "REPLACE"}}))
    with pytest.raises(ConfigError):
        load_config(f)
    A = ad.AnnData(X=tissue.counts[:50].astype(np.float32),
                   obs=pd.DataFrame({"ct": tissue.labels[:50]}, index=[f"c{i}" for i in range(50)]))
    A.write_h5ad(tmp_path / "d.h5ad")
    assert "ct" in describe_file(tmp_path / "d.h5ad")


def test_load_subset_and_integer_check(tissue, tmp_path):
    import anndata as ad, numpy as np, pandas as pd, pytest
    from claimbreak import load
    n = 3000
    obs = pd.DataFrame({"ct": tissue.labels[:n], "sec": tissue.section[:n], "mouse": tissue.unit[:n],
                        "grp": tissue.condition[:n], "x": tissue.xy[:n, 0], "y": tissue.xy[:n, 1]},
                       index=[f"c{i}" for i in range(n)])
    A = ad.AnnData(X=tissue.counts[:n].toarray().astype(np.float32), obs=obs)   # dense X, like many files
    A.write_h5ad(tmp_path / "raw.h5ad")
    t = load(tmp_path / "raw.h5ad", "ct", "sec", "mouse", "grp", subset={"grp": ["ctrl"]})
    assert set(t.condition) == {"ctrl"} and t.n == (obs.grp == "ctrl").sum()
    with pytest.raises(ValueError):
        load(tmp_path / "raw.h5ad", "ct", "sec", subset={"grp": ["nope"]})
    B = A.copy(); B.X = np.log1p(B.X)
    B.write_h5ad(tmp_path / "log.h5ad")
    with pytest.raises(ValueError, match="not integers"):
        load(tmp_path / "log.h5ad", "ct", "sec")


def test_labelled_questions(tissue, tmp_path):
    import pytest
    from claimbreak.experiment import ConfigError
    c = {"out": str(tmp_path), "reps": 1,
         "questions": [{"name": "proximity_de", "n_genes": 30, "aggregate": "pooled"},
                       {"name": "proximity_de", "label": "proximity_de_eff", "n_genes": 30, "aggregate": "pooled",
                        "min_effect": 0.3}],
         "perturbations": [{"kind": "label_errors", "confusion": "native", "levels": [0.1]}]}
    S = run(c, tissue=tissue, verbose=False)["summary"]
    assert set(S.question) == {"proximity_de", "proximity_de_eff"}
    c["questions"][1].pop("label")
    with pytest.raises(ConfigError):
        run({**c, "out": str(tmp_path / "b")}, tissue=tissue, verbose=False)


def test_native_by_condition_run(tissue, tmp_path):
    import json
    c = {"out": str(tmp_path), "reps": 1,
         "questions": [{"name": "colocalization", "n_perm": 10}, {"name": "niches", "n_niches": 5, "n_seeds": 4}],
         "perturbations": [{"kind": "label_errors", "name": "cond", "confusion": "native_by_condition",
                            "type_rates": "native_by_condition", "levels": [0.1]}]}
    r = run(c, tissue=tissue, verbose=False)
    nat = json.load(open(tmp_path / "native.json"))
    assert set(nat["by_condition"]) == {"ctrl", "dis"}
    S = r["summary"]
    assert abs(S.realised_eps.iloc[0] - 0.1) < 0.02 and "n_unstable" in S


def test_explain(tissue, tmp_path):
    import glob
    import pandas as pd
    from claimbreak.compare import explain
    from claimbreak.cli import main
    c = {"out": str(tmp_path), "reps": 2, "questions": [{"name": "colocalization", "n_perm": 10}],
         "perturbations": [{"kind": "label_errors", "name": "e", "confusion": "native", "levels": [0.3]}]}
    run(c, tissue=tissue, verbose=False)
    D = pd.concat([pd.read_csv(f) for f in glob.glob(str(tmp_path / "decisions_*.csv.gz"))])
    base = pd.read_csv(tmp_path / "baseline_claims.csv")
    E = explain(D, base, "colocalization")
    S = pd.read_csv(tmp_path / "summary.csv")
    assert np.isclose(E.gained.sum() * 1, S.spurious.mean(), atol=1e-6) or E.empty
    assert np.isclose(E.lost.sum(), (S.n_base - S.kept).mean())
    assert main(["explain", str(tmp_path), "--question", "colocalization"]) == 0


def test_joint_in_runner(tissue, tmp_path):
    c = {"out": str(tmp_path), "reps": 1,
         "plant": [{"kind": None}, {"kind": "spike", "sender": "Inf", "receiver": "A", "genes": 5, "effect": 1.5,
                                    "lam": 20}],
         "questions": [{"name": "proximity_de_joint", "n_genes": 30, "aggregate": "pooled", "sigma": 20}],
         "perturbations": [{"kind": "label_errors", "confusion": "native", "levels": [0, 0.1]}]}
    S = run(c, tissue=tissue, verbose=False)["summary"]
    assert "power" in S and "marker_share_up" in S and S.power.notna().all()


def test_check_truth_detail_and_diagnostics(tissue, tmp_path, capsys):
    from claimbreak.experiment import check
    c = {"out": str(tmp_path), "reps": 1,
         "plant": [{"kind": None},
                   {"kind": "spike", "sender": "Inf", "receiver": "A", "genes": 5, "effect": 1.5, "lam": 20},
                   {"kind": "spike", "sender": "Epi", "receiver": "B", "genes": 5, "effect": 1.5, "lam": 20,
                    "marker_rule": "pair"}],
         "questions": [{"name": "proximity_de", "n_genes": 30, "aggregate": "pooled"},
                       {"name": "proximity_de", "label": "proximity_de_kernel", "n_genes": 30, "aggregate": "pooled",
                        "k": 30, "sigma": 20},
                       {"name": "proximity_de_joint", "n_genes": 30, "aggregate": "pooled", "k": 30, "sigma": 20}],
         "perturbations": [{"kind": "label_errors", "confusion": "native", "levels": [0, 0.1]}]}
    recs = check(c, tissue=tissue)
    assert "check passed" in capsys.readouterr().out and len(recs) == 3
    S = run(c, tissue=tissue, verbose=False)["summary"]
    for col in ["power_1", "power_2", "misattrib_up", "misattrib_down"]:
        assert col in S and S[col].notna().all()
    assert set(S.question) == {"proximity_de", "proximity_de_kernel", "proximity_de_joint"}
    assert (tmp_path / "diagnostics_proximity_de_joint.csv").exists()
    b = S[S.level == 0]
    assert np.allclose((b.power_1 + b.power_2) / 2, b.power)          # equal gene counts: pooled = mean


def test_fixed_threshold_rescoring(tissue, tmp_path):
    """Fixed-threshold scoring equals the run's own scoring at level 0 and is never more lenient than the baseline."""
    import gzip
    c = {"out": str(tmp_path), "reps": 1,
         "plant": [{"kind": None}, {"kind": "spike", "sender": "Inf", "receiver": "A", "genes": 5, "effect": 1.5,
                                    "lam": 20}],
         "questions": [{"name": "proximity_de", "n_genes": 30, "aggregate": "pooled"},
                       {"name": "proximity_de_joint", "n_genes": 30, "aggregate": "pooled", "k": 30, "sigma": 20}],
         "perturbations": [{"kind": "label_errors", "confusion": "native", "levels": [0, 0.2]}]}
    S = run(c, tissue=tissue, verbose=False)["summary"]
    z = S[S.level == 0]
    assert np.allclose(z.power_fixed, z.power) and np.allclose(z.fp_fixed, z.fp)
    assert (S.p_thr_base >= 0).all() and "power_1_fixed" in S
    D = pd.read_csv(tmp_path / "decisions_label_errors_0.csv.gz")
    assert "p" in D
    import json
    rec = json.loads((tmp_path / "planted.json").read_text())[1]
    assert set(rec["profiles"]) == set(rec["genes"])
    d = pd.read_csv(tmp_path / "diagnostics_proximity_de_joint.csv")
    assert {"vif_section_median", "share_exposed"} <= set(d.columns)


def test_correctors_in_runner(tissue, tmp_path):
    c = {"out": str(tmp_path), "reps": 1,
         "plant": [{"kind": None}, {"kind": "spike", "sender": "Inf", "receiver": "A", "genes": 5, "effect": 1.5,
                                    "lam": 20}],
         "questions": [{"name": "proximity_de", "n_genes": 60, "aggregate": "units",
                        "correctors": [{"name": "twode"}, {"name": "simex", "pi": "oracle", "B": 1},
                                       {"name": "simex", "pi": "native", "B": 1},
                                       {"name": "ppi", "fraction": 0.2, "patch": 150}]}],
         "perturbations": [{"kind": "label_errors", "confusion": "native", "levels": [0, 0.2]}]}
    S = run(c, tissue=tissue, verbose=False)["summary"]
    names = {"proximity_de", "proximity_de+twode", "proximity_de+simex_oracle", "proximity_de+simex_native",
             "proximity_de+ppi_20"}
    assert set(S.question) == names
    z = S[S.level == 0].set_index("question")
    assert z.loc["proximity_de+simex_oracle", "n_run"] == z.loc["proximity_de", "n_run"]   # no error: unchanged
    assert z.loc["proximity_de+ppi_20", "ppi_revealed"] >= 0.2
    assert {"power_fixed", "fp_fixed", "corrector_seconds"} <= set(S.columns)
