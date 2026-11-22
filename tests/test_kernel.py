"""claimbreak.kernel recovers a planted spillover kernel (sigma, nu_b, R_b) and the pair-resolved kept ratio."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

from claimbreak import kernel as kn
from claimbreak.cli import main

OWN = {"T": ["CD3E", "CD3D", "CD2"], "EPI": ["KRT8", "KRT18", "EPCAM"], "FIB": ["COL1A1", "COL1A2", "DCN"]}
GENES = [g for v in OWN.values() for g in v] + ["SPP1"]
SIGMA, NU_B, NU_OWN, LEVEL = 0.08, 0.1, 0.5, 20.0


def _section(d, sample, rng, n=4000):
    xy = rng.uniform(0, 800, (n, 2))
    lab = rng.choice(["T", "EPI", "FIB"], n, p=[0.2, 0.5, 0.3])
    gi = {g: i for i, g in enumerate(GENES)}
    W = np.zeros((n, len(GENES))); N = np.zeros_like(W)
    for t, gs in OWN.items():
        m = lab == t
        for g in gs:
            c = rng.poisson(LEVEL, n) * m
            nu = rng.binomial(c, NU_OWN)
            W[:, gi[g]] += c; N[:, gi[g]] += nu
    # EPI also makes SPP1 (own); T cells make none
    m = lab == "EPI"
    c = rng.poisson(LEVEL, n) * m
    W[:, gi["SPP1"]] += c; N[:, gi["SPP1"]] += rng.binomial(c, NU_OWN)
    # spillover: per touching neighbor of type t, a cell of another type gains SIGMA x LEVEL of each of t's genes;
    # NU_B of it over the nucleus
    _, j = cKDTree(xy).query(xy, k=7, distance_upper_bound=20.0)
    j = j[:, 1:]
    for t, gs in OWN.items():
        nT = np.where(j < n, lab[np.where(j < n, j, 0)] == t, False).sum(1) * (lab != t)
        for g in gs + (["SPP1"] if t == "EPI" else []):
            c = rng.poisson(SIGMA * LEVEL * nT)
            W[:, gi[g]] += c; N[:, gi[g]] += rng.binomial(c, NU_B)
    obs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}, index=[f"{sample}_c{i}" for i in range(n)])
    A = ad.AnnData(X=sp.csr_matrix(W), obs=obs, var=pd.DataFrame(index=GENES))
    A.layers["nucleus"] = sp.csr_matrix(N)
    (d / "h").mkdir(exist_ok=True); (d / "l").mkdir(exist_ok=True)
    A.write_h5ad(d / "h" / f"{sample}.h5ad")
    pd.DataFrame({"cell_id": obs.index, "sample": sample, "L1_nucleus": lab}).to_csv(
        d / "l" / f"{sample}.labels.csv.gz", index=False)


@pytest.fixture(scope="module")
def kres(tmp_path_factory):
    d = tmp_path_factory.mktemp("kern")
    rng = np.random.default_rng(5)
    for s in ("I0001_a", "I0001_b", "I0002_a"):
        _section(d, s, rng)
    res = kn.spill_kernel(d / "h", d / "l", d / "o", unit_pattern=r"I0\d{3}", verbose=False, min_prod=1.0)
    return d, res


def test_planted_spillover_recovered(kres):
    d, res = kres
    P = res["spill_pairs"]
    assert set(P.unit) == {"I0001", "I0002"}
    # KRT8 in T cells: T does not make it -> calibration row with the planted kernel
    r = P[(P.gene == "KRT8") & (P.receiver == "T")]
    assert r.decoy.all() and (r.producer == "EPI").all()
    assert r.sigma.between(0.06, 0.10).all()
    assert r.nu_b.between(0.06, 0.14).all()
    assert r.R_b.between(0.12, 0.28).all()
    # own markers found, own nuclear fraction recovered
    M = res["own_markers"]
    assert M[M.type == "T"].markers.str.contains("CD3E").all()
    assert M.nu_own.between(0.45, 0.55).all()


def test_r_matrix_and_summary(kres):
    d, res = kres
    R = res["r_matrix"]
    own = R[(R.gene == "CD3E") & (R.type == "T")].R
    borrowed = R[(R.gene == "SPP1") & (R.type == "T")].R
    assert own.between(0.9, 1.1).all() and (borrowed < 0.35).all()
    S = kn.kernel_summary(d / "o", None, d / "s", "toy")
    E = S["expr"].set_index("unit")
    assert 0.06 <= E.loc["ALL", "sigma"] <= 0.10 and 0.12 <= E.loc["ALL", "R_b"] <= 0.28
    assert (d / "o" / "decoy_pairs.csv.gz").exists() and (d / "o" / "spill_pairs.csv.gz").exists()


def test_refuses_existing_output(kres):
    d, _ = kres
    with pytest.raises(FileExistsError):
        kn.spill_kernel(d / "h", d / "l", d / "o", verbose=False)


def test_label_kernel(tmp_path):
    # a loses its label where it touches b (kept 0.6 at share > 0, 1.0 at 0); the null keeps all
    rows = []
    for u in ("U1", "U2"):
        for b_, kt, tb in (("0", 100, 0), ("(0,0.25]", 60, 10), ("(0.25,0.5]", 60, 20), ("(0.5,0.75]", 60, 30)):
            rows.append(dict(sample=u, unit=u, level="L1", a="T", b="EPI", bin=b_, cells=100, to_b_test=tb,
                             to_b_null=0, kept_test=kt, kept_null=100))
    (tmp_path / "e").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "e" / "destinations.csv", index=False)
    U, F = kn.label_kernel(tmp_path / "e", tmp_path / "k")
    assert np.allclose(F.K, 0.6) and (F.q_slope > 0).all()
    main(["compartment", "kernel-summary", "--out", str(tmp_path / "k2"), "--exposure", str(tmp_path / "e"),
          "--dataset", "toy"])
    L = pd.read_csv(tmp_path / "k2" / "kernel_labels.csv")
    assert L.K.iloc[0] == pytest.approx(0.6)


def test_cli_kernel(tmp_path):
    rng = np.random.default_rng(9)
    _section(tmp_path, "I0003_a", rng, n=2500)
    main(["compartment", "kernel", "--h5ad-dir", str(tmp_path / "h"), "--labels", str(tmp_path / "l"),
          "--out", str(tmp_path / "o"), "--min-prod", "1"])
    assert (tmp_path / "o" / "own_markers.csv").exists()


def test_mixed_panels_pooled_per_panel(tmp_path):
    # one unit imaged on two panels (lung, gut): sections are grouped by unit and panel, not refused
    rng = np.random.default_rng(11)
    _section(tmp_path, "I0005_a", rng)
    _section(tmp_path, "I0005_b", rng)
    f = tmp_path / "h" / "I0005_b.h5ad"
    A = ad.read_h5ad(f)
    A[:, [g for g in GENES if g != "SPP1"]].copy().write_h5ad(f)
    res = kn.spill_kernel(tmp_path / "h", tmp_path / "l", tmp_path / "o", verbose=False, min_prod=1.0)
    S = res["sections"]
    assert sorted(S.panel) == ["p10", "p9"]
    P = res["spill_pairs"]
    assert set(P.panel) == {"p10", "p9"} and "SPP1" not in set(P[P.panel == "p9"].gene)
    E = kn.kernel_summary(tmp_path / "o", None, tmp_path / "s", "toy")["expr"]
    assert set(E.panel) >= {"p10", "p9"}


def test_rb_interval_from_genes(kres):
    """R_b and nu_b get gene-resampled intervals that bracket the pooled value; boot=0 leaves them empty."""
    d, _ = kres
    S = kn.kernel_summary(d / "o", None, d / "s_boot", "toy", boot=400)
    E = S["expr"]
    row = E[E.unit == "ALL"].iloc[0]
    assert row.R_b_lo <= row.R_b <= row.R_b_hi and row.nu_b_lo <= row.nu_b <= row.nu_b_hi
    assert row.R_b_hi - row.R_b_lo < 0.2                                  # planted R_b 0.20, many genes
    assert {"R_b_lo", "R_b_hi"} <= set(S["expr_types"].columns)
    E0 = kn.kernel_summary(d / "o", None, d / "s_noboot", "toy", boot=0)["expr"]
    assert E0.R_b_lo.isna().all()
    assert np.allclose(E0.R_b, E.R_b, equal_nan=True)                    # the point value is unchanged
