"""nuclear_share.py separates a group's own gene (nuclear like its markers) from neighbors' RNA."""
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "compartment" / "nuclear_share.py"
GENES = ["CXCL9", "CXCR3", "CD3E", "CD8A", "CD68", "C1QA"]


def _write(d, unit, rng, n=1500):
    xy = rng.uniform(0, 600, (n, 2))
    lab = rng.choice(["T_cell_CD8", "Macrophage", "PT"], n, p=[0.3, 0.3, 0.4])
    W = np.zeros((n, len(GENES))); Nn = np.zeros_like(W)
    g = {x: i for i, x in enumerate(GENES)}
    isT, isM = lab == "T_cell_CD8", lab == "Macrophage"
    # own genes: nuclear fraction 0.5 (T) and 0.3 (macrophage)
    for gene, who, fr in (("CD3E", isT, 0.5), ("CD8A", isT, 0.5), ("CXCR3", isT, 0.5), ("CD68", isM, 0.3),
                          ("C1QA", isM, 0.3), ("CXCL9", isM, 0.3)):
        c = rng.poisson(8, n) * who
        nuc = rng.binomial(c, fr)
        W[:, g[gene]] += c; Nn[:, g[gene]] += nuc
    # T cells pick up CXCL9 from neighboring macrophages, outside the nucleus only
    from scipy.spatial import cKDTree
    t = cKDTree(xy); _, j = t.query(xy, k=7)
    nm = isM[j[:, 1:]].sum(1)
    spill = rng.poisson(1.5 * nm) * isT
    W[:, g["CXCL9"]] += spill
    A = ad.AnnData(X=sp.csr_matrix(W), obs=pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "unit": unit, "category": "AIN",
                                                         "author_label": lab, "author_tls": "False"},
                                                        index=[f"{unit}_c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=GENES))
    A.layers["nucleus"] = sp.csr_matrix(Nn)
    A.write_h5ad(d / f"{unit}.h5ad")


def test_spillover_detected(tmp_path):
    rng = np.random.default_rng(1)
    (tmp_path / "h").mkdir()
    for u in ("P1", "P2", "P3"):
        _write(tmp_path / "h", u, rng)
    r = subprocess.run([sys.executable, str(SCRIPT), "--h5ad-dir", str(tmp_path / "h"), "--label-obs", "author_label",
                        "--groups", "T=T_cell_CD8;MAC=Macrophage", "--gene", "CXCL9",
                        "--markers", "T=CD3E,CD8A;MAC=CD68,C1QA", "--cond-col", "category", "--pair-genes", "CXCL9,CXCR3",
                        "--niche-col", "author_tls", "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    U = pd.read_csv(tmp_path / "o" / "nuclear_fraction.csv").set_index(["unit", "group"])
    for u in ("P1", "P2", "P3"):
        assert U.loc[(u, "T"), "R"] < 0.05                      # T-cell CXCL9 is all outside the nucleus
        assert abs(U.loc[(u, "MAC"), "R"] - 1) < 0.15            # macrophage CXCL9 is nuclear like CD68
    S = pd.read_csv(tmp_path / "o" / "share.csv")
    t = S[S.group == "T"]
    assert (t.share_whole > 0.2).all() and (t.share_nuc < 0.05).all()
    P = pd.read_csv(tmp_path / "o" / "pairs.csv")
    tt = P[(P.sender == "T") & (P.receiver == "T")]
    assert tt[tt.basis == "whole"].pairs.sum() > 0 and tt[tt.basis == "nucleus"].pairs.sum() == 0
    E = pd.read_csv(tmp_path / "o" / "exposure.csv")
    e = E[E.group == "T"].groupby("sources").apply(lambda d: np.average(d.cyto_per_cell, weights=d.cells),
                                                   include_groups=False)
    assert e.iloc[-1] > e.iloc[0]                                 # outside counts rise with macrophage neighbors
