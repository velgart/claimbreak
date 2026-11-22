import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

from claimbreak import Tissue

TYPES = ["Epi", "Inf", "Endo", "A", "A2", "B", "B2"]


def synthetic(n_sec=6, n=2000, depth=100, m=60, seed=0, G=120):
    """Small niche-structured tissue: 7 types (two confusable target pairs), one section per unit,
    alternating conditions."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.6, G)
    prof = {}
    for j, t in enumerate(["Epi", "Inf", "Endo"]):
        p = base.copy(); p[10 * j:10 * (j + 1)] += 2.5; prof[t] = p
    for t in ["A", "A2", "B", "B2"]:
        p = base.copy(); p[30:40] += 2.0
        fam = t in ("A", "A2")
        p[40:50] += 1.0 if fam else -1.0
        p[50:56] += 0.7 if t in ("A", "B") else -0.7
        prof[t] = p
    Ys, obs = [], []
    for s in range(n_sec):
        L = np.sqrt(n) * 10
        xy = rng.uniform(0, L, (n, 2))
        nd = max(2, n // m)
        dom = cKDTree(rng.uniform(0, L, (nd, 2))).query(xy)[1]
        comp = rng.integers(0, 4, nd)
        mix = rng.dirichlet([1, 1, 1], nd)
        lab = np.array([["A", "A2", "B", "B2"][comp[d]] if rng.random() < 0.55 else
                        ["Epi", "Inf", "Endo"][rng.choice(3, p=mix[d])] for d in dom])
        P = np.vstack([prof[t] for t in lab]) + rng.normal(0, 0.3, (n, G))
        R = np.exp(P); R /= R.sum(1, keepdims=True)
        Ys.append(rng.poisson(R * depth * rng.lognormal(0, 0.3, n)[:, None]))
        obs.append(pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "label": lab, "section": f"s{s}",
                                 "unit": f"u{s}", "cond": "ctrl" if s % 2 == 0 else "dis"}))
    o = pd.concat(obs, ignore_index=True)
    return Tissue(counts=sp.csr_matrix(np.vstack(Ys).astype(float)), xy=o[["x", "y"]].to_numpy(),
                  labels=o.label.to_numpy(), section=o.section.to_numpy(), unit=o.unit.to_numpy(),
                  genes=np.array([f"g{i}" for i in range(G)]), condition=o.cond.to_numpy())


@pytest.fixture(scope="session")
def tissue():
    return synthetic()
