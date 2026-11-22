"""claimbreak.spill: neighbor transcripts make a gradient in whole-cell counts that nuclear counts do not show."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

from claimbreak import spill as sl
from claimbreak.cli import main

GENES = ["NAMPT", "VCAM1", "CXCL9", "CXCR3", "CD3E", "LRP2"] + [f"G{i}" for i in range(30)]


def _section(root, name, n=3000, seed=0):
    """PT cells express NAMPT (nucleus and cytoplasm) and VCAM1; T cells express CD3E and CXCR3 and none of NAMPT.
    Each T cell receives NAMPT transcripts in its cytoplasm from every PT cell it touches (segmentation spill-over)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 2000, (n, 2))
    lab = np.where(rng.random(n) < 0.5, "PT", "T_NK").astype(object)
    G = len(GENES)
    nuc = rng.poisson(0.3, (n, G)).astype(float)
    cyt = rng.poisson(0.6, (n, G)).astype(float)
    pt = lab == "PT"
    for g, lam in (("NAMPT", 3), ("VCAM1", 1.5), ("LRP2", 4)):
        j = GENES.index(g)
        nuc[pt, j] = rng.poisson(lam, pt.sum()); cyt[pt, j] = rng.poisson(2 * lam, pt.sum())
        nuc[~pt, j] = 0; cyt[~pt, j] = 0
    for g, lam in (("CD3E", 3), ("CXCR3", 1.0)):
        j = GENES.index(g)
        nuc[~pt, j] = rng.poisson(lam, (~pt).sum()); cyt[~pt, j] = rng.poisson(2 * lam, (~pt).sum())
        nuc[pt, j] = 0; cyt[pt, j] = 0
    d, idx = cKDTree(xy).query(xy, k=7)
    touch_pt = ((d[:, 1:] <= 30) & pt[idx[:, 1:]]).sum(1)
    j = GENES.index("NAMPT")
    cyt[~pt, j] += rng.poisson(0.5 * touch_pt[~pt])
    A = ad.AnnData(X=sp.csr_matrix(nuc + cyt), obs=pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]},
                                                              index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=GENES))
    A.layers["nucleus"] = sp.csr_matrix(nuc)
    A.write_h5ad(root / "h5ad" / f"{name}.h5ad")
    pd.DataFrame({"sample": name, "cell_id": A.obs_names, "x": xy[:, 0], "y": xy[:, 1],
                  "L1_nucleus": lab.astype(str)}).to_csv(root / "labels" / f"{name}.labels.csv.gz", index=False,
                                                        compression="gzip")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("spill")
    (root / "h5ad").mkdir(); (root / "labels").mkdir()
    for u, name in enumerate(["GSMx_AIN_01_D_cortex", "GSMx_AIN_02_D_cortex", "GSMx_ATI_01_D_cortex"]):
        _section(root, name, seed=u)
    return root


def test_gradient_in_whole_not_in_nucleus(data, tmp_path):
    res = sl.spill(data / "h5ad", data / "labels", tmp_path / "o", "NAMPT", source_types=["PT"], source_gene="VCAM1",
                   radius=50, ligand="NAMPT", receptor="CXCR3", unit_pattern=r"(AIN|ATI|REF)_\d+_D",
                   group_pattern=r"(?P<g>AIN|ATI|REF)", verbose=False)
    R = res["readouts"].set_index(["unit", "arm"]).sort_index()
    for u in ("AIN_01_D", "AIN_02_D", "ATI_01_D"):
        assert R.loc[(u, "wholeB"), "diff_contact"] > 0.08
        assert abs(R.loc[(u, "nucB"), "diff_contact"]) < 0.02          # T cells hold no nuclear NAMPT
    P = res["positivity"]
    t = P[(P.type == "T_NK")].set_index(["unit", "arm"]).share_counts
    assert (t.xs("wholeB", level="arm") > t.xs("nucB", level="arm") + 0.05).all()
    I = res["interactions"]
    # PT -> T interactions exist in every arm; T -> T (NAMPT from spill-over in T cells) only outside the nucleus
    tt = I[(I.sender == "T_NK") & (I.receiver == "T_NK")].groupby("arm").n.sum()
    assert tt.get("whole", 0) > 300 and tt.get("wholeB", 0) > 20 and tt.get("nucB", 0) == 0 and tt.get("nucleus", 0) == 0
    V = sl.unit_view(res)
    assert {"positivity", "share_counts", "gradient", "interactions_per1000", "homotypic_share"} <= set(V)
    for f in ("sections.csv", "positivity.csv", "gradient.csv", "gradient_readouts.csv", "interactions.csv", "meta.json"):
        assert (tmp_path / "o" / f).exists()


def test_cli_and_refusals(data, tmp_path):
    rc = main(["compartment", "spill", "--h5ad-dir", str(data / "h5ad"), "--labels", str(data / "labels"),
               "--out", str(tmp_path / "c"), "--gene", "NAMPT", "--samples", "AIN"])
    assert rc == 0 and (tmp_path / "c" / "positivity.csv").exists()
    assert not (tmp_path / "c" / "gradient.csv").exists()
    with pytest.raises(FileExistsError):
        sl.spill(data / "h5ad", data / "labels", tmp_path / "c", "NAMPT", verbose=False)
    with pytest.raises(ValueError, match="not on the panel"):
        sl.spill(data / "h5ad", data / "labels", tmp_path / "d", "NOPE", verbose=False)
