"""claimbreak.calls: reference and analyst calls, and the field's random stress test on the analyst's labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from claimbreak import calls as cl
from claimbreak import placement as pl
from claimbreak.cli import main


def _sections(root: Path, units=3, n=6000, seed=3):
    """Reference labels placed at random (no spatial structure); the analyst's labels turn T cells that touch a
    tumor cell into tumor cells, which manufactures T-tumor depletion and T-T enrichment."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for u in range(units):
        xy = rng.uniform(0, 1000, (n, 2))
        ref = rng.choice(np.array(["Tumour", "T_NK", "Fibroblast"]), n, p=[0.3, 0.4, 0.3])
        _, nn = cKDTree(xy).query(xy, k=7)
        touch = (ref[nn[:, 1:]] == "Tumour").any(1)
        test = ref.copy()
        test[(ref == "T_NK") & touch & (rng.random(n) < 0.95)] = "Tumour"
        name = f"GSM{u}_I0{200 + u}"
        pd.DataFrame({"sample": name, "x": xy[:, 0], "y": xy[:, 1], "L1_nucleus": ref, "L1_whole": test}).to_csv(
            root / f"{name}.labels.csv.gz", index=False, compression="gzip")
    return root


@pytest.fixture(scope="module")
def secs(tmp_path_factory):
    return _sections(tmp_path_factory.mktemp("calls") / "labels")


def test_random_relabel_rate_and_modes():
    rng = np.random.default_rng(0)
    c = rng.integers(0, 4, 10000)
    for mode in ("uniform", "composition"):
        r = cl.random_relabel(c, 0.1, 4, rng, mode, np.bincount(c) / len(c))
        assert abs((r != c).mean() - 0.1) < 1e-9


def test_status_rules():
    assert cl.status_of(0, -1, 0.0, -0.5) == "manufactured"
    assert cl.status_of(-1, -1, -0.3, -0.6) == "exaggerated"
    assert cl.status_of(-1, -1, -0.6, -0.3) == "attenuated"
    assert cl.status_of(1, 0, 0.4, 0.1) == "lost"
    assert cl.status_of(1, -1, 0.4, -0.4) == "reversed"


def test_manufactured_depletion_certified_by_random_stress(secs, tmp_path):
    S, U, summ = cl.calls(secs, tmp_path / "out", "L1", immune=["T_NK"], partners=["Tumour"], nperm=40,
                          nperm_stress=20, reps=4, eps=(0.02, 0.05), verbose=False)
    tt = U[(U.a == "T_NK") & (U.b == "Tumour")]
    assert (tt.call_ref == "neither").all() and (tt.call_test == "depleted").all()
    assert (tt.status == "manufactured").all()
    assert (tt.random_toward_zero).all() and (tt.random_robust).all() and (tt.certified_wrongly).all()
    hom = U[(U.a == "T_NK") & (U.b == "T_NK")]
    assert (hom.L_test > hom.L_ref).all() and (hom.dL_random_5 < 0).all()
    s = summ[(summ.a == "T_NK") & (summ.b == "Tumour")].iloc[0]
    assert s.units == 3 and s.manufactured == 3 and s.certified_wrongly == 3
    # at a higher random rate the diluted T/NK population loses the call: the stress test can also fail a claim
    S2, U2, _ = cl.calls(secs, tmp_path / "out15", "L1", immune=["T_NK"], partners=["Tumour"], nperm=40,
                         nperm_stress=20, reps=4, eps=(0.15,), verbose=False)
    t2 = U2[(U2.a == "T_NK") & (U2.b == "Tumour")]
    assert (t2.persist_15 < 0.9).all() and not t2.certified_wrongly.any()
    for f in ("sections.csv", "units.csv", "summary.csv", "meta.json"):
        assert (tmp_path / "out" / f).exists()
    with pytest.raises(FileExistsError):
        cl.calls(secs, tmp_path / "out", "L1", verbose=False)


def test_cli(secs, tmp_path):
    rc = main(["compartment", "calls", "--labels", str(secs), "--out", str(tmp_path / "c"), "--immune", "T_NK",
               "--partners", "Tumour,Fibroblast", "--nperm", "20", "--nperm-stress", "10", "--reps", "2",
               "--eps", "0.1", "--mode", "composition"])
    assert rc == 0
    m = json.loads((tmp_path / "c" / "meta.json").read_text())
    assert m["mode"] == "composition" and m["eps"] == [0.1]
    assert "persist_10" in pd.read_csv(tmp_path / "c" / "units.csv")


def test_placement_neighbours_from_another_arm(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_placement import _labels
    lab = _labels(tmp_path / "lab", units=3, n=1500)
    r = pl.placement(lab, tmp_path / "pl", "L1", ["T_NK"], ["Epithelial"], nbr_arm="wholeB", draws=2, min_cells=50,
                     transfer=(), verbose=False)
    assert json.loads((tmp_path / "pl" / "meta.json").read_text())["nbr_arm"] == "wholeB"
    assert "louo_full" in r["arms"]
