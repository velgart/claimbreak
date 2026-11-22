"""examples/compartment/protein_contacts.py: T-cell contacts with protein-typed partners, T called by protein
and by each label arm, with paired block intervals."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("protein_contacts", ROOT / "examples" / "compartment" / "protein_contacts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _section(root, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1200, (n, 2))
    true = np.where(xy[:, 0] < 500, "Tumour", rng.choice(["T", "B", "Stroma"], n))
    true[(xy[:, 0] < 500) & (rng.random(n) < 0.15)] = "T"
    cells = [f"c{i}" for i in range(n)]
    sig = {"CD3E": true == "T", "CD45": np.isin(true, ["T", "B"]), "CD68": np.zeros(n, bool), "PanCK": true == "Tumour"}
    P = pd.DataFrame({"cell_id": cells})
    for p, s in sig.items():
        P[p] = np.where(s, rng.lognormal(5, 0.4, n), rng.lognormal(0.5, 0.8, n) * (rng.random(n) < 0.4)).round()
    P.to_csv(root / "XP_T.protein.csv.gz", index=False, compression="gzip")
    from scipy.spatial import cKDTree
    _, nn = cKDTree(xy).query(xy, k=7)
    touch_tum = (true[nn[:, 1:]] == "Tumour").any(1)
    lab_map = {"Tumour": "PT", "T": "T_NK", "B": "B", "Stroma": "Stromal"}
    nuc = np.array([lab_map[t] for t in true])
    whole = nuc.copy()
    whole[(true == "T") & touch_tum & (rng.random(n) < 0.6)] = "PT"
    nuc_lab = nuc.copy()
    nuc_lab[rng.random(n) < 0.1] = ""                      # cells without a nuclear label
    L = pd.DataFrame({"sample": "XP_T", "cell_id": cells, "x": xy[:, 0], "y": xy[:, 1], "L1_nucleus": nuc_lab,
                      "L1_whole": whole})
    (root / "lab").mkdir()
    L.to_csv(root / "lab" / "XP_T.labels.csv.gz", index=False, compression="gzip")
    return root


def test_protein_contacts(tmp_path):
    m = _mod()
    root = _section(tmp_path)
    m.main(["--labels", str(root / "lab"), "--protein", str(root / "XP_T.protein.csv.gz"), "--sample", "XP_T",
            "--out", str(tmp_path / "o"), "--arms", "nucleus,whole,nucB", "--block-draws", "100"])
    R = pd.read_csv(tmp_path / "o" / "contacts.csv").set_index(["variant", "arm", "metric"])
    C = pd.read_csv(tmp_path / "o" / "counts.csv").set_index(["variant", "arm"])
    meta = json.loads((tmp_path / "o" / "meta.json").read_text())
    assert meta["missing_arms"] == ["nucB"] and meta["arms"] == ["nucleus", "whole", "nucleus+whole"]
    assert meta["t_labels"] == ["T_NK"]
    # per100 is 100 * touching T / partners
    c = C.loc[("all", "protein")]
    assert R.loc[("all", "protein", "per100"), "estimate"] == pytest.approx(100 * c.touching / c.n_partner)
    # whole-cell labels lose T cells at contact; on common cells the nuclear arm matches protein
    assert R.loc[("all", "whole", "per100_ratio_to_protein"), "estimate"] < 0.9
    assert R.loc[("common", "nucleus", "per100_ratio_to_protein"), "estimate"] == pytest.approx(1, abs=0.03)
    assert R.loc[("all", "whole", "contact_share"), "estimate"] < R.loc[("all", "protein", "contact_share"), "estimate"]
    # the hybrid fills the unlabeled nuclei from whole-cell labels: between the two
    h = R.loc[("all", "nucleus+whole", "per100"), "estimate"]
    assert R.loc[("all", "whole", "per100"), "estimate"] < h <= R.loc[("all", "protein", "per100"), "estimate"] * 1.01
    # arms against each other, from the same draws
    w, n_ = R.loc[("all", "whole", "per100"), "estimate"], R.loc[("all", "nucleus", "per100"), "estimate"]
    assert R.loc[("all", "whole", "per100_ratio_to_nucleus"), "estimate"] == pytest.approx(w / n_)
    assert R.loc[("common", "whole", "per100_ratio_to_nucleus"), "estimate"] < 0.95
    lo, hi = R.loc[("all", "whole", "per100_ratio_to_protein"), ["lo", "hi"]]
    assert lo < R.loc[("all", "whole", "per100_ratio_to_protein"), "estimate"] < hi
    with pytest.raises(SystemExit):
        m.main(["--labels", str(root / "lab"), "--protein", str(root / "XP_T.protein.csv.gz"), "--sample", "XP_T",
                "--out", str(tmp_path / "o")])


def test_partner_parse():
    m = _mod()
    assert m.parse_partner("CD45-,PanCK+") == [("CD45", False), ("PANCK", True)]
    with pytest.raises(SystemExit):
        m.parse_partner("CD45")


def test_confusion_and_label_partners(tmp_path):
    """the same changes placed at random (arm conf) and partners typed by labels."""
    m = _mod()
    root = _section(tmp_path)
    L = pd.read_csv(root / "lab" / "XP_T.labels.csv.gz", keep_default_na=False)
    L["L1_nucB"], L["L1_wholeB"] = L.L1_nucleus, L.L1_whole
    L.to_csv(root / "lab" / "XP_T.labels.csv.gz", index=False, compression="gzip")
    common = ["--labels", str(root / "lab"), "--protein", str(root / "XP_T.protein.csv.gz"), "--sample", "XP_T",
              "--block-draws", "50", "--confusion", "5"]
    m.main(common + ["--out", str(tmp_path / "p"), "--partner", "PANCK+"])
    m.main(common + ["--out", str(tmp_path / "l"), "--partner-from", "labels"])
    m.main(common + ["--out", str(tmp_path / "n"), "--partner-from", "nucB"])       # partners fixed at half B
    for d in ("p", "l", "n"):
        R = pd.read_csv(tmp_path / d / "contacts.csv").set_index(["variant", "arm", "metric"])
        ch = R.loc[("all", "wholeB", "change_pct"), "estimate"]
        cf = R.loc[("all", "conf", "change_pct"), "estimate"]
        pos = R.loc[("all", "wholeB", "position_pct"), "estimate"]
        assert pos == pytest.approx(ch - cf, abs=1e-6)
        assert ch < 0 and pos < 0          # T cells lost at tumor contact: beyond the same changes placed at random
    R = pd.read_csv(tmp_path / "l" / "contacts.csv")
    assert "protein" not in set(R.arm)
    assert json.loads((tmp_path / "l" / "meta.json").read_text())["partner_from"] == "labels"
