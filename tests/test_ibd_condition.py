"""examples/compartment/ibd_condition.py: IELs per 100 epithelial cells by condition and labeling."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    import sys
    sys.path.insert(0, str(ROOT / "examples" / "compartment"))
    spec = importlib.util.spec_from_file_location("ibd_condition", ROOT / "examples" / "compartment" / "ibd_condition.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _section(root, name, iel_rate, lose, seed):
    """epithelium x < 300 with intraepithelial T cells at iel_rate, lamina propria beyond; whole-cell labels lose a
    share `lose` of the IELs to epithelium"""
    rng = np.random.default_rng(seed)
    n = 5000
    xy = rng.uniform(0, 800, (n, 2))
    epi = xy[:, 0] < 300
    lab = np.where(epi, np.where(rng.random(n) < iel_rate, "T_NK", "Epithelial"),
                   rng.choice(["T_NK", "Fibroblast", "Myeloid"], n, p=[0.3, 0.4, 0.3]))
    whole = lab.copy()
    lost = epi & (lab == "T_NK") & (rng.random(n) < lose)
    whole[lost] = "Epithelial"
    nn = rng.poisson(30, n)
    L = pd.DataFrame({"sample": name, "x": xy[:, 0], "y": xy[:, 1], "L1_nucleus": lab, "L1_whole": whole,
                      "n_nucleus": nn})
    L.to_csv(root / f"{name}.labels.csv.gz", index=False, compression="gzip")


def _matrix(path, units, cond):
    lines = ["!Series_title\t\"x\"",
             "!Sample_title\t" + "\t".join(f'"Ileum_CD_{c.capitalize()} [{u} ITDF00A Xenium337]"'
                                           for u, c in zip(units, cond)),
             "!Sample_geo_accession\t" + "\t".join(f'"GSM{i}"' for i in range(len(units))),
             "!Sample_characteristics_ch1\t" + "\t".join('"tissue: ileum"' for _ in units)]
    Path(path).write_text("\n".join(lines) + "\n")


def test_ibd_condition(tmp_path):
    m = _mod()
    lab = tmp_path / "lab"; lab.mkdir()
    meta = tmp_path / "meta"; meta.mkdir()
    units = [f"I03{i:02d}" for i in range(8)]
    cond = ["inflamed" if i % 2 else "non-inflamed" for i in range(8)]
    for i, (u, c) in enumerate(zip(units, cond)):
        # inflamed: twice the IELs, and whole-cell labels lose more of them (0.7 against 0.3)
        _section(lab, f"{u}_337", 0.16 if c == "inflamed" else 0.08, 0.7 if c == "inflamed" else 0.3, i)
    _section(lab, "I0300_withSegment_x", 0.1, 0.5, 99)            # excluded by default
    _matrix(meta / "GSE1_series_matrix.txt", units, cond)
    m.main(["--labels", str(lab), "--meta", str(meta), "--out", str(tmp_path / "o"), "--draws", "200"])
    P = pd.read_csv(tmp_path / "o" / "patients.csv")
    assert len(P) == 8 and set(P.inflammation) == {"inflamed", "non-inflamed"}
    C = pd.read_csv(tmp_path / "o" / "contrasts.csv")
    assert "tissue=ileum" in set(C.stratum)                     # within each level of the other characteristics
    C = C[C.stratum == "all"]
    c = C[(C.characteristic == "inflammation") & (C.readout == "iel_per100_epi")].set_index("arm")
    assert c.loc["nucleus", "reference"] == "non-inflamed"     # the unaffected level is the reference
    assert c.loc["nucleus", "ratio"] == pytest.approx(2.0, abs=0.5)
    assert c.loc["whole", "ratio"] < c.loc["nucleus", "ratio"]   # the whole cell shrinks the contrast
    assert c.loc["whole/nucleus", "ratio"] < 0.8
    assert c.loc["whole/nucleus", "hi"] < 1
    assert "tissue" not in set(C.characteristic) and "site" not in set(C.characteristic)   # one level only
    meta_j = json.loads((tmp_path / "o" / "meta.json").read_text())
    assert meta_j["skipped"] == ["I0300_withSegment_x"]
    with pytest.raises(SystemExit):
        m.main(["--labels", str(lab), "--meta", str(meta), "--out", str(tmp_path / "o")])
