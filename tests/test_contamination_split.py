"""contamination.py --split-typing: typing a cell on the counts on which its nucleus foreign share is
measured biases that share low; typing on one nuclear half and measuring on the other does not."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from claimbreak.cli import main
from test_compartment import ARMS, EX, _make


def test_split_typing_removes_selection_bias(tmp_path):
    d = _make(tmp_path)
    assert main(["compartment", "label", "--ref", str(d / "ref.h5ad"), "--inputs", str(d / "sec0.h5ad"),
                 str(d / "sec1.h5ad"), "--arms", ARMS, "--seed", "3", "--out", str(d / "lab")]) == 0
    res = {}
    for tag, extra in (("plain", []), ("split", ["--split-typing"])):
        r = subprocess.run([sys.executable, str(EX / "contamination.py"), "--h5ad-dir", str(d), "--labels",
                            str(d / "lab"), "--ref", str(d / "ref.h5ad"), "--unit-pattern", r"sec\d",
                            "--out", str(d / f"ct_{tag}"), *extra], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        C = pd.read_csv(d / f"ct_{tag}" / "cells.csv.gz")
        w = C.marker_nucleus
        res[tag] = float((C.foreign_nucleus.fillna(0) * w).sum() / w.sum())
        assert C.foreign_cyto.notna().any()
        S = pd.read_csv(d / f"ct_{tag}" / "source.csv")
        assert {"share_of_far_cyto", "share_of_section_cyto"} <= set(S.columns)
        assert "nulls: far neighbours" in r.stdout
    # synthetic nuclei hold only their own profile: typing on the measured counts can only lower the nucleus share
    # (selection), so the split estimate is not below the plain one beyond noise
    assert res["split"] >= res["plain"] - 0.02, res


def test_rho_split_and_pooled_shares(tmp_path):
    d = _make(tmp_path)
    assert main(["compartment", "label", "--ref", str(d / "ref.h5ad"), "--inputs", str(d / "sec0.h5ad"),
                 str(d / "sec1.h5ad"), "--arms", ARMS, "--seed", "3", "--out", str(d / "lab")]) == 0
    r = subprocess.run([sys.executable, str(EX / "contamination.py"), "--h5ad-dir", str(d), "--labels",
                        str(d / "lab"), "--ref", str(d / "ref.h5ad"), "--unit-pattern", r"sec\d", "--split-typing",
                        "--rho-split", "--out", str(d / "ct")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    C = pd.read_csv(d / "ct" / "cells.csv.gz")
    assert {"rho_c1", "flipped_split", "foreign_cyto_n", "marker_cyto_n"} <= set(C.columns)
    assert C.flipped_split.notna().mean() > 0.3
    P = pd.read_csv(d / "ct" / "pooled.csv")
    g = C.groupby(["sample", "type", "touch_other"])
    want = (g.foreign_whole_n.sum() / g.marker_whole_n.sum()).reset_index(name="w")
    m = P.merge(want, on=["sample", "type", "touch_other"])
    assert len(m) and np.allclose(m.pooled_foreign_whole, m.w, equal_nan=True)
    PU = pd.read_csv(d / "ct" / "pooled_units.csv")
    assert any(c.startswith("contact_adds_") for c in PU.columns) or len(PU)
    F = pd.read_csv(d / "ct" / "flip_curve_split.csv")
    assert {"flipped_split", "null", "log10_rho_mid"} <= set(F.columns)
    assert "pooled over cells" in r.stdout


def test_split_typing_uses_the_labelling_exclusions(tmp_path):
    """a gene the labeling excluded (an over-detected probe, as CCL5 in synovium) is not used to type cells
    in --split-typing either, so relabeling without it changes the foreign shares."""
    import anndata as ad
    from test_label_audit import _make as make_hot
    (tmp_path / "hot").mkdir()
    d = make_hot(tmp_path / "hot", hot=True)
    common = ["--ref", str(d / "ref.h5ad"), "--ref-levels", "L1=coarse", "--inputs", str(d / "sec0.h5ad"),
              str(d / "sec1.h5ad"), "--arms", "whole,nucleus,nucA,nucB,wholeB", "--seed", "2"]
    (d / "excl.txt").write_text("g0\n")
    assert main(["compartment", "label", *common, "--exclude-genes", str(d / "excl.txt"), "--out", str(d / "lab")]) == 0
    T = {}
    for tag, extra in (("new", []), ("old", ["--ignore-label-settings"])):
        r = subprocess.run([sys.executable, str(EX / "contamination.py"), "--h5ad-dir", str(d), "--labels",
                            str(d / "lab"), "--ref", str(d / "ref.h5ad"), "--ref-levels", "L1=coarse", "--ref-merge=",
                            "--level", "L1", "--unit-pattern", r"sec\d", "--split-typing",
                            "--out", str(d / f"ct_{tag}"), *extra], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        C = pd.read_csv(d / f"ct_{tag}" / "cells.csv.gz", dtype={"cell_id": str})
        truth = pd.concat([ad.read_h5ad(d / f"sec{i}.h5ad").obs["truth"] for i in (0, 1)])
        C["truth"] = truth.reindex(C.cell_id).to_numpy()
        T[tag] = float((C[C.truth == "Fib"].type == "T").mean())
    s = __import__("json").loads((d / "ct_new" / "settings.json").read_text())
    assert s["samples"]["sec0"]["excluded"] == ["g0"]
    assert T["new"] < 0.1 < T["old"], T                          # the probe no longer types fibroblasts as T
