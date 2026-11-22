"""Slide-GoTags negative control (examples/compartment/slidetags_tcr.py) on synthetic portal files."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "compartment" / "slidetags_tcr.py"


def _write(tmp, loss_at_contact):
    rng = np.random.default_rng(0)
    rows, meta = [], []
    for s in range(3):
        sec = f"RCC_{s}_A"
        n = 8000
        xy = rng.uniform(0, 1600, (n, 2))
        tum = xy[:, 0] < 800
        lab = np.where(tum, "Tumor", "Fibroblast").astype(object)
        it = rng.choice(n, 400, replace=False)
        lab[it] = "T cells"
        tcr = np.zeros(n, bool); tcr[it] = rng.random(400) < 0.5
        near = xy[:, 0] < 700                      # T cells inside the tumor half: in contact with tumor
        if loss_at_contact:
            lost = tcr & near & (rng.random(n) < 0.6)
            lab[lost] = "Tumor"
        for i in range(n):
            name = f"{sec}_BC{i}-1"
            rows.append([name, xy[i, 0], xy[i, 1], lab[i], "CAVR" if tcr[i] else "", ""])
            meta.append([name, sec, lab[i]])
    S = pd.DataFrame(rows, columns=["NAME", "X", "Y", "cell_type_original", "TRA_cdr3", "TRB_cdr3"])
    M = pd.DataFrame(meta, columns=["NAME", "biosample_id", "Merged_Cluster_Name_scevan_broad"])
    for df, f, types in ((S, "spatial.txt", ["TYPE", "numeric", "numeric", "group", "group", "group"]),
                         (M, "meta.txt", ["TYPE", "group", "group"])):
        with open(tmp / f, "w") as fh:
            fh.write("\t".join(df.columns) + "\n" + "\t".join(types) + "\n")
            df.to_csv(fh, sep="\t", header=False, index=False)


def _run(tmp, name):
    out = tmp / name
    r = subprocess.run([sys.executable, str(SCRIPT), "run", "--spatial", str(tmp / "spatial.txt"), "--meta",
                        str(tmp / "meta.txt"), "--out", str(out), "--t-regex", "^T cells", "--tumor-regex", "Tumor",
                        "--boot", "200"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads((out / "summary.json").read_text())


def test_no_loss_gives_flat_recall(tmp_path):
    _write(tmp_path, False)
    S = _run(tmp_path, "flat")
    assert S["recall_contact"] == 1.0 and S["recall_no_contact"] == 1.0


def test_planted_loss_at_contact_is_found(tmp_path):
    _write(tmp_path, True)
    S = _run(tmp_path, "loss")
    assert S["recall_contact"] < S["recall_no_contact"] - 0.2
    assert S["mh_or_T_contact_vs_not"] < 0.5
    assert S["tumor_label_contact"] > S["tumor_label_no_contact"]


def _write_mtx(tmp, contaminate):
    """Portal-like files with a matrix: genes 0-19 tumor-specific, 20-29 T genes (named CD3E...), 30-59 shared.
    With contaminate=True, T nuclei touching tumor carry tumor-gene counts (a whole-cell mask)."""
    import scipy.io
    import scipy.sparse as sp
    rng = np.random.default_rng(1)
    tg = ["CD3E", "CD3D", "CD3G", "CD2", "CD247", "TRAC", "LCK", "ZAP70", "BCL11B", "ITK"]
    genes = [f"TUM{i}" for i in range(20)] + tg + [f"G{i}" for i in range(30)]
    rows, meta, prof = [], [], []
    for s in range(3):
        sec = f"MEL_{s}"
        n = 4000
        xy = rng.uniform(0, 1600, (n, 2))
        lab = np.where(xy[:, 0] < 800, "Tumor", "Fibroblast").astype(object)
        lab[rng.choice(n, 600, replace=False)] = "T-cell"
        for i in range(n):
            name = f"{sec}_BC{i}-1"
            rows.append([name, xy[i, 0], xy[i, 1], lab[i], "", "", "MUT" if lab[i] == "Tumor" and rng.random() < .3 else ""])
            meta.append([name, sec, lab[i]])
            lam = np.full(60, 0.3)
            lam[:20] = 0.08
            if lab[i] == "Tumor":
                lam[:20] = 3.0
            elif lab[i] == "T-cell":
                lam[20:30] = 3.0
                if contaminate and xy[i, 0] < 780:
                    lam[:20] += 0.3
            prof.append(rng.poisson(lam))
    S = pd.DataFrame(rows, columns=["NAME", "X", "Y", "cell_type_original", "TRA_cdr3", "TRB_cdr3", "BRAF"])
    M = pd.DataFrame(meta, columns=["NAME", "biosample_id", "Merged_Cluster_Name_scevan_broad"])
    for df, f, types in ((S, "spatial.txt", ["TYPE", "numeric", "numeric", "group", "group", "group", "group"]),
                         (M, "meta.txt", ["TYPE", "group", "group"])):
        with open(tmp / f, "w") as fh:
            fh.write("\t".join(df.columns) + "\n" + "\t".join(types) + "\n")
            df.to_csv(fh, sep="\t", header=False, index=False)
    X = sp.csr_matrix(np.array(prof).T.astype(float))                  # genes x cells
    scipy.io.mmwrite(str(tmp / "m.mtx"), X, field="integer")
    pd.DataFrame({"id": genes, "name": genes}).to_csv(tmp / "f.tsv", sep="\t", header=False, index=False)
    pd.DataFrame({"bc": S.NAME}).to_csv(tmp / "b.tsv", sep="\t", header=False, index=False)


def _contam(tmp, name):
    out = tmp / name
    r = subprocess.run([sys.executable, str(SCRIPT), "contamination", "--spatial", str(tmp / "spatial.txt"),
                        "--meta", str(tmp / "meta.txt"), "--mtx", str(tmp / "m.mtx"), "--features",
                        str(tmp / "f.tsv"), "--barcodes", str(tmp / "b.tsv"), "--out", str(out), "--t-regex",
                        "^T-cell$", "--tumor-regex", "^Tumor$", "--n-markers", "20", "--min-ratio", "5", "--boot", "200", "--genotype",
                        "--chunk", "50000"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads((out / "summary.json").read_text())


def test_contamination_flat_without_masks(tmp_path):
    _write_mtx(tmp_path, False)
    S = _contam(tmp_path, "flat")
    assert 0.8 < S["T_tumorload"]["median_ratio_contact_far"] < 1.25
    assert S["integer_counts"]


def test_contamination_found_with_masks(tmp_path):
    _write_mtx(tmp_path, True)
    S = _contam(tmp_path, "mask")
    assert S["T_tumorload"]["median_ratio_contact_far"] > 2
    assert S["T_tumorload"]["n_sections_ratio_gt1"] == 3
