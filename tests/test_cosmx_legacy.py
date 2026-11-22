"""compartment_to_h5ad.py reads 2021 CosMx flat files (He et al. 2022 NSCLC: cells identified by fov + cell_ID, no
cell id column, CellComp "0" for extracellular transcripts), with the pixel size given."""
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

EX = Path(__file__).resolve().parents[1] / "examples" / "compartment" / "compartment_to_h5ad.py"


def _write(raw: Path, sample: str, seed=0):
    rng = np.random.default_rng(seed)
    genes = ["CD3E", "CD8A", "EPCAM", "KRT8", "HLA-A", "COL1A1"]
    meta, tx = [], []
    for fov in (1, 2):
        for cid in range(1, 16):
            meta.append({"fov": fov, "cell_ID": cid, "Area": 2000 + cid, "AspectRatio": 1.0,
                         "CenterX_local_px": 10 * cid, "CenterY_local_px": 5 * cid,
                         "CenterX_global_px": 1000 * fov + 10 * cid, "CenterY_global_px": 5000 + 5 * cid,
                         "Width": 50, "Height": 50, "Mean.MembraneStain": 1.0, "Mean.PanCK": 2.0,
                         "Mean.CD45": 3.0, "Mean.CD3": float(cid), "Mean.DAPI": 4.0})
            for _ in range(int(rng.integers(5, 40))):
                tx.append({"fov": fov, "cell_ID": cid, "target": rng.choice(genes),
                           "CellComp": rng.choice(["Nuclear", "Cytoplasm", "Membrane"], p=[0.5, 0.3, 0.2])})
        for _ in range(20):                                    # not in a cell
            tx.append({"fov": fov, "cell_ID": 0, "target": rng.choice(genes), "CellComp": "0"})
        tx.append({"fov": fov, "cell_ID": 3, "target": "NegPrb5", "CellComp": "Nuclear"})
    T = pd.DataFrame(tx)
    T["x_local_px"] = rng.uniform(0, 5000, len(T)).round(3)            # distinct positions: file not doubled
    T["y_local_px"] = rng.uniform(0, 3000, len(T)).round(3)
    T["x_global_px"] = T.x_local_px + 1000 * T.fov
    T["y_global_px"] = T.y_local_px + 5000
    T["z"] = rng.integers(0, 8, len(T))
    T = T[["fov", "cell_ID", "x_global_px", "y_global_px", "x_local_px", "y_local_px", "z", "target", "CellComp"]]
    T.to_csv(raw / f"{sample}_tx_file.csv", index=False)
    M = pd.DataFrame(meta)
    M.to_csv(raw / f"{sample}_metadata_file.csv", index=False)
    c = T[(T.cell_ID > 0) & ~T.target.str.startswith("NegPrb")]
    E = c.groupby(["fov", "cell_ID", "target"]).size().unstack(fill_value=0).reset_index()
    E.columns = [str(x).replace("-", ".") for x in E.columns]            # exprMat writes HLA.A
    E["NegPrb5"] = 0
    E.to_csv(raw / f"{sample}_exprMat_file.csv", index=False)
    return T, M


def test_legacy_cosmx_flat_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    T, M = _write(raw, "Lung5_Rep1")
    r = subprocess.run([sys.executable, str(EX), "--platform", "cosmx", "--raw", str(raw), "--sample", "Lung5_Rep1",
                        "--out", str(tmp_path / "h5ad"), "--tmp", str(tmp_path / "tmp"), "--px-um", "0.18",
                        "--no-sha", "--min-piece-cells", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    files = sorted((tmp_path / "h5ad").glob("Lung5_Rep1_p*.h5ad"))
    A = ad.concat([ad.read_h5ad(f) for f in files])
    assert A.n_obs == len(M) == 30
    c = T[(T.cell_ID > 0) & ~T.target.str.startswith("NegPrb")]
    assert int(A.X.sum()) == len(c)
    assert int(A.layers["nucleus"].sum()) == int((c.CellComp == "Nuclear").sum())
    assert int(A.layers["cyto"].sum()) == int((c.CellComp != "Nuclear").sum())
    assert "NegPrb5" not in A.var_names and "HLA-A" in A.var_names
    # per cell: totals of a known cell, and coordinates in um at 0.18 um per pixel
    one = c[(c.fov == 2) & (c.cell_ID == 7)]
    i = list(A.obs_names).index("2_7")
    assert A.obs.n_whole.iloc[i] == len(one) and A.obs.n_nucleus.iloc[i] == (one.CellComp == "Nuclear").sum()
    m = M[(M.fov == 2) & (M.cell_ID == 7)].iloc[0]
    assert abs(A.obs.x.iloc[i] - m.CenterX_global_px * 0.18) < 1e-6 and A.obs["Mean.CD3"].iloc[i] == 7.0
    import json
    chk = json.loads(ad.read_h5ad(files[0]).uns["check"])
    assert chk["share_cells_total_equal"] == 1.0 and chk["cells_matched"] == 30
    assert "not doubled" in json.loads(ad.read_h5ad(files[0]).uns["source"])["duplicates"]["dedup"]
