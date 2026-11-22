#!/usr/bin/env python
"""resolvi_rcc.py -- resolVI's semi-supervised relabeling on renal carcinoma windows, for scoring against protein
(a correction readers will ask about after SPLIT, RCTD and cellAdmix).

resolVI (Ergen & Yosef, Nat Methods 23:2003-2014, 2026; scvi-tools) models each cell's whole-cell counts as its own
expression, its neighbors' expression and background, and in semi-supervised mode relabels every cell. We follow the
authors' published relabeling route (their colitis Methods), on the labels a user would have (our whole-cell arm):
  1. unsupervised resolVI on the window's whole-cell counts;
  2. k-nearest-neighbor graph (k = 100, scanpy, connectivities as weights) in its latent space; a cell's certainty is
     the connectivity carrying its own label over its total connectivity; labels with uncertainty above 5% (the
     authors' threshold for coarse labels) are set to 'unknown';
  3. semi-supervised resolVI on the remaining labels; every cell relabeled with its classifier (predict).
Windows: square windows of about --window-cells cells on a grid over the section; the --windows windows holding the
most cells labeled T/NK by the training arm (the same cells are then scored for every arm). Training settings are
the package defaults (50 epochs, KL warm-up over 20) unless given.

Writes OUT/<sample>.resolvi.csv.gz (cell_id, window, <level>_rvTrain, rv_certainty, <level>_rvSemi) and
OUT/settings.json (versions, windows, times). Scoring against protein: resolvi_rcc.sh score.

  python resolvi_rcc.py --h5ad XP_RCC.h5ad --labels labels/XP_RCC.labels.csv.gz --out OUT [--windows 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def windows(xy, lab, focus, n_target, n_windows, min_share=0.5):
    """Grid windows with about n_target cells each; the n_windows with most cells labeled `focus`."""
    lo, hi = xy.min(0), xy.max(0)
    area = float(np.prod(hi - lo))
    side = np.sqrt(n_target * area / len(xy))
    ij = np.floor((xy - lo) / side).astype(int)
    key = pd.Series([f"w{i:02d}_{j:02d}" for i, j in ij])
    G = pd.DataFrame({"w": key, "f": (lab == focus)}).groupby("w").agg(cells=("f", "size"), focus=("f", "sum"))
    G = G[G.cells >= min_share * n_target].sort_values(["focus", "cells"], ascending=False)
    pick = list(G.index[:n_windows])
    return key.to_numpy(), pick, G, side


def certainty_of_own_label(C, lab):
    """Share of each cell's kNN connectivity that carries its own label."""
    C = sp.csr_matrix(C)
    codes, uniq = pd.factorize(pd.Series(lab))
    M = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(uniq)))
    per = np.asarray((C @ M).todense())
    tot = per.sum(1)
    own = per[np.arange(len(codes)), codes]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, own / tot, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True, help="compartment .h5ad: whole-cell counts in X, obs x, y")
    ap.add_argument("--labels", required=True, help="claimbreak labels file of the same section")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="L1")
    ap.add_argument("--train-arm", default="whole", help="the labels resolVI starts from (a user's whole-cell labels)")
    ap.add_argument("--focus", default="T_NK", help="type used to choose windows")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--window-cells", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-uncertainty", type=float, default=0.05)
    ap.add_argument("--num-samples", type=int, default=30, help="posterior samples for predict (package default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows-cells", type=int, default=0, help="testing: cap cells per window (0: none)")
    a = ap.parse_args()
    import anndata as ad
    import scanpy as sc
    import scvi
    import torch
    from scvi.external import RESOLVI
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    scvi.settings.seed = a.seed
    A = ad.read_h5ad(Path(a.h5ad).expanduser())
    sample = Path(a.h5ad).name.replace(".h5ad", "")
    L = pd.read_csv(Path(a.labels).expanduser(), keep_default_na=False, low_memory=False, dtype={"cell_id": str},
                    usecols=["cell_id", f"{a.level}_{a.train_arm}"])
    lab = L.set_index("cell_id")[f"{a.level}_{a.train_arm}"].reindex(A.obs_names.astype(str)).fillna("").to_numpy()
    xy = A.obs[["x", "y"]].to_numpy(float)
    key, pick, G, side = windows(xy, lab, a.focus, a.window_cells, a.windows)
    picked = ", ".join(f"{G.loc[w, 'cells']:,} cells, {int(G.loc[w, 'focus']):,} {a.focus}" for w in pick)
    print(f"{sample}: {A.n_obs:,} cells; windows of {side:.0f} um; picked {pick} ({picked})", flush=True)
    info = dict(sample=sample, scvi_tools=scvi.__version__, torch=torch.__version__,
                device=str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
                train_arm=a.train_arm, k=a.k, max_uncertainty=a.max_uncertainty, epochs=a.epochs,
                num_samples=a.num_samples, seed=a.seed, window_side_um=float(side), windows={})
    rows = []
    for w in pick:
        m = key == w
        W = A[m].copy()
        if a.max_windows_cells and W.n_obs > a.max_windows_cells:
            W = W[np.random.default_rng(a.seed).choice(W.n_obs, a.max_windows_cells, replace=False)].copy()
        W.X = sp.csr_matrix(W.X).astype(np.float32)
        W.layers["counts"] = W.X.copy()
        W.obsm["X_spatial"] = W.obs[["x", "y"]].to_numpy(float)
        W.obsm["spatial"] = W.obsm["X_spatial"]
        wl = pd.Series(lab, index=A.obs_names.astype(str)).reindex(W.obs_names.astype(str)).to_numpy()
        keep_cells = np.asarray(W.X.sum(1)).ravel() > 0
        W = W[keep_cells].copy()
        wl = wl[keep_cells]
        t0 = time.time()
        # 1. unsupervised model
        RESOLVI.setup_anndata(W, layer="counts")
        m0 = RESOLVI(W)
        m0.train(max_epochs=a.epochs)
        Z = m0.get_latent_representation()
        t1 = time.time()
        # 2. kNN in the latent space, certainty of the cell's own label
        W.obsm["X_rv"] = Z
        sc.pp.neighbors(W, n_neighbors=a.k, use_rep="X_rv", random_state=a.seed)
        cert = certainty_of_own_label(W.obsp["connectivities"], np.where(wl == "", "unknown", wl))
        train = np.where((wl != "") & (1 - cert <= a.max_uncertainty), wl, "unknown")
        # 3. semi-supervised model on the confident labels, every cell relabeled
        S = ad.AnnData(W.X.copy(), obs=W.obs[[]].copy(), var=W.var[[]].copy())
        S.layers["counts"] = S.X.copy()
        S.obsm["X_spatial"] = W.obsm["X_spatial"]
        S.obsm["spatial"] = W.obsm["X_spatial"]
        S.obs["rv_train"] = pd.Categorical(train)
        RESOLVI.setup_anndata(S, layer="counts", labels_key="rv_train", unlabeled_category="unknown")
        m1 = RESOLVI(S, semisupervised=True)
        m1.train(max_epochs=a.epochs)
        pred = np.asarray(m1.predict(S, soft=False, num_samples=a.num_samples)).ravel().astype(str)
        t2 = time.time()
        kept = float((train != "unknown").mean())
        same = float(np.mean(pred[wl != ""] == wl[wl != ""]))
        info["windows"][w] = dict(cells=int(W.n_obs), labels_kept_for_training=kept, agreement_with_start=same,
                                  minutes_unsupervised=(t1 - t0) / 60, minutes_semisupervised=(t2 - t1) / 60)
        print(f"  {w}: {W.n_obs:,} cells; training labels kept {kept:.3f}; resolVI agrees with the starting labels "
              f"{same:.3f}; {(t1 - t0) / 60:.1f} + {(t2 - t1) / 60:.1f} min", flush=True)
        rows.append(pd.DataFrame({"cell_id": W.obs_names.astype(str), "window": w,
                                  f"{a.level}_rvTrain": np.where(train == "unknown", "", train),
                                  "rv_certainty": cert, f"{a.level}_rvSemi": pred}))
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(out / f"{sample}.resolvi.csv.gz", index=False, compression="gzip")
    (out / "settings.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {out}/{sample}.resolvi.csv.gz ({len(R):,} cells) and settings.json")


if __name__ == "__main__":
    main()
