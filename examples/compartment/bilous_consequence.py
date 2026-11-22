"""What organized label errors do to colocalization, against random relabeling at the same confusion rates
(GSE311609).

Input: <sample>.labels.csv.gz from bilous_label.py (first run, arms 'nucleus' and 'whole'). For each section, cells
labeled in both arms; kNN graph (k = --k) on the cell centroids, fixed for all label sets. Label sets:
  ref       nucleus labels (fewer neighbor transcripts; not ground truth)
  test      whole-cell labels (what an analysis would use)
  typelevel ref labels relabeled at random with the section's own confusion matrix P(test label | ref label),
            independently of position (the random-relabeling stress test at exactly the observed rates; --draws
            draws, averaged)
Colocalization of an ordered pair (a, b): log2 of the mean number of b among the k neighbors of a-cells over its
expectation k * n_b / (n - 1) (the label-permutation expectation).
Outputs (OUT/): coloc.csv (sample, a, b, coloc_ref, coloc_test, coloc_typelevel, n_a_ref, n_b_ref), and a printed
summary: for each pair, the median over sections of delta_test = test - ref and delta_type = typelevel - ref, with
the number of sections in which delta_test < delta_type.
Usage: python bilous_consequence.py --labels ~/claimbreak_data/Bilous_labels --out ~/claimbreak_results/bilous_conseq
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--level", default="L2m")
ap.add_argument("--ref", default="nucleus")
ap.add_argument("--test", default="whole")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--draws", type=int, default=20)
ap.add_argument("--min-cells", type=int, default=200, help="types with fewer reference cells in a section are skipped")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out).expanduser()
if out.exists() and any(out.iterdir()):
    raise SystemExit(f"{out} exists and is not empty")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)


def coloc(lab, idx, types):
    """log2 O/E for all ordered pairs; lab: int codes (-1 = none)."""
    n = len(lab)
    K = len(types)
    cnt = np.bincount(lab, minlength=K).astype(float)
    nb = lab[idx]                       # cells x k
    M = np.zeros((K, K))
    for j in range(idx.shape[1]):
        np.add.at(M, (lab, nb[:, j]), 1)
    O = M / np.maximum(cnt[:, None], 1)
    E = idx.shape[1] * (cnt[None, :] - np.eye(K)) / max(n - 1, 1)     # the cell is not its own neighbor
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(O / E), cnt


rows = []
for f in sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz")):
    cols = ["sample", "x", "y", f"{a.level}_{a.ref}", f"{a.level}_{a.test}"]
    D = pd.read_csv(f, usecols=cols, keep_default_na=False)
    D = D[(D[cols[3]] != "") & (D[cols[4]] != "")].reset_index(drop=True)
    sample = D["sample"].iloc[0]
    types = np.array(sorted(set(D[cols[3]]) | set(D[cols[4]])))
    code = {t: i for i, t in enumerate(types)}
    r = D[cols[3]].map(code).to_numpy()
    t = D[cols[4]].map(code).to_numpy()
    K = len(types)
    _, idx = cKDTree(D[["x", "y"]].to_numpy(float)).query(D[["x", "y"]].to_numpy(float), k=a.k + 1)
    idx = idx[:, 1:]
    C = np.zeros((K, K))
    np.add.at(C, (r, t), 1)
    Prow = C / np.maximum(C.sum(1, keepdims=True), 1)
    cr, cnt_r = coloc(r, idx, types)
    ct, _ = coloc(t, idx, types)
    acc = np.zeros((K, K))
    cum = np.cumsum(Prow, 1)
    for _ in range(a.draws):
        u = rng.random(len(r))
        tl = (u[:, None] > cum[r]).sum(1)
        tl = np.minimum(tl, K - 1)
        c_, _ = coloc(tl, idx, types)
        acc += c_
    cl = acc / a.draws
    dis = float(np.mean(r != t))
    print(f"{sample}: {len(D):,} cells, disagreement {dis:.3f}", flush=True)
    for i in range(K):
        for j in range(K):
            if cnt_r[i] < a.min_cells or cnt_r[j] < a.min_cells:
                continue
            rows.append(dict(sample=sample, a=types[i], b=types[j], coloc_ref=cr[i, j], coloc_test=ct[i, j],
                             coloc_typelevel=cl[i, j], n_a_ref=int(cnt_r[i]), n_b_ref=int(cnt_r[j]),
                             disagreement=dis))
R = pd.DataFrame(rows)
R.to_csv(out / "coloc.csv", index=False)
R["d_test"] = R.coloc_test - R.coloc_ref
R["d_type"] = R.coloc_typelevel - R.coloc_ref
S = R.groupby(["a", "b"]).agg(sections=("d_test", "size"), d_test=("d_test", "median"), d_type=("d_type", "median"),
                              test_below_type=("d_test", lambda v: int((v < R.loc[v.index, "d_type"]).sum()))).reset_index()
S.to_csv(out / "summary.csv", index=False)
focus = S[S.a.isin(["T_NK", "Myeloid", "B"]) & S.b.isin(["Tumour", "Fibro_muscle", "Vessel"])]
print("\nimmune -> partner colocalization change (median over sections; log2):")
print(focus.round(3).to_string(index=False))
hom = S[S.a == S.b]
print("\nhomotypic:")
print(hom.round(3).to_string(index=False))
print(f"\nall pairs: median |d_test| {R.d_test.abs().median():.3f}, median |d_type| {R.d_type.abs().median():.3f}")
print(f"wrote {out}/coloc.csv, summary.csv")
