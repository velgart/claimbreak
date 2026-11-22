"""Tissue container, loading, neighbor graph, count splitting and thinning.

A Tissue holds integer counts, positions, reference labels ("truth" for the experiment), section, independent
unit (mouse, donor, patient) and optional condition. Perturbations return new Tissue objects; the neighbor graph
depends only on positions and sections and is shared between copies.
"""
from __future__ import annotations

import dataclasses
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


class Graph:
    """k-nearest-neighbor graph within sections (global indices), computed lazily per k and cached."""

    def __init__(self, xy: np.ndarray, section: np.ndarray):
        self.xy = xy
        self.section = section
        self._knn: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._trees: dict = {}
        self._sec_idx = {s: np.flatnonzero(section == s) for s in pd.unique(section)}

    def tree(self, s):
        if s not in self._trees:
            self._trees[s] = cKDTree(self.xy[self._sec_idx[s]])
        return self._trees[s]

    def section_index(self, s) -> np.ndarray:
        return self._sec_idx[s]

    def knn(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (idx, dist), each (n, k): the k nearest other cells of the same section."""
        if k not in self._knn:
            n = len(self.xy)
            idx = np.empty((n, k), dtype=np.int64)
            dist = np.empty((n, k), dtype=np.float64)
            for s, ii in self._sec_idx.items():
                kk = min(k + 1, len(ii))
                d, j = self.tree(s).query(self.xy[ii], k=kk)
                d, j = np.atleast_2d(d)[:, 1:], np.atleast_2d(j)[:, 1:]
                if kk - 1 < k:  # tiny section: pad by repeating the last neighbor
                    pad = k - (kk - 1)
                    d = np.hstack([d, np.repeat(d[:, -1:], pad, 1)])
                    j = np.hstack([j, np.repeat(j[:, -1:], pad, 1)])
                idx[ii] = ii[j]
                dist[ii] = d
            self._knn[k] = (idx, dist)
        return self._knn[k]


@dataclasses.dataclass
class Tissue:
    counts: sp.csr_matrix          # cells x genes, non-negative integers
    xy: np.ndarray                 # (n, 2)
    labels: np.ndarray             # reference labels (str)
    section: np.ndarray            # section id per cell
    unit: np.ndarray               # independent unit per cell (mouse, donor, patient)
    genes: np.ndarray
    condition: np.ndarray | None = None
    obs_names: np.ndarray | None = None
    graph: Graph | None = None
    meta: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not sp.isspmatrix_csr(self.counts):
            self.counts = sp.csr_matrix(self.counts)
        self.labels = np.asarray(self.labels).astype(str)
        self.section = np.asarray(self.section).astype(str)
        self.unit = np.asarray(self.unit).astype(str)
        if self.condition is not None:
            self.condition = np.asarray(self.condition).astype(str)
        if self.graph is None:
            self.graph = Graph(np.asarray(self.xy, dtype=float), self.section)

    @property
    def n(self) -> int:
        return self.counts.shape[0]

    @property
    def types(self) -> np.ndarray:
        return np.array(sorted(pd.unique(self.labels)))

    def replace(self, **kw) -> "Tissue":
        return dataclasses.replace(self, **kw)

    def codes(self, labels=None, types=None) -> tuple[np.ndarray, np.ndarray]:
        """Integer codes of labels against a fixed type list (default: the reference types)."""
        types = self.types if types is None else np.asarray(types)
        lab = self.labels if labels is None else np.asarray(labels).astype(str)
        lut = {t: i for i, t in enumerate(types)}
        return np.array([lut[x] for x in lab]), types

    def summary(self) -> str:
        tot = np.asarray(self.counts.sum(1)).ravel()
        lines = [f"{self.n} cells, {self.counts.shape[1]} genes, {len(pd.unique(self.section))} sections, "
                 f"{len(pd.unique(self.unit))} units" +
                 (f", conditions {[str(c) for c in sorted(pd.unique(self.condition))]}" if self.condition is not None else ""),
                 f"median counts per cell {np.median(tot):.0f}"]
        vc = pd.Series(self.labels).value_counts()
        lines.append("labels: " + ", ".join(f"{k} {v}" for k, v in vc.items()))
        return "\n".join(lines)


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def check_output_dir(out_dir, data_path=None):
    """Refuse to write outputs next to (or inside the directory of) the input data file."""
    if data_path is None:
        return
    data_dir = Path(data_path).resolve().parent
    if _is_inside(Path(out_dir), data_dir):
        raise ValueError(f"output directory {out_dir} is inside the data directory {data_dir}; "
                         "choose an output location outside the data directory")


def describe_file(path) -> str:
    """Obs columns (with a few values), layers, obsm keys and shape, reading the file in backed mode."""
    import anndata as ad
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found (is the drive mounted?)")
    A = ad.read_h5ad(path, backed="r")
    try:
        lines = [f"{path}: {A.n_obs} cells x {A.n_vars} genes",
                 f"layers: {[k for k in A.layers.keys() if k is not None]}", f"obsm: {list(A.obsm.keys())}", "obs columns:"]
        for c in A.obs.columns:
            v = A.obs[c]
            nu = v.nunique()
            ex = ", ".join(str(x)[:20] for x in pd.unique(v)[:5])
            lines.append(f"  {c:30s} {str(v.dtype):10s} {nu:>8d} distinct   e.g. {ex}")
        return "\n".join(lines)
    finally:
        A.file.close()


def load(path_or_adata, label_key: str, section_key: str, unit_key: str | None = None,
         condition_key: str | None = None, spatial=("x", "y"), counts_layer: str | None = None,
         keep_labels=None, min_cells_per_label: int = 0, subset: dict | None = None,
         allow_noninteger: bool = False) -> Tissue:
    """Build a Tissue from an AnnData object or an .h5ad path. The file is only read, never written.

    spatial: a pair of obs columns, or the name of an obsm entry. unit_key defaults to section_key.
    subset: {obs column: [values to keep]}, applied before the counts are loaded (a .h5ad path is opened in
    backed mode, so only the selected cells are read into memory).
    Counts must be raw integers (count splitting and thinning assume them) unless allow_noninteger is set.
    """
    import anndata as ad

    if isinstance(path_or_adata, (str, os.PathLike)):
        if not Path(path_or_adata).exists():
            raise FileNotFoundError(f"{path_or_adata} not found (is the drive mounted?)")
        A = ad.read_h5ad(path_or_adata, backed="r")      # read-only; nothing is written back
        src = str(path_or_adata)
    else:
        A, src = path_or_adata, None
    o = A.obs
    for col in [label_key, section_key, unit_key, condition_key] + list((subset or {}).keys()):
        if col is not None and col not in o.columns:
            raise ValueError(f"obs column '{col}' not found; available: {list(o.columns)}")
    keep = np.ones(A.n_obs, bool)
    for col, vals in (subset or {}).items():
        vals = [str(v) for v in (vals if isinstance(vals, (list, tuple)) else [vals])]
        miss = sorted(set(vals) - set(o[col].astype(str).unique()))
        if miss:
            raise ValueError(f"subset values not found in '{col}': {miss}")
        keep &= o[col].astype(str).isin(vals).to_numpy()
    lab_all = o[label_key].astype(str).to_numpy()
    if keep_labels is not None:
        keep &= np.isin(lab_all, list(keep_labels))
    if min_cells_per_label:
        vc = pd.Series(lab_all[keep]).value_counts()
        keep &= np.isin(lab_all, vc.index[vc >= min_cells_per_label])
    if keep.sum() == 0:
        raise ValueError("no cells left after subsetting")
    rows = np.flatnonzero(keep)
    genes = np.asarray(A.var_names)
    if counts_layer == "raw":                          # counts kept in AnnData's .raw (X normalized)
        if A.raw is None:
            raise ValueError("counts_layer 'raw' requested but the file has no .raw")
        X = A.raw.X[rows]
        genes = np.asarray(A.raw.var_names)
    elif counts_layer:
        if counts_layer not in A.layers.keys():
            raise ValueError(f"layer '{counts_layer}' not found; layers: {[k for k in A.layers.keys() if k]}"
                             + ("; the file has .raw: use counts_layer: raw" if A.raw is not None else ""))
        X = A.layers[counts_layer][rows]
    else:
        X = A.X[rows]
    X = sp.csr_matrix(X, dtype=np.float64)
    sample = X.data[: min(X.nnz, 200000)]
    if X.nnz and not np.allclose(sample, np.round(sample)):
        msg = ("counts are not integers (the matrix looks normalised or log-transformed); count splitting and "
               "thinning need raw counts. Point counts_layer at the raw counts"
               + (" (this file has .raw: counts_layer: raw)" if getattr(A, "raw", None) is not None else ""))
        if not allow_noninteger:
            raise ValueError(msg + ", or set allow_noninteger: true to round them (not recommended)")
        warnings.warn(msg + "; rounding")
        X.data = np.round(X.data)
        X.eliminate_zeros()
    if isinstance(spatial, str):
        xy = np.asarray(A.obsm[spatial])[rows, :2]
    else:
        xy = o[list(spatial)].to_numpy(dtype=float)[rows]
    ok = np.isfinite(xy).all(1)
    if not ok.all():
        warnings.warn(f"{(~ok).sum()} cells without coordinates dropped")
    T = Tissue(counts=X[ok], xy=xy[ok], labels=lab_all[rows][ok], section=o[section_key].to_numpy()[rows][ok],
               unit=o[unit_key or section_key].to_numpy()[rows][ok], genes=genes,
               condition=None if condition_key is None else o[condition_key].to_numpy()[rows][ok],
               obs_names=np.asarray(A.obs_names)[rows][ok], meta={"source": src})
    if src is not None and getattr(A, "file", None) is not None:
        A.file.close()
    return T


def take(t: Tissue, idx) -> Tissue:
    """A new Tissue with the cells idx (graph rebuilt)."""
    idx = np.asarray(idx)
    return Tissue(counts=t.counts[idx], xy=t.xy[idx], labels=t.labels[idx], section=t.section[idx],
                  unit=t.unit[idx], genes=t.genes, condition=None if t.condition is None else t.condition[idx],
                  obs_names=None if t.obs_names is None else t.obs_names[idx], meta=dict(t.meta))


def subset_genes(t: Tissue, gidx) -> Tissue:
    """The same cells with only the genes gidx (indices); graph shared (positions unchanged)."""
    gidx = np.asarray(gidx, int)
    return t.replace(counts=t.counts[:, gidx].tocsr(), genes=t.genes[gidx])


def counts_checksum(t: Tissue) -> str:
    """SHA-256 over cell identifiers, gene names and the count matrix (identifies an exported half A)."""
    import hashlib
    h = hashlib.sha256()
    ids = t.obs_names if t.obs_names is not None else np.arange(t.n)
    h.update("\n".join(map(str, ids)).encode()); h.update("\n".join(map(str, t.genes)).encode())
    X = t.counts.tocsr()
    for a in (X.indptr, X.indices, X.data):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def crop(t: Tissue, frac: float) -> Tissue:
    """Keep, in every section, the central square (Chebyshev distance to the section's median position) holding
    `frac` of its cells: fewer cells per unit at unchanged density (the cells-per-unit axis of the design sweep)."""
    if frac >= 1:
        return t
    keep = []
    for s in pd.unique(t.section):
        m = np.flatnonzero(t.section == s)
        c = np.median(t.xy[m], 0)
        d = np.abs(t.xy[m] - c).max(1)
        keep.append(m[d <= np.quantile(d, frac)])
    return take(t, np.sort(np.concatenate(keep)))


def _binom_sparse(X: sp.csr_matrix, p, rng) -> sp.csr_matrix:
    """Binomially thin a sparse count matrix; p scalar or per-cell vector."""
    X = X.tocsr()
    if np.isscalar(p):
        pp = p
    else:
        pp = np.repeat(np.asarray(p, float), np.diff(X.indptr))
    data = rng.binomial(X.data.astype(np.int64), pp).astype(np.float64)
    Y = sp.csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    Y.eliminate_zeros()
    return Y


def split(t: Tissue, frac: float, rng) -> tuple[Tissue, Tissue]:
    """Binomial count split: A ~ Bin(Y, frac), B = Y - A. Returns (A, B) sharing labels and graph."""
    A = _binom_sparse(t.counts, frac, rng)
    B = (t.counts - A).tocsr()
    B.eliminate_zeros()
    return t.replace(counts=A), t.replace(counts=B)


def thin(t: Tissue, p, rng) -> Tissue:
    """Binomial thinning to a fraction p (scalar or per cell)."""
    return t.replace(counts=_binom_sparse(t.counts, p, rng))


def lognorm(X: sp.csr_matrix, scale: float | None = None, genes=None, size_genes=None) -> np.ndarray:
    """log1p of counts scaled to a common total; dense (cells x selected genes).
    size_genes: genes whose counts define each cell's total (size factor); default all genes. The stress test
    computes totals from non-planted genes, so planted genes cannot shift other genes by normalisation."""
    tot = np.asarray((X if size_genes is None else X[:, size_genes]).sum(1)).ravel()
    tot[tot == 0] = 1
    scale = np.median(tot) if scale is None else scale
    Xs = X[:, genes] if genes is not None else X
    Z = sp.diags(scale / tot) @ Xs
    return np.log1p(Z.toarray())


def neighbour_counts(codes: np.ndarray, idx: np.ndarray, n_types: int) -> np.ndarray:
    """(n, T) counts of each label among each cell's neighbors."""
    n, k = idx.shape
    flat = np.repeat(np.arange(n), k) * n_types + codes[idx].ravel()
    return np.bincount(flat, minlength=n * n_types).reshape(n, n_types)
