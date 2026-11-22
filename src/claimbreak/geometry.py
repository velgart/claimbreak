"""Physical neighborhoods of cells from nucleus centroids and nucleus sizes.

The default contact elsewhere is the six nearest centroids within 30 um: graph proximity with a distance cap. Here neighbors are defined by physical distance between nuclei:

  gap_ij = |x_i - x_j| - r_i - r_j   (floored at 0)

with r the radius of a disk of the cell's nucleus area (obs `nucleus_area`, um^2). Where the area is missing, the
section median is used (and a default of 3 um if no cell has one). Nucleus polygons would be more exact; disks keep
the computation independent of the deposited polygon files, which differ between platforms.

Two further quantities per cell, at one contact gap g:

  border share   the circumference of the cell's nucleus is divided into angular bins; each bin is given to the
                 neighbor (gap <= g) whose nucleus disk, seen from the cell's center, covers that angle, the
                 smallest gap winning. The share of a neighbor is its number of bins over all bins; the
                 heterotypic border share is the share held by neighbors of another type over the covered share.
                 It approximates the share of the cell's boundary in the nearest-nucleus tessellation that touches
                 another type.
  exposure       e_i = sum over heterotypic neighbors j of share_ij * m_j / m_i, with m the cell's RNA content
                 (whole-cell counts): the RNA of touching cells of another type per unit of the cell's own RNA,
                 weighted by how much of the cell's border each neighbor holds (theory Box: spillover scales with the
                 neighbor's RNA, not the receiver's).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_RADIUS = 3.0     # um; used only when no cell of a section has a nucleus area
MAX_RADIUS = 8.0         # um; larger nucleus areas (segmentation merges) are clipped for the search radius


def nucleus_radius(area) -> tuple[np.ndarray, str]:
    """Radius (um) of a disk with each cell's nucleus area; missing or non-positive areas take the median radius.
    Returns (radius, note)."""
    a = np.asarray(area, float) if area is not None else np.array([], float)
    ok = np.isfinite(a) & (a > 0)
    if a.size and ok.any():
        r = np.sqrt(np.where(ok, a, np.nan) / np.pi)
        med = float(np.nanmedian(r))
        note = "" if ok.all() else f"{int((~ok).sum())} cells without nucleus area: median radius {med:.2f} um used"
        return np.clip(np.where(ok, r, med), 0.5, MAX_RADIUS), note
    n = a.size
    return np.full(n, DEFAULT_RADIUS), f"no nucleus areas: default radius {DEFAULT_RADIUS} um for every cell"


def gap_pairs(xy, r, gmax: float):
    """All ordered pairs (i, j), i != j, with nucleus gap <= gmax. Returns i, j, gap, distance (arrays)."""
    xy = np.asarray(xy, float)
    r = np.asarray(r, float)
    tree = cKDTree(xy)
    P = tree.query_pairs(gmax + 2 * float(r.max() if r.size else 0), output_type="ndarray")
    if len(P) == 0:
        e = np.array([], int)
        return e, e, np.array([], float), np.array([], float)
    i, j = P[:, 0], P[:, 1]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    gap = np.maximum(d - r[i] - r[j], 0.0)
    keep = gap <= gmax
    i, j, d, gap = i[keep], j[keep], d[keep], gap[keep]
    return np.r_[i, j], np.r_[j, i], np.r_[gap, gap], np.r_[d, d]


def counts_within(n: int, i, gap, g: float) -> np.ndarray:
    """Number of neighbors with gap <= g per cell."""
    return np.bincount(i[gap <= g], minlength=n)


def border_shares(n: int, xy, r, i, j, gap, d, g: float, bins: int = 72, chunk: int = 2_000_000):
    """Angular border ownership at contact gap g. Returns (owner pairs as arrays i_p, j_p, share_p) and the covered
    share per cell. share_p = bins held by neighbor j / bins (of the full circle)."""
    xy = np.asarray(xy, float)
    m = gap <= g
    i, j, gap, d = i[m], j[m], gap[m], d[m]
    if len(i) == 0:
        return np.array([], int), np.array([], int), np.array([], float), np.zeros(n)
    theta = np.arctan2(xy[j, 1] - xy[i, 1], xy[j, 0] - xy[i, 0])
    half = np.arcsin(np.clip(r[j] / np.maximum(d, 1e-9), 0.0, 1.0))
    w = 2 * np.pi / bins
    b0 = np.floor((theta - half) / w).astype(int)
    b1 = np.floor((theta + half) / w).astype(int)
    nb = np.maximum(b1 - b0 + 1, 1)
    # expand pairs to the bins they cover, in chunks of expanded rows
    keys, gaps, pid = [], [], []
    starts = np.r_[0, np.cumsum(nb)]
    total = int(starts[-1])
    for lo in range(0, total, chunk):
        hi = min(total, lo + chunk)
        rows = np.arange(lo, hi)
        p = np.searchsorted(starts, rows, side="right") - 1
        off = rows - starts[p]
        b = (b0[p] + off) % bins
        keys.append(i[p].astype(np.int64) * bins + b)
        gaps.append(gap[p])
        pid.append(p)
    keys, gaps, pid = np.concatenate(keys), np.concatenate(gaps), np.concatenate(pid)
    o = np.lexsort((pid, gaps, keys))          # per (cell, bin): smallest gap first, then lowest pair index
    keys, pid = keys[o], pid[o]
    first = np.r_[True, keys[1:] != keys[:-1]]
    win = pid[first]                           # the owning pair of each covered (cell, bin)
    held = np.bincount(win, minlength=len(i))  # bins held by each pair
    covered = np.bincount(i[win], minlength=n) / bins
    has = held > 0
    return i[has], j[has], held[has] / bins, covered


def heterotypic(n: int, i_p, j_p, share_p, focal, nbr, mass=None):
    """Per cell: heterotypic border share (share of the covered border, among neighbors of known type, held by
    neighbors whose type differs from the cell's; NaN without such a border or without a focal type), and the
    exposure sum_j het share_ij * m_j / m_i (0 without heterotypic neighbors; NaN without a focal type).
    focal: the cell's own type code (e.g. from one nuclear half); nbr: each cell's type code as a neighbor (e.g. from
    its full nuclear counts); -1 = unknown. Neighbors of unknown type are left out of both shares."""
    focal, nbr = np.asarray(focal), np.asarray(nbr)
    known = (focal[i_p] >= 0) & (nbr[j_p] >= 0)
    het = known & (focal[i_p] != nbr[j_p])
    het_share = np.bincount(i_p[het], weights=share_p[het], minlength=n)
    known_share = np.bincount(i_p[known], weights=share_p[known], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where((known_share > 0) & (focal >= 0), het_share / known_share, np.nan)
    if mass is None:
        return frac, np.full(n, np.nan)
    m = np.asarray(mass, float)
    wm = share_p[het] * m[j_p[het]] / np.maximum(m[i_p[het]], 1.0)
    expo = np.bincount(i_p[het], weights=wm, minlength=n)
    expo = np.where(focal >= 0, expo, np.nan)
    return frac, expo


def type_share(n: int, i_p, j_p, share_p, nbr, code: int):
    """Per cell: share of the covered border held by neighbors of type `code` (over all covered border; 0 without
    a covered border)."""
    nbr = np.asarray(nbr)
    hit = nbr[j_p] == code
    num = np.bincount(i_p[hit], weights=share_p[hit], minlength=n)
    den = np.bincount(i_p, weights=share_p, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)
