"""Run integrity checks of the stress-test runner.

A run that fails a check is void: the runner writes void.json with the reason, raises IntegrityError and never
writes a manifest. Summaries and uploads are made only from runs with a complete manifest.json, written last.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

MARGIN_CAP = 0.15
DEFAULTS = {"level0_absz": 0.5, "level0_ratio": [0.95, 1.05], "level0_min_strong": 10}   # level-0 tolerances of the stress-test reports


class IntegrityError(RuntimeError):
    """A run integrity check failed; the run is void."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_claims(cl: pd.DataFrame, where: str):
    """Explicit numerical checks on a niche-DE claims table: every claim with a p-value has a finite estimate and
    a finite, non-negative standard error; no estimate without a p-value. (Empty-slice warnings are silenced
    deliberately, so warnings cannot be the check.)"""
    if "se" not in cl:
        raise IntegrityError(f"{where}: claims have no standard errors (per-claim storage needs them)")
    p, st, se = cl.p.to_numpy(float), cl.stat.to_numpy(float), cl.se.to_numpy(float)
    hasp = np.isfinite(p)
    bad = hasp & ~(np.isfinite(st) & np.isfinite(se) & (se >= 0))
    if bad.any():
        raise IntegrityError(f"{where}: {int(bad.sum())} claims have a p-value but no finite estimate or SE, "
                             f"e.g. {cl.claim.to_numpy()[bad][:3].tolist()}")
    orphan = ~hasp & np.isfinite(st)
    if orphan.any():
        raise IntegrityError(f"{where}: {int(orphan.sum())} claims have an estimate but no p-value "
                             f"(a value was substituted for a missing one?), e.g. {cl.claim.to_numpy()[orphan][:3].tolist()}")


def identity_exempt(cr) -> str | None:
    """Correctors that cannot reproduce the naive estimate at zero label error by design (reason), else None."""
    name = type(cr).__name__
    if name == "Simex" and getattr(cr, "pi", None) == "native":
        return "native SIMEX applies a non-zero error rate at level 0"
    if name == "PPI":
        if getattr(cr, "validation_error", 0) or 0:
            return "validator errors"
        if getattr(cr, "mode", "ppi") == "classical":
            return "validated cells only"
        if getattr(cr, "predictor", "hard") == "soft":
            return "soft predictor differs from the observed labels"
    return None


def level0_identity(naive: pd.DataFrame, corr: pd.DataFrame, settings: dict) -> dict:
    """Agreement of a corrector with the naive model at level 0 (labels = reference): median |difference| in
    units of the naive SE, and median ratio of estimates on claims with |t| > 3 (catches shrinkage). The tuned-lambda blend makes PPI close to, not equal to, naive."""
    m = naive[["claim", "stat", "se"]].merge(corr[["claim", "stat"]], on="claim", suffixes=("_n", "_c"))
    m = m[np.isfinite(m.stat_n) & np.isfinite(m.stat_c) & (m.se > 0)]
    if m.empty:
        return {"level0_absz_median": np.nan, "level0_ratio_median": np.nan, "level0_n_strong": 0}
    z = (m.stat_c - m.stat_n) / m.se
    strong = (m.stat_n / m.se).abs() > 3
    ratio = (m.stat_c[strong] / m.stat_n[strong]).median() if strong.sum() else np.nan
    return {"level0_absz_median": float(z.abs().median()), "level0_ratio_median": float(ratio),
            "level0_n_strong": int(strong.sum())}


def check_level0(stats: dict, settings: dict, where: str):
    s = {**DEFAULTS, **(settings or {})}
    if stats["level0_absz_median"] > s["level0_absz"]:
        raise IntegrityError(f"{where}: at level 0 the corrector differs from naive by a median "
                             f"{stats['level0_absz_median']:.2f} SE (tolerance {s['level0_absz']})")
    lo, hi = s["level0_ratio"]
    r = stats["level0_ratio_median"]
    if stats["level0_n_strong"] >= s["level0_min_strong"] and not (lo <= r <= hi):
        raise IntegrityError(f"{where}: at level 0 the corrector's estimates are {r:.3f} x naive on strong claims "
                             f"(tolerance {lo}-{hi}): shrinkage or inflation")


def write_void(out: Path, reason: str):
    (out / "void.json").write_text(json.dumps({"void": True, "reason": reason,
                                               "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def write_manifest(out: Path, cfg: dict, summary: pd.DataFrame, extra: dict):
    """Written last. Lists version and config hash, the levels and replicates finished per arm, and
    the checksum of every output file."""
    from . import __version__
    files = {}
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.name not in ("manifest.json",):
            files[str(f.relative_to(out))] = sha256_file(f)
    fin = {}
    if not summary.empty:
        for (sp_, lv), g in summary.groupby(["spec", "level"]):
            fin.setdefault(str(sp_), {})[f"{lv:g}"] = int(g.rep.nunique())
    man = {"claimbreak_version": __version__,
           "config_sha256": sha256_file(out / "config.yaml") if (out / "config.yaml").exists() else None,
           "finished": fin, "outputs": files, "python": platform.python_version(),
           "numpy": np.__version__, "pandas": pd.__version__,
           "completed": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=1))
    tmp.replace(out / "manifest.json")
    return man
