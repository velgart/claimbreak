"""Experiment runner.

Order of operations (fixed, so that each layer acts on what the previous one produced):
  1. load the tissue; its labels are the reference ("truth") for this experiment
  2. plant known signals on the full counts (null / spike / shift)
  3. count perturbations of the arm, if any (contamination, thinning), on the full counts
  4. count split with a fixed seed: half A is used for re-calling labels, half B is analyzed
     (so labels and the analyzed counts never share sampling noise)
  5. labels of the arm: reference, perturbed reference (label_errors), or re-called on half A (recall)
  6. every question on (half B, labels); claims compared with the baseline (steps 1, 2, 4 with reference labels)
     and with planted truth

Outputs in cfg["out"] (never inside the data directory): config.yaml, planted.json, baseline_claims.csv,
native.json, summary.csv (one row per spec x level x rep x question; the run resumes from it),
decisions_<spec>.csv.gz (claims that are non-zero in the run or in the baseline), diagnostics_<question>.csv (joint
niche DE: collinearity of neighbor types at baseline), breakpoints_by_support.csv (per spec, question and support quartile: share broken ever and at or below the flip
rate), breakpoints.csv (break level in order of increasing severity; fragility = break level / resampling flip rate
for label-error arms; baseline support and effect of each claim).

Also written:
  claims_index_<question>.csv.gz   niche-DE questions: every baseline claim with its class , proxy target
                                   (stat_ref, se_ref, df_ref, p_ref), averaged target (stat_avg, sd_avg, n_avg; if
                                   analysis.target_seeds) and the half-A reference p-value (p_A; if
                                   analysis.half_a_reference)
  claims/<spec>/L<level>_r<rep>.npz  estimate, SE and df of every claim of every niche-DE question and corrector,
                                   aligned to the claims index (NaN = not estimable)
  target_avg.csv.gz                averaged target, reused when the run resumes
  manifest.json                    written last (integrity.write_manifest); void.json instead if a check failed
Panel arm: analysis.panel (gene list, file, or {random: n}) restricts labeling to the panel genes and the
analyzed gene set to the panel plus planted genes; `claimbreak export` writes half A for an external labeler and
a perturbation of kind `imported` reads its labels back.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import compare, perturb, plant
from . import questions as Q
from .integrity import (MARGIN_CAP, IntegrityError, check_level0, identity_exempt, level0_identity,
                        sha256_file, validate_claims, write_manifest, write_void)
from .labeling import NearestCentroid, native_confusion, native_flip, recall_labels
from .tissue import Tissue, check_output_dir, counts_checksum, load, split, subset_genes, thin

DEFAULTS = {"seed": 0, "reps": 3, "analysis": {"split": 0.5, "native_depth": 1.0},
            "questions": ["colocalization", "niches", "proximity_de"], "plant": [], "perturbations": []}


class ConfigError(ValueError):
    pass


def _check_placeholders(obj, where="config"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_placeholders(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_placeholders(v, f"{where}[{i}]")
    elif isinstance(obj, str) and "REPLACE" in obj:
        raise ConfigError(f"{where} still contains the placeholder REPLACE; run `claimbreak inspect FILE` "
                          "to see the available columns and edit the config")


def load_config(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _check_placeholders(cfg)
    out = {**DEFAULTS, **cfg}
    out["analysis"] = {**DEFAULTS["analysis"], **cfg.get("analysis", {})}
    return out


def _rng(*keys):
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in keys]))


def _apply_plant(t: Tissue, specs, seed):
    for i, sp_ in enumerate(specs):
        sp_ = dict(sp_)
        kind = sp_.pop("kind")
        rng = _rng(seed, 1000 + i)
        if kind is None or kind == "null":      # YAML reads an unquoted `null` as None
            t, _ = plant.null(t, rng, **sp_)
        elif kind == "spike":
            t, _ = plant.spike(t, rng=rng, **sp_)
        elif kind == "shift":
            t, _ = plant.shift(t, rng=rng, **sp_)
        elif kind == "spike_set":
            t, _ = plant.spike_set(t, rng=rng, **sp_)
        else:
            raise ValueError(f"unknown plant kind {kind}")
    return t


def _lineage(spec):
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, str) and spec.startswith("prefix:"):
        n = int(spec.split(":")[1])
        return lambda x: x[:n]
    if isinstance(spec, str) and spec.startswith("sep:"):
        sep = spec.split(":", 1)[1]
        return lambda x: x.split(sep)[0]
    raise ValueError(f"cannot read lineage spec {spec}")


def _keep_units(t, n: int):
    """Keep n units (design sweep), balanced across conditions, the first in sorted order within each condition."""
    from .tissue import take
    conds = pd.unique(t.condition) if t.condition is not None else np.array(["all"])
    per = max(n // len(conds), 1)
    keep = []
    for c in conds:
        m = (t.condition == c) if t.condition is not None else np.ones(t.n, bool)
        keep += sorted(pd.unique(t.unit[m]))[:per]
    return take(t, np.flatnonzero(np.isin(t.unit, keep)))


def _validate(cfg: dict):
    """Configuration rules checked before anything is loaded."""
    for p in cfg.get("perturbations", []):
        if p.get("kind") == "label_errors" and p.get("selection") == "margin":
            bad = [lv for lv in p["levels"] if float(lv) > MARGIN_CAP + 1e-12]
            if bad:
                raise ConfigError(f"arm {p.get('name', '?')}: margin levels {bad} exceed the cap "
                                  f"{MARGIN_CAP} (flip probabilities clip and destinations become deterministic)")
        if p.get("kind") == "imported":
            for k in ("file", "export"):
                if not p.get(k):
                    raise ConfigError(f"imported arm {p.get('name', '?')} needs `{k}` (labels file and the "
                                      "export.json it was produced from)")
    if (cfg.get("analysis", {}) or {}).get("target_seeds") and any(
            (p or {}).get("kind") == "shift" for p in cfg.get("plant", [])):
        raise ConfigError("target_seeds is implemented for null and spike plantings, not shift")


def _label_map(spec, labels=None) -> dict:
    """label_map: {fine: coarse}, a two-column CSV, or a rule on the names: "sep:<s>" (text before the first
    <s>, e.g. "sep: " maps "mAdamdec1+ Grem2-hi" to "mAdamdec1+") or "prefix:<n>" (first n characters)."""
    if isinstance(spec, dict):
        return {str(k): str(v) for k, v in spec.items()}
    if isinstance(spec, str) and (spec.startswith("sep:") or spec.startswith("prefix:")) and labels is not None:
        f = _lineage(spec)
        return {str(l): str(f(str(l))) for l in pd.unique(np.asarray(labels))}
    m = pd.read_csv(spec)
    if m.shape[1] < 2:
        raise ConfigError(f"label_map file {spec} needs two columns: fine label, coarse label")
    return dict(zip(m.iloc[:, 0].astype(str), m.iloc[:, 1].astype(str)))


def _load_tissue(cfg: dict) -> Tissue:
    d = dict(cfg["data"])
    crop_frac = d.pop("crop", None)
    n_units = d.pop("n_units", None)
    lmap = d.pop("label_map", None)
    t = load(d.pop("path"), **d)
    if lmap is not None:                     # coarser label level (granularity analysis)
        mp = _label_map(lmap, t.labels)
        miss = sorted(set(t.labels) - set(mp))
        if miss:
            raise ConfigError(f"label_map does not map the labels {miss}")
        t = t.replace(labels=np.array([mp[x] for x in t.labels]))
    if n_units:
        t = _keep_units(t, int(n_units))
    if crop_frac:
        from .tissue import crop
        t = crop(t, float(crop_frac))
    return t


def _panel(cfg: dict, t: Tissue):
    """Panel gene names from analysis.panel: a list, a file with one gene per line, or
    {random: n, seed: s} (a random gene subset, for rehearsing the panel arm on data without a 5K panel)."""
    spec = (cfg.get("analysis", {}) or {}).get("panel")
    if spec is None:
        return None
    if isinstance(spec, dict) and "random" in spec:
        rng = _rng(int(spec.get("seed", 0)), 9100)
        genes = list(t.genes[np.sort(rng.choice(len(t.genes), int(spec["random"]), replace=False))])
    elif isinstance(spec, (list, tuple)):
        genes = [str(g) for g in spec]
    else:
        genes = [l.strip() for l in Path(spec).read_text().splitlines() if l.strip() and not l.startswith("#")]
    found = [g for g in genes if g in set(t.genes)]
    if len(found) < 20:
        raise ConfigError(f"panel: only {len(found)} of {len(genes)} panel genes are in the data")
    return np.array(found)


def _setup(cfg: dict, tissue: Tissue | None, log=lambda *a: None):
    """Load, plant and split (shared by run, check and export so they see the same cells and counts)."""
    if tissue is None:
        tissue = _load_tissue(cfg)
    seed = int(cfg["seed"])
    panel = _panel(cfg, tissue)
    planted = _apply_plant(tissue, cfg["plant"], seed)
    if panel is not None:
        planted = planted.replace(meta={**planted.meta, "panel": [str(g) for g in panel]})
    frac = cfg["analysis"]["split"]
    base_A, base_B = split(planted, frac, _rng(seed, 7)) if frac else (planted, planted)
    pidx = None if panel is None else np.flatnonzero(np.isin(planted.genes, panel))
    lab_A = base_A if pidx is None else subset_genes(base_A, pidx)
    return tissue, planted, base_A, base_B, lab_A, pidx


def _folds(units, k: int, seed: int) -> dict:
    """Units grouped into k folds (deterministic in the seed), for training an external labeler on other folds."""
    u = np.array(sorted(pd.unique(units)))
    u = u[_rng(seed, 6000).permutation(len(u))]
    return {str(x): int(i % k) for i, x in enumerate(u)}


def export(cfg: dict | str, out: str, folds: int = 5, tissue: Tissue | None = None) -> dict:
    """Write half A of the planted tissue (panel genes only, if a panel is set) for an external labeler:
    half_a.h5ad (counts; obs: position, cell id, section, unit, reference label, fold) and
    export.json (seed, gene and cell counts, folds, SHA-256 of the exported counts). The labeler trains on the
    reference labels of the other folds and writes one label per row of half_a.h5ad (CSV, column `label`, same
    order); an `imported` perturbation reads that file back and checks the hash."""
    import anndata as ad
    if isinstance(cfg, (str, Path)):
        cfg = load_config(cfg)
    _validate(cfg)
    outp = Path(out)
    data_path = cfg.get("data", {}).get("path") if isinstance(cfg.get("data"), dict) else None
    check_output_dir(outp, data_path)
    outp.mkdir(parents=True, exist_ok=True)
    _, planted, base_A, _, lab_A, pidx = _setup(cfg, tissue)
    fmap = _folds(lab_A.unit, folds, int(cfg["seed"]))
    ids = lab_A.obs_names if lab_A.obs_names is not None else np.arange(lab_A.n).astype(str)
    obs = pd.DataFrame({"pos": np.arange(lab_A.n), "cell_id": np.asarray(ids).astype(str),
                        "section": lab_A.section, "unit": lab_A.unit, "ref_label": lab_A.labels,
                        "fold": [fmap[u] for u in lab_A.unit]}, index=[f"c{i}" for i in range(lab_A.n)])
    A = ad.AnnData(X=lab_A.counts.astype(np.float32), obs=obs,
                   var=pd.DataFrame(index=[str(g) for g in lab_A.genes]))
    A.write_h5ad(outp / "half_a.h5ad")
    from . import __version__
    man = {"claimbreak_version": __version__, "seed": int(cfg["seed"]), "n_cells": int(lab_A.n),
           "n_genes": int(lab_A.counts.shape[1]), "panel": pidx is not None, "folds": fmap, "n_folds": folds,
           "null_tissue": any((p or {}).get("kind") in (None, "null") for p in cfg["plant"]),
           "counts_sha256": counts_checksum(lab_A), "h5ad_sha256": sha256_file(outp / "half_a.h5ad")}
    (outp / "export.json").write_text(json.dumps(man, indent=1))
    return man


def _pair_covariates(base: Tissue, recs: list, native_C: pd.DataFrame, k: int = 10) -> None:
    """Co-location, confusion and abundance of every planted pair, added to the records."""
    idx, _ = base.graph.knn(k)
    for r in recs:
        if r.get("kind") != "spike":
            continue
        s_, r_ = r["sender"], r["receiver"]
        rc = np.flatnonzero(base.labels == r_)
        has = (base.labels[idx[rc]] == s_).any(1) if len(rc) else np.array([])
        f = np.exp(-plant.distance_to_type(base, rc, s_) / float(r["lam"])) if len(rc) else np.array([])
        conf = lambda a, b: float(native_C.at[a, b]) if (a in native_C.index and b in native_C.columns) else np.nan
        r["covariates"] = {"n_sender": int((base.labels == s_).sum()), "n_receiver": int(len(rc)),
                           "exposure_share_k10": float(has.mean()) if len(rc) else np.nan,
                           "mean_planted_weight": float(f.mean()) if len(rc) else np.nan,
                           "confusion_sender_to_receiver": conf(s_, r_),
                           "confusion_receiver_to_sender": conf(r_, s_)}


def _replant(tissue: Tissue, planted_recs: list, rng_for):
    """The same planting (pairs, genes, effects) on a new draw: a new null permutation and new spike counts."""
    t = tissue
    for i, r in enumerate(planted_recs):
        rng = rng_for(i)
        if r["kind"] == "null":
            t, _ = plant.null(t, rng, within=tuple(r.get("within", ("section", "labels"))))
        elif r["kind"] == "spike":
            t, _ = plant.spike(t, r["sender"], r["receiver"], list(r["genes"]), r["effect"], r["lam"], rng)
        else:
            raise ConfigError(f"averaged target: cannot replant kind {r['kind']}")
    return t


def averaged_target(tissue: Tissue, planted: Tissue, qs: list, T: int, seed: int, frac, log=lambda *a: None):
    """Averaged target: the reference-label analysis repeated over T draws that vary the null
    permutation, the spike counts and the count split, with the planting specification and geometry fixed.
    Returns {question: DataFrame(claim, stat_avg, sd_avg, n_avg)}."""
    recs = [r for r in planted.meta.get("planted", []) if r.get("kind") in ("null", "spike")]
    acc = {q.name: {} for q in qs}
    t0 = time.time()
    for j in range(T):
        tj = _replant(tissue, recs, lambda i: _rng(seed, 5000, j, i))
        _, Bj = split(tj, frac, _rng(seed, 5001, j)) if frac else (tj, tj)
        for q in qs:
            c = q.run(Bj, Bj.labels, rep=0)
            a = acc[q.name]
            for cl_, v in zip(c.claim.to_numpy(), c.stat.to_numpy(float)):
                if np.isfinite(v):
                    s0 = a.get(cl_)
                    a[cl_] = (s0[0] + v, s0[1] + v * v, s0[2] + 1) if s0 else (v, v * v, 1)
        if (j + 1) % 5 == 0 or j == T - 1:
            log(f"  averaged target: {j + 1}/{T} draws [{time.time() - t0:.0f} s]")
    out = {}
    for name, a in acc.items():
        cl_ = np.array(list(a.keys()), dtype=object)
        v = np.array(list(a.values()), float).reshape(-1, 3)
        n = v[:, 2]
        mu = v[:, 0] / n
        sd = np.sqrt(np.maximum(v[:, 1] / n - mu ** 2, 0) * n / np.maximum(n - 1, 1))
        out[name] = pd.DataFrame({"claim": cl_, "stat_avg": mu, "sd_avg": sd, "n_avg": n.astype(int)})
    return out


class ClaimStore:
    """Per-claim storage: estimate, SE and df of every claim of every niche-DE question and
    corrector, per (spec, level, rep), aligned to the question's claims index."""

    def __init__(self, out: Path, universes: dict):
        self.out = out
        self.pos = {name: pd.Index(u) for name, u in universes.items()}

    def path(self, spec, level, rep) -> Path:
        return self.out / "claims" / str(spec) / f"L{float(level):.6g}_r{int(rep)}.npz"

    def save(self, spec, level, rep, tables: dict):
        arrs = {}
        for name, (base_q, cl) in tables.items():
            u = self.pos[base_q]
            r = cl.drop_duplicates("claim").set_index("claim").reindex(u)
            arrs[f"{name}@est"] = r.stat.to_numpy(np.float32)
            arrs[f"{name}@se"] = r.se.to_numpy(np.float32) if "se" in r else np.full(len(u), np.nan, np.float32)
            arrs[f"{name}@df"] = r.df.to_numpy(np.float32) if "df" in r else np.full(len(u), np.nan, np.float32)
        p = self.path(spec, level, rep)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrs)
        tmp.replace(p)

    def load(self, spec, level, rep) -> dict:
        with np.load(self.path(spec, level, rep)) as z:
            return {k: z[k] for k in z.files}


def _overlap_with_recall(ctx, labels) -> dict:
    """Jaccard overlap between the cells this run mislabels and the cells the second labeler mislabels on half-A
    counts thinned to 0.9 (one fixed draw), with the overlap expected for independent sets of the same sizes.
    High overlap means margin injection and the re-called arm select the same cells (a mechanical match)."""
    if "recall_ref" not in ctx:
        ctx["recall_ref"] = recall_labels(ctx["lab_A"], 0.9, _rng(ctx["seed"], 4242), labeller=ctx["labeller"])
    truth = ctx["base_B"].labels
    F = np.asarray(labels, dtype=object) != truth
    R = np.asarray(ctx["recall_ref"], dtype=object) != truth
    n, nf, nr = len(F), F.sum(), R.sum()
    inter = (F & R).sum()
    union = nf + nr - inter
    e_inter = nf * nr / max(n, 1)
    return {"jaccard_recall": float(inter / max(union, 1)),
            "jaccard_expected": float(e_inter / max(nf + nr - e_inter, 1))}


class Arm:
    """One perturbation spec: builds (analysis tissue, labels, info) for a level and replicate."""

    def __init__(self, spec: dict, i: int, ctx: dict):
        self.spec = dict(spec)
        self.kind = self.spec["kind"]
        self.name = self.spec.get("name", f"{self.kind}_{i}")
        self.levels = list(self.spec["levels"])
        self.i, self.ctx = i, ctx
        if self.kind == "label_errors":
            conf = self.spec.get("confusion", "native")
            types = ctx["base_B"].types
            if conf == "native":
                self.C = ctx["native_C"]
            elif conf == "native_by_condition":
                if not ctx.get("native_by_cond"):
                    raise ConfigError("confusion: native_by_condition needs a condition_key")
                self.C = {c: v["confusion"] for c, v in ctx["native_by_cond"].items()}
            elif conf == "uniform":
                self.C = perturb.uniform_confusion(types)
            elif isinstance(conf, dict) and "lineage" in conf:
                self.C = perturb.lineage_confusion(types, _lineage(conf["lineage"]))
            else:
                raise ValueError(f"unknown confusion {conf}")
            base = ctx["base_B"]
            elig = np.isin(base.labels, self.spec["types"]) if self.spec.get("types") else None
            tr = self.spec.get("type_rates")
            cell_mult = None
            if tr == "native":
                tr = ctx["native_type_flip"]
            elif tr == "native_by_condition":
                if not ctx.get("native_by_cond"):
                    raise ConfigError("type_rates: native_by_condition needs a condition_key")
                nb = ctx["native_by_cond"]
                cell_mult = np.array([nb[c]["flip_rate_by_type"].get(l, nb[c]["flip_rate"])
                                      for c, l in zip(base.condition, base.labels)])
                tr = None
            elif tr is not None and not isinstance(tr, dict):
                raise ConfigError("type_rates must be 'native', 'native_by_condition' or a dict of label -> rate, "
                                  f"got {tr}")
            self.elig = elig
            self.selection = self.spec.get("selection", "random")
            if self.selection not in ("random", "margin"):
                raise ConfigError(f"selection must be 'random' or 'margin', got {self.selection}")
            if self.selection == "margin":
                self.P, self.q = ctx["margin"]()
            if self.spec.get("spatial") == "patch":
                if self.selection == "margin":
                    raise ConfigError("spatial: patch and selection: margin cannot be combined")
                self.w = None
                return
            self.w = perturb.spatial_weights(base, self.spec.get("spatial", "random"), eligible=elig,
                                             k=self.spec.get("k", 10), gamma=self.spec.get("gamma", 2.0),
                                             condition_mult=self.spec.get("condition_mult"), type_mult=tr,
                                             cell_mult=cell_mult)

    def build(self, level, rep):
        B, labels, info = self._build(level, rep)
        if self.kind in ("label_errors", "recall") and B is self.ctx["base_B"]:
            info.update(_overlap_with_recall(self.ctx, labels))
        return B, labels, info

    def _build(self, level, rep):
        c = self.ctx
        rng = _rng(c["seed"], 2000 + self.i, int(round(level * 1e6)), rep)
        info = {}
        if self.kind == "label_errors" and self.selection == "margin":
            labels, info = perturb.margin_errors(c["base_B"].labels, self.P, self.q, c["base_B"].types, level, rng,
                                                 eligible=self.elig)
            return c["base_B"], labels, info
        if self.kind == "label_errors" and self.spec.get("spatial") == "patch":
            labels, info = perturb.patch_errors(c["base_B"], level, self.C, rng, size=self.spec.get("patch_size", 200.0),
                                                within=self.spec.get("patch_within", 0.5), eligible=self.elig)
            return c["base_B"], labels, info
        if self.kind == "label_errors":
            labels, info = perturb.label_errors(c["base_B"], level, self.C, rng, weights=self.w,
                                                context_bias=self.spec.get("context_bias", 0.0),
                                                k=self.spec.get("k", 10), types=self.spec.get("types"))
            return c["base_B"], labels, info
        if self.kind == "recall":
            labels = recall_labels(c["lab_A"], level, rng, labeller=c["labeller"])
            info["realised_eps"] = float((labels != c["base_B"].labels).mean())
            return c["base_B"], labels, info
        if self.kind == "imported":
            return c["base_B"], *self._imported()
        if self.kind in ("contamination", "thinning"):
            t = c["planted"]
            if self.kind == "contamination":
                t, info = perturb.contaminate(t, level, rng, k=self.spec.get("k", 6), lam=self.spec.get("lam"),
                                              source_mult=self.spec.get("source_mult"),
                                              receiver_mult=self.spec.get("receiver_mult"))
            else:
                t = thin(t, level, rng)
            A, B = split(t, c["split_frac"], _rng(c["seed"], 7)) if c["split_frac"] else (t, t)
            if self.spec.get("labels", "reference") == "recall":
                # the run's labeler (centroids from clean half-A counts, panel genes if set) re-calls the
                # contaminated half A at the operating-point depth (0.9)
                if c.get("panel_idx") is not None:
                    A = subset_genes(A, c["panel_idx"])
                depth = float(self.spec.get("recall_depth", 0.9))
                labels = recall_labels(A, depth, rng, labeller=c["labeller"])
                info["realised_eps"] = float((labels != B.labels).mean())
                info["recall_depth"] = depth
            else:
                labels = B.labels
            return B, labels, info
        raise ValueError(f"unknown perturbation kind {self.kind}")

    def _imported(self):
        """Labels from an external labeler, checked against the export they were made from."""
        c = self.ctx
        if "imported" not in self.__dict__:
            man = json.loads(Path(self.spec["export"]).read_text())
            got = counts_checksum(c["lab_A"])
            if man.get("counts_sha256") != got:
                raise ConfigError(f"imported arm {self.name}: the run's half A does not match {self.spec['export']} "
                                  "(different seed, data, crop, panel or planting)")
            f = Path(self.spec["file"])
            L = pd.read_csv(f)
            if "label" not in L or len(L) != c["lab_A"].n:
                raise ConfigError(f"imported arm {self.name}: {f} needs a `label` column with one row per exported "
                                  f"cell ({c['lab_A'].n}); got {len(L)} rows")
            if "pos" in L and not np.array_equal(L["pos"].to_numpy(), np.arange(len(L))):
                raise ConfigError(f"imported arm {self.name}: rows of {f} are not in the exported order")
            labels = L["label"].astype(str).to_numpy()
            unknown = sorted(set(labels) - set(c["base_B"].types))
            if unknown:
                raise ConfigError(f"imported arm {self.name}: labels not in the reference set: {unknown[:10]}")
            sha = sha256_file(f)
            c.setdefault("labels_files", {})[str(f)] = sha
            self.imported = (labels, {"realised_eps": float((labels != c["base_B"].labels).mean()),
                                      "labels_sha256": sha})
        return self.imported[0], dict(self.imported[1])


def check(cfg: dict | str, tissue: Tissue | None = None) -> list[dict]:
    """Load the data and apply the planting only (no analysis): reports, for each spike, the candidate genes under
    each marker rule and the genes chosen, and the panel (genes found, planted genes inside it). Raises the same
    errors a run would raise at configuration or planting. Seconds to minutes."""
    if isinstance(cfg, (str, Path)):
        cfg = load_config(cfg)
    else:
        cfg = {**DEFAULTS, **cfg, "analysis": {**DEFAULTS["analysis"], **cfg.get("analysis", {})}}
    _validate(cfg)
    if tissue is None:
        tissue = _load_tissue(cfg)
    print(tissue.summary(), flush=True)
    for sp_ in cfg["plant"]:
        if sp_.get("kind") == "spike" and isinstance(sp_.get("genes"), int):
            n = plant.candidate_counts(tissue, sp_["sender"], sp_["receiver"],
                                       min_mean=sp_.get("min_mean", 0.2), max_ratio=sp_.get("max_ratio", 2.0))
            print(f"spike {sp_['sender']} -> {sp_['receiver']}: candidates " +
                  ", ".join(f"{k} {v}" for k, v in n.items()) +
                  f"; rule used: {sp_.get('marker_rule', 'all')}, genes wanted {sp_['genes']}", flush=True)
    panel = _panel(cfg, tissue)
    planted = _apply_plant(tissue, cfg["plant"], int(cfg["seed"]))
    recs = planted.meta.get("planted", [])
    for r in recs:
        if r["kind"] == "spike":
            print(f"planted {r['sender']} -> {r['receiver']} (effect {r['effect']}, lam {r['lam']}): "
                  + ", ".join(r["genes"]), flush=True)
    if panel is not None:
        pg = [g for r in recs if r["kind"] == "spike" for g in r["genes"]]
        print(f"panel: {len(panel)} genes in the data; planted genes inside the panel {sum(g in set(panel) for g in pg)}"
              f" of {len(pg)} (genes outside are analysed but never used for labels)", flush=True)
    print("check passed", flush=True)
    return recs


def run(cfg: dict | str, tissue: Tissue | None = None, out: str | None = None, verbose: bool = True) -> dict:
    if isinstance(cfg, (str, Path)):
        cfg = load_config(cfg)
    else:
        cfg = {**DEFAULTS, **cfg, "analysis": {**DEFAULTS["analysis"], **cfg.get("analysis", {})}}
    _validate(cfg)
    out = Path(out or cfg["out"])
    data_path = cfg.get("data", {}).get("path") if isinstance(cfg.get("data"), dict) else None
    check_output_dir(out, data_path)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "void.json").exists():
        raise ConfigError(f"{out} holds a void run (void.json); rerun into a new directory after the fix")
    (out / "manifest.json").unlink(missing_ok=True)       # outputs are about to change
    try:
        return _run(cfg, tissue, out, verbose)
    except IntegrityError as e:
        write_void(out, str(e))
        raise


def _run(cfg, tissue, out: Path, verbose: bool) -> dict:
    seed = int(cfg["seed"])
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t0 = time.time()
    integ = cfg.get("integrity", {}) or {}

    tissue, planted, base_A, base_B, lab_A, pidx = _setup(cfg, tissue)
    log(tissue.summary())
    if pidx is not None:
        log(f"panel: {len(pidx)} genes; labels on panel genes, analysis on panel + planted genes")
    for i, r in enumerate([r for r in planted.meta.get("planted", []) if r["kind"] == "spike"], 1):
        log(f"spike {i}: {r['sender']} -> {r['receiver']}, effect {r['effect']}, lam {r['lam']}, "
            f"{len(r['genes'])} genes ({r.get('marker_rule')})")
    frac = cfg["analysis"]["split"]

    from . import correct as CR
    corr = {}
    qspecs = []
    for qd in cfg["questions"]:
        qd = {"name": qd} if isinstance(qd, str) else dict(qd)
        cs = qd.pop("correctors", None) or []
        qspecs.append(qd)
        corr[qd.get("label") or qd["name"]] = [CR.make(c) for c in cs]
    qs = [Q.make(q) for q in qspecs]
    for q in qs:
        if corr.get(q.name) and not isinstance(q, Q.ProximityDE):
            raise ConfigError(f"correctors are implemented for proximity_de only, not {q.name}")
        if corr.get(q.name) and isinstance(q, Q.ProximityDEJoint) and \
                any(isinstance(c, CR.TwoDE) and c.k_exposure is None for c in corr[q.name]):
            raise ConfigError("twode on proximity_de_joint needs k_exposure (the joint model's k is its kernel "
                              "truncation, not an exposure neighbourhood)")
    names = [q.name for q in qs]
    if len(set(names)) != len(names):
        raise ConfigError(f"question names must be unique; give repeated questions a label: {names}")
    for q in qs:
        q.prepare(base_B.replace(meta=planted.meta))
    pqs = [q for q in qs if isinstance(q, Q.ProximityDE)]
    base = pd.concat([q.run(base_B, base_B.labels, rep=0).assign(question=q.name) for q in qs],
                     ignore_index=True)
    for q in pqs:
        validate_claims(base[base.question == q.name], f"baseline {q.name}")
    base.to_csv(out / "baseline_claims.csv", index=False)
    for q in qs:                   # collinearity of neighbor types in the joint model, on the baseline
        if isinstance(getattr(q, "diagnostics", None), pd.DataFrame) and not q.diagnostics.empty:
            q.diagnostics.to_csv(out / f"diagnostics_{q.name}.csv", index=False)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "tissue"}, f, sort_keys=False)
    log(f"baseline: " + ", ".join(f"{q.name} {int((base.question == q.name).sum())} claims, "
                                  f"{int(((base.question == q.name) & (base.decision != 0)).sum())} positive"
                                  for q in qs) + f" [{time.time() - t0:.0f} s]")

    # averaged target and half-A reference
    T = int(cfg["analysis"].get("target_seeds") or 0)
    tgt = {}
    if T and pqs:
        tf = out / "target_avg.csv.gz"
        if tf.exists():
            prev = pd.read_csv(tf)
            if int(prev.n_draws.iloc[0]) == T:
                tgt = {k: g.drop(columns=["question", "n_draws"]) for k, g in prev.groupby("question")}
        if not tgt:
            log(f"averaged target over {T} draws (reference labels only)")
            tgt = averaged_target(tissue, planted, pqs, T, seed, frac, log)
            pd.concat([d.assign(question=k, n_draws=T) for k, d in tgt.items()]).to_csv(tf, index=False)
    half_a = {}
    if cfg["analysis"].get("half_a_reference") and pqs:
        for q in pqs:
            half_a[q.name] = q.run(base_A, base_A.labels, rep=0)[["claim", "stat", "p"]].rename(
                columns={"stat": "stat_A", "p": "p_A"})
        log("half-A reference analysis done (claim selection for the real-tissue test)")
    universes = {}
    spikes_meta = planted.meta.get("planted", [])
    for q in pqs:
        b = base[base.question == q.name][["claim", "stat", "se", "df", "p"]].drop_duplicates("claim")
        idx = compare.claim_classes(b.claim.to_numpy(), spikes_meta).merge(
            b.rename(columns={"stat": "stat_ref", "se": "se_ref", "df": "df_ref", "p": "p_ref"}), on="claim")
        if q.name in tgt:
            idx = idx.merge(tgt[q.name], on="claim", how="left")
        if q.name in half_a:
            idx = idx.merge(half_a[q.name], on="claim", how="left")
        idx.to_csv(out / f"claims_index_{q.name}.csv.gz", index=False)
        universes[q.name] = idx.claim.to_numpy()
        if q.name in tgt:                                     # coverage of the averaged target in summary rows
            base = base.merge(tgt[q.name][["claim", "stat_avg"]].assign(question=q.name),
                              on=["question", "claim"], how="left", suffixes=("", "_new"))
            if "stat_avg_new" in base:
                base["stat_avg"] = base.stat_avg.fillna(base.stat_avg_new)
                base = base.drop(columns=["stat_avg_new"])
    store = ClaimStore(out, universes)

    labeller = NearestCentroid(lab_A)
    native_C, native_rate = native_confusion(lab_A, cfg["analysis"]["native_depth"], _rng(seed, 11),
                                             labeller=labeller, min_errors=1)
    flip, type_flip = native_flip(lab_A, _rng(seed, 12), labeller=labeller, by_type=True)
    by_cond = {}
    if lab_A.condition is not None:
        for j, c in enumerate(sorted(pd.unique(lab_A.condition))):
            m = lab_A.condition == c
            Cc, dc = native_confusion(lab_A, cfg["analysis"]["native_depth"], _rng(seed, 13, j),
                                      labeller=labeller, mask=m, types=lab_A.types, fallback=native_C)
            fc, tfc = native_flip(lab_A, _rng(seed, 14, j), labeller=labeller, by_type=True, mask=m)
            by_cond[c] = {"disagreement": dc, "flip_rate": fc, "flip_rate_by_type": tfc, "confusion": Cc}
            log(f"  {c}: disagreement {dc:.3f}, flip rate {fc:.3f}")
    (out / "native.json").write_text(json.dumps({
        "disagreement": native_rate, "flip_rate": flip, "flip_rate_by_type": type_flip,
        "by_condition": {c: {**{k: v for k, v in d.items() if k != "confusion"},
                             "confusion": d["confusion"].round(4).to_dict(orient="index")}
                         for c, d in by_cond.items()},
        "note": ("disagreement: re-called (half counts) vs reference labels, includes systematic labeller "
                 "difference; flip_rate: labels changing under Poisson resampling of half-A counts (noise only, "
                 "the fragility denominator; by type: per reference label, used by type_rates: native); by_condition: the "
                 "same within each condition (rows with < 20 errors in a condition take the pooled row), used by "
                 "confusion/type_rates: native_by_condition; confusion: rows true label -> columns called label, off-diagonal, "
                 "row-normalised; panel: labeller on panel genes only"),
        "panel_genes": None if pidx is None else int(len(pidx)),
        "confusion": native_C.round(4).to_dict(orient="index")}, indent=1))
    log(f"re-called labels (half counts): disagreement with reference {native_rate:.3f}, "
        f"resampling flip rate {flip:.3f}")
    _pair_covariates(base_B, spikes_meta, native_C)
    (out / "planted.json").write_text(json.dumps(spikes_meta, indent=1))

    null_tissue = any(p.get("kind") is None or p.get("kind") == "null" for p in cfg["plant"])
    n_units = len(pd.unique(base_B.unit))
    planted_ids = {f"{r['receiver']}|{r['sender']}|{g}" for r in spikes_meta
                   if r.get("kind") == "spike" for g in r["genes"]}
    truth = {q.name: compare.planted_truth(spikes_meta, type(q).name, base_B.types) for q in qs}
    ctx = {"seed": seed, "base_A": base_A, "lab_A": lab_A, "base_B": base_B, "planted": planted, "split_frac": frac,
           "labeller": labeller, "native_C": native_C, "native_type_flip": type_flip, "native_by_cond": by_cond,
           "native_flip": flip, "panel_idx": pidx}
    _mcache = {}

    def _margin():                                            # labeler posterior, computed once when needed
        if "P" not in _mcache:
            from .labeling import centroid_scores
            sc = centroid_scores(labeller, lab_A.counts, lab_A.section)
            _mcache["P"], _mcache["q"] = perturb.margin_probabilities(sc, base_B.labels, base_B.types, flip)
        return _mcache["P"], _mcache["q"]

    ctx["margin"] = _margin
    arms = [Arm(s, i, ctx) for i, s in enumerate(cfg["perturbations"])]
    cctx = {"types": base_B.types, "native_C": native_C, "native_type_flip": type_flip, "native_flip": flip,
            "margin": _margin,
            "native_by_cond": by_cond, "native_disagreement": native_rate}
    has_contrast = any(getattr(q, "contrast", None) or getattr(q, "conditions", None) for q in qs)
    for a in arms:
        if a.kind == "label_errors" and has_contrast and a.spec.get("confusion", "native") != "native_by_condition":
            log(f"note: arm {a.name} uses condition-blind errors with a condition contrast; "
                "native_by_condition reproduces the labeller's condition-dependent errors")

    reps = int(cfg["reps"])
    sfile = out / "summary.csv"
    done = pd.read_csv(sfile) if sfile.exists() else pd.DataFrame(columns=["spec", "level", "rep"])
    done_keys = set(zip(done.spec, done.level.astype(float).round(9), done.rep))
    for arm in arms:
        for level in arm.levels:
            for rep in range(reps):
                if (arm.name, round(float(level), 9), rep) in done_keys:
                    continue
                t_an, labels, info = arm.build(float(level), rep)
                rows, dec, stored = [], [], {}
                for q in qs:
                    cl = q.run(t_an, labels, rep=rep)
                    if isinstance(q, Q.ProximityDE):
                        validate_claims(cl, f"{arm.name} level {level} rep {rep} {q.name}")
                        stored[q.name] = (q.name, cl)
                    b = base[base.question == q.name]
                    r = {"spec": arm.name, "kind": arm.kind, "level": float(level), "rep": rep,
                         "question": q.name, **compare.versus_baseline(b, cl), **info}
                    r.update(compare.versus_truth(cl, truth[q.name], null_tissue))
                    if "unstable" in cl:
                        r["n_unstable"] = int(cl.unstable.sum())
                        r["n_unstable_base"] = int(b.unstable.sum()) if "unstable" in b else np.nan
                    if isinstance(q, Q.ProximityDE):
                        r.update(q.marker_share(cl))
                        r.update(compare.truth_detail(cl, spikes_meta))
                        r.update(compare.fixed_threshold(cl, b, spikes_meta, null_tissue,
                                                         getattr(q, "min_effect", 0.0)))
                        r.update(compare.threshold_free(cl, b, spikes_meta, null_tissue))
                        r.update(compare.coverage(cl, b, spikes_meta, n_units - 1))
                    rows.append(r)
                    keep = cl.claim.isin(b.claim[b.decision != 0]) | (cl.decision != 0) | cl.claim.isin(planted_ids)
                    cols = [c for c in ["claim", "decision", "stat", "effect", "support", "p"] if c in cl]
                    dec.append(cl[keep][cols].assign(
                        spec=arm.name, level=float(level), rep=rep, question=q.name))
                    for j, cr in enumerate(corr.get(q.name, [])):
                        tc = time.time()
                        sseed = int(_rng(seed, 3999, arm.i, int(round(float(level) * 1e6)), rep).integers(2**31))
                        cc = cr.apply(q, t_an, labels, cl, {**cctx, "sample_seed": sseed},
                                      _rng(seed, 3000 + j, arm.i, int(round(float(level) * 1e6)), rep))
                        name = f"{q.name}+{cr.label}"
                        validate_claims(cc, f"{arm.name} level {level} rep {rep} {name}")
                        stored[name] = (q.name, cc)
                        rc = {"spec": arm.name, "kind": arm.kind, "level": float(level), "rep": rep,
                              "question": name, "corrector": cr.label, **compare.versus_baseline(b, cc), **info}
                        if arm.kind == "label_errors" and float(level) == 0:     # level-0 identity
                            l0 = level0_identity(cl, cc, integ)
                            rc.update(l0)
                            why = identity_exempt(cr)
                            rc["level0_exempt"] = why or ""
                            if not why:
                                check_level0(l0, integ, f"{arm.name} rep {rep} {name}")
                        rc.update(compare.versus_truth(cc, truth[q.name], null_tissue))
                        rc.update(q.marker_share(cc))
                        rc.update(compare.truth_detail(cc, spikes_meta))
                        rc.update(compare.fixed_threshold(compare.on_universe(cc, cl.claim), b,
                                                          spikes_meta, null_tissue,
                                                          getattr(q, "min_effect", 0.0)))
                        ccu = compare.on_universe(cc, cl.claim)          # common claim set (naive's claims)
                        rc.update(compare.threshold_free(ccu, b, spikes_meta, null_tissue))
                        own = compare.threshold_free(cc, b, spikes_meta, null_tissue)
                        rc.update({f"{k}_own": v for k, v in own.items() if k in ("auprc", "null_typeI_05")})
                        rc["estimable"] = float(ccu.estimable.mean())
                        est_claims = set(cc.claim[cc.p.notna()])            # naive scored on the same claims
                        nv = compare.threshold_free(cl[cl.claim.isin(est_claims)], b, spikes_meta, null_tissue)
                        if "null_typeI_05" in nv:
                            rc["naive_typeI_on_est"] = nv["null_typeI_05"]
                        rc.update(compare.coverage(cc, b, spikes_meta, n_units - 1))
                        for extra in ("simex_clipped", "ppi_revealed", "ppi_interior", "ppi_units_kept", "ppi_lambda",
                                      "simex_curve"):
                            if extra in cc:
                                v = cc[extra].iloc[0] if len(cc) else np.nan
                                rc[extra] = v if isinstance(v, str) else float(v)
                        if "filtered" in cc:
                            rc["n_filtered"] = int(cc.filtered.sum())
                        rc["corrector_seconds"] = round(time.time() - tc, 1)
                        rows.append(rc)
                        kc = cc.claim.isin(b.claim[b.decision != 0]) | (cc.decision != 0) | cc.claim.isin(planted_ids)
                        dec.append(cc[kc][[c for c in cols if c in cc]].assign(
                            spec=arm.name, level=float(level), rep=rep, question=name))
                if stored:
                    store.save(arm.name, level, rep, stored)
                new = pd.DataFrame(rows)       # rewrite: columns differ between kinds of perturbation
                (pd.concat([pd.read_csv(sfile), new], ignore_index=True) if sfile.exists() else new).to_csv(
                    sfile, index=False)
                pd.concat(dec).to_csv(out / f"decisions_{arm.name}.csv.gz", mode="a", index=False,
                                      header=not (out / f"decisions_{arm.name}.csv.gz").exists(),
                                      compression="gzip")
                log(f"{arm.name} level={level} rep={rep}: " + "; ".join(
                    f"{r['question']} survival {r['survival']:.2f} spurious {r['spurious']}" for r in rows)
                    + f" [{time.time() - t0:.0f} s]")
                if rep == reps - 1:
                    _check_degenerate(store, arm, level, reps, pqs)

    summary = pd.read_csv(sfile)
    decs = [pd.read_csv(f) for f in sorted(out.glob("decisions_*.csv.gz"))]
    if decs:
        bp = []
        for arm in arms:
            d = pd.concat(decs)
            d = d[d.spec == arm.name]
            if d.empty:
                continue
            r = flip if arm.kind == "label_errors" else None
            bp.append(compare.break_points(d, base, native_rate=r,
                                           descending=arm.kind in ("recall", "thinning")))
        if bp:
            BP = pd.concat(bp, ignore_index=True)
            BP.to_csv(out / "breakpoints.csv", index=False)
            compare.break_summary(BP, flip).to_csv(out / "breakpoints_by_support.csv", index=False)
    write_manifest(out, cfg, summary, {"labels_files": ctx.get("labels_files", {}),
                                             "seed": seed, "target_seeds": T,
                                             "panel_genes": None if pidx is None else int(len(pidx))})
    log(f"done [{time.time() - t0:.0f} s]; outputs in {out}; manifest written")
    return {"summary": summary, "baseline": base, "native_rate": native_rate, "flip_rate": flip}


STOCHASTIC = {"label_errors", "recall", "contamination", "thinning"}


def _check_degenerate(store: ClaimStore, arm, level, reps: int, pqs):
    """At a non-zero level of a stochastic arm, replicates must differ (identical estimates across all
    replicates mean a degenerate, near-deterministic error draw)."""
    if reps < 2 or float(level) == 0 or arm.kind not in STOCHASTIC or not pqs:
        return
    if arm.kind in ("recall", "thinning") and float(level) >= 1:      # no thinning: deterministic
        return
    key = f"{pqs[0].name}@est"
    arrs = [store.load(arm.name, level, r)[key] for r in range(reps)]
    if all(np.array_equal(arrs[0], a, equal_nan=True) for a in arrs[1:]):
        raise IntegrityError(f"{arm.name} level {level}: all {reps} replicates give identical estimates "
                             "(degenerate error draw)")
