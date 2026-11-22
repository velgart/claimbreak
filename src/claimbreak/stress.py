"""Stress-test the claims of one analysis against realistic and random label errors.

For a tissue with labels (taken as the reference), a question (default: colocalization, homotypic pairs included)
and error rates eps, every replicate draws one set of error counts per pair from a confusion matrix and places the
same counts twice: at random (the standard stress test) and where whole-cell relabeling puts them (flipmodel).
For every claim the output gives the reference effect, the mean change of the effect under each placement, how
often the decision survives, and whether the two placements move the effect in opposite directions.

The point is the comparison: a claim that random relabeling calls robust, or merely weakened, can be exaggerated
or created by errors placed where segmentation and neighbors' transcripts put them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import questions
from .flipmodel import CELL_FEATURES, NB_FEATURES, Design, FlipModel, Neighbourhood, cell_features, draw_counts, \
    margin_second, place
from .labeling import NearestCentroid, centroid_scores, native_confusion
from .perturb import uniform_confusion
from .tissue import Tissue


def stress_test(t: Tissue, question="colocalization", eps=(0.05, 0.10), reps: int = 5, model=None,
                confusion="native", nucleus_counts=None, cell_area=None, nucleus_area=None, scores=None,
                seed: int = 0, k: int = 20, r_contact: float | None = None, r_far: float | None = None,
                log=print) -> tuple[pd.DataFrame, dict]:
    """Return (claims table, info). question: a name or a dict for questions.make; colocalization defaults to
    include_self=True and n_perm=20. model: FlipModel, a JSON path, or None for the packaged default.
    confusion: "native" (labels re-called on counts thinned to half), "uniform", or a DataFrame.
    scores: labeler scores (n x T, columns t.types); default: the leave-one-section-out nearest-centroid
    correlations."""
    rng = np.random.default_rng(seed)
    spec = {"name": question} if isinstance(question, str) else dict(question)
    if spec["name"] == "colocalization":
        spec.setdefault("include_self", True)
        spec.setdefault("n_perm", 20)
    q = questions.make(spec)
    q.prepare(t)
    types = np.asarray(t.types)
    ref_codes = t.codes(None, types)[0]

    default_model = model is None
    if model is None:
        model = FlipModel.default()
    elif not isinstance(model, FlipModel):
        model = FlipModel.from_json(model)

    lab = None
    if scores is None or isinstance(confusion, str) and confusion == "native":
        log("labeller: leave-one-section-out nearest centroid")
        lab = NearestCentroid(t)
    if scores is None:
        scores = centroid_scores(lab, t.counts, t.section)
    margin, second = margin_second(scores, ref_codes)
    cell = cell_features(t, margin=margin, nucleus_counts=nucleus_counts, cell_area=cell_area,
                         nucleus_area=nucleus_area)
    nb = Neighbourhood.build(t, types=types, k=k, r_contact=r_contact, r_far=r_far)
    if default_model and "outside_nucleus_share" not in cell and FlipModel.has_variant("no_nucleus"):
        model = FlipModel.default("no_nucleus")
        log("no nucleus counts: using the packaged flip model fitted without nucleus-based features")
    wanted = [f for f in NB_FEATURES + CELL_FEATURES if f in model.beta]
    design = Design(nb, cell, second, wanted)
    missing = [f for f in wanted if f not in design.features]
    if missing:
        log(f"flip model: features not available here, dropped: {missing}")

    if isinstance(confusion, pd.DataFrame):
        C = confusion.reindex(index=types, columns=types).fillna(0)
    elif confusion == "uniform":
        C = uniform_confusion(types)
    elif confusion == "native":
        C, rate = native_confusion(t, 0.5, rng, labeller=lab)
        log(f"native confusion at half depth (disagreement {rate:.3f})")
    else:
        raise ValueError(f"unknown confusion {confusion}")

    ref = q.run(t, t.labels, 0).set_index("claim")
    rows, infos = [], []
    for e in eps:
        for r in range(reps):
            n = draw_counts(t.labels, e, C, rng)
            for arm, mdl in (("random", None), ("flipmodel", model)):
                L, info = place(t.labels, n, types, rng, design=design, model=mdl)
                cl = q.run(t, L, r + 1).set_index("claim")
                d = pd.DataFrame({"effect": cl["effect"], "decision": cl["decision"]}).reindex(ref.index)
                d["eps"], d["rep"], d["arm"] = e, r, arm
                rows.append(d.reset_index())
                infos.append(dict(eps=e, rep=r, arm=arm, **info))
            log(f"eps {e}: replicate {r + 1}/{reps}")
    R = pd.concat(rows, ignore_index=True)
    out = summarise(ref, R)
    inf = pd.DataFrame(infos)
    info = {"question": spec, "eps": list(eps), "reps": reps, "features_used": design.features,
            "features_missing": missing, "model_note": model.note,
            "realised_eps": inf.groupby(["eps", "arm"]).realised_eps.mean().round(4).reset_index().to_dict("records"),
            "contact_share": (inf.groupby(["eps", "arm"])["contact_share"].mean().round(3).reset_index()
                              .to_dict("records") if "contact_share" in inf else None)}
    return out, info


def summarise(ref: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """Per claim and eps: reference effect and decision, mean change of the effect and decision survival under
    each placement, and a reading."""
    R = R.merge(ref[["effect", "decision"]].rename(columns={"effect": "ref_effect", "decision": "ref_decision"}),
                left_on="claim", right_index=True)
    R["delta"] = R["effect"] - R["ref_effect"]
    R["kept"] = (R["decision"] == R["ref_decision"]).astype(float)
    g = R.groupby(["claim", "eps", "arm"]).agg(delta=("delta", "mean"), delta_sd=("delta", "std"),
                                                kept=("kept", "mean")).unstack("arm")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index().merge(ref[["effect", "decision"]].rename(columns={"effect": "ref_effect",
                                                                          "decision": "ref_decision"}),
                              left_on="claim", right_index=True)

    def reading(r):
        dr, df_ = r.get("delta_random", np.nan), r.get("delta_flipmodel", np.nan)
        kr, kf = r.get("kept_random", np.nan), r.get("kept_flipmodel", np.nan)
        if not (np.isfinite(dr) and np.isfinite(df_)):
            return "not estimable"
        notes = []
        if np.sign(dr) != np.sign(df_) and max(abs(dr), abs(df_)) > 0.02:
            toward = np.sign(df_) == np.sign(r["ref_effect"])
            notes.append("opposite directions; realistic errors " + ("exaggerate" if toward else "weaken") +
                         " the effect")
        if kr >= 0.9 > kf:
            notes.append("robust to random errors, not to realistic ones")
        elif kf >= 0.9 > kr:
            notes.append("fragile to random errors only")
        elif kr < 0.9 and kf < 0.9:
            notes.append("fragile to both")
        return "; ".join(notes) or "agree"
    g["reading"] = g.apply(reading, axis=1)
    return g


def load_extras(path, obs_names, nucleus_layer=None, cell_area_key=None, nucleus_area_key=None) -> dict:
    """Per-cell nucleus counts (row sums of a layer) and areas (obs columns), aligned to obs_names."""
    import anndata as ad
    A = ad.read_h5ad(path, backed="r")
    pos = pd.Index(A.obs_names).get_indexer(obs_names)
    if (pos < 0).any():
        raise ValueError("cells of the tissue not found in the file")
    out = {}
    if nucleus_layer:
        if nucleus_layer not in A.layers.keys():
            raise ValueError(f"layer '{nucleus_layer}' not found; layers: {list(A.layers.keys())}")
        L = A.layers[nucleus_layer]
        tot = np.asarray(L.sum(1)).ravel()
        out["nucleus_counts"] = tot[pos]
    for key, name in ((cell_area_key, "cell_area"), (nucleus_area_key, "nucleus_area")):
        if key:
            if key not in A.obs.columns:
                raise ValueError(f"obs column '{key}' not found")
            out[name] = A.obs[key].to_numpy(float)[pos]
    return out


def write(out_dir, table: pd.DataFrame, info: dict):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    table.to_csv(out / "claims.csv", index=False)
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str))
    flagged = table[table.reading != "agree"]
    lines = [f"# claimbreak stress test: {info['question'].get('name')}", "",
             f"Error rates: {info['eps']}; {info['reps']} replicates; same error counts per pair in both placements.",
             f"Flip-model features used: {', '.join(info['features_used'])}" +
             (f"; missing here: {', '.join(info['features_missing'])}" if info["features_missing"] else ""), "",
             f"{len(flagged)} of {len(table)} claim x rate rows where random and realistic placement differ:", ""]
    cols = ["claim", "eps", "ref_effect", "delta_random", "delta_flipmodel", "kept_random", "kept_flipmodel",
            "reading"]
    if len(flagged):
        lines += ["```", flagged.sort_values(["eps", "claim"])[cols].round(3).to_string(index=False), "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
