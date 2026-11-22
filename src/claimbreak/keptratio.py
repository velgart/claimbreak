"""`claimbreak kept-ratio`: one command from Xenium output folders (or compartment .h5ad files) and a
single-cell reference to the label-loss check of the accompanying paper.

Steps, each written to its own subfolder of OUT so that every intermediate can be inspected or rerun:
  h5ad/     one compartment .h5ad per section (claimbreak.ingest; skipped with --h5ad)
  labels/   cells labeled from nuclear half A, half B, whole-cell counts minus A thinned to the depth of B (wholeB),
            and from the whole nucleus and whole cell for the label audit (`compartment label`)
  audit/    over-detected genes and marker support (`compartment audit`)
  compare/  label changes and where they land (`compartment compare`, comparisons nucB|nucA and wholeB|nucA)
  summary/  extra-change ratio and kept ratio per unit with block-bootstrap intervals (`compartment summary`)
and three files at the top: report.md (what was found, in words), kept_ratio.csv and sections.csv.

What the numbers mean
  halves agreement   share of cells whose label from nuclear half A equals that from half B. Labels unstable at
                     nuclear depth (below about 0.6) make the rest unreliable (liver).
  kept ratio         of the cells labeled type t from half A, the share that wholeB also calls t, over the share
                     that half B calls t. One means whole-cell counts lose no more labels than counting noise does;
                     0.7 means they lose 30% of t's labels relative to the nucleus. The nucleus carries some
                     neighboring RNA too, so the loss is a lower bound. It counts labels lost, not labels gained.
  extra-change ratio of the label changes beyond the null, the share that take a touching neighbor's type, over
                     the share expected if changes ignored the neighbors. Above one: changes follow the neighbors.
Two segmentations can be ranked by their kept ratios only if they share the nuclear counts (nuclear expansions of the
same section do); a method that also changes the nucleus needs protein or another external reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVEL = "L1"
ARMS = ("nucleus", "whole", "nucA", "nucB", "wholeB")
TEST, NULL = "wholeB|nucA", "nucB|nucA"
GATE = 0.6
TYPE_GATE = 0.5          # a type's own null keep rate below this: unstable at nuclear depth
MIN_KEPT_CELLS = 100
MIN_LOSS = 0.05         # losses smaller than this are not listed in the summary sentence


def run(out, ref, ref_level, xenium=(), h5ad=(), xenium_prefix: str = "", ref_merge: str = "",
        ref_donor: str = "donor_id", exclude_genes: str = "", unit_pattern: str = r"(?P<unit>.+)",
        min_qv: float = 20.0, tile: float = 200.0, block_draws: int = 1000, min_null_cells: int = 1000,
        seed: int = 0, verbose: bool = True):
    """Run the chain into OUT (must be new or empty). Returns the report text."""
    from . import compartment as cm
    from . import ingest
    say = print if verbose else (lambda *a, **k: None)
    out = Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; choose a new output folder")
    if bool(xenium) == bool(h5ad):
        raise ValueError("give either --xenium folders or --h5ad files")
    plan = []                                           # find every input before anything is written
    for root in xenium:
        secs = (ingest.find_xenium_sections(root) if not xenium_prefix
                else [(Path(root).expanduser(), xenium_prefix, xenium_prefix)])
        if not secs:
            raise FileNotFoundError(f"no Xenium section (transcripts.parquet) in {root} or its subfolders")
        plan += secs
    for f in h5ad:
        if not Path(f).expanduser().exists():
            raise FileNotFoundError(f)
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(ref=str(ref), ref_level=ref_level, ref_merge=ref_merge, exclude_genes=exclude_genes,
                    unit_pattern=unit_pattern, min_qv=min_qv, tile=tile, block_draws=block_draws,
                    min_null_cells=min_null_cells, seed=seed, xenium=[str(x) for x in xenium],
                    h5ad=[str(x) for x in h5ad])
    from . import __version__
    settings["claimbreak"] = __version__
    (out / "settings.json").write_text(json.dumps(settings, indent=1))

    # 1. compartment .h5ad files
    if xenium:
        files = []
        for folder, pre, name in plan:
            if any(f.stem == name for f in files):
                name = f"{name}_{len(files)}"
            say(f"== section {name} ({folder})", flush=True)
            files.append(ingest.xenium_to_h5ad(folder, out / "h5ad", name=name, sample=pre, min_qv=min_qv,
                                               verbose=verbose))
    else:
        files = [Path(f).expanduser() for f in h5ad]

    # 2. labels
    excl = ()
    if exclude_genes:
        rows = [ln.split(",")[0].strip() for ln in open(Path(exclude_genes).expanduser()).read().splitlines()]
        excl = tuple(g for g in rows if g and not g.startswith("#") and g.lower() != "gene")
    say(f"== labeling {len(files)} section(s) against {ref} ({ref_level})", flush=True)
    R = cm.Reference.load(ref, {LEVEL: ref_level}, merge=ref_merge, donor=ref_donor, verbose=verbose)
    cfg = cm.LabelConfig(exclude_genes=excl)
    cm.label_files(R, files, out / "labels", [LEVEL], list(ARMS), cfg, seed=seed, verbose=verbose)

    # 3. audit
    say("== label audit", flush=True)
    excl_found, over, _ = cm.audit_summary(out / "labels", out / "audit", verbose=verbose)

    # 4. compare and summary
    say("== label changes and where they land", flush=True)
    cm.compare(out / "labels", out / "compare", [LEVEL], f"{NULL},{TEST}", null=NULL, tile=tile, verbose=verbose)
    say("== extra-change ratio and kept ratio", flush=True)
    r = cm.summarise(out / "compare", out / "summary", None, test=TEST, null=NULL, level=LEVEL,
                     min_null_cells=min_null_cells, unit_pattern=unit_pattern, block_draws=block_draws, seed=seed,
                     focus=r".")

    # 5. tables and report
    S = pd.read_csv(out / "compare" / "summary.csv")
    gate = (S[(S.comparison == NULL) & (S.level == LEVEL)][["sample", "change_rate"]]
            .assign(halves_agreement=lambda t: 1 - t.change_rate).drop(columns="change_rate"))
    sec = r["sections"][["sample", "unit", "used"]].merge(gate, on="sample", how="left")
    sec["stable_at_nuclear_depth"] = sec.halves_agreement >= GATE
    sec.to_csv(out / "sections.csv", index=False)
    K = r.get("units_kept")
    if K is None or not len(K):
        K = pd.DataFrame(columns=["unit", "type", "kept_ratio", "kept_ratio_lo", "kept_ratio_hi", "cells_null"])
    K = K.copy()
    K["enough_cells"] = K.cells_null >= MIN_KEPT_CELLS
    # how stable each type is at nuclear depth -- of the cells half A calls t, the share half B also calls t.
    # A type the null itself rarely keeps has an unstable denominator (kept ratios far from one, even above one).
    rows = []
    for f in sorted((out / "labels").glob("*.labels.csv.gz")):
        lab = pd.read_csv(f, keep_default_na=False, low_memory=False, usecols=[f"{LEVEL}_nucA", f"{LEVEL}_nucB"])
        sample = f.name.replace(".labels.csv.gz", "")
        a, b = lab[f"{LEVEL}_nucA"].astype(str), lab[f"{LEVEL}_nucB"].astype(str)
        ok = (a != "") & (b != "")
        g = pd.DataFrame({"type": a[ok], "same": (a[ok] == b[ok])}).groupby("type").same.agg(["sum", "size"])
        g["unit"] = cm.unit_of(sample, unit_pattern)
        rows.append(g.reset_index())
    if rows:
        N = pd.concat(rows).groupby(["unit", "type"])[["sum", "size"]].sum().reset_index()
        N["null_kept"] = N["sum"] / N["size"]
        K = K.merge(N[["unit", "type", "null_kept"]], on=["unit", "type"], how="left")
    else:
        K["null_kept"] = np.nan
    K["stable"] = K.null_kept >= TYPE_GATE
    K = K.sort_values(["unit", "kept_ratio"])
    K.to_csv(out / "kept_ratio.csv", index=False)
    U = r["units"]
    Su = pd.read_csv(out / "audit" / "support.csv") if (out / "audit" / "support.csv").exists() else pd.DataFrame()
    text = report(sec, K, U, excl_found, over, Su, settings, out)
    (out / "report.md").write_text(text)
    say(f"\n{text}\nwrote {out}/report.md, kept_ratio.csv, sections.csv", flush=True)
    return text


def _f(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def report(sec, K, U, excl, over, Su, settings, out):
    L = [f"# Kept-ratio check", "",
         f"claimbreak {settings['claimbreak']}; reference {settings['ref']} ({settings['ref_level']}); "
         f"{len(sec)} section(s); output {out}.", ""]
    # stability
    bad = sec[~sec.stable_at_nuclear_depth.fillna(False)]
    L += ["## Are the labels stable at nuclear depth?", ""]
    L.append(f"Agreement between the two nuclear halves ranged from {_f(sec.halves_agreement.min())} to "
             f"{_f(sec.halves_agreement.max())}.")
    if len(bad):
        L.append(f"{len(bad)} section(s) fall below {GATE}: {', '.join(bad['sample'].astype(str))}. Their labels "
                 "change a lot from counting noise alone, so the ratios below are unreliable there. A coarser level "
                 "of the reference, or a reference closer to the tissue, usually helps.")
    else:
        L.append(f"All sections are at or above {GATE}, so the ratios below can be read.")
    L.append("")
    # audit
    L += ["## Label audit", ""]
    if len(excl):
        top = over[over.gene.isin(excl)].drop_duplicates("gene").head(8)
        L.append(f"{len(excl)} gene(s) are detected far above their share in every reference type and can act as "
                 f"false markers: {', '.join(map(str, excl[:12]))}{' ...' if len(excl) > 12 else ''}.")
        if len(top):
            L.append("Largest: " + "; ".join(f"{g.gene} (marker of {g.type}, median {100 * g.median_share:.1f}% of "
                                             f"counts)" for g in top.itertuples()) + ".")
        L.append(f"Review the list ({out}/audit/overdetected.csv) before excluding anything. A gene can also be "
                 "detected above every reference type because the tissue holds a cell state the reference lacks "
                 "(tumor cells in a reference of normal tissue, for example); then it is real signal, and the "
                 "better fix is a reference that covers the tissue. To exclude the listed genes, rerun with "
                 f"`--exclude-genes {out}/audit/exclude_overdetected.txt` into a new folder.")
    else:
        L.append("No gene is detected far above its reference share.")
    if len(Su) and "ratio" in Su:
        s = (Su[Su.arm == "nucleus"].groupby("type").ratio.median().dropna().sort_values())
        low = s[s < 0.7]
        if len(low):
            L.append("Calls of " + ", ".join(f"{t} ({_f(v)})" for t, v in low.items()) + " carry their own markers "
                     "less often than true members would at that depth (observed over expected support, nuclear "
                     "labels). Treat those classes with care.")
        else:
            L.append("Every class's calls carry their own markers about as often as true members would.")
    L.append("")
    # kept ratio
    L += ["## How many labels do whole-cell counts lose?", "",
          "Kept ratio per type and unit, whole-cell counts against nuclear half B at equal depth (one: no loss "
          "beyond counting noise; 95% block-bootstrap interval over tiles). Lower bound: the nucleus carries some "
          "neighboring RNA too.", "",
          "| unit | type | kept ratio | 95% interval | null keeps | cells (null) |", "|---|---|---|---|---|---|"]
    for k in K[K.enough_cells].itertuples():
        flag = "" if k.stable else " (unstable)"
        L.append(f"| {k.unit} | {k.type} | {_f(k.kept_ratio)}{flag} | {_f(k.kept_ratio_lo)}–{_f(k.kept_ratio_hi)} | "
                 f"{_f(k.null_kept)} | {int(k.cells_null):,} |")
    L += ["", f"Null keeps: of the cells half A calls that type, the share half B also calls it. Below {TYPE_GATE}, "
          "the type is not stable at nuclear depth and its kept ratio can stray far from one, even above one; read "
          "those rows as unreliable."]
    few = K[~K.enough_cells]
    if len(few):
        L.append("")
        L.append(f"Not shown (fewer than {MIN_KEPT_CELLS} cells in the null): "
                 + ", ".join(sorted(set(few.type.astype(str)))) + ". All rows are in kept_ratio.csv.")
    lost = K[K.enough_cells & K.stable & (K.kept_ratio_hi < 1 - MIN_LOSS)]
    if len(lost):
        g = lost.groupby("type").kept_ratio.median().sort_values()
        L += ["", "Among the stable types, whole-cell labels lose " +
              ", ".join(f"{t} ({100 * (1 - v):.0f}%)" for t, v in g.items()) +
              f" beyond counting noise (median over units; intervals below {1 - MIN_LOSS:.2f}). Small, RNA-poor "
              "cells that border larger ones usually lose most; consider labeling them from nuclear counts."]
    elif len(K):
        L += ["", f"No stable type loses more than {100 * MIN_LOSS:.0f}% of its labels beyond counting noise."]
    L.append("")
    # extra-change ratio
    L += ["## Do the changes follow the neighbors?", ""]
    if len(U) and "evaluable" in U:
        ev = U[U.evaluable]
        if len(ev):
            col_lo = "block_contact_lo" if "block_contact_lo" in ev else "pooled_contact_lo"
            above = int((ev[col_lo] > 1).sum()) if col_lo in ev else 0
            L.append(f"Extra-change ratio at contact: median {_f(ev.pooled_contact.median())} over {len(ev)} "
                     f"evaluable unit(s), with its interval above one in {above}. Above one means label changes "
                     "beyond counting noise take the type of a nearby cell more often than chance.")
            if "pooled_contact_reg" in ev:
                lo_r = "block_contact_reg_lo" if "block_contact_reg_lo" in ev else "pooled_contact_reg_lo"
                above_r = int((ev[lo_r] > 1).sum()) if lo_r in ev else 0
                L.append(f"Matched on each cell's surroundings (the composition of cells 30-100 um away, and the "
                         f"number of contacts): median {_f(ev.pooled_contact_reg.median())}, interval above one in "
                         f"{above_r}. Above one here means the changes follow the touching cells, not just the "
                         "region; about one means they follow the region.")
        else:
            L.append("No unit has enough extra changes (at least 50) for a stable ratio.")
    L += ["", "## Reading these numbers", "",
          "- The kept ratio counts labels lost, not labels gained: the types that absorb the lost labels are not "
          "scored.",
          "- Compare two segmentations by their kept ratios only if they share the nuclear counts; a method that "
          "also changes the nucleus needs protein or another reference.",
          "- Errors that nucleus and whole cell share (reference, label granularity, classifier) are invisible to "
          "this check."]
    return "\n".join(L) + "\n"
