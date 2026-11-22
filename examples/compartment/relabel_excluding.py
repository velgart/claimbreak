#!/usr/bin/env python
"""relabel_excluding.py -- relabel a labels folder with extra genes excluded, every other
setting taken from the folder's own meta.json files, then pool the new label audit.

  python relabel_excluding.py --from LABELS --h5ad-dir H5 --out NEW --exclude CCL5,EEF1G
  python relabel_excluding.py --from LABELS --h5ad-dir H5 --out NEW --exclude @LABELS_audit/exclude_overdetected.txt

Read-only on LABELS and H5; refuses a non-empty NEW. All sections of LABELS must share the labeling settings
(reference, levels, arms, seed, classifier options); the exclusion list is LABELS' own plus --exclude. Writes NEW
(labels, gamma, meta.json, audit.csv per section, as 'compartment label') and NEW_audit (compartment audit).
Example (synovium): gseh2_pre -> gseh4_pre with CCL5 and EEF1G excluded.
"""
import argparse
import glob
import json
import os
import sys
from dataclasses import fields

KEYS = ("reference", "ref_levels", "ref_merge", "ref_map", "ref_subset", "ref_layer", "cv_group", "levels", "arms",
        "seed")


def genes_from(spec):
    """'A,B' or '@file' (one gene per line or a CSV's first field; '#' comments and a 'gene' header skipped)."""
    if spec.startswith("@"):
        rows = [ln.split(",")[0].strip() for ln in open(os.path.expanduser(spec[1:])).read().splitlines()]
    else:
        rows = [g.strip() for g in spec.split(",")]
    return [g for g in rows if g and not g.startswith("#") and g.lower() != "gene"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, help="labels folder written by 'compartment label'")
    ap.add_argument("--h5ad-dir", required=True, help="compartment .h5ad files (<sample>.h5ad)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", required=True, help="genes to add to the exclusion list: A,B or @file")
    ap.add_argument("--reference", default="", help="reference .h5ad (default: the path in meta.json)")
    ap.add_argument("--samples", default="", help="comma-separated subset of samples (default: all)")
    ap.add_argument("--min-sections", type=float, default=0.5, help="for the pooled audit")
    a = ap.parse_args(argv)
    from claimbreak import __version__
    from claimbreak import compartment as cm
    src, out = os.path.expanduser(a.src), os.path.expanduser(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} exists and is not empty")
    metas = sorted(glob.glob(os.path.join(src, "*.meta.json")))
    if a.samples:
        keep = set(a.samples.split(","))
        metas = [f for f in metas if os.path.basename(f).replace(".meta.json", "") in keep]
    if not metas:
        sys.exit(f"no meta.json in {src}")
    M = [json.load(open(f)) for f in metas]
    cfgnames = [f.name for f in fields(cm.LabelConfig)]
    m = M[0]
    for f, o in zip(metas[1:], M[1:]):
        diff = [k for k in KEYS + tuple(cfgnames) if k in m and o.get(k) != m.get(k)]
        if diff:
            sys.exit(f"{os.path.basename(f)} was labelled with other settings ({', '.join(diff)}); relabel it apart")
    kw = {k: (tuple(m[k]) if isinstance(m.get(k), list) else m[k]) for k in cfgnames if k in m}
    add = genes_from(a.exclude)
    kw["exclude_genes"] = tuple(sorted(set(kw.get("exclude_genes", ())) | set(add)))
    cfg = cm.LabelConfig(**kw)
    layer = m.get("ref_layer", "X")
    ref = cm.Reference.load(a.reference or m["reference"], m["ref_levels"], merge=m.get("ref_merge", ""),
                            map_file=m.get("ref_map", ""), raw=(layer == "raw"),
                            layer="" if layer in ("X", "raw") else layer, subset=m.get("ref_subset", ""),
                            cv_group=m.get("cv_group", ""))
    inputs = [os.path.join(os.path.expanduser(a.h5ad_dir), os.path.basename(f).replace(".meta.json", ".h5ad"))
              for f in metas]
    missing = [f for f in inputs if not os.path.exists(f)]
    if missing:
        sys.exit(f"missing compartment files: {missing[:3]}")
    print(f"claimbreak {__version__} | {len(inputs)} sections from {src} | arms {m['arms']} | excluded "
          f"{', '.join(cfg.exclude_genes)} ({len(add)} added)", flush=True)
    cm.label_files(ref, inputs, out, m["levels"], m["arms"], cfg, seed=m["seed"])
    with open(os.path.join(out, "relabel_from.json"), "w") as fh:
        json.dump(dict(source=src, added=add, exclude_genes=list(cfg.exclude_genes), claimbreak=__version__,
                       samples=[os.path.basename(f).replace(".meta.json", "") for f in metas]), fh, indent=1)
    cm.audit_summary(out, out.rstrip("/") + "_audit", min_sections=a.min_sections)
    return 0


if __name__ == "__main__":
    sys.exit(main())
