#!/bin/zsh
# cytob_add.zsh -- add the pure-cytoplasm arm (cytoB) to label runs
# that lack it (lung bilous_v4, CosMx x171), then run the exposure analysis with cytoB as the test.
#   zsh cytob_add.zsh lung cosmx 2>&1 | tee ~/claimbreak_results/logs/cytob_add_$(date +%Y%m%d_%H%M).log
# Per tissue: (1) `claimbreak compartment label` with arms nucA,nucB,cytoB and the options of the existing run (read
# from its meta.json, as in celltypist_batch.zsh); the nuclear halves come from the same seed, so nucA/nucB must equal
# the existing labels (checked; stops otherwise). (2) copies of the existing labels with the cytoB columns added, in
# ~/claimbreak_results/cytoB_<tissue>/merged. (3) exposure with --test cytoB. Reads the existing labels and h5ad files only.
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; R=${R:-~/claimbreak_results}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

typeset -A LAB H5 REF LEVCOL MERGE LEVEL UNIT
LAB[lung]=$R/bilous_v4;  H5[lung]=$D/Bilous_h5ad
LAB[cosmx]=$R/cosmx_nsclc_x171/multinomial/labels;  H5[cosmx]=$D/data/cosmx_nsclc/h5ad
for t in lung cosmx; do
  REF[$t]=$D/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad; LEVCOL[$t]=L2m=Level2; MERGE[$t]='^Tu_L\d=>Tumour'; LEVEL[$t]=L2m
done
UNIT[lung]='L\d(?=_L)'; UNIT[cosmx]='Lung\d+'

optpy() { python - "$@" <<'EOF'
import json, sys, glob, os
lab, out = sys.argv[1], sys.argv[2]
m = json.load(open(sorted(glob.glob(os.path.join(lab, "*.meta.json")))[0]))
o = []
lay = m.get("ref_layer", "X")
if lay == "raw": o += ["--ref-raw"]
elif lay not in ("X", "", None): o += ["--ref-layer", lay]
for k, f in (("ref_map", "--ref-map"), ("ref_subset", "--ref-subset")):
    if m.get(k): o += [f, m[k]]
if m.get("cv_group"): o += ["--cv-group", m["cv_group"]]
o += ["--min-counts", str(m.get("min_counts", 10)), "--seed", str(m.get("seed", 0)), "--floor", str(m.get("floor", 0.01)),
      "--gamma-iter", str(m.get("gamma_iter", 5))]
if m.get("markers", 0): o += ["--markers", str(m["markers"]), "--marker-min", str(m.get("marker_min", 1e-4)),
                             "--marker-min-log2", str(m.get("marker_min_log2", 1.0))]
if m.get("exclude_genes"):
    open(out, "w").write("\n".join(m["exclude_genes"]) + "\n"); o += ["--exclude-genes", out]
print(" ".join(f"'{x}'" for x in o))
print(f"existing labels: reference {os.path.basename(str(m.get('reference')))}, levels {m.get('ref_levels')}, "
      f"merge {m.get('ref_merge')!r}, arms {m.get('arms')}, seed {m.get('seed')}", file=sys.stderr)
EOF
}

merge() { python - "$@" <<'EOF'
# copies of the existing labels with the new cytoB columns; nucA and nucB must be identical in both runs
import glob, os, sys
import pandas as pd
old, new, out, lv = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(new, "*.labels.csv.gz"))):
    s = os.path.basename(f)
    A = pd.read_csv(os.path.join(old, s), keep_default_na=False, low_memory=False, dtype={"cell_id": str})
    B = pd.read_csv(f, keep_default_na=False, low_memory=False, dtype={"cell_id": str})
    B = B.set_index("cell_id").reindex(A["cell_id"])
    for arm in ("nucA", "nucB"):
        same = (A[f"{lv}_{arm}"].to_numpy() == B[f"{lv}_{arm}"].fillna("").to_numpy()).mean()
        print(f"  {s}: {arm} identical for {same:.4f} of cells")
        if same < 0.999:
            sys.exit(f"STOP {s}: {arm} differs from the existing run ({same:.4f}); not merged")
    for c in (f"{lv}_cytoB", f"{lv}_cytoB_second", f"{lv}_cytoB_margin", "n_cytoB"):
        if c in B:
            A[c] = B[c].fillna("").to_numpy()
    A.to_csv(os.path.join(out, s), index=False, compression="gzip")
print(f"merged into {out}")
EOF
}

for t in "$@"; do
  [[ -n ${LAB[$t]} ]] || stop "unknown tissue $t (lung, cosmx)"
  W=$R/cytoB_$t; [[ -e $W ]] && stop "$W exists; move it away"
  mkdir -p $W
  stamp "$t: label nucA, nucB, cytoB"
  opts=$(optpy ${LAB[$t]} $W/exclude_genes.txt) || stop "$t: no meta.json in ${LAB[$t]}"
  ins=()
  for f in ${LAB[$t]}/*.labels.csv.gz; do s=${f:t}; s=${s%.labels.csv.gz}; [[ -f ${H5[$t]}/$s.h5ad ]] && ins+=(${H5[$t]}/$s.h5ad); done
  print "$t: ${#ins} samples; options $opts"
  (( ${#ins} )) || stop "$t: no h5ad for the labelled samples"
  eval claimbreak compartment label --ref ${REF[$t]} --ref-levels ${LEVCOL[$t]} --ref-merge "'${MERGE[$t]}'" \
    --inputs ${(q)ins} --out $W/labels --arms nucA,nucB,cytoB $opts 2>&1 | grep -v "^  wrote" | cut -c1-200
  stamp "$t: merge"
  merge ${LAB[$t]} $W/labels $W/merged ${LEVEL[$t]} || stop "$t: merge"
  stamp "$t: exposure, test cytoB"
  claimbreak compartment exposure --labels $W/merged --h5ad-dir ${H5[$t]} --out $W/exposure --levels ${LEVEL[$t]} \
    --unit-pattern "${UNIT[$t]}" --test cytoB --gaps 2,6,10,15 --contact-gap 10 --focus '(?i)^t_?nk$' 2>&1 | tail -45
done
