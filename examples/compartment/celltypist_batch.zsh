#!/bin/zsh
# celltypist_batch.zsh -- the compartment test with a second labeler (CellTypist).
#   zsh $CB/examples/compartment/celltypist_batch.zsh check                 # settings per tissue, nothing run
#   PILOT=1 caffeinate -ims zsh $CB/examples/compartment/celltypist_batch.zsh lung 2>&1 | tee ~/claimbreak_results/logs/batchD_pilot_$(date +%Y%m%d_%H%M).log
#   caffeinate -ims zsh $CB/examples/compartment/celltypist_batch.zsh lung kidney gut ra 2>&1 | tee ~/claimbreak_results/logs/batchD_$(date +%Y%m%d_%H%M).log
#   [PILOT=1] [TNK='<regex>'] zsh $CB/examples/compartment/celltypist_batch.zsh tnk lung   # T/NK step only
#   zsh $CB/examples/compartment/celltypist_batch.zsh harvest               # ~/Downloads/cb_batchD_<time>.zip
# Per tissue: the samples, reference options, min-counts, markers, excluded genes and seed of the existing
# multinomial labels (read from their meta.json), relabeled with --labeler celltypist on arms nucleus, nucA, nucB,
# wholeB (same seed, so the same count sets); reference check (5-fold, by donor) per panel; compare (nucB|nucA,
# wholeB|nucA; neighbors typed by the celltypist nucleus labels); summary with pooled unit ratios; T/NK kept at equal
# depth (wholeB over nucB, common cells) for both labelers side by side. Reads only; writes ~/claimbreak_results/ct_<tissue>.
# PILOT=1: first sample only, into ~/claimbreak_results/ct_<tissue>_pilot.
# Memory: CellTypist trains with one job (CTJOBS=1; more jobs multiply memory) and predicts in chunks of 5,000
# cells. On the WSL workstation: D=/mnt/e/fibrosis R=~/claimbreak_work/batchD LUNG_LABELS=<folder with the lung v4 labels>
# CB_DOWNLOADS=/mnt/c/Users/$USER/Downloads zsh celltypist_batch.zsh check   (then lung kidney gut ra; no caffeinate).
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
# RA_LAB = synovium labels folder (default gseh2_pre); with R=<new root> the synovium results of an
#   audited relabeling (e.g. RA_LAB=$D/data/RA_labels/gseh4_pre/all R=~/claimbreak_results) go to a new folder.
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

# tissue: labels folder of the multinomial run | h5ad folder | reference | level=column | merge | level | unit pattern
typeset -A LAB H5 REF LEVCOL MERGE LEVEL UNIT
LAB[lung]=${LUNG_LABELS:-$R/bilous_v4};                    H5[lung]=$D/Bilous_h5ad
REF[lung]=$D/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad
LEVCOL[lung]=L2m=Level2; MERGE[lung]='^Tu_L\d=>Tumour'; LEVEL[lung]=L2m; UNIT[lung]='L\d(?=_L)'
LAB[gut]=$D/IBD/labels/xenium_v1;          H5[gut]=$D/IBD/h5ad;  REF[gut]=$D/IBD/ref/gut_census_v2.h5ad
LEVCOL[gut]=L1=coarse; MERGE[gut]=''; LEVEL[gut]=L1; UNIT[gut]='I0\d{3}'
LAB[kidney]=$D/AIN/labels/v1;              H5[kidney]=$D/AIN/h5ad; REF[kidney]=$D/AIN/ref/kidney_census.h5ad
LEVCOL[kidney]=L1=coarse; MERGE[kidney]=''; LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'
LAB[ra]=${RA_LAB:-$D/data/RA_labels/gseh2_pre/all};   H5[ra]=$D/data/RA_h5ad/compartment_pre; REF[ra]=$D/data/RA_ref/gse200815h2_ref.h5ad
LEVCOL[ra]=L1=coarse; MERGE[ra]=''; LEVEL[ra]=L1; UNIT[ra]='p\d{2}'

# options of the existing labels -> command-line options (and a file of excluded genes)
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
o += ["--min-counts", str(m.get("min_counts", 10)), "--seed", str(m.get("seed", 0))]
if m.get("markers", 0): o += ["--markers", str(m["markers"]), "--marker-min", str(m.get("marker_min", 1e-4)),
                             "--marker-min-log2", str(m.get("marker_min_log2", 1.0))]
if m.get("exclude_genes"):
    open(out, "w").write("\n".join(m["exclude_genes"]) + "\n"); o += ["--exclude-genes", out]
print(" ".join(f"'{x}'" for x in o))
print(f"existing labels: reference {os.path.basename(str(m.get('reference')))}, levels {m.get('ref_levels')}, "
      f"merge {m.get('ref_merge')!r}, arms {m.get('arms')}, floor {m.get('floor')}", file=sys.stderr)
EOF
}

samples() {   # samples of the existing labels that have an h5ad
  local t=$1 f s
  for f in ${LAB[$t]}/*.labels.csv.gz; do s=${f:t}; s=${s%.labels.csv.gz}; [[ -f ${H5[$t]}/$s.h5ad ]] && print ${H5[$t]}/$s.h5ad; done
}

tnk() { python - "$@" <<'EOF'
# T/NK kept at equal depth: per unit, T/NK labels from wholeB over T/NK labels from nucB on cells labeled in both
import glob, os, re, sys
import numpy as np, pandas as pd
mn, ct, lv, unit, out = sys.argv[1:6]
# T/NK types: a name with a stand-alone "T" or "NK" token ("T/NK cells", "T_NK", "T cells", "NK", "CD8 T"); TNK=<regex>
# overrides. Types matched are printed; if none, all type names are printed.
rx = re.compile(os.environ.get("TNK") or r"(?i)(^|[^a-z])(t|nk|tnk)([^a-z]|$)")
rows = []
for f in sorted(glob.glob(os.path.join(ct, "*.labels.csv.gz"))):
    s = os.path.basename(f).replace(".labels.csv.gz", "")
    u = re.search(unit, s); u = u.group(0) if u else s
    for name, path in (("celltypist", f), ("multinomial", os.path.join(mn, os.path.basename(f)))):
        if not os.path.exists(path): continue
        L = pd.read_csv(path, usecols=lambda c: c in (f"{lv}_nucB", f"{lv}_wholeB"), keep_default_na=False)
        ok = (L[f"{lv}_nucB"] != "") & (L[f"{lv}_wholeB"] != "")
        types = sorted(set(L.loc[ok, f"{lv}_nucB"]) | set(L.loc[ok, f"{lv}_wholeB"]))
        tn = [t for t in types if rx.match(t)]
        rows.append({"sample": s, "unit": u, "labeller": name, "tnk_types": "|".join(tn), "cells": int(ok.sum()),
                     "tnk_nucB": int(L.loc[ok, f"{lv}_nucB"].isin(tn).sum()),
                     "tnk_wholeB": int(L.loc[ok, f"{lv}_wholeB"].isin(tn).sum())})
T = pd.DataFrame(rows)
if not len(T) or T.tnk_types.eq("").all():
    seen = set()
    for f in sorted(glob.glob(os.path.join(ct, "*.labels.csv.gz")))[:2]:
        seen |= set(pd.read_csv(f, usecols=[f"{lv}_nucB"], keep_default_na=False)[f"{lv}_nucB"]) - {""}
    print("no T/NK type matched; types in the labels:", sorted(seen), "-> rerun with TNK='<regex>' ... tnk <tissue>")
T.to_csv(os.path.join(out, "tnk_sections.csv"), index=False)
U = T.groupby(["labeller", "unit"])[["tnk_nucB", "tnk_wholeB", "cells"]].sum().reset_index()
U["kept"] = U.tnk_wholeB / U.tnk_nucB.where(U.tnk_nucB > 0)
U.to_csv(os.path.join(out, "tnk_units.csv"), index=False)
print("T/NK types matched:", T.groupby("labeller").tnk_types.first().to_dict())
for name, g in U.groupby("labeller"):
    print(f"{name}: T/NK kept at equal depth, median over {g.kept.notna().sum()} units {g.kept.median():.3f} "
          f"(range {g.kept.min():.3f}-{g.kept.max():.3f}); below one in {(g.kept < 1).sum()}")
EOF
}

run() {
  local t=$1 W=$R/ct_$1 opts ins
  [[ -n ${LAB[$t]} ]] || stop "unknown tissue $t (lung, gut, kidney, ra)"
  [[ -n $PILOT ]] && W=${W}_pilot
  stamp "$t: settings"
  [[ -d ${LAB[$t]} && -d ${H5[$t]} && -f ${REF[$t]} ]] || stop "$t: missing ${LAB[$t]}, ${H5[$t]} or ${REF[$t]}"
  mkdir -p $W/logs
  opts=$(optpy ${LAB[$t]} $W/exclude_genes.txt) || stop "$t: no meta.json in ${LAB[$t]}"
  ins=( ${(f)"$(samples $t)"} ); [[ -n $PILOT ]] && ins=( $ins[1] )
  print "$t: ${#ins} samples; options ${opts}"
  (( ${#ins} )) || stop "$t: no sample with both labels and an h5ad"
  stamp "$t: label (celltypist)"
  eval claimbreak compartment label --ref ${REF[$t]} --ref-levels ${LEVCOL[$t]} --ref-merge "'${MERGE[$t]}'" \
    --inputs ${(q)ins} --out $W/labels --arms nucleus,nucA,nucB,wholeB --labeller celltypist --ct-jobs ${CTJOBS:-1} \
    --ref-cv --cv-folds 5 --cv-depths 20,50,100,200 $opts 2>&1 | tee -a $W/logs/label.log | grep -v "^  wrote"
  stamp "$t: compare + summary"
  claimbreak compartment compare --labels $W/labels --out $W/compare --levels ${LEVEL[$t]} \
    --comparisons "nucB|nucA,wholeB|nucA" > $W/logs/compare.log 2>&1 || { tail -5 $W/logs/compare.log; stop "$t compare" }
  claimbreak compartment summary --compare $W/compare --out $W/summary --level ${LEVEL[$t]} --unit-pattern "${UNIT[$t]}" \
    2>&1 | tee $W/logs/summary.log | grep -E "^sections used|^per unit, pooled|^extra-change ratio" | cut -c1-220
  stamp "$t: T/NK kept, both labellers"
  tnk ${LAB[$t]} $W/labels ${LEVEL[$t]} "${UNIT[$t]}" $W 2>&1 | tee $W/logs/tnk.log
}

case $1 in
  check)
    python -c "import claimbreak, celltypist, sklearn; print('claimbreak', claimbreak.__version__, '| celltypist', celltypist.__version__, '| scikit-learn', sklearn.__version__)" 2>/dev/null \
      || print "claimbreak 1.7.103 or later and celltypist needed: pip install celltypist"
    for t in lung gut kidney ra; do
      print "\n$t: labels ${LAB[$t]}; h5ad ${H5[$t]}; reference ${REF[$t]:t}"
      [[ -d ${LAB[$t]} ]] || { print "  labels folder missing"; continue }
      print "  $(samples $t | wc -l | tr -d ' ') samples with labels and h5ad; reference file $([[ -f ${REF[$t]} ]] && print found || print MISSING)"
      optpy ${LAB[$t]} /dev/null
    done ;;
  tnk)          # T/NK step only, on finished runs: zsh celltypist_batch.zsh tnk lung [kidney ...]
    shift
    for t in "$@"; do W=$R/ct_$t; [[ -n $PILOT ]] && W=${W}_pilot
      stamp "$t: T/NK kept, both labellers"; tnk ${LAB[$t]} $W/labels ${LEVEL[$t]} "${UNIT[$t]}" $W 2>&1 | tee $W/logs/tnk.log; done ;;
  harvest)
    DL=${CB_DOWNLOADS:-~/Downloads}; cd $R && zip -q -r $DL/cb_batchD_$(date +%Y%m%d_%H%M).zip ct_*/{compare/summary.csv,compare/did.csv,summary,logs} \
      ct_*/tnk_*.csv ct_*/labels/*.meta.json ct_*/labels/ref_cv*.csv logs/batchD_*.log(N)
    ls -lh $DL/cb_batchD_*.zip(om[1]) ;;
  *)
    python -c "import claimbreak, sys; sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
      || stop "claimbreak older than 1.7.103"
    python -c "import celltypist" 2>/dev/null || stop "celltypist missing: pip install celltypist"
    [[ -d $D ]] || stop "drive not mounted at $D"
    for t in "$@"; do run $t; done
    stamp "batch D done" ;;
esac
