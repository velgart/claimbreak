#!/bin/zsh
# stats_batch.zsh -- statistics for the paper on the existing labels (no
# relabeling): block bootstrap over 200 um tiles, sensitivity to the reference-label margin, pooled contamination
# shares with the homotypic baseline, and the flip curve with rho and the label change read from independent halves.
#   zsh $CB/examples/compartment/stats_batch.zsh check
#   caffeinate -ims zsh $CB/examples/compartment/stats_batch.zsh lung gut kidney ra cosmx 2>&1 | tee ~/claimbreak_results/logs/batchS_$(date +%Y%m%d_%H%M).log
#   SKIP_CONTAM=1 zsh $CB/examples/compartment/stats_batch.zsh lung        # compare / summary only
#   zsh $CB/examples/compartment/stats_batch.zsh harvest                   # ~/Downloads/cb_batchS_<time>.zip
# Per tissue, into ~/claimbreak_results/stat_<tissue>:
#  (1) for each margin m in MARGINS (nats, reference arm nucA): `compartment compare` (nucB|nucA, wholeB|nucA,
#      cytoB|nucA; contact = 6 nearest within 30 um, as in the paper) with tiles of 200 um, and `compartment summary`
#      for test wholeB and test cytoB: per unit pooled extra-change ratio with the parametric and the block interval,
#      T/NK kept ratio with its block interval (units_kept.csv);
#  (2) contamination.py --split-typing --rho-split: pooled foreign shares (pooled.csv, pooled_units.csv with the
#      homotypic baseline subtracted), rho > 1 shares, flip_curve_split.csv;
#  (3) margin_sensitivity.csv: one row per margin and test.
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
# LR = root of the lung and CosMx labels (default R), so a rerun can write stat_<tissue> under a new R while
#   reading the labels where they are, e.g. LR=~/claimbreak_results R=~/claimbreak_results MARGINS=0 SKIP_CONTAM=1
LR=${LR:-$R}
# RA_LAB = synovium labels folder (default gseh2_pre); with R=<new root> the synovium results of an
#   audited relabeling (e.g. RA_LAB=$D/data/RA_labels/gseh4_pre/all R=~/claimbreak_results) go to a new folder.
MARGINS=${MARGINS:-0,5,10,20}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

# tissue: labels for compare (with cytoB) | labels for contamination (with gamma.csv) | h5ad | reference | levels | merge | level | unit
typeset -A CMPL ORIG H5 REF LEVCOL MERGE LEVEL UNIT
CMPL[lung]=$LR/cytoB_lung/merged;   ORIG[lung]=${LUNG_LABELS:-$LR/bilous_v4};  H5[lung]=$D/Bilous_h5ad
CMPL[cosmx]=$LR/cytoB_cosmx/merged; ORIG[cosmx]=$LR/cosmx_nsclc_x171/multinomial/labels; H5[cosmx]=$D/data/cosmx_nsclc/h5ad
for t in lung cosmx; do
  REF[$t]=$D/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad; LEVCOL[$t]=L2m=Level2; MERGE[$t]='^Tu_L\d=>Tumour'; LEVEL[$t]=L2m
done
UNIT[lung]='L\d(?=_L)'; UNIT[cosmx]='Lung\d+'
CMPL[gut]=$D/IBD/labels/xenium_v1;  ORIG[gut]=${CMPL[gut]};  H5[gut]=$D/IBD/h5ad;  REF[gut]=$D/IBD/ref/gut_census_v2.h5ad
LEVCOL[gut]=L1=coarse; MERGE[gut]=''; LEVEL[gut]=L1; UNIT[gut]='I0\d{3}'
CMPL[kidney]=$D/AIN/labels/v1;      ORIG[kidney]=${CMPL[kidney]}; H5[kidney]=$D/AIN/h5ad; REF[kidney]=$D/AIN/ref/kidney_census.h5ad
LEVCOL[kidney]=L1=coarse; MERGE[kidney]=''; LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'
CMPL[ra]=${RA_LAB:-$D/data/RA_labels/gseh2_pre/all}; ORIG[ra]=${CMPL[ra]}; H5[ra]=$D/data/RA_h5ad/compartment_pre
REF[ra]=$D/data/RA_ref/gse200815h2_ref.h5ad; LEVCOL[ra]=L1=coarse; MERGE[ra]=''; LEVEL[ra]=L1; UNIT[ra]='p\d{2}'

table() { python - "$@" <<'EOF'
# one row per margin and test: units evaluable, pooled ratio, intervals above one (parametric, block); T/NK kept
import glob, os, re, sys
import pandas as pd
W = sys.argv[1]
rows = []
for d in sorted(glob.glob(os.path.join(W, "sum_m*_*"))):
    m, test = re.match(r".*sum_m([\d.]+)_(\w+)$", d).groups()
    U = pd.read_csv(os.path.join(d, "units.csv")) if os.path.exists(os.path.join(d, "units.csv")) else pd.DataFrame()
    K = pd.read_csv(os.path.join(d, "units_kept.csv")) if os.path.exists(os.path.join(d, "units_kept.csv")) else pd.DataFrame()
    S = pd.read_csv(os.path.join(d, "sections.csv"))
    ev = U[U.evaluable] if len(U) else U
    r = {"margin": float(m), "test": test, "null_cells": int(S.null_cells.sum()), "units": len(U), "evaluable": len(ev),
         "pooled_contact_median": ev.pooled_contact.median() if len(ev) else float("nan"),
         "param_ci_above_1": int(ev.ci_excludes_1.sum()) if len(ev) else 0,
         "block_ci_above_1": int(ev.block_ci_excludes_1.sum()) if len(ev) and "block_ci_excludes_1" in ev else 0}
    if len(K):
        k = K[K.cells_null >= 100]
        r.update({"tnk_units": len(k), "tnk_kept_median": k.kept_ratio.median(),
                  "tnk_block_hi_below_1": int((k.kept_ratio_hi < 1).sum())})
    rows.append(r)
T = pd.DataFrame(rows).sort_values(["test", "margin"])
T.to_csv(os.path.join(W, "margin_sensitivity.csv"), index=False)
pd.set_option("display.width", 220)
print(T.round(3).to_string(index=False))
EOF
}

run() {
  local t=$1 W=$R/stat_$1 L=${CMPL[$1]} m x
  [[ -n ${LEVEL[$t]} ]] || stop "unknown tissue $t (lung, gut, kidney, ra, cosmx)"
  [[ -d $L ]] || stop "$t: no labels $L"
  [[ -e $W ]] && stop "$W exists; move it away"
  mkdir -p $W/logs
  for m in ${(s:,:)MARGINS}; do
    stamp "$t: compare, margin $m"
    claimbreak compartment compare --labels $L --out $W/cmp_m$m --levels ${LEVEL[$t]} --min-margin $m --tile 200 \
      --comparisons "nucB|nucA,wholeB|nucA,cytoB|nucA" > $W/logs/compare_m$m.log 2>&1 || { tail -5 $W/logs/compare_m$m.log; stop "$t compare $m" }
    for x in wholeB cytoB; do
      claimbreak compartment summary --compare $W/cmp_m$m --out $W/sum_m${m}_$x --level ${LEVEL[$t]} --test "$x|nucA" \
        --unit-pattern "${UNIT[$t]}" > $W/logs/summary_m${m}_$x.log 2>&1 || { tail -5 $W/logs/summary_m${m}_$x.log; stop "$t summary $m $x" }
      grep -E "^per unit, pooled|^block bootstrap|^matched on regional" $W/logs/summary_m${m}_$x.log | sed "s/^/  $x: /" | cut -c1-230
    done
  done
  stamp "$t: margin sensitivity"
  table $W 2>&1 | tee $W/logs/margins.log
  if [[ -z $SKIP_CONTAM ]]; then
    stamp "$t: contamination, split typing and rho split"
    [[ -f ${REF[$t]} ]] || stop "$t: no reference ${REF[$t]}"
    python -u $CB/examples/compartment/contamination.py --h5ad-dir ${H5[$t]} --labels ${ORIG[$t]} \
      --ref ${REF[$t]} --ref-levels ${LEVCOL[$t]} --ref-merge "${MERGE[$t]}" --level ${LEVEL[$t]} --unit-pattern "${UNIT[$t]}" \
      --max-cells 50000 --split-typing --rho-split --out $W/contam > $W/logs/contam.log 2>&1 || { tail -5 $W/logs/contam.log; stop "$t contamination" }
    sed -n '/^pooled over cells/,/^wrote/p' $W/logs/contam.log | cut -c1-230
  fi
}

case $1 in
  check)
    python -c "import claimbreak, sys; print('claimbreak', claimbreak.__version__); sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
      || stop "claimbreak 1.7.103 or later needed"
    for t in lung gut kidney ra cosmx; do
      n=( ${CMPL[$t]}/*.labels.csv.gz ); f=( ${CMPL[$t]}/*.labels.csv.gz([1]) )
      c=$( (( ${#f} )) && python -c "import pandas as pd,sys; h=pd.read_csv(sys.argv[1],nrows=1).columns; print('cytoB' if any(x.endswith('_cytoB') for x in h) else 'NO cytoB')" $f[1])
      print "$t: compare labels ${CMPL[$t]} (${#n} files, $c); contamination labels ${ORIG[$t]}; reference ${REF[$t]:t} $([[ -f ${REF[$t]} ]] && print ok || print MISSING)"
    done ;;
  harvest)
    DL=${CB_DOWNLOADS:-~/Downloads}; cd $R && zip -q -r $DL/cb_batchS_$(date +%Y%m%d_%H%M).zip stat_*/margin_sensitivity.csv \
      stat_*/sum_m*/{sections,units,units_kept,pairs,pairs_units}.csv(N) stat_*/cmp_m*/{summary,did}.csv(N) \
      stat_*/contam/{by_type,pooled,pooled_units,source,flip_curve,flip_curve_split}.csv(N) stat_*/logs/*.log(N)
    ls -lh $DL/cb_batchS_*.zip(om[1]) ;;
  *)
    python -c "import claimbreak, sys; sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
      || stop "claimbreak older than 1.7.103"
    for t in "$@"; do run $t; done ;;
esac
