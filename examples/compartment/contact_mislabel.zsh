#!/bin/zsh
# contact_mislabel.zsh -- can the nuclear excess of
# partner RNA at contact (contact_contamination.zsh) come from partner cells labeled T/NK? Three steps, each
# refusing to overwrite:
#   1  shape of the existing runs (no counts read): contact_contamination.py --from-cells on
#      $R/contact_contamination/<tissue> -> .../tail/<tissue>; mean ratio against quantile ratios and AUC
#   2  label checks per tissue -> $R/contact_contamination/mislabel/<tissue>_<check>:
#        nucB         load also from nuclear half B (the half no label saw), all genes restricted to the label genes
#        nucB_m20     the same, measured cells with label margin >= 20 nats
#        nucB_m20_own3  and >= 3 nuclear counts of their own type's markers
#        thirds       labels nucA3, agreeing with nucC3, load from third B (only where the labels have the thirds)
#   3  renal carcinomas, T cells and partners by protein (no RNA label), 500 um tiles as units
#      -> $R/contact_contamination/mislabel/<rcc|ccrcc>_protein_<T|Tstrict>_<partner>
#
#   caffeinate -ims zsh $CB/examples/compartment/contact_mislabel.zsh 2>&1 | tee ~/claimbreak_results/logs/contact_mislabel_$(date +%Y%m%d_%H%M).log
#
# Env: D, LR, R, RA_LAB as contact_contamination.zsh; TISSUES (gut lung), STEPS (1 2 3), MARGIN (20), OWN (3).
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; LR=${LR:-~/claimbreak_results}; R=${R:-~/claimbreak_results}
RA_LAB=${RA_LAB:-$D/data/RA_labels/gseh4_pre/all}; TISSUES=(${=TISSUES:-gut lung}); STEPS=(${=STEPS:-1 2 3})
MARGIN=${MARGIN:-20}; OWN=${OWN:-3}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
python -c "import claimbreak, sys; v = tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(v < (1, 7, 103))" \
  || stop "claimbreak 1.7.103 or later needed"
[[ -d $D ]] || stop "data drive not mounted ($D)"
CC=$CB/examples/compartment/contact_contamination.py
O=$R/contact_contamination; M=$O/mislabel; mkdir -p $M/logs $O/tail

typeset -A LAB META LEVEL UNIT H5 PART
LAB[gut]=$D/IBD/labels/xenium_v1;  META[gut]=$D/IBD/labels/xenium_v1; LEVEL[gut]=L1;    UNIT[gut]='I0\d{3}';               H5[gut]=$D/IBD/h5ad;      PART[gut]=Epithelial
LAB[lung]=$LR/cytoB_lung/merged;   META[lung]=$LR/bilous_v4;          LEVEL[lung]=L2m;  UNIT[lung]='L\d(?=_L)';            H5[lung]=$D/Bilous_h5ad;  PART[lung]=Tumour
LAB[kidney]=$D/AIN/labels/v1;      META[kidney]=$D/AIN/labels/v1;     LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'; H5[kidney]=$D/AIN/h5ad;   PART[kidney]=PT
LAB[ra]=$RA_LAB;                   META[ra]=$RA_LAB;                  LEVEL[ra]=L1;     UNIT[ra]='p\d{2}';                 H5[ra]=$D/data/RA_h5ad/compartment_pre; PART[ra]=Fibroblast

show() { grep -E 'contact / away|^ +(whole|nucleus|nucB|nucB3)|by part of|^STOP|skipped|split|Error' $1 }

if (( ${STEPS[(Ie)1]} )); then
  for t in gut lung ra kidney; do
    s=$O/$t; o=$O/tail/$t
    [[ -f $s/cells.csv.gz ]] || { print "no $s/cells.csv.gz; step 1 skipped for $t"; continue }
    [[ -e $o ]] && { print "exists, kept: $o"; continue }
    stamp "1 $t: shape of the existing run"
    python -u $CC --from-cells $s --out $o 2>&1 | tee $M/logs/tail_$t.log | grep -vE '^wrote'
  done
fi

if (( ${STEPS[(Ie)2]} )); then
  for t in $TISSUES; do
    [[ -n ${LAB[$t]} ]] || stop "unknown tissue $t (gut lung ra kidney)"
    f=(${LAB[$t]}/*.labels.csv.gz); (( ${#f} )) || { print "no labels in ${LAB[$t]}; $t skipped"; continue }
    hdr=$(python -c "import pandas as pd, sys; print(' '.join(pd.read_csv(sys.argv[1], nrows=0).columns))" $f[1])
    typeset -a CHECKS; CHECKS=(nucB nucB_m$MARGIN nucB_m${MARGIN}_own$OWN)
    [[ " $hdr " == *" ${LEVEL[$t]}_nucA3 "* && " $hdr " == *" ${LEVEL[$t]}_nucC3 "* ]] && CHECKS+=(thirds) \
      || print "$t: no ${LEVEL[$t]}_nucA3 / _nucC3 columns; thirds check skipped"
    for c in $CHECKS; do
      o=$M/${t}_$c
      [[ -e $o ]] && { print "exists, kept: $o"; continue }
      case $c in
        nucB) XA=(--measure nucB) ;;
        nucB_m*_own*) XA=(--measure nucB --min-margin $MARGIN --min-own $OWN) ;;
        nucB_m*) XA=(--measure nucB --min-margin $MARGIN) ;;
        thirds) XA=(--measure nucB3 --arm nucA3 --agree nucC3) ;;
      esac
      stamp "2 $t $c: ${XA[*]}"
      python -u $CC --labels ${LAB[$t]} --meta ${META[$t]} --h5ad ${H5[$t]} --level ${LEVEL[$t]} --immune T_NK \
        --partner ${PART[$t]} --unit-pattern "${UNIT[$t]}" $XA --out $o > $M/logs/${t}_$c.log 2>&1 \
        || { tail -5 $M/logs/${t}_$c.log; print "  ($t $c failed; skipped)"; continue }
      show $M/logs/${t}_$c.log
      python - $o/markers.csv <<'EOF'
import sys, pandas as pd
m = pd.read_csv(sys.argv[1])
print(f"  measured T/NK cells: {m.n_immune_measured.sum():,} of {m.n_immune.sum():,} ({m.n_immune_measured.sum() / m.n_immune.sum():.0%}); "
      f"split reproduced (min over sections): {m.split_reproduced.min():.4f}")
EOF
    done
  done
fi

if (( ${STEPS[(Ie)3]} )); then
  typeset -A SAMPLE PROT H5P
  SAMPLE[rcc]=XP_RCC;     PROT[rcc]=$D/data/xenium_protein/h5ad/XP_RCC.protein.csv.gz;           H5P[rcc]=$D/data/xenium_protein/h5ad
  SAMPLE[ccrcc]=XP_CCRCC; PROT[ccrcc]=$D/data/xenium_protein_ccrcc/h5ad/XP_CCRCC.protein.csv.gz; H5P[ccrcc]=$D/data/xenium_protein_ccrcc/h5ad
  # PanCK does not separate in renal carcinoma 1: partners there are CD45-negative cells only
  for spec in "rcc T CD45-" "rcc T_strict CD45-" "ccrcc T CD45-" "ccrcc T_strict CD45-" "ccrcc T PANCK+" "ccrcc T_strict PANCK+"; do
    s=${spec%% *}; rest=${spec#* }; tc=${rest%% *}; pp=${rest#* }
    lab=""; for l in $R/xp_$s/labels $LR/xp_$s/labels; do [[ -f $l/${SAMPLE[$s]}.labels.csv.gz ]] && { lab=$l; break }; done
    [[ -n $lab ]] || { print "$s: no labels (for positions) in $R/xp_$s/labels or $LR/xp_$s/labels; skipped"; continue }
    [[ -f ${PROT[$s]} && -f ${H5P[$s]}/${SAMPLE[$s]}.h5ad ]] || { print "$s: no protein table or h5ad; skipped"; continue }
    pn=${${pp//-/neg}//+/pos}; o=$M/${s}_protein_${tc/_/}_${(L)pn}
    [[ -e $o ]] && { print "exists, kept: $o"; continue }
    stamp "3 $s: $tc by protein, partners $pp by protein, 500 um tiles"
    python -u $CC --labels $lab --h5ad ${H5P[$s]} --sample ${SAMPLE[$s]} --protein ${PROT[$s]} --t-class $tc \
      --partner-protein $pp --immune T --partner $pp --tile 500 --out $o > $M/logs/${o:t}.log 2>&1 \
      || { tail -5 $M/logs/${o:t}.log; print "  ($s $tc $pp failed; skipped)"; continue }
    show $M/logs/${o:t}.log
  done
fi
stamp "done; send $M/logs/*.log, $O/tail/*/tail_summary.json and $M/*/markers.csv"
