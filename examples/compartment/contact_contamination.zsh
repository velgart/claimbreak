#!/bin/zsh
# contact_contamination.zsh -- the partner's RNA in T/NK cells by contact, from
# whole-cell and nucleus counts of the same cells, in the tissues of region_check.zsh (gut, lung, synovium, kidney),
# on the scale of the Slide-GoTags isolated-nuclei control (slidetags_tcr.py contamination).
#
#   caffeinate -ims zsh $CB/examples/compartment/contact_contamination.zsh 2>&1 | tee ~/claimbreak_results/logs/contact_contamination_$(date +%Y%m%d_%H%M).log
#
# Positional check: JITTER="0,5,10,20" (um RMS) and DROP=0.4 (share of cells removed) write to
# $R/contact_contamination/<tissue>_jitter[_drop<DROP>]:
#   TISSUES=lung JITTER=0,5,10,20 caffeinate -ims zsh $CB/examples/compartment/contact_contamination.zsh 2>&1 | tee ...
#
# Env: D, LR, R, RA_LAB, TISSUES (gut lung ra kidney). Output $R/contact_contamination/<tissue>. Refuses to overwrite.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; LR=${LR:-~/claimbreak_results}; R=${R:-~/claimbreak_results}
RA_LAB=${RA_LAB:-$D/data/RA_labels/gseh4_pre/all}; TISSUES=(${=TISSUES:-gut lung ra kidney})
JITTER=${JITTER:-}; DROP=${DROP:-0}
typeset -a XA; SUF=""
if [[ -n $JITTER || $DROP != 0 ]]; then
  XA=(--jitter ${JITTER:-0} --drop-frac $DROP); SUF=_jitter; [[ $DROP != 0 ]] && SUF=${SUF}_drop$DROP
fi
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
python -c "import claimbreak, sys; v = tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(v < (1, 7, 103))" \
  || stop "claimbreak 1.7.103 or later needed"
[[ -d $D ]] || stop "data drive not mounted ($D)"
CC=$CB/examples/compartment/contact_contamination.py
O=$R/contact_contamination; mkdir -p $O/logs

typeset -A LAB LEVEL UNIT H5 PART
LAB[gut]=$D/IBD/labels/xenium_v1;  LEVEL[gut]=L1;    UNIT[gut]='I0\d{3}';               H5[gut]=$D/IBD/h5ad;      PART[gut]=Epithelial
LAB[lung]=$LR/cytoB_lung/merged;   LEVEL[lung]=L2m;  UNIT[lung]='L\d(?=_L)';            H5[lung]=$D/Bilous_h5ad;  PART[lung]=Tumour
LAB[kidney]=$D/AIN/labels/v1;      LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'; H5[kidney]=$D/AIN/h5ad;   PART[kidney]=PT
LAB[ra]=$RA_LAB;                   LEVEL[ra]=L1;     UNIT[ra]='p\d{2}';                 H5[ra]=$D/data/RA_h5ad/compartment_pre; PART[ra]=Fibroblast

for t in $TISSUES; do
  o=$O/$t$SUF
  [[ -e $o ]] && { print "exists, kept: $o"; continue }
  stamp "$t: T_NK cells, ${PART[$t]}-marker load at contact, whole cell vs nucleus"
  python -u $CC --labels ${LAB[$t]} --h5ad ${H5[$t]} --level ${LEVEL[$t]} --immune T_NK --partner ${PART[$t]} \
    --unit-pattern "${UNIT[$t]}" $XA --out $o > $O/logs/$t$SUF.log 2>&1 \
    || { tail -5 $O/logs/$t$SUF.log; print "  ($t failed; skipped)"; continue }
  if [[ -n $SUF ]]; then
    grep -E '^#####|contact / away|^ +(whole|nucleus)|^ *[0-9.]+ +[0-9.]' $O/logs/$t$SUF.log
  else
    sed -n '/per 1,000 counts by distance/,$p' $O/logs/$t.log
  fi
done
stamp "done; send $O/logs/*.log and $O/*/markers.csv"
