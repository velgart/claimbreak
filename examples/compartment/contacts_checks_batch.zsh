#!/bin/zsh
# contacts_checks_batch.zsh -- checks of the contact readouts, for lung, gut and kidney:
#   (1) `claimbreak compartment contacts` with --mode composition (random relabeling that keeps the type mix) for the
#       noise null (nucA -> nucB) and the depth-matched pair (nucB -> wholeB);
#   (2) contacts_decompose.py for null, depth-matched and analyst (nucleus -> whole) pairs, one run per partner.
# Then a harvest zip in ~/Downloads. Read only on the labels; every output folder is new (existing ones are skipped).
#   zsh contacts_checks_batch.zsh                 (all tissues whose labels folder exists)
#   LUNG=... GUT=... KIDNEY=... zsh contacts_checks_batch.zsh     (override the labels folders)
setopt no_bang_hist
R=~/claimbreak_results; LOG=$R/logs; mkdir -p $LOG
T=~/software/tools
LUNG=${LUNG:-$R/bilous_v4}
if [[ -z ${GUT:-} ]]; then                       # IBD Xenium labels: newest folder under IBD/labels with Xenium labels
  GUT=$(ls -td ~/claimbreak_data/IBD/labels/*(N/) | while read d; do
          ls $d/*Xenium*.labels.csv.gz >/dev/null 2>&1 && { echo $d; break }; done)
fi
KIDNEY=${KIDNEY:-~/claimbreak_data/AIN/labels/v1}
STAMP=$(date +%Y%m%d_%H%M)
OUTS=()

run_tissue() {   # name labels level immune partners(comma) unit_pattern
  local name=$1 lab=$2 lv=$3 imm=$4 parts=$5 up=$6
  [[ -n $lab && -d $lab ]] || { echo "skip $name (no labels folder: '$lab')"; return }
  echo "=== $name $(date +%H:%M) labels $lab (level $lv)"
  for p in "nucA nucB null" "nucB wholeB dm"; do set -- ${=p}
    local o=$R/rv_${name}_contacts_${3}_comp
    OUTS+=$o
    [[ -d $o ]] && { echo "skip $o"; continue }
    claimbreak compartment contacts --labels $lab --out $o --level $lv --immune $imm --partners $parts \
      --ref $1 --test $2 --unit-pattern $up --mode composition 2>&1 | tee $LOG/rv_${name}_contacts_${3}_comp.log
  done
  for p in "nucA nucB null" "nucB wholeB dm" "nucleus whole an"; do set -- ${=p}
    for b in ${(s:,:)parts}; do
      local o=$R/rv_${name}_decomp_${3}_${b}
      OUTS+=$o
      [[ -d $o ]] && { echo "skip $o"; continue }
      python $T/contacts_decompose.py --labels $lab --ref $1 --test $2 --level $lv --immune $imm --partner $b \
        --unit-pattern $up --out $o 2>&1 | tee $LOG/rv_${name}_decomp_${3}_${b}.log
    done
  done
}

run_tissue lung   "$LUNG"   L2m T_NK,Myeloid,B        Tumour            'L\d(?=_L)'
run_tissue gut    "$GUT"    L1  T_NK,Myeloid,B,Plasma Epithelial        'I0\d{3}'
run_tissue kidney "$KIDNEY" L1  T_NK,Myeloid,B        PT,LOH,Distal_CD  '(AIN|ATI|REF)_\d+_D'

# harvest: result tables (csv, json) and the logs of this batch
Z=~/Downloads/cb_contacts_checks_$STAMP.zip
( cd $R && zip -q -r $Z rv_*/*.csv rv_*/*.json logs/rv_*.log ) && echo "harvest: $Z ($(du -h $Z | cut -f1))"
