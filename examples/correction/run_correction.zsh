#!/bin/zsh
# run_correction.zsh -- the correction test on the
# GSE311609 Lung-panel sections, end to end. Each step skips sections already done (the scripts never overwrite).
#   CORES=6 zsh run_correction.zsh            (all steps)
#   zsh run_correction.zsh split              (one step: export | split | admix | import | contacts | readouts)
setopt no_bang_hist
set -e
F=${CB_F:-~/claimbreak_data}
C=${CB_C:-$F/Bilous_corr}                                   # inputs and tool outputs (large) on the drive
R=${CB_R:-~/claimbreak_results}                                      # labels and readouts (small)
LOG=$R/logs; mkdir -p $LOG $C
HERE=${0:A:h}                                      # this folder (examples/correction of the claimbreak repo)
REF=${CB_REF:-$F/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad}
RAW=${CB_RAW:-$F/Bilous/raw}
H5=${CB_H5:-$F/Bilous_h5ad}
LAB=${CB_LAB:-$R/bilous_v4}
OUT=${CB_OUT:-$R/bilous_corr_v1}
CORES=${CORES:-6}
S=(GSM9509145_NSCLC_lung_L1_L1 GSM9509147_NSCLC_lung_L2_L2_1 GSM9509148_NSCLC_lung_L2_L2_2 GSM9509149_NSCLC_lung_L3_L3
   GSM9509150_NSCLC_lung_L4_L4)
[[ -n ${CB_SECTIONS:-} ]] && S=(${(s:,:)CB_SECTIONS})     # paths and sections can be overridden (CB_* variables)
MAXC=${CB_MAX_CELLS:-30000}
ARMS=(whole rerun split splitsp shiftc shift rctd admix)
step=${1:-all}
for p in $REF $RAW $H5 $LAB; do [[ -e $p ]] || { echo "missing $p"; exit 1; }; done

if [[ $step == all || $step == export ]]; then
  todo=(); for s in $S; do [[ -d $C/input/$s ]] || todo+=$s; done
  if (( $#todo )); then
    python -u $HERE/corr_export.py --h5ad-dir $H5 --labels $LAB --raw $RAW --ref $REF --samples ${(j:,:)todo} \
      --out $C/input --tmp $F/Bilous_tmp --max-cells $MAXC 2>&1 | tee -a $LOG/corr_export.log
  fi
fi
if [[ $step == all || $step == split ]]; then
  for s in $S; do
    [[ -f $C/split/$s/versions.txt ]] && continue
    [[ -d $C/split/$s ]] && { echo "incomplete $C/split/$s (an interrupted run): remove it and rerun"; exit 1; }
    Rscript $HERE/run_split.R $C/input/$s $C/input/ref $C/split/$s $CORES 2>&1 | tee $LOG/corr_split_$s.log
  done
fi
if [[ $step == all || $step == admix ]]; then
  mkdir -p $C/work                                 # NMF writes temporary folders into the working directory
  for s in $S; do
    [[ -f $C/admix/$s/admix_counts.mtx ]] && continue        # an interrupted run resumes after its CRF step
    ( cd $C/work && Rscript $HERE/run_celladmix.R $C/input/$s $C/admix/$s $CORES 12 ) 2>&1 | tee $LOG/corr_admix_$s.log
  done
fi
if [[ $step == all || $step == import ]]; then
  [[ -d $OUT ]] || python -u $HERE/corr_import.py --input $C/input --split $C/split --admix $C/admix --labels $LAB \
      --ref $REF --out $OUT 2>&1 | tee $LOG/corr_import.log
fi
if [[ $step == all || $step == contacts ]]; then
  for a in $ARMS; do
    [[ -d ${OUT}_contacts_$a ]] && continue
    claimbreak compartment contacts --labels $OUT --out ${OUT}_contacts_$a --level L2m --immune T_NK,Myeloid,B \
      --partners Tumour --ref nucleus --test $a --unit-pattern 'L\d(?=_L)' 2>&1 | tee $LOG/corr_contacts_$a.log
  done
fi
if [[ $step == all || $step == readouts ]]; then
  CT=(); for a in $ARMS; do CT+="$a=${OUT}_contacts_$a"; done
  [[ -d ${OUT}_readouts ]] || python -u $HERE/corr_readouts.py --labels $OUT --contacts ${(j:,:)CT} \
      --out ${OUT}_readouts 2>&1 | tee $LOG/corr_readouts.log
fi
