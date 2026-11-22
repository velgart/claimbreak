#!/usr/bin/env bash
# ihc_wsl.sh -- lung IHC ground truth on a Linux or WSL workstation.
# Never touches an existing claimbreak clone or venv: setup clones claimbreak (CB_REPO) into ~/claimbreak_work/ihc/claimbreak
# and makes its own venv ~/claimbreak_work/ihc/venv.
#   bash ihc_wsl.sh setup     copy inputs to the Linux disk, clone claimbreak, make the venv, check the environment
#   bash ihc_wsl.sh A         nuclei (3 components in parallel), then registrations (3 in parallel)
#   bash ihc_wsl.sh A2        registration diagnostics (match shares against random shifts, tiles, scale)
#   bash ihc_wsl.sh B         truth on the aligned registrations: matches within 2 and 1 um, with and without the
#                             autofluorescence correction (4 runs in parallel); harvest ~/ihc_results_<time>.tgz
#   bash ihc_wsl.sh C         truth again at 2 um, with and without AF, adding fate.csv (what protein CD8 and
#                             panCK cells are labeled), label_protein.csv and the CD8iso class; harvest ~/ihc_fate_<time>.tgz
#   bash ihc_wsl.sh channels  which protein each deposited Opal channel carries, read against the Xenium
#                             transcripts of the same cells (ihc_channel_check.py; the authors' panel lists CD8 on
#                             Opal 780, absent from the file); H5 = folder of the section .h5ad files (default the
#                             drive, /mnt/e/fibrosis/Bilous_h5ad); harvest ~/ihc_channels_<time>.tgz
# Settings by environment variables: SRC (where the transferred files are), CB_REPO (claimbreak repository URL),
# REG_OPTS (default --fix-scale; REG_OPTS= runs the default search for a section failing R0),
# BASEPY (python used to make the venv; default python3).
set -uo pipefail
SRC=${SRC:-/mnt/e/transfer}
CB_REPO=${CB_REPO:-https://github.com/velgart/claimbreak}
W=~/claimbreak_work/ihc; mkdir -p $W/logs; cd $W
CB=$W/claimbreak
PY=$W/venv/bin/python
IMG=$W/GSE311609_ihc_NSCLC_lung_L3_L3_NSCLC_lung_L6_L6_NSCLC_lung_L1_L1.ome.tif
RAW=${RAW:-~/claimbreak_data/Bilous/raw}
T=$CB/examples/correction/ihc_truth.py
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
step=${1:-}

fail() { echo "STOP: $*"; exit 1; }
check_jobs() {   # wait for every background job; stop if any failed, and show the end of its log
  local bad=0
  for j in "$@"; do wait ${j%%:*} || { echo "FAILED ${j#*:}"; tail -5 logs/${j#*:}.log; bad=1; }; done
  (( bad == 0 )) || fail "a job failed (logs above)"
}

if [[ $step == setup ]]; then
  # inputs of earlier runs are reused when present
  if [[ ! -d $W/bilous_v4 ]]; then
    [[ -f $SRC/ihc_for_wsl.tgz ]] || fail "missing $SRC/ihc_for_wsl.tgz"
    cp $SRC/ihc_for_wsl.tgz . && tar xzf ihc_for_wsl.tgz
  fi
  if [[ ! -f $IMG ]]; then
    [[ -f $SRC/$(basename $IMG).gz ]] || fail "missing $SRC/$(basename $IMG).gz"
    cp $SRC/$(basename $IMG).gz . && gunzip $(basename $IMG).gz
  fi
  if [[ -d $CB/.git ]]; then
    git -C $CB pull -q --ff-only || fail "updating $CB failed"
  else
    git clone -q $CB_REPO $CB || fail "clone of $CB_REPO failed"
  fi
  echo "repo: $(git -C $CB log --oneline -1)  $(git -C $CB describe --tags)"
  [[ -x $PY ]] || ${BASEPY:-python3} -m venv $W/venv || fail "venv (try BASEPY=/usr/bin/python3.12)"
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q -e $CB scikit-image zarr scikit-learn pillow tifffile imagecodecs pyarrow || fail "pip"
  $PY -c "import claimbreak, tifffile, zarr, skimage, sklearn; print('python ok', claimbreak.__version__, claimbreak.__file__)"
  $PY - "$IMG" <<'EOF2'
import sys, tifffile, re
with tifffile.TiffFile(sys.argv[1]) as t:
    s = t.series[0]; x = t.ome_metadata or ""
    print("image", s.shape, s.dtype, "levels", len(s.levels), "um/px", re.findall(r'PhysicalSizeX="([^"]+)"', x)[:1])
EOF2
  ls $RAW/GSM9509145_NSCLC_lung_L1_L1_nucleus_boundaries.parquet* $RAW/GSM9509149_NSCLC_lung_L3_L3_nucleus_boundaries.parquet* \
    || fail "nucleus boundaries missing in $RAW"
  echo "setup done; next: bash ihc_wsl.sh A"
fi

if [[ $step == A ]]; then
  jobs_=()
  for c in 1 5 22; do
    [[ -f nuc_c$c/nuclei.json ]] && continue
    $PY -u $T nuclei --image $IMG --components $W/ihc_lung_overview/components.csv --component $c --out $W/nuc_c$c \
      > logs/nuclei_c$c.log 2>&1 & jobs_+=("$!:nuclei_c$c")
  done
  (( ${#jobs_[@]} )) && check_jobs "${jobs_[@]}"
  grep -h '"n"' logs/nuclei_c*.log | cut -c1-200
  jobs_=()
  for pair in "c5 GSM9509149_NSCLC_lung_L3_L3 L3" "c1 GSM9509145_NSCLC_lung_L1_L1 L1" "c22 GSM9509145_NSCLC_lung_L1_L1 L1"; do
    set -- $pair
    [[ -f reg_$3_$1/register.json ]] && continue
    $PY -u $T register --nuclei $W/nuc_$1 --raw $RAW --sample $2 --out $W/reg_$3_$1 ${REG_OPTS---fix-scale} \
      > logs/reg_$3_$1.log 2>&1 & jobs_+=("$!:reg_$3_$1")
  done
  (( ${#jobs_[@]} )) && check_jobs "${jobs_[@]}"
  for f in reg_*/register.json; do
    $PY -c "import json,sys; d=json.load(open('$f')); print('$f', {k: d.get(k) for k in ('R0','matched_share_small','excess_share','median_residual_um','um_per_px')})"
  done
  echo "stage A done. Expected: reg_L3_c5 valid; exactly one of reg_L1_c1 / reg_L1_c22 valid."
  tar czf ~/ihc_stageA_$(date +%Y%m%d_%H%M).tgz logs reg_*/register.json nuc_*/nuclei.json && ls -lh ~/ihc_stageA_*.tgz | tail -1
fi

if [[ $step == A2 ]]; then
  [[ -d reg_diag ]] && fail "reg_diag exists; remove it to rerun"
  R=$(ls -d reg_*/ | grep -v reg_diag | sed 's#/$##' | sed "s#^#$W/#" | paste -sd,)
  $PY -u $CB/examples/correction/reg_diag.py --reg $R --raw $RAW --out $W/reg_diag > logs/reg_diag.log 2>&1 \
    || { tail -20 logs/reg_diag.log; fail "reg_diag failed"; }
  cat logs/reg_diag.log; echo; cat reg_diag/r0b.csv
  Z=~/ihc_regdiag_$(date +%Y%m%d_%H%M).tgz
  tar czf $Z logs reg_diag reg_*/register.json nuc_*/nuclei.json && ls -lh $Z
fi

if [[ $step == B ]]; then
  git -C $CB log --oneline -1
  [[ -f reg_diag/r0b.csv ]] || fail "run stage A2 first (registration diagnostics)"
  # registrations that are aligned (match share at 2 um far above random shifts: R0b_match); the wrong section fails it
  M=$($PY -c "import pandas as pd; R=pd.read_csv('reg_diag/r0b.csv'); print(','.join('$W/'+r for r in R.loc[R.R0b_match,'registration']))")
  [[ -n $M ]] || fail "no aligned registration"
  echo "registrations used: $M"
  L=$W/bilous_v4,$W/bilous_corr_v1
  jobs_=()
  for d in 2 1; do
    for af in "" "--no-af"; do
      o=truth_d${d}${af:+_noaf}
      [[ -d $o ]] && continue
      $PY -u $T truth --matches $M --labels $L --out $W/$o --max-match-um $d $af > logs/$o.log 2>&1 & jobs_+=("$!:$o")
    done
  done
  (( ${#jobs_[@]} )) && check_jobs "${jobs_[@]}"
  Z=~/ihc_results_$(date +%Y%m%d_%H%M).tgz
  tar czf $Z logs truth_d* reg_diag reg_*/register.json nuc_*/nuclei.json && ls -lh $Z
  echo "copy $Z to /mnt/c/Users/$USER/Downloads"
fi

if [[ $step == C ]]; then
  git -C $CB log --oneline -1
  grep -q "fate.csv" $T || fail "this clone is out of date; run setup"
  M=$($PY -c "import pandas as pd; R=pd.read_csv('reg_diag/r0b.csv'); print(','.join('$W/'+r for r in R.loc[R.R0b_match,'registration']))")
  [[ -n $M ]] || fail "no aligned registration"
  L=$W/bilous_v4,$W/bilous_corr_v1
  jobs_=()
  for af in "" "--no-af"; do
    o=truth_c_d2${af:+_noaf}
    [[ -d $o ]] && continue
    $PY -u $T truth --matches $M --labels $L --out $W/$o --max-match-um 2 $af > logs/$o.log 2>&1 & jobs_+=("$!:$o")
  done
  (( ${#jobs_[@]} )) && check_jobs "${jobs_[@]}"
  sed -n '/labels of protein CD8any/,$p' logs/truth_c_d2.log | head -80
  Z=~/ihc_fate_$(date +%Y%m%d_%H%M).tgz
  tar czf $Z logs/truth_c_* truth_c_d2 truth_c_d2_noaf && ls -lh $Z
  echo "copy $Z to /mnt/c/Users/$USER/Downloads"
fi
if [[ $step == channels ]]; then
  git -C $CB log --oneline -1
  [[ -f $CB/examples/correction/ihc_channel_check.py ]] || fail "this clone is out of date; run setup"
  H5=${H5:-/mnt/e/fibrosis/Bilous_h5ad}
  ls $H5/GSM9509145_NSCLC_lung_L1_L1*.h5ad $H5/GSM9509149_NSCLC_lung_L3_L3*.h5ad || fail "section h5ad files not in $H5 (mount the drive, or set H5=)"
  M=$($PY -c "import pandas as pd; R=pd.read_csv('reg_diag/r0b.csv'); print(','.join('$W/'+r for r in R.loc[R.R0b_match,'registration']))")
  [[ -n $M ]] || fail "no aligned registration"
  echo "registrations used: $M"
  for af in "" "--no-af"; do
    o=channel_check${af:+_noaf}
    [[ -d $o ]] && fail "$o exists; move it away to rerun"
    io=(); [[ -z $af ]] && io=(--image $IMG)
    $PY -u $CB/examples/correction/ihc_channel_check.py --matches $M --h5ad $H5 --out $W/$o --max-match-um 2 $af \
      "${io[@]}" > logs/$o.log 2>&1 || { tail -20 logs/$o.log; fail "$o failed"; }
  done
  cat logs/channel_check.log
  Z=~/ihc_channels_$(date +%Y%m%d_%H%M).tgz
  tar czf $Z logs/channel_check*.log channel_check channel_check_noaf && ls -lh $Z
  echo "copy $Z to /mnt/c/Users/$USER/Downloads"
fi
[[ -n $step ]] || echo "usage: bash ihc_wsl.sh setup|A|A2|B|C|channels"
