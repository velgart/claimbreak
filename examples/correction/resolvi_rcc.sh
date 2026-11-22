#!/usr/bin/env bash
# resolvi_rcc.sh -- resolVI's semi-supervised relabeling on the two renal carcinoma donors, scored against protein
# (uncertainty filter and seed as settings, sensitivity runs). Linux or WSL, GPU; ~/claimbreak_work/ihc/resolvi; never touches ~/software/claimbreak.
#   bash resolvi_rcc.sh check      inputs, GPU, scvi-tools in ~/venvs/resolvi (writes nothing)
#   bash resolvi_rcc.sh labels     claimbreak labels of each section with the options of the deposited-mask labels
#                                  (kidney reference, AIN options) -- the same labels as the deposited-mask analysis
#   bash resolvi_rcc.sh run        resolvi_rcc.py: windows, unsupervised model, kNN filter, semi-supervised relabeling
#   bash resolvi_rcc.sh score      protein recall and precision of the resolVI labels against our whole-cell and
#                                  nuclear labels on the same cells (xenium_protein.py truth)
#   bash resolvi_rcc.sh harvest    ~/resolvi_<time>.tgz (tables and logs, no labels or models), copied to Downloads
#   bash resolvi_rcc.sh all        labels, run, score, harvest (each step skips work already done)
#   bash resolvi_rcc.sh sens       run and score each setting in SENS ("0.40:0 0.05:1" = uncertainty:seed), each
#                                  into its own folders (rv_u40_s0, protein_rv_u40_s0, ...), then harvest
# Settings: D (/mnt/e/fibrosis), WINDOWS (3), WINDOW_CELLS (30000), EPOCHS (50), DONORS ("XP_RCC XP_CCRCC"),
#   UNC (0.05, largest uncertainty kept for training), SEED (0). The defaults write rv/, labels_rv/, protein/ and
#   summary.txt as before; any other setting writes rv_u<UNC>_s<SEED>/, labels_<tag>/, protein_<tag>/, summary_<tag>.txt.
#   The windows depend only on the starting labels, so every setting scores the same cells.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)       # before the cd below: the folder this script was started from
D=${D:-/mnt/e/fibrosis}; W=~/claimbreak_work/ihc/resolvi; mkdir -p $W/logs; cd $W
CB=~/claimbreak_work/ihc/claimbreak; CBX=~/claimbreak_work/ihc/venv/bin/claimbreak; CPY=~/claimbreak_work/ihc/venv/bin/python
RPY=~/venvs/resolvi/bin/python
RVPY=${RVPY:-$HERE/resolvi_rcc.py}; [[ -f $RVPY ]] || RVPY=$W/resolvi_rcc.py; [[ -f $RVPY ]] || RVPY=$CB/examples/correction/resolvi_rcc.py
REF=${REF:-$D/AIN/ref/kidney_census.h5ad}; AINLAB=${AINLAB:-$D/AIN/labels/v1}
WINDOWS=${WINDOWS:-3}; WINDOW_CELLS=${WINDOW_CELLS:-30000}; EPOCHS=${EPOCHS:-50}
DONORS=${DONORS:-"XP_RCC XP_CCRCC"}
UNC=${UNC:-0.05}; SEED=${SEED:-0}
set_tag() {   # folder names of one setting
  if [[ $UNC == 0.05 && $SEED == 0 ]]; then TAG=rv; PDIR=protein; SUM=summary.txt
  else TAG=rv_u${UNC#0.}_s$SEED; PDIR=protein_$TAG; SUM=summary_$TAG.txt; fi
}
set_tag
DL=${CB_DOWNLOADS:-/mnt/c/Users/$USER/Downloads}
step=${1:-}
fail() { echo "STOP: $*"; return 1 2>/dev/null || exit 1; }
stamp() { echo; echo "=== $(date '+%H:%M:%S') $*"; }
xdir() { [[ $1 == XP_CCRCC ]] && echo $D/data/xenium_protein_ccrcc || echo $D/data/xenium_protein; }

do_check() {
  stamp "check"
  $RPY -c "import torch, scvi; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| scvi-tools', scvi.__version__)" \
    || echo "  resolVI environment not ready (~/venvs/resolvi)"
  $CPY -c "import claimbreak; print('claimbreak', claimbreak.__version__)"
  local s x
  for s in $DONORS; do x=$(xdir $s)
    for f in $x/h5ad/$s.h5ad $x/h5ad/$s.protein.csv.gz; do [[ -s $f ]] && echo "  ok      $f ($(du -h $f | cut -f1))" || echo "  MISSING $f"; done
  done
  for f in $REF $AINLAB; do [[ -e $f ]] && echo "  ok      $f" || echo "  MISSING $f"; done
  ls $AINLAB/*.meta.json > /dev/null 2>&1 || echo "  MISSING $AINLAB/*.meta.json (label options)"
  [[ -f $RVPY ]] && echo "  ok      $RVPY" || echo "  MISSING $RVPY"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null
}

do_labels() {
  local opts s x
  opts=$($CPY - $AINLAB $W/exclude_genes.txt <<'EOF'
import json, sys, glob, os
lab, out = sys.argv[1], sys.argv[2]
m = json.load(open(sorted(glob.glob(os.path.join(lab, "*.meta.json")))[0]))
o = []
lay = m.get("ref_layer", "X")
if lay == "raw": o += ["--ref-raw"]
elif lay not in ("X", "", None): o += ["--ref-layer", lay]
for k, f in (("ref_map", "--ref-map"), ("ref_subset", "--ref-subset")):
    if m.get(k): o += [f, m[k]]
o += ["--min-counts", str(m.get("min_counts", 10)), "--seed", str(m.get("seed", 0)), "--floor", str(m.get("floor", 0.01)),
      "--gamma-iter", str(m.get("gamma_iter", 5))]
if m.get("markers", 0): o += ["--markers", str(m["markers"]), "--marker-min", str(m.get("marker_min", 1e-4)),
                             "--marker-min-log2", str(m.get("marker_min_log2", 1.0))]
if m.get("exclude_genes"):
    open(out, "w").write("\n".join(m["exclude_genes"]) + "\n"); o += ["--exclude-genes", out]
print(" ".join(o))
EOF
) || { fail "no label options in $AINLAB"; return 1; }
  for s in $DONORS; do x=$(xdir $s)
    [[ -f $W/$s/labels/$s.labels.csv.gz ]] && { echo "exists: $W/$s/labels"; continue; }
    stamp "label $s with $opts"
    $CBX compartment label --ref $REF --ref-levels L1=coarse --levels L1 --ref-merge '' --inputs $x/h5ad/$s.h5ad \
      --out $W/$s/labels --arms nucA,nucB,wholeB,cytoB,nucleus,whole $opts > logs/label_$s.log 2>&1 \
      || { tail -8 logs/label_$s.log; fail "label $s"; return 1; }
    grep -E "L1:|WARNING" logs/label_$s.log | cut -c1-200 | head -8
  done
}

do_run() {
  local s x
  for s in $DONORS; do x=$(xdir $s)
    [[ -f $W/$s/$TAG/$s.resolvi.csv.gz ]] && { echo "exists: $W/$s/$TAG"; continue; }
    [[ -f $W/$s/labels/$s.labels.csv.gz ]] || { fail "no labels for $s (bash resolvi_rcc.sh labels)"; return 1; }
    [[ -d $W/$s/$TAG ]] && mv $W/$s/$TAG $W/$s/${TAG}_failed_$(date +%H%M)
    stamp "resolVI $s ($TAG): $WINDOWS windows of ~$WINDOW_CELLS cells, $EPOCHS epochs, uncertainty <= $UNC, seed $SEED"
    $RPY -u $RVPY --h5ad $x/h5ad/$s.h5ad --labels $W/$s/labels/$s.labels.csv.gz \
      --out $W/$s/$TAG --windows $WINDOWS --window-cells $WINDOW_CELLS --epochs $EPOCHS \
      --max-uncertainty $UNC --seed $SEED > logs/resolvi_${s}_$TAG.log 2>&1 \
      || { grep -v "it/s\]" logs/resolvi_${s}_$TAG.log | tail -15; fail "resolVI $s $TAG"; return 1; }
    grep -E "^[A-Z]|^  w|^wrote" logs/resolvi_${s}_$TAG.log
  done
}

do_score() {
  local s x
  for s in $DONORS; do x=$(xdir $s)
    [[ -f $W/$s/$TAG/$s.resolvi.csv.gz ]] || { fail "no resolVI labels for $s ($TAG)"; return 1; }
    [[ -d $W/$s/$PDIR ]] && { echo "exists: $W/$s/$PDIR"; continue; }
    mkdir -p $W/$s/labels_$TAG
    $CPY - $W/$s/labels/$s.labels.csv.gz $W/$s/$TAG/$s.resolvi.csv.gz $W/$s/labels_$TAG/$s.labels.csv.gz <<'EOF'
import sys, pandas as pd
L = pd.read_csv(sys.argv[1], keep_default_na=False, low_memory=False, dtype={"cell_id": str})
R = pd.read_csv(sys.argv[2], keep_default_na=False, dtype={"cell_id": str})
M = L.merge(R[["cell_id", "window", "L1_rvSemi", "L1_rvTrain", "rv_certainty"]], on="cell_id", how="inner")
M.to_csv(sys.argv[3], index=False, compression="gzip")
print(f"{len(M):,} cells in the resolVI windows (of {len(L):,}); scored on these cells for every arm")
EOF
    stamp "protein truth $s ($TAG)"
    $CPY -u $CB/examples/compartment/xenium_protein.py truth --labels $W/$s/labels_$TAG --protein $x/h5ad/$s.protein.csv.gz \
      --sample $s --out $W/$s/$PDIR --level L1 --arms nucleus,whole,nucB,wholeB,rvSemi \
      --pairs nucleus:whole,whole:rvSemi,nucleus:rvSemi,nucB:wholeB > logs/truth_${s}_$TAG.log 2>&1 \
      || { tail -8 logs/truth_${s}_$TAG.log; fail "truth $s $TAG"; return 1; }
    $CPY - $W/$s/$PDIR <<'EOF' | tee $W/$s/$SUM
import os, sys, pandas as pd
P = sys.argv[1]
R = pd.read_csv(os.path.join(P, "recall.csv")); R = R[(R.variant == "all") & (R.stratum == "all")]
print("recall of protein classes by each arm (same cells):")
print(R.pivot_table(index="protein_class", columns="arm", values="recall").round(3).to_string())
Pr = pd.read_csv(os.path.join(P, "precision.csv"))
cols = [c for c in ("label", "arm", "precision") if c in Pr.columns]
if {"arm", "precision"} <= set(Pr.columns):
    idx = "label" if "label" in Pr.columns else Pr.columns[0]
    print("\nprecision:"); print(Pr.pivot_table(index=idx, columns="arm", values="precision").round(3).to_string())
EOF
  done
}

do_harvest() {
  local f=~/resolvi_$(date +%Y%m%d_%H%M).tgz
  tar czf $f -C $W logs $(cd $W && ls -d */rv*/settings.json */protein* */summary*.txt 2>/dev/null) 2>/dev/null
  cp $f $DL/ 2>/dev/null && echo "copied to $DL"; ls -lh $f
}

case $step in
  check) do_check ;; labels) do_labels ;; run) do_run ;; score) do_score ;; harvest) do_harvest ;;
  all) do_labels && do_run && do_score && do_harvest ;;
  sens) SENS=${SENS:-0.40:0 0.05:1}   # unquoted default: split into settings below
        for c in $SENS; do UNC=${c%:*}; SEED=${c#*:}; set_tag
          do_run && do_score || exit 1; done; do_harvest ;;
  *) sed -n '2,19p' $0 ;;
esac
