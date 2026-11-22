#!/bin/zsh
# vendor_calls.zsh -- the analyses a
# typical user runs (squidpy-style neighborhood enrichment, contacts per 100 partner cells, infiltration, the Keren
# mixing class), per 800 um field, from whole-cell, nuclear, hybrid and (Xenium Protein) protein labels of the same
# cells, on the vendor datasets: 10x renal carcinomas 1 and 2 (Xenium Protein), Vizgen uterine cancer (MERSCOPE,
# CD45RO), NanoString lung cancer (CosMx). vendor_calls.py on existing labels; nothing is relabeled.
#
#   zsh $CB/examples/compartment/vendor_calls.zsh check        label vocabulary and class sizes per dataset (seconds)
#   caffeinate -ims zsh $CB/examples/compartment/vendor_calls.zsh run 2>&1 | tee ~/claimbreak_results/logs/vendor_calls_$(date +%Y%m%d_%H%M).log
#
# Env: D, R (~/claimbreak_results), LR (~/claimbreak_results; where labels are looked for when R has none), SETS (rcc ccrcc uterine
# cosmx), MAXF (fields per section, default 40: a pilot), NPERM (200), FOV (800), PART_<set> to override a partner
# regex. Writes $R/vendor_calls/<set>; refuses to overwrite.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}; LR=${LR:-~/claimbreak_results}
SETS=(${=SETS:-rcc ccrcc uterine cosmx}); MAXF=${MAXF:-40}; NPERM=${NPERM:-200}; FOV=${FOV:-800}
MODE=${1:-check}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
python -c "import claimbreak, sys; v = tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(v < (1, 7, 103))" \
  || stop "claimbreak 1.7.103 or later needed"
VC=$CB/examples/compartment/vendor_calls.py

typeset -A SUB LEVEL PART PROT PPART
SUB[rcc]=xp_rcc;              LEVEL[rcc]=L1;     PART[rcc]='(?i)^(pt|tumou?r|epithel\w*|renal\w*|cancer\w*)$'
PROT[rcc]=$D/data/xenium_protein/h5ad/XP_RCC.protein.csv.gz;           PPART[rcc]=''
SUB[ccrcc]=xp_ccrcc;          LEVEL[ccrcc]=L1;   PART[ccrcc]=${PART[rcc]}
PROT[ccrcc]=$D/data/xenium_protein_ccrcc/h5ad/XP_CCRCC.protein.csv.gz; PPART[ccrcc]='PANCK+'
SUB[uterine]=ms_uterine_ro;   LEVEL[uterine]=L1; PART[uterine]='(?i)(tumou?r|epithel\w*|cancer\w*|malignant|carcinoma)'
SUB[cosmx]=cosmx_nsclc_x171/multinomial; LEVEL[cosmx]=L2m; PART[cosmx]='(?i)^tumou?r$'
for s in $SETS; do
  v=PART_$s; [[ -n ${(P)v} ]] && PART[$s]=${(P)v}
done

labdir() { local l; for l in $R/${SUB[$1]}/labels $LR/${SUB[$1]}/labels; do [[ -d $l ]] && { print $l; return } done }

for s in $SETS; do
  [[ -n ${SUB[$s]} ]] || stop "unknown set $s (rcc ccrcc uterine cosmx)"
  L=$(labdir $s)
  [[ -n $L ]] || { print "$s: no labels in $R/${SUB[$s]}/labels or $LR/${SUB[$s]}/labels; skipped"; continue }
  args=(--labels $L --level ${LEVEL[$s]} --partner "${PART[$s]}")
  if [[ -n ${PROT[$s]} ]]; then
    [[ -f ${PROT[$s]} ]] && args+=(--protein ${PROT[$s]}) || print "$s: no protein table ${PROT[$s]}; RNA arms only"
    [[ -n ${PPART[$s]} && -f ${PROT[$s]} ]] && args+=(--protein-partner ${PPART[$s]})
  fi
  if [[ $MODE == check ]]; then
    stamp "check $s: $L (level ${LEVEL[$s]}, partner ${PART[$s]})"
    python -u $VC $args --check
  elif [[ $MODE == run ]]; then
    o=$R/vendor_calls/$s
    [[ -e $o ]] && { print "exists, kept: $o"; continue }
    stamp "run $s: $L, at most $MAXF fields of $FOV um per section, $NPERM permutations"
    python -u $VC $args --out $o --max-fovs $MAXF --nperm $NPERM --fov $FOV || print "  ($s failed)"
  else
    stop "mode: check or run"
  fi
done
[[ $MODE == run ]] && stamp "done; send $R/vendor_calls/*/flips.csv, medians.csv and the log"
