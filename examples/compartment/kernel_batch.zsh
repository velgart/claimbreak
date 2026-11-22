#!/bin/zsh
# kernel_batch.zsh -- error kernel per dataset.
# Reads the existing labels, h5ad and polygon files only.
#   zsh $CB/examples/compartment/kernel_batch.zsh check
#   caffeinate -ims zsh $CB/examples/compartment/kernel_batch.zsh all 2>&1 | tee ~/claimbreak_results/logs/kernel_$(date +%Y%m%d_%H%M).log
#   zsh $CB/examples/compartment/kernel_batch.zsh expr lung gut      # one stage, chosen tissues
#   zsh $CB/examples/compartment/kernel_batch.zsh harvest            # ~/Downloads/cb_kernel_<time>.zip (small files)
# Stages per tissue:
#   expr     claimbreak compartment kernel: spillover per contact by gene, producer and receiver; nuclear share of
#            borrowed RNA (calibration pairs: receivers that do not make the gene); R matrix. Cells typed by the
#            full-nucleus labels; neighbors = 6 nearest within 30 um (the contact used throughout).
#            Sections are pooled per unit and panel (lung and gut image one donor on several panels).
#   labels   claimbreak compartment exposure (focal nucA, test wholeB, null nucB; polygons where the tissue has
#            them; no regional destinations): destinations carry kept counts by the share of each partner
#   summary  claimbreak compartment kernel-summary: pooled expression kernel and pair-resolved label kernel (K, q)
#            (with 95% intervals of nu_b and R_b, genes resampled; BOOT draws, default 1000)
#   audit    claimbreak compartment audit --recompute: the label audit on labels written without it,
#            from their stored labels, runner-ups and gamma (no relabeling); writes $KR/<tissue>/audit. reproduce.csv
#            checks that the stored margins are recomputed (label_is_max ~1); REF_<TISSUE> overrides the reference
#            path recorded in meta.json (e.g. labels made on WSL). Not part of `all`: kernel_batch.zsh audit [tissues]
# Tissues (default all = lung gut kidney cosmx rcc ccrcc rpgn); ra (synovium) runs only when named.
# Outputs: $KR/<tissue>/{expr,labels,summary} on the data drive (spill_pairs and r_matrix are large); the harvest
# carries the summaries, the calibration rows and the logs only.
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
# RA_LAB = synovium labels folder (default gseh2_pre); with R=<new root> the synovium results of an
#   audited relabeling (e.g. RA_LAB=$D/data/RA_labels/gseh4_pre/all R=~/claimbreak_results) go to a new folder.
KR=${KR:-~/claimbreak_work/results/kernel}
K=${K:-6}; KRAD=${KRAD:-30}
BOOT=${BOOT:-1000}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

# tissue: labels | h5ad folder | level | unit pattern | polygons (as exposure_batch.zsh)
typeset -A LAB H5 LEVEL UNIT BND
LAB[lung]=${LUNG_LABELS:-$R/bilous_v4};            H5[lung]=$D/Bilous_h5ad;                  LEVEL[lung]=L2m;   UNIT[lung]='L\d(?=_L)'
LAB[gut]=$D/IBD/labels/xenium_v1;                  H5[gut]=$D/IBD/h5ad;                      LEVEL[gut]=L1;     UNIT[gut]='I0\d{3}'
LAB[kidney]=$D/AIN/labels/v1;                      H5[kidney]=$D/AIN/h5ad;                   LEVEL[kidney]=L1;  UNIT[kidney]='(AIN|ATI|REF)_\d+_D'
LAB[cosmx]=$R/cosmx_nsclc_x171/multinomial/labels; H5[cosmx]=$D/data/cosmx_nsclc/h5ad;       LEVEL[cosmx]=L2m;  UNIT[cosmx]='Lung\d+'
LAB[rcc]=$R/xp_rcc/labels;                         H5[rcc]=$D/data/xenium_protein/h5ad;      LEVEL[rcc]=L1;     UNIT[rcc]='XP_RCC'
LAB[ccrcc]=$R/xp_ccrcc/labels;                     H5[ccrcc]=$D/data/xenium_protein_ccrcc/h5ad; LEVEL[ccrcc]=L1; UNIT[ccrcc]='XP_CCRCC'
LAB[rpgn]=${RPGN_W:-$R/rpgn}/labels;               H5[rpgn]=$D/data/rpgn_sultana/h5ad;       LEVEL[rpgn]=L1;    UNIT[rpgn]='RPGN_[0-9]+'
LAB[ra]=${RA_LAB:-$D/data/RA_labels/gseh2_pre/all};           H5[ra]=$D/data/RA_h5ad/compartment_pre;   LEVEL[ra]=L1;      UNIT[ra]='p\d{2}'
BND[lung]=${BND_LUNG:-$D/Bilous/raw}; BND[gut]=${BND_GUT:-$D/IBD/raw}; BND[kidney]=${BND_KIDNEY:-$D/AIN/boundaries}
BND[rcc]=$D/data/xenium_protein/links; BND[ccrcc]=$D/data/xenium_protein_ccrcc/links; BND[rpgn]=$D/data/rpgn_sultana/links
typeset -A REF
REF[lung]=${REF_LUNG:-}; REF[gut]=${REF_GUT:-}; REF[kidney]=${REF_KIDNEY:-}; REF[cosmx]=${REF_COSMX:-}
REF[rcc]=${REF_RCC:-}; REF[ccrcc]=${REF_CCRCC:-}; REF[rpgn]=${REF_RPGN:-}; REF[ra]=${REF_RA:-}
ALL=(lung gut kidney cosmx rcc ccrcc rpgn)

known() { [[ -n ${LEVEL[$1]} ]] || stop "unknown tissue $1 (${(k)LEVEL})" }

check() {
  python -c "import claimbreak, sys; print('claimbreak', claimbreak.__version__); sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
    || stop "claimbreak 1.7.103 or later needed"
  [[ -d ${KR:h} ]] || mkdir -p $KR || stop "cannot make $KR (is the data drive mounted?)"
  print "outputs: $KR ($(df -h ${KR:h} | tail -1 | awk '{print $4}') free)"
  local t nl nh
  for t in ${@:-$ALL}; do
    known $t
    nl=$(ls ${LAB[$t]}/*.labels.csv.gz 2>/dev/null | wc -l | tr -d ' '); nh=$(ls ${H5[$t]}/*.h5ad 2>/dev/null | wc -l | tr -d ' ')
    print "$t: labels $nl in ${LAB[$t]}; h5ad $nh in ${H5[$t]}; polygons ${BND[$t]:-none}$([[ -n ${BND[$t]} && ! -d ${BND[$t]} ]] && print ' (MISSING)')"
    [[ $nl -gt 0 && $nh -gt 0 ]] || print "  WARNING: $t has no labels or no h5ad; it will stop there"
    [[ $nl -gt 0 ]] && python - ${LAB[$t]} ${LEVEL[$t]} <<'EOF'
import sys, glob, pandas as pd
f = sorted(glob.glob(sys.argv[1] + "/*.labels.csv.gz"))[0]
c = pd.read_csv(f, nrows=5).columns
need = [f"{sys.argv[2]}_{a}" for a in ("nucleus", "nucA", "nucB", "wholeB")]
miss = [x for x in need if x not in c]
print("  arms:", "ok" if not miss else f"MISSING {miss}")
EOF
  done
}

expr() {
  local t=$1 W=$KR/$1/expr
  known $t; [[ -e $W ]] && { print "$t: $W exists; skipped (move it away to rerun)"; return 0 }
  mkdir -p $KR/$t
  stamp "$t: expression kernel"
  claimbreak compartment kernel --h5ad-dir ${H5[$t]} --labels ${LAB[$t]} --out $W --level ${LEVEL[$t]} \
    --unit-pattern "${UNIT[$t]}" --k $K --r $KRAD 2>&1 | tee $KR/$t/expr.log | cut -c1-240
  [[ -f $W/decoy_pairs.csv.gz ]] || stop "$t: kernel wrote nothing"
}

labels() {
  local t=$1 W=$KR/$1/labels bx=()
  known $t; [[ -e $W ]] && { print "$t: $W exists; skipped (move it away to rerun)"; return 0 }
  [[ -n ${BND[$t]} && -d ${BND[$t]} ]] && bx=(--boundaries ${BND[$t]} --tol 1)
  mkdir -p $KR/$t
  stamp "$t: label kernel (exposure; $([[ ${#bx} -gt 0 ]] && print polygons || print nuclear disks))"
  claimbreak compartment exposure --labels ${LAB[$t]} --h5ad-dir ${H5[$t]} --out $W --levels ${LEVEL[$t]} \
    --unit-pattern "${UNIT[$t]}" --test wholeB --gaps 2 --contact-gap 10 --regions '' $bx 2>&1 \
    | tee $KR/$t/labels.log | cut -c1-240
  [[ -f $W/destinations.csv ]] || stop "$t: exposure wrote nothing"
}

summary() {
  local t=$1 S=$KR/$1/summary a=()
  known $t
  [[ -f $KR/$t/expr/decoy_pairs.csv.gz ]] && a+=(--spill $KR/$t/expr)
  [[ -f $KR/$t/labels/destinations.csv ]] && a+=(--exposure $KR/$t/labels)
  [[ ${#a} -gt 0 ]] || { print "$t: nothing to summarise"; return 0 }
  rm -rf $S
  stamp "$t: summary"
  claimbreak compartment kernel-summary --out $S --dataset $t --boot $BOOT $a 2>&1 | tee $KR/$t/summary.log | cut -c1-240
}

audit() {
  local t=$1 W=$KR/$1/audit rf=() rp
  known $t; [[ -e $W ]] && { print "$t: $W exists; skipped (move it away to rerun)"; return 0 }
  rp=${REF[$t]}
  if [[ -z $rp ]]; then
    rp=$(python -c "import json,glob,sys; f=sorted(glob.glob(sys.argv[1]+'/*.meta.json')); print(json.load(open(f[0])).get('reference','') if f else '')" ${LAB[$t]})
  else
    rf=(--reference $rp)
  fi
  [[ -n $rp && -f ${~rp} ]] || { print "$t: reference '$rp' not found; set REF_${(U)t}=<path to the reference .h5ad>"; return 0 }
  mkdir -p $KR/$t
  stamp "$t: label audit of the stored labels (reference $rp)"
  claimbreak compartment audit --labels ${LAB[$t]} --out $W --recompute ${H5[$t]} --arms nucleus,whole $rf 2>&1 \
    | tee $KR/$t/audit.log | cut -c1-240
  [[ -f $W/reproduce.csv ]] || print "$t: audit wrote nothing"
}

harvest() {
  local DL=${CB_DOWNLOADS:-~/Downloads} Z t
  Z=$DL/cb_kernel_$(date +%Y%m%d_%H%M).zip
  (cd $KR && for t in *(/); do
     zip -q -r $Z $t/summary $t/*.log $t/expr/decoy_pairs.csv.gz(N) $t/expr/{own_markers,sections}.csv(N) $t/expr/meta.json(N) \
       $t/labels/{notes.txt,dest_fit.csv,units.csv}(N) $t/audit/*.csv(N) $t/audit/*.txt(N) $t/audit/*.json(N) 2>/dev/null
   done)
  ls -lh $Z
}

mkdir -p $R/logs
cmd=$1; shift
case $cmd in
  check) check $@ ;;
  expr|labels|summary|audit) for t in ${@:-$ALL}; do $cmd $t; done ;;
  harvest) harvest ;;
  all) check $@ && for t in ${@:-$ALL}; do expr $t; labels $t; summary $t; done; harvest ;;
  *) stop "usage: kernel_batch.zsh check|expr|labels|summary|audit|harvest|all [tissues]" ;;
esac
