#!/bin/zsh
# region_check.zsh -- checks of the region result of region_loss.zsh (the paper; gut:
# T/NK cells inside the epithelium lost about 69% of their labels, in the lamina propria about 25%).
#
#   caffeinate -ims zsh $CB/examples/compartment/region_check.zsh 2>&1 | tee ~/claimbreak_results/logs/region_check_$(date +%Y%m%d_%H%M).log
#
# A  identity and state (region_loss.py with --h5ad): are the T/NK calls inside the partner region T cells (size,
#    T and partner markers in the nucleus, against the partner cells), and which T/NK cells lose labels (cytotoxic,
#    proliferating, naive-like, exhausted, activated, CD103; nuclear RNA content quartile). Gut, lung, synovium, and
#    kidney with the proximal tubule as partner. Into $R/region_check/<tissue>_T.
# B  kidney with all non-immune cells as partner; other small immune cells (B, with plasma cells as an RNA-rich
#    control) in gut, lung and synovium. Into $R/region_check/<tissue>_<type>. Missing labels are skipped.
# C  protein truth (protein_region.py): renal carcinomas, protein T cells (CD3E+, and the strict class) inside
#    CD45-negative regions, and in donor 2 inside PanCK+ tumor, against away. Into $R/xp_<s>/region_<region>[_strict].
# Env: D, LR, R, RA_LAB, RADIUS (25), DRAWS (1000), PARTS (A B C). Refuses to overwrite.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; LR=${LR:-~/claimbreak_results}; R=${R:-~/claimbreak_results}
RA_LAB=${RA_LAB:-$D/data/RA_labels/gseh4_pre/all}; RADIUS=${RADIUS:-25}; DRAWS=${DRAWS:-1000}; PARTS=(${=PARTS:-A B C})
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
python -c "import claimbreak, sys; v = tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(v < (1, 7, 103))" \
  || stop "claimbreak 1.7.103 or later needed"
[[ -d $D ]] || stop "data drive not mounted ($D)"
RL=$CB/examples/compartment/region_loss.py
O=$R/region_check; mkdir -p $O/logs

typeset -A LAB LEVEL UNIT H5 STAIN PMK
LAB[gut]=$D/IBD/labels/xenium_v1;  LEVEL[gut]=L1;    UNIT[gut]='I0\d{3}';               H5[gut]=$D/IBD/h5ad;      STAIN[gut]='(withSegment|Colon|5k)'
LAB[lung]=$LR/cytoB_lung/merged;   LEVEL[lung]=L2m;  UNIT[lung]='L\d(?=_L)';            H5[lung]=$D/Bilous_h5ad;  STAIN[lung]='5k'
LAB[kidney]=$D/AIN/labels/v1;      LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'; H5[kidney]=$D/AIN/h5ad;   STAIN[kidney]='.'
LAB[ra]=$RA_LAB;                   LEVEL[ra]=L1;     UNIT[ra]='p\d{2}';                 H5[ra]=$D/data/RA_h5ad/compartment_pre; STAIN[ra]='.'
typeset -A PART
PART[gut]=Epithelial; PART[lung]=Tumour; PART[kidney]=PT; PART[ra]=Fibroblast
PMK[gut]=EPCAM,KRT8,KRT18,KRT19,KRT20,CDH1,CDX1,CDX2,VIL1,LGALS4,PIGR,CEACAM5,MUC2,ELF3,CLDN4,CLDN7,FABP1,OLFM4,LGR5
PMK[lung]=EPCAM,KRT8,KRT18,KRT19,KRT7,KRT5,TP63,NKX2-1,SFTPB,NAPSA,CEACAM5,CEACAM6,MUC1,ELF3,CLDN4
PMK[kidney]=LRP2,CUBN,SLC34A1,HNF4A,ALDOB,GPX3,SLC22A6,SLC5A12,MIOX,PDZK1
PMK[ra]=COL1A1,COL1A2,COL3A1,DCN,LUM,PDGFRA,FAP,PDPN,THY1,PRG4,CLU,FN1
OWN_T=CD3E,CD3D,CD3G,CD2,CD247,TRAC,TRBC1,TRBC2,CD7,LCK,CD8A,CD8B,IL7R
STATE='cytotox=GZMB,GZMA,GZMK,GZMH,PRF1,NKG7,GNLY;prolif=MKI67,TOP2A,CENPF,TYMS,STMN1;naive=CCR7,SELL,TCF7,LEF1;exhaust=PDCD1,HAVCR2,LAG3,TOX,TIGIT,CTLA4;activ=CD69,IFNG,TNFRSF9,HLA-DRA,ICOS;cd103=ITGAE'

run_rl() {   # tissue, immune label, partner, outdir, extra args...
  local t=$1 imm=$2 part=$3 o=$4; shift 4
  [[ -e $o ]] && { print "exists, kept: $o"; return 0 }
  python -u $RL --labels ${LAB[$t]} --out $o --level ${LEVEL[$t]} --immune $imm --partner $part --radius $RADIUS \
    --unit-pattern "${UNIT[$t]}" --stain-pattern "${STAIN[$t]}" "$@" > $O/logs/${o:t}.log 2>&1 \
    || { tail -3 $O/logs/${o:t}.log; print "  (failed or no $imm cells; skipped)"; return 0 }
  sed -n '/^median over units/,$p' $O/logs/${o:t}.log
}

if (( ${PARTS[(I)A]} )); then
  for t in gut lung ra kidney; do
    stamp "A $t: T/NK identity and state, partner ${PART[$t]} (h5ad ${H5[$t]})"
    run_rl $t T_NK ${PART[$t]} $O/${t}_T --h5ad ${H5[$t]} --own-markers $OWN_T --partner-markers ${PMK[$t]} --state-genes "$STATE"
  done
fi
if (( ${PARTS[(I)B]} )); then
  stamp "B kidney: T/NK against all non-immune cells"
  run_rl kidney T_NK nonimmune $O/kidney_T_nonimmune
  for t in gut lung ra; do
    for imm in B Plasma; do
      stamp "B $t: $imm cells, partner ${PART[$t]}"
      run_rl $t $imm ${PART[$t]} $O/${t}_$imm
    done
  done
fi
if (( ${PARTS[(I)C]} )); then
  typeset -A SAMPLE PROT XLAB
  SAMPLE[rcc]=XP_RCC;     PROT[rcc]=$D/data/xenium_protein/h5ad/XP_RCC.protein.csv.gz
  SAMPLE[ccrcc]=XP_CCRCC; PROT[ccrcc]=$D/data/xenium_protein_ccrcc/h5ad/XP_CCRCC.protein.csv.gz
  for s in rcc ccrcc; do
    for l in $R/xp_$s/labels $LR/xp_$s/labels; do [[ -f $l/${SAMPLE[$s]}.labels.csv.gz ]] && { XLAB[$s]=$l; break }; done
    regs=(CD45-); [[ $s == ccrcc ]] && regs+=(PANCK+)
    for reg in $regs; do
      for tc in T T_strict; do
        tag=${${reg//-/neg}//+/pos}; [[ $tc == T_strict ]] && tag=${tag}_strict
        o=$R/xp_$s/region_${(L)tag}
        [[ -e $o ]] && { print "exists, kept: $o"; continue }
        stamp "C $s: protein $tc cells inside $reg regions against away"
        python -u $CB/examples/compartment/protein_region.py --labels ${XLAB[$s]} --protein ${PROT[$s]} \
          --sample ${SAMPLE[$s]} --out $o --region $reg --t-class $tc --radius $RADIUS --block-draws $DRAWS \
          || print "  ($s $reg $tc failed)"
      done
    done
  done
fi
print "done"
