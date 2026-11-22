#!/bin/zsh
# state_depth.zsh -- which T cells lose their labels, matched on nuclear RNA depth
# (region result). In region_check.zsh, part A, a T/NK call counted as "positive" for a state when its
# nucleus held >= 1 count of the gene set, which is more likely in a nucleus with more counts; cytotoxic T cells lost
# more labels in gut, but the state and RNA depth were mixed. This run repeats part A with region_loss.py --depth-strata
# (depth.csv: loss of positive minus negative cells within five strata of nuclear counts per patient), adds CD8, CD4,
# gamma-delta and tissue-resident sets, and asks whether the region contrast itself survives matching on depth.
#
#   caffeinate -ims zsh $CB/examples/compartment/state_depth.zsh 2>&1 | tee ~/claimbreak_results/logs/state_depth_$(date +%Y%m%d_%H%M).log
#
# Env: as region_check.zsh (D, LR, R, RA_LAB, RADIUS), TISSUES (gut lung ra kidney), STRATA (5). Writes
# $R/region_check/<tissue>_T_depth and $R/region_check/state_depth_summary.csv; refuses to overwrite.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; LR=${LR:-~/claimbreak_results}; R=${R:-~/claimbreak_results}
RA_LAB=${RA_LAB:-$D/data/RA_labels/gseh4_pre/all}; RADIUS=${RADIUS:-25}; STRATA=${STRATA:-5}
TISSUES=(${=TISSUES:-gut lung ra kidney})
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
python -c "import claimbreak, sys; v = tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(v < (1, 7, 103))" \
  || stop "claimbreak 1.7.103 or later needed"
[[ -d $D ]] || stop "data drive not mounted ($D)"
RL=$CB/examples/compartment/region_loss.py
O=$R/region_check; mkdir -p $O/logs

typeset -A LAB LEVEL UNIT H5 STAIN PART PMK
LAB[gut]=$D/IBD/labels/xenium_v1;  LEVEL[gut]=L1;    UNIT[gut]='I0\d{3}';               H5[gut]=$D/IBD/h5ad;      STAIN[gut]='(withSegment|Colon|5k)'
LAB[lung]=$LR/cytoB_lung/merged;   LEVEL[lung]=L2m;  UNIT[lung]='L\d(?=_L)';            H5[lung]=$D/Bilous_h5ad;  STAIN[lung]='5k'
LAB[kidney]=$D/AIN/labels/v1;      LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'; H5[kidney]=$D/AIN/h5ad;   STAIN[kidney]='.'
LAB[ra]=$RA_LAB;                   LEVEL[ra]=L1;     UNIT[ra]='p\d{2}';                 H5[ra]=$D/data/RA_h5ad/compartment_pre; STAIN[ra]='.'
PART[gut]=Epithelial; PART[lung]=Tumour; PART[kidney]=PT; PART[ra]=Fibroblast
PMK[gut]=EPCAM,KRT8,KRT18,KRT19,KRT20,CDH1,CDX1,CDX2,VIL1,LGALS4,PIGR,CEACAM5,MUC2,ELF3,CLDN4,CLDN7,FABP1,OLFM4,LGR5
PMK[lung]=EPCAM,KRT8,KRT18,KRT19,KRT7,KRT5,TP63,NKX2-1,SFTPB,NAPSA,CEACAM5,CEACAM6,MUC1,ELF3,CLDN4
PMK[kidney]=LRP2,CUBN,SLC34A1,HNF4A,ALDOB,GPX3,SLC22A6,SLC5A12,MIOX,PDZK1
PMK[ra]=COL1A1,COL1A2,COL3A1,DCN,LUM,PDGFRA,FAP,PDPN,THY1,PRG4,CLU,FN1
OWN_T=CD3E,CD3D,CD3G,CD2,CD247,TRAC,TRBC1,TRBC2,CD7,LCK,CD8A,CD8B,IL7R
STATE='cytotox=GZMB,GZMA,GZMK,GZMH,PRF1,NKG7,GNLY;prolif=MKI67,TOP2A,CENPF,TYMS,STMN1;naive=CCR7,SELL,TCF7,LEF1;exhaust=PDCD1,HAVCR2,LAG3,TOX,TIGIT,CTLA4;activ=CD69,IFNG,TNFRSF9,HLA-DRA,ICOS;cd8=CD8A,CD8B;cd4=CD4,CD40LG;gd=TRDC,TRGC1,TRGC2;trm=CXCR6,ITGA1,ZNF683,ITGAE'

for t in $TISSUES; do
  [[ -n ${LAB[$t]} ]] || stop "unknown tissue $t (gut lung ra kidney)"
  [[ -e $O/${t}_T_depth ]] && stop "$O/${t}_T_depth exists; move it away"
done
[[ -e $O/state_depth_summary.csv ]] && stop "$O/state_depth_summary.csv exists; move it away"
for t in $TISSUES; do
  stamp "$t: T/NK loss by state, matched on nuclear RNA depth ($STRATA strata), partner ${PART[$t]}"
  python -u $RL --labels ${LAB[$t]} --out $O/${t}_T_depth --level ${LEVEL[$t]} --immune T_NK --partner ${PART[$t]} \
    --radius $RADIUS --unit-pattern "${UNIT[$t]}" --stain-pattern "${STAIN[$t]}" --h5ad ${H5[$t]} --own-markers $OWN_T \
    --partner-markers ${PMK[$t]} --state-genes "$STATE" --depth-strata $STRATA > $O/logs/${t}_T_depth.log 2>&1 \
    || { tail -5 $O/logs/${t}_T_depth.log; stop "$t failed" }
  sed -n '/^loss positive minus negative/,$p' $O/logs/${t}_T_depth.log
done

stamp "summary: loss of positive minus negative T/NK calls (points), crude and matched on depth; median over patients"
python - $O ${TISSUES[@]} <<'EOF'
import os, sys
import pandas as pd
O, tissues = sys.argv[1], sys.argv[2:]
rows = []
for t in tissues:
    f = os.path.join(O, f"{t}_T_depth", "depth.csv")
    if not os.path.exists(f):
        print(f"  {t}: no depth.csv"); continue
    X = pd.read_csv(f)
    X = X[X.unit == "median"].assign(tissue=t)
    rows.append(X)
S = pd.concat(rows, ignore_index=True)
S = S[["tissue", "bin", "feature", "units", "units_positive", "cells_pos", "cells_neg", "crude_diff", "adjusted_diff",
       "pooled_adjusted"]]
S.to_csv(os.path.join(O, "state_depth_summary.csv"), index=False)
pd.set_option("display.width", 200)
for b in ("all", "away", "inside"):
    print(f"\nbin {b}:")
    print(S[S.bin == b].drop(columns="bin").round(1).to_string(index=False))
print(f"\nwrote {os.path.join(O, 'state_depth_summary.csv')}")
EOF
print "done"
