#!/bin/zsh
# region_loss.zsh -- where T/NK cells lose their labels, by the share of partner cells
# within 25 um (the paper: gut intraepithelial against lamina propria T cells; lung tumor nests against stroma; kidney
# and synovium with the Fig. 4a partners for comparison). Kept ratio of whole-cell counts (equal depth) against half B
# for T/NK cells of half A, in three regions: away (< 0.2 partner cells around), border (0.2-0.5), inside (>= 0.5),
# per patient and by mask type. Labels and settings as per100_fixed.zsh.
#
#   caffeinate -ims zsh $CB/examples/compartment/region_loss.zsh 2>&1 | tee ~/claimbreak_results/logs/region_loss_$(date +%Y%m%d_%H%M).log
#
# Env: D, LR, R, RA_LAB, TISSUES (gut lung kidney ra), RADIUS (25). Writes $R/region_loss/<tissue>; refuses to overwrite.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; LR=${LR:-~/claimbreak_results}; R=${R:-~/claimbreak_results}
RA_LAB=${RA_LAB:-$D/data/RA_labels/gseh4_pre/all}; TISSUES=(${=TISSUES:-gut lung kidney ra}); RADIUS=${RADIUS:-25}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
python -c "import claimbreak, sys; v = tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(v < (1, 7, 103))" \
  || stop "claimbreak 1.7.103 or later needed"
typeset -A LAB LEVEL UNIT PART STAIN
LAB[gut]=$D/IBD/labels/xenium_v1; LEVEL[gut]=L1; UNIT[gut]='I0\d{3}'; PART[gut]=Epithelial; STAIN[gut]='(withSegment|Colon|5k)'
LAB[lung]=$LR/cytoB_lung/merged; LEVEL[lung]=L2m; UNIT[lung]='L\d(?=_L)'; PART[lung]=Tumour; STAIN[lung]='5k'
LAB[kidney]=$D/AIN/labels/v1; LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'; PART[kidney]=LOH; STAIN[kidney]='.'
LAB[ra]=$RA_LAB; LEVEL[ra]=L1; UNIT[ra]='p\d{2}'; PART[ra]=Fibroblast; STAIN[ra]='.'
O=$R/region_loss
for t in $TISSUES; do
  [[ -n ${LAB[$t]} ]] || stop "unknown tissue $t (gut lung kidney ra)"
  ls ${LAB[$t]}/*.labels.csv.gz >/dev/null 2>&1 || stop "$t: no labels in ${LAB[$t]}"
  [[ -e $O/$t ]] && stop "$O/$t exists; move it away"
done
mkdir -p $O/logs
for t in $TISSUES; do
  stamp "$t: T/NK label loss by the share of ${PART[$t]} cells within $RADIUS um"
  python -u $CB/examples/compartment/region_loss.py --labels ${LAB[$t]} --out $O/$t --level ${LEVEL[$t]} --immune T_NK \
    --partner ${PART[$t]} --radius $RADIUS --unit-pattern "${UNIT[$t]}" --stain-pattern "${STAIN[$t]}" \
    > $O/logs/$t.log 2>&1 || { tail -5 $O/logs/$t.log; stop "$t failed" }
  sed -n '/^median over units/,$p' $O/logs/$t.log
done
print "done"
