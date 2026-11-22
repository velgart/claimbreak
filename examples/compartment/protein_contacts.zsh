#!/bin/zsh
# protein_contacts.zsh -- T-cell contacts scored against protein in both renal
# carcinomas (Fig. 4 and Supplementary Table "protein contacts"). protein_contacts.py on each Xenium Protein
# section, with two partner sets typed by protein: CD45-negative cells (all non-immune) and PanCK+ cells (tumor).
#
#   caffeinate -ims zsh $CB/examples/compartment/protein_contacts.zsh 2>&1 | tee ~/claimbreak_results/logs/protein_contacts_$(date +%Y%m%d_%H%M).log
#
# Env: D data root (~/claimbreak_data), R results root holding xp_rcc and xp_ccrcc (~/claimbreak_results), LR where xp_<s>/labels is looked for when R has none (~/claimbreak_results), SECTIONS
# (rcc ccrcc), DRAWS (1000). Writes $R/xp_<s>/protein_contacts_{cd45neg,panck} (T = CD3E+) and
# $R/xp_<s>/protein_contacts_strict_{cd45neg,panck} (T = CD3E+ CD45+ CD68- PanCK-); refuses to overwrite.
# PanCK does not separate in renal carcinoma 1 (no PanCK+ partner cells): its panck rows are NaN.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}; LR=${LR:-~/claimbreak_results}
SECTIONS=(${=SECTIONS:-rcc ccrcc}); DRAWS=${DRAWS:-1000}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

python -c "import claimbreak, sys; v = tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(v < (1, 7, 103))" \
  || stop "claimbreak 1.7.103 or later needed"
typeset -A SAMPLE PROT LAB
SAMPLE[rcc]=XP_RCC;     PROT[rcc]=$D/data/xenium_protein/h5ad/XP_RCC.protein.csv.gz
SAMPLE[ccrcc]=XP_CCRCC; PROT[ccrcc]=$D/data/xenium_protein_ccrcc/h5ad/XP_CCRCC.protein.csv.gz
for s in $SECTIONS; do
  [[ -n ${SAMPLE[$s]} ]] || stop "unknown section $s (rcc ccrcc)"
  for l in $R/xp_$s/labels $LR/xp_$s/labels; do [[ -f $l/${SAMPLE[$s]}.labels.csv.gz ]] && { LAB[$s]=$l; break }; done
  [[ -n ${LAB[$s]} ]] && print "$s labels: ${LAB[$s]}" || stop "$s: no labels in $R/xp_$s/labels or $LR/xp_$s/labels"
  [[ -f ${PROT[$s]} ]] || stop "$s: no protein table ${PROT[$s]}"
  for p in cd45neg panck strict_cd45neg strict_panck; do [[ -e $R/xp_$s/protein_contacts_$p ]] && stop "$R/xp_$s/protein_contacts_$p exists; move it away"; done
done
for s in $SECTIONS; do
  for tc in T T_strict; do
    for p in cd45neg panck; do
      [[ $p == cd45neg ]] && spec=CD45- || spec=PANCK+
      [[ $tc == T ]] && o=$R/xp_$s/protein_contacts_$p || o=$R/xp_$s/protein_contacts_strict_$p
      stamp "$s: T ($tc) contacts with $spec cells, T by protein and by each arm"
      python -u $CB/examples/compartment/protein_contacts.py --labels ${LAB[$s]} --protein ${PROT[$s]} \
        --sample ${SAMPLE[$s]} --out $o --partner $spec --hybrid nucleus:whole --t-class $tc \
        --arms nucleus,whole,nucB,wholeB --block-draws $DRAWS || stop "$s $tc $p failed"
    done
  done
done
stamp "summary (all cells): per100 and contact-share ratios, estimate [95% interval]"
python - $R ${SECTIONS[@]} <<'EOF'
import sys, os, pandas as pd
R, secs = sys.argv[1], sys.argv[2:]
show = [("wholeB", "per100_ratio_to_nucB"), ("wholeB", "share_ratio_to_nucB"), ("whole", "per100_ratio_to_nucleus"),
        ("whole", "share_ratio_to_nucleus"), ("nucleus+whole", "per100_ratio_to_whole"),
        ("nucB", "share_ratio_to_protein"), ("wholeB", "share_ratio_to_protein"), ("nucleus+whole", "share_ratio_to_protein")]
for s in secs:
    for tag in ("", "strict_"):
        for p in ("cd45neg", "panck"):
            f = os.path.join(R, f"xp_{s}", f"protein_contacts_{tag}{p}", "contacts.csv")
            if not os.path.exists(f):
                continue
            C = pd.read_csv(f)
            C = C[C.variant == "all"].set_index(["arm", "metric"])
            cells = []
            for arm, m in show:
                if (arm, m) in C.index:
                    r = C.loc[(arm, m)]
                    cells.append(f"{arm} {m.replace('_ratio_to_', '/')} {r.estimate:.2f} [{r.lo:.2f}, {r.hi:.2f}]")
            print(f"{s:6s} {tag + p:15s} " + "; ".join(cells))
EOF
print "done"
