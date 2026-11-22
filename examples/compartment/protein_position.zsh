#!/bin/zsh
# protein_position.zsh -- the control of Fig. 4a (the same label changes placed at
# random) in the two renal carcinomas, with partners typed by protein and by labels.
# Question: does the label-only analysis find the same small positional part that protein found (protein_contacts.zsh: the share of
# T cells touching a CD45- cell barely changed between half B and whole-cell counts at equal depth)? If labels and
# protein agree here, the renal carcinomas join the label-protein agreements; if labels report a large positional
# part where protein reports a small one, the Fig. 4a gaps need a caveat.
#
#   caffeinate -ims zsh $CB/examples/compartment/protein_position.zsh 2>&1 | tee ~/claimbreak_results/logs/protein_position_$(date +%Y%m%d_%H%M).log
#
# Runs per section (protein_contacts.py --confusion 10, 1,000 tile draws; into $R/xp_<s>/position_<partner>):
#   cd45neg   partners CD45- by protein (fixed for every labeling)
#   panck     partners PanCK+ by protein (renal carcinoma 2 only; PanCK does not separate in 1)
#   labels    partners typed by each labeling's own non-immune labels (the label-only version)
# and, for the exact Fig. 4a statistic, claimbreak compartment contacts --confusion with the kidney label PT as the
# partner (as xp_batch.zsh test), into $R/xp_<s>/contacts_conf_PT. Env as protein_contacts.zsh. Refuses to overwrite.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}; LR=${LR:-~/claimbreak_results}
SECTIONS=(${=SECTIONS:-rcc ccrcc}); DRAWS=${DRAWS:-1000}; NCONF=${NCONF:-10}; PART=${PART:-PT}
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
  for p in position_cd45neg position_panck position_labels contacts_conf_$PART; do
    [[ -e $R/xp_$s/$p ]] && stop "$R/xp_$s/$p exists; move it away"
  done
done
for s in $SECTIONS; do
  for p in cd45neg panck labels; do
    [[ $p == panck && $s == rcc ]] && continue
    case $p in
      cd45neg) opt=(--partner CD45-) ;;
      panck)   opt=(--partner PANCK+) ;;
      labels)  opt=(--partner-from labels) ;;
    esac
    stamp "$s: T contacts, partners $p, with the same changes placed at random ($NCONF draws)"
    python -u $CB/examples/compartment/protein_contacts.py --labels ${LAB[$s]} --protein ${PROT[$s]} \
      --sample ${SAMPLE[$s]} --out $R/xp_$s/position_$p $opt --hybrid nucleus:whole --arms nucleus,whole,nucB,wholeB \
      --confusion $NCONF --block-draws $DRAWS | grep -E "change_pct|position_pct|conf |^wrote" || stop "$s $p failed"
  done
  stamp "$s: compartment contacts --confusion, T_NK per 100 $PART cells (Fig. 4a statistic)"
  claimbreak compartment contacts --labels ${LAB[$s]} --out $R/xp_$s/contacts_conf_$PART --level L1 --immune T_NK \
    --partners $PART --ref nucB --test wholeB --unit-pattern "${SAMPLE[$s]}" --mode composition --contact knn \
    --confusion > $R/xp_$s/contacts_conf_$PART.log 2>&1 || { tail -5 $R/xp_$s/contacts_conf_$PART.log; stop "$s contacts" }
done

stamp "summary: change of T contacts from half B to whole-cell counts (equal depth), the same changes at random, and the part from position (percentage points)"
python - $R $PART ${SECTIONS[@]} <<'EOF'
import sys, os, pandas as pd
R, part, secs = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for s in secs:
    for p in ("cd45neg", "panck", "labels"):
        f = os.path.join(R, f"xp_{s}", f"position_{p}", "contacts.csv")
        if not os.path.exists(f):
            continue
        C = pd.read_csv(f)
        C = C[C.variant == "all"].set_index(["arm", "metric"])
        g = lambda a, m: C.loc[(a, m)] if (a, m) in C.index else None
        ch, cf, po, sh = g("wholeB", "change_pct"), g("conf", "change_pct"), g("wholeB", "position_pct"), g("wholeB", "share_ratio_to_nucB")
        rows.append(dict(section=s, partners=p, change=ch.estimate, conf=cf.estimate, position=po.estimate,
                         position_lo=po.lo, position_hi=po.hi, share_ratio=sh.estimate))
    f = os.path.join(R, f"xp_{s}", f"contacts_conf_{part}", "units.csv")
    if os.path.exists(f):
        U = pd.read_csv(f)
        col = f"per100|T_NK|{part}"
        if col in U:
            v = U[U.level == "unit"].set_index("labels")[col]
            if not len(v):
                v = U.groupby("labels")[col].mean()
            rows.append(dict(section=s, partners=f"label {part} (contacts)", change=100 * (v["test"] / v["ref"] - 1),
                             conf=100 * (v["confusion"] / v["ref"] - 1),
                             position=100 * (v["test"] - v["confusion"]) / v["ref"]))
        else:
            print(f"  {f}: no column {col}; columns {[c for c in U.columns if c.startswith('per100|')]}")
T = pd.DataFrame(rows)
T.to_csv(os.path.join(R, "position_rcc.csv"), index=False)
pd.set_option("display.width", 200)
print(T.round(2).to_string(index=False))
print("Fig. 4a, four Xenium tissues: position part 17-33 points (per100_conf/per100_units.csv)")
print(f"wrote {os.path.join(R, 'position_rcc.csv')}")
EOF
print "done"
