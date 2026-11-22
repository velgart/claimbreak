#!/bin/zsh
# per100_fixed.zsh -- Fig. 4a with the partner cells fixed at the half-B labels. protein_contacts.zsh showed, in the renal carcinomas, that the random-placement control also moves
# the partners' labels, and that placing those changes at random raises contacts; with partners fixed by protein, the
# part of the contact loss that comes from where the T-cell label changes fall was 2-3 points (10 at tumor contact),
# against 16-18 with label-typed partners. This run separates the two parts where there is no protein.
#
#   caffeinate -ims zsh $CB/examples/compartment/per100_fixed.zsh 2>&1 | tee ~/claimbreak_results/logs/per100_fixed_$(date +%Y%m%d_%H%M).log
#
# Part A, lung, gut, kidney, synovium: claimbreak compartment contacts with the settings of per100_conf.zsh (same seed,
#   so ref, test, random_10 and confusion must reproduce per100_conf/<tissue>/units.csv: checked) plus --fix-partners:
#   test_fixed and confusion_fixed keep every partner cell at its half-B label, so only the T/NK side varies.
#   Into $R/per100_fixed/<tissue>; per unit table $R/per100_fixed/per100_fixed_units.csv.
# Part B, renal carcinomas 1 and 2: protein_contacts.py --partner-from nucB (non-immune half-B labels, fixed for every
#   arm) with the same random-placement control, into $R/xp_<s>/position_nucB, next to the protein-typed partners of
#   protein_position.zsh (position_cd45neg): if labels with fixed partners give what protein gives, the label-only split of
#   the four tissues can be trusted.
# Env: as per100_conf.zsh (D, LR, R, RA_LAB, RA_DIR, TISSUES) and protein_contacts.zsh (LR for xp labels). Refuses to
# overwrite.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; LR=${LR:-~/claimbreak_results}; R=${R:-~/claimbreak_results}
RA_LAB=${RA_LAB:-$D/data/RA_labels/gseh4_pre/all}
TISSUES=(${=TISSUES:-lung gut kidney ra}); SECTIONS=(${=SECTIONS:-rcc ccrcc}); DRAWS=${DRAWS:-1000}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
python -c "import claimbreak, sys; v = tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(v < (1, 7, 103))" \
  || stop "claimbreak 1.7.103 or later needed"

typeset -A LAB LEVEL UNIT IMM PART
LAB[lung]=$LR/cytoB_lung/merged; LEVEL[lung]=L2m; UNIT[lung]='L\d(?=_L)'; IMM[lung]=T_NK; PART[lung]=Tumour
LAB[gut]=$D/IBD/labels/xenium_v1; LEVEL[gut]=L1; UNIT[gut]='I0\d{3}'; IMM[gut]=T_NK; PART[gut]=Epithelial
LAB[kidney]=$D/AIN/labels/v1; LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'; IMM[kidney]=T_NK; PART[kidney]=LOH
LAB[ra]=$RA_LAB; LEVEL[ra]=L1; UNIT[ra]='p\d{2}'; IMM[ra]=T_NK,Myeloid,B,Plasma; PART[ra]=Fibroblast,Endothelial
typeset -A SAMPLE PROT XLAB
SAMPLE[rcc]=XP_RCC;     PROT[rcc]=$D/data/xenium_protein/h5ad/XP_RCC.protein.csv.gz
SAMPLE[ccrcc]=XP_CCRCC; PROT[ccrcc]=$D/data/xenium_protein_ccrcc/h5ad/XP_CCRCC.protein.csv.gz

O=$R/per100_fixed
[[ -e $O ]] && stop "$O exists; move it away"
for t in $TISSUES; do
  [[ -n ${LAB[$t]} ]] || stop "unknown tissue $t (lung gut kidney ra)"
  ls ${LAB[$t]}/*.labels.csv.gz >/dev/null 2>&1 || stop "$t: no labels in ${LAB[$t]}"
done
for s in $SECTIONS; do
  for l in $R/xp_$s/labels $LR/xp_$s/labels; do [[ -f $l/${SAMPLE[$s]}.labels.csv.gz ]] && { XLAB[$s]=$l; break }; done
  [[ -n ${XLAB[$s]} ]] || stop "$s: no labels"
  [[ -e $R/xp_$s/position_nucB ]] && stop "$R/xp_$s/position_nucB exists; move it away"
done
mkdir -p $O/logs

for t in $TISSUES; do
  stamp "$t: contacts, nucB vs wholeB, random placement, partners fixed at half B"
  claimbreak compartment contacts --labels ${LAB[$t]} --out $O/$t --level ${LEVEL[$t]} --immune ${IMM[$t]} \
    --partners ${PART[$t]} --ref nucB --test wholeB --unit-pattern "${UNIT[$t]}" --mode composition --contact knn \
    --confusion --fix-partners > $O/logs/$t.log 2>&1 || { tail -5 $O/logs/$t.log; stop "$t contacts" }
  grep -E "^wrote" $O/logs/$t.log
done
for s in $SECTIONS; do
  stamp "$s: T contacts with non-immune cells fixed at their half-B labels, random placement (10 draws)"
  python -u $CB/examples/compartment/protein_contacts.py --labels ${XLAB[$s]} --protein ${PROT[$s]} \
    --sample ${SAMPLE[$s]} --out $R/xp_$s/position_nucB --partner-from nucB --hybrid nucleus:whole \
    --arms nucleus,whole,nucB,wholeB --confusion 10 --block-draws $DRAWS | grep -E "change_pct|position_pct|^wrote" \
    || stop "$s failed"
done

stamp "summary: change from half B to whole-cell counts (equal depth), the same changes at random, and the part from position (points); partners by label as in Fig. 4a, and fixed"
python - $R "${(j:,:)TISSUES}" "${(j:,:)SECTIONS}" <<'EOF'
import os, sys
import pandas as pd
R, tissues, secs = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
O = os.path.join(R, "per100_fixed")
pair = {"lung": ("T_NK", "Tumour"), "gut": ("T_NK", "Epithelial"), "kidney": ("T_NK", "LOH"), "ra": ("T_NK", "Fibroblast")}
rows = []
for t in tissues:
    col = "per100|%s|%s" % pair[t]
    U = pd.read_csv(os.path.join(O, t, "units.csv"))
    P = U[U.level == "unit"].pivot_table(index="unit", columns="labels", values=col)
    f = os.path.join(R, "per100_conf", t, "units.csv")
    if os.path.exists(f):
        V = pd.read_csv(f)
        V = V[V.level == "unit"].pivot_table(index="unit", columns="labels", values=col)
        d = max((V[c] - P[c]).abs().max() for c in ("ref", "test", "random_10", "confusion") if c in V)
        print(f"  {t}: largest difference from per100_conf in ref, test, random_10, confusion: {d:.2e}"
              + ("" if d < 1e-6 else "  <-- NOT reproduced; check settings before using"))
    else:
        print(f"  {t}: no per100_conf run to check against ({f})")
    r = pd.DataFrame({"tissue": t, "partner": pair[t][1]}, index=P.index)
    r["change_pct"] = 100 * (P["test"] / P["ref"] - 1)
    r["confusion_pct"] = 100 * (P["confusion"] / P["ref"] - 1)
    r["position_pct"] = r.change_pct - r.confusion_pct
    r["change_fixed_pct"] = 100 * (P["test_fixed"] / P["ref"] - 1)
    r["confusion_fixed_pct"] = 100 * (P["confusion_fixed"] / P["ref"] - 1)
    r["position_fixed_pct"] = r.change_fixed_pct - r.confusion_fixed_pct
    rows.append(r.reset_index())
T = pd.concat(rows)
T.to_csv(os.path.join(O, "per100_fixed_units.csv"), index=False)
pd.set_option("display.width", 220)
cols = ["change_pct", "confusion_pct", "position_pct", "change_fixed_pct", "confusion_fixed_pct", "position_fixed_pct"]
print(T.round(1).to_string(index=False))
print("\nmedians per tissue:\n" + T.groupby("tissue")[cols].median().round(1).to_string())
print("\nrenal carcinomas, all non-immune partners: protein-typed and fixed at half B (this run)")
for s in secs:
    for tag in ("position_cd45neg", "position_nucB", "position_labels"):
        f = os.path.join(R, f"xp_{s}", tag, "contacts.csv")
        if not os.path.exists(f):
            continue
        C = pd.read_csv(f)
        C = C[C.variant == "all"].set_index(["arm", "metric"])
        g = lambda a, m: C.loc[(a, m)]
        po = g("wholeB", "position_pct")
        print(f"  {s:6s} {tag:17s} change {g('wholeB', 'change_pct').estimate:6.1f}  random {g('conf', 'change_pct').estimate:6.1f}  "
              f"position {po.estimate:6.1f} [{po.lo:.1f}, {po.hi:.1f}]")
print(f"wrote {O}/per100_fixed_units.csv")
EOF
print "done"
