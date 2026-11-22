#!/bin/zsh
# per100_conf.zsh -- T/NK contacts per 100 partner cells under random relabeling at the
# real confusion rates (Fig. 4a). The usual stress test reassigns 10% of the whole-cell labels; this control
# relabels the nuclear labels (nucB) at random with each section's own confusion P(wholeB | nucB), so it makes as many
# changes of each kind as the whole-cell counts do (e.g. as many T/NK -> tumor), but places them without regard to
# contact. The real change minus this one is the part of the contact loss that comes from where the errors fall.
#
#   caffeinate -ims zsh $CB/examples/compartment/per100_conf.zsh 2>&1 | tee ~/claimbreak_results/logs/per100_conf_$(date +%Y%m%d_%H%M).log
#
# Env: D data root (~/claimbreak_data), LR labels root of batch T (~/claimbreak_results; lung labels in
# $LR/cytoB_lung/merged), R results root (~/claimbreak_results), RA_LAB synovium labels ($D/data/RA_labels/gseh4_pre/all),
# RA_DIR synovium results folder under R holding contacts_dm (ra_gseh4_pre), TISSUES (lung gut kidney ra).
# Same settings as batch T (robust_batch.zsh boundary, contact knn) and the synovium contacts_dm run: --ref nucB
# --test wholeB --mode composition, default eps, reps and seed, so ref, test and random_10 must reproduce those runs;
# the script checks this. Writes $R/per100_conf/<tissue>/ and $R/per100_conf/per100_units.csv. Refuses to overwrite.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; LR=${LR:-~/claimbreak_results}; R=${R:-~/claimbreak_results}
RA_LAB=${RA_LAB:-$D/data/RA_labels/gseh4_pre/all}; RA_DIR=${RA_DIR:-ra_gseh4_pre}
TISSUES=(${=TISSUES:-lung gut kidney ra})
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

typeset -A LAB LEVEL UNIT IMM PART OLD
LAB[lung]=$LR/cytoB_lung/merged; LEVEL[lung]=L2m; UNIT[lung]='L\d(?=_L)'; IMM[lung]=T_NK; PART[lung]=Tumour
LAB[gut]=$D/IBD/labels/xenium_v1; LEVEL[gut]=L1; UNIT[gut]='I0\d{3}'; IMM[gut]=T_NK; PART[gut]=Epithelial
LAB[kidney]=$D/AIN/labels/v1; LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'; IMM[kidney]=T_NK; PART[kidney]=LOH
LAB[ra]=$RA_LAB; LEVEL[ra]=L1; UNIT[ra]='p\d{2}'; IMM[ra]=T_NK,Myeloid,B,Plasma; PART[ra]=Fibroblast,Endothelial
for t in lung gut kidney; do OLD[$t]=$R/rob_boundary_$t/contacts_knn/units.csv; done
OLD[ra]=$R/$RA_DIR/contacts_dm/units.csv

O=$R/per100_conf
[[ -e $O ]] && stop "$O exists; move it away"
for t in $TISSUES; do
  [[ -n ${LAB[$t]} ]] || stop "unknown tissue $t (lung gut kidney ra)"
  ls ${LAB[$t]}/*.labels.csv.gz >/dev/null 2>&1 || stop "$t: no labels in ${LAB[$t]}"
done
mkdir -p $O/logs
for t in $TISSUES; do
  stamp "$t: contacts, nucB vs wholeB, with confusion-matched relabelling"
  claimbreak compartment contacts --labels ${LAB[$t]} --out $O/$t --level ${LEVEL[$t]} --immune ${IMM[$t]} \
    --partners ${PART[$t]} --ref nucB --test wholeB --unit-pattern "${UNIT[$t]}" --mode composition --contact knn \
    --confusion > $O/logs/$t.log 2>&1 || { tail -5 $O/logs/$t.log; stop "$t contacts" }
  grep -E "^wrote" $O/logs/$t.log
done

stamp "per unit table and check against the earlier runs"
python - $O "${(j:,:)TISSUES}" lung=${OLD[lung]} gut=${OLD[gut]} kidney=${OLD[kidney]} ra=${OLD[ra]} <<'EOF'
import os, sys
import pandas as pd
O, tissues = sys.argv[1], sys.argv[2].split(",")
old = dict(a.split("=", 1) for a in sys.argv[3:])
pair = {"lung": ("T_NK", "Tumour"), "gut": ("T_NK", "Epithelial"), "kidney": ("T_NK", "LOH"), "ra": ("T_NK", "Fibroblast")}
rows = []
for t in tissues:
    col = "per100|%s|%s" % pair[t]
    U = pd.read_csv(os.path.join(O, t, "units.csv"))
    P = U[U.level == "unit"].pivot_table(index="unit", columns="labels", values=col)
    r = pd.DataFrame({"tissue": t, "partner": pair[t][1], "ref_nucB": P["ref"], "test_wholeB": P["test"],
                      "random_10": P["random_10"], "confusion": P["confusion"]})
    r["change_pct"] = 100 * (r.test_wholeB / r.ref_nucB - 1)          # real: whole cell against nucleus
    r["random10_pct"] = 100 * (r.random_10 / r.test_wholeB - 1)       # usual test: 10% of the whole-cell labels
    r["confusion_pct"] = 100 * (r.confusion / r.ref_nucB - 1)         # same changes as real, placed at random
    r["position_pct"] = r.change_pct - r.confusion_pct                # the part that comes from where errors fall
    rows.append(r.reset_index())
    f = old.get(t, "")
    if os.path.exists(f):
        V = pd.read_csv(f)
        V = V[V.level == "unit"].pivot_table(index="unit", columns="labels", values=col)
        d = max((V[c] - P[c]).abs().max() for c in ("ref", "test", "random_10") if c in V)
        print(f"  {t}: largest difference from {f} in ref, test, random_10: {d:.2e}"
              + ("" if d < 1e-6 else "  <-- NOT the same settings or labels; check before using"))
    else:
        print(f"  {t}: no earlier run at {f}; not checked")
T = pd.concat(rows)
T.to_csv(os.path.join(O, "per100_units.csv"), index=False)
pd.set_option("display.width", 200)
print(T.round(1).to_string(index=False))
print(T.groupby("tissue")[["change_pct", "confusion_pct", "position_pct", "random10_pct"]].median().round(1).to_string())
print(f"wrote {O}/per100_units.csv")
EOF
