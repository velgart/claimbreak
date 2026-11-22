#!/bin/zsh
# robust_batch.zsh -- robustness checks.
#   zsh $CB/examples/compartment/robust_batch.zsh check
#   caffeinate -ims zsh $CB/examples/compartment/robust_batch.zsh thirds lung gut kidney ra cosmx 2>&1 | tee ~/claimbreak_results/logs/batchT_thirds_$(date +%Y%m%d_%H%M).log
#   caffeinate -ims zsh $CB/examples/compartment/robust_batch.zsh boundary lung gut kidney 2>&1 | tee ~/claimbreak_results/logs/batchT_boundary_$(date +%Y%m%d_%H%M).log
#   caffeinate -ims zsh $CB/examples/compartment/robust_batch.zsh sweep lung gut 2>&1 | tee ~/claimbreak_results/logs/batchT_sweep_$(date +%Y%m%d_%H%M).log
#   PILOT=1 zsh $CB/examples/compartment/robust_batch.zsh thirds lung      # one section first
#   zsh $CB/examples/compartment/robust_batch.zsh harvest                  # ~/Downloads/cb_batchT_<time>.zip
# Into ~/claimbreak_results/rob_<step>_<tissue>:
#  thirds    The correction confound : some changes nucA -> wholeB are corrections of a wrong
#            nucA label, and corrections land on the (spatially clustered) true type, i.e. often a contact neighbor's.
#            Labels from three independent nuclear thirds (nucA3, nucB3, nucC3) and wholeB3 / cytoB3 (whole minus
#            nucA3 and nucC3, and cyto, thinned to the nucB3 depth). compare (a) on all cells, (b) only where nucC3
#            confirms nucA3 (--agree nucC3), (c) the same with reference margin >= 5 nats; summary with null
#            nucB3|nucA3. If the extra-change ratio, the T/NK -> partner excess and T/NK kept hold in (b)-(c), the
#            confound does not carry the result. Reading: (a) differs from the halves of batch S only by depth.
#  boundary  Contact = touching segmentation boundaries (tol 1 um; Xenium cell_boundaries.parquet of the masks that
#            collected the counts) instead of the 6 nearest within 30 um, on the existing labels (no relabeling):
#            compare + summary (wholeB, cytoB), and contacts (per 100 partner cells, nucB vs wholeB). A kNN run of the
#            same labels is made alongside so the two differ only in the contact definition.
#  sweep     Classifier floor (FLOORS, default 0.001,0.01,0.05) x prior (PRIORS, uniform,fit): label nucA, nucB,
#            wholeB, nucleus; summary (extra-change ratio, T/NK kept with block interval), T/NK wholeB / nucB per unit
#            (the composition readout of the paper), contacts per 100 partner cells (nucB vs wholeB).
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
# RA_LAB = synovium labels folder (default gseh2_pre); with R=<new root> the synovium results of an
#   audited relabeling (e.g. RA_LAB=$D/data/RA_labels/gseh4_pre/all R=~/claimbreak_results) go to a new folder.
FLOORS=${FLOORS:-0.001,0.01,0.05}; PRIORS=${PRIORS:-uniform,fit}
FOCUS=${FOCUS:-'(?i)(?:^|[^a-z])(?:t|nk|tnk)(?:[^a-z]|$)'}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

# tissue: labels with cytoB (compare) | labels whose meta.json gives the label options | h5ad | reference | ...
typeset -A CMPL ORIG H5 REF LEVCOL MERGE LEVEL UNIT IMM PART BND
CMPL[lung]=$R/cytoB_lung/merged;   ORIG[lung]=${LUNG_LABELS:-$R/bilous_v4};  H5[lung]=$D/Bilous_h5ad
CMPL[cosmx]=$R/cytoB_cosmx/merged; ORIG[cosmx]=$R/cosmx_nsclc_x171/multinomial/labels; H5[cosmx]=$D/data/cosmx_nsclc/h5ad
for t in lung cosmx; do
  REF[$t]=$D/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad; LEVCOL[$t]=L2m=Level2; MERGE[$t]='^Tu_L\d=>Tumour'; LEVEL[$t]=L2m
  IMM[$t]=T_NK; PART[$t]=Tumour
done
UNIT[lung]='L\d(?=_L)'; UNIT[cosmx]='Lung\d+'
CMPL[gut]=$D/IBD/labels/xenium_v1;  ORIG[gut]=${CMPL[gut]};  H5[gut]=$D/IBD/h5ad;  REF[gut]=$D/IBD/ref/gut_census_v2.h5ad
LEVCOL[gut]=L1=coarse; MERGE[gut]=''; LEVEL[gut]=L1; UNIT[gut]='I0\d{3}'; IMM[gut]=T_NK; PART[gut]=Epithelial
CMPL[kidney]=$D/AIN/labels/v1;      ORIG[kidney]=${CMPL[kidney]}; H5[kidney]=$D/AIN/h5ad; REF[kidney]=$D/AIN/ref/kidney_census.h5ad
LEVCOL[kidney]=L1=coarse; MERGE[kidney]=''; LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'; IMM[kidney]=T_NK; PART[kidney]=LOH
CMPL[ra]=${RA_LAB:-$D/data/RA_labels/gseh2_pre/all}; ORIG[ra]=${CMPL[ra]}; H5[ra]=$D/data/RA_h5ad/compartment_pre
REF[ra]=$D/data/RA_ref/gse200815h2_ref.h5ad; LEVCOL[ra]=L1=coarse; MERGE[ra]=''; LEVEL[ra]=L1; UNIT[ra]='p\d{2}'
IMM[ra]=T_NK; PART[ra]=Fibroblast
# boundaries of the masks that made the counts (<sample>_cell_boundaries.parquet[.gz] or <sample>/cell_boundaries...)
BND[lung]=${BND_LUNG:-$D/Bilous/raw}; BND[gut]=${BND_GUT:-$D/IBD/raw}; BND[kidney]=${BND_KIDNEY:-$D/AIN/boundaries}

optpy() { python - "$@" <<'EOF'
# label options of the existing run (meta.json), without the floor when the sweep sets it
import json, sys, glob, os
lab, out, nofloor = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
m = json.load(open(sorted(glob.glob(os.path.join(lab, "*.meta.json")))[0]))
o = []
lay = m.get("ref_layer", "X")
if lay == "raw": o += ["--ref-raw"]
elif lay not in ("X", "", None): o += ["--ref-layer", lay]
for k, f in (("ref_map", "--ref-map"), ("ref_subset", "--ref-subset")):
    if m.get(k): o += [f, m[k]]
if m.get("cv_group"): o += ["--cv-group", m["cv_group"]]
o += ["--min-counts", str(m.get("min_counts", 10)), "--seed", str(m.get("seed", 0)), "--gamma-iter", str(m.get("gamma_iter", 5))]
if not nofloor: o += ["--floor", str(m.get("floor", 0.01))]
if m.get("markers", 0): o += ["--markers", str(m["markers"]), "--marker-min", str(m.get("marker_min", 1e-4)),
                             "--marker-min-log2", str(m.get("marker_min_log2", 1.0))]
if m.get("exclude_genes"):
    open(out, "w").write("\n".join(m["exclude_genes"]) + "\n"); o += ["--exclude-genes", out]
print(" ".join(f"'{x}'" for x in o))
print(f"existing labels: reference {os.path.basename(str(m.get('reference')))}, levels {m.get('ref_levels')}, "
      f"merge {m.get('ref_merge')!r}, floor {m.get('floor')}, arms {m.get('arms')}, seed {m.get('seed')}", file=sys.stderr)
EOF
}

inputs() {   # h5ad files of the samples labelled in the existing run
  local t=$1 f s
  ins=()
  for f in ${ORIG[$t]}/*.labels.csv.gz; do s=${f:t}; s=${s%.labels.csv.gz}; [[ -f ${H5[$t]}/$s.h5ad ]] && ins+=(${H5[$t]}/$s.h5ad); done
  [[ -n $PILOT ]] && ins=( $ins[1] )
  (( ${#ins} )) || stop "$t: no h5ad for the labelled samples in ${H5[$t]}"
}

label() {   # tissue out arms [extra options]
  local t=$1 out=$2 arms=$3; shift 3
  local opts
  opts=$(optpy ${ORIG[$t]} ${out:h}/exclude_genes.txt ${NOFLOOR:-0}) || stop "$t: no meta.json in ${ORIG[$t]}"
  inputs $t
  print "$t: ${#ins} samples; options $opts $*"
  eval claimbreak compartment label --ref ${REF[$t]} --ref-levels ${LEVCOL[$t]} --levels ${LEVEL[$t]} \
    --ref-merge "'${MERGE[$t]}'" --inputs ${(q)ins} --out $out --arms $arms $opts "$@" 2>&1 \
    | tee ${out:h}/logs/label_${out:t}.log | grep -v "^  wrote" | cut -c1-240
  ls $out/*.labels.csv.gz >/dev/null 2>&1 || stop "$t: labelling wrote nothing ($out)"
}

summ() {   # compare_dir out test null tissue
  claimbreak compartment summary --compare $1 --out $2 --level ${LEVEL[$5]} --test "$3" --null "$4" \
    --unit-pattern "${UNIT[$5]}" --focus "$FOCUS" > ${2:h}/logs/${2:t}.log 2>&1 || { tail -5 ${2:h}/logs/${2:t}.log; stop "summary $2" }
  grep -E "^per unit, pooled|^block bootstrap" ${2:h}/logs/${2:t}.log | sed "s/^/  ${2:t}: /" | cut -c1-230
}

table() { python - "$@" <<'EOF'
# one row per summary folder: units, evaluable, pooled ratio, intervals above one, T/NK kept, T/NK -> partner excess
import glob, os, re, sys
import pandas as pd
W, imm, part = sys.argv[1:4]
rows = []
for d in sorted(glob.glob(os.path.join(W, "sum_*"))):
    f = os.path.join(d, "units.csv")
    if not os.path.exists(f) or os.path.getsize(f) < 5:
        rows.append({"run": os.path.basename(d)[4:], "units": 0, "evaluable": 0}); continue
    U = pd.read_csv(f)
    ev = U[U.evaluable] if "evaluable" in U else U
    r = {"run": os.path.basename(d)[4:], "units": len(U), "evaluable": len(ev),
         "pooled_median": ev.pooled_contact.median() if len(ev) else float("nan"),
         "pooled_min": ev.pooled_contact.min() if len(ev) else float("nan"),
         "param_ci_gt1": int(ev.ci_excludes_1.sum()) if "ci_excludes_1" in ev else None,
         "block_ci_gt1": int(ev.block_ci_excludes_1.sum()) if "block_ci_excludes_1" in ev else None,
         "far_only_median": ev.pooled_far_only.median() if "pooled_far_only" in ev and len(ev) else float("nan")}
    S = pd.read_csv(os.path.join(d, "sections.csv"))
    r["cells_null"] = int(S.null_cells.sum()) if "null_cells" in S else None
    k = os.path.join(d, "units_kept.csv")
    if os.path.exists(k) and os.path.getsize(k) > 5:
        K = pd.read_csv(k); K = K[K.cells_null >= 100]
        r.update({"tnk_units": len(K), "tnk_kept_median": K.kept_ratio.median(),
                  "tnk_block_hi_lt1": int((K.kept_ratio_hi < 1).sum())})
    p = os.path.join(d, "pairs.csv")
    if os.path.exists(p) and os.path.getsize(p) > 5:
        P = pd.read_csv(p)
        x = P[(P.a == imm) & (P.b == part)]
        if len(x):
            x = x.iloc[0]
            r.update({"excess_imm_part": x.excess_median, "sections_pos": f"{int(x.sections_pos)}/{int(x.sections)}",
                      "reverse_excess": x.get("reverse_excess")})
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(W, "robust_table.csv"), index=False)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))
EOF
}

thirds() {
  local t=$1 W=$R/rob_thirds_$1 v x
  [[ -n ${LEVEL[$t]} ]] || stop "unknown tissue $t"
  [[ -n $PILOT ]] && W=${W}_pilot
  [[ -e $W ]] && stop "$W exists; move it away"
  mkdir -p $W/logs
  stamp "$t: label nuclear thirds"
  label $t $W/labels nucA3,nucB3,nucC3,wholeB3,cytoB3,nucleus
  python - $W/labels ${LEVEL[$t]} <<'EOF'
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.meta.json"))):
    a = json.load(open(f))["agreement"]; lv = sys.argv[2]
    g = lambda k: a.get(f"{lv}:{k}", [float("nan"), 0])
    print(f"  {os.path.basename(f)[:-10]}: nucB3==nucA3 {g('nucB3==nucA3')[0]:.3f}, nucC3==nucA3 {g('nucC3==nucA3')[0]:.3f}"
          f" (cells {g('nucC3==nucA3')[1]:,}), wholeB3==nucA3 {g('wholeB3==nucA3')[0]:.3f}")
EOF
  for v in all agree agree_m5; do
    local extra=()
    [[ $v == agree* ]] && extra+=(--agree nucC3)
    [[ $v == agree_m5 ]] && extra+=(--min-margin 5)
    stamp "$t: compare ($v)"
    claimbreak compartment compare --labels $W/labels --out $W/cmp_$v --levels ${LEVEL[$t]} --null "nucB3|nucA3" \
      --comparisons "nucB3|nucA3,wholeB3|nucA3,cytoB3|nucA3" --tile 200 $extra > $W/logs/cmp_$v.log 2>&1 \
      || { tail -5 $W/logs/cmp_$v.log; stop "$t compare $v" }
    for x in wholeB3 cytoB3; do summ $W/cmp_$v $W/sum_${v}_$x "$x|nucA3" "nucB3|nucA3" $t; done
  done
  stamp "$t: thirds table"
  table $W ${IMM[$t]} ${PART[$t]} | tee $W/logs/table.log
}

boundary() {
  local t=$1 W=$R/rob_boundary_$1 c x
  [[ -n ${BND[$t]} ]] || stop "$t: no boundaries folder defined (lung, gut, kidney)"
  [[ -d ${BND[$t]} ]] || stop "$t: boundaries folder ${BND[$t]} not found (set BND_${(U)t})"
  [[ -e $W ]] && stop "$W exists; move it away"
  mkdir -p $W/logs
  local smp=()
  [[ -n $PILOT ]] && { local f=( ${CMPL[$t]}/*.labels.csv.gz ); smp=(--samples ${${f[1]:t}%.labels.csv.gz}) }
  for c in knn boundary; do
    stamp "$t: compare, contact $c"
    local ex=()
    [[ $c == boundary ]] && ex=(--boundaries ${BND[$t]} --tol 1)
    claimbreak compartment compare --labels ${CMPL[$t]} --out $W/cmp_$c --levels ${LEVEL[$t]} --contact $c $ex $smp \
      --comparisons "nucB|nucA,wholeB|nucA,cytoB|nucA" --tile 200 > $W/logs/cmp_$c.log 2>&1 \
      || { tail -5 $W/logs/cmp_$c.log; stop "$t compare $c" }
    [[ $c == boundary ]] && grep "boundary contacts" $W/logs/cmp_$c.log | head -3
    for x in wholeB cytoB; do summ $W/cmp_$c $W/sum_${c}_$x "$x|nucA" "nucB|nucA" $t; done
    stamp "$t: contacts per 100 ${PART[$t]} cells, nucB vs wholeB, contact $c"
    claimbreak compartment contacts --labels ${CMPL[$t]} --out $W/contacts_$c --level ${LEVEL[$t]} --immune ${IMM[$t]} \
      --partners ${PART[$t]} --ref nucB --test wholeB --unit-pattern "${UNIT[$t]}" --mode composition --contact $c $ex \
      > $W/logs/contacts_$c.log 2>&1 || { tail -5 $W/logs/contacts_$c.log; stop "$t contacts $c" }
  done
  stamp "$t: kNN against touching boundaries"
  table $W ${IMM[$t]} ${PART[$t]} | tee $W/logs/table.log
  p100 $W ${IMM[$t]} ${PART[$t]} | tee -a $W/logs/table.log
}

p100() { python - "$@" <<'EOF'
# per unit: T/NK per 100 partner cells from nucB (ref) and wholeB (test), and random 10% on the test labels
import glob, os, sys
import pandas as pd
W, imm, part = sys.argv[1:4]
col = f"per100|{imm}|{part}"
out = []
for d in sorted(glob.glob(os.path.join(W, "contacts_*"))):
    U = pd.read_csv(os.path.join(d, "units.csv"))
    if col not in U: print(f"{d}: no column {col}"); continue
    P = U.pivot_table(index="unit", columns="labels", values=col)
    r = pd.DataFrame({"run": os.path.basename(d), "ref_nucB": P["ref"], "test_wholeB": P["test"]})
    r["change_pct"] = 100 * (r.test_wholeB / r.ref_nucB - 1)
    if "random_10" in P: r["random10_pct"] = 100 * (P["random_10"] / P["test"] - 1)
    out.append(r.reset_index())
if out:
    T = pd.concat(out); T.to_csv(os.path.join(W, "per100_units.csv"), index=False)
    pd.set_option("display.width", 200)
    print(T.round(2).to_string(index=False))
    print(T.groupby("run")[["change_pct"] + (["random10_pct"] if "random10_pct" in T else [])].agg(["median", "min", "max"]).round(1).to_string())
EOF
}

sweep() {
  local t=$1 W=$R/rob_sweep_$1 fl pr tag
  [[ -n ${LEVEL[$t]} ]] || stop "unknown tissue $t"
  [[ -n $PILOT ]] && W=${W}_pilot
  [[ -e $W ]] && stop "$W exists; move it away"
  mkdir -p $W/logs
  for fl in ${(s:,:)FLOORS}; do for pr in ${(s:,:)PRIORS}; do
    tag=f${fl}_$pr
    stamp "$t: label, floor $fl, prior $pr"
    NOFLOOR=1 label $t $W/lab_$tag nucA,nucB,wholeB,nucleus --floor $fl --prior $pr
    claimbreak compartment compare --labels $W/lab_$tag --out $W/cmp_$tag --levels ${LEVEL[$t]} \
      --comparisons "nucB|nucA,wholeB|nucA" --tile 200 > $W/logs/cmp_$tag.log 2>&1 || { tail -5 $W/logs/cmp_$tag.log; stop "$t compare $tag" }
    summ $W/cmp_$tag $W/sum_$tag "wholeB|nucA" "nucB|nucA" $t
    claimbreak compartment contacts --labels $W/lab_$tag --out $W/contacts_$tag --level ${LEVEL[$t]} --immune ${IMM[$t]} \
      --partners ${PART[$t]} --ref nucB --test wholeB --unit-pattern "${UNIT[$t]}" --mode composition \
      > $W/logs/contacts_$tag.log 2>&1 || { tail -5 $W/logs/contacts_$tag.log; stop "$t contacts $tag" }
  done; done
  stamp "$t: sweep tables"
  table $W ${IMM[$t]} ${PART[$t]} | tee $W/logs/table.log
  p100 $W ${IMM[$t]} ${PART[$t]} | tee -a $W/logs/table.log
  python - $W ${LEVEL[$t]} "${UNIT[$t]}" "$FOCUS" <<'EOF' | tee -a $W/logs/table.log
# composition readout: T/NK labels from wholeB over T/NK labels from nucB, per unit, for every setting
import glob, os, re, sys
import pandas as pd
W, lv, unit, focus = sys.argv[1:5]
rx = re.compile(focus)
rows = []
for d in sorted(glob.glob(os.path.join(W, "lab_*"))):
    for f in sorted(glob.glob(os.path.join(d, "*.labels.csv.gz"))):
        s = os.path.basename(f).replace(".labels.csv.gz", "")
        u = re.search(unit, s); u = u.group(0) if u else s
        L = pd.read_csv(f, usecols=lambda c: c in (f"{lv}_nucB", f"{lv}_wholeB", f"{lv}_nucA"), keep_default_na=False)
        tn = [t for t in set(L[f"{lv}_nucB"]) | set(L[f"{lv}_wholeB"]) if t and rx.search(t)]
        rows.append({"setting": os.path.basename(d)[4:], "unit": u, "nucB": int(L[f"{lv}_nucB"].isin(tn).sum()),
                     "wholeB": int(L[f"{lv}_wholeB"].isin(tn).sum()), "nucA": int(L[f"{lv}_nucA"].isin(tn).sum())})
T = pd.DataFrame(rows).groupby(["setting", "unit"]).sum()
T["comp_wholeB_over_nucB"] = T.wholeB / T.nucB
T["null_nucB_over_nucA"] = T.nucB / T.nucA
T.reset_index().to_csv(os.path.join(W, "composition_units.csv"), index=False)
print("T/NK labels, wholeB over nucB (equal depth), per unit:")
print(T.comp_wholeB_over_nucB.unstack("setting").round(3).to_string())
print(T.groupby("setting")[["comp_wholeB_over_nucB", "null_nucB_over_nucA"]].agg(["median", "min", "max"]).round(3).to_string())
EOF
}

case $1 in
  check)
    python -c "import claimbreak, sys; print('claimbreak', claimbreak.__version__); sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
      || stop "claimbreak 1.7.103 or later needed"
    for t in lung gut kidney ra cosmx; do
      n=( ${ORIG[$t]}/*.labels.csv.gz ); h=( ${H5[$t]}/*.h5ad )
      print "$t: labels ${ORIG[$t]} (${#n} files); h5ad ${H5[$t]} (${#h}); reference ${REF[$t]:t} $([[ -f ${REF[$t]} ]] && print ok || print MISSING); compare labels ${CMPL[$t]} $([[ -d ${CMPL[$t]} ]] && print ok || print MISSING)"
      [[ -d ${CMPL[$t]} ]] && python - ${CMPL[$t]} ${LEVEL[$t]} ${IMM[$t]} ${PART[$t]} <<'EOF'
import glob, sys, pandas as pd
f = sorted(glob.glob(sys.argv[1] + "/*.labels.csv.gz"))[:1]
if f:
    L = pd.read_csv(f[0], nrows=20000, keep_default_na=False)
    c = f"{sys.argv[2]}_nucB"
    ty = sorted(set(L[c]) - {""}) if c in L else []
    miss = [x for x in sys.argv[3:5] if x not in ty]
    print(f"  types in {c}: {ty}" + (f"  WARNING not found: {miss}" if miss else ""))
EOF
      if [[ -n ${BND[$t]} ]]; then
        python - ${CMPL[$t]} ${BND[$t]} <<'EOF'
import glob, os, sys
from claimbreak.contacts import find_boundaries
lab, bd = sys.argv[1:3]
s = [os.path.basename(f).replace(".labels.csv.gz", "") for f in sorted(glob.glob(lab + "/*.labels.csv.gz"))]
if not os.path.isdir(bd):
    print(f"  boundaries: folder {bd} not found"); sys.exit(0)
hit = [x for x in s if find_boundaries(bd, x) is not None]
print(f"  boundaries {bd}: {len(hit)} of {len(s)} samples have a cell_boundaries.parquet" +
      (f"; e.g. missing {sorted(set(s) - set(hit))[:3]}" if len(hit) < len(s) else ""))
EOF
      fi
    done ;;
  harvest)
    DL=${CB_DOWNLOADS:-~/Downloads}; cd $R && zip -q -r $DL/cb_batchT_$(date +%Y%m%d_%H%M).zip \
      rob_*/robust_table.csv(N) rob_*/per100_units.csv(N) rob_*/composition_units.csv(N) \
      rob_*/sum_*/{sections,units,units_kept,pairs,pairs_units}.csv(N) rob_*/cmp_*/{summary,did,settings,boundary_contacts}.*(N) \
      rob_*/contacts_*/{units,classes}.csv(N) rob_*/contacts_*/meta.json(N) rob_*/lab_*/*.meta.json(N) rob_*/labels/*.meta.json(N) \
      rob_*/logs/*.log(N)
    ls -lh $DL/cb_batchT_*.zip(om[1]) ;;
  thirds|boundary|sweep)
    python -c "import claimbreak, sys; sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
      || stop "claimbreak older than 1.7.103"
    step=$1; shift
    (( $# )) || stop "name the tissues (lung gut kidney ra cosmx)"
    for t in "$@"; do $step $t; done ;;
  *) stop "usage: robust_batch.zsh check | thirds|boundary|sweep TISSUE... | harvest" ;;
esac
