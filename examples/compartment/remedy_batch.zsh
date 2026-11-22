#!/bin/zsh
# remedy_batch.zsh -- neighbor-aware labeling on the five
# compartment datasets, scored without protein (protein recall: ihc_truth.py truth on lung, ../correction/ihc_wsl.sh).
#   zsh $CB/examples/compartment/remedy_batch.zsh check
#   PILOT=1 zsh $CB/examples/compartment/remedy_batch.zsh lung            # one section first (timing, sanity)
#   caffeinate -ims zsh $CB/examples/compartment/remedy_batch.zsh lung gut kidney ra cosmx 2>&1 | tee ~/claimbreak_results/logs/batchR_$(date +%Y%m%d_%H%M).log
#   zsh $CB/examples/compartment/remedy_batch.zsh harvest                 # ~/Downloads/cb_batchR_<time>.zip
# Per tissue, into ~/claimbreak_results/rem_<tissue> (reads the existing labels' meta.json, the h5ad files and the reference):
#  (1) `claimbreak compartment label` with arms nucA, nucB, cytoB, cytoM, nucleus, whole, wholeM, wholeMX and the
#      options of the existing multinomial run; nucA/nucB must equal the existing labels (checked);
#  (2) exposure with test cytoB and with test cytoM (same counts; cytoM labels them with the neighbors'
#      contamination in the likelihood): does the T/NK border-share dose flatten?
#  (3) remedy.csv: per unit, T/NK kept (cytoB, cytoM over nucB), T/NK recall against the nucleus labels (whole,
#      wholeM, wholeMX), agreement with the nucleus, T/NK share, fitted catchment constant; precision and F1
#      per type (cytoplasm arms against nucA, independent counts; whole-cell arms against the nucleus, which wholeM
#      contains, so partly circular) in remedy_types.csv, and the T/NK border-share dose of both cytoplasm arms.
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
# RA_LAB = synovium labels folder (default gseh2_pre); with R=<new root> the synovium results of an
#   audited relabeling (e.g. RA_LAB=$D/data/RA_labels/gseh4_pre/all R=~/claimbreak_results) go to a new folder.
ARMS=${ARMS:-nucA,nucB,cytoB,cytoM,nucleus,whole,wholeM,wholeMX,wBall,wBallM}
# DECON='<extra label options>' (e.g. '--decon-w 1,0,0.1' or '--decon-fit-on auto') and TAG=<name> for variants
DECON=${DECON:-}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

typeset -A LAB H5 REF LEVCOL MERGE LEVEL UNIT
LAB[lung]=${LUNG_LABELS:-$R/bilous_v4};    H5[lung]=$D/Bilous_h5ad
LAB[cosmx]=$R/cosmx_nsclc_x171/multinomial/labels; H5[cosmx]=$D/data/cosmx_nsclc/h5ad
for t in lung cosmx; do
  REF[$t]=$D/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad; LEVCOL[$t]=L2m=Level2; MERGE[$t]='^Tu_L\d=>Tumour'; LEVEL[$t]=L2m
done
UNIT[lung]='L\d(?=_L)'; UNIT[cosmx]='Lung\d+'
LAB[gut]=$D/IBD/labels/xenium_v1;          H5[gut]=$D/IBD/h5ad;  REF[gut]=$D/IBD/ref/gut_census_v2.h5ad
LEVCOL[gut]=L1=coarse; MERGE[gut]=''; LEVEL[gut]=L1; UNIT[gut]='I0\d{3}'
LAB[kidney]=$D/AIN/labels/v1;              H5[kidney]=$D/AIN/h5ad; REF[kidney]=$D/AIN/ref/kidney_census.h5ad
LEVCOL[kidney]=L1=coarse; MERGE[kidney]=''; LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'
LAB[ra]=${RA_LAB:-$D/data/RA_labels/gseh2_pre/all};   H5[ra]=$D/data/RA_h5ad/compartment_pre; REF[ra]=$D/data/RA_ref/gse200815h2_ref.h5ad
LEVCOL[ra]=L1=coarse; MERGE[ra]=''; LEVEL[ra]=L1; UNIT[ra]='p\d{2}'
FOCUS=${FOCUS:-'(?i)(^|[^a-z])(t|nk|tnk)([^a-z]|$)'}

optpy() { python - "$@" <<'EOF'
import json, sys, glob, os
lab, out = sys.argv[1], sys.argv[2]
m = json.load(open(sorted(glob.glob(os.path.join(lab, "*.meta.json")))[0]))
o = []
lay = m.get("ref_layer", "X")
if lay == "raw": o += ["--ref-raw"]
elif lay not in ("X", "", None): o += ["--ref-layer", lay]
for k, f in (("ref_map", "--ref-map"), ("ref_subset", "--ref-subset")):
    if m.get(k): o += [f, m[k]]
if m.get("cv_group"): o += ["--cv-group", m["cv_group"]]
o += ["--min-counts", str(m.get("min_counts", 10)), "--seed", str(m.get("seed", 0)), "--floor", str(m.get("floor", 0.01)),
      "--gamma-iter", str(m.get("gamma_iter", 5))]
if m.get("markers", 0): o += ["--markers", str(m["markers"]), "--marker-min", str(m.get("marker_min", 1e-4)),
                             "--marker-min-log2", str(m.get("marker_min_log2", 1.0))]
if m.get("exclude_genes"):
    open(out, "w").write("\n".join(m["exclude_genes"]) + "\n"); o += ["--exclude-genes", out]
print(" ".join(f"'{x}'" for x in o))
print(f"existing labels: reference {os.path.basename(str(m.get('reference')))}, levels {m.get('ref_levels')}, "
      f"merge {m.get('ref_merge')!r}, arms {m.get('arms')}, seed {m.get('seed')}", file=sys.stderr)
EOF
}

same() { python - "$@" <<'EOF'
# nucA / nucB of the new run must equal the existing labels (same seed, same halves)
import glob, os, sys
import pandas as pd
old, new, lv = sys.argv[1:4]
bad = 0
for f in sorted(glob.glob(os.path.join(new, "*.labels.csv.gz"))):
    s = os.path.basename(f)
    if not os.path.exists(os.path.join(old, s)): continue
    A = pd.read_csv(os.path.join(old, s), usecols=["cell_id", f"{lv}_nucA", f"{lv}_nucB"], keep_default_na=False, dtype=str)
    B = pd.read_csv(f, usecols=["cell_id", f"{lv}_nucA", f"{lv}_nucB"], keep_default_na=False, dtype=str).set_index("cell_id").reindex(A.cell_id)
    r = [(A[f"{lv}_{a}"].to_numpy() == B[f"{lv}_{a}"].fillna("").to_numpy()).mean() for a in ("nucA", "nucB")]
    print(f"  {s}: nucA {r[0]:.4f}, nucB {r[1]:.4f} identical")
    bad += min(r) < 0.999
sys.exit(1 if bad else 0)
EOF
}

remedy() { python - "$@" <<'EOF'
# per unit: T/NK kept (cytoB, cytoM over nucB on cells whose nucA says T/NK), T/NK recall against the nucleus arm,
# agreement with the nucleus arm, T/NK share per arm, catchment constant
import glob, json, os, re, sys
import numpy as np, pandas as pd
lab, lv, unit, out, focus = sys.argv[1:6]
rx = re.compile(focus)
rows, trows = [], []
for f in sorted(glob.glob(os.path.join(lab, "*.labels.csv.gz"))):
    s = os.path.basename(f).replace(".labels.csv.gz", "")
    u = re.search(unit, s); u = u.group(0) if u else s
    L = pd.read_csv(f, keep_default_na=False, low_memory=False)
    arms = [c[len(lv) + 1:] for c in L.columns if c.startswith(lv + "_") and not c.endswith(("_second", "_margin", "_psi"))]
    types = sorted(set().union(*[set(L[f"{lv}_{a}"]) for a in arms]) - {""})
    tn = [t for t in types if rx.search(t)]
    meta = json.load(open(f.replace(".labels.csv.gz", ".meta.json")))
    r = {"sample": s, "unit": u, "cells": len(L), "tnk_types": "|".join(tn)}
    foc = L[f"{lv}_nucA"].isin(tn)
    for a in ("nucB", "cytoB", "cytoM"):
        if f"{lv}_{a}" in L:
            r[f"tnk_{a}"] = int((foc & L[f"{lv}_{a}"].isin(tn)).sum())
    nuc = L[f"{lv}_nucleus"]
    for a in ("nucleus", "whole", "wholeM", "wholeMX", "cytoB", "cytoM", "nucB"):
        if f"{lv}_{a}" not in L: continue
        x = L[f"{lv}_{a}"]
        ok = (x != "") & (nuc != "")
        r[f"agree_{a}"] = round(float((x[ok] == nuc[ok]).mean()), 4)
        r[f"tnk_share_{a}"] = round(float(x[x != ""].isin(tn).mean()), 4)
        t = ok & nuc.isin(tn)
        r[f"tnk_in_nuc_{a}"] = int(t.sum()); r[f"tnk_kept_nuc_{a}"] = int((t & x.isin(tn)).sum())
        if f"{lv}_{a}_psi" in L:
            r[f"psi_med_{a}"] = round(float(pd.to_numeric(L[f"{lv}_{a}_psi"], errors="coerce").median()), 3)
    for k, v in (meta.get("decon_kappa") or {}).items():
        r[f"kappa_{k.split('_', 1)[1]}"] = v
    for k, v in (meta.get("decon_beta") or {}).items():
        r[f"beta_{k.split('_', 1)[1]}"] = v
    rows.append(r)
    for a, refc in (("nucB", "nucA"), ("cytoB", "nucA"), ("cytoM", "nucA"), ("wBall", "nucA"), ("wBallM", "nucA"),
                    ("whole", "nucleus"), ("wholeM", "nucleus"), ("wholeMX", "nucleus")):
        if f"{lv}_{a}" not in L or f"{lv}_{refc}" not in L: continue
        x, y = L[f"{lv}_{a}"], L[f"{lv}_{refc}"]
        ok = (x != "") & (y != "")
        for t in types:
            trows.append({"unit": u, "sample": s, "arm": a, "ref": refc, "type": t, "ref_n": int((ok & (y == t)).sum()),
                          "called": int((ok & (x == t)).sum()), "tp": int((ok & (x == t) & (y == t)).sum())})
S = pd.DataFrame(rows)
TT = pd.DataFrame(trows).groupby(["arm", "ref", "type"])[["ref_n", "called", "tp"]].sum().reset_index()
TT["recall"] = TT.tp / TT.ref_n.where(TT.ref_n > 0); TT["precision"] = TT.tp / TT.called.where(TT.called > 0)
TT["f1"] = 2 * TT.recall * TT.precision / (TT.recall + TT.precision)
TT.round(4).to_csv(os.path.join(out, "remedy_types.csv"), index=False)
S.to_csv(os.path.join(out, "remedy_sections.csv"), index=False)
num = [c for c in S.columns if c.startswith(("tnk_nucB", "tnk_cyto", "tnk_in_nuc", "tnk_kept_nuc"))]
U = S.groupby("unit")[num].sum()
res = pd.DataFrame(index=U.index)
for a in ("cytoB", "cytoM"):
    if f"tnk_{a}" in U: res[f"kept_{a}"] = U[f"tnk_{a}"] / U["tnk_nucB"].where(U["tnk_nucB"] > 0)
for a in ("whole", "wholeM", "wholeMX", "cytoB", "cytoM", "nucB"):
    if f"tnk_in_nuc_{a}" in U: res[f"recall_{a}"] = U[f"tnk_kept_nuc_{a}"] / U[f"tnk_in_nuc_{a}"].where(U[f"tnk_in_nuc_{a}"] > 0)
for c in [c for c in S.columns if c.startswith(("agree_", "tnk_share_", "kappa_", "beta_", "psi_med_"))]:
    res[c] = S.groupby("unit")[c].median()
res.round(4).to_csv(os.path.join(out, "remedy.csv"))
pd.set_option("display.width", 250)
print("T/NK types:", sorted(set("|".join(S.tnk_types).split("|")) - {""}))
print(res.round(3).to_string())
print("\nmedian over units:\n" + res.median().round(3).to_string())
print("\nper type, pooled over sections: recall / precision / F1 (cytoplasm arms and nucB against nucA; whole-cell arms "
      "against the nucleus, partly circular for wholeM)")
P = TT.pivot_table(index="type", columns="arm", values=["recall", "precision", "f1"])
print(P.round(2).to_string())
print("\nmacro F1 by arm:", TT.groupby("arm").f1.mean().round(3).to_dict())
if {"wBall", "wBallM", "nucB"} <= set(TT.arm):
    m = TT.groupby("arm").f1.mean()
    print(f"fair test against nucA (no shared counts): nucB {m['nucB']:.3f}, wBall {m['wBall']:.3f}, wBallM {m['wBallM']:.3f}"
          f" -> wBallM {'beats' if m['wBallM'] > max(m['nucB'], m['wBall']) else 'does not beat'} both")
EOF
}

run() {
  local t=$1 W=$R/rem_$1 opts ins
  [[ -n ${LAB[$t]} ]] || stop "unknown tissue $t (lung, gut, kidney, ra, cosmx)"
  [[ -n $PILOT ]] && W=${W}_pilot
  [[ -n $TAG ]] && W=${W}_$TAG
  [[ -e $W ]] && stop "$W exists; move it away"
  [[ -d ${LAB[$t]} && -d ${H5[$t]} && -f ${REF[$t]} ]] || stop "$t: missing ${LAB[$t]}, ${H5[$t]} or ${REF[$t]}"
  mkdir -p $W/logs
  opts=$(optpy ${LAB[$t]} $W/exclude_genes.txt) || stop "$t: no meta.json in ${LAB[$t]}"
  ins=()
  for f in ${LAB[$t]}/*.labels.csv.gz; do s=${f:t}; s=${s%.labels.csv.gz}; [[ -f ${H5[$t]}/$s.h5ad ]] && ins+=(${H5[$t]}/$s.h5ad); done
  [[ -n $PILOT ]] && ins=( $ins[1] )
  print "$t: ${#ins} samples; options $opts"
  (( ${#ins} )) || stop "$t: no h5ad for the labelled samples"
  stamp "$t: label $ARMS"
  eval claimbreak compartment label --ref ${REF[$t]} --ref-levels ${LEVCOL[$t]} --ref-merge "'${MERGE[$t]}'" \
    --inputs ${(q)ins} --out $W/labels --arms $ARMS $opts ${=DECON} 2>&1 | tee $W/logs/label.log | grep -v "^  wrote" | cut -c1-240
  stamp "$t: nuclear halves against the existing labels"
  same ${LAB[$t]} $W/labels ${LEVEL[$t]} || stop "$t: nucA/nucB differ from the existing run"
  for x in cytoB cytoM; do
    stamp "$t: exposure, test $x"
    claimbreak compartment exposure --labels $W/labels --h5ad-dir ${H5[$t]} --out $W/exp_$x --levels ${LEVEL[$t]} \
      --unit-pattern "${UNIT[$t]}" --focus "$FOCUS" --test $x --gaps 2,6,10,15 --contact-gap 10 > $W/logs/exp_$x.log 2>&1 \
      || { tail -5 $W/logs/exp_$x.log; stop "$t exposure $x" }
  done
  stamp "$t: T/NK kept by border share (contact 10 um), cytoB vs cytoM"
  python - $W <<'EOF'
import sys, pandas as pd
W = sys.argv[1]
t = {}
for x in ("cytoB", "cytoM"):
    U = pd.read_csv(f"{W}/exp_{x}/units.csv")
    U = U[U.type.astype(str).str.contains(r"(?i)(?:^|[^a-z])(?:t|nk|tnk)(?:[^a-z]|$)") & (U.scheme == "border")]
    g = U.groupby("bin")[["kept_test", "kept_null"]].sum()
    t[x] = g.kept_test / g.kept_null
    t[f"cells"] = U.groupby("bin").cells.sum()
order = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1)", "1", "none"]
print(pd.DataFrame(t).reindex([o for o in order if o in t["cells"].index]).round(3).to_string())
EOF
  stamp "$t: remedy table"
  remedy $W/labels ${LEVEL[$t]} "${UNIT[$t]}" $W "$FOCUS" 2>&1 | tee $W/logs/remedy.log
}

case $1 in
  check)
    python -c "import claimbreak, sys; print('claimbreak', claimbreak.__version__); sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
      || stop "claimbreak 1.7.103 or later needed"
    for t in lung gut kidney ra cosmx; do
      n=( ${LAB[$t]}/*.labels.csv.gz )
      print "$t: labels ${LAB[$t]} (${#n} files); h5ad ${H5[$t]} $([[ -d ${H5[$t]} ]] && print ok || print MISSING); reference ${REF[$t]:t} $([[ -f ${REF[$t]} ]] && print ok || print MISSING)"
    done ;;
  harvest)
    DL=${CB_DOWNLOADS:-~/Downloads}; cd $R && zip -q -r $DL/cb_batchR_$(date +%Y%m%d_%H%M).zip rem_*/remedy*.csv rem_*/logs/*.log \
      rem_*/exp_*/{units,dose,tissue,dest_units,dest_fit,dest_region_fit}.csv(N) rem_*/exp_*/notes.txt(N) rem_*/labels/*.meta.json(N)
    ls -lh $DL/cb_batchR_*.zip(om[1]) ;;
  *)
    python -c "import claimbreak, sys; sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
      || stop "claimbreak older than 1.7.103"
    for t in "$@"; do run $t; done ;;
esac
