#!/bin/zsh
# xp_baysor.zsh -- renal carcinoma with Baysor cells: labels, compartment test and
# protein recall on the Baysor cells and on the deposited-mask cells that Baysor keeps, in the layout of the Proseg
# runs (<W>/proseg), so the numbers scripts read both the same way:
#   <W>/baysor/labels, cmp_knn, sum_knn_wholeB, sum_knn_cytoB, protein            Baysor cells
#   <W>/baysor/labels_dep_common, cmp_dep_common, sum_dep_common_wholeB, protein_dep_common   deposited, same cells
#   <W>/baysor/masks.txt                                                          cells per segmentation
# Input: the Baysor .h5ad made on WSL (baysor_batch.sh rcc), copied to $BH. Run once per donor:
#   zsh xp_baysor.zsh                                             # donor 1 (XP_RCC, ~/claimbreak_results/xp_rcc)
#   S=XP_CCRCC W=~/claimbreak_results/xp_ccrcc zsh xp_baysor.zsh           # donor 2
# Every step skips a folder that exists; nothing is overwritten.
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
X=${X:-$D/data/xenium_protein}; S=${S:-XP_RCC}; W=${W:-$R/xp_rcc}; BH=${BH:-$X/baysor_h5ad}
REF=${REF:-$D/AIN/ref/kidney_census.h5ad}; AINLAB=${AINLAB:-$D/AIN/labels/v1}
LV=L1; B=$W/baysor
stop() { print "STOP: $*"; exit 1 }
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
python -c "import claimbreak, sys; print('claimbreak', claimbreak.__version__); sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
  || stop "claimbreak 1.7.103 or later needed"
for f in $BH/$S.h5ad $X/h5ad/$S.protein.csv.gz $W/labels/$S.labels.csv.gz $REF; do
  [[ -f $f ]] || stop "missing $f"
done
mkdir -p $B/logs
opts=$(python - $AINLAB $B/exclude_genes.txt <<'EOF'
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
EOF
) || stop "AIN label options"

if [[ ! -e $B/labels ]]; then
  stamp "label $S (Baysor cells) with $opts"
  eval claimbreak compartment label --ref $REF --ref-levels $LV=coarse --levels $LV --ref-merge "''" \
    --inputs $BH/$S.h5ad --out $B/labels --arms nucA,nucB,wholeB,cytoB,nucleus,whole $opts > $B/logs/label.log 2>&1 \
    || { tail -8 $B/logs/label.log; stop "label" }
  grep -E "$LV:|WARNING" $B/logs/label.log | cut -c1-230
fi
if [[ ! -e $B/labels_dep_common ]]; then
  mkdir -p $B/labels_dep_common
  python - $B/labels/$S.labels.csv.gz $W/labels/$S.labels.csv.gz $B/labels_dep_common/$S.labels.csv.gz <<'EOF' | tee $B/masks.txt
import sys, pandas as pd
a, d, o = sys.argv[1:4]
A = pd.read_csv(a, keep_default_na=False, dtype={"cell_id": str}, usecols=["cell_id"])
Dp = pd.read_csv(d, keep_default_na=False, dtype={"cell_id": str}, low_memory=False)
C = Dp[Dp.cell_id.isin(set(A.cell_id))]
C.to_csv(o, index=False, compression="gzip")
s = a.split("/")[-1].replace(".labels.csv.gz", "")
print(f"{s} baysor     all cells {len(A):,}")
print(f"{s} deposited  all cells {len(Dp):,}")
print(f"{s} common         cells {len(C):,}")
EOF
fi
for pair in "knn labels" "dep_common labels_dep_common"; do
  set -- ${=pair}
  [[ -e $B/cmp_$1 ]] || {
    stamp "compare $1"
    claimbreak compartment compare --labels $B/$2 --out $B/cmp_$1 --levels $LV \
      --comparisons "nucB|nucA,wholeB|nucA,cytoB|nucA" --tile 200 > $B/logs/cmp_$1.log 2>&1 || { tail -5 $B/logs/cmp_$1.log; stop "compare $1" }
  }
  for x in wholeB cytoB; do
    [[ -e $B/sum_${1}_$x ]] && continue
    claimbreak compartment summary --compare $B/cmp_$1 --out $B/sum_${1}_$x --level $LV --test "$x|nucA" \
      --unit-pattern "$S" > $B/logs/sum_${1}_$x.log 2>&1 || { tail -5 $B/logs/sum_${1}_$x.log; stop "summary $1 $x" }
    print "$1 $x:"; grep -E "^per unit, pooled|^block bootstrap|^kept ratio" -A2 $B/logs/sum_${1}_$x.log | cut -c1-200
  done
done
for pair in "protein labels" "protein_dep_common labels_dep_common"; do
  set -- ${=pair}
  [[ -e $B/$1 ]] && continue
  stamp "protein truth: $1"
  python -u $CB/examples/compartment/xenium_protein.py truth --labels $B/$2 --protein $X/h5ad/$S.protein.csv.gz \
    --sample $S --out $B/$1 --level $LV > $B/logs/$1.log 2>&1 || { tail -5 $B/logs/$1.log; stop "truth $1" }
done
python - $B <<'EOF'
import os, sys, pandas as pd
B = sys.argv[1]
for nm in ("protein", "protein_dep_common"):
    R = pd.read_csv(os.path.join(B, nm, "recall.csv"))
    R = R[(R.variant == "all") & (R.stratum == "all") & (R.protein_class == "T")].set_index("arm").recall
    if {"wholeB", "nucB"} <= set(R.index):
        print(f"{nm:20s} protein T recall nucB {R['nucB']:.3f}, wholeB {R['wholeB']:.3f}, ratio {R['wholeB'] / R['nucB']:.3f}")
for nm in ("sum_knn_wholeB", "sum_dep_common_wholeB"):
    K = pd.read_csv(os.path.join(B, nm, "units_kept.csv"))
    k = K[K.type.astype(str).str.contains("T_NK")]
    if len(k):
        r = k.iloc[0]; print(f"{nm:22s} T/NK kept ratio {r.kept_ratio:.3f} ({r.kept_ratio_lo:.3f}-{r.kept_ratio_hi:.3f})")
EOF
print "done: $B"
