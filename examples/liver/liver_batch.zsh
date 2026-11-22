#!/bin/zsh
# liver_batch.zsh -- the compartment test on the MASLD liver CosMx data (Boesch et al.
# 2026, Nat Genet; GSE312698; 1,000-plex Universal panel + 14 custom genes; 4 slides, 10 samples; 11 usable pieces
# after liver_pieces.py qc/filter, in $L/h5ad_qc). Reference: Census liver, normal (liver_reference.py).
#   zsh $CB/examples/liver/liver_batch.zsh check
#   caffeinate -ims zsh $CB/examples/liver/liver_batch.zsh all 2>&1 | tee ~/claimbreak_results/logs/liver_$(date +%Y%m%d_%H%M).log
#   zsh $CB/examples/liver/liver_batch.zsh grid               # settings against label collapse + marker check
#   zsh $CB/examples/liver/liver_batch.zsh gtest              # test on the pieces passing the label gate
#   zsh $CB/examples/liver/liver_batch.zsh harvest            # ~/Downloads/cb_liver_run_<time>.zip
# steps (each refuses to overwrite its output; `all` = pilot drivers label test stains harvest):
#   pilot    label every piece without exclusions (agreement of the arms; input of drivers)
#   drivers  label_drivers.py per piece: genes detected > 2x their highest share in any reference type; genes
#            over-detected in at least half of the pieces -> $W/exclude_half.txt (as the CosMx lung data)
#   label    label every piece with exclude_half.txt; arms nucA nucB wholeB cytoB nucleus whole
#   test     compare (contact: 6 nearest within 30 um) + summary, unit = piece; contacts per 100 hepatocytes
#            (nucB vs wholeB) for T_NK, B, Plasma, Myeloid
#   stains   per piece: is each stain a property of the cell (Spearman of the stain with the cell's own marker
#            transcripts vs its nearest neighbor's)? then xenium_protein.py truth with the CosMx stains (CD45 = any
#            immune label, CD68 = myeloid; contact strata at CD68+ and PanCK+ cells), pooled over pieces
# Reads $L/h5ad_qc and the reference only; writes $W (default ~/claimbreak_results/liver_mash).
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}
L=${L:-$D/data/liver_mash}; H=${H:-$L/h5ad_qc}; REF=${REF:-$L/ref/liver_census.h5ad}
W=${W:-~/claimbreak_results/liver_mash}; LV=L1
UNIT=${UNIT:-'GSM\d+_Leuven_\d_p\d+'}
LBL=(--ref $REF --ref-levels L1=coarse --ref-merge '' --levels $LV --cv-group donor_id --min-counts 10 --seed 0
     --floor 0.01 --arms nucA,nucB,wholeB,cytoB,nucleus,whole)
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
ins=( $H/*_p*.h5ad(N) )

check() {
  python -c "import claimbreak, sys; print('claimbreak', claimbreak.__version__); sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
    || stop "claimbreak 1.7.103 or later needed"
  (( ${#ins} )) && print "${#ins} pieces in $H" || stop "no pieces in $H (liver_pieces.py filter)"
  [[ -f $REF ]] && print "reference ok: $REF" || stop "reference missing: $REF"
  df -h ${W:h} | tail -1
}

agree() {  # agreement of the arms, from the label meta files
  python - $1 <<'EOF'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.meta.json"))):
    a = json.load(open(f)).get("agreement", {})
    rows.append((os.path.basename(f).split(".")[0], {k.split(":")[1]: round(v[0], 3) for k, v in a.items()
                                                    if k.split(":")[1] in ("nucB==nucA", "wholeB==nucA", "cytoB==nucA")}))
for s, d in rows:
    print(f"  {s:28s} {d}")
EOF
}

pilot() {
  [[ -e $W/pilot ]] && { print "exists: $W/pilot"; return }
  mkdir -p $W/logs
  stamp "pilot labels (${#ins} pieces, no exclusions)"
  claimbreak compartment label "${LBL[@]}" --ref-cv --inputs $ins --out $W/pilot > $W/logs/pilot.log 2>&1 \
    || { tail -5 $W/logs/pilot.log; stop "pilot" }
  grep -m1 -A3 "overall accuracy" $W/logs/pilot.log; grep -m1 -A3 "mean per-type accuracy" $W/logs/pilot.log
  agree $W/pilot
}

drivers() {
  [[ -f $W/exclude_half.txt ]] && { print "exists: $W/exclude_half.txt"; return }
  local f s
  for f in $ins; do
    s=${f:t:r}
    [[ -f $W/drivers/$s/exclude_overdetected.txt ]] && continue
    python $CB/examples/ibd/label_drivers.py --ref $REF --ref-levels L1=coarse --h5ad $f --labels $W/pilot \
      --level $LV --out $W/drivers/$s > $W/logs/drivers_$s.log 2>&1 || { tail -3 $W/logs/drivers_$s.log; stop "drivers $s" }
    print "  $s: $(grep -vc '^#' $W/drivers/$s/exclude_overdetected.txt) genes over-detected"
  done
  local n=${#ins}
  cat $W/drivers/*/exclude_overdetected.txt | grep -v '^#' | sort | uniq -c | awk -v h=$(( (n + 1) / 2 )) '$1>=h{print $2}' \
    > $W/exclude_half.txt
  print "over-detected in >= half of the $n pieces: $(wc -l < $W/exclude_half.txt | tr -d ' ') genes -> $W/exclude_half.txt"
  head -40 $W/exclude_half.txt | tr '\n' ' '; print
}

label() {
  [[ -e $W/labels ]] && { print "exists: $W/labels"; return }
  [[ -f $W/exclude_half.txt ]] || stop "run drivers first"
  stamp "labels with exclude_half.txt"
  claimbreak compartment label "${LBL[@]}" --exclude-genes $W/exclude_half.txt --inputs $ins --out $W/labels \
    > $W/logs/label.log 2>&1 || { tail -5 $W/logs/label.log; stop "label" }
  agree $W/labels
  python - $W/labels <<'EOF'
import glob, os, sys, pandas as pd
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.labels.csv.gz"))):
    L = pd.read_csv(f, usecols=["L1_nucleus", "L1_whole"], keep_default_na=False)
    rows.append(L.L1_whole.value_counts(normalize=True).rename(os.path.basename(f).split(".")[0]))
print("composition (whole-cell labels):\n" + pd.concat(rows, axis=1).fillna(0).round(3).to_string())
EOF
}

test_() {
  [[ -e $W/cmp_knn ]] && { print "exists: $W/cmp_knn"; return }
  local x
  stamp "compare, contact knn"
  claimbreak compartment compare --labels $W/labels --out $W/cmp_knn --levels $LV --contact knn \
    --comparisons "nucB|nucA,wholeB|nucA,cytoB|nucA" --tile 200 > $W/logs/cmp_knn.log 2>&1 \
    || { tail -5 $W/logs/cmp_knn.log; stop "compare" }
  for x in wholeB cytoB; do
    claimbreak compartment summary --compare $W/cmp_knn --out $W/sum_knn_$x --level $LV --test "$x|nucA" \
      --unit-pattern "$UNIT" > $W/logs/sum_knn_$x.log 2>&1 || { tail -5 $W/logs/sum_knn_$x.log; stop "summary" }
    print -r -- "-- $x|nucA"; grep -E "^sections used|^per unit, pooled|^block bootstrap" -A2 $W/logs/sum_knn_$x.log | cut -c1-200
  done
  claimbreak compartment contacts --labels $W/labels --out $W/contacts_knn --level $LV --immune T_NK,B,Plasma,Myeloid \
    --partners Hepatocyte --ref nucB --test wholeB --unit-pattern "$UNIT" --mode composition --contact knn \
    > $W/logs/contacts_knn.log 2>&1 || tail -5 $W/logs/contacts_knn.log
  python - $W <<'EOF'
import os, sys, pandas as pd
f = os.path.join(sys.argv[1], "contacts_knn", "units.csv")
if os.path.exists(f):
    U = pd.read_csv(f); U = U[U.level == "unit"]; cols = [x for x in U.columns if x.startswith("per100|")]
    print("contacts per 100 hepatocytes (ref = nucB, test = wholeB, random_k = random relabelling controls):\n"
          + U.set_index(["unit", "labels"])[cols].round(2).to_string())
EOF
}

stains() {
  [[ -e $W/stains ]] && { print "exists: $W/stains"; return }
  mkdir -p $W/stains/in
  stamp "stains: property of the cell?"
  python - $H $W/stains <<'EOF' 2>&1 | tee $W/stains/stain_check.log
import glob, os, sys
import anndata as ad, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
H, O = sys.argv[1:3]
TX = {"CD45": ["PTPRC"], "CD68": ["CD68"], "PanCK": ["KRT8", "KRT18", "KRT19", "KRT7"]}
rows = []
for f in sorted(glob.glob(os.path.join(H, "*_p*.h5ad"))):
    s = os.path.basename(f)[:-5]
    A = ad.read_h5ad(f)
    P = pd.DataFrame({"cell_id": A.obs_names.astype(str)})
    xy = A.obs[["x", "y"]].to_numpy(float)
    nn = cKDTree(xy).query(xy, k=2)[1][:, 1]
    for st, genes in TX.items():
        c = f"Mean.{st}"
        if c not in A.obs:
            continue
        v = A.obs[c].to_numpy(float)
        P[st] = v
        g = [x for x in genes if x in A.var_names]
        if not g:
            continue
        t = np.asarray(A[:, g].X.sum(1)).ravel()
        rows.append(dict(piece=s, stain=st, genes="+".join(g), cells=A.n_obs,
                         rho_own=spearmanr(v, t)[0], rho_neighbour=spearmanr(v, t[nn])[0],
                         rho_own_nucleus=spearmanr(v, np.asarray(A[:, g].layers["nucleus"].sum(1)).ravel())[0]))
    P.to_csv(os.path.join(O, "in", f"{s}.protein.csv.gz"), index=False)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(O, "stain_check.csv"), index=False)
print(T.round(3).to_string(index=False))
print("\nmedian over pieces:\n" + T.groupby("stain")[["rho_own", "rho_own_nucleus", "rho_neighbour"]].median().round(3).to_string())
print("(a stain describes the cell only if rho_own clearly exceeds rho_neighbour; lung CosMx CD3: 0.01)")
EOF
  local f s
  for f in $ins; do
    s=${f:t:r}
    print -r -- "-- truth $s"
    python -u $CB/examples/compartment/xenium_protein.py truth --labels $W/labels --protein $W/stains/in/$s.protein.csv.gz \
      --sample $s --out $W/stains/$s --level $LV --partners CD68,PanCK --block-draws ${DRAWS:-200} \
      > $W/logs/truth_$s.log 2>&1 || { tail -3 $W/logs/truth_$s.log; print "  (truth failed for $s)" }
    grep -A8 "^markers" $W/logs/truth_$s.log | cut -c1-150
  done
  python - $W/stains <<'EOF' 2>&1 | tee $W/stains/pooled.log
import glob, os, sys, pandas as pd
O = sys.argv[1]
R = []
for f in sorted(glob.glob(os.path.join(O, "*", "recall.csv"))):
    r = pd.read_csv(f); r["piece"] = os.path.basename(os.path.dirname(f)); R.append(r)
if not R:
    sys.exit("no recall.csv")
R = pd.concat(R)
R = R[(R.variant == "all") & R.n.gt(0)]
R["hits"] = R.recall * R.n
P = R.groupby(["protein_class", "stratum", "arm"])[["hits", "n"]].sum()
P["recall"] = P.hits / P.n
T = P.recall.unstack("arm").round(3)
T["n_nucB"] = P.n.xs("nucB", level="arm") if "nucB" in P.index.get_level_values("arm") else pd.NA
if {"nucB", "wholeB"} <= set(T.columns):
    T["loss_wholeB_vs_nucB"] = (1 - T.wholeB / T.nucB).round(3)
T.to_csv(os.path.join(O, "recall_pooled.csv"))
print("recall of stain-positive cells by the expected label, pooled over pieces:\n" + T.to_string())
M = pd.concat([pd.read_csv(f).assign(piece=os.path.basename(os.path.dirname(f)))
               for f in glob.glob(os.path.join(O, "*", "markers.csv"))])
print("\nstain separation (Ashman D; >= 2 used), per piece:\n" +
      M.pivot_table(index="piece", columns="marker", values="ashman_d").round(2).to_string())
EOF
}

grid() {   # classifier settings against label collapse (7 of 11 pieces collapsed with the defaults)
  [[ -f $W/exclude_half.txt ]] || stop "run drivers first"
  mkdir -p $W/grid $W/logs
  local -A V=( g1 "--gamma-iter 0" g2 "--prior fit" g3 "--gamma-iter 0 --prior fit" g4 "--markers 30"
               g5 "--markers 30 --gamma-iter 0" g6 "--markers 30 --prior fit" )
  local g
  for g in ${(ok)V}; do
    [[ -e $W/grid/$g ]] && { print "exists: $W/grid/$g"; continue }
    stamp "grid $g: ${V[$g]}"
    claimbreak compartment label "${LBL[@]}" ${=V[$g]} --arms nucA,nucB,wholeB,nucleus,whole \
      --exclude-genes $W/exclude_half.txt --inputs $ins --out $W/grid/$g > $W/logs/grid_$g.log 2>&1 \
      || { tail -5 $W/logs/grid_$g.log; print "  grid $g failed"; continue }
    print -r -- "${V[$g]}" > $W/grid/$g/SETTINGS.txt
    agree $W/grid/$g
  done
  stamp "label check (reference-free marker calls)"
  local runs=$W/labels; for g in ${(ok)V}; do [[ -d $W/grid/$g ]] && runs+=,$W/grid/$g; done
  python $CB/examples/liver/liver_label_check.py --h5ad $H --runs $runs --out $W/grid 2>&1 | tee $W/grid/label_check.log
}

gtest() {  # the compartment test on the pieces that pass the label gate
  # gate: the two nuclear halves agree in >= GATE of cells (default 0.6, as for the CosMx lung data) with labels from
  # the run GRUN (default g5: 30 markers per type, no platform factor, among the settings without collapse, the
  # one with the highest agreement with the marker call). Runs compare + summary + contacts for GRUN and, as sensitivity, g6 and
  # the default labels, on the same pieces.
  local gate=${GATE:-0.6} grun=${GRUN:-g5} r f s
  [[ -d $W/grid/$grun ]] || stop "run grid first"
  python - $W/grid/$grun $gate $W/gate.csv <<'EOF2'
import glob, json, os, sys
d, gate, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.meta.json"))):
    m = json.load(open(f)); a = m["agreement"]
    rows.append((os.path.basename(f).split(".")[0], a["L1:nucB==nucA"][0], a["L1:wholeB==nucA"][0]))
with open(out, "w") as fh:
    fh.write("piece,nucB_nucA,wholeB_nucA,pass\n")
    for s, x, y in rows:
        fh.write(f"{s},{x:.4f},{y:.4f},{x >= gate}\n")
        print(f"  {s:28s} nucB==nucA {x:.3f}  {'pass' if x >= gate else 'fail'}")
EOF2
  local pass=( ${(f)"$(awk -F, 'NR>1 && $4=="True"{print $1}' $W/gate.csv)"} )
  print "gate ${gate}: ${#pass} of $(( $(wc -l < $W/gate.csv) - 1 )) pieces pass: $pass"
  (( ${#pass} )) || stop "no piece passes the gate"
  for r in $grun g6 labels; do
    local src=$W/grid/$r; [[ $r == labels ]] && src=$W/labels
    local G=$W/gt_$r
    [[ -d $src ]] || { print "no $src; skipped"; continue }
    [[ -e $G ]] && { print "exists: $G"; continue }
    mkdir -p $G/labels
    for s in $pass; do for f in $src/$s.*(N); do ln -s $f $G/labels/${f:t}; done; done
    stamp "gated test: $r (${#pass} pieces)"
    claimbreak compartment compare --labels $G/labels --out $G/cmp_knn --levels $LV --contact knn \
      --comparisons "nucB|nucA,wholeB|nucA" --tile 200 > $W/logs/gt_${r}_cmp.log 2>&1 \
      || { tail -5 $W/logs/gt_${r}_cmp.log; print "  compare failed"; continue }
    claimbreak compartment summary --compare $G/cmp_knn --out $G/sum_knn_wholeB --level $LV --test "wholeB|nucA" \
      --unit-pattern "$UNIT" > $W/logs/gt_${r}_sum.log 2>&1 || { tail -5 $W/logs/gt_${r}_sum.log; continue }
    grep -E "^sections used|^per unit, pooled|^block bootstrap" -A13 $W/logs/gt_${r}_sum.log | cut -c1-230
    claimbreak compartment contacts --labels $G/labels --out $G/contacts_knn --level $LV --immune T_NK,B,Plasma,Myeloid \
      --partners Hepatocyte --ref nucB --test wholeB --unit-pattern "$UNIT" --mode composition --contact knn \
      > $W/logs/gt_${r}_contacts.log 2>&1 || tail -3 $W/logs/gt_${r}_contacts.log
    python - $G <<'EOF2'
import os, sys, pandas as pd
G = sys.argv[1]
f = os.path.join(G, "sum_knn_wholeB", "units_kept.csv")
if os.path.exists(f):
    K = pd.read_csv(f); print("kept (wholeB over null), per unit and type:\n" + K.round(3).to_string(index=False)[:3000])
f = os.path.join(G, "contacts_knn", "units.csv")
if os.path.exists(f):
    U = pd.read_csv(f); U = U[U.level == "unit"]; cols = [x for x in U.columns if x.startswith("per100|")]
    print("contacts per 100 hepatocytes:\n" + U.set_index(["unit", "labels"])[cols].round(2).to_string())
EOF2
  done
}

harvest() {
  local Z=~/Downloads/cb_liver_run_$(date +%Y%m%d_%H%M).zip; mkdir -p ~/Downloads
  (cd ${W:h} && zip -q -r $Z ${W:t} -x '*.labels.csv.gz' '*/stains/in/*' '*.h5ad')
  local lg=( ~/claimbreak_results/logs/liver_*.log(N) ); (( ${#lg} )) && zip -q -j $Z $lg
  ls -lh $Z
}

case $1 in
  check) check ;;
  pilot) pilot ;;
  drivers) drivers ;;
  label) label ;;
  test) test_ ;;
  stains) stains ;;
  grid) grid ;;
  gtest) gtest ;;
  harvest) harvest ;;
  all) check && pilot && drivers && label && test_ && stains && harvest ;;
  *) stop "usage: liver_batch.zsh check|pilot|drivers|label|test|stains|grid|gtest|harvest|all" ;;
esac
