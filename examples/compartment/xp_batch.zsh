#!/bin/zsh
# xp_batch.zsh -- Xenium Protein, 10x FFPE human renal cell carcinoma (XOA 4.0; one
# section; 27 proteins on the same cells). Protein replication of the T-cell loss on an independent dataset.
#   zsh $CB/examples/compartment/xp_batch.zsh check
#   caffeinate -ims zsh $CB/examples/compartment/xp_batch.zsh all 2>&1 | tee ~/claimbreak_results/logs/xp_rcc_$(date +%Y%m%d_%H%M).log
#   (steps: prepare label test truth harvest; each refuses to overwrite its output)
# prepare  links with the sample prefix to the unpacked bundle (originals untouched), compartment_to_h5ad.py
#          (whole, nucleus, cyto counts), xenium_protein.py extract (protein per cell)
# label    claimbreak compartment label against the kidney reference (Census kidney, L1 coarse) with the options of
#          the AIN kidney labels; arms nucA, nucB, wholeB, cytoB, nucleus, whole
# test     compare + summary (contact: 6 nearest within 30 um, and shared segmentation boundaries); contacts per 100
#          PT cells (nucB vs wholeB) for T_NK, B, Myeloid
# truth    xenium_protein.py truth: recall of protein T (CD3E), CD8, CD4, B (CD20), myeloid (CD68) cells by each arm,
#          overall and at PanCK+ contact; precision; fates
# remedy   the neighbor-aware arms scored by protein on every class:
#          label again into rem_labels with arms $RARMS and the options of `label` (same seed, so nucA and nucB must
#          equal labels/: checked), then truth into protein_rem with recall, precision and block intervals of the
#          losses nucB->wholeB, nucB->wBall, nucB->wBallM, nucleus->whole, nucleus->wholeM and of the share of the
#          wBall->nucB gap that wBallM closes. Not part of `all`; run after `label`:
#            caffeinate -ims zsh $CB/examples/compartment/xp_batch.zsh remedy 2>&1 | tee ~/claimbreak_results/logs/xp_remedy_$(date +%Y%m%d_%H%M).log
# remsens  the remedy for each contamination weight set in WSETS (default "1,0,0 1,0.25,0.05 1,1,0.2"),
#          into rem_labels_w*/protein_rem_w*, and remedy_weights.csv (gap closed and recalls per class)
# border   protein T, CD8, CD4 and B loss (wholeB vs nucB) by the CD45-negative share of
#          the polygon perimeter, into protein_border (xenium_protein.py border; polygons from links/); the
#          share over the whole outline by default, into protein_border_outline (BBASE=touched: the touched part only)
# groups   a published labeling of the same cells scored against protein, into groups_pub; GROUPS=<csv with
#          cell_id, group> (default: the first *_cell_groups.csv in $X)
# gborder  the published labels and the claimbreak arms on the same cells, recall by the non-immune share of
#          the polygon outline (touched part and whole outline), into groups_border
# strict   truth with lineage-consistent protein classes, posterior 0.5 and 0.9, into protein_strict_p05 and
#          protein_strict_p09
# Both are separate steps (not in `all`); run after `label`. Another section: X, OUTS, S, W (one export per line).
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
X=${X:-$D/data/xenium_protein}; OUTS=${OUTS:-$X/rcc_outs}; S=${S:-XP_RCC}; W=${W:-$R/xp_rcc}
REF=${REF:-$D/AIN/ref/kidney_census.h5ad}; AINLAB=${AINLAB:-$D/AIN/labels/v1}
LV=L1
RARMS=${RARMS:-nucA,nucB,wholeB,cytoB,cytoM,nucleus,whole,wholeM,wBall,wBallM}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
bundle() { local t=( $OUTS/transcripts.parquet(N) $OUTS/*/transcripts.parquet(N) $OUTS/*/*/transcripts.parquet(N) ); (( ${#t} )) && print ${t[1]:h} }

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
EOF
}

check() {
  python -c "import claimbreak, sys; print('claimbreak', claimbreak.__version__); sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
    || stop "claimbreak 1.7.103 or later needed"
  local b; b=$(bundle)
  [[ -n $b ]] && print "bundle: $b" || stop "no transcripts.parquet under $OUTS (unpack the bundle first)"
  for f in transcripts.parquet cells.parquet cell_feature_matrix.h5 cell_boundaries.parquet nucleus_boundaries.parquet; do
    [[ -f $b/$f ]] && print "  ok      $f ($(du -h $b/$f | cut -f1))" || { print "  MISSING $f"; stop "bundle incomplete" }
  done
  [[ -f $REF ]] && print "reference ok: $REF" || stop "reference missing: $REF"
  ls $AINLAB/*.meta.json >/dev/null 2>&1 && print "AIN label options: $AINLAB" || stop "no AIN labels meta in $AINLAB"
  python - $b/cell_feature_matrix.h5 <<'EOF'
import sys, h5py, collections
with h5py.File(sys.argv[1]) as f:
    ft = f["matrix/features"]; t = [x.decode() for x in ft["feature_type"][:]]; n = [x.decode() for x in ft["name"][:]]
print("features:", dict(collections.Counter(t)))
print("proteins:", ", ".join(x for x, y in zip(n, t) if y == "Protein Expression"))
EOF
}

prepare() {
  local b; b=$(bundle); [[ -n $b ]] || stop "no bundle"
  mkdir -p $X/links $X/h5ad $X/tmp
  for f in transcripts.parquet cells.parquet cell_feature_matrix.h5 cell_boundaries.parquet nucleus_boundaries.parquet; do
    ln -sf $b/$f $X/links/${S}_$f
  done
  [[ -f $X/h5ad/$S.h5ad ]] && print "exists: $X/h5ad/$S.h5ad" || {
    stamp "convert"
    python -u $CB/examples/compartment/compartment_to_h5ad.py --platform xenium --raw $X/links --sample $S \
      --out $X/h5ad --tmp $X/tmp || stop "convert"; }
  [[ -f $X/h5ad/$S.protein.csv.gz ]] && print "exists: protein table" || {
    stamp "protein"
    python -u $CB/examples/compartment/xenium_protein.py extract --outs $b --sample $S --out $X/h5ad || stop "protein"; }
}

label() {
  [[ -e $W/labels ]] && stop "$W/labels exists; move it away"
  mkdir -p $W/logs
  local opts; opts=$(optpy $AINLAB $W/exclude_genes.txt) || stop "AIN options"
  stamp "label with $opts"
  eval claimbreak compartment label --ref $REF --ref-levels $LV=coarse --levels $LV --ref-merge "''" \
    --inputs $X/h5ad/$S.h5ad --out $W/labels --arms nucA,nucB,wholeB,cytoB,nucleus,whole $opts 2>&1 \
    | tee $W/logs/label.log | grep -v "^  wrote" | cut -c1-240
  [[ -f $W/labels/$S.labels.csv.gz ]] || stop "labelling wrote nothing"
}

test_() {
  mkdir -p $W/logs
  local c ex x
  for c in knn boundary; do
    [[ -e $W/cmp_$c ]] && { print "exists: $W/cmp_$c"; continue }
    ex=(); [[ $c == boundary ]] && ex=(--boundaries $X/links --tol 1)
    stamp "compare, contact $c"
    claimbreak compartment compare --labels $W/labels --out $W/cmp_$c --levels $LV --contact $c $ex \
      --comparisons "nucB|nucA,wholeB|nucA,cytoB|nucA" --tile 200 > $W/logs/cmp_$c.log 2>&1 || { tail -5 $W/logs/cmp_$c.log; stop "compare $c" }
    for x in wholeB cytoB; do
      claimbreak compartment summary --compare $W/cmp_$c --out $W/sum_${c}_$x --level $LV --test "$x|nucA" \
        --unit-pattern "$S" > $W/logs/sum_${c}_$x.log 2>&1 || { tail -5 $W/logs/sum_${c}_$x.log; stop "summary" }
      grep -E "^per unit, pooled|^block bootstrap|^matched on regional" -A2 $W/logs/sum_${c}_$x.log | cut -c1-200
    done
    claimbreak compartment contacts --labels $W/labels --out $W/contacts_$c --level $LV --immune T_NK,B,Myeloid \
      --partners PT --ref nucB --test wholeB --unit-pattern "$S" --mode composition --contact $c $ex \
      > $W/logs/contacts_$c.log 2>&1 || tail -5 $W/logs/contacts_$c.log
  done
  python - $W <<'EOF'
import os, sys, pandas as pd
W = sys.argv[1]
for c in ("knn", "boundary"):
    f = os.path.join(W, f"contacts_{c}", "units.csv")
    if os.path.exists(f):
        U = pd.read_csv(f); cols = [x for x in U.columns if x.startswith("per100|")]
        print(c, "per 100 PT cells:\n", U.set_index("labels")[cols].round(2).to_string())
EOF
}

truth() {
  [[ -e $W/protein ]] && stop "$W/protein exists; move it away"
  stamp "protein truth"
  python -u $CB/examples/compartment/xenium_protein.py truth --labels $W/labels --protein $X/h5ad/$S.protein.csv.gz \
    --sample $S --out $W/protein --level $LV 2>&1 | tee $W/logs/truth.log
}

strict() {
  # the recall ratios with lineage-consistent protein classes (T_strict = CD3E+ CD45+ CD68-
  # PanCK-; CD8, CD4, B likewise) at the usual posterior (0.5) and a stricter one (0.9); class_overlap.csv shows how
  # many loose class cells are CD45-negative or carry CD68 or PanCK. Not part of `all`; run after `label`.
  local post o
  for post in 0.5 0.9; do
    o=$W/protein_strict_p${post/./}
    [[ -e $o ]] && { print "$o exists; skipped (move it away to rerun)"; continue }
    stamp "protein truth, strict classes, posterior $post"
    python -u $CB/examples/compartment/xenium_protein.py truth --labels $W/labels --protein $X/h5ad/$S.protein.csv.gz \
      --sample $S --out $o --level $LV --strict --posterior $post 2>&1 | tee $W/logs/truth_strict_p${post/./}.log
  done
}

same() { python - "$@" <<'EOF'
# nucA / nucB of the remedy labels must equal the existing labels (same seed, same halves)
import os, sys
import pandas as pd
old, new, s, lv = sys.argv[1:5]
A = pd.read_csv(os.path.join(old, f"{s}.labels.csv.gz"), usecols=["cell_id", f"{lv}_nucA", f"{lv}_nucB"], keep_default_na=False, dtype=str)
B = pd.read_csv(os.path.join(new, f"{s}.labels.csv.gz"), usecols=["cell_id", f"{lv}_nucA", f"{lv}_nucB"], keep_default_na=False, dtype=str).set_index("cell_id").reindex(A.cell_id)
r = [(A[f"{lv}_{a}"].to_numpy() == B[f"{lv}_{a}"].fillna("").to_numpy()).mean() for a in ("nucA", "nucB")]
print(f"  nucA {r[0]:.4f}, nucB {r[1]:.4f} identical to labels/")
sys.exit(0 if min(r) >= 0.999 else 1)
EOF
}

remedy() {
  # DECON='<label options>' (e.g. '--decon-w 1,0,0') and TAG=<name> write rem_labels_<TAG>, protein_rem_<TAG>
  local T=${TAG:+_$TAG} RL PR
  RL=$W/rem_labels$T; PR=$W/protein_rem$T
  [[ -f $W/labels/$S.labels.csv.gz ]] || stop "run label first ($W/labels)"
  [[ -e $RL ]] && stop "$RL exists; move it away"
  [[ -e $PR ]] && stop "$PR exists; move it away"
  mkdir -p $W/logs
  local opts; opts=$(optpy $AINLAB $W/exclude_genes_rem$T.txt) || stop "AIN options"
  stamp "remedy$T: label $RARMS with $opts ${DECON:-}"
  eval claimbreak compartment label --ref $REF --ref-levels $LV=coarse --levels $LV --ref-merge "''" \
    --inputs $X/h5ad/$S.h5ad --out $RL --arms $RARMS $opts ${DECON:-} 2>&1 \
    | tee $W/logs/rem_label$T.log | grep -v "^  wrote" | cut -c1-240
  [[ -f $RL/$S.labels.csv.gz ]] || stop "remedy labelling wrote nothing"
  stamp "remedy$T: nuclear halves against labels/"
  same $W/labels $RL $S $LV || stop "nucA/nucB differ from labels/"
  stamp "remedy$T: protein truth of every arm"
  python -u $CB/examples/compartment/xenium_protein.py truth --labels $RL --protein $X/h5ad/$S.protein.csv.gz \
    --sample $S --out $PR --level $LV --arms nucleus,whole,nucB,wholeB,cytoB,cytoM,wBall,wBallM,wholeM \
    --pairs nucB:wholeB,nucB:wBall,nucB:wBallM,nucleus:whole,nucleus:wholeM --gaps nucB:wBall:wBallM \
    2>&1 | tee $W/logs/truth_rem$T.log
}

remsens() {
  # the remedy again for each weight
  # set in WSETS (contact,regional,ambient; default: contact only, half and twice the non-contact terms), each into
  # rem_labels_w<a>_<b>_<c> and protein_rem_w<a>_<b>_<c> (dots as p); a set whose protein folder exists is skipped.
  # Then remedy_weights.csv: share of the T (and other) recall gap wBall->nucB that wBallM closes, and the recalls.
  local w t
  for w in ${=WSETS:-1,0,0 1,0.25,0.05 1,1,0.2}; do
    t=w${w//,/_}; t=${t//./p}
    if [[ -e $W/protein_rem_$t ]]; then print "exists: $W/protein_rem_$t"; continue; fi
    TAG=$t DECON="--decon-w $w" remedy || return 1
  done
  python - $W <<'PYEOF'
import sys, glob, os, pandas as pd
W = sys.argv[1]
rows = []
for d in sorted(glob.glob(os.path.join(W, "protein_rem*"))):
    tag = os.path.basename(d).replace("protein_rem", "").lstrip("_") or "w1_0p5_0p1"
    try:
        C = pd.read_csv(os.path.join(d, "recall_ci.csv"))
        R = pd.read_csv(os.path.join(d, "recall.csv"))
    except FileNotFoundError:
        continue
    g = C[(C.metric == "gap_closed_wBallM_wBall") & (C.stratum == "all")].drop_duplicates("protein_class")
    g = g.set_index("protein_class")
    R = R[(R.variant == "all") & (R.stratum == "all")].pivot_table(index="protein_class", columns="arm", values="recall")
    for cls in ("T", "CD8", "CD4", "B", "Myeloid"):
        if cls in R.index:
            rows.append(dict(weights=tag, protein_class=cls,
                             gap_closed=g.estimate.get(cls) if "estimate" in g else None,
                             gap_lo=g.lo.get(cls) if "lo" in g else None, gap_hi=g.hi.get(cls) if "hi" in g else None,
                             nucB=R.loc[cls].get("nucB"), wBall=R.loc[cls].get("wBall"), wBallM=R.loc[cls].get("wBallM")))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(W, "remedy_weights.csv"), index=False)
print(S.round(3).to_string(index=False))
print(f"wrote {W}/remedy_weights.csv")
PYEOF
}

border() {
  [[ -f $W/labels/$S.labels.csv.gz ]] || stop "run label first ($W/labels)"
  local BB=${BBASE:-outline} OD
  OD=$W/protein_border; [[ $BB == outline ]] && OD=$W/protein_border_outline
  [[ -e $OD ]] && stop "$OD exists; move it away"
  mkdir -p $W/logs
  stamp "border: loss by the non-immune share of the polygon perimeter"
  python -u $CB/examples/compartment/xenium_protein.py border --labels $W/labels --protein $X/h5ad/$S.protein.csv.gz \
    --boundaries $X/links --sample $S --out $OD --level $LV --base $BB 2>&1 | tee $W/logs/${OD:t}.log
}

groups() {
  local G=${GROUPS:-$(print -l $X/*_cell_groups.csv(N) | head -1)}
  [[ -f $G ]] || stop "no cell groups file (set GROUPS=<csv>)"
  [[ -e $W/groups_pub ]] && stop "$W/groups_pub exists; move it away"
  mkdir -p $W/logs
  stamp "groups: $G against protein"
  python -u $CB/examples/compartment/xenium_protein.py groups --groups $G --protein $X/h5ad/$S.protein.csv.gz \
    --out $W/groups_pub 2>&1 | tee $W/logs/groups.log
}

gborder() {
  local G=${GROUPS:-$(print -l $X/*_cell_groups.csv(N) | head -1)}
  [[ -f $G ]] || stop "no cell groups file (set GROUPS=<csv>)"
  [[ -f $W/labels/$S.labels.csv.gz ]] || stop "run label first ($W/labels)"
  [[ -e $W/groups_border ]] && stop "$W/groups_border exists; move it away"
  mkdir -p $W/logs
  stamp "groups-border: published labels and claimbreak arms by the non-immune share of the outline"
  python -u $CB/examples/compartment/xenium_protein.py groups-border --groups $G --labels $W/labels \
    --protein $X/h5ad/$S.protein.csv.gz --boundaries $X/links --sample $S --out $W/groups_border --level $LV \
    2>&1 | tee $W/logs/groups_border.log
}

harvest() {
  local DL=${CB_DOWNLOADS:-~/Downloads} Z
  Z=$DL/cb_${W:t}_$(date +%Y%m%d_%H%M).zip
  (cd ${W:h} && zip -q -r $Z ${W:t}/protein ${W:t}/protein_strict_*(N) ${W:t}/protein_rem(N) ${W:t}/protein_border*(N) ${W:t}/groups_pub(N) ${W:t}/groups_border(N) ${W:t}/rem_labels/*.meta.json(N) ${W:t}/logs ${W:t}/labels/*.meta.json ${W:t}/sum_*/{sections,units,units_kept,pairs,pairs_units}.csv(N) \
     ${W:t}/cmp_*/{summary,did,settings,boundary_contacts}.*(N) ${W:t}/contacts_*/{units,classes}.csv(N))
  (cd $X && zip -q -u $Z h5ad/$S.protein_markers.json)
  ls -lh $Z
}

case $1 in
  check) check ;;
  prepare) prepare ;;
  label) label ;;
  test) test_ ;;
  truth) truth ;;
  strict) strict ;;
  remedy) remedy ;;
  remsens) remsens ;;
  border) border ;;
  groups) groups ;;
  gborder) gborder ;;
  harvest) harvest ;;
  all) check && prepare && label && test_ && truth && harvest ;;
  *) stop "usage: xp_batch.zsh check|prepare|label|test|truth|strict|remedy|remsens|border|groups|gborder|harvest|all" ;;
esac
