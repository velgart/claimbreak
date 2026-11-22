#!/bin/zsh
# rpgn_batch.zsh -- a published claim tested (Sultana et al., Nat Immunol 2025;
# GSE294965, Xenium 480 genes, kidney biopsies of rapidly progressive glomerulonephritis). The authors' Baysor labels
# put adaptive immune cells nearly absent inside glomeruli; this labels the vendor cells of the same slides from nuclear
# and whole-cell counts and compares T/NK inside glomeruli with the 100-um ring (rule in rpgn_glom.py).
#   zsh $CB/examples/compartment/rpgn_batch.zsh check
#   caffeinate -ims zsh $CB/examples/compartment/rpgn_batch.zsh all 2>&1 | tee ~/claimbreak_results/logs/rpgn_$(date +%Y%m%d_%H%M).log
# steps: fetch prepare label test glom harvest (each refuses to overwrite its output)
# fetch    per slide only transcripts, cells, cell and nucleus boundaries, count matrix, experiment.xenium from GEO
#          (no morphology images), resumable, into $X/raw (originals; never modified)
# prepare  links with the sample prefix, compartment_to_h5ad.py (whole, nucleus, cyto counts; .gz read via a temp copy)
# label    claimbreak compartment label against the kidney reference with the options of the AIN kidney labels
# test     compartment compare + summary (6 nearest within 30 um), the extra-change ratio of this dataset
# glom     rpgn_glom.py: T/NK inside glomeruli vs ring per biopsy, nuclear / whole / halves / published Baysor labels
#          (needs $X/derived/baysor_cells.parquet, written from the deposited processed_data.h5ad)
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
X=${RPGN_X:-$D/data/rpgn_sultana}; W=${RPGN_W:-$R/rpgn}   # own names, so X and W left over from other batches are ignored
REF=${REF:-$D/AIN/ref/kidney_census.h5ad}; AINLAB=${AINLAB:-$D/AIN/labels/v1}
LV=L1
GEO=https://ftp.ncbi.nlm.nih.gov/geo/samples
# slide -> GEO sample; default: the three slides with 6 of the 7 control biopsies (override SLIDES="0011546 ...")
typeset -A GSM; GSM=(0011216 GSM8933087 0011284 GSM8933088 0011287 GSM8933089 0011546 GSM8933090 0011695 GSM8933091
  0011707 GSM8933092 0011762 GSM8933093 0018775 GSM8933094)
SLIDES=(${=SLIDES:-0011546 0011762 0011284})
FILES=(transcripts.parquet.gz cells.parquet.gz cell_boundaries.parquet.gz nucleus_boundaries.parquet.gz
  cell_feature_matrix.h5 experiment.xenium.gz)
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
smp() { print RPGN_$1 }

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
  [[ -f $REF ]] && print "reference ok: $REF" || stop "reference missing: $REF"
  ls $AINLAB/*.meta.json >/dev/null 2>&1 && print "AIN label options: $AINLAB" || stop "no AIN labels meta in $AINLAB"
  [[ -f $X/derived/baysor_cells.parquet ]] && print "Baysor cells ok" || stop "missing $X/derived/baysor_cells.parquet"
  local s f g
  for s in $SLIDES; do
    g=${GSM[$s]}; [[ -n $g ]] || stop "unknown slide $s"
    for f in $FILES; do
      [[ -f $X/raw/${g}_${s}_$f ]] && print "  ok      ${g}_${s}_$f ($(du -h $X/raw/${g}_${s}_$f | cut -f1))" \
        || print "  absent  ${g}_${s}_$f (fetch)"
    done
  done
  df -h $X | tail -1
}

fetch() {
  mkdir -p $X/raw
  local s f g u n
  for s in $SLIDES; do
    g=${GSM[$s]}
    for f in $FILES; do
      u=$GEO/${g[1,7]}nnn/$g/suppl/${g}_${s}_$f
      n=$(curl -sIL $u | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}')
      [[ $n -gt 0 ]] || stop "no size for $u"
      until [[ -f $X/raw/${g}_${s}_$f && $(stat -f%z $X/raw/${g}_${s}_$f) == $n ]]; do
        stamp "fetch ${g}_${s}_$f ($((n/1000000)) MB)"
        curl -L -C - --retry 5 --retry-delay 10 -o $X/raw/${g}_${s}_$f $u || sleep 20
      done
    done
  done
  ls -lh $X/raw | grep -v morphology
}

prepare() {
  mkdir -p $X/links $X/h5ad $X/tmp
  local s g f S
  for s in $SLIDES; do
    g=${GSM[$s]}; S=$(smp $s)
    for f in transcripts.parquet.gz cells.parquet.gz cell_boundaries.parquet.gz nucleus_boundaries.parquet.gz cell_feature_matrix.h5; do
      [[ -f $X/raw/${g}_${s}_$f ]] || stop "missing ${g}_${s}_$f (run fetch)"
      ln -sf $X/raw/${g}_${s}_$f $X/links/${S}_$f
    done
    [[ -f $X/h5ad/$S.h5ad ]] && { print "exists: $X/h5ad/$S.h5ad"; continue }
    stamp "convert $S"
    python -u $CB/examples/compartment/compartment_to_h5ad.py --platform xenium --raw $X/links --sample $S \
      --out $X/h5ad --tmp $X/tmp || stop "convert $S"
  done
}

label() {
  [[ -e $W/labels ]] && stop "$W/labels exists; move it away"
  mkdir -p $W/logs
  local opts ins s; opts=$(optpy $AINLAB $W/exclude_genes.txt) || stop "AIN options"
  ins=(); for s in $SLIDES; do ins+=($X/h5ad/$(smp $s).h5ad); done
  stamp "label with $opts"
  eval claimbreak compartment label --ref $REF --ref-levels $LV=coarse --levels $LV --ref-merge "''" \
    --inputs ${ins[@]} --out $W/labels --arms nucA,nucB,wholeB,cytoB,nucleus,whole $opts 2>&1 \
    | tee $W/logs/label.log | grep -v "^  wrote" | cut -c1-240
  for s in $SLIDES; do [[ -f $W/labels/$(smp $s).labels.csv.gz ]] || stop "labelling wrote nothing for $s"; done
}

test_() {
  mkdir -p $W/logs
  [[ -e $W/cmp_knn ]] && { print "exists: $W/cmp_knn"; return }
  stamp "compare, contact knn"
  claimbreak compartment compare --labels $W/labels --out $W/cmp_knn --levels $LV --contact knn \
    --comparisons "nucB|nucA,wholeB|nucA,cytoB|nucA" --tile 200 > $W/logs/cmp_knn.log 2>&1 || { tail -5 $W/logs/cmp_knn.log; stop "compare" }
  claimbreak compartment summary --compare $W/cmp_knn --out $W/sum_knn_wholeB --level $LV --test "wholeB|nucA" \
    --unit-pattern "RPGN_[0-9]+" > $W/logs/sum_knn_wholeB.log 2>&1 || { tail -5 $W/logs/sum_knn_wholeB.log; stop "summary" }
  grep -E "^per unit, pooled|^block bootstrap" -A4 $W/logs/sum_knn_wholeB.log | cut -c1-200
}

glom() {
  [[ -e $W/glom ]] && stop "$W/glom exists; move it away"
  mkdir -p $W/logs
  local pairs=() s; for s in $SLIDES; do pairs+=($(smp $s)=$s); done
  stamp "glom: T/NK inside glomeruli vs ring"
  python -u $CB/examples/compartment/rpgn_glom.py --labels $W/labels --baysor $X/derived/baysor_cells.parquet \
    --slides ${(j:,:)pairs} --out $W/glom --level $LV 2>&1 | tee $W/logs/glom.log
}

harvest() {
  local DL=${CB_DOWNLOADS:-~/Downloads} Z
  Z=$DL/cb_${W:t}_$(date +%Y%m%d_%H%M).zip
  (cd ${W:h} && zip -q -r $Z ${W:t}/glom ${W:t}/logs ${W:t}/labels/*.meta.json ${W:t}/sum_*/{sections,units,units_kept,pairs}.csv(N) \
     ${W:t}/cmp_*/{summary,settings}.*(N))
  ls -lh $Z
}

case $1 in
  check) check ;;
  fetch) fetch ;;
  prepare) prepare ;;
  label) label ;;
  test) test_ ;;
  glom) glom ;;
  harvest) harvest ;;
  all) check && fetch && prepare && label && test_ && glom && harvest ;;
  *) stop "usage: rpgn_batch.zsh check|fetch|prepare|label|test|glom|harvest|all" ;;
esac
