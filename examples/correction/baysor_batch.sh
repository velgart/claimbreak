#!/usr/bin/env bash
# baysor_batch.sh -- Baysor as a second transcript-based segmentation (next to Proseg). WSL, next to the Proseg runs of proseg_wsl.sh, in ~/claimbreak_work/ihc; never
# touches ~/software/claimbreak. Every step writes into its own new folder and skips work already done.
#   bash baysor_batch.sh check               what is installed and which inputs exist (writes nothing)
#   bash baysor_batch.sh install             Baysor's Linux binary from its GitHub release into ~/opt/baysor
#   bash baysor_batch.sh probe               one small tile of lung L1: Baysor's flags, run time, memory, output columns
#   bash baysor_batch.sh tiles SECTION RAW   tile, run Baysor tile by tile (resumable), merge -> $W/baysor/h5ad/SECTION.h5ad
#   bash baysor_batch.sh lung                lung L1 and L3: tiles, label like the deposited masks, compartment test
#                                            against the deposited masks on the cells both keep, CD8 protein recall
#   bash baysor_batch.sh rcc                 the two renal carcinoma sections: tiles only (labels and protein on the
#                                            Mac: xp_baysor.zsh)
#   bash baysor_batch.sh harvest             ~/baysor_<time>.tgz (no tiles, no h5ad), copied to the Windows Downloads
# Settings: RAW (~/claimbreak_data/Bilous/raw), REF (lung reference), LUNG (GSM9509145_NSCLC_lung_L1_L1 GSM9509149_NSCLC_lung_L3_L3),
# RCC_RAW (/mnt/e/fibrosis/data/xenium_protein/links), RCC (XP_RCC XP_CCRCC), TILE_UM (1000), MARGIN_UM (30),
# MIN_MOL and PRIOR_CONF (empty: the values of Baysor's Xenium preset, -m 50 and confidence 0.5), NTHREADS (8),
# BAYSOR_VERSION (0.9.0): Baysor 0.9.0 (the C++ release), its Xenium preset, checksum-verified install.
set -uo pipefail
W=~/claimbreak_work/ihc; mkdir -p $W/logs $W/baysor; cd $W
CB=$W/claimbreak; PY=$W/venv/bin/python; CBX=$W/venv/bin/claimbreak
RAW=${RAW:-~/claimbreak_data/Bilous/raw}
REF=${REF:-$(ls ~/claimbreak_data/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad ~/claimbreak_work/segser/lung/ref.h5ad 2>/dev/null | head -1)}
LUNG=${LUNG:-"GSM9509145_NSCLC_lung_L1_L1 GSM9509149_NSCLC_lung_L3_L3"}
RCC_RAW=${RCC_RAW:-/mnt/e/fibrosis/data/xenium_protein/links}
RCC=${RCC:-"XP_RCC XP_CCRCC"}
TILE_UM=${TILE_UM:-1000}; MARGIN_UM=${MARGIN_UM:-30}; MIN_MOL=${MIN_MOL:-}; PRIOR_CONF=${PRIOR_CONF:-}
BAYSOR_VERSION=${BAYSOR_VERSION:-0.9.0}
BAYSOR_SHA=${BAYSOR_SHA:-6848f2b924a20a32b19db568e286b156649f0a7ebc8c7ab54d1900f74f9b0e8a}   # linux-x86_64, cpp-0.9.0 release page
NTHREADS=${NTHREADS:-8}
T=$CB/examples/correction/ihc_truth.py
BT=$CB/examples/compartment/baysor_tiles.py
DL=${CB_DOWNLOADS:-/mnt/c/Users/$USER/Downloads}
export JULIA_NUM_THREADS=$NTHREADS OMP_NUM_THREADS=$NTHREADS
export PATH=~/opt/baysor/bin:$PATH
step=${1:-}

fail() { echo "STOP: $*"; return 1 2>/dev/null || exit 1; }
stamp() { echo; echo "=== $(date '+%H:%M:%S') $*"; }
regs() { $PY -c "import pandas as pd; R=pd.read_csv('$W/reg_diag/r0b.csv'); print(','.join('$W/'+r for r in R.loc[R.R0b_match,'registration']))"; }

do_check() {
  stamp "check"
  $PY -c "import claimbreak; print('claimbreak', claimbreak.__version__)" || fail "no claimbreak in $W/venv"
  $PY -c "import claimbreak, sys; sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
    || echo "  claimbreak 1.7.103 or later needed"
  command -v baysor >/dev/null && echo "  baysor: $(baysor --version 2>&1 | head -1)" || echo "  baysor: not installed (bash baysor_batch.sh install)"
  echo "  memory: $(free -g | awk '/Mem:/{print $2" GB total, "$7" GB available"}'); cores: $(nproc)"
  for s in $LUNG; do ls $RAW/${s}_transcripts.parquet* $RAW/${s}_cells.parquet* 2>/dev/null | sed 's/^/  ok      /' || true; done
  for s in $RCC; do ls $RCC_RAW/${s}_transcripts.parquet* $RCC_RAW/${s}_cells.parquet* 2>/dev/null | sed 's/^/  ok      /' \
    || echo "  MISSING $RCC_RAW/${s}_transcripts.parquet (RCC_RAW=...)"; done
  for p in $W/bilous_v4 $W/reg_diag/r0b.csv "$REF"; do [[ -e $p ]] && echo "  ok      $p" || echo "  MISSING $p"; done
}

do_install() {
  stamp "install Baysor $BAYSOR_VERSION (Linux x86-64 binary, checksum from the release page) and its Xenium preset"
  local tgz=baysor-$BAYSOR_VERSION-linux-x86_64.tar.gz rel=https://github.com/kharchenkolab/Baysor/releases/download/cpp-$BAYSOR_VERSION
  mkdir -p ~/opt && cd ~/opt
  if [[ ! -x ~/opt/baysor-$BAYSOR_VERSION-linux-x86_64/bin/baysor ]]; then
    curl -fsSL -o $tgz $rel/$tgz || { cd $W; fail "download $rel/$tgz"; return 1; }
    echo "$BAYSOR_SHA  $tgz" | sha256sum -c - || { cd $W; fail "checksum of $tgz differs from the release page"; return 1; }
    tar xzf $tgz || { cd $W; fail "unpack $tgz"; return 1; }
  fi
  mkdir -p ~/opt/baysor/bin && ln -sf ~/opt/baysor-$BAYSOR_VERSION-linux-x86_64/bin/baysor ~/opt/baysor/bin/baysor
  cd $W
  [[ -f $W/baysor/xenium.toml ]] || curl -fsSL -o $W/baysor/xenium.toml \
    https://raw.githubusercontent.com/kharchenkolab/Baysor/cpp-$BAYSOR_VERSION/configs/xenium.toml \
    || { fail "download the Xenium preset"; return 1; }
  baysor --version 2>&1 | head -1 | tee $W/baysor/version.txt
  baysor run --help > $W/baysor/help.txt 2>&1
  echo "preset ($W/baysor/xenium.toml):"; grep -vE '^\s*(#|$)' $W/baysor/xenium.toml | head -30
}

# Baysor's own flags, checked against its help text (names changed between versions)
baysor_cmd() {   # baysor_cmd INPUT_CSV OUTDIR -> prints the command
  # Baysor 0.9.0 with its Xenium preset (-m 50, prior confidence 0.5, control genes dropped); our tile tables keep
  # the Xenium column names and carry the Xenium cell as an integer prior column (xcell, 0 = none), so the
  # unassigned label is set to 0 here (the preset's is UNASSIGNED). MIN_MOL / PRIOR_CONF override the preset.
  local f=(run -c $W/baysor/xenium.toml -x x_location -y y_location -z z_location -g feature_name
           --unassigned-prior-label 0 --threads $NTHREADS)
  [[ -n $MIN_MOL ]] && f+=(-m $MIN_MOL)
  [[ -n $PRIOR_CONF ]] && f+=(--prior-segmentation-confidence $PRIOR_CONF)
  echo baysor "${f[@]}" -o "$2" "$1" :xcell
}

tile_run() {   # tile_run TILES_DIR: Baysor on every tile not done yet
  local D=$1 t n=0 done_=0
  for t in $(awk -F, 'NR>1 && $NF=="True"{print $1}' $D/tiles.csv); do
    n=$((n+1))
    if [[ -f $D/$t/out/segmentation.csv ]]; then done_=$((done_+1)); continue; fi
    mkdir -p $D/$t/out
    local cmd; cmd=$(baysor_cmd $D/$t/transcripts.csv $D/$t/out)
    local tm=(); [[ -x /usr/bin/time ]] && tm=(/usr/bin/time -v)
    ( cd $D/$t && "${tm[@]}" $cmd ) > $D/$t/baysor.log 2>&1 || { tail -15 $D/$t/baysor.log; fail "baysor on $t (log $D/$t/baysor.log)"; return 1; }
    # some versions name the file with a prefix: link the first *segmentation.csv
    [[ -f $D/$t/out/segmentation.csv ]] || { local s; s=$(ls $D/$t/out/*segmentation.csv 2>/dev/null | head -1)
      [[ -n $s ]] && ln -s $(basename $s) $D/$t/out/segmentation.csv || { fail "no segmentation.csv in $D/$t/out"; return 1; }; }
    echo "  $t: $(grep -E 'Elapsed|Maximum resident' $D/$t/baysor.log | tr -s ' ' | tr '\n' ' ')"
    done_=$((done_+1))
  done
  echo "  tiles done: $done_ of $n"
  [[ $done_ == $n ]]
}

do_tiles() {   # do_tiles SECTION RAW
  local S=$1 R=$2 D=$W/baysor/tiles/$1
  command -v baysor >/dev/null || { fail "baysor not installed"; return 1; }
  [[ -f $W/baysor/xenium.toml ]] || { fail "no Xenium preset (bash baysor_batch.sh install)"; return 1; }
  [[ -f $W/baysor/h5ad/$S.h5ad ]] && { echo "exists: $W/baysor/h5ad/$S.h5ad"; return 0; }
  if [[ ! -f $D/tiles.csv ]]; then
    stamp "tiles $S ($TILE_UM um, margin $MARGIN_UM um)"
    $PY -u $BT prep --raw $R --sample $S --tiles $D --tile-um $TILE_UM --margin-um $MARGIN_UM \
      > logs/baysor_prep_$S.log 2>&1 || { tail -5 logs/baysor_prep_$S.log; fail "prep $S"; return 1; }
    cat logs/baysor_prep_$S.log
  fi
  stamp "baysor $S: $(baysor_cmd TILE/transcripts.csv TILE/out)"
  tile_run $D || return 1
  stamp "merge $S"
  $PY -u $BT merge --raw $R --sample $S --tiles $D --out $W/baysor/h5ad --panel ${PANEL:-Xenium} \
    > logs/baysor_merge_$S.log 2>&1 || { tail -8 logs/baysor_merge_$S.log; fail "merge $S"; return 1; }
  cat logs/baysor_merge_$S.log
}

do_probe() {   # one tile of L1, to see time, memory and columns before the full runs
  local S=${LUNG%% *} D=$W/baysor/probe
  command -v baysor >/dev/null || { fail "baysor not installed"; return 1; }
  [[ -f $W/baysor/xenium.toml ]] || { fail "no Xenium preset (bash baysor_batch.sh install)"; return 1; }
  [[ -e $D ]] && { fail "$D exists; move it away to probe again"; return 1; }
  stamp "probe: one 500 um tile of $S"
  $PY -u $BT prep --raw $RAW --sample $S --tiles $D --tile-um 500 --margin-um $MARGIN_UM > logs/baysor_probe_prep.log 2>&1 \
    || { tail -5 logs/baysor_probe_prep.log; fail "prep"; return 1; }
  local t; t=$(awk -F, 'NR>1 && $NF=="True"{print $7" "$1}' $D/tiles.csv | sort -n | tail -1 | cut -d' ' -f2)
  awk -F, -v t=$t 'NR==1 || $1==t' $D/tiles.csv > $D/tiles1.csv && mv $D/tiles.csv $D/tiles_all.csv && mv $D/tiles1.csv $D/tiles.csv
  tile_run $D || return 1
  head -3 $D/$t/out/segmentation.csv | cut -c1-300
  echo "columns: $(head -1 $D/$t/out/segmentation.csv)"
  [[ -f $D/$t/out/segmentation_params.dump.toml ]] && grep -E "min_molecules|confidence|scale|unassigned|threads" $D/$t/out/segmentation_params.dump.toml | head -12
  $PY - $D/$t/out/segmentation.csv <<'EOF'
import sys, pandas as pd
S = pd.read_csv(sys.argv[1], low_memory=False)
if "cell" not in S.columns:
    sys.exit(f"no 'cell' column; columns: {list(S.columns)}")
c = S["cell"].astype(str)
print(f"molecules {len(S):,}; assigned {float((c != '0').mean()):.3f}; cells {c[c != '0'].nunique():,}; "
      f"transcript_id kept: {'transcript_id' in S.columns}")
EOF
  echo "transcripts in the tile: $(awk -F, -v t=$t '$1==t{print $7}' $D/tiles.csv)"
}

do_lung() {
  local s ok=() P=$W/baysor
  for s in $LUNG; do PANEL=Lung do_tiles $s $RAW && ok+=($s); done
  (( ${#ok[@]} )) || { fail "no lung section finished"; return 1; }
  [[ -f "$REF" ]] || { fail "reference not found (REF=...)"; return 1; }
  local opts
  opts=$($PY - $W/bilous_v4 $P/exclude_genes.txt <<'EOF'
import json, sys, glob, os
m = json.load(open(sorted(glob.glob(os.path.join(sys.argv[1], "*.meta.json")))[0]))
o = []
lay = m.get("ref_layer", "X")
if lay == "raw": o += ["--ref-raw"]
elif lay not in ("X", "", None): o += ["--ref-layer", lay]
for k, f in (("ref_map", "--ref-map"), ("ref_subset", "--ref-subset")):
    if m.get(k): o += [f, m[k]]
o += ["--min-counts", str(m.get("min_counts", 10)), "--seed", str(m.get("seed", 0)), "--floor", str(m.get("floor", 0.01)),
      "--gamma-iter", str(m.get("gamma_iter", 5))]
if m.get("exclude_genes"):
    open(sys.argv[2], "w").write("\n".join(m["exclude_genes"]) + "\n"); o += ["--exclude-genes", sys.argv[2]]
print(" ".join(o))
EOF
) || { fail "no meta.json in $W/bilous_v4"; return 1; }
  for s in ${ok[@]}; do
    [[ -f $P/labels/$s.labels.csv.gz ]] && continue
    stamp "label $s (Baysor) with: $opts"
    $CBX compartment label --ref "$REF" --ref-levels L2m=Level2 --levels L2m --ref-merge '^Tu_L\d=>Tumour' \
      --inputs $P/h5ad/$s.h5ad --out $P/labels --arms nucA,nucB,wholeB,cytoB,nucleus,whole $opts \
      > logs/baysor_label_$s.log 2>&1 || { tail -8 logs/baysor_label_$s.log; fail "label $s"; return 1; }
    grep -E "L2m:|WARNING" logs/baysor_label_$s.log | cut -c1-230
  done
  # the deposited-mask labels of the cells Baysor keeps, so both are compared on the same cells
  local sm; sm=$(IFS=,; echo "${ok[*]}")
  if [[ ! -d $P/labels_dep_common ]]; then
    mkdir -p $P/labels_dep_common
    $PY - $P/labels $W/bilous_v4 $P/labels_dep_common "$sm" <<'EOF'
import os, sys, pandas as pd
lab, dep, out, sm = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
for s in sm:
    A = pd.read_csv(os.path.join(lab, f"{s}.labels.csv.gz"), keep_default_na=False, dtype={"cell_id": str}, usecols=["cell_id"])
    B = pd.read_csv(os.path.join(dep, f"{s}.labels.csv.gz"), keep_default_na=False, dtype={"cell_id": str}, low_memory=False)
    C = B[B.cell_id.isin(set(A.cell_id))]
    C.to_csv(os.path.join(out, f"{s}.labels.csv.gz"), index=False, compression="gzip")
    print(f"{s}: Baysor cells {len(A):,}; deposited cells {len(B):,}; common {len(C):,}")
EOF
  fi
  stamp "compartment test: Baysor and deposited masks, same cells"
  for set_ in "by $P/labels" "dep_common $P/labels_dep_common"; do
    set -- $set_
    [[ -d $P/cmp_$1 ]] && continue
    $CBX compartment compare --labels $2 --out $P/cmp_$1 --levels L2m --samples $sm \
      --comparisons "nucB|nucA,wholeB|nucA,cytoB|nucA" --tile 200 > logs/baysor_cmp_$1.log 2>&1 || { tail -5 logs/baysor_cmp_$1.log; continue; }
    $CBX compartment summary --compare $P/cmp_$1 --out $P/sum_$1 --level L2m --test "wholeB|nucA" --null "nucB|nucA" \
      --unit-pattern 'L\d(?=_L)' --min-null-cells 500 > logs/baysor_sum_$1.log 2>&1 || tail -5 logs/baysor_sum_$1.log
    echo "$1:"; grep -E "^per unit, pooled|^block bootstrap|^kept ratio" -A3 logs/baysor_sum_$1.log | cut -c1-200
  done
  # CD8 protein recall: Baysor arms renamed (by*) so truth keeps them next to the deposited-mask arms
  if [[ ! -d $W/truth_baysor_d2 ]]; then
    stamp "protein recall with Baysor labels"
    rm -rf $P/labels_by && mkdir -p $P/labels_by
    $PY - $P/labels $P/labels_by <<'EOF'
import glob, os, re, sys, pandas as pd
for f in glob.glob(os.path.join(sys.argv[1], "*.labels.csv.gz")):
    L = pd.read_csv(f, keep_default_na=False, dtype={"cell_id": str}, low_memory=False)
    ren = {c: re.sub(r"^L2m_(\w)", lambda m: "L2m_by" + m.group(1).upper(), c) for c in L.columns if c.startswith("L2m_")}
    L.rename(columns=ren)[["cell_id"] + list(ren.values())].to_csv(os.path.join(sys.argv[2], os.path.basename(f)),
                                                                   index=False, compression="gzip")
EOF
    local M; M=$(regs)
    $PY -u $T truth --matches $M --labels $W/bilous_v4,$P/labels_by --out $W/truth_baysor_d2 --max-match-um 2 \
      > logs/baysor_truth_d2.log 2>&1 || { tail -5 logs/baysor_truth_d2.log; fail "truth with Baysor labels"; return 1; }
  fi
  $PY -c "
import pandas as pd; T=pd.read_csv('$W/truth_baysor_d2/t6_counts.csv'); T=T[T.protein_class.isin(['CD8','CD8any'])]
print(T.pivot_table(index=['sample','protein_class'], columns='arm', values='recall').round(3).to_string())" | tee $P/recall_cd8.txt
}

do_rcc() {
  local s
  for s in $RCC; do PANEL=XeniumProtein do_tiles $s $RCC_RAW || return 1; done
  echo "next: copy $W/baysor/h5ad/{${RCC// /,}}.h5ad to the drive and run xp_baysor.zsh"
}

do_harvest() {
  local f=~/baysor_$(date +%Y%m%d_%H%M).tgz
  tar czf $f -C $W baysor/version.txt baysor/help.txt baysor/xenium.toml baysor/tiles/*/tiles.csv baysor/tiles/*/prep.json \
    $(cd $W && ls -d baysor/sum_* baysor/recall_cd8.txt truth_baysor_d2 2>/dev/null) \
    $(cd $W && ls logs/baysor_* baysor/tiles/*/t*/baysor.log baysor/tiles/*/t00_00/out/segmentation_params.dump.toml baysor/probe/*/out/segmentation_params.dump.toml 2>/dev/null) 2>/dev/null
  cp $f $DL/ 2>/dev/null && echo "copied to $DL"
  ls -lh $f
}

case $step in
  check) do_check ;; install) do_install ;; probe) do_probe ;;
  tiles) do_tiles ${2:?section} ${3:?raw folder} ;;
  lung) do_lung ;; rcc) do_rcc ;; harvest) do_harvest ;;
  *) sed -n '2,16p' $0 ;;
esac
