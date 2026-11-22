#!/usr/bin/env bash
# proseg_wsl.sh -- Proseg runs and protein checks on a Linux or WSL workstation. Works in ~/claimbreak_work/ihc next to
# the earlier IHC runs (ihc_wsl.sh) and uses their clone of claimbreak.
#   bash proseg_wsl.sh setup            update the clone, install, copy the remedy labels, list inputs
#   bash proseg_wsl.sh remedy           protein recall of the neighbor-aware arms (truth on rem_lung labels; Fig. 5d)
#   bash proseg_wsl.sh per100           why L3 gives ~88 T/NK per 100 tumor cells (correction windows vs sections)
#   bash proseg_wsl.sh stains           label-free channel roles on L1 / L3 nuclei and the L6 candidate (usability)
#   bash proseg_wsl.sh proseg-install   Rust + `cargo install proseg` (asks for sudo only if a C compiler is missing)
#   bash proseg_wsl.sh proseg [SAMPLE]  Proseg on one Lung-panel section (default L1), convert, label, compartment
#                                       test against the deposited masks, protein recall
#   bash proseg_wsl.sh day              remedy, per100, stains, proseg (each continues past a failure), then harvest
#   bash proseg_wsl.sh harvest          ~/proseg_wsl_<time>.tgz, copied to the Windows Downloads folder
# Settings: SRC (/mnt/e/transfer), RAW (~/claimbreak_data/Bilous/raw),
# REF (lung reference h5ad), COMPS (IHC components for stains; default 1,5,22), WINDOWS (correction windows),
# PROSEG_SAMPLES (default GSM9509145_NSCLC_lung_L1_L1; add GSM9509149_NSCLC_lung_L3_L3 if time allows).
set -uo pipefail
SRC=${SRC:-/mnt/e/transfer}
W=~/claimbreak_work/ihc; mkdir -p $W/logs; cd $W
CB=$W/claimbreak
PY=$W/venv/bin/python
CBX=$W/venv/bin/claimbreak
IMG=$W/GSE311609_ihc_NSCLC_lung_L3_L3_NSCLC_lung_L6_L6_NSCLC_lung_L1_L1.ome.tif
RAW=${RAW:-~/claimbreak_data/Bilous/raw}
REF=${REF:-$(ls ~/claimbreak_data/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad ~/claimbreak_work/segser/lung/ref.h5ad 2>/dev/null | head -1)}
T=$CB/examples/correction/ihc_truth.py
COMPS=${COMPS:-1,5,22}
WINDOWS=${WINDOWS:-/mnt/e/fibrosis/Bilous_corr/input}
PROSEG_SAMPLES=${PROSEG_SAMPLES:-GSM9509145_NSCLC_lung_L1_L1}
DL=${CB_DOWNLOADS:-/mnt/c/Users/$USER/Downloads}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
[[ -f ~/.cargo/env ]] && . ~/.cargo/env
step=${1:-}

fail() { echo "STOP: $*"; return 1 2>/dev/null || exit 1; }
stamp() { echo; echo "=== $(date '+%H:%M:%S') $*"; }
regs() {   # aligned registrations (reg_diag R0b), as in ihc_wsl.sh B/C
  $PY -c "import pandas as pd; R=pd.read_csv('$W/reg_diag/r0b.csv'); print(','.join('$W/'+r for r in R.loc[R.R0b_match,'registration']))"
}
version_ok() { $PY -c "import claimbreak, sys; v=tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(0 if v >= (1, 7, 103) else 1)"; }

do_setup() {
  stamp "setup"
  [[ -d $CB/.git ]] || { fail "no clone at $CB (run ihc_wsl.sh setup first)"; return 1; }
  [[ -z $(git -C $CB status --porcelain --untracked-files=no) ]] || { git -C $CB status --short; fail "clone $CB has local changes"; return 1; }
  git -C $CB pull -q --ff-only || { fail "fast-forward of $CB failed"; return 1; }
  echo "repo: $(git -C $CB log --oneline -1)  $(git -C $CB describe --tags)"
  $PY -m pip install -q -e $CB || { fail "pip install"; return 1; }
  version_ok || { fail "claimbreak 1.7.103 or later not installed"; return 1; }
  if [[ ! -d $W/rem_lung/labels ]]; then
    [[ -d $SRC/rem_lung_labels ]] || { fail "missing $SRC/rem_lung_labels (copy from the machine that made them)"; return 1; }
    mkdir -p $W/rem_lung/labels && cp $SRC/rem_lung_labels/* $W/rem_lung/labels/
  fi
  echo "inputs:"
  for p in $W/rem_lung/labels $W/bilous_v4 $W/bilous_corr_v1 $W/reg_diag/r0b.csv $W/truth_c_d2 $IMG \
           $W/ihc_lung_overview/components.csv $WINDOWS "$REF"; do
    [[ -e $p ]] && echo "  ok      $p" || echo "  MISSING $p"
  done
  ls $W/rem_lung/labels/*.labels.csv.gz | sed 's/^/  remedy labels: /'
  ls -d $W/nuc_c* $W/reg_L* 2>/dev/null | sed 's/^/  /'
  for s in $PROSEG_SAMPLES; do ls $RAW/${s}_transcripts.parquet* $RAW/${s}_cells.parquet* 2>/dev/null | sed 's/^/  raw: /' \
    || echo "  MISSING raw transcripts/cells for $s in $RAW"; done
  command -v proseg >/dev/null && echo "  proseg: $(proseg --version 2>&1 | head -1)" || echo "  proseg: not installed (bash proseg_wsl.sh proseg-install)"
}

do_remedy() {
  stamp "remedy: protein recall of the neighbour-aware arms"
  local M; M=$(regs) && [[ -n $M ]] || { fail "no aligned registration in reg_diag/r0b.csv"; return 1; }
  [[ -d $W/rem_lung/labels ]] || { fail "no $W/rem_lung/labels (run setup)"; return 1; }
  local o pids=()
  for af in "" "--no-af"; do
    o=truth_rem_d2${af:+_noaf}
    [[ -d $W/$o ]] && { echo "skip $o (exists)"; continue; }
    $PY -u $T truth --matches $M --labels $W/rem_lung/labels --out $W/$o --max-match-um 2 $af > logs/wsl_$o.log 2>&1 &
    pids+=($!)
  done
  local bad=0; for p in "${pids[@]}"; do wait $p || bad=1; done
  (( bad == 0 )) || { tail -5 logs/wsl_truth_rem_d2*.log; fail "truth on the remedy labels failed"; return 1; }
  sed -n '/^sections:/,/^channels:/p' logs/wsl_truth_rem_d2.log
  $PY - $W <<'EOF'
# recall of protein CD8 cells as T/NK by arm, with the earlier run (same matches) for the shared arms
import os, sys, pandas as pd
W = sys.argv[1]
for o in ("truth_rem_d2", "truth_rem_d2_noaf", "truth_c_d2"):
    f = os.path.join(W, o, "t6_counts.csv")
    if not os.path.exists(f): continue
    T = pd.read_csv(f)
    T = T[T.protein_class.isin(["CD8", "CD8any"])] if "protein_class" in T else T
    print(f"\n{o}: recall of protein CD8 cells as T_NK (T6)")
    print(T.pivot_table(index=["sample", "protein_class"], columns="arm", values="recall").round(3).to_string())
EOF
}

do_per100() {
  stamp "per100: L3 against the other donors, windows against sections"
  [[ -d $W/bilous_corr_v1 ]] || { fail "no $W/bilous_corr_v1"; return 1; }
  [[ -d $W/per100_check ]] && { echo "skip per100_check (exists)"; return 0; }
  local wopt=(); [[ -d $WINDOWS ]] && wopt=(--windows $WINDOWS)
  $PY -u $CB/examples/correction/per100_check.py --labels $W/bilous_corr_v1,$W/bilous_v4 --arms ${P100_ARMS:-nucleus,whole} \
    "${wopt[@]}" --out $W/per100_check > logs/wsl_per100.log 2>&1 || { tail -5 logs/wsl_per100.log; fail "per100_check"; return 1; }
  sed -n '/^per donor/,$p' logs/wsl_per100.log
}

do_stains() {
  stamp "stains: label-free channel roles; L6 candidate"
  [[ -f $W/ihc_lung_overview/components.csv ]] && $PY -c "
import pandas as pd; C=pd.read_csv('$W/ihc_lung_overview/components.csv'); print(C.sort_values(C.columns[-1], ascending=False).head(8).to_string(index=False))"
  local c dirs=() pids=()
  for c in ${COMPS//,/ }; do
    if [[ ! -f $W/nuc_c$c/nuclei.json ]]; then
      [[ -f $IMG ]] || { fail "image $IMG missing"; return 1; }
      $PY -u $T nuclei --image $IMG --components $W/ihc_lung_overview/components.csv --component $c --out $W/nuc_c$c \
        > logs/wsl_nuclei_c$c.log 2>&1 & pids+=($!)
    fi
  done
  for p in "${pids[@]}"; do wait $p || echo "nuclei failed (see logs/wsl_nuclei_c*.log)"; done
  for c in ${COMPS//,/ }; do [[ -f $W/nuc_c$c/nuclei.json ]] && dirs+=($W/nuc_c$c); done
  (( ${#dirs[@]} )) || { fail "no nuclei folders"; return 1; }
  local L; L=$(IFS=,; echo "${dirs[*]}")
  for af in "" "--no-af"; do
    o=stains${af:+_noaf}
    [[ -d $W/$o ]] && { echo "skip $o (exists)"; continue; }
    $PY -u $T stains --nuclei $L --out $W/$o $af > logs/wsl_$o.log 2>&1 || { tail -5 logs/wsl_$o.log; fail "stains"; return 1; }
  done
  cat logs/wsl_stains.log
  [[ -f $W/truth_c_d2/channels.csv ]] && { echo "label-based roles (truth_c_d2/channels.csv):"; \
    $PY -c "import pandas as pd; C=pd.read_csv('$W/truth_c_d2/channels.csv'); print(C[['sample','channel','ashman_d','positive_share','role','role_from_whole']].round(3).to_string(index=False))"; }
}

do_proseg_install() {
  stamp "proseg install"
  if ! command -v cc >/dev/null; then
    echo "a C compiler is needed: sudo apt-get update && sudo apt-get install -y build-essential pkg-config"
    sudo apt-get update -q && sudo apt-get install -y -q build-essential pkg-config || { fail "apt"; return 1; }
  fi
  if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal || { fail "rustup"; return 1; }
    . ~/.cargo/env
  fi
  cargo install proseg > logs/wsl_proseg_install.log 2>&1 || { tail -20 logs/wsl_proseg_install.log; fail "cargo install proseg"; return 1; }
  mkdir -p $W/proseg && proseg --version | tee $W/proseg/version.txt
  proseg --help > $W/proseg/help.txt 2>&1
  grep -E -- "--(xenium|output-[a-z-]+|nthreads|overwrite)" $W/proseg/help.txt | head -20
}

proseg_one() {   # one section: run, probe, convert, label, compare, truth
  local S=$1 O=$W/proseg/$1 TMP=$W/proseg/tmp
  command -v proseg >/dev/null || { fail "proseg not installed"; return 1; }
  mkdir -p $O $TMP $W/proseg/h5ad
  local TR
  TR=$(ls $RAW/${S}_transcripts.parquet $RAW/${S}_transcripts.parquet.gz 2>/dev/null | head -1)
  [[ -n $TR ]] || { fail "no transcripts for $S in $RAW"; return 1; }
  if [[ ! -f $O/transcript-metadata.csv.gz ]]; then
    if [[ $TR == *.gz ]]; then gunzip -c $TR > $TMP/${S}_transcripts.parquet; TR=$TMP/${S}_transcripts.parquet; fi
    proseg --help > $W/proseg/help.txt 2>&1
    local flags=() f
    for f in "output-transcript-metadata transcript-metadata.csv.gz" "output-cell-metadata cell-metadata.csv.gz" \
             "output-counts counts.mtx.gz" "output-cell-polygons cell-polygons.geojson.gz" "output-spatialdata proseg.zarr"; do
      set -- $f; grep -q -- "--$1" $W/proseg/help.txt && flags+=(--$1 $O/$2)
    done
    grep -q -- "--output-transcript-metadata" $W/proseg/help.txt || { fail "this proseg has no --output-transcript-metadata; send help.txt"; return 1; }
    stamp "proseg $S: ${flags[*]}"
    local tcmd=(); [[ -x /usr/bin/time ]] && tcmd=(/usr/bin/time -v)
    ( cd $O && "${tcmd[@]}" proseg --xenium $TR "${flags[@]}" ) > logs/wsl_proseg_$S.log 2>&1 \
      || { tail -20 logs/wsl_proseg_$S.log; fail "proseg $S"; return 1; }
    grep -E "Maximum resident|Elapsed" logs/wsl_proseg_$S.log
    rm -f $TMP/${S}_transcripts.parquet
  fi
  $PY $CB/examples/compartment/proseg_to_h5ad.py probe --proseg $O | tee logs/wsl_proseg_probe_$S.log
  if [[ ! -f $W/proseg/h5ad/$S.h5ad ]]; then
    $PY -u $CB/examples/compartment/proseg_to_h5ad.py convert --proseg $O --raw $RAW --sample $S --out $W/proseg/h5ad \
      --tmp $TMP > logs/wsl_proseg_convert_$S.log 2>&1 || { tail -8 logs/wsl_proseg_convert_$S.log; fail "convert $S (send logs/wsl_proseg_probe_$S.log)"; return 1; }
  fi
  tail -25 logs/wsl_proseg_convert_$S.log | grep -v "^wrote"
  # label with the options of the deposited-mask labels (bilous_v4)
  [[ -f "$REF" ]] || { fail "reference not found (REF=...)"; return 1; }
  local opts
  opts=$($PY - $W/bilous_v4 $W/proseg/exclude_genes.txt <<'EOF'
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
  if [[ ! -f $W/proseg/labels/$S.labels.csv.gz ]]; then
    stamp "label $S (Proseg) with: $opts"
    $CBX compartment label --ref "$REF" --ref-levels L2m=Level2 --levels L2m --ref-merge '^Tu_L\d=>Tumour' \
      --inputs $W/proseg/h5ad/$S.h5ad --out $W/proseg/labels --arms nucA,nucB,wholeB,cytoB,nucleus,whole $opts \
      > logs/wsl_proseg_label_$S.log 2>&1 || { tail -8 logs/wsl_proseg_label_$S.log; fail "label $S"; return 1; }
  fi
  grep -E "L2m:|WARNING|composition" logs/wsl_proseg_label_$S.log | cut -c1-230
}

do_proseg() {
  local s ok=()
  for s in ${1:-$PROSEG_SAMPLES}; do proseg_one $s && ok+=($s); done
  (( ${#ok[@]} )) || { fail "no Proseg section finished"; return 1; }
  local P=$W/proseg tag=$(date +%H%M)
  # compartment test: Proseg cells against the deposited masks (same sections, same classifier and options)
  stamp "compartment test: Proseg vs deposited masks"
  local sm; sm=$(IFS=,; echo "${ok[*]}")
  for set_ in "proseg $P/labels" "deposited $W/bilous_v4"; do
    set -- $set_
    [[ -d $P/cmp_$1 ]] && mv $P/cmp_$1 $P/cmp_${1}_$tag.old
    [[ -d $P/sum_$1 ]] && mv $P/sum_$1 $P/sum_${1}_$tag.old
    $CBX compartment compare --labels $2 --out $P/cmp_$1 --levels L2m --samples $sm \
      --comparisons "nucB|nucA,wholeB|nucA,cytoB|nucA" --tile 200 > logs/wsl_proseg_cmp_$1.log 2>&1 || { tail -5 logs/wsl_proseg_cmp_$1.log; continue; }
    $CBX compartment summary --compare $P/cmp_$1 --out $P/sum_$1 --level L2m --test "wholeB|nucA" --null "nucB|nucA" \
      --unit-pattern 'L\d(?=_L)' --min-null-cells 500 > logs/wsl_proseg_sum_$1.log 2>&1 || tail -5 logs/wsl_proseg_sum_$1.log
    echo "$1:"; grep -E "^per unit, pooled|^block bootstrap|^kept ratio" -A3 logs/wsl_proseg_sum_$1.log | cut -c1-200
  done
  $PY - $P $W/bilous_v4 "$sm" <<'EOF' | tee $P/compare_masks.txt
# T/NK at equal depth (wholeB / nucB), whole / nucleus, and agreement with the nucleus, Proseg vs deposited masks,
# on all cells and on the cells both segmentations keep
import os, sys, pandas as pd
P, dep, sm = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
for s in sm:
    A = pd.read_csv(os.path.join(P, "labels", f"{s}.labels.csv.gz"), keep_default_na=False, dtype={"cell_id": str})
    B = pd.read_csv(os.path.join(dep, f"{s}.labels.csv.gz"), keep_default_na=False, dtype={"cell_id": str})
    com = set(A.cell_id) & set(B.cell_id)
    for nm, L in (("proseg", A), ("deposited", B)):
        for scope, D in (("all", L), ("common", L[L.cell_id.isin(com)])):
            t = lambda c: int((D[f"L2m_{c}"] == "T_NK").sum()) if f"L2m_{c}" in D else 0
            ok = (D.L2m_nucleus != "") & (D.L2m_whole != "")
            print(f"{s} {nm:9s} {scope:6s} cells {len(D):7,}  T/NK wholeB/nucB {t('wholeB') / max(t('nucB'), 1):.3f}  "
                  f"whole/nucleus {t('whole') / max(t('nucleus'), 1):.3f}  nucB/nucA {t('nucB') / max(t('nucA'), 1):.3f}  "
                  f"agree whole==nucleus {(D.L2m_whole[ok] == D.L2m_nucleus[ok]).mean():.3f}")
EOF
  # protein recall: Proseg arms renamed (ps*) so that truth keeps them next to the deposited-mask arms
  stamp "protein recall with Proseg labels"
  rm -rf $P/labels_ps && mkdir -p $P/labels_ps
  $PY - $P/labels $P/labels_ps <<'EOF'
import glob, os, re, sys, pandas as pd
for f in glob.glob(os.path.join(sys.argv[1], "*.labels.csv.gz")):
    L = pd.read_csv(f, keep_default_na=False, dtype={"cell_id": str}, low_memory=False)
    ren = {c: re.sub(r"^L2m_(\w)", lambda m: "L2m_ps" + m.group(1).upper(), c) for c in L.columns if c.startswith("L2m_")}
    L = L.rename(columns=ren)[["cell_id"] + list(ren.values())]
    L.to_csv(os.path.join(sys.argv[2], os.path.basename(f)), index=False, compression="gzip")
    print(os.path.basename(f), "arms:", sorted({c[4:].split("_")[0] for c in ren.values()}))
EOF
  local M; M=$(regs)
  [[ -d $W/truth_proseg_d2 ]] && mv $W/truth_proseg_d2 $W/truth_proseg_d2_$tag.old
  $PY -u $T truth --matches $M --labels $W/bilous_v4,$P/labels_ps --out $W/truth_proseg_d2 --max-match-um 2 \
    > logs/wsl_truth_proseg_d2.log 2>&1 || { tail -5 logs/wsl_truth_proseg_d2.log; fail "truth with Proseg labels"; return 1; }
  $PY -c "
import pandas as pd; T=pd.read_csv('$W/truth_proseg_d2/t6_counts.csv'); T=T[T.protein_class.isin(['CD8','CD8any'])]
print(T.pivot_table(index=['sample','protein_class'], columns='arm', values='recall').round(3).to_string())"
}

do_harvest() {
  stamp "harvest"
  local Z=~/proseg_wsl_$(date +%Y%m%d_%H%M).tgz
  local items=(logs/wsl_*)
  for p in truth_rem_d2 truth_rem_d2_noaf truth_proseg_d2 per100_check stains stains_noaf reg_diag \
           truth_c_d2/t6_counts.csv truth_c_d2/channels.csv truth_c_d2/sections.csv \
           proseg/version.txt proseg/help.txt proseg/compare_masks.txt proseg/labels proseg/labels_ps; do
    [[ -e $p ]] && items+=($p)
  done
  for p in proseg/cmp_proseg proseg/cmp_deposited proseg/sum_proseg proseg/sum_deposited; do
    [[ -d $p ]] && items+=($(ls $p/*.csv $p/*.json 2>/dev/null))
  done
  for p in proseg/GSM*; do [[ -d $p ]] && items+=($(ls $p/cell-metadata.csv.gz 2>/dev/null)); done
  for p in nuc_c*; do [[ -f $p/nuclei.json ]] && items+=($p/nuclei.json); done
  tar czf $Z "${items[@]}" && ls -lh $Z
  if [[ -d $DL ]]; then cp $Z $DL/ && echo "copied to $DL/$(basename $Z)"; else echo "NOT copied: $DL not found (set CB_DOWNLOADS); the file is $Z"; fi
}

case $step in
  setup) do_setup ;;
  remedy) do_remedy ;;
  per100) do_per100 ;;
  stains) do_stains ;;
  proseg-install) do_proseg_install ;;
  proseg) do_proseg "${2:-}" ;;
  harvest) do_harvest ;;
  day)
    version_ok || { echo "run setup first"; exit 1; }
    st=()
    for s in remedy per100 stains proseg; do do_$s && st+=("$s ok") || st+=("$s FAILED"); done
    do_harvest
    stamp "status"; printf '  %s\n' "${st[@]}" ;;
  *) echo "usage: bash proseg_wsl.sh setup|remedy|per100|stains|proseg-install|proseg [SAMPLE]|day|harvest" ;;
esac
