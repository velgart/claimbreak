#!/bin/zsh
# exposure_batch.zsh -- labels kept against physical exposure to other types
# (analyses A and B). Reads the existing labels and h5ad files only; writes ~/claimbreak_results/exp_<tissue>.
#   zsh $CB/examples/compartment/exposure_batch.zsh check                       # folders, samples, nucleus areas
#   caffeinate -ims zsh $CB/examples/compartment/exposure_batch.zsh lung gut kidney ra cosmx 2>&1 | tee ~/claimbreak_results/logs/batchE_$(date +%Y%m%d_%H%M).log
#   LABELLER=celltypist zsh $CB/examples/compartment/exposure_batch.zsh lung gut kidney ra cosmx   # CellTypist labels
#   zsh $CB/examples/compartment/exposure_batch.zsh harvest                     # ~/Downloads/cb_batchE_<time>.zip
# Per tissue: `claimbreak compartment exposure` on the labels of the compartment test (focal type from nucA, test
# wholeB, null nucB, neighbors typed from the nucleus arm), nucleus gaps 2, 4, 6 um, border share and RNA-weighted
# exposure at 2 um; summary printed for the T/NK type. Units as in the compartment test.
# POLY=1 takes the border share from the segmentation polygons (shared perimeter, --boundaries, 1 um) instead
# of nuclear disks, for tissues with polygons (lung, gut, kidney, rcc, ccrcc); tissues rcc and ccrcc are the two Xenium
# Protein renal carcinoma sections. The paper's Fig. 2b rerun with polygons:
#   POLY=1 TEST=cytoB TAG=poly LUNG_LABELS=~/claimbreak_results/cytoB_lung/merged \
#     caffeinate -ims zsh $CB/examples/compartment/exposure_batch.zsh lung gut kidney rcc ccrcc 2>&1 | tee ~/claimbreak_results/logs/batchE_poly_$(date +%Y%m%d_%H%M).log
# On the WSL workstation: D=/mnt/e/fibrosis R=~/claimbreak_work/batchE LUNG_LABELS=<lung v4 labels> CB_DOWNLOADS=/mnt/c/Users/$USER/Downloads
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
# RA_LAB = synovium labels folder (default gseh2_pre); with R=<new root> the synovium results of an
#   audited relabeling (e.g. RA_LAB=$D/data/RA_labels/gseh4_pre/all R=~/claimbreak_results) go to a new folder.
LABELLER=${LABELLER:-multinomial}
# TEST=<arm> (default wholeB; cytoB = cytoplasm only, where the labels have it), GAPS, CGAP (contact gap for
# border share; 10 um = two 5 um masks touching), TAG=<name> appended to the output folder
TEST=${TEST:-wholeB}; GAPS=${GAPS:-2,6,10,15}; CGAP=${CGAP:-10}; POLY=${POLY:-}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

# tissue: labels (multinomial) | labels (celltypist) | h5ad folder | level | unit pattern | T/NK regex
typeset -A LAB CTL H5 LEVEL UNIT TNK BND
LAB[lung]=${LUNG_LABELS:-$R/bilous_v4};        CTL[lung]=$R/ct_lung/labels;    H5[lung]=$D/Bilous_h5ad
LEVEL[lung]=L2m; UNIT[lung]='L\d(?=_L)'
LAB[gut]=$D/IBD/labels/xenium_v1;              CTL[gut]=$R/ct_gut/labels;      H5[gut]=$D/IBD/h5ad
LEVEL[gut]=L1; UNIT[gut]='I0\d{3}'
LAB[kidney]=$D/AIN/labels/v1;                  CTL[kidney]=$R/ct_kidney/labels; H5[kidney]=$D/AIN/h5ad
LEVEL[kidney]=L1; UNIT[kidney]='(AIN|ATI|REF)_\d+_D'
LAB[ra]=${RA_LAB:-$D/data/RA_labels/gseh2_pre/all};       CTL[ra]=$R/ct_ra/labels;        H5[ra]=$D/data/RA_h5ad/compartment_pre
LEVEL[ra]=L1; UNIT[ra]='p\d{2}'
LAB[cosmx]=$R/cosmx_nsclc_x171/multinomial/labels; CTL[cosmx]=$R/cosmx_nsclc_x171/celltypist/labels
H5[cosmx]=$D/data/cosmx_nsclc/h5ad; LEVEL[cosmx]=L2m; UNIT[cosmx]='Lung\d+'
LAB[rcc]=$R/xp_rcc/labels;    H5[rcc]=$D/data/xenium_protein/h5ad;        LEVEL[rcc]=L1;   UNIT[rcc]='XP_RCC'
LAB[ccrcc]=$R/xp_ccrcc/labels; H5[ccrcc]=$D/data/xenium_protein_ccrcc/h5ad; LEVEL[ccrcc]=L1; UNIT[ccrcc]='XP_CCRCC'
# segmentation polygons (as robust_batch.zsh; the Xenium Protein sections through the links of xp_batch.zsh)
BND[lung]=${BND_LUNG:-$D/Bilous/raw}; BND[gut]=${BND_GUT:-$D/IBD/raw}; BND[kidney]=${BND_KIDNEY:-$D/AIN/boundaries}
BND[rcc]=$D/data/xenium_protein/links; BND[ccrcc]=$D/data/xenium_protein_ccrcc/links
FOCUS=${FOCUS:-'(?i)(^|[^a-z])(t|nk|tnk)([^a-z]|$)'}

labdir() { [[ $LABELLER == celltypist ]] && print ${CTL[$1]} || print ${LAB[$1]} }

areas() { python - "$@" <<'EOF'
# share of cells with a nucleus area in the first two h5ad files of a folder
import glob, os, sys
import anndata as ad, numpy as np
fs = sorted(glob.glob(os.path.join(sys.argv[1], "*.h5ad")))[:2]
for f in fs:
    A = ad.read_h5ad(f, backed="r")
    a = A.obs["nucleus_area"].to_numpy(float) if "nucleus_area" in A.obs else np.array([np.nan])
    print(f"  {os.path.basename(f)}: nucleus_area for {np.isfinite(a).mean():.3f} of {A.n_obs:,} cells"
          + ("" if np.isfinite(a).any() else " (default radius 3 um will be used)"))
    A.file.close()
EOF
}

run() {
  local t=$1 L=$(labdir $1) W=$R/exp_$1
  [[ $LABELLER == celltypist ]] && W=${W}_ct
  [[ $TEST != wholeB ]] && W=${W}_$TEST
  [[ -n $TAG ]] && W=${W}_$TAG
  [[ -n ${LEVEL[$t]} ]] || stop "unknown tissue $t (lung, gut, kidney, ra, cosmx, rcc, ccrcc)"
  local bx=()
  if [[ -n $POLY ]]; then
    [[ -n ${BND[$t]} && -d ${BND[$t]} ]] || stop "$t: POLY=1 needs a polygon folder (${BND[$t]:-none defined})"
    bx=(--boundaries ${BND[$t]} --tol 1)
  fi
  [[ -d $L ]] || stop "$t: no labels folder $L"
  [[ -e $W ]] && stop "$t: $W exists; move it away or set R"
  stamp "$t ($LABELLER): exposure from $L"
  mkdir -p $R/logs
  claimbreak compartment exposure --labels $L --h5ad-dir ${H5[$t]} --out $W --levels ${LEVEL[$t]} \
    --unit-pattern "${UNIT[$t]}" --focus "$FOCUS" --test $TEST --gaps $GAPS --contact-gap $CGAP $bx 2>&1 | tee $R/logs/batchE_${t}_${LABELLER}_${TEST}${TAG:+_$TAG}.log | cut -c1-240
}

case $1 in
  check)
    python -c "import claimbreak, sys; print('claimbreak', claimbreak.__version__); sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
      || stop "claimbreak 1.7.103 or later needed"
    for t in lung gut kidney ra cosmx rcc ccrcc; do
      L=$(labdir $t); n=( $L/*.labels.csv.gz )
      print "\n$t: labels $L (${#n} files); h5ad ${H5[$t]}; level ${LEVEL[$t]}; unit ${UNIT[$t]}"
      [[ -n ${BND[$t]} ]] && print "  polygons ${BND[$t]} $([[ -d ${BND[$t]} ]] && print ok || print MISSING)"
      [[ -d ${H5[$t]} ]] && areas ${H5[$t]} || print "  h5ad folder missing: default radius for every cell"
    done ;;
  harvest)
    DL=${CB_DOWNLOADS:-~/Downloads}; cd $R && zip -q -r $DL/cb_batchE_$(date +%Y%m%d_%H%M).zip exp_*/{units,dose,tissue,dest_units,dest_fit,dest_region_units,dest_region_fit}.csv(N) \
      exp_*/notes.txt logs/batchE_*.log(N)
    ls -lh $DL/cb_batchE_*.zip(om[1]) ;;
  *)
    python -c "import claimbreak, sys; sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
      || stop "claimbreak older than 1.7.103"
    for t in "$@"; do run $t; done ;;
esac
