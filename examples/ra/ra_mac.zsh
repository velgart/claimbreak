#!/bin/zsh
# ra_mac.zsh -- RA synovium (Bhamidipati et al. 2026, pretreatment) as an ordinary dataset
#.
# Inputs on the data drive are only read; derived files go to the
# drive (data/RA_h5ad/compartment, data/RA_ref, data/RA_labels) and results to ~/claimbreak_results/ra_v1 (small tables).
#   zsh ra_mac.zsh convert                 per-sample compartment files (cells passing the authors' filter)
#   zsh ra_mac.zsh gseref                  GSE200815 synovium reference, step 1: cells, clusters, marker scores (review)
#   zsh ra_mac.zsh gsebuild                step 2: build the reference (default REF; REFTAG=gse names the outputs)
#   zsh ra_mac.zsh reflist                 (alternative) Census tissues matching synovium / joint
#   zsh ra_mac.zsh reftypes 'TISSUE|...'   cell types and coarse classes of those tissues (review before building)
#   zsh ra_mac.zsh refbuild 'TISSUE|...'   build the reference (panel genes only)
#   zsh ra_mac.zsh pilot                   label one sample with the reference check (--ref-cv); look at composition
#   zsh ra_mac.zsh label                   label every sample (resumable)
#   zsh ra_mac.zsh analyse                 compartment test, counts, contacts (+ random relabeling), decomposition,
#                                          aggregates, contamination; writes ~/claimbreak_results/ra_v1
#   zsh ra_mac.zsh harvest                 ~/Downloads/cb_ra_v1_<time>.zip (tables and logs)
# Environment: V (drive root, default ~/claimbreak_data), CB (repo, default ~/claimbreak),
# TP (timepoint, default pre), LABEL_OPTS (extra options for compartment label), REFOPTS (extra options for the
# reference builder, e.g. --per-type 1500), GW (GSE200815 work folder), REF / REFTAG (reference file / output tag). Tissues: exact Census labels, or general:A|B for tissue_general (composite).
setopt no_nomatch
V=${V:-~/claimbreak_data}
CB=${CB:-~/claimbreak}
TP=${TP:-pre}
SRC=$V/data/RA_h5ad/samples
H=$V/data/RA_h5ad/compartment_$TP
REF=${REF:-$V/data/RA_ref/gse200815_ref.h5ad}
REFTAG=${REFTAG:-gse}
GSE=$V/data/RA_ref/GSE200815
GW=${GW:-$GSE/work}        # reference work folder (GW=$GSE/work_harmony for the Harmony variant)
L=$V/data/RA_labels/${REFTAG}_$TP
R=~/claimbreak_results/ra_${REFTAG}_$TP
LOG=$R/logs
UNITS='p\d{2}'
IMM=T_NK,Myeloid,B,Plasma
PARTS=Fibroblast,Endothelial
LBL=(--levels L1 --ref-levels L1=coarse --ref-merge= --ref-donor donor_id --gamma-iter 0
     --arms whole,nucleus,nucA,nucB,cytoB,wholeB)
mkdir -p $LOG
step=${1:-}
stop() { print -r -- "STOP: $*"; exit 1 }
[[ -d $V ]] || stop "drive not mounted at $V"
python -c "import claimbreak, sys; v=claimbreak.__version__; sys.exit(0 if tuple(map(int, v.split('.'))) >= (1, 7, 103) else 1)" \
  || stop "claimbreak not found in this python (pip install -e $CB)"

case $step in
convert)
  python -u $CB/examples/ra/ra_to_compartment.py --samples $SRC --out $H --timepoint $TP 2>&1 | tee $LOG/convert.log ;;
gseref)
  python -u $CB/examples/ra/gse200815_reference.py cluster --tar $GSE/GSE200815_RAW.tar \
    --amp-markers $V/data/RA_ref/amp2_supp/41586_2023_6708_MOESM5_ESM.xlsx --work $GW ${=REFOPTS:-} \
    2>&1 | tee $LOG/gseref_cluster_${GW:t}.log ;;
gsebuild)
  ls $H/*.h5ad >/dev/null 2>&1 || stop "run convert first"
  C=$GW/clusters_reviewed.csv; [[ -f $C ]] || C=$GW/clusters.csv
  python -u $CB/examples/ra/gse200815_reference.py build --work $GW --clusters $C --genes-from "$H/*.h5ad" \
    --out $REF 2>&1 | tee $LOG/gseref_build.log ;;
reflist)
  python -u $CB/examples/ra/synovium_reference.py --list-tissues --out $REF 2>&1 | tee $LOG/ref_tissues.log ;;
reftypes)
  [[ -n ${2:-} ]] || stop "give the tissue labels: zsh ra_mac.zsh reftypes 'label1|label2'"
  python -u $CB/examples/ra/synovium_reference.py --list-types --tissues "$2" --out $REF ${=REFOPTS:-} 2>&1 | tee $LOG/ref_types.log ;;
refbuild)
  [[ -n ${2:-} ]] || stop "give the tissue labels"
  ls $H/*.h5ad >/dev/null 2>&1 || stop "run convert first"
  python -u $CB/examples/ra/synovium_reference.py --tissues "$2" --genes-from "$H/*.h5ad" --out $REF ${=REFOPTS:-} \
    2>&1 | tee $LOG/ref_build.log ;;
pilot)
  [[ -f $REF ]] || stop "no reference at $REF"
  f=$(ls $H/*.h5ad | head -1)
  caffeinate -ims claimbreak compartment label --ref $REF --inputs $f --out $L/pilot $LBL --ref-cv ${=LABEL_OPTS:-} \
    2>&1 | tee $LOG/pilot.log ;;
label)
  [[ -f $REF ]] || stop "no reference at $REF"
  caffeinate -ims claimbreak compartment label --ref $REF --inputs $H/*.h5ad --out $L/all $LBL ${=LABEL_OPTS:-} \
    2>&1 | tee $LOG/label.log ;;
analyse)
  A=$L/all
  ls $A/*.labels.csv.gz >/dev/null 2>&1 || stop "run label first"
  run() { local name=$1; shift; [[ -e $R/$name ]] && { print "skip $name"; return }
          print "=== $name $(date +%H:%M)"; "$@" 2>&1 | tee $LOG/$name.log }
  run compare caffeinate -ims claimbreak compartment compare --labels $A --out $R/compare --levels L1 \
    --comparisons "nucB|nucA,cytoB|nucA,wholeB|nucA,whole|nucleus"
  run summary claimbreak compartment summary --compare $R/compare --out $R/summary --level L1 --unit-pattern $UNITS
  run counts python -u $CB/examples/compartment/compartment_variants.py --labels $A --level L1 --group timepoint \
    --unit-pattern $UNITS --summary $R/summary --out $R/counts
  for p in "nucA nucB null" "nucB wholeB dm" "nucleus whole an"; do set -- ${=p}
    run contacts_$3 caffeinate -ims claimbreak compartment contacts --labels $A --out $R/contacts_$3 --level L1 \
      --immune $IMM --partners $PARTS --ref $1 --test $2 --unit-pattern $UNITS --mode composition
    for b in ${(s:,:)PARTS}; do
      run decomp_$3_$b python -u $CB/examples/compartment/contacts_decompose.py --labels $A --ref $1 --test $2 \
        --level L1 --immune $IMM --partner $b --unit-pattern $UNITS --out $R/decomp_$3_$b
    done
  done
  run aggregates caffeinate -ims claimbreak compartment aggregates --labels $A --out $R/aggregates --level L1 \
    --immune $IMM --ref nucB --test wholeB,whole --unit-pattern $UNITS --mode composition
  run contamination caffeinate -ims python -u $CB/examples/compartment/contamination.py --h5ad-dir $H --labels $A \
    --ref $REF --ref-levels L1=coarse --ref-merge= --level L1 --unit-pattern $UNITS --max-cells 50000 \
    --out $R/contamination
  run contamination_split caffeinate -ims python -u $CB/examples/compartment/contamination.py --h5ad-dir $H --labels $A \
    --ref $REF --ref-levels L1=coarse --ref-merge= --level L1 --unit-pattern $UNITS --max-cells 50000 --split-typing \
    --out $R/contamination_split
  print "analyse done $(date +%H:%M); next: zsh ra_mac.zsh harvest" ;;
harvest)
  mkdir -p ~/Downloads; Z=~/Downloads/cb_ra_${REFTAG}_${TP}_$(date +%Y%m%d_%H%M).zip
  ( cd $R && zip -q -r $Z . -x '*/cells.csv.gz' ) && ( cd $L && zip -q -r $Z pilot/*.json pilot/ref_cv_* all/*.meta.json 2>/dev/null )
  ( cd $V/data && zip -q $Z RA_h5ad/compartment_$TP/conversion.csv RA_ref/*.csv RA_ref/GSE200815/work*/clusters*.csv 2>/dev/null )
  ls -lh $Z ;;
*) print "usage: zsh ra_mac.zsh convert|reflist|reftypes TISSUES|refbuild TISSUES|pilot|label|analyse|harvest" ;;
esac
