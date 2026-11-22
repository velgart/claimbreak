#!/bin/zsh
# ms_batch.zsh -- MERSCOPE FFPE human uterine cancer with CD45RO co-staining (Vizgen
# showcase, HumanUterineCancerPatient2-ROCostain; 500 genes, 758,506 cells). A second same-section protein truth for
# the T-cell loss, on another platform and tissue, with no registration (the antibody is imaged in the same run).
# CD45RO marks memory T cells (most tumor T cells) and some activated myeloid cells; it passed the pair test of
# merscope_protein_check.py (T signature 4.9x its nearest CD45RO-low neighbor; Xenium Protein CD45RO in RCC 2.7/4.4).
#   zsh $CB/examples/compartment/ms_batch.zsh check
#   caffeinate -ims zsh $CB/examples/compartment/ms_batch.zsh all 2>&1 | tee ~/claimbreak_results/logs/ms_uterine_$(date +%Y%m%d_%H%M).log
#   (steps: nuclei owners prepare reflist ref label truth test; each refuses to overwrite its output)
# nuclei   merscope_nuclei.py: StarDist nuclei on DAPI z3 at /2 (environment NUCPY; output NUC)
# owners   merscope_owners.py: each nucleus to the cell whose z3 outline contains it (cell_boundaries.parquet)
# prepare  merscope_to_h5ad.py --nuclei: whole and nucleus counts (nucleus = in a StarDist nucleus owned by the
#          transcript's cell; a DAPI threshold merges packed nuclei), protein table
# reflist  uterus_reference.py --list-types: Census uterus cell types and their coarse class (no download of counts)
# ref      uterus_reference.py: the reference on the panel genes
# label    claimbreak compartment label with the options of the AIN kidney labels (as xp_batch.zsh); arms nucA, nucB,
#          wholeB, cytoB, nucleus, whole
# truth    xenium_protein.py truth with --alias CD3E=CD45RO_hp: recall of CD45RO+ cells as T_NK by each arm; no
#          contact strata by protein (no epithelial antibody), so --partners none; protein_truth_raw uses the raw signal
# test     compare + summary (6 nearest within 30 um), as xp_batch.zsh, contacts per 100 epithelial cells
setopt no_bang_hist null_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
X=${X:-$D/data/merscope_uterine_cd45ro}; P=${P:-HumanUterineCancerPatient2-ROCostain}; S=${S:-MS_UTERINE_RO}
W=${W:-$R/ms_uterine_ro}; REF=${REF:-$X/ref/uterus_census.h5ad}; AINLAB=${AINLAB:-$D/AIN/labels/v1}
NUCPY=${NUCPY:-$D/envs/stardist/bin/python}; NUC=${NUC:-$X/nuclei}     # StarDist environment and output
LV=L1
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
# an image file under either name (bucket: <prefix>_<file>; browser downloads: <prefix>_images_<file>)
img() { local f=( $X/download/images/${P}_$1(N) $X/download/images/${P}_images_$1(N) )
        (( ${#f} == 1 )) && print -r -- $f[1] || print -r -- "$X/download/images/${P}_$1 (found ${#f})" }
stop() { print "STOP: $*"; exit 1 }

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
  local f
  for f in $X/download/${P}_cell_metadata.csv $X/download/${P}_cell_by_gene.csv $X/download/${P}_detected_transcripts.csv \
           $(img mosaic_DAPI_z3.tif) $(img micron_to_mosaic_pixel_transform.csv); do
    [[ -f $f ]] && print "  ok      ${f:t} ($(du -h $f | cut -f1))" || stop "missing $f"
  done
  local n; n=$(stat -f %z $(img mosaic_DAPI_z3.tif) 2>/dev/null || stat -c %s $(img mosaic_DAPI_z3.tif))
  (( n >= 23353919850 )) && print "  DAPI z3 size $n" || stop "DAPI z3 has $n bytes; download incomplete?"
  head -c 300 $X/download/${P}_detected_transcripts.csv | head -2
  python -c "import cellxgene_census, tifffile, pyarrow; print('census', cellxgene_census.__version__, 'tifffile', tifffile.__version__)" \
    || stop "pip install cellxgene-census tifffile pyarrow"
  ls $AINLAB/*.meta.json >/dev/null 2>&1 && print "AIN label options: $AINLAB" || stop "no AIN labels meta in $AINLAB"
  [[ -x $NUCPY ]] && $NUCPY -c "import stardist, tensorflow as tf; print('stardist', stardist.__version__, 'tensorflow', tf.__version__)" 2>/dev/null \
    || print "StarDist environment missing or broken: $NUCPY (needed by the nuclei step)"
  [[ -f $REF ]] && print "reference ok: $REF" || print "reference not built yet: $REF (steps reflist, ref)"
}

nuclei() {
  # StarDist nuclei on the DAPI plane (separate environment: tensorflow stays out of the main environment)
  [[ -f $NUC/$S.nuclei.npy ]] && { print "exists: $NUC/$S.nuclei.npy"; return }
  [[ -x $NUCPY ]] || stop "no StarDist python at $NUCPY (make one: python3.12 -m venv --copies ...; pip install stardist tensorflow)"
  local q; q=( $X/nucleus_qc_*(/N) ); q=${q[-1]:-}
  stamp "nuclei (StarDist, /2 resolution)"
  $NUCPY -u $CB/examples/compartment/merscope_nuclei.py --img $(img mosaic_DAPI_z3.tif) \
    --meta $X/download/${P}_cell_metadata.csv --sample $S --out $NUC ${q:+--qc-dir} ${q:+$q} || stop "nuclei"
}

owners() {
  # nucleus owner = the cell whose z3 outline contains the nucleus centroid (Vizgen cell_boundaries.parquet)
  [[ -f $NUC/$S.owners.csv.gz ]] && { print "exists: $NUC/$S.owners.csv.gz"; return }
  local b=( $X/download/${P}*cell_boundaries.parquet(N) )
  (( ${#b} == 1 )) || stop "need one $X/download/${P}*cell_boundaries.parquet (found ${#b})"
  python -c "import shapely" 2>/dev/null || stop "pip install shapely"
  stamp "owners by cell outline"
  python -u $CB/examples/compartment/merscope_owners.py --boundaries $b[1] --nuclei $NUC || stop "owners"
}

prepare() {
  [[ -f $X/h5ad/$S.h5ad ]] && { print "exists: $X/h5ad/$S.h5ad"; return }
  [[ -f $NUC/$S.owners.csv.gz ]] || stop "run the owners step first"
  [[ -f $NUC/$S.nuclei.npy ]] || stop "run the nuclei step first ($NUC)"
  stamp "convert (reads 40 GB of transcripts; nuclei from $NUC; about 40 min)"
  python -u $CB/examples/compartment/merscope_to_h5ad.py --dir $X/download --prefix $P --sample $S --out $X/h5ad \
    --donor UterinePatient2 --nuclei $NUC || stop "convert"
}

reflist() {
  stamp "Census uterus cell types"
  python -u $CB/examples/compartment/uterus_reference.py --list-types --out $REF 2>&1 | tee $X/ref_list_types.log | tail -80
  # no tissue of the adnexa may be kept (cell type names are not a test: one endometrium dataset calls its
  # stromal cells "stromal cell of ovary")
  grep -E "^ *True " $X/ref_list_types.log | grep -qiE "adnexa|ovar|fallopian" && stop "adnexal tissue kept in the reference" \
    || print "reference: no adnexal tissue kept"
}

ref() {
  [[ -f $REF ]] && { print "exists: $REF"; return }
  [[ -f $X/h5ad/$S.h5ad ]] || stop "run prepare first (the reference takes the panel genes from the h5ad)"
  stamp "Census uterus reference"
  python -u $CB/examples/compartment/uterus_reference.py --genes-from "$X/h5ad/$S.h5ad" --out $REF || stop "reference"
}

label() {
  [[ -e $W/labels ]] && stop "$W/labels exists; move it away"
  [[ -f $REF ]] || stop "no reference"
  mkdir -p $W/logs
  local opts; opts=$(optpy $AINLAB $W/exclude_genes.txt) || stop "AIN options"
  stamp "label with $opts"
  eval claimbreak compartment label --ref $REF --ref-levels $LV=coarse --levels $LV --ref-merge "''" \
    --inputs $X/h5ad/$S.h5ad --out $W/labels --arms nucA,nucB,wholeB,cytoB,nucleus,whole $opts 2>&1 \
    | tee $W/logs/label.log | grep -v "^  wrote" | cut -c1-240
  [[ -f $W/labels/$S.labels.csv.gz ]] || stop "labelling wrote nothing"
}

truth() {
  local col dst
  for col in CD45RO_hp CD45RO; do
    dst=$W/protein_truth; [[ $col == CD45RO ]] && dst=$W/protein_truth_raw
    [[ -e $dst ]] && { print "exists: $dst"; continue }
    stamp "truth: T class = $col positive"
    python -u $CB/examples/compartment/xenium_protein.py truth --labels $W/labels --protein $X/h5ad/$S.protein.csv.gz \
      --sample $S --out $dst --level $LV --alias CD3E=$col --partners none \
      --arms nucleus,whole,nucA,nucB,wholeB,cytoB --pairs nucB:wholeB,nucleus:whole 2>&1 | tee $W/logs/${dst:t}.log
  done
}

test_() {
  mkdir -p $W/logs
  [[ -e $W/cmp_knn ]] && { print "exists: $W/cmp_knn"; return }
  stamp "compare, contact knn"
  claimbreak compartment compare --labels $W/labels --out $W/cmp_knn --levels $LV --contact knn \
    --comparisons "nucB|nucA,wholeB|nucA,cytoB|nucA" --tile 200 > $W/logs/cmp_knn.log 2>&1 || { tail -5 $W/logs/cmp_knn.log; stop "compare" }
  local x
  for x in wholeB cytoB; do
    claimbreak compartment summary --compare $W/cmp_knn --out $W/sum_knn_$x --level $LV --test "$x|nucA" \
      --unit-pattern "$S" > $W/logs/sum_knn_$x.log 2>&1 || { tail -5 $W/logs/sum_knn_$x.log; stop "summary" }
    grep -E "^per unit, pooled|^block bootstrap|^matched on regional" -A2 $W/logs/sum_knn_$x.log | cut -c1-200
  done
  claimbreak compartment contacts --labels $W/labels --out $W/contacts_knn --level $LV --immune T_NK,B,Myeloid \
    --partners Epithelial --ref nucB --test wholeB --unit-pattern "$S" --mode composition --contact knn \
    > $W/logs/contacts_knn.log 2>&1 || tail -5 $W/logs/contacts_knn.log
}

case ${1:-check} in
  check) check ;;
  nuclei) nuclei ;;
  owners) owners ;;
  prepare) prepare ;;
  reflist) reflist ;;
  ref) ref ;;
  label) label ;;
  truth) truth ;;
  test) test_ ;;
  all) check; nuclei; owners; prepare; ref; label; truth; test_ ;;
  *) stop "steps: check|nuclei|owners|prepare|reflist|ref|label|truth|test|all" ;;
esac
