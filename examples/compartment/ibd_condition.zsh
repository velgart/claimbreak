#!/bin/zsh
# ibd_condition.zsh -- intraepithelial lymphocytes per 100 epithelial cells by IBD
# condition (inflamed or not, disease, tissue: every characteristic in the GEO series matrix), counted from nuclear,
# whole-cell and hybrid labels of the same cells (does the label choice change the condition
# contrast?).
#
#   caffeinate -ims zsh $CB/examples/compartment/ibd_condition.zsh 2>&1 | tee ~/claimbreak_results/logs/ibd_condition_$(date +%Y%m%d_%H%M).log
#
# Runs ibd_condition.py twice: on the 337-panel sections only (every patient on the same panel and masks; default),
# into $O, and on every labeled section, into ${O}_all (O default $R/ibd_condition2; inflammation and
# disease from the sample titles, contrasts within each site). Env: D, R, O, META ($D/IBD/meta),
# LAB ($D/IBD/labels/xenium_v1), REF (reference level, optional). Refuses to overwrite.
setopt no_bang_hist null_glob pipefail
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}
META=${META:-$D/IBD/meta}; LAB=${LAB:-$D/IBD/labels/xenium_v1}; O=${O:-$R/ibd_condition2}
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
python -c "import claimbreak, sys; v = tuple(map(int, claimbreak.__version__.split('.'))); print('claimbreak', claimbreak.__version__); sys.exit(v < (1, 7, 103))" \
  || stop "claimbreak 1.7.103 or later needed"
[[ -d $D ]] || stop "data drive not mounted ($D)"
ls $LAB/*.labels.csv.gz >/dev/null 2>&1 || stop "no labels in $LAB"
for o in $O ${O}_all; do [[ -e $o ]] && stop "$o exists; move it away"; done
stamp "metadata files in $META"
ls -l $META
for f in $META/*series_matrix*; do
  print -r -- "== $f:"
  if [[ $f == *.gz ]]; then gzip -dc $f; else cat $f; fi | grep -E '^!Sample_(title|characteristics_ch1)' | cut -c1-240
done
extra=(); [[ -n $REF ]] && extra=(--ref-level "$REF")
stamp "337-panel sections (same panel and masks for every patient)"
python -u $CB/examples/compartment/ibd_condition.py --labels $LAB --meta $META --out $O $extra \
  | grep -v "^GSM" || stop "ibd_condition failed"
stamp "every labelled section"
python -u $CB/examples/compartment/ibd_condition.py --labels $LAB --meta $META --out ${O}_all --exclude '' $extra \
  | grep -v "^GSM" || stop "ibd_condition (all sections) failed"
print "done"
