#!/bin/zsh
# IBD pair (GSE312415 CosMx, GSE312420 Xenium; Li et al., Nat Commun 2026): convert every Xenium sample and every
# CosMx slide to compartment h5ad files on the drive. Resumable: finished outputs are skipped. Sequential (memory).
#   caffeinate -ims zsh ~/software/tools/ibd_convert.zsh 2>&1 | tee ~/claimbreak_results/ibd_convert.log
# CosMx doubled tx_files: compartment rule for disagreeing copies, DUPCOMP=first|second|both|either (default first)
setopt no_bang_hist pipefail
B=~/claimbreak_data/IBD
T=~/software/tools
OUT=$B/h5ad; TMP=$B/tmp; mkdir -p $OUT $TMP
# Xenium: one sample per file prefix
for f in $B/raw/xenium/*_transcripts.parquet.gz; do
  s=${${f:t}%_transcripts.parquet.gz}
  [ -f $OUT/$s.h5ad ] && { echo "skip $s"; continue; }
  echo "=== Xenium $s $(date +%H:%M)"
  python -u $T/compartment_to_h5ad.py --platform xenium --raw $B/raw/xenium --sample $s --out $OUT --tmp $TMP \
    | grep -v 'transcripts read' || echo "FAILED $s"
done
# CosMx: one slide per file prefix (tissue pieces become sections <slide>_p<k>)
for f in $B/raw/cosmx/*_tx_file.csv; do
  s=${${f:t}%_tx_file.csv}
  done_p=($OUT/${s}_p*.h5ad(N))           # (N): an empty list, not an error, when nothing matches
  (( $#done_p )) && { echo "skip $s"; continue; }
  echo "=== CosMx $s $(date +%H:%M)"
  python -u $T/compartment_to_h5ad.py --platform cosmx --raw $B/raw/cosmx --sample $s --out $OUT --tmp $TMP \
    --batch 10000000 --dup-comp ${DUPCOMP:-first} | grep -v 'transcripts read' || echo "FAILED $s"
done
echo "=== done $(date +%H:%M)"
ls $OUT | wc -l; du -sh $OUT
