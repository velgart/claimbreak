#!/bin/zsh
# GSE311609 (Bilous et al. 2026): 12 lung sections of the four donors with matched snRNA-seq (Suppl. Table 1),
# four files each, into ~/claimbreak_data/Bilous/raw. Resumable (curl -C -); sizes checked against the server afterwards.
set -u
RAW=~/claimbreak_data/Bilous/raw; mkdir -p $RAW
S=(GSM9509145_NSCLC_lung_L1_L1 GSM9509147_NSCLC_lung_L2_L2_1 GSM9509148_NSCLC_lung_L2_L2_2 GSM9509149_NSCLC_lung_L3_L3
   GSM9509150_NSCLC_lung_L4_L4 GSM9509140_NSCLC_chuvio_L1_L1_1 GSM9509141_NSCLC_chuvio_L1_L1_2 GSM9509142_NSCLC_chuvio_L2_L2
   GSM9509143_NSCLC_chuvio_L3_L3 GSM9509144_NSCLC_chuvio_L4_L4 GSM9509134_NSCLC_5k_L1_L1 GSM9509135_NSCLC_5k_L3_L3)
F=(transcripts.parquet.gz cells.parquet.gz cell_feature_matrix.h5 nucleus_boundaries.parquet.gz)
bad=0
for s in $S; do
  g=${s%%_*}
  for f in $F; do
    u="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9509nnn/${g}/suppl/${s}_${f}"
    o="${RAW}/${s}_${f}"
    want=$(curl -sI "$u" | grep -i content-length | tr -dc '0-9')
    have=$( [ -f "$o" ] && stat -c %s "$o" || echo 0 )
    if [ "$have" = "$want" ]; then echo "ok   ${s}_${f}"; continue; fi
    echo "get  ${s}_${f} ($want bytes)"
    curl -fL --retry 5 --retry-delay 10 -C - -o "$o" "$u" || { echo "FAILED ${s}_${f}"; bad=1; continue; }
    have=$(stat -c %s "$o")
    [ "$have" = "$want" ] || { echo "SIZE MISMATCH ${s}_${f}: $have vs $want"; bad=1; }
  done
done
( cd $RAW && sha256sum * > ../SHA256SUMS_raw.txt ) && echo "checksums: ~/claimbreak_data/Bilous/SHA256SUMS_raw.txt"
[ $bad = 0 ] && echo "ALL OK" || echo "SOME FILES FAILED: rerun this script"
