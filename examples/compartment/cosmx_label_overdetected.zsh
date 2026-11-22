#!/bin/zsh
# cosmx_label_overdetected.zsh -- over-detected probes per section, all eight CosMx NSCLC sections.
# 1. label_drivers.py on every section (its composition-free list: genes above twice any reference type's share).
# 2. Default classifier settings on all eight sections with three exclusion lists:
#      common  genes over-detected in all three sections of pilot 2 (142; as pilot 2)
#      half    genes over-detected in at least four of the eight sections
#      own     each section's own list (one run per section)
# Writes ~/claimbreak_results/cosmx_nsclc/{drivers_<section>, exclude_{common,half}.txt, pilot3_<list>}.
setopt no_bang_hist null_glob
D=~/claimbreak_data; C=$D/data/cosmx_nsclc; R=~/claimbreak_results/cosmx_nsclc; CB=~/claimbreak
REF=$D/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad
LBL=(--ref $REF --ref-levels L2m=Level2 --ref-merge '^Tu_L\d=>Tumour' --min-counts 10 --seed 0 --arms whole,nucleus,nucA,nucB,wholeB)
secs=( ${${(f)"$(ls $C/h5ad/*_p0.h5ad)"}:t:r} )
for s in $secs; do
  [[ -f $R/drivers_${s%_p0}/exclude_overdetected.txt ]] && continue
  print "drivers $s"
  python $CB/examples/ibd/label_drivers.py --ref $REF --ref-levels L2m=Level2 --ref-merge '^Tu_L\d=>Tumour' \
    --h5ad $C/h5ad/$s.h5ad --labels $R/multinomial/labels --level L2m --out $R/drivers_${s%_p0} > $R/drivers_${s%_p0}.log 2>&1 \
    || { tail -3 $R/drivers_${s%_p0}.log; exit 1 }
done
/bin/cp $R/exclude_overdetected_3of3.txt $R/exclude_common.txt
cat $R/drivers_*/exclude_overdetected.txt | grep -v '^#' | sort | uniq -c | awk '$1>=4{print $2}' > $R/exclude_half.txt
for s in $secs; do print "$s: $(wc -l < $R/drivers_${s%_p0}/exclude_overdetected.txt | tr -d ' ') genes over-detected"; done
print "common $(wc -l < $R/exclude_common.txt | tr -d ' '), half $(wc -l < $R/exclude_half.txt | tr -d ' ')"
for v in common half; do
  print "=== $(date '+%H:%M:%S') $v"
  claimbreak compartment label $LBL --exclude-genes $R/exclude_$v.txt --inputs $C/h5ad/*_p0.h5ad --out $R/pilot3_$v \
    > $R/pilot3_$v.log 2>&1 || tail -3 $R/pilot3_$v.log
done
print "=== $(date '+%H:%M:%S') own"
for s in $secs; do
  claimbreak compartment label $LBL --exclude-genes $R/drivers_${s%_p0}/exclude_overdetected.txt --inputs $C/h5ad/$s.h5ad \
    --out $R/pilot3_own >> $R/pilot3_own.log 2>&1 || tail -3 $R/pilot3_own.log
done
python - $R <<'PY'
import glob, json, os, sys
R = sys.argv[1]
for v in ("common", "half", "own"):
    print(f"\n{v}")
    for f in sorted(glob.glob(os.path.join(R, f"pilot3_{v}", "*.meta.json"))):
        m = json.load(open(f)); a = m["agreement"]; c = m["composition"]["L2m_whole"]
        top = ", ".join(f"{k} {x:.2f}" for k, x in list(c.items())[:6])
        print(f"  {os.path.basename(f)[:13]:13s} genes {m['genes_used']:4d}  nucB==nucA {a['L2m:nucB==nucA'][0]:.3f}  "
              f"wholeB==nucA {a['L2m:wholeB==nucA'][0]:.3f} | T_NK {c.get('T_NK', 0):.3f} | {top}")
PY
