#!/bin/zsh
# cosmx_label_exclude.zsh -- CosMx NSCLC labels with the over-detected probes excluded.
# Exclusion list = genes detected at more than twice any reference type's share in all three sections checked
# (label_drivers.py on Lung12, Lung5_Rep1, Lung5_Rep2). Same four sections as pilot 1; settings default, markers20,
# nogamma, each with the exclusion. Writes ~/claimbreak_results/cosmx_nsclc/{exclude_overdetected_3of3.txt,pilotx_<setting>}.
setopt no_bang_hist null_glob
D=~/claimbreak_data; C=$D/data/cosmx_nsclc; R=~/claimbreak_results/cosmx_nsclc
REF=$D/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad
X=$R/exclude_overdetected_3of3.txt
cat $R/drivers_{Lung12,Lung5_Rep1,Lung5_Rep2}/exclude_overdetected.txt | grep -v '^#' | sort | uniq -c | awk '$1==3{print $2}' > $X
print "excluded genes (in all three lists): $(wc -l < $X | tr -d ' ')"; head -30 $X | tr '\n' ' '; print
ins=( $C/h5ad/{Lung12,Lung5_Rep1,Lung5_Rep2,Lung5_Rep3}_p0.h5ad )
typeset -A V
V[default]=""
V[markers20]="--markers 20"
V[nogamma]="--gamma-iter 0"
for v in default markers20 nogamma; do
  print "\n=== $(date '+%H:%M:%S') $v + exclusion"
  claimbreak compartment label --ref $REF --ref-levels L2m=Level2 --ref-merge '^Tu_L\d=>Tumour' --min-counts 10 --seed 0 \
    --arms whole,nucleus,nucA,nucB,wholeB --exclude-genes $X ${=V[$v]} --inputs $ins --out $R/pilotx_$v > $R/pilotx_$v.log 2>&1 \
    || { tail -3 $R/pilotx_$v.log; continue }
done
python - $R <<'PY'
import glob, json, os, sys
R = sys.argv[1]
for v in ("default", "markers20", "nogamma"):
    print(f"\n{v} + exclusion")
    for f in sorted(glob.glob(os.path.join(R, f"pilotx_{v}", "*.meta.json"))):
        m = json.load(open(f)); a = m["agreement"]; c = m["composition"]["L2m_whole"]
        top = ", ".join(f"{k} {x:.2f}" for k, x in list(c.items())[:6])
        print(f"  {os.path.basename(f)[:13]:13s} genes {m['genes_used']:4d}  nucB==nucA {a['L2m:nucB==nucA'][0]:.3f}  "
              f"wholeB==nucA {a['L2m:wholeB==nucA'][0]:.3f} | T_NK {c.get('T_NK', 0):.3f} | {top}")
PY
