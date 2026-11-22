#!/bin/zsh
# cosmx_label_settings.zsh -- which classifier settings label the CosMx NSCLC sections without collapse.
# Four sections (Lung12, where the default collapsed onto Vessel; Lung5 Rep1-3, serial sections of one patient whose
# compositions should agree), five settings, arms whole, nucleus, nucA, nucB, wholeB; no reference check (fast).
# Writes ~/claimbreak_results/cosmx_nsclc/pilot_<setting>; prints composition and agreement per section and setting.
setopt no_bang_hist null_glob
D=~/claimbreak_data; C=$D/data/cosmx_nsclc; R=~/claimbreak_results/cosmx_nsclc
REF=$D/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad
ins=( $C/h5ad/{Lung12,Lung5_Rep1,Lung5_Rep2,Lung5_Rep3}_p0.h5ad )
typeset -A V
V[default]=""
V[soft]="--gamma-fit soft"
V[damp05]="--gamma-damp 0.5"
V[nogamma]="--gamma-iter 0"
V[markers20]="--markers 20"
for v in default soft damp05 nogamma markers20; do
  print "\n=== $(date '+%H:%M:%S') $v (${V[$v]:-defaults})"
  claimbreak compartment label --ref $REF --ref-levels L2m=Level2 --ref-merge '^Tu_L\d=>Tumour' --min-counts 10 --seed 0 \
    --arms whole,nucleus,nucA,nucB,wholeB ${=V[$v]} --inputs $ins --out $R/pilot_$v > $R/pilot_$v.log 2>&1 \
    || { tail -3 $R/pilot_$v.log; continue }
done
python - $R <<'PY'
import glob, json, os, sys
R = sys.argv[1]
for v in ("default", "soft", "damp05", "nogamma", "markers20"):
    print(f"\n{v}")
    for f in sorted(glob.glob(os.path.join(R, f"pilot_{v}", "*.meta.json"))):
        m = json.load(open(f)); a = m["agreement"]; c = m["composition"]["L2m_whole"]
        top = ", ".join(f"{k} {x:.2f}" for k, x in list(c.items())[:5])
        print(f"  {os.path.basename(f)[:13]:13s} nucB==nucA {a['L2m:nucB==nucA'][0]:.3f}  wholeB==nucA {a['L2m:wholeB==nucA'][0]:.3f}"
              f"  whole==nucleus {a['L2m:whole==nucleus'][0]:.3f} | T_NK {c.get('T_NK', 0):.3f} | {top}")
PY
