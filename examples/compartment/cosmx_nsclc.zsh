#!/bin/zsh
# cosmx_nsclc.zsh -- the compartment test on CosMx (NSCLC, He et al. 2022, Nat
# Biotechnol 40:1794; Bruker/NanoString public dataset, 960-gene panel, 8 sections of 5 patients: Lung5 Rep1-3,
# Lung6, Lung9 Rep1-2, Lung12, Lung13). Same lung reference, level and classifier settings as the Xenium lung data.
#   zsh $CB/examples/compartment/cosmx_nsclc.zsh list       # what the tar holds (nothing written)
#   zsh $CB/examples/compartment/cosmx_nsclc.zsh unpack     # flat files only -> $C/flat (minutes)
#   caffeinate -ims zsh $CB/examples/compartment/cosmx_nsclc.zsh convert 2>&1 | tee ~/claimbreak_results/logs/cosmx_convert_$(date +%Y%m%d_%H%M).log
#   caffeinate -ims zsh $CB/examples/compartment/cosmx_nsclc.zsh gate    2>&1 | tee ~/claimbreak_results/logs/cosmx_gate_$(date +%Y%m%d_%H%M).log
#   caffeinate -ims zsh $CB/examples/compartment/cosmx_nsclc.zsh test    2>&1 | tee ~/claimbreak_results/logs/cosmx_test_$(date +%Y%m%d_%H%M).log
#   zsh $CB/examples/compartment/cosmx_nsclc.zsh harvest    # ~/Downloads/cb_cosmx_<time>.zip
# gate: one section (Lung5_Rep1) labeled by both labelers; prints how often the two nuclear halves agree. The test
# is worth running if nucB==nucA is near the Xenium values (0.6-0.9); test: all sections, both labelers, compare,
# summary (unit = patient, 'Lung\d+'), T/NK kept at equal depth, and the CD3 protein stain on the same cells.
# Reads the tar and the reference only. Writes $C/{flat,h5ad,tmp} and ~/claimbreak_results/cosmx_nsclc.
setopt no_bang_hist null_glob extended_glob
D=${D:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; R=${R:-~/claimbreak_results}/cosmx_nsclc${TAG:+_$TAG}
C=${C:-$D/data/cosmx_nsclc}; TAR=${TAR:-$C/raw/All+SMI+Flat+data.tar}
REF=${REF:-$D/Bilous/ref/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad}
LBL=(--ref $REF --ref-levels L2m=Level2 --ref-merge '^Tu_L\d=>Tumour' --cv-group donor_id --min-counts 10 --seed 0
     --arms whole,nucleus,nucA,nucB,wholeB)
# EXCLUDE=<gene list>: genes never used for labeling (e.g. probes over-detected on this prototype panel, from
# cosmx_label_overdetected.zsh), copied into the results folder. TAG=<name>: results in ~/claimbreak_results/cosmx_nsclc_<name>.
if [[ -n $EXCLUDE ]]; then
  [[ -f $EXCLUDE ]] || { print "STOP: no $EXCLUDE"; exit 1 }
  LBL+=(--exclude-genes $EXCLUDE); mkdir -p $R; /bin/cp $EXCLUDE $R/exclude_genes.txt
fi
UNIT='Lung\d+'
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }
want='*_(tx_file|metadata_file|exprMat_file|fov_positions_file).csv(|.gz)'

label() {   # label <labeller> <out> <h5ad files...>
  local lab=$1 out=$2; shift 2
  local extra=(); [[ $lab == celltypist ]] && extra=(--labeller celltypist --ct-jobs ${CTJOBS:-1})
  claimbreak compartment label $LBL $extra --ref-cv --inputs "$@" --out $out 2>&1 | grep -v "^  wrote"
}

case $1 in
  list)
    [[ -f $TAR ]] || stop "no $TAR"
    tar tvf $TAR | tee $C/listing.txt | head -60; print "... $(wc -l < $C/listing.txt) members (listing in $C/listing.txt)" ;;
  unpack)
    [[ -f $TAR ]] || stop "no $TAR"
    mkdir -p $C/flat/x
    tar tf $TAR | grep -E '_(tx_file|metadata_file|exprMat_file|fov_positions_file)\.csv(\.gz)?$|Flat.*data.*\.(tar\.gz|tgz)$' > $C/flat/members.txt
    print "$(wc -l < $C/flat/members.txt) members to extract"; (( $(wc -l < $C/flat/members.txt) )) || stop "no flat files in $TAR"
    tar xvf $TAR -C $C/flat/x -T $C/flat/members.txt 2>&1 | tail -20
    for z in $C/flat/x/**/*.(tar.gz|tgz); do     # per-sample archives inside the bundle
      print "inner archive ${z:t}"
      tar tzf $z | grep -E '_(tx_file|metadata_file|exprMat_file|fov_positions_file)\.csv(\.gz)?$' > $z.members.txt
      tar xzf $z -C ${z:h} -T $z.members.txt
    done
    mkdir -p $C/flat/all
    for f in $C/flat/x/**/${~want}; do ln -sf $f $C/flat/all/${f:t}; done
    ls -l $C/flat/all ;;
  convert)
    mkdir -p $C/h5ad $C/tmp
    for f in $C/flat/all/*_tx_file.csv(|.gz); do
      s=${${f:t}%_tx_file.csv*}
      [[ -n $(print $C/h5ad/${s}_p*.h5ad(N)) ]] && { print "$s converted; skipped"; continue }
      stamp "convert $s"
      python -u $CB/examples/compartment/compartment_to_h5ad.py --platform cosmx --raw $C/flat/all --sample $s \
        --out $C/h5ad --tmp $C/tmp --px-um 0.18 --no-sha --min-piece-cells ${MINPIECE:-2000} || stop "convert $s"
      [[ -n $(print $C/h5ad/${s}_p*.h5ad(N)) ]] || print "WARNING $s: no section written (pieces below --min-piece-cells)"
    done ;;
  gate)
    f=( $C/h5ad/Lung5_Rep1_p*.h5ad(N) ); (( ${#f} )) || stop "Lung5_Rep1 not converted"
    mkdir -p $R/gate
    for lab in multinomial celltypist; do stamp "gate: $lab"; label $lab $R/gate/$lab $f[1]; done
    print "\nXenium lung for comparison (nucB==nucA): $(python -c "import json,glob;print(sorted({round(json.load(open(m))['agreement'].get('L2m:nucB==nucA',[0])[0],2) for m in glob.glob('$HOME/cbresults/bilous_v4/*.meta.json')}))" 2>/dev/null)" ;;
  test)
    ins=( $C/h5ad/*_p*.h5ad(N) ); (( ${#ins} )) || stop "nothing converted"
    for lab in multinomial celltypist; do
      W=$R/$lab; stamp "label: $lab (${#ins} sections)"; label $lab $W/labels $ins
      stamp "compare + summary: $lab"
      claimbreak compartment compare --labels $W/labels --out $W/compare --levels L2m --comparisons "nucB|nucA,wholeB|nucA,whole|nucleus" \
        > $W/compare.log 2>&1 || { tail -5 $W/compare.log; stop "compare $lab" }
      claimbreak compartment summary --compare $W/compare --out $W/summary --level L2m --unit-pattern $UNIT 2>&1 \
        | tee $W/summary.log | grep -E "^sections used|^per unit, pooled|^extra-change ratio" | cut -c1-220
    done
    stamp "T/NK kept at equal depth, and the CD3 stain"
    python - $C/h5ad $R $UNIT <<'EOF' 2>&1 | tee $R/tnk_cd3.log
import glob, os, re, sys
import anndata as ad, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
h5, R, unit = sys.argv[1:4]
rx = re.compile(r"(?i)(^|[^a-z])(t|nk|tnk)([^a-z]|$)")
rows, prot = [], []
for f in sorted(glob.glob(os.path.join(h5, "*_p*.h5ad"))):
    s = os.path.basename(f)[:-5]
    A = ad.read_h5ad(f)
    u = re.search(unit, s).group(0)
    # is the CD3 stain a property of the cell? its own CD3 transcripts against its nearest neighbor's
    g = [x for x in ("CD3D", "CD3E", "CD3G") if x in A.var_names]
    cd3tx = np.asarray(A[:, g].X.sum(1)).ravel()
    xy = A.obs[["x", "y"]].to_numpy(float)
    nn = cKDTree(xy).query(xy, k=2)[1][:, 1]
    st = A.obs["Mean.CD3"].to_numpy(float) if "Mean.CD3" in A.obs else None
    for lab in ("multinomial", "celltypist"):
        L = pd.read_csv(os.path.join(R, lab, "labels", f"{s}.labels.csv.gz"), keep_default_na=False).set_index("cell_id")
        L = L.reindex(A.obs_names.astype(str))
        tn = sorted({t for t in set(L["L2m_nucleus"]) if t and rx.search(t)})
        ok = (L["L2m_nucB"] != "") & (L["L2m_wholeB"] != "")
        rows.append({"sample": s, "unit": u, "labeller": lab, "tnk_types": "|".join(tn), "cells": int(ok.sum()),
                     "tnk_nucB": int(L.loc[ok, "L2m_nucB"].isin(tn).sum()),
                     "tnk_wholeB": int(L.loc[ok, "L2m_wholeB"].isin(tn).sum())})
        if st is not None:
            hi = st >= np.quantile(st, 0.95)                    # CD3-stain-high cells (top 5% of the section)
            r = {"sample": s, "unit": u, "labeller": lab, "cd3_high": int(hi.sum()),
                 "rho_stain_own_cd3tx": spearmanr(st, cd3tx)[0], "rho_stain_neighbour_cd3tx": spearmanr(st, cd3tx[nn])[0]}
            for arm in ("nucleus", "whole", "nucB", "wholeB"):
                c = L[f"L2m_{arm}"].to_numpy()
                m = hi & (c != "")
                r[f"tnk_share_cd3high_{arm}"] = float(np.isin(c[m], tn).mean()) if m.any() else np.nan
            prot.append(r)
T = pd.DataFrame(rows); T.to_csv(os.path.join(R, "tnk_sections.csv"), index=False)
U = T.groupby(["labeller", "unit"])[["tnk_nucB", "tnk_wholeB"]].sum().reset_index()
U["kept"] = U.tnk_wholeB / U.tnk_nucB.where(U.tnk_nucB >= 100)      # at least 100 T/NK labels from nucB
U.to_csv(os.path.join(R, "tnk_units.csv"), index=False)
print("T/NK types:", T.groupby("labeller").tnk_types.first().to_dict())
for lab, g in U.groupby("labeller"):
    print(f"{lab}: T/NK kept at equal depth (wholeB / nucB; patients with >= 100 nucB T/NK labels), per patient "
          f"{dict(zip(g.unit, g.kept.round(3)))}; T/NK labels from nucB {dict(zip(g.unit, g.tnk_nucB))}; "
          f"below one in {(g.kept < 1).sum()} of {g.kept.notna().sum()}")
if prot:
    P = pd.DataFrame(prot); P.to_csv(os.path.join(R, "cd3_stain.csv"), index=False)
    q = P[P.labeller == "multinomial"]
    print(f"CD3 stain vs CD3 transcripts, Spearman (median over sections): own cell {q.rho_stain_own_cd3tx.median():.3f}, "
          f"nearest neighbour {q.rho_stain_neighbour_cd3tx.median():.3f}  (the stain is usable only if own >> neighbour)")
    for lab, g in P.groupby("labeller"):
        print(f"{lab}: CD3-high cells labelled T/NK, median over sections: " + ", ".join(
            f"{a} {g[f'tnk_share_cd3high_{a}'].median():.3f}" for a in ("nucleus", "whole", "nucB", "wholeB")))
EOF
    ;;
  harvest)
    cd ${R:h} && zip -q -r ~/Downloads/cb_cosmx${TAG:+_$TAG}_$(date +%Y%m%d_%H%M).zip ${R:t} logs/cosmx_*.log(N) -x '*.labels.csv.gz' \
      && ls -lh ~/Downloads/cb_cosmx_*.zip(om[1]) ;;
  *) print "steps: list | unpack | convert | gate | test | harvest" ;;
esac
