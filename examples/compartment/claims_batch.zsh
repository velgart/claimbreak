#!/bin/zsh
# claims_batch.zsh -- published expression claims tested with nuclear counts
# (claims R2 and R3).
#   zsh $CB/examples/compartment/claims_batch.zsh check
#   caffeinate -ims zsh $CB/examples/compartment/claims_batch.zsh all 2>&1 | tee ~/claimbreak_results/logs/claims_$(date +%Y%m%d_%H%M).log
# r2  Baker et al. 2026 (AIN kidney, Xenium 5K): is CXCL9 in T cells their own (nuclear like CD3E, CD8A) or the
#     neighbors'? Authors' labels (author_label), cortex pieces, AIN / ATI / REF; CXCL9 -> CXCR3 pairs within 20 um;
#     in and out of the authors' lymphoid aggregates (author_tls).
# r3  He et al. 2022 (CosMx lung): is SPP1 in macrophages of the tumor interior their own? Our nuclear labels (L2m);
#     tumor interior = at least half tumor among the 50 nearest cells.
# Inputs are read only; outputs in ~/claimbreak_results/claims_r2, claims_r3 (each refuses to overwrite).
setopt no_bang_hist null_glob
CLD=${CLD:-~/claimbreak_data}; CB=${CB:-~/claimbreak}; CLR=${CLR:-~/claimbreak_results}
AIN_H5=${AIN_H5:-$CLD/AIN/h5ad}
CX_H5=${CX_H5:-$CLD/data/cosmx_nsclc/h5ad}; CX_LAB=${CX_LAB:-$CLR/cosmx_nsclc_x171/multinomial/labels}
NS=$CB/examples/compartment/nuclear_share.py
stamp() { print "\n=== $(date '+%H:%M:%S') $*" }
stop() { print "STOP: $*"; exit 1 }

check() {
  python -c "import claimbreak, sys; print('claimbreak', claimbreak.__version__); sys.exit(0 if tuple(map(int, claimbreak.__version__.split('.'))) >= (1, 7, 103) else 1)" \
    || stop "claimbreak 1.7.103 or later needed"
  ls $AIN_H5/*.h5ad >/dev/null 2>&1 && print "AIN h5ad: $(ls $AIN_H5/*.h5ad | wc -l | tr -d ' ') files" || stop "no AIN h5ad in $AIN_H5"
  python - $AIN_H5 <<'EOF'
import sys, glob, anndata as ad
f = sorted(glob.glob(sys.argv[1] + "/*.h5ad"))[0]
A = ad.read_h5ad(f, backed="r")
print("  obs:", [c for c in A.obs.columns if c in ("unit", "category", "tissue", "author_label", "author_tls", "x", "y")])
print("  layers:", list(A.layers.keys()), "; CXCL9, CXCR3 in panel:", "CXCL9" in A.var_names, "CXCR3" in A.var_names)
print("  tissue values:", dict(A.obs["tissue"].astype(str).value_counts()))
EOF
  ls $CX_H5/*.h5ad >/dev/null 2>&1 && print "CosMx h5ad: $(ls $CX_H5/*.h5ad | wc -l | tr -d ' ') files" || stop "no CosMx h5ad in $CX_H5"
  ls $CX_LAB/*.labels.csv.gz >/dev/null 2>&1 && print "CosMx labels: $CX_LAB" || stop "no CosMx labels in $CX_LAB"
}

r2() {
  [[ -e $CLR/claims_r2 ]] && stop "$CLR/claims_r2 exists; move it away"
  stamp "R2: CXCL9 in T cells (AIN, authors' labels, cortex)"
  python -u $NS --h5ad-dir $AIN_H5 --label-obs author_label \
    --groups "CD8=T_cell_CD8;CD4=T_cell_CD4,T_cell_DN,T_cell_gd;MAC=Macrophage;DC=Dendritic_cell,APC;PT=PT-INJ,PT-H,PT-dd;FIB=Fibroblasts,Myofibroblast,Stromal;EC=EC" \
    --gene CXCL9 \
    --markers "CD8=CD3E,CD3D,CD2,CD8A,CD8B;CD4=CD3E,CD3D,CD2;MAC=CD68,C1QA,C1QB,CD163,CD14;DC=CLEC9A,CD1C,FLT3,LAMP3;PT=LRP2,CUBN,SLC34A1;FIB=COL1A1,COL1A2,DCN,LUM;EC=PECAM1,VWF,CDH5" \
    --unit-col unit --cond-col category --keep "tissue=kidney_cortex" --pair-genes CXCL9,CXCR3 --pair-r 20 \
    --niche-col author_tls --out $CLR/claims_r2 2>&1 | tee $CLR/logs/claims_r2.log
}

r3() {
  [[ -e $CLR/claims_r3 ]] && stop "$CLR/claims_r3 exists; move it away"
  stamp "R3: SPP1 in macrophages (CosMx lung, nuclear labels, tumour interior)"
  python -u $NS --h5ad-dir $CX_H5 --labels $CX_LAB --arm L2m_nucleus \
    --groups "MAC=Myeloid;TUM=Tumour;T=T_NK;FIB=Fibro_muscle" --gene SPP1 \
    --markers "MAC=CD68,C1QA,C1QB,CD163,CD14;TUM=EPCAM,KRT8,KRT18,KRT19;T=CD3E,CD3D,CD2;FIB=COL1A1,COL1A2,DCN,LUM" \
    --unit-col unit --niche-type Tumour --niche-k 50 --niche-min 0.5 --out $CLR/claims_r3 2>&1 | tee $CLR/logs/claims_r3.log
}

harvest() {
  local DL=${CB_DOWNLOADS:-~/Downloads} Z
  Z=$DL/cb_claims_$(date +%Y%m%d_%H%M).zip
  (cd $CLR && zip -q -r $Z claims_r2 claims_r3 logs/claims_r2.log logs/claims_r3.log 2>/dev/null)
  ls -lh $Z
}

mkdir -p $CLR/logs
case $1 in
  check) check ;;
  r2) r2 ;;
  r3) r3 ;;
  harvest) harvest ;;
  all) check && r2 && r3; harvest ;;
  *) stop "usage: claims_batch.zsh check|r2|r3|harvest|all" ;;
esac
