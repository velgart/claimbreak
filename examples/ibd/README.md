# examples/ibd

The IBD pair (Li et al., Nat Commun 2026; GEO GSE312415 CosMx, GSE312420 Xenium): conversion, depth QC, CosMx diagnostics and the gut reference. In the GSE312415 1k-panel transcript files every transcript is listed twice with identical fields; `compartment_to_h5ad.py` keeps one copy, after which per-cell totals equal the deposited count matrix exactly.

| script | what it does |
|---|---|
| `cellscape_check.py` | is each CellScape protein row on the right CosMx cell? |
| `cellscape_classes.py` | major cell classes from CellScape protein on the same cells as CosMx. |
| `cosmx_background.py` | why do CosMx cells fall into one broad class? |
| `cosmx_exprmat_diag.py` | why do whole-cell totals from a CosMx tx_file disagree with the deposited exprMat? |
| `cosmx_pairs.py` | the two copies of each transcript in a doubled CosMx tx_file: where do they differ, and which compartment call is more plausible? |
| `cosmx_stain_rna.py` | do CosMx's own stains agree with CosMx RNA on the same cell? |
| `cosmx_tx_dups.py` | are transcripts repeated in a CosMx tx_file? |
| `eval_protein.py` | transcript labels against protein classes on the same cells. |
| `gut_reference.py` | a single-cell reference of the human small and large intestine from the CZ CELLxGENE Census, for labeling the IBD spatial data with compartment_label.py. |
| `ibd_convert.zsh` | IBD pair: convert every Xenium sample and every CosMx slide to compartment h5ad files on the drive. |
| `ibd_qc.py` | one row per converted compartment h5ad: depth and how many cells survive a count threshold. |
| `label_drivers.py` | which genes make the labeler call a class? |
| `seurat_export.R` | For the Li et al. 2026 figshare object (doi:10.6084/m9.figshare.31595527), '3_smi_so_qcnorm.RDS': CosMx counts with the authors' protein-gated labels (ct_major_new, ct_minor_new), Max.PanCK / Max.CD45 and the gate category. |
| `seurat_to_h5ad.py` | files from seurat_export.R (+ optional CellScape tables) -> one .h5ad. |
