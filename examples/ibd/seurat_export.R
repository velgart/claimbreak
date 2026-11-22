# seurat_export.R -- export a Seurat (v4/v5) object to plain files for seurat_to_h5ad.py (read only).
# For the Li et al. 2026 figshare object (doi:10.6084/m9.figshare.31595527), '3_smi_so_qcnorm.RDS': CosMx counts with
# the authors' protein-gated labels (ct_major_new, ct_minor_new), Max.PanCK / Max.CD45 and the gate category.
#
# Usage:  Rscript seurat_export.R 3_smi_so_qcnorm.RDS OUT_DIR [ASSAY]
# Writes OUT_DIR/counts.mtx (genes x cells, raw counts), genes.txt, cells.txt, meta.csv (all meta.data columns).
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript seurat_export.R OBJECT.RDS OUT_DIR [ASSAY]")
obj <- readRDS(args[1])
out <- args[2]
if (dir.exists(out) && length(list.files(out)) > 0) stop(paste(out, "exists and is not empty"))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
assay <- if (length(args) >= 3) args[3] else DefaultAssay(obj)
cat("assays:", paste(Assays(obj), collapse = ", "), "; using", assay, "\n")
m <- tryCatch(LayerData(obj, assay = assay, layer = "counts"),                  # Seurat v5
              error = function(e) GetAssayData(obj, assay = assay, slot = "counts"))  # Seurat v4
m <- as(m, "CsparseMatrix")
cat("counts:", nrow(m), "genes x", ncol(m), "cells; integer:", all(m@x == round(m@x)), "\n")
writeMM(m, file.path(out, "counts.mtx"))
writeLines(rownames(m), file.path(out, "genes.txt"))
writeLines(colnames(m), file.path(out, "cells.txt"))
md <- obj@meta.data[colnames(m), , drop = FALSE]
write.csv(md, file.path(out, "meta.csv"), row.names = TRUE)
cat("meta.data columns:", paste(colnames(md), collapse = ", "), "\n")
for (cn in intersect(c("ct_major_new", "ct_major", "category"), colnames(md))) {
  cat("\n", cn, ":\n"); print(sort(table(md[[cn]]), decreasing = TRUE))
}
