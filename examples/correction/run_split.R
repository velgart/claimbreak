#!/usr/bin/env Rscript
# run_split.R -- RCTD (doublet mode) and SPLIT on one exported section.
# Follows the SPLIT Xenium vignette (bdsc-tds/SPLIT, vignettes/Run_RCTD_and_SPLIT_on_Xenium.Rmd, v0.3.0):
#   RCTD: create.RCTD(UMI_min = 10, counts_MIN = 10, UMI_min_sigma = 100, CELL_MIN_INSTANCE = 25), doublet mode;
#         SPLIT::run_post_process_RCTD
#   split    SPLIT::purify(DO_remove_residual_contamination = TRUE) on convert_rctd_result_to_purify_input (default SPLIT;
#            singlets are purified by default since 0.2.0)
#   splitsp  spatially aware SPLIT: balance_raw_and_purified_data_by_score(threshold = 0.05,
#            score = neighborhood_weights_second_type; spatial network k 20, pruned at 15 um)
#   shift    SPLIT-shift: the same with DO_swap_lables = TRUE (transcriptomic network k 100 on SCT + PCA)
# The reference is the one used for the claimbreak labels (corr_export.py OUT/ref), at the same level.
#
#   Rscript run_split.R IN/<sample> IN/ref OUT/<sample> [cores]
# Writes OUT/<sample>/: rctd.csv (RCTD results per cell), {split,splitsp,shift}_counts.mtx + _cells.txt + genes.txt,
# split_meta.csv, shift_meta.csv (first_type after SPLIT-shift, swap), versions.txt. Refuses an existing OUT/<sample>.
suppressPackageStartupMessages({
  library(Matrix); library(spacexr); library(SPLIT); library(Seurat); library(dplyr)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: Rscript run_split.R IN/<sample> IN/ref OUT/<sample> [cores]")
sec_dir <- args[1]; ref_dir <- args[2]; out_dir <- args[3]
cores <- if (length(args) >= 4) as.integer(args[4]) else max(1, parallel::detectCores() - 1)
if (dir.exists(out_dir)) stop(out_dir, " exists; not overwritten")
dir.create(out_dir, recursive = TRUE)
say <- function(...) { cat(format(Sys.time(), "%H:%M:%S"), ..., "\n"); flush.console() }

read_mtx <- function(dir, stem = "counts", cells_file = "cells.csv", id_col = "cell_id") {
  M <- as(readMM(file.path(dir, paste0(stem, ".mtx"))), "CsparseMatrix")
  rownames(M) <- readLines(file.path(dir, "genes.txt"))
  cells <- read.csv(file.path(dir, cells_file), stringsAsFactors = FALSE, colClasses = c(cell_id = "character"))
  colnames(M) <- cells[[id_col]]
  list(M = M, cells = cells)
}
write_mtx <- function(M, stem) {
  M <- as(M, "CsparseMatrix")
  M@x <- round(M@x, 6)
  writeMM(M, file.path(out_dir, paste0(stem, "_counts.mtx")))
  writeLines(colnames(M), file.path(out_dir, paste0(stem, "_cells.txt")))
  writeLines(rownames(M), file.path(out_dir, paste0(stem, "_genes.txt")))
}

sec <- read_mtx(sec_dir)
ref_M <- as(readMM(file.path(ref_dir, "counts.mtx")), "CsparseMatrix")
rownames(ref_M) <- readLines(file.path(ref_dir, "genes.txt"))
ref_cells <- read.csv(file.path(ref_dir, "cells.csv"), stringsAsFactors = FALSE)
colnames(ref_M) <- ref_cells$cell
common <- intersect(rownames(sec$M), rownames(ref_M))
say("section", basename(sec_dir), ":", ncol(sec$M), "cells,", nrow(sec$M), "genes,", length(common), "in reference")

ref_types <- factor(ref_cells$type); names(ref_types) <- ref_cells$cell
ref.obj <- Reference(ref_M[common, ], cell_types = ref_types, min_UMI = 10, require_int = TRUE)
coords <- data.frame(x = sec$cells$x, y = sec$cells$y, row.names = sec$cells$cell_id)
test.obj <- SpatialRNA(coords = coords, counts = sec$M[common, ], require_int = TRUE)
RCTD <- create.RCTD(test.obj, ref.obj, UMI_min = 10, counts_MIN = 10, UMI_min_sigma = 100,
                    max_cores = cores, CELL_MIN_INSTANCE = 25)
say("RCTD doublet mode on", ncol(RCTD@spatialRNA@counts), "cells")
RCTD <- run.RCTD(RCTD, doublet_mode = "doublet")
RCTD <- SPLIT::run_post_process_RCTD(RCTD)
rd <- RCTD@results$results_df
rd$cell_id <- rownames(rd)
write.csv(rd, file.path(out_dir, "rctd.csv"), row.names = FALSE)
say("RCTD spot classes:", paste(names(table(rd$spot_class)), table(rd$spot_class), collapse = ", "))

counts <- sec$M[common, ]
pin <- SPLIT::convert_rctd_result_to_purify_input(rctd = RCTD)      # SPLIT >= 0.2.0 interface
res <- SPLIT::purify(counts = counts, primary_cell_type = pin$primary_cell_type,
                     deconvolution_weights = pin$deconvolution_weights, reference = t(pin$reference),
                     DO_remove_residual_contamination = TRUE, DO_output_sce = FALSE)
write_mtx(res$purified_counts, "split")
meta <- res$cell_meta; meta$cell_id <- rownames(meta)
write.csv(meta, file.path(out_dir, "split_meta.csv"), row.names = FALSE)
say("SPLIT purified:", ncol(res$purified_counts), "cells; status",
    paste(names(table(meta$purification_status)), table(meta$purification_status), collapse = ", "))

# spatially aware SPLIT and SPLIT-shift (vignette chunks spatial-nw, transcriptomics-nw, run-split-shift)
cells_rctd <- rownames(rd)
xe <- CreateSeuratObject(counts = counts[, cells_rctd], assay = "Xenium", meta.data = rd[cells_rctd, ])
sp_xy <- as.matrix(coords[cells_rctd, c("x", "y")]); colnames(sp_xy) <- c("ST_1", "ST_2")
xe[["spatial"]] <- CreateDimReducObject(sp_xy, assay = "Xenium", key = "ST_")
sp_nw <- SPLIT::build_spatial_network(xe, reduction = "spatial", dims = 1:2, DO_prune = TRUE, rad_pruning = 15,
                                      k_knn = 20)
sp_nw <- SPLIT::add_spatial_metric(spatial_neighborhood = sp_nw, rctd = RCTD)
xe <- AddMetaData(xe, SPLIT::neighborhood_analysis_to_metadata(sp_nw))
rm(sp_nw)
xe_pur <- CreateSeuratObject(counts = res$purified_counts, meta.data = res$cell_meta, assay = "Xenium")
xe_sp <- SPLIT::balance_raw_and_purified_data_by_score(xe_raw = xe, xe_purified = xe_pur, default_assay = "Xenium",
                                                       spot_class_key = "spot_class", threshold = 0.05,
                                                       score_name = "neighborhood_weights_second_type")
write_mtx(GetAssayData(xe_sp, assay = "Xenium", layer = "counts"), "splitsp")
xe <- SCTransform(xe, assay = "Xenium", verbose = FALSE)
xe <- RunPCA(xe, verbose = FALSE)
tr_nw <- SPLIT::build_transcriptomics_network(xe, DO_prune = FALSE, k_knn = 100)
tr_nw <- SPLIT::add_transcriptomics_metric(transcriptomics_neighborhood = tr_nw, rctd = RCTD)
xe <- AddMetaData(xe, SPLIT::neighborhood_analysis_to_metadata(tr_nw))
rm(tr_nw)
DefaultAssay(xe) <- "Xenium"
xe_shift <- SPLIT::balance_raw_and_purified_data_by_score(xe_raw = xe, xe_purified = xe_pur, default_assay = "Xenium",
                                                          spot_class_key = "spot_class", threshold = 0.05,
                                                          score_name = "neighborhood_weights_second_type",
                                                          DO_swap_lables = TRUE)
write_mtx(GetAssayData(xe_shift, assay = "Xenium", layer = "counts"), "shift")
sm <- xe_shift@meta.data
keep <- intersect(c("first_type", "second_type", "spot_class", "purification_status", "swap",
                    "first_type_before_swap"), colnames(sm))
sm <- sm[, keep, drop = FALSE]; sm$cell_id <- rownames(sm)
write.csv(sm, file.path(out_dir, "shift_meta.csv"), row.names = FALSE)
say("SPLIT-shift: swapped", sum(sm$swap), "of", nrow(sm), "cells")
writeLines(c(R.version.string, paste("spacexr", packageVersion("spacexr")), paste("SPLIT", packageVersion("SPLIT")),
             paste("Seurat", packageVersion("Seurat"))), file.path(out_dir, "versions.txt"))
say("done")
