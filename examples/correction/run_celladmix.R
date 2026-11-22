#!/usr/bin/env Rscript
# run_celladmix.R -- cellAdmix (Mitchel et al., Nat Genet 2026) on one exported section. Follows the package's NSCLC tutorial (kharchenkolab/cellAdmix, vignettes/NSCLC_tutorial_fulldata):
# joint NMF of molecular neighborhoods (k factors, h = 20, 10,000 molecules from up to 2,000 cells per type), CRF
# assignment of every molecule (num.nn 10, same.label.ratio 5, normalize.by gene), cell-bridging annotation
# (ncells.samp 200, knn.k 20, p < 0.1 unadjusted), false-positive check (knn_k 100, median_thresh 0.1), then removal of
# the molecules of factors called foreign to the cell's type. The cell types given to cellAdmix are the analyst's
# whole-cell labels (cells.csv column <level>_whole), as an analyst would use them.
#
#   Rscript run_celladmix.R IN/<sample> OUT/<sample> [cores] [k] [level]
# Writes OUT/<sample>/: admix_counts.mtx + admix_cells.txt + admix_genes.txt (cleaned whole-cell counts, genes x cells,
# every exported cell), removed.csv (factor x cell type pairs removed, molecules removed), loadings.csv (NMF gene
# loadings), nmf_crf.rds (NMF and CRF results; a rerun after an interruption resumes from it), versions.txt.
# Refuses a completed OUT/<sample>.
suppressPackageStartupMessages({ library(Matrix); library(cellAdmix); library(data.table) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript run_celladmix.R IN/<sample> OUT/<sample> [cores] [k] [level]")
sec_dir <- args[1]; out_dir <- args[2]
cores <- if (length(args) >= 3) as.integer(args[3]) else max(1, parallel::detectCores() - 1)
K <- if (length(args) >= 4) as.integer(args[4]) else 12L
level <- if (length(args) >= 5) args[5] else "L2m"
if (file.exists(file.path(out_dir, "admix_counts.mtx"))) stop(out_dir, " is complete; not overwritten")
resume <- file.exists(file.path(out_dir, "nmf_crf.rds"))       # an interrupted run: NMF and CRF are kept
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
say <- function(...) { cat(format(Sys.time(), "%H:%M:%S"), ..., "\n"); flush.console() }
set.seed(20261012)

cells <- fread(file.path(sec_dir, "cells.csv"), colClasses = list(character = "cell_id"))
genes <- readLines(file.path(sec_dir, "genes.txt"))
df <- fread(cmd = paste("gzip -dc", shQuote(file.path(sec_dir, "molecules.csv.gz"))),   # no R.utils needed
            colClasses = list(character = c("cell", "mol_id")))
df <- as.data.frame(df)
df$celltype[is.na(df$celltype)] <- ""
lab <- cells[[paste0(level, "_whole")]]; lab[is.na(lab)] <- ""
cell_meta <- data.frame(x = cells$x, y = cells$y, z = 0, cell = cells$cell_id, celltype = lab,
                        row.names = cells$cell_id, stringsAsFactors = FALSE)
cell_meta <- cell_meta[cell_meta$celltype != "", ]
dfl <- df[df$celltype != "", ]
rownames(dfl) <- dfl$mol_id
say("section", basename(sec_dir), ":", nrow(cell_meta), "labelled cells,", nrow(dfl), "molecules; types",
    paste(names(table(cell_meta$celltype)), table(cell_meta$celltype), collapse = ", "))

if (resume) {
  saved <- readRDS(file.path(out_dir, "nmf_crf.rds")); res <- saved$res; crf_res <- saved$crf_res
  say("resumed: NMF and CRF read from nmf_crf.rds")
} else {
  df_sub <- samp_ct_equal(dfl, cell_meta, num.cells.samp = 2000)
  res <- run_knn_nmf(df_sub, k = K, h = 20, nmol.dsamp = 10000, n.cores = cores)
  say("NMF done (k =", K, ")")
  crf_res <- run_crf_all(dfl, res, num.nn = 10, same.label.ratio = 5, normalize.by = "gene", n.cores = cores)
  say("CRF done")
  saveRDS(list(res = res, crf_res = crf_res, k = K), file.path(out_dir, "nmf_crf.rds"))
}
knn_res <- cellAdmix:::get_mol_knn(dfl, knn_k = 10)        # not exported by cellAdmix 0.1.0
bridge_raw <- run_bridge_test(dfl, crf_res, cell_meta, knn_res[[1]], knn_res[[2]], ncells.samp = 200, knn.k = 20,
                              n.cores = cores)
all_ctypes <- unique(cell_meta$celltype)
bridge_res <- extract_bridge_res(bridge_raw, all.ctypes = all_ctypes, n.factors = K, nmf.type = "joint",
                                 p.thresh = .1, adj.pvals = FALSE)
fp_checks <- cellAdmix:::check_fp(dfl, cell_meta, crf_res, bridge_res, do_clean = TRUE, knn_k = 100,
                                  median_thresh = .1)
say("factor x type pairs removed:", length(fp_checks), ":", paste(fp_checks, collapse = " "))
dfl$factor <- crf_res[, 1]
rm_mol <- paste(dfl$factor, dfl$celltype, sep = "_") %in% fp_checks
removed <- as.data.frame(table(pair = paste(dfl$factor, dfl$celltype, sep = "_")[rm_mol]))
write.csv(removed, file.path(out_dir, "removed.csv"), row.names = FALSE)
H <- tryCatch(res@fit@H, error = function(e) NULL)
if (!is.null(H)) write.csv(as.data.frame(t(H)), file.path(out_dir, "loadings.csv"))

# cleaned counts for every exported cell: original molecules of unlabeled cells kept, removed molecules dropped
keep <- rep(TRUE, nrow(df))
keep[match(dfl$mol_id[rm_mol], df$mol_id)] <- FALSE
dk <- df[keep, ]
gi <- match(dk$gene, genes); ci <- match(dk$cell, cells$cell_id)
ok <- !is.na(gi) & !is.na(ci)
M <- sparseMatrix(i = gi[ok], j = ci[ok], x = 1, dims = c(length(genes), nrow(cells)),
                  dimnames = list(genes, cells$cell_id))
writeMM(M, file.path(out_dir, "admix_counts.mtx"))
writeLines(colnames(M), file.path(out_dir, "admix_cells.txt"))
writeLines(rownames(M), file.path(out_dir, "admix_genes.txt"))
say("molecules removed:", sum(rm_mol), "of", nrow(df), sprintf("(%.3f)", sum(rm_mol) / nrow(df)))
writeLines(c(R.version.string, paste("cellAdmix", packageVersion("cellAdmix")), paste("k", K)),
           file.path(out_dir, "versions.txt"))
say("done")
