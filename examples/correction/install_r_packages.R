# install_r_packages.R -- R packages for run_split.R and run_celladmix.R (Mac; CRAN binaries where available).
#   Rscript install_r_packages.R
repos <- "https://cloud.r-project.org"
need <- c("remotes", "BiocManager", "Matrix", "data.table", "dplyr", "Seurat", "NMF", "Rfast", "arrow", "scatterpie",
          "doParallel", "foreach", "quadprog", "readr", "pals", "fields", "CompQuadForm", "locfdr", "metafor",
          "forcats", "entropy", "gtools", "igraph", "FNN", "nnls", "ggridges", "ggrastr", "RANN", "pROC")
miss <- need[!sapply(need, requireNamespace, quietly = TRUE)]
if (length(miss)) install.packages(miss, repos = repos)
bioc <- c("SingleCellExperiment", "rhdf5", "BiocParallel", "glmGamPoi")
miss <- bioc[!sapply(bioc, requireNamespace, quietly = TRUE)]
if (length(miss)) BiocManager::install(miss, update = FALSE, ask = FALSE)
gh <- c(CRF = "wulingyun/CRF", sccore = "kharchenkolab/sccore", spacexr = "dmcable/spacexr",
        SPLIT = "bdsc-tds/SPLIT", cellAdmix = "kharchenkolab/cellAdmix")
for (p in names(gh)) if (!requireNamespace(p, quietly = TRUE)) remotes::install_github(gh[[p]], upgrade = "never")
for (p in c(names(gh), "Seurat", "NMF")) cat(p, as.character(packageVersion(p)), "\n")
