# examples/correction

Correction methods (SPLIT, RCTD, cellAdmix, resolVI, Baysor, Proseg) and the protein truth from the post-Xenium IHC of the lung sections. The R scripts need the packages in `install_r_packages.R`; `proseg_wsl.sh` and `ihc_wsl.sh` drive the runs on a Linux or WSL workstation.

| script | what it does |
|---|---|
| `baysor_batch.sh` | Baysor as a second transcript-based segmentation. |
| `corr_export.py` | inputs for the correction test (SPLIT, cellAdmix) on GSE311609 lung sections. |
| `corr_import.py` | labels of corrected counts, as arms of the claimbreak labels files. |
| `corr_readouts.py` | readouts C1-C6 from the labels written by corr_import.py and the `claimbreak compartment contacts` runs on them. |
| `ihc_channel_check.py` | which protein does each deposited Opal channel of the lung IHC carry? |
| `ihc_truth.py` | protein identity (post-Xenium multiplex IHC, CD8 and pan-cytokeratin) as ground truth for the labels of the same cells. |
| `ihc_wsl.sh` | lung IHC ground truth on a Linux or WSL workstation. |
| `install_r_packages.R` | Rscript install_r_packages.R |
| `make_synth.py` | Synthetic lung-like data: reference h5ad, one section h5ad, transcripts parquet; T cells next to tumor receive tumor transcripts in the cytoplasm (spill-over). |
| `per100_check.py` | why one donor's T/NK per 100 tumor cells is far from the others. |
| `proseg_wsl.sh` | Proseg runs and protein checks on a Linux or WSL workstation. |
| `reg_diag.py` | is an IHC -> Xenium nucleus registration right? |
| `resolvi_rcc.py` | resolVI's semi-supervised relabeling on renal carcinoma windows, for scoring against protein. |
| `resolvi_rcc.sh` | resolVI's semi-supervised relabeling on the two renal carcinoma donors, scored against protein. |
| `run_celladmix.R` | cellAdmix on one exported section. |
| `run_correction.zsh` | the correction test on the GSE311609 Lung-panel sections, end to end. |
| `run_split.R` | RCTD (doublet mode) and SPLIT on one exported section. |
