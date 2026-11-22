# examples/compartment

Scripts behind the nucleus-against-whole-cell analyses of the paper: conversion of Xenium, CosMx and MERSCOPE output to compartment .h5ad files, labeling drivers, the contact, exposure, region and protein analyses, and the simulations. Run the conversion first (`compartment_to_h5ad.py`, or `merscope_to_h5ad.py` with `merscope_nuclei.py`), then `claimbreak compartment label`, then the analysis scripts; each script's header gives its command.

| script | what it does |
|---|---|
| `baysor_tiles.py` | a compartment .h5ad from a Baysor segmentation of a Xenium section, run tile by tile. |
| `bilous_compare.py` | Do label changes between count sets land on the neighbors' types? |
| `bilous_consequence.py` | What organized label errors do to colocalization, against random relabeling at the same confusion rates. |
| `bilous_download.zsh` | GSE311609: 12 lung sections of the four donors with matched snRNA-seq (Suppl. Table 1), four files each, into ~/claimbreak_data/Bilous/raw. |
| `bilous_flipcells.py` | which cells do whole-cell counts relabel? |
| `bilous_label.py` | Label Xenium cells of GSE311609 from the matched single-nucleus reference, four ways per cell. |
| `bilous_placement.py` | same flips, different placement, held out by donor. |
| `bilous_to_h5ad.py` | One .h5ad per Xenium section of GSE311609, with whole-cell and nucleus-only counts of the same cells. |
| `cell_crop.py` | one real cell for Fig. 1a of the paper: a T cell whose nucleus is labeled T/NK but whose whole segmented cell is labeled tumor, next to a tumor cell, with its nucleus and mask outlines and the transcripts of T-cell and tumor markers. |
| `celltypist_batch.zsh` | the compartment test with a second labeler (CellTypist). |
| `claims_batch.zsh` | published expression claims tested with nuclear counts (claims R2 and R3). |
| `compartment_label.py` | label every cell of a compartment h5ad (compartment_to_h5ad.py or bilous_to_h5ad.py) from a single-cell reference, several ways per cell. |
| `compartment_to_h5ad.py` | one .h5ad per section with whole-cell and nucleus-only counts of the same cells, from Xenium or CosMx transcript tables. |
| `compartment_variants.py` | the same patient measured several ways (panel, segmentation stain, platform): label effects of whole-cell counts side by side. |
| `contact_contamination.py` | the neighbor's RNA in a cell, by contact, from whole-cell and from nucleus counts of the same cells. |
| `contact_contamination.zsh` | the partner's RNA in T/NK cells by contact, from whole-cell and nucleus counts of the same cells, in the tissues of region_check.zsh (gut, lung, synovium, kidney), on the scale of the Slide-GoTags isolated-nuclei control (slidetags_tcr.py contamination). |
| `contact_mislabel.zsh` | can the nuclear excess of partner RNA at contact (contact_contamination.zsh) come from partner cells labeled T/NK? |
| `contacts_checks_batch.zsh` | checks of the contact readouts, for lung, gut and kidney: (1) `claimbreak compartment contacts` with --mode composition (random relabeling that keeps the type mix) for the noise null (nucA -> nucB) and the |
| `contacts_decompose.py` | where does the per100 change come from? |
| `contamination.py` | how many of a cell's whole-cell counts belong to other cell types, where they come from, and how the label change depends on it. |
| `cosmx_label_exclude.zsh` | CosMx NSCLC labels with the over-detected probes excluded. |
| `cosmx_label_overdetected.zsh` | over-detected probes per section, all eight CosMx NSCLC sections. |
| `cosmx_label_settings.zsh` | which classifier settings label the CosMx NSCLC sections without collapse. |
| `cosmx_nsclc.zsh` | the compartment test on CosMx. |
| `cosmx_protein_check.py` | can CosMx same-cell protein (64-plex Immuno-Oncology panel) serve as protein truth? |
| `cytob_add.zsh` | add the pure-cytoplasm arm (cytoB) to label runs that lack it (lung bilous_v4, CosMx x171), then run the exposure analysis with cytoB as the test. |
| `exposure_batch.zsh` | labels kept against physical exposure to other types (analyses A and B). |
| `fit_flipmodel.py` | Fit the packaged flip models from the GSE311609 compartment labels (bilous_label.py output). |
| `hnscc_stain_check.py` | does the CosMx CD3 morphology stain identify T cells well enough to serve as protein truth? |
| `ibd_condition.py` | intraepithelial lymphocytes by condition, counted from nuclear, whole-cell and hybrid labels. |
| `ibd_condition.zsh` | intraepithelial lymphocytes per 100 epithelial cells by IBD condition (inflamed or not, disease, tissue: every characteristic in the GEO series matrix), counted from nuclear, whole-cell and hybrid labels of the same cell |
| `kept_ratio_sim.py` | does the label-only kept ratio recover the true loss? |
| `kernel_batch.zsh` | error kernel per dataset. |
| `merscope_crops.py` | test crops of a MERSCOPE DAPI plane for tuning nucleus segmentation. |
| `merscope_nuclei.py` | nuclei of a MERSCOPE section from its DAPI plane with StarDist. |
| `merscope_nucleus_qc.py` | does the DAPI threshold of merscope_to_h5ad.py outline nuclei? |
| `merscope_owners.py` | which cell owns each StarDist nucleus, by the platform's cell outlines. |
| `merscope_protein_check.py` | does MERSCOPE antibody co-staining mark the cell itself, or its region? |
| `merscope_to_h5ad.py` | whole-cell and nucleus-only counts of the same cells from a Vizgen MERSCOPE run. |
| `ms_batch.zsh` | MERSCOPE FFPE human uterine cancer with CD45RO co-staining (Vizgen showcase, HumanUterineCancerPatient2-ROCostain; 500 genes, 758,506 cells). |
| `nuclear_share.py` | is a gene a cell's own, or its neighbors'? |
| `per100_conf.zsh` | T/NK contacts per 100 partner cells under random relabeling at the real confusion rates (Fig. 4a). |
| `per100_fixed.zsh` | Fig. 4a with the partner cells fixed at the half-B labels. |
| `proseg_to_h5ad.py` | a compartment .h5ad (whole-cell, nucleus and cytoplasm counts of the same cells) from a Proseg segmentation of a Xenium section. |
| `protein_contacts.py` | T-cell contacts scored against protein on the same cells. |
| `protein_contacts.zsh` | T-cell contacts scored against protein in both renal carcinomas (Fig. 4 and Supplementary Table "protein contacts"). |
| `protein_position.zsh` | the control of Fig. 4a (the same label changes placed at random) in the two renal carcinomas, with partners typed by protein and by labels. |
| `protein_region.py` | the region analysis of region_loss.py with protein truth. |
| `reexpand.py` | the same cells, segmented again from their nuclei at several expansion distances, so that the mask is the only thing that changes. |
| `region_check.zsh` | checks of the region result of region_loss.zsh (gut: T/NK cells inside the epithelium lost about 69% of their labels, in the lamina propria about 25%). |
| `region_loss.py` | where in the tissue the immune cells lose their labels. |
| `region_loss.zsh` | where T/NK cells lose their labels, by the share of partner cells within 25 um (gut intraepithelial against lamina propria T cells; lung tumor nests against stroma; kidney and synovium with the Fig. 4a partners for comparison). |
| `regional_sim.py` | does the regionally matched extra-change ratio separate changes that follow the touching cells from changes that follow the region? |
| `relabel_excluding.py` | relabel a labels folder with extra genes excluded, every other setting taken from the folder's own meta.json files, then pool the new label audit. |
| `remedy_batch.zsh` | neighbor-aware labeling on the five compartment datasets, scored without protein (protein recall: ihc_truth.py truth on lung, ../correction/ihc_wsl.sh). |
| `robust_batch.zsh` | robustness checks. zsh $CB/examples/compartment/robust_batch.zsh check caffeinate -ims zsh $CB/examples/compartment/robust_batch.zsh thirds lung gut kidney ra cosmx 2>&1 / tee ~/claimbreak_results/logs/batchT_thirds_$(da |
| `rpgn_batch.zsh` | a published claim tested. |
| `rpgn_glom.py` | a published spatial claim tested with nuclear labels. |
| `segser_wsl.sh` | segmentation series (the same cells re-segmented from their nuclei at 0-15 um) on a Linux or WSL workstation. |
| `segseries_summary.py` | what the mask does to labels, on the same cells. |
| `slidetags_tcr.py` | negative control for the segmentation mechanism: are T cells with a recovered TCR labeled T less often next to tumor cells when there is no segmentation? |
| `state_depth.zsh` | which T cells lose their labels, matched on nuclear RNA depth (region result). |
| `stats_batch.zsh` | statistics for the paper on the existing labels (no relabeling): block bootstrap over 200 um tiles, sensitivity to the reference-label margin, pooled contamination shares with the homotypic baseline, and the flip curve w |
| `uterus_reference.py` | a single-cell reference of the adult human uterus (endometrium, myometrium, endometrial cancer) from the CZ CELLxGENE Census, for labeling the MERSCOPE uterine cancer data with `claimbreak compartment label`. |
| `vendor_calls.py` | the analyses a typical user runs, from whole-cell, nuclear, hybrid and protein labels of the same cells, per field of view: which calls change? |
| `vendor_calls.zsh` | the analyses a typical user runs (squidpy-style neighborhood enrichment, contacts per 100 partner cells, infiltration, the Keren mixing class), per 800 um field, from whole-cell, nuclear, hybrid and (Xenium Protein) prot |
| `xenium_protein.py` | protein on the same cells as the labels. |
| `xenium_protein_check.py` | the cosmx_protein_check.py tests on the protein we trust: Xenium Protein, renal carcinoma. |
| `xp_batch.zsh` | Xenium Protein, 10x FFPE human renal cell carcinoma (XOA 4.0; one section; 27 proteins on the same cells). |
| `xp_baysor.zsh` | renal carcinoma with Baysor cells: labels, compartment test and protein recall on the Baysor cells and on the deposited-mask cells that Baysor keeps, in the layout of the Proseg runs (<W>/proseg), so the numbers scripts  |
