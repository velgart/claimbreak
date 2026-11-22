# examples/ain

Acute interstitial nephritis kidney (Xenium): the kidney reference from the CELLxGENE Census and the checks of the authors' labels.

| script | what it does |
|---|---|
| `ain_author_arm.py` | add the authors' published labels as an arm `<level>_author` of the claimbreak labels files, mapped to our coarse classes, so that `claimbreak compartment calls / contacts / aggregates` can compare them with nucleus labels. |
| `ain_split.py` | split the converted Xenium regions of Baker et al. 2026 (acute interstitial nephritis, GSE336890) into one compartment .h5ad per patient and tissue region, using the authors' per-cell metadata. |
| `kidney_reference.py` | a single-cell reference of the adult human kidney from the CZ CELLxGENE Census, for labeling the acute interstitial nephritis Xenium data with `claimbreak compartment label`. |
| `xenium_count_check.py` | why do whole-cell totals from the transcript table differ from the platform's cell-feature matrix for some Xenium regions? |
