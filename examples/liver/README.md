# examples/liver

Liver CosMx: reference mapping from the CELLxGENE Census, piece QC and FOV layout.

| script | what it does |
|---|---|
| `liver_batch.zsh` | the compartment test on the MASLD liver CosMx data. |
| `liver_label_check.py` | are the liver labels plausible? |
| `liver_pieces.py` | tissue pieces of the MASLD CosMx slides: quality per piece and the layout of the fields of view, to decide which pieces enter the compartment test and to map pieces to biopsies. |
| `liver_reference.py` | a single-cell reference of the adult human liver from the CZ CELLxGENE Census, for labeling the MASLD CosMx data with `claimbreak compartment label`. |
