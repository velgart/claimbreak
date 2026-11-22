# examples/ra

Rheumatoid-arthritis synovium (Xenium 5K): conversion to the compartment layout and the synovium references (CELLxGENE Census; GEO GSE200815).

| script | what it does |
|---|---|
| `gse200815_reference.py` | a synovium single-cell reference from GEO GSE200815, annotated at the coarse level by canonical lineage markers and, for information, by the AMP RA phase 2 cell states. |
| `ra_mac.zsh` | RA synovium as an ordinary dataset. |
| `ra_to_compartment.py` | RA synovium samples (xenium_to_h5ad.py output) -> compartment .h5ad files for `claimbreak compartment label` and the rest of the chain. |
| `synovium_reference.py` | a single-cell reference of human synovium from the CZ CELLxGENE Census, for labeling the RA Xenium data with `claimbreak compartment label`. |
