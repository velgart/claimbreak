# How many labels does your segmentation cost? `claimbreak kept-ratio`

Every segmentation gives each cell some RNA of its neighbors, and cell-type labels called from those counts are
pulled toward the neighbors. Small, RNA-poor cells (T and NK cells, most of all) lose their labels to larger cells
they touch. `claimbreak kept-ratio` measures this in your own Xenium or CosMx data, without protein or manual
annotation. It needs only the nucleus flag that both platforms record for each transcript, and a single-cell
reference for the tissue.

## Install

    python -m pip install "claimbreak[xenium] @ git+https://github.com/velgart/claimbreak"   # or, from a copy:
    python -m pip install "./claimbreak[xenium]"

Python 3.10 or later; numpy, scipy, pandas, anndata, scikit-learn. No R, GPU or compiled tools.

## Run

Xenium (the instrument's output folder, or a folder holding several):

    claimbreak kept-ratio --xenium path/to/output-XETG.../outs --ref reference.h5ad --ref-level cell_type \
        --out kept_check

CosMx, or Xenium files with GEO-style prefixes: convert first with
`examples/compartment/compartment_to_h5ad.py` (one .h5ad per section), then

    claimbreak kept-ratio --h5ad sections/*.h5ad --ref reference.h5ad --ref-level cell_type --out kept_check

The reference is an .h5ad of raw counts with the cell types in an obs column (`--ref-level`). Merge fine types with
`--ref-merge 'regex=>label;...'`. Use the coarsest level that answers your question: labels must be stable at
nuclear depth for the check to mean anything (below). Sections from the same donor or patient can be pooled with
`--unit-pattern`, a regular expression that finds the donor in the section name (default: every section is its own
unit).

Nothing is written outside `--out`, and the input folders are only read.

## What you get

`kept_check/report.md` says what was found, in words. The tables behind it:

| file | content |
|---|---|
| `kept_ratio.csv` | per unit and type: kept ratio, 95% block-bootstrap interval, cells in the null |
| `sections.csv` | per section: agreement of the two nuclear halves, whether it was used |
| `audit/` | over-detected genes (`exclude_overdetected.txt`), marker support per class |
| `summary/` | extra-change ratio per unit, pair excess, every intermediate table |
| `labels/`, `compare/`, `h5ad/` | the labels of every arm, the label changes, the converted counts |

## How it works, and how to read it

Each cell is labeled three times at the same depth: from one random half of its nuclear transcripts (A), from the
other half (B), and from its whole-cell transcripts minus A, thinned to the depth of B (wholeB). A and B differ only
by counting noise, so B against A is the null.

- **Halves agreement** is the share of cells whose label from A equals the label from B. Below about 0.6, labels
  change a lot from counting noise alone and the other numbers are unreliable. A coarser reference level, or a
  reference closer to the tissue, usually helps.
- **Kept ratio** of a type: of the cells labeled that type from A, the share that wholeB also calls that type, over
  the share that B calls that type. One means the whole cell loses no more labels than counting noise; 0.7 means
  the whole-cell labels lose 30% of that type relative to the nucleus. In four Xenium tissues, T and NK cells kept
  0.56–0.77 of their labels, and the kept ratio agreed with the loss measured by protein on the same cells within
  0.05.
- **Null keeps** of a type: of the cells labeled that type from A, the share that B also calls that type. It is
  the denominator of the kept ratio. Below 0.5 the type is not stable at nuclear depth, its kept ratio can stray far
  from one (even above one), and the report marks the row "unstable". This happens most for types the reference
  describes poorly, or that the tissue lacks while it holds a type the reference lacks (tumor cells against a
  normal-tissue reference take the labels of several normal types).
- **Extra-change ratio**: of the label changes beyond the null, the share that take a touching neighbor's type,
  over the share expected if changes ignored the neighbors. Above one, label changes follow the neighbors.
- **Label audit**: genes detected far above their share in every reference type can act as false markers (in one
  synovium dataset, a single probe created most T/NK calls). They can also be real signal of a cell state the
  reference lacks (as VIM may be in a renal carcinoma section labeled against a normal kidney reference). Review
  `audit/overdetected.csv`; exclude probe artifacts by rerunning into a new folder with
  `--exclude-genes kept_check/audit/exclude_overdetected.txt`, and fix missing cell states with a better reference.

Limits worth keeping in mind:

- The nucleus also carries some neighboring RNA, so the loss is a lower bound.
- The kept ratio counts labels lost, not labels gained. The types that absorb the lost labels are not scored.
- Two segmentations can be ranked by their kept ratios only if they share the nuclear counts, as nuclear expansions
  of the same section do. A method that also changes the nucleus (some transcript-based segmentations do) needs
  protein or another external reference.
- Errors that nucleus and whole cell share (from the reference, the label granularity or the classifier) are
  invisible to this check.

## What to do with a low kept ratio

Label small cell types from their nuclear counts, and large ones from the whole cell: in two renal carcinoma
donors with protein on the same cells, nuclear labels recalled more T, CD8, CD4 and B cells than any count set
that included the cytoplasm, while myeloid cells were recalled better by the whole cell. resolVI's relabeling
recalled more T cells still, but at lower T precision and with costs in other classes that varied between donors and
runs; check any relabeling against protein before relying on it (`examples/correction/resolvi_rcc.sh`). Do not test
contact or
colocalization claims against random relabeling alone: random label errors move these statistics in the opposite
direction to the errors segmentation makes.

## Cite

Elgart V, Loscalzo J. Measuring and reducing segmentation-driven cell-type labeling errors in imaging spatial transcriptomics (2026). Software: claimbreak, Zenodo, doi:10.5281/zenodo.<record> (see `CITATION.cff`).
