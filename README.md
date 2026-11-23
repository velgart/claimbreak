# claimbreak

Measures how segmentation changes cell-type labels in imaging spatial transcriptomics, without ground truth, and
stress-tests label-based spatial conclusions under controlled label and count errors. It accompanies Elgart V,
Loscalzo J, *Measuring and reducing segmentation-driven cell-type labeling errors in imaging spatial
transcriptomics* (2026; preprint identifier to be added). Software archive: doi:10.5281/zenodo.23987641.

## Install

    git clone https://github.com/velgart/claimbreak claimbreak && cd claimbreak
    python -m pip install -e ".[test,xenium]"
    python -m pytest -q            # synthetic data only

Pure Python (numpy, scipy, pandas, anndata, scikit-learn, pyyaml); no R, GPU or compiled tools. Runs the same on
macOS, Linux and Windows (native Python or WSL2).

## Check your own data: how many labels does segmentation cost?

    claimbreak kept-ratio --xenium path/to/xenium/outs --ref reference.h5ad --ref-level cell_type --out kept_check

Labels every cell from halves of its nuclear counts and from its whole-cell counts at the same depth, and reports,
per cell type, the share of labels the whole cell keeps (the kept ratio), whether label changes follow the touching
neighbors, and a label audit. Read `kept_check/report.md` first. CosMx and GEO-style files: convert with
`examples/compartment/compartment_to_h5ad.py`, then `--h5ad`. MERSCOPE: segment nuclei from the DAPI image first
(`examples/compartment/merscope_nuclei.py`). Details and how to read the numbers: `docs/kept_ratio.md`.

## Compartment analyses
Cells labeled from different subsets of their own counts (nucleus, whole cell, cytoplasm, independent halves),
against a single-cell reference; then where label changes land and what they do to colocalization. Inputs are one
.h5ad per section with X = whole-cell counts and layers `nucleus`, `cyto` (`examples/compartment/compartment_to_h5ad.py`
makes them from Xenium or CosMx output).

    claimbreak compartment label --ref REF.h5ad --inputs H5AD/*.h5ad --out LABELS \
        --ref-levels L1=coarse --ref-merge '' --ref-donor donor_id --arms whole,nucleus,nucA,nucB,cytoB,wholeB [--ref-cv]
    claimbreak compartment compare --labels LABELS --out LABELS/compare --levels L1 --comparisons "nucB|nucA,cytoB|nucA,wholeB|nucA"
    claimbreak compartment consequence --labels LABELS --out LABELS/conseq --level L1 --ref nucB --test wholeB
    claimbreak compartment summary --compare LABELS/compare --consequence LABELS/conseq --out LABELS/summary
    claimbreak compartment placement --labels LABELS --out LABELS/placement --level L1 \
        --immune T_NK,Myeloid --partners Epithelial,Fibroblast --use-sections LABELS/summary/sections.csv

Label audit. Every `compartment label` run audits the nucleus and whole-cell arms and writes the result to
`<sample>.audit.csv`, `meta.json` (`label_audit`) and the warnings; the labels themselves are unchanged. Three
checks, none needing protein: genes over-detected beyond every reference type (they act as false markers of the
type with their highest share), the share of a type's calls that exist only through such a gene, and the share of a
type's calls with one of its own marker transcripts. Pool the sections and relabel without the listed genes:

    claimbreak compartment audit --labels LABELS --out LABELS/audit           # LABELS/audit/exclude_overdetected.txt
    claimbreak compartment label ... --exclude-genes LABELS/audit/exclude_overdetected.txt --out LABELS_EXCL

In synovium (Xenium 5K, GSE200815 reference) CCL5 held 4-14% of nuclear counts against 0.65% in reference T cells
and turned low-count fibroblasts into T/NK calls; excluding it alone brought the nuclear T share from 0.19 to 0.08,
next to the dataset authors' 0.07. Review the list before excluding: by default it also
holds low-abundance genes above 2x a tiny reference share; `--min-sections` and the `audit_min_share` floor (0.5% of
counts) keep it to genes that matter.

`placement` (the packaged form of `examples/compartment/bilous_placement.py`, without its criteria
file) applies each section's excess changes (wholeB against nucB, beyond the nucA null) to the nucB labels, placed
at random, by ambiguity, on the cells that really changed (oracle), by contact kernels or by flip models: the
packaged lung models as fixed (`transfer_default`, `transfer_no_nucleus`; `--model NAME=JSON` for others) and models
refitted leaving one unit out (`louo_full`, `louo_no_nucleus`). Units (patients, donors) come from
`--unit-pattern` on the sample name. Readout: sign agreement of the colocalization change with the real one, by
pair over units and within each unit; `models.json` has the held-out within-pair AUC of every model.
`--save-models` writes the models fitted on all units, for transfer to another dataset. `--nbr-arm whole` types the
neighbors by whole-cell labels (data without a nucleus split).

    claimbreak compartment calls --labels LABELS --out LABELS/calls --level L1 --ref nucleus --test whole \
        --immune T_NK,Myeloid --partners Epithelial,Fibroblast --use-sections LABELS/summary/sections.csv

    claimbreak compartment contacts --labels LABELS --out LABELS/contacts --level L1 --immune T_NK,Myeloid,B \
        --partners Epithelial --mixing-immune T_NK,Myeloid,B,Plasma,Mast [--contact boundary --boundaries RAW]

`contacts`: contact-scale readouts from the reference and the analyst's labels and under the field's random
stress test: infiltration (share of immune cells touching the partner type), immune cells touching the partner per
100 partner cells, and the Keren et al. (Cell 2018) mixing score (immune-partner over immune-immune contacts; cold
< 250 immune cells, compartmentalized < 0.22, mixed), per section and per unit from summed contacts. Contacts: 6
nearest cells within 30 um, or touching segmentation boundaries (Xenium cell_boundaries.parquet; needs pyarrow).

    claimbreak compartment exposure --labels LABELS --h5ad-dir H5AD --out LABELS/exposure --levels L1 [--gaps 2,4,6]

`exposure`: does a cell keep its label from whole-cell counts when no cell of another type touches it?
Each cell's own type is its label from one nuclear half (nucA); among those cells, the test (wholeB) and the
counting-noise null (nucB) keep the type or not, and the ratio kept_test / kept_null (same cells) is resolved by
physical exposure: nuclei of another type within a nucleus gap g (2, 4, 6 um; nucleus radii from obs
`nucleus_area`, else the section median, else 3 um), stratified by all nuclei within g; the share of the cell's
border held by other types; and the RNA-weighted exposure (neighbors' whole-cell counts over the cell's own,
weighted by border share). Neighbor-driven loss is absent at zero exposure and grows with it; intrinsic loss is the
same at every exposure. `destinations.csv` / `dest_fit.csv`: the excess rate a -> b against the share of the border
held by b (an intercept above zero is a change toward b without b at the border). Outputs: strata.csv,
destinations.csv, units.csv, dose.csv, tissue.csv, dest_units.csv, dest_fit.csv, notes.txt. Use `--test cytoB`
(cytoplasm only) and `--contact-gap 10` when comparing strata, so that the test arm has the same composition in every
stratum and contact means touching masks; `--regions` adds destinations against the regional composition
(dest_region*.csv) and the scheme `outside` crosses the outside-nucleus share with the border share.

    claimbreak compartment label ... --arms nucA,nucB,cytoB,cytoM,nucleus,whole,wholeM,wholeMX

Neighbor-aware arms (`claimbreak.decontam`): the counts outside the nucleus are modeled as the
cell's own profile mixed with an expected contamination profile built from geometry and the neighbors' nucleus
labels only (touching cells by border share at `--decon-gap` 10 um, cells within `--decon-region` 20 um, section
ambient; each weighted by its RNA). The foreign share psi_i = kappa E_i / (1 + kappa E_i) grows with the cell's
exposure E_i (foreign RNA per unit of its own), with one catchment constant kappa per section and arm fitted by
maximum likelihood (meta.json `decon_kappa`). `wholeM`: nuclear counts plus the outside counts so modeled (the
remedy as a user runs it); `cytoM`: the cytoB counts so modeled (the compartment test of the remedy: include cytoB
in the arms so both arms use the same counts); `wholeMX`: all counts treated as outside (no nucleus split). Extra
columns `<level>_<arm>_psi`, `border_covered`, `foreign_exposure`. Multinomial labeler only.

Block bootstrap: `compartment compare` writes per-tile sums (`--tile 200` um) and `compartment summary`
resamples the tiles of each unit (`--block-draws 1000`): `block_contact_lo/hi` in units.csv next to the parametric
interval, and `units_kept.csv` (kept ratio of the `--focus` types, test over null, with its block interval).

Robustness options: arms `nucA3,nucB3,nucC3,wholeB3,cytoB3` (nuclear thirds) with `compartment compare
--agree nucC3 --null "nucB3|nucA3"` (changes from labels an independent third confirms: departures, not corrections);
`compartment compare --contact boundary --boundaries DIR` (contact = touching segmentation boundaries); `compartment
label --prior fit` (class prior fitted per section and arm). Driver: `examples/compartment/robust_batch.zsh`.

`calls`: neighborhood-enrichment calls (kNN, permutation z, |z| >= 2 and |log2 O/E| >= 0.2) from the
reference labels and from the analyst's labels, the status of each claim (manufactured, exaggerated, attenuated, lost,
reversed), and the field's stress test on the analyst's labels (5, 10, 15% reassigned at random, 10 draws): does the
call survive, and which way does random error move it. A claim is "certified wrongly" when whole-cell labeling made
or inflated it and the random test calls it robust while moving it toward zero.

Same options and outputs as the example scripts. Check the composition printed for each section (and the WARNING
lines) before reading any comparison: a label collapse is a classifier failure, not biology. Platform-factor
options: `--gamma-iter 0`, `--gamma-clip`, `--gamma-damp`, `--gamma-fit soft`.

## Stress-test the claims of one analysis

    claimbreak stress data.h5ad --label-key cell_type --section-key sample --out results/stress \
        [--question colocalization] [--eps 0.05,0.1] [--reps 5] \
        [--nucleus-layer nucleus --cell-area-key cell_area --nucleus-area-key nucleus_area]

For every claim (default: kNN neighborhood enrichment of every ordered pair, homotypic pairs included) the same
numbers of label errors per pair are placed twice: at random, as published robustness tests do, and where
whole-cell relabeling puts them (`claimbreak.flipmodel`: cells with much of their counts outside the nucleus,
touching the destination type, called with moderate confidence). `report.md` lists the claims where the two
placements disagree: opposite directions, or robust to random errors but not to realistic ones. Without a nucleus
layer the model drops the nucleus-based features (and uses the packaged no-nucleus model once fitted). Python:
`claimbreak.stress.stress_test(tissue, ...)`.

## Stress-test framework

Plant known signals in a real tissue, add label and count errors in a controlled way, and record which
conclusions survive, disappear, flip or appear. The accompanying paper's results do not use this framework.

    claimbreak inspect data.h5ad                     # lists obs columns, layers, obsm keys (metadata only)
    claimbreak inspect data.h5ad --label-key cell_type --section-key section --unit-key mouse --condition-key group
    cp examples/template.yaml my_experiment.yaml     # edit keys, questions, perturbations
    claimbreak check my_experiment.yaml              # loads the data and applies the planting only (seconds)
    claimbreak run my_experiment.yaml
    claimbreak summarize results/claimbreak/my_experiment
    claimbreak explain results/claimbreak/my_experiment --question composition   # which claims changed, and how often

Runs resume: rows already in `summary.csv` are skipped. `run` stops with a message if the config still contains
`REPLACE` placeholders or the data file is missing (for example, the drive is not mounted). Outputs are refused
inside the data directory, and the input file is only read.

With more than two conditions, give the contrast explicitly: `{name: composition, conditions: [ref, alt]}` and
`{name: niches, conditions: [ref, alt]}`. `data.subset: {column: [values]}` keeps only selected cells; a .h5ad
path is opened in backed mode, so only those cells are read. Counts must be raw integers.

In YAML, write the null planting as `{kind: "null"}` (quoted); an unquoted null is also accepted.

## Order of operations

1. load the tissue; its labels are the reference ("truth")
2. plant known signals on the full counts: `null` (profiles permuted within section and label: every
   neighborhood effect beyond identity removed), `spike` (genes raised or lowered in receivers by
   exp(effect * exp(-d/lam)), d = distance to the nearest sender), `shift` (receivers near senders move toward
   a partner type's profile)
3. count perturbations of the arm: `contamination` (molecules move to the k nearest cells with weights
   exp(-d/lam); totals conserved; optional source/receiver type multipliers for asymmetry) or `thinning`
4. binomial count split with a fixed seed: half A for re-calling labels, half B analyzed
5. labels of the arm: `label_errors` (reference labels perturbed at rate eps; confusion `native`, `uniform` or
   `{lineage: ...}`; spatial `random`, `boundary` or `interior`; `context_bias` toward neighborhood-typical
   labels; `condition_mult`; `type_rates: native` distributes eps across types in proportion to each type's
   native flip rate, so rare types receive a realistic error flux; `confusion: native_by_condition` and
   `type_rates: native_by_condition` use the confusion structure and per-type rates estimated within each
   condition, so errors differ between conditions as the labeler's do), `recall` (nearest-centroid re-calling on half A thinned to a depth), or the
   reference labels
6. questions on half B with those labels; claims compared with the baseline and with planted truth

## Questions

| name | claims | decision |
|---|---|---|
| composition | label proportion differs between two conditions | Welch t over units, p < alpha (effect: log2 ratio of means) |
| colocalization | every ordered pair of different labels | abs(z) > 2 vs within-section label permutations **and** observed/expected >= `min_ratio` (1.5) or <= 1/`min_ratio`; `include_self` |
| niches | `co` (two labels enriched in the same niche; independent of niche identity), `enriched` (label enriched >= 2-fold in a matched niche), `abundance` (niche fraction differs between conditions) | k-means on neighbor composition with `n_seeds` (20) seeds; a claim needs `consensus` (0.7) of the seeds; claims made by 0.3–0.7 of the seeds are flagged `unstable`; stat = share of seeds |
| proximity_de_joint | as proximity_de, but all neighbor types in one regression (kernel-weighted counts over the k = 50 nearest cells, bandwidth `sigma`), as Niche-DE fits its effective niche; writes `diagnostics_<name>.csv` (variance inflation of each neighbor type, most correlated partner) | per-section OLS with all neighbor types; `aggregate: units` or `pooled`; BH |
| proximity_de | expression of a gene in the index type vs neighbors of another type | OLS per section; `aggregate: units` or `pooled`; BH q < alpha and abs(effect) >= `min_effect`, effect = slope x k |

Questions with random elements (colocalization permutations, niche k-means) vary their seed with the replicate
(`vary_seed: true`), so level-0 replicates measure each method's own variability. Every claim carries `effect` and
`support` (cells behind it). A question can be listed twice with different settings by giving it a `label`, e.g.
`{name: proximity_de, label: proximity_de_eff, min_effect: 0.3}`.

## Outputs

- `baseline_claims.csv`: every claim with stat, p, decision, effect, support
- `native.json`: `disagreement` (re-called vs reference labels, half counts; includes systematic labeler
  difference), `flip_rate` (labels changing under Poisson resampling: noise only) and `flip_rate_by_type`, and all of these
  per condition under `by_condition`, `confusion` as
  `{true: {called: p}}` (off-diagonal, row-normalized)
- `summary.csv`: per spec x level x replicate x question: calls in the run (`n_run`), kept, lost, flipped,
  spurious, `survival` (kept / baseline), `jaccard` (kept / union), `precision` (kept / calls in run),
  `sign_flip_rate`, realised error; FDR and power against planted truth; neighbor-marker shares for proximity_de
- `decisions_<spec>.csv.gz`: claims non-zero in the run or baseline, with stat, effect, support
- `breakpoints_by_support.csv`: break points by support quartile (share broken ever and within the flip rate)
- `breakpoints.csv`: per baseline claim, the first level (in order of increasing severity: rising error, falling
  depth) at which it survives in fewer than half the replicates; `fragility` = break level / `flip_rate` for
  label-error arms; baseline support and effect

Survival alone hides the main effect of label error, which is new claims: read it together with Jaccard and
precision.

## Caveats

- The reference labels are not error-free; break levels are measured above the native error. Compare them with
  the native disagreement (fragility index) and, where depth allows, build the reference on a deeper truth share.
- Robustness is not correctness. Use planted signals (`spike`, `null`) to measure false discoveries and power.
- Include level 0 in label-error arms: it measures run-to-run variation of methods with random elements.
- Raising genes in `spike` raises library sizes; normalized analyses then see small decreases in other genes.

## Correctors

A `proximity_de` question can list `correctors`; each produces its own claims (question name
`proximity_de+<label>`), scored against the planting like the uncorrected claims (BH and fixed threshold).

    questions:
      - name: proximity_de
        correctors:
          - {name: twode}                         # 2DE filter (Levy et al. 2025), tau 0.9
          - {name: simex, pi: oracle}             # MC-SIMEX, matrix from reference vs observed labels (upper bound)
          - {name: simex, pi: native}             # MC-SIMEX, matrix from the labeler's resampling flips
          - {name: ppi, fraction: 0.05, patch: 200}   # difference estimator from revealed patches

See `src/claimbreak/correct.py` for the exact definitions and assumptions. Spike-ins accept `marker_rule: sender`
(genes that are markers of the sender: truly induced genes a neighbor-marker filter would remove).

## Examples

`examples/` holds the scripts behind the analyses of the accompanying paper (`compartment/`, `correction/`, `ibd/`,
`liver/`, `ra/`, `ain/`, `analysis/`) and the Xenium conversion helper (`xenium/`); each folder's README lists its
scripts. Paths in them are examples: set `D` (data folder), `R` (results folder) and `CB` (this repository) as in
each script's header. The batch drivers (`*.zsh`, `*.sh`) were written for the authors' machines (zsh on macOS,
bash on a Linux or WSL workstation) and need their data folders; the Python scripts run anywhere the package does.

## Version history

`CHANGELOG.md`.
