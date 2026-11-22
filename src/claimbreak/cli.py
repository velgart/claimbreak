"""Command line.

claimbreak inspect FILE                         list obs columns, layers and obsm keys (reads metadata only)
claimbreak inspect FILE --label-key L --section-key S [--unit-key U --condition-key C --counts-layer X]
claimbreak check CONFIG                          load the data and apply the planting only (seconds): spike gene
                                                candidates by marker rule, genes chosen
claimbreak run CONFIG [--out DIR]               exit code 2 if an integrity check voids the run (void.json)
claimbreak export CONFIG --out DIR [--folds 5]  half A (panel genes) for an external labeler
claimbreak summarize OUT
claimbreak explain OUT --question Q [--spec S] [--level L] [--top N]   which claims were gained, lost, flipped
claimbreak stress FILE --label-key L --section-key S --out DIR [--question colocalization] [--eps 0.05,0.1]
                [--nucleus-layer N --cell-area-key A --nucleus-area-key B]   claims under random vs realistic errors
claimbreak compartment label --ref REF --inputs H5AD... --out DIR [--arms ...] [--ref-cv]   labels per count set
claimbreak compartment audit --labels DIR --out DIR [--min-sections 0.5]                       label audit, pooled
claimbreak compartment audit --labels DIR --out DIR --recompute H5AD_DIR                        audit of older labels
claimbreak compartment compare --labels DIR --out DIR [--levels L] [--comparisons T|R,...]    where changes land
claimbreak compartment consequence --labels DIR --out DIR [--level L --ref A --test B]       colocalization
claimbreak compartment summary --compare DIR [--consequence DIR] --out DIR                   by section and unit
claimbreak compartment placement --labels DIR --out DIR --level L --immune A,B --partners C,D  same changes, placed
claimbreak compartment calls --labels DIR --out DIR --level L --immune A,B --partners C,D      calls and stress test
claimbreak compartment contacts --labels DIR --out DIR --level L --immune A,B --partners C     infiltration, mixing
claimbreak compartment aggregates --labels DIR --out DIR --level L --immune A,B               immune aggregates
claimbreak compartment spill --h5ad-dir DIR --labels DIR --out DIR --gene G [--source-types ...]  gene spill-over
claimbreak compartment exposure --labels DIR --out DIR [--h5ad-dir DIR] [--levels L] [--gaps 2,4,6]   label kept vs
                physical exposure to other types (nucleus gaps, border share, RNA-weighted exposure); destinations
claimbreak compartment exposure-summary --exposure DIR                                     summary tables only
claimbreak compartment kernel --h5ad-dir DIR --labels DIR --out DIR [--level L --unit-pattern P]  error kernel per
                dataset: spillover per contact, nuclear share of borrowed RNA, R matrix
claimbreak compartment kernel-summary --out DIR [--spill DIR] [--exposure DIR] [--dataset N]  pooled kernel tables
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="claimbreak")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment from a YAML config")
    r.add_argument("config"); r.add_argument("--out"); r.add_argument("--quiet", action="store_true")
    c = sub.add_parser("check", help="load the data and apply the planting only (catches config errors fast)")
    c.add_argument("config")
    i = sub.add_parser("inspect", help="summarize a dataset before configuring an experiment")
    i.add_argument("path"); i.add_argument("--label-key"); i.add_argument("--section-key")
    i.add_argument("--unit-key"); i.add_argument("--condition-key"); i.add_argument("--x", default="x")
    i.add_argument("--y", default="y"); i.add_argument("--obsm"); i.add_argument("--counts-layer")
    x = sub.add_parser("export", help="write half A for an external labeller (panel arm)")
    x.add_argument("config"); x.add_argument("--out", required=True); x.add_argument("--folds", type=int, default=5)
    s = sub.add_parser("summarize", help="print survival by spec, level and question")
    s.add_argument("out")
    e = sub.add_parser("explain", help="list the claims gained, lost or flipped in a finished run")
    e.add_argument("out"); e.add_argument("--question", required=True); e.add_argument("--spec")
    e.add_argument("--level", type=float); e.add_argument("--top", type=int, default=30)
    e.add_argument("--csv", help="also write the full table to this file")
    st = sub.add_parser("stress", help="stress-test the claims of one analysis: random vs realistic label errors")
    st.add_argument("path"); st.add_argument("--label-key", required=True); st.add_argument("--section-key", required=True)
    st.add_argument("--unit-key"); st.add_argument("--condition-key"); st.add_argument("--x", default="x")
    st.add_argument("--y", default="y"); st.add_argument("--obsm"); st.add_argument("--counts-layer")
    st.add_argument("--question", default="colocalization")
    st.add_argument("--eps", default="0.05,0.1", help="error rates, comma-separated")
    st.add_argument("--reps", type=int, default=5)
    st.add_argument("--confusion", default="native", choices=["native", "uniform"])
    st.add_argument("--model", help="flip model JSON (default: the packaged model)")
    st.add_argument("--nucleus-layer", help="layer with the counts over the nucleus (Xenium overlaps_nucleus)")
    st.add_argument("--cell-area-key"); st.add_argument("--nucleus-area-key")
    st.add_argument("--r-contact", type=float, help="contact distance cap, units of the coordinates")
    st.add_argument("--min-cells-per-label", type=int, default=50)
    st.add_argument("--seed", type=int, default=0); st.add_argument("--out", required=True)
    cp = sub.add_parser("compartment", help="label cells from count subsets (nucleus, whole cell, ...); where do "
                                            "label changes land; what they do to colocalization")
    csub = cp.add_subparsers(dest="ccmd", required=True)
    cl = csub.add_parser("label", help="label compartment .h5ad files against a single-cell reference")
    cl.add_argument("--ref", required=True)
    cl.add_argument("--inputs", nargs="+", required=True, help="compartment .h5ad files")
    cl.add_argument("--out", required=True)
    cl.add_argument("--levels", default="", help="levels to label (default: all of --ref-levels)")
    cl.add_argument("--arms", default="whole,nucleus,thinned,cyto,nucA,nucB,cytoB")
    cl.add_argument("--floor", type=float, default=0.01)
    cl.add_argument("--prior", default="uniform", choices=["uniform", "fit"],
                    help="class prior: uniform (default) or fitted per section and arm by EM")
    cl.add_argument("--min-counts", type=int, default=10)
    cl.add_argument("--gamma-iter", type=int, default=5)
    cl.add_argument("--gamma-clip", default="0.05,20", help="lower,upper bound of the platform factor")
    cl.add_argument("--gamma-damp", type=float, default=1.0, help="gamma *= (observed/expected) ** damp")
    cl.add_argument("--gamma-fit", default="hard", choices=["hard", "soft"],
                    help="expected pseudobulk from hard labels (default) or from posteriors")
    cl.add_argument("--markers", type=int, default=0,
                    help="label on the N most specific genes per type only (0 = all panel genes)")
    cl.add_argument("--marker-min", type=float, default=1e-4, help="minimum mean proportion of a marker in its type")
    cl.add_argument("--marker-min-log2", type=float, default=1.0,
                    help="minimum log2 ratio of a marker's mean in its type over the next type's")
    cl.add_argument("--exclude-genes", default="", help="file of genes never used (one per line, or a CSV with a "
                                                        "'gene' column)")
    cl.add_argument("--labeller", default="multinomial", choices=["multinomial", "celltypist"],
                    help="classifier for every arm: the multinomial classifier (default) or a CellTypist "
                         "logistic-regression model trained on the same reference (pip install celltypist)")
    cl.add_argument("--ct-train-depths", default="20,50,100,200",
                    help="celltypist: thin training cells to these depths ('full' = no thinning)")
    cl.add_argument("--ct-per-type", type=int, default=500, help="celltypist: training cells per type")
    cl.add_argument("--ct-max-iter", type=int, default=500, help="celltypist: logistic-regression iterations")
    cl.add_argument("--ct-jobs", type=int, default=1, help="celltypist: parallel jobs in training (-1 = all)")
    cl.add_argument("--decon-gap", type=float, default=10.0,
                    help="neighbor-aware arms (wholeM, cytoM, wholeMX): nucleus gap counted as mask contact, um")
    cl.add_argument("--decon-region", type=float, default=20.0, help="neighbor-aware arms: regional radius, um")
    cl.add_argument("--decon-w", default="1,0.5,0.1",
                    help="neighbor-aware arms: weights of the contact, regional and ambient contamination terms")
    cl.add_argument("--decon-psi-max", type=float, default=0.9,
                    help="neighbor-aware arms: largest foreign share of the counts outside the nucleus")
    cl.add_argument("--decon-fit-on", default="max",
                    help="neighbor-aware arms: labels the catchment constant is fitted to: max (best candidate per "
                         "cell, default), auto (cytoM: nucB, otherwise nucleus) or an arm name")
    cl.add_argument("--decon-betas", default="0,0.5,1,2",
                    help="neighbor-aware arms: exponents of the outside-share term tried in the fit (0 = none)")
    cl.add_argument("--decon-psi-mode", default="fit", choices=["fit", "profile"],
                    help="neighbor-aware arms: foreign share from exposure with one fitted constant per section "
                         "(default), or free per cell and candidate type")
    cl.add_argument("--seed", type=int, default=0)
    cl.add_argument("--no-audit", action="store_true", help="skip the label audit (labels are the same)")
    cl.add_argument("--audit-arms", default="nucleus,whole", help="arms audited when labeled")
    cl.add_argument("--ref-cv", action="store_true")
    cl.add_argument("--cv-depths", default="20,50,100,200")
    cl.add_argument("--cv-folds", type=int, default=5)
    cl.add_argument("--cv-group", default="")
    cl.add_argument("--cv-per-type", type=int, default=500)
    cl.add_argument("--ref-layer", default="")
    cl.add_argument("--ref-levels", default="L2m=Level2,L3m=Level3")
    cl.add_argument("--ref-merge", default=r"^Tu_L\d=>Tumour")
    cl.add_argument("--ref-map", default="")
    cl.add_argument("--ref-donor", default="donor_id")
    cl.add_argument("--ref-raw", action="store_true")
    cl.add_argument("--ref-subset", default="")
    au = csub.add_parser("audit", help="pool the label audits of a labels folder: over-detected genes (exclusion "
                                       "list for --exclude-genes), driver concentration, marker support")
    au.add_argument("--labels", required=True, help="folder written by 'compartment label'")
    au.add_argument("--out", required=True)
    au.add_argument("--arms", default="nucleus,whole")
    au.add_argument("--min-sections", type=float, default=0.5,
                    help="over-detected in at least this share of sections (or this many, if >= 1) to be listed")
    au.add_argument("--recompute", default="", metavar="H5AD_DIR",
                    help="audit labels written without the audit from their stored labels, runner-ups and gamma "
                         "with the counts of H5AD_DIR (no relabeling); --out must be new or empty")
    au.add_argument("--reference", default="", help="with --recompute: reference .h5ad (default: meta.json)")
    au.add_argument("--ref-levels", default="", help="with --recompute: e.g. L1=coarse (default: meta.json)")
    au.add_argument("--samples", default="", help="with --recompute: comma-separated subset of samples")
    au.add_argument("--all-warnings", action="store_true",
                    help="with --recompute: print every over-detected gene (default: only the count per arm)")
    cc = csub.add_parser("compare", help="do label changes between arms land on contact neighbors' types")
    cc.add_argument("--labels", required=True); cc.add_argument("--out", required=True)
    cc.add_argument("--levels", default="L2m,L3m")
    cc.add_argument("--comparisons", default="nucB|nucA,cytoB|nucA,wholeB|nucA,whole|nucleus,thinned|nucleus,"
                                             "cyto|nucleus,whole|thinned")
    cc.add_argument("--null", default="nucB|nucA"); cc.add_argument("--nbr-arm", default="nucleus")
    cc.add_argument("--k-contact", type=int, default=6); cc.add_argument("--k-far", type=int, default=20)
    cc.add_argument("--radius", type=float, default=30.0); cc.add_argument("--radius-far", type=float, default=60.0)
    cc.add_argument("--min-margin", type=float, default=0.0)
    cc.add_argument("--common-cells", default=""); cc.add_argument("--samples", default="")
    cc.add_argument("--tile", type=float, default=200.0, help="tile side (um) for the block bootstrap of summary")
    cc.add_argument("--agree", default="", help="only cells whose reference label equals this arm's label (e.g. nucC3)")
    cc.add_argument("--contact", default="knn", choices=["knn", "boundary"],
                    help="contact: 6 nearest within 30 um (default) or touching segmentation boundaries")
    cc.add_argument("--boundaries", default="", help="folder of <sample>_cell_boundaries.parquet(.gz) for --contact boundary")
    cc.add_argument("--tol", type=float, default=1.0, help="boundary contact: largest gap between boundaries, um")
    cc.add_argument("--region", default="30,100",
                    help="inner,outer radius (um) of the ring whose composition the matched expectation "
                         "(exp_contact_reg) is stratified on; '' switches it off")
    cc.add_argument("--region-bins", type=int, default=10, help="bins of a type's regional share")
    cc.add_argument("--region-min", type=int, default=20,
                    help="cells a stratum needs before it is used; thinner strata fall back to coarser ones")
    cc.add_argument("--no-match-degree", action="store_true",
                    help="match the expectation on regional share only, not also on the number of contacts")
    cq = csub.add_parser("consequence", help="kNN colocalization from two arms vs random relabeling")
    cq.add_argument("--labels", required=True); cq.add_argument("--out", required=True)
    cq.add_argument("--level", default="L2m"); cq.add_argument("--ref", default="nucleus")
    cq.add_argument("--test", default="whole"); cq.add_argument("--k", type=int, default=10)
    cq.add_argument("--draws", type=int, default=20); cq.add_argument("--min-cells", type=int, default=200)
    cq.add_argument("--seed", type=int, default=0)
    cs = csub.add_parser("summary", help="extra-change ratios, pair excess and colocalization change, by section and unit")
    cs.add_argument("--compare", required=True, help="folder written by 'compartment compare'")
    cs.add_argument("--consequence", default="", help="folder written by 'compartment consequence'")
    cs.add_argument("--out", required=True)
    cs.add_argument("--test", default="wholeB|nucA"); cs.add_argument("--null", default="nucB|nucA")
    cs.add_argument("--level", default="")
    cs.add_argument("--min-null-cells", type=int, default=1000, help="sections with fewer cells in the null are not used")
    cs.add_argument("--min-stratum", type=int, default=50, help="cells per contact and absent stratum for a pair")
    cs.add_argument("--unit-pattern", default="",
                    help="regex giving the unit (donor, patient) from the sample name; default '': every section is "
                         "its own unit (before, the gut default I0\\d{3})")
    cs.add_argument("--top", type=int, default=10)
    cs.add_argument("--min-extra", type=int, default=50,
                    help="extra changes (test minus null) a section or unit needs for its extra-change ratio")
    cs.add_argument("--draws", type=int, default=2000, help="bootstrap draws for the pooled unit ratio (0: none)")
    cs.add_argument("--block-draws", type=int, default=1000,
                    help="block bootstrap over the tiles of compare (0: none): units.csv block_* columns, "
                         "units_kept.csv")
    cs.add_argument("--focus", default=r"(?i)(?:^|[^a-z])(?:t|nk|tnk)(?:[^a-z]|$)",
                    help="types for units_kept.csv (regex)")
    sl = csub.add_parser("spill", help="gene positivity, neighbor gradients and ligand-receptor proximity from "
                                       "whole, nucleus and depth-matched counts of the same cells")
    sl.add_argument("--h5ad-dir", required=True); sl.add_argument("--labels", required=True)
    sl.add_argument("--out", required=True); sl.add_argument("--gene", required=True)
    sl.add_argument("--level", default="L1"); sl.add_argument("--label-arm", default="nucleus")
    sl.add_argument("--source-types", default=""); sl.add_argument("--source-gene", default="")
    sl.add_argument("--source-arm", default="whole"); sl.add_argument("--recipients", default="")
    sl.add_argument("--radius", type=float, default=50.0); sl.add_argument("--last-bin", type=int, default=4)
    sl.add_argument("--k", type=int, default=6); sl.add_argument("--r", type=float, default=30.0)
    sl.add_argument("--arms", default="whole,nucleus,nucB,wholeB")
    sl.add_argument("--ligand", default=""); sl.add_argument("--receptor", default="")
    sl.add_argument("--lr-radius", type=float, default=20.0)
    sl.add_argument("--unit-pattern", default=r"I0\d{3}"); sl.add_argument("--group-pattern", default="")
    sl.add_argument("--samples", default=""); sl.add_argument("--seed", type=int, default=20261017)
    kn = csub.add_parser("kernel", help="expression error kernel of a dataset: spillover per contact by gene, producer "
                                        "and receiver; nuclear share of borrowed RNA; R matrix")
    kn.add_argument("--h5ad-dir", required=True); kn.add_argument("--labels", required=True)
    kn.add_argument("--out", required=True)
    kn.add_argument("--level", default="L1"); kn.add_argument("--label-arm", default="nucleus")
    kn.add_argument("--unit-pattern", default=r"I0\d{3}")
    kn.add_argument("--k", type=int, default=6); kn.add_argument("--r", type=float, default=30.0)
    kn.add_argument("--samples", default="")
    kn.add_argument("--n-markers", type=int, default=10, help="own markers per type (isolated-cell specificity)")
    kn.add_argument("--eps", type=float, default=0.02, help="calibration pair: receiver's own level <= eps x producer's")
    kn.add_argument("--min-cells", type=int, default=30); kn.add_argument("--min-counts", type=float, default=50)
    kn.add_argument("--min-prod", type=float, default=0.3, help="producer's mean whole counts of the gene, isolated cells")
    ks = csub.add_parser("kernel-summary", help="pool the expression kernel (decoy rows) and the label kernel "
                                                "(from an exposure folder) of one dataset")
    ks.add_argument("--out", required=True); ks.add_argument("--spill", default="")
    ks.add_argument("--exposure", default=""); ks.add_argument("--dataset", default="")
    ks.add_argument("--min-contact-cells", type=int, default=50)
    ks.add_argument("--boot", type=int, default=1000, help="draws for the nu_b and R_b intervals, genes resampled "
                                                           "(0 = none)")
    pl = csub.add_parser("placement", help="the test's excess label changes placed at random, by ambiguity, by "
                                           "contact kernels or by flip models (fixed or held out by unit)")
    pl.add_argument("--labels", required=True); pl.add_argument("--out", required=True)
    pl.add_argument("--level", default="L1")
    pl.add_argument("--immune", required=True, help="comma list of immune types")
    pl.add_argument("--partners", required=True, help="comma list of partner types")
    pl.add_argument("--h5ad-dir", default="", help="compartment h5ad files (cell and nucleus areas)")
    pl.add_argument("--unit-pattern", default=r"I0\d{3}",
                    help="regex giving the unit from the sample name (whole match, or the named group (?P<unit>...))")
    pl.add_argument("--use-sections", default="", help="'compartment summary' sections.csv (column used) or a "
                                                       "file of sample names")
    pl.add_argument("--transfer", default="default,no_nucleus", help="packaged flip models applied as fixed "
                                                                     "('' for none)")
    pl.add_argument("--model", action="append", default=[], metavar="NAME=JSON",
                    help="another fixed flip model (repeatable), arm transfer_NAME")
    pl.add_argument("--louo", default="full,no_nucleus", help="flip models fitted leaving one unit out ('' for none)")
    pl.add_argument("--no-kernels", action="store_true"); pl.add_argument("--no-oracle", action="store_true")
    pl.add_argument("--save-models", action="store_true", help="also write flip models fitted on all units")
    pl.add_argument("--draws", type=int, default=20); pl.add_argument("--min-cells", type=int, default=200)
    pl.add_argument("--informative", type=float, default=0.02)
    pl.add_argument("--neg-ratio", type=int, default=10); pl.add_argument("--seed", type=int, default=20261013)
    pl.add_argument("--max-pos", type=int, default=2000, help="real changes per section and pair used to fit")
    pl.add_argument("--nbr-arm", default="nucleus", help="arm whose labels type the neighbors (nucleus; whole or "
                                                          "wholeB for data without a nucleus split)")
    ck = csub.add_parser("calls", help="enrichment calls from reference and analyst labels, and the field's random "
                                       "stress test on the analyst's labels")
    ck.add_argument("--labels", required=True); ck.add_argument("--out", required=True)
    ck.add_argument("--level", default="L1")
    ck.add_argument("--ref", default="nucleus", help="reference arm (nucleus; nucB for the depth-matched pair)")
    ck.add_argument("--test", default="whole", help="analyst's arm (whole; wholeB for the depth-matched pair)")
    ck.add_argument("--immune", default=""); ck.add_argument("--partners", default="")
    ck.add_argument("--k", type=int, default=10)
    ck.add_argument("--nperm", type=int, default=100); ck.add_argument("--nperm-stress", type=int, default=50)
    ck.add_argument("--z", type=float, default=2.0); ck.add_argument("--min-effect", type=float, default=0.2)
    ck.add_argument("--eps", default="0.05,0.10,0.15"); ck.add_argument("--reps", type=int, default=10)
    ck.add_argument("--mode", default="uniform", choices=["uniform", "composition"])
    ck.add_argument("--robust", type=float, default=0.9)
    ck.add_argument("--min-cells", type=int, default=200)
    ck.add_argument("--unit-pattern", default=r"I0\d{3}")
    ck.add_argument("--use-sections", default="")
    ck.add_argument("--seed", type=int, default=20261014)
    cn = csub.add_parser("contacts", help="contact-scale infiltration, per-100 counts and the Keren mixing score from "
                                          "reference and analyst labels, with the field's random stress test")
    cn.add_argument("--labels", required=True); cn.add_argument("--out", required=True)
    cn.add_argument("--level", default="L1")
    cn.add_argument("--ref", default="nucleus"); cn.add_argument("--test", default="whole")
    cn.add_argument("--immune", required=True, help="immune types for infiltration (comma list)")
    cn.add_argument("--partners", required=True, help="partner types (comma list); the first is the mixing partner "
                                                      "unless --mixing-partner is given")
    cn.add_argument("--mixing-immune", default="", help="types counted as immune in the mixing score (default --immune)")
    cn.add_argument("--mixing-partner", default="", help="tumor (or epithelial) types of the mixing score")
    cn.add_argument("--contact", default="knn", choices=["knn", "boundary"])
    cn.add_argument("--k", type=int, default=6); cn.add_argument("--r", type=float, default=30.0)
    cn.add_argument("--boundaries", default="", help="folder of <sample>_cell_boundaries.parquet(.gz) for --contact boundary")
    cn.add_argument("--tol", type=float, default=1.0, help="boundary vertex distance counted as touching (um)")
    cn.add_argument("--confusion", action="store_true",
                    help="also relabel the reference labels at random with the section's own confusion P(test | ref) "
                         "(labels 'confusion' in units.csv): the real number of changes of each kind, placed at random")
    cn.add_argument("--fix-partners", action="store_true",
                    help="also the test (and confusion) labels with the partner cells fixed at the reference "
                         "labels (labels 'test_fixed', 'confusion_fixed'): only the immune side of the changes varies")
    cn.add_argument("--eps", default="0.05,0.10,0.15"); cn.add_argument("--reps", type=int, default=10)
    cn.add_argument("--mode", default="uniform", choices=["uniform", "composition"])
    cn.add_argument("--robust", type=float, default=0.9)
    cn.add_argument("--unit-pattern", default=r"I0\d{3}"); cn.add_argument("--use-sections", default="")
    cn.add_argument("--seed", type=int, default=20261015)
    ag = csub.add_parser("aggregates", help="immune aggregates (connected immune components on the contact graph) "
                                            "from reference and analyst labels, with random relabeling")
    ag.add_argument("--labels", required=True); ag.add_argument("--out", required=True)
    ag.add_argument("--level", default="L1"); ag.add_argument("--immune", required=True)
    ag.add_argument("--ref", default="nucleus"); ag.add_argument("--test", default="whole", help="comma list of arms")
    ag.add_argument("--k", type=int, default=6); ag.add_argument("--r", type=float, default=30.0)
    ag.add_argument("--min-size", type=int, default=50)
    ag.add_argument("--eps", default="0.05,0.10,0.15"); ag.add_argument("--reps", type=int, default=10)
    ag.add_argument("--mode", default="uniform", choices=["uniform", "composition"])
    ag.add_argument("--unit-pattern", default=r"I0\d{3}"); ag.add_argument("--use-sections", default="")
    ag.add_argument("--seed", type=int, default=20261016)
    ex = csub.add_parser("exposure", help="labels kept from whole-cell counts against physical exposure to other "
                                          "types (nucleus gaps, border share, RNA-weighted exposure) and where "
                                          "changed labels go")
    ex.add_argument("--labels", required=True); ex.add_argument("--out", required=True)
    ex.add_argument("--h5ad-dir", default="", help="compartment .h5ad files (<sample>.h5ad; obs nucleus_area)")
    ex.add_argument("--levels", default="L2m")
    ex.add_argument("--focal", default="nucA", help="arm giving each cell's own type")
    ex.add_argument("--test", default="wholeB"); ex.add_argument("--null", default="nucB")
    ex.add_argument("--nbr-arm", default="nucleus", help="arm typing the neighbors")
    ex.add_argument("--gaps", default="2,4,6", help="nucleus gaps (um) for the neighbor counts")
    ex.add_argument("--contact-gap", type=float, default=2.0, help="gap (um) for border share and exposure")
    ex.add_argument("--mcap", type=int, default=8, help="neighbor counts above this are pooled")
    ex.add_argument("--min-margin", type=float, default=0.0, help="focal-arm margin a cell needs")
    ex.add_argument("--unit-pattern", default=r"I0\d{3}", help="unit from the sample name where the labels have "
                                                               "no unit column")
    ex.add_argument("--samples", default="")
    ex.add_argument("--focus", default=r"(?i)^t.?nk$|^t$|t_?cell", help="types printed in the summary (regex)")
    ex.add_argument("--min-kept", type=int, default=20, help="null-kept cells a unit needs in a stratum")
    ex.add_argument("--regions", default="10,20,30", help="radii (um) for regional destinations ('' = none)")
    ex.add_argument("--boundaries", default="", help="folder of segmentation polygons (<sample>_cell_boundaries"
                                                    ".parquet): border share from shared perimeter")
    ex.add_argument("--tol", type=float, default=1.0, help="with --boundaries: points of two polygons within tol um "
                                                          "touch")
    es = csub.add_parser("exposure-summary", help="(re)write the summary tables of an existing 'compartment exposure' "
                                                  "folder")
    es.add_argument("--exposure", required=True, help="folder written by 'compartment exposure'")
    es.add_argument("--focus", default=r"(?i)^t.?nk$|^t$|t_?cell")
    es.add_argument("--min-kept", type=int, default=20)
    kr = sub.add_parser("kept-ratio", help="one command: label-loss check of a Xenium or CosMx dataset against a "
                                           "single-cell reference (labels from nucleus and whole cell, kept ratio, "
                                           "extra-change ratio, label audit)",
                        description="From Xenium output folders (or compartment .h5ad files) and a single-cell "
                                    "reference to OUT/report.md, kept_ratio.csv and sections.csv. See "
                                    "docs/kept_ratio.md.")
    src_ = kr.add_mutually_exclusive_group(required=True)
    src_.add_argument("--xenium", nargs="+", default=[], metavar="DIR",
                      help="Xenium output folders (transcripts.parquet, cells.parquet), or a folder holding several")
    src_.add_argument("--h5ad", nargs="+", default=[], metavar="FILE",
                      help="compartment .h5ad files (examples/compartment/compartment_to_h5ad.py; needed for CosMx)")
    kr.add_argument("--xenium-prefix", default="", help="GEO-style file prefix (<prefix>_transcripts.parquet), "
                                                        "one section per call")
    kr.add_argument("--ref", required=True, help="single-cell reference .h5ad (raw counts in X)")
    kr.add_argument("--ref-level", required=True, help="reference obs column with the cell types to label")
    kr.add_argument("--ref-merge", default="", help="merge reference types: 'regex=>label;...'")
    kr.add_argument("--ref-donor", default="donor_id")
    kr.add_argument("--exclude-genes", default="", help="genes never used (file; e.g. OUT/audit/exclude_overdetected.txt "
                                                        "of a first run)")
    kr.add_argument("--unit-pattern", default=r"(?P<unit>.+)",
                    help="regex giving the unit (donor, patient) from the section name (default: each section)")
    kr.add_argument("--min-qv", type=float, default=20.0)
    kr.add_argument("--tile", type=float, default=200.0, help="tile side (um) for the block bootstrap")
    kr.add_argument("--block-draws", type=int, default=1000)
    kr.add_argument("--min-null-cells", type=int, default=1000, help="sections with fewer labeled cells are not used")
    kr.add_argument("--seed", type=int, default=0)
    kr.add_argument("--out", required=True, help="new or empty output folder")
    kr.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "compartment":
        return _compartment(a)
    if a.cmd == "kept-ratio":
        from .keptratio import run as kept_run
        try:
            kept_run(a.out, a.ref, a.ref_level, xenium=a.xenium, h5ad=a.h5ad, xenium_prefix=a.xenium_prefix,
                     ref_merge=a.ref_merge, ref_donor=a.ref_donor, exclude_genes=a.exclude_genes,
                     unit_pattern=a.unit_pattern, min_qv=a.min_qv, tile=a.tile, block_draws=a.block_draws,
                     min_null_cells=a.min_null_cells, seed=a.seed, verbose=not a.quiet)
        except (FileExistsError, FileNotFoundError, ValueError) as e:
            print(f"claimbreak kept-ratio: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "run":
        from .experiment import run, ConfigError
        from .integrity import IntegrityError
        try:
            run(a.config, out=a.out, verbose=not a.quiet)
        except IntegrityError as e:
            print(f"claimbreak: RUN VOID (integrity check failed): {e}", file=sys.stderr)
            return 2
        except (ConfigError, FileNotFoundError, ValueError) as e:
            print(f"claimbreak: {e}", file=sys.stderr)
            return 1
    elif a.cmd == "export":
        from .experiment import export, ConfigError
        try:
            m = export(a.config, a.out, folds=a.folds)
            print(f"exported {m['n_cells']} cells x {m['n_genes']} genes, {m['n_folds']} folds, to {a.out}")
        except (ConfigError, FileNotFoundError, ValueError) as e:
            print(f"claimbreak: {e}", file=sys.stderr)
            return 1
    elif a.cmd == "check":
        from .experiment import check, ConfigError
        try:
            check(a.config)
        except (ConfigError, FileNotFoundError, ValueError) as e:
            print(f"claimbreak: {e}", file=sys.stderr)
            return 1
    elif a.cmd == "inspect" and not (a.label_key and a.section_key):
        from .tissue import describe_file
        print(describe_file(a.path))
    elif a.cmd == "inspect":
        from .tissue import load
        t = load(a.path, a.label_key, a.section_key, a.unit_key, a.condition_key,
                 spatial=a.obsm or (a.x, a.y), counts_layer=a.counts_layer)
        print(t.summary())
        print(pd.crosstab(pd.Series(t.unit, name="unit"), pd.Series(t.condition, name="condition")) if t.condition is not None else
              pd.Series(t.unit).value_counts().to_string())
    elif a.cmd == "stress":
        from .stress import load_extras, stress_test, write
        from .tissue import load
        try:
            t = load(a.path, a.label_key, a.section_key, a.unit_key, a.condition_key, spatial=a.obsm or (a.x, a.y),
                     counts_layer=a.counts_layer, min_cells_per_label=a.min_cells_per_label)
            ex = load_extras(a.path, t.obs_names, a.nucleus_layer, a.cell_area_key, a.nucleus_area_key)
            tab, info = stress_test(t, question=a.question, eps=[float(x) for x in a.eps.split(",")], reps=a.reps,
                                    model=a.model, confusion=a.confusion, seed=a.seed, r_contact=a.r_contact, **ex)
            write(a.out, tab, info)
        except (FileNotFoundError, ValueError, FileExistsError) as e:
            print(f"claimbreak: {e}", file=sys.stderr)
            return 1
        print(open(f"{a.out}/report.md").read())
    elif a.cmd == "summarize":
        import os
        f = os.path.join(a.out, "summary.csv")
        if not os.path.exists(f):
            print(f"no summary.csv in {a.out}: the run has not produced results yet", file=sys.stderr)
            return 1
        S = pd.read_csv(f)
        cols = [c for c in ["n_run", "survival", "jaccard", "precision", "sign_flip_rate", "spurious", "fdr", "power",
                            "realised_eps", "marker_share_up", "marker_share_down", "power_1", "power_2",
                            "misattrib_up", "misattrib_down", "p_thr_run", "power_fixed", "power_1_fixed",
                            "power_2_fixed", "power_3_fixed", "fp_fixed", "n_filtered", "auprc", "pauroc_01",
                            "power_p3", "fp_p3", "null_typeI_05", "planted_bias"] if c in S]
        print(S.groupby(["spec", "question", "level"])[cols].mean().round(3).to_string())
    elif a.cmd == "explain":
        import glob, os
        from .compare import explain
        files = sorted(glob.glob(os.path.join(a.out, "decisions_*.csv.gz")))
        if not files:
            print(f"no decisions_*.csv.gz in {a.out}", file=sys.stderr)
            return 1
        D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        base = pd.read_csv(os.path.join(a.out, "baseline_claims.csv"))
        E = explain(D, base, a.question, a.spec, a.level)
        if a.csv:
            E.to_csv(a.csv, index=False)
        if E.empty:
            print(f"no claims of {a.question} changed" + (f" in {a.spec}" if a.spec else ""))
        cols = ["spec", "level", "claim", "gained", "lost", "flipped", "sign", "base_effect", "run_effect",
                "base_support"]
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            for (sp, lv), g in E.groupby(["spec", "level"]):
                print(f"== {sp} level {lv}: {int((g.gained > 0).sum())} gained, {int((g.lost > 0).sum())} lost, "
                      f"{int((g.flipped > 0).sum())} flipped")
                print(g[cols].head(a.top).round(3).to_string(index=False))
    return 0


def _compartment(a):
    from . import compartment as cm
    try:
        if a.ccmd == "label":
            levcol = dict(x.split("=", 1) for x in a.ref_levels.split(",") if x)
            levels = [s for s in a.levels.split(",") if s and s in levcol] if a.levels else list(levcol)
            if not levels:
                raise ValueError(f"--levels {a.levels} not among --ref-levels {list(levcol)}")
            lo, hi = (float(x) for x in a.gamma_clip.split(","))
            excl = ()
            if a.exclude_genes:            # one gene per line (first field of a CSV line), '#' comments, 'gene' header
                rows = [ln.split(",")[0].strip() for ln in open(a.exclude_genes).read().splitlines()]
                excl = tuple(g for g in rows if g and not g.startswith("#") and g.lower() != "gene")
            if a.prior != "uniform" and a.labeller != "multinomial":
                raise ValueError("--prior fit needs the multinomial labeller")
            cfg = cm.LabelConfig(floor=a.floor, prior=a.prior, min_counts=a.min_counts, gamma_iter=a.gamma_iter, gamma_clip=(lo, hi),
                                 gamma_damp=a.gamma_damp, gamma_fit=a.gamma_fit, markers=a.markers,
                                 marker_min=a.marker_min, marker_min_log2=a.marker_min_log2,
                                 exclude_genes=excl, labeller=a.labeller,
                                 ct_train_depths=() if a.ct_train_depths == "full" else
                                 tuple(int(x) for x in a.ct_train_depths.split(",") if x),
                                 ct_per_type=a.ct_per_type, ct_max_iter=a.ct_max_iter, ct_jobs=a.ct_jobs,
                                 decon_gap=a.decon_gap, decon_region=a.decon_region,
                                 decon_w=tuple(float(x) for x in a.decon_w.split(",")),
                                 decon_psi_max=a.decon_psi_max, decon_psi_mode=a.decon_psi_mode,
                                 decon_fit_on=a.decon_fit_on,
                                 decon_betas=tuple(float(x) for x in a.decon_betas.split(",")),
                                 audit=not a.no_audit, audit_arms=tuple(x for x in a.audit_arms.split(",") if x))
            if a.labeller == "celltypist":
                try:
                    import celltypist  # noqa: F401
                except ImportError:
                    raise ValueError("--labeller celltypist needs CellTypist: pip install celltypist")
            ref = cm.Reference.load(a.ref, {lv: levcol[lv] for lv in levels}, merge=a.ref_merge, map_file=a.ref_map,
                                    donor=a.ref_donor, raw=a.ref_raw, layer=a.ref_layer, subset=a.ref_subset,
                                    cv_group=a.cv_group)
            cm.label_files(ref, a.inputs, a.out, levels, [s for s in a.arms.split(",") if s], cfg, seed=a.seed,
                           ref_cv=a.ref_cv, cv_folds=a.cv_folds, cv_per_type=a.cv_per_type,
                           cv_depths=[int(x) for x in a.cv_depths.split(",")])
        elif a.ccmd == "compare":
            P, S, did, _ = cm.compare(a.labels, a.out, a.levels.split(","), a.comparisons, null=a.null,
                                      samples=a.samples, nbr_arm=a.nbr_arm, k_contact=a.k_contact, k_far=a.k_far,
                                      radius=a.radius, radius_far=a.radius_far, min_margin=a.min_margin,
                                      common_cells=[x for x in a.common_cells.split(",") if x], tile=a.tile,
                                      agree=a.agree, contact=a.contact, boundaries=a.boundaries or None, tol=a.tol,
                                      region=tuple(float(x) for x in a.region.split(",")) if a.region else None,
                                      region_bins=a.region_bins, region_min=a.region_min,
                                      match_degree=not a.no_match_degree)
            if did is not None and len(did):
                print("ratio_test / ratio_null (median over sections):")
                print(did.groupby(["level", "test"])[["change_test", "change_null", "ratio_test", "ratio_null",
                                                      "ratio_of_ratios"]].median().round(3).to_string())
            if len(S):
                print(S.groupby(["level", "comparison"])[["change_rate", "ratio_contact", "ratio_far_only"]].median()
                      .round(3).to_string())
            print(f"wrote {a.out}/pairs.csv, summary.csv")
        elif a.ccmd == "audit" and a.recompute:
            cm.audit_existing(a.labels, a.recompute, a.out, arms=tuple(x for x in a.arms.split(",") if x),
                              reference=a.reference, ref_levels=a.ref_levels, samples=a.samples,
                              min_sections=a.min_sections, verbose_overdetected=a.all_warnings)
            print(f"wrote {a.out}/<sample>.audit.csv, reproduce.csv and the pooled audit tables")
        elif a.ccmd == "audit":
            excl, _, _ = cm.audit_summary(a.labels, a.out, arms=tuple(x for x in a.arms.split(",") if x),
                                          min_sections=a.min_sections)
            print(f"wrote {a.out}/exclude_overdetected.txt ({len(excl)} genes), overdetected.csv, drivers.csv, "
                  f"support.csv")
        elif a.ccmd == "exposure-summary":
            from . import exposure as xp
            xp.exposure_summary(a.exposure, focus=a.focus, min_kept=a.min_kept)
            print(f"wrote the summary tables in {a.exposure}")
        elif a.ccmd == "exposure":
            from . import exposure as xp
            xp.exposure_files(a.labels, a.out, [x for x in a.levels.split(",") if x], h5ad_dir=a.h5ad_dir,
                              samples=a.samples, focal=a.focal, test=a.test, null=a.null, nbr=a.nbr_arm,
                              gaps=tuple(float(x) for x in a.gaps.split(",") if x), contact_gap=a.contact_gap,
                              mcap=a.mcap, unit_pattern=a.unit_pattern, min_margin=a.min_margin,
                              regions=tuple(float(x) for x in a.regions.split(",") if x),
                              boundaries=a.boundaries, tol=a.tol)
            xp.exposure_summary(a.out, focus=a.focus, min_kept=a.min_kept)
            print(f"wrote {a.out}/strata.csv, destinations.csv, dest_region.csv, units.csv, dose.csv, tissue.csv, "
                  f"dest_units.csv, dest_fit.csv, dest_region_units.csv, dest_region_fit.csv")
        elif a.ccmd == "consequence":
            R, S = cm.consequence(a.labels, a.out, a.level, a.ref, a.test, k=a.k, draws=a.draws,
                                  min_cells=a.min_cells, seed=a.seed)
            print(S.round(3).to_string(index=False))
            print(f"all pairs: median |d_test| {R.d_test.abs().median():.3f}, median |d_type| "
                  f"{R.d_type.abs().median():.3f}\nwrote {a.out}/coloc.csv, summary.csv")
        elif a.ccmd == "summary":
            r = cm.summarise(a.compare, a.out, a.consequence or None, test=a.test, null=a.null, level=a.level or None,
                             min_null_cells=a.min_null_cells, min_stratum=a.min_stratum, unit_pattern=a.unit_pattern,
                             min_extra=a.min_extra, draws=a.draws, block_draws=a.block_draws, focus=a.focus)
            sec = r["sections"]
            u = sec[sec.used]
            print(f"sections used: {len(u)} of {len(sec)} (null cells >= {a.min_null_cells}); units: "
                  f"{u.unit.nunique()}")
            if len(u) and (~u.test_changes_more).any():
                print(f"note: {a.test} changes fewer labels than {a.null} in {(~u.test_changes_more).sum()} of {len(u)} "
                      f"sections; the extra-change ratio is undefined there (NaN). Use a depth-matched test (wholeB|nucA).")
            if len(u) and a.test.split("|")[1] != a.null.split("|")[1]:
                print(f"note: {a.test} and {a.null} have different reference arms; pair excess below is the test's own "
                      f"contrast (contact minus absent), without the null subtracted")
            if len(u):
                print(f"extra-change ratio {a.test} over {a.null}, contact: median {u.extra_ratio_contact.median():.2f} "
                      f"(range {u.extra_ratio_contact.min():.2f}-{u.extra_ratio_contact.max():.2f}); far only: median "
                      f"{u.extra_ratio_far_only.median():.2f}; per unit (median of sections): "
                      f"{u.groupby('unit').extra_ratio_contact.median().round(2).to_dict()}")
                U = r["units"]
                if len(U):
                    ev = U[U.evaluable]
                    print(f"per unit, pooled over sections (report these): {len(ev)} of {len(U)} units evaluable "
                          f"(>= {a.min_extra} extra changes and a stable denominator); pooled ratio at contact median "
                          f"{ev.pooled_contact.median():.2f}, above one in {int((ev.pooled_contact > 1).sum())}, "
                          f"95% interval above one in {int(ev.ci_excludes_1.sum()) if 'ci_excludes_1' in ev else 'n/a'}")
                    if "block_ci_excludes_1" in ev:
                        print(f"block bootstrap over {a.block_draws} draws of tiles: 95% interval above one in "
                              f"{int(ev.block_ci_excludes_1.sum())} of {len(ev)} evaluable units")
                    if "pooled_contact_reg" in ev:
                        lo_ = "block_contact_reg_lo" if "block_contact_reg_lo" in ev else "pooled_contact_reg_lo"
                        print(f"matched on regional composition and contact count (~1 when changes follow the "
                              f"region rather than the touching cells): pooled median "
                              f"{ev.pooled_contact_reg.median():.2f}, 95% interval above one in "
                              f"{int((ev[lo_] > 1).sum()) if lo_ in ev else 'n/a'} of {len(ev)}")
                    cols = [c for c in ("unit", "sections", "extra_changes", "expected_extra_contact", "pooled_contact",
                                        "pooled_contact_lo", "pooled_contact_hi", "tiles", "block_contact_lo",
                                        "block_contact_hi", "pooled_contact_reg", "block_contact_reg_lo",
                                        "block_contact_reg_hi", "pooled_far_only", "evaluable", "ci_excludes_1",
                                        "block_ci_excludes_1") if c in U]
                    with pd.option_context("display.width", 200):
                        print(U[cols].round(2).to_string(index=False))
                    K_ = r.get("units_kept")
                    if K_ is not None and len(K_):
                        print("kept ratio of the focal type(s), test over null, block-bootstrap 95% interval:")
                        with pd.option_context("display.width", 200):
                            print(K_.round(3).to_string(index=False))
            with pd.option_context("display.width", 200):
                print(f"\npairs by median excess (top {a.top}), with the reverse pair:")
                print(r["pairs"].head(a.top).round(3).to_string(index=False))
                print(f"\nsame pairs by unit:")
                print(r["pairs_units"].head(a.top).round(3).to_string(index=False))
                if "coloc_pairs" in r:
                    cp = r["coloc_pairs"]
                    print("\ncolocalization change, real against random errors (sections and units):")
                    print(cp.sort_values("d_test").round(3).to_string(index=False))
            print(f"wrote {a.out}")
        elif a.ccmd == "spill":
            from .spill import spill, unit_view
            sp_ = lambda x: [v for v in x.split(",") if v]
            res = spill(a.h5ad_dir, a.labels, a.out, a.gene, a.level, a.label_arm, sp_(a.source_types),
                        a.source_gene or None, a.source_arm, sp_(a.recipients), a.radius, a.last_bin, a.k, a.r,
                        sp_(a.arms), a.ligand or None, a.receptor or None, a.lr_radius, a.unit_pattern,
                        a.group_pattern, a.samples, a.seed)
            V = unit_view(res)
            with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 400):
                for name, t in V.items():
                    print(f"\n{name} (median of sections per unit):")
                    print(t.round(4).to_string())
            print(f"wrote {a.out}")
        elif a.ccmd == "kernel":
            from .kernel import spill_kernel
            res = spill_kernel(a.h5ad_dir, a.labels, a.out, a.level, a.label_arm, a.unit_pattern, a.k, a.r, a.samples,
                               a.n_markers, a.eps, a.min_cells, a.min_counts, a.min_prod)
            print(f"wrote {a.out}: {len(res['spill_pairs']):,} gene x receiver rows, "
                  f"{int(res['spill_pairs'].decoy.sum()) if len(res['spill_pairs']) else 0:,} calibration rows")
        elif a.ccmd == "kernel-summary":
            from .kernel import kernel_summary, label_kernel
            if a.exposure:
                label_kernel(a.exposure, a.out)
            res = kernel_summary(a.spill or None, a.out if a.exposure else None, a.out, a.dataset,
                                 a.min_contact_cells, boot=a.boot)
            with pd.option_context("display.width", 220, "display.max_rows", 200):
                if "expr" in res:
                    print("expression kernel, calibration pairs (receiver does not make the gene), by unit:")
                    print(res["expr"].round(4).to_string(index=False))
                    print("\nby receiver type:")
                    print(res["expr_types"].round(4).to_string(index=False))
                if "labels" in res:
                    L = res["labels"]
                    print("\nlabel kernel, K = kept ratio touching b / not touching b (median over units), "
                          "pairs with >= 200 touching cells:")
                    print(L[L.cells_touch >= 200].sort_values("K").round(3).to_string(index=False))
            print(f"wrote {a.out}")
        elif a.ccmd == "aggregates":
            from .contacts import aggregates
            sp_ = lambda x: [v for v in x.split(",") if v]
            S, U = aggregates(a.labels, a.out, a.level, sp_(a.immune), a.ref, sp_(a.test), k=a.k, r=a.r,
                              min_size=a.min_size, eps=[float(x) for x in sp_(a.eps)], reps=a.reps, mode=a.mode,
                              unit_pattern=a.unit_pattern, use_sections=a.use_sections or None, seed=a.seed)
            with pd.option_context("display.width", 250, "display.max_columns", 40):
                print("\naggregates per unit (summed over sections):")
                print(U.pivot_table(index="unit", columns="labels", values="aggregates").round(1).to_string())
                print("\nshare of immune cells in aggregates per unit:")
                print(U.pivot_table(index="unit", columns="labels", values="share_in_aggregates").round(3).to_string())
            print(f"wrote {a.out}")
        elif a.ccmd == "contacts":
            from .contacts import contacts, compare_units
            sp_ = lambda x: [v for v in x.split(",") if v]
            S, U, C = contacts(a.labels, a.out, a.level, sp_(a.immune), sp_(a.partners), a.ref, a.test,
                               mixing_immune=sp_(a.mixing_immune) or None, mixing_partner=sp_(a.mixing_partner) or None,
                               contact=a.contact, k=a.k, r=a.r, boundaries=a.boundaries or None, tol=a.tol,
                               eps=[float(x) for x in sp_(a.eps)], reps=a.reps, mode=a.mode, robust=a.robust,
                               unit_pattern=a.unit_pattern, use_sections=a.use_sections or None, seed=a.seed,
                               confusion=a.confusion, fix_partners=a.fix_partners)
            cmp_ = compare_units(U)
            cmp_.to_csv(Path(a.out).expanduser() / "compare_units.csv", index=False)
            with pd.option_context("display.width", 250, "display.max_columns", 40):
                print(f"\nby unit (summed contacts): medians of reference ({a.ref}), analyst ({a.test}) and random "
                      "relabeling of the analyst's labels; units with analyst below reference; units where random "
                      "relabeling moves the analyst's value up")
                print(cmp_.round(3).to_string(index=False))
                cu = C[C.level == "unit"]
                print("\nKeren mixing class by unit (cold < 250 immune cells; compartmentalized < 0.22):")
                print(cu.drop(columns=["level", "name"]).round(3).to_string(index=False))
                cs = C[C.level == "section"]
                print(f"\nsections: class changed in {int(cs.class_changed.sum())} of {len(cs)}; certified wrongly in "
                      f"{int(cs.certified_wrongly.sum())}")
            print(f"wrote {a.out}")
        elif a.ccmd == "calls":
            from .calls import calls
            sp_ = lambda x: [v for v in x.split(",") if v]
            S, U, summ = calls(a.labels, a.out, a.level, a.ref, a.test, sp_(a.immune), sp_(a.partners), k=a.k,
                               nperm=a.nperm, nperm_stress=a.nperm_stress, z_thr=a.z, min_effect=a.min_effect,
                               eps=[float(x) for x in sp_(a.eps)], reps=a.reps, mode=a.mode, robust=a.robust,
                               min_cells=a.min_cells, unit_pattern=a.unit_pattern, use_sections=a.use_sections or None,
                               seed=a.seed)
            e = f"{int(round(100 * max(float(x) for x in sp_(a.eps))))}"
            with pd.option_context("display.width", 250, "display.max_columns", 40):
                print(f"\nby pair, medians over units of {a.ref} (reference) and {a.test} (analyst) log2 O/E; random "
                      f"relabeling at {e}% of the analyst's labels: mean shift and share of draws keeping the call; "
                      f"counts of units by status (calls need |z| >= {a.z} and |log2 O/E| >= {a.min_effect})")
                print(summ.round(3).to_string(index=False))
                print("\nby unit:")
                print(U[["a", "b", "unit", "sections", "L_ref", "L_test", "call_ref", "call_test", "status",
                         f"dL_random_{e}", f"persist_{e}", "certified_wrongly"]].round(3).to_string(index=False))
            print(f"wrote {a.out}")
        elif a.ccmd == "placement":
            from .placement import placement
            sp_ = lambda x: [t for t in x.split(",") if t]
            r = placement(a.labels, a.out, a.level, sp_(a.immune), sp_(a.partners), h5ad_dir=a.h5ad_dir or None,
                          unit_pattern=a.unit_pattern, use_sections=a.use_sections or None,
                          transfer=sp_(a.transfer), models=dict(m.split("=", 1) for m in a.model),
                          louo=sp_(a.louo), kernels=not a.no_kernels, oracle=not a.no_oracle, draws=a.draws,
                          min_cells=a.min_cells, informative=a.informative, neg_ratio=a.neg_ratio,
                          max_pos=a.max_pos, seed=a.seed, nbr_arm=a.nbr_arm,
                          save_models=a.save_models)
            with pd.option_context("display.width", 250, "display.max_columns", 50):
                sm = r["summary"]
                print("\ncolocalization change from nucB, median over units (real = wholeB):")
                print(sm[["pair", "group", "units", "real"] + r["arms"]].round(3).to_string(index=False))
                print(f"\nsign agreement with the real change (pairs with |real| >= {a.informative}), unit medians:")
                print(r["sign_match"][["informative_pairs"] + [f"sign_{x}" for x in r["arms"]]].to_string(index=False))
                u = r["sign_match_units"]
                print("\nby unit (share of that unit's informative pairs):")
                fr = u[["unit", "informative_pairs"]].copy()
                for x in r["arms"]:
                    fr[x] = (u[f"sign_{x}"] / u.informative_pairs).round(2)
                print(fr.to_string(index=False))
            print(f"wrote {a.out}")
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as e:
        print(f"claimbreak: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
