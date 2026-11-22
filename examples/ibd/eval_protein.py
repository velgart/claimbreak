r"""eval_protein.py -- transcript labels against protein classes on the same cells (read only).

Inputs: a labels folder from `claimbreak compartment label` run on protein_section.h5ad (cellscape_classes.py) and
that script's protein_classes.csv. Cells with a protein class (not mixed / unassigned / no_protein) are scored.
Transcript labels are mapped to protein classes with --map (default: Fibroblast, Muscle -> Stromal; labels with no
protein counterpart, e.g. Mast, Neural, Neuron, count as 'other' and are always wrong).

Prints and writes OUT/confusion.csv (protein class x transcript label), OUT/by_class.csv (per protein class: cells,
recall; per mapped transcript label: precision), overall agreement. With several label columns (--cols) each is
scored.

Usage: python eval_protein.py --labels LABELS_DIR --protein CLASSES_DIR/protein_classes.csv \
           --cols L1_whole --out ~/claimbreak_results/eval_protein_census [--map "Fibroblast=Stromal,Muscle=Stromal"]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--protein", required=True)
ap.add_argument("--cols", default="L1_whole")
ap.add_argument("--map", default="Fibroblast=Stromal,Muscle=Stromal,Mesenchymal=Stromal,T/NK=T_NK")
ap.add_argument("--out", required=True)
a = ap.parse_args()

out = Path(a.out).expanduser()
if out.exists() and any(out.iterdir()):
    sys.exit(f"{out} exists and is not empty; choose a new --out")
out.mkdir(parents=True, exist_ok=True)
P = pd.read_csv(a.protein, index_col=0, dtype=str, keep_default_na=False)
files = sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz"))
if not files:
    sys.exit(f"no labels files in {a.labels}")
L = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]).set_index("cell_id")
mp = dict(x.split("=", 1) for x in a.map.split(",") if x)
ok = ~P.protein_class.isin(["mixed", "unassigned", "no_protein"])
truth = P.protein_class[ok]
classes = sorted(truth.unique())
rows, conf_all = [], []
for col in a.cols.split(","):
    if col not in L:
        sys.exit(f"{col} not in the labels files; columns {[c for c in L.columns if not c.endswith(('_second', '_margin'))]}")
    lab = L[col].reindex(truth.index).fillna("")
    mapped = lab.map(lambda x: mp.get(x, x))
    mapped = mapped.where(mapped.isin(classes) | (mapped == ""), "other")
    m = mapped != ""
    conf = pd.crosstab(truth[m], lab[m])
    conf.index.name, conf.columns.name = "protein", col
    conf_all.append(conf.assign(labels=col))
    acc = float((mapped[m] == truth[m]).mean())
    print(f"\n{col}: {int(m.sum()):,} cells with a protein class and a label; agreement {acc:.3f}")
    print(conf.to_string())
    for c in classes:
        t = m & (truth == c)
        p = m & (mapped == c)
        rows.append({"labels": col, "class": c, "protein_cells": int(t.sum()),
                     "recall": float((mapped[t] == c).mean()) if t.any() else float("nan"),
                     "label_cells": int(p.sum()),
                     "precision": float((truth[p] == c).mean()) if p.any() else float("nan")})
    rows.append({"labels": col, "class": "(all)", "protein_cells": int(m.sum()), "recall": acc,
                 "label_cells": int(m.sum()), "precision": acc})
B = pd.DataFrame(rows)
print("\nper protein class: recall (share of the class's cells labelled as it) and precision of the label:")
print(B.round(3).to_string(index=False))
B.to_csv(out / "by_class.csv", index=False)
pd.concat(conf_all).to_csv(out / "confusion.csv")
print(f"wrote {out}")
