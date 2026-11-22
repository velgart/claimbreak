#!/usr/bin/env python
"""contacts_decompose.py -- where does the per100 change come from?

per100(a|b) = 100 x (a cells touching b) / (b cells) can fall because (i) a cells that touch b are relabeled
(absorption at contact), (ii) a cells away from b are relabeled (a general loss of a labels, which also lowers the
numerator only through its contact share), (iii) b gains cells (the denominator), or (iv) contacts change because
neighbors are relabeled. This script splits the change between a reference arm and a test arm on the same cells and
the same contact graph as `claimbreak compartment contacts` (cells labeled in both arms; 6 nearest within 30 um).

Per section and per unit (sums), for each immune type a and the partner b:
  n_a_ref, n_a_test, n_b_ref, n_b_test          cells per label set
  num_ref, num_test                             a cells touching >= 1 b cell (each label set with its own neighbors)
  ref_a_contact, kept_contact                   ref-a cells touching a ref-b cell, and the share still a in test
  ref_a_away, kept_away                         ref-a cells touching no ref-b cell, and the share still a in test
  contact_excess_loss = (1 - kept_contact) - (1 - kept_away)    loss at contact beyond the loss away from b
  dest_contact_<type>                           where lost contact cells go (shares of the lost)
  gained_num                                    test-a touching test-b that were not a in ref (new a cells at contact)
  factor_num, factor_den                        num_test / num_ref and n_b_ref / n_b_test: per100_test / per100_ref
                                                = factor_num x factor_den
Read only; writes OUT/sections.csv and OUT/units.csv.

  python contacts_decompose.py --labels ~/claimbreak_results/bilous_v4 --ref nucB --test wholeB --level L2m \
      --immune T_NK,Myeloid,B --partner Tumor --unit-pattern 'L\\d(?=_L)' --out ~/claimbreak_results/bilous_v4_decomp_dm
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from claimbreak.contacts import knn_edges


def touches(codes, E, n, t):
    """bool per cell: has a contact neighbor of type code t."""
    m = np.zeros(n, bool)
    a, b = codes[E[:, 0]], codes[E[:, 1]]
    m[E[b == t, 0]] = True
    m[E[a == t, 1]] = True
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ref", default="nucB")
    ap.add_argument("--test", default="wholeB")
    ap.add_argument("--level", default="L2m")
    ap.add_argument("--immune", default="T_NK,Myeloid,B")
    ap.add_argument("--partner", default="Tumour")
    ap.add_argument("--unit-pattern", default=r"L\d(?=_L)")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--r", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    cr, ct = f"{a.level}_{a.ref}", f"{a.level}_{a.test}"
    rows = []
    for f in sorted(Path(a.labels).expanduser().glob("*.labels.csv.gz")):
        D = pd.read_csv(f, usecols=lambda c: c in (cr, ct, "x", "y"), keep_default_na=False, low_memory=False)
        D = D[(D[cr].astype(str) != "") & (D[ct].astype(str) != "")].reset_index(drop=True)
        sample = f.name.split(".labels")[0]
        m = re.search(a.unit_pattern, sample)
        unit = m.group(0) if m else sample
        types = sorted(set(D[cr].astype(str)) | set(D[ct].astype(str)))
        code = {t: i for i, t in enumerate(types)}
        R, T = D[cr].astype(str).map(code).to_numpy(), D[ct].astype(str).map(code).to_numpy()
        n = len(D)
        E = knn_edges(D[["x", "y"]].to_numpy(float), a.k, a.r)
        if a.partner not in code:
            continue
        b = code[a.partner]
        tb_ref, tb_test = touches(R, E, n, b), touches(T, E, n, b)
        for imm in a.immune.split(","):
            if imm not in code:
                continue
            i = code[imm]
            ra, ta = R == i, T == i
            c_ref, away = ra & tb_ref, ra & ~tb_ref
            lost_c = c_ref & ~ta
            row = dict(sample=sample, unit=unit, immune=imm, partner=a.partner, cells=n,
                       n_a_ref=int(ra.sum()), n_a_test=int(ta.sum()), n_b_ref=int((R == b).sum()),
                       n_b_test=int((T == b).sum()), num_ref=int(c_ref.sum()), num_test=int((ta & tb_test).sum()),
                       ref_a_contact=int(c_ref.sum()), kept_contact_n=int((c_ref & ta).sum()),
                       ref_a_away=int(away.sum()), kept_away_n=int((away & ta).sum()),
                       gained_num=int((ta & tb_test & ~ra).sum()), lost_contact_n=int(lost_c.sum()))
            for t, j in code.items():
                if j != i:
                    row[f"dest_contact_{t}"] = int((lost_c & (T == j)).sum())
            rows.append(row)
    S = pd.DataFrame(rows).fillna(0)

    def derive(X):
        X = X.copy()
        X["kept_contact"] = X.kept_contact_n / X.ref_a_contact.clip(lower=1)
        X["kept_away"] = X.kept_away_n / X.ref_a_away.clip(lower=1)
        X["contact_excess_loss"] = (1 - X.kept_contact) - (1 - X.kept_away)
        X["per100_ref"] = 100 * X.num_ref / X.n_b_ref.clip(lower=1)
        X["per100_test"] = 100 * X.num_test / X.n_b_test.clip(lower=1)
        X["factor_num"] = X.num_test / X.num_ref.clip(lower=1)
        X["factor_den"] = X.n_b_ref / X.n_b_test.clip(lower=1)
        X["infil_ref"] = X.num_ref / X.n_a_ref.clip(lower=1)
        X["infil_test"] = X.num_test / X.n_a_test.clip(lower=1)
        return X

    S = derive(S)
    num = [c for c in S.columns if c not in ("sample", "unit", "immune", "partner") and not c.startswith(
        ("kept_contact", "kept_away", "contact_excess", "per100", "factor", "infil"))] + ["kept_contact_n", "kept_away_n"]
    num = list(dict.fromkeys(c for c in num if c in S.columns and pd.api.types.is_numeric_dtype(S[c])))
    U = derive(S.groupby(["unit", "immune", "partner"])[num].sum().reset_index())
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "sections.csv", index=False)
    U.to_csv(out / "units.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    show = ["unit", "immune", "n_a_ref", "n_a_test", "n_b_ref", "n_b_test", "per100_ref", "per100_test", "factor_num",
            "factor_den", "infil_ref", "infil_test", "kept_contact", "kept_away", "contact_excess_loss", "gained_num"]
    print(f"{a.ref} -> {a.test}, partner {a.partner}, by unit:")
    print(U[show].round(3).to_string(index=False))
    dc = [c for c in U.columns if c.startswith("dest_contact_")]
    D = U[["unit", "immune", "lost_contact_n"] + dc].copy()
    D[dc] = D[dc].div(D.lost_contact_n.clip(lower=1), axis=0)
    print("\nwhere the lost contact cells go (share of lost):")
    print(D.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
