"""I2 -- does controlling SEX as well as age change anything?

Sex became a validity item rather than a nicety once two facts landed together:

    the embeddings encode sex at AUROC 0.971   (embedding_probes.py)
    mPower patients are 34.6% female against 18.6% of controls

and age matching does NOT fix it -- it only narrows the gap to 33.4% vs 22.3%
(chi-square p = 5.6e-05). Sex alone predicts the label at 0.58. So a model trained on
these embeddings has a second demographic shortcut available, independent of age, and
nothing in this project has ever controlled it.

This script asks three questions:

    1. how much does the demographics-only baseline gain from adding sex to age?
    2. what does the honest AUROC become when the cohort is matched on age AND sex?
    3. does the model behave differently for men and women (stratified evaluation)?

Question 3 matters on its own: a screening tool that works for one sex and not the
other is a fairness problem regardless of its overall AUROC.

Run:  .venv\\Scripts\\python.exe -u scripts\\sex_control.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from pdvoice.cohort import (age_match_ids, attach_sex, cap_per_subject,
                            demographic_match_ids, load_mpower_cohort)
from pdvoice.config import EMBEDDINGS, LOGS
from pdvoice.eval.discrimination import bootstrap_ci
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL, LAYER = "xls-r-300m", 15


def probe(X, y, groups, seed=42, n_splits=5):
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))

    df = load_mpower_cohort(cap=1, seed=args.seed)
    # join on the SAMPLED RECORDING's path; this used to keep each speaker's first cached
    # embedding instead, a different recording for 2,885 of 4,809 speakers
    df = attach_sex(df).merge(emb, on="path", how="inner", validate="one_to_one")
    df = df.dropna(subset=["sex", "age"]).reset_index(drop=True)
    X = d["X"][df.row.to_numpy()].astype(np.float32)
    y = df.label.to_numpy()
    print(f"cohort with sex, age and an embedding: {len(df):,} speakers\n")

    cohorts = {
        "raw": np.ones(len(df), bool),
        "age-matched": df.subject_id.isin(age_match_ids(df, seed=args.seed)).to_numpy(),
        "age+sex-matched": df.subject_id.isin(
            demographic_match_ids(df, on=("age", "sex"), seed=args.seed)).to_numpy(),
    }

    rows = []
    for name, m in cohorts.items():
        sub, ys, Xs = df[m], y[m], X[m]
        f_hc = 100 * (sub[(sub.label == 0)].sex == "Female").mean()
        f_pd = 100 * (sub[(sub.label == 1)].sex == "Female").mean()
        a_age = roc_auc_score(ys, sub.age.to_numpy())
        a_sex = roc_auc_score(ys, (sub.sex == "Female").astype(int).to_numpy())
        both = np.c_[sub.age.to_numpy(), (sub.sex == "Female").astype(float).to_numpy()]
        a_both = roc_auc_score(ys, probe(both, ys, sub.subject_id.to_numpy(), args.seed))
        oof = probe(Xs, ys, sub.subject_id.to_numpy(), args.seed)
        a_ssl, lo, hi = bootstrap_ci(ys, oof)

        print(f"=== {name} ===  {len(sub):,} speakers "
              f"(PD {int(ys.sum()):,} / HC {int((ys == 0).sum()):,})")
        print(f"   female: HC {f_hc:.1f}%  PD {f_pd:.1f}%   gap {f_pd - f_hc:+.1f} pts")
        print(f"   age alone       AUROC {a_age:.3f}")
        print(f"   sex alone       AUROC {a_sex:.3f}")
        print(f"   age + sex       AUROC {a_both:.3f}   <- demographics-only baseline")
        print(f"   SSL embeddings  AUROC {a_ssl:.3f} [{lo:.3f}, {hi:.3f}]")

        # ---- stratified: does it work equally for men and women? ----
        strat = {}
        for s in ("Male", "Female"):
            sm = (sub.sex == s).to_numpy()
            if sm.sum() > 60 and len(np.unique(ys[sm])) == 2:
                strat[s] = float(roc_auc_score(ys[sm], oof[sm]))
                print(f"   ...within {s:<6} AUROC {strat[s]:.3f}  (n={int(sm.sum()):,})")
        print()

        rows.append(dict(cohort=name, n=len(sub), pct_female_hc=round(f_hc, 1),
                         pct_female_pd=round(f_pd, 1),
                         auroc_age=round(a_age, 4), auroc_sex=round(a_sex, 4),
                         auroc_age_sex=round(a_both, 4),
                         auroc_ssl=round(a_ssl, 4), ci=[round(lo, 4), round(hi, 4)],
                         **{f"ssl_within_{k.lower()}": round(v, 4) for k, v in strat.items()}))

    R = pd.DataFrame(rows)
    R.to_csv(LOGS / "sex_control.csv", index=False)
    (LOGS / "sex_control.json").write_text(json.dumps(rows, indent=2))

    print("=" * 92)
    print("WHAT CONTROLLING SEX COSTS AND BUYS")
    print("=" * 92)
    a, b = R[R.cohort == "age-matched"].iloc[0], R[R.cohort == "age+sex-matched"].iloc[0]
    print(f"  speakers        {a.n:,}  ->  {b.n:,}   ({b.n - a.n:+,})")
    print(f"  sex gap         {a.pct_female_pd - a.pct_female_hc:+.1f} pts  ->  "
          f"{b.pct_female_pd - b.pct_female_hc:+.1f} pts")
    print(f"  sex alone       {a.auroc_sex:.3f}  ->  {b.auroc_sex:.3f}"
          f"   (0.5 = fully controlled)")
    print(f"  demographics    {a.auroc_age_sex:.3f}  ->  {b.auroc_age_sex:.3f}")
    print(f"  SSL embeddings  {a.auroc_ssl:.3f}  ->  {b.auroc_ssl:.3f}"
          f"   ({b.auroc_ssl - a.auroc_ssl:+.3f})")
    if "ssl_within_male" in R.columns and "ssl_within_female" in R.columns:
        print(f"\n  fairness on the age+sex-matched cohort: "
              f"male {b.ssl_within_male:.3f} vs female {b.ssl_within_female:.3f} "
              f"(difference {b.ssl_within_male - b.ssl_within_female:+.3f})")
    print(f"\nwrote {LOGS / 'sex_control.csv'}")


if __name__ == "__main__":
    main()
