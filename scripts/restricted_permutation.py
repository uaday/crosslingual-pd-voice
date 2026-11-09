"""Restricted permutation test for the age confound (review point 7; Chaibub Neto et al. 2019).

An ordinary permutation test shuffles the labels freely, which destroys the association
between age and diagnosis along with everything else. A model that has only learned age then
beats the null easily. A RESTRICTED permutation shuffles labels only within strata of the
measured confounder (here 5-year age bins, and age-bin x sex cells), so every permuted
dataset keeps the original age-diagnosis association. The null distribution is then "what
this pipeline scores when it can use age but nothing else", and the observed score is
compared with it.

Pipeline (refitted for every permutation): one recording per participant from the
uncorrected mPower cohort, XLS-R layer 11 mean+std embeddings, PCA to 64 dimensions inside
each training fold, logistic regression, 5-fold participant-disjoint cross-validation.
A linear probe is used so that hundreds of refits are affordable; the observed score is
reported for the same probe.

Run:  .venv\\Scripts\\python.exe -u scripts\\restricted_permutation.py --n-perm 200
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdvoice.cohort import attach_sex, load_mpower_cohort
from pdvoice.config import EMBEDDINGS, LOGS

MODEL, LAYER = "xls-r-300m", 11


def cv_auroc(X, y, seed):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), PCA(64, random_state=seed),
                            LogisticRegression(max_iter=2000))
        oof[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def fold_projections(X, seed, k=5):
    """Fixed, label-independent folds with each fold's scaler and PCA fitted once.

    PCA never sees the labels, so with folds that do not depend on the labels (plain
    KFold) the projection can be computed once and reused for every permutation; only the
    logistic regression is refitted. This makes 1,000 permutations feasible on a CPU."""
    from sklearn.model_selection import KFold
    out = []
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        pre = make_pipeline(StandardScaler(), PCA(64, random_state=seed)).fit(X[tr])
        out.append((tr, te, pre.transform(X[tr]), pre.transform(X[te])))
    return out


def cv_auroc_fast(proj, y):
    oof = np.zeros(len(y))
    for tr, te, Ztr, Zte in proj:
        oof[te] = LogisticRegression(max_iter=2000).fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
    return roc_auc_score(y, oof)


def permute_within(y, strata, rng):
    yp = y.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        yp[idx] = rng.permutation(y[idx])
    return yp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="fixed folds, PCA fitted once (see fold_projections)")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))
    df = load_mpower_cohort(cap=1, seed=args.seed)
    # join on the SAMPLED RECORDING's path; this used to keep each speaker's first cached
    # embedding instead, a different recording for 2,885 of 4,809 speakers
    df = attach_sex(df).merge(emb, on="path", how="inner", validate="one_to_one")
    df = df.dropna(subset=["age"]).reset_index(drop=True)
    X = d["X"][df.row.to_numpy()].astype(np.float32)
    y = df.label.to_numpy().astype(int)
    age_bin = (df.age // 5 * 5).astype(int).to_numpy()
    sex = df.sex.fillna("unknown").to_numpy()
    print(f"{len(df):,} participants (PD {y.sum():,} / HC {(y == 0).sum():,})")

    proj = fold_projections(X, args.seed) if args.fast else None
    score = (lambda yy, sd: cv_auroc_fast(proj, yy)) if args.fast else (lambda yy, sd: cv_auroc(X, yy, sd))
    observed = score(y, args.seed)
    age_only = roc_auc_score(y, df.age.to_numpy())
    print(f"observed AUROC (PCA-64 logistic probe): {observed:.3f} | age alone: {age_only:.3f}")

    rng = np.random.default_rng(args.seed)
    nulls = {"unrestricted": [], "within_age_bins": [], "within_age_x_sex": []}
    strata = {"unrestricted": np.zeros(len(y), int), "within_age_bins": age_bin,
              "within_age_x_sex": np.array([f"{a}_{s}" for a, s in zip(age_bin, sex)])}
    for i in range(args.n_perm):
        for k in nulls:
            nulls[k].append(score(permute_within(y, strata[k], rng), args.seed + i + 1))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n_perm}: " + "  ".join(
                f"{k} mean {np.mean(v):.3f}" for k, v in nulls.items()), flush=True)

    res = {"n": int(len(y)), "observed": round(float(observed), 4),
           "age_alone": round(float(age_only), 4), "n_perm": args.n_perm}
    for k, v in nulls.items():
        v = np.asarray(v)
        res[k] = dict(null_mean=round(float(v.mean()), 4), null_sd=round(float(v.std()), 4),
                      null_95th=round(float(np.quantile(v, 0.95)), 4),
                      exceedances=int(np.sum(v >= observed)),
                      p_value=round(float((np.sum(v >= observed) + 1) / (len(v) + 1)), 4),
                      excess_over_null=round(float(observed - v.mean()), 4))
        print(f"{k:<18} null {v.mean():.3f} +/- {v.std():.3f} (95th {np.quantile(v, .95):.3f}) "
              f"-> observed {observed:.3f}, excess {observed - v.mean():+.3f}, "
              f"p = {res[k]['p_value']}")
    res["fast"] = bool(args.fast)
    name = f"restricted_permutation__L{LAYER}" + ("__fast" if args.fast else "") + (f"__n{args.n_perm}" if args.n_perm != 100 else "")
    (LOGS / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {LOGS / (name + '.json')}")


if __name__ == "__main__":
    main()
