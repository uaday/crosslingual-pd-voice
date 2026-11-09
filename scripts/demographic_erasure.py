"""Erase age and sex from the embeddings, then ask what is left.

Supervisor's question (meeting, Oct 2026): instead of discarding participants to match on
age, can the age and sex information be removed from the encoder output itself, so the
classifier can only use what remains?

Method: LEACE (Belrose et al., 2023), a closed-form linear eraser. Fitted on the SOURCE
training data only, it removes every direction of the embedding that is linearly
correlated with the concept vector z = [age, age^2, female], so that no linear model can
predict z from the erased embedding better than a constant. The same fixed map is then
applied unchanged to the external datasets (their demographics are never used).

    r(x) = x - W^+ P W (x - mu),   W = Sigma^{-1/2},   P = projector onto col(W Sigma_xz)

What it does NOT remove: non-linear traces of age and sex. Part 1 measures those with a
non-linear probe, so the result is not over-read.

Three parts:
  1. leakage   - can age / sex still be predicted after erasure? (linear and non-linear)
  2. within    - mPower PD AUROC, participant-disjoint CV, eraser refitted in every fold
  3. transfer  - locked head trained on erased mPower, applied to NeuroVoz, IPVS, MDVR-KCL,
                 confounded and age-matched source, five seeds

Run:  .venv\\Scripts\\python.exe -u scripts\\demographic_erasure.py
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdvoice.calibration import select_best_by_brier
from pdvoice.cohort import age_match_ids, attach_sex, load_mpower_cohort
from pdvoice.config import EMBEDDINGS, LOGS, VAL_FRACTION
from pdvoice.data import LOADERS
from pdvoice.eval.discrimination import auroc
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control
from pdvoice.data.splits import ensure_two_class_val
from pdvoice.features.ssl import load_aligned

MODEL = "xls-r-300m"
LAYER = int(os.environ.get("PDV_LAYER", 15))
EXTERNALS = [("neurovoz", "vowel"), ("ipvs", "vowel"), ("mdvr_kcl", "speech")]


# --------------------------------------------------------------------------- eraser
class LeaceEraser:
    """Closed-form linear concept erasure (Belrose et al., 2023)."""

    def fit(self, X, Z, ridge=1e-3):
        X = X.astype(np.float64)
        Z = Z.astype(np.float64)
        self.mu = X.mean(0)
        Xc, Zc = X - self.mu, Z - Z.mean(0)
        sigma = Xc.T @ Xc / len(X)
        evals, evecs = np.linalg.eigh(sigma)
        evals = np.clip(evals, 0, None) + ridge * evals.max()
        W = (evecs / np.sqrt(evals)) @ evecs.T          # Sigma^{-1/2}
        W_pinv = (evecs * np.sqrt(evals)) @ evecs.T     # Sigma^{1/2}
        A = W @ (Xc.T @ Zc / len(X))                    # whitened cross-covariance
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        U = U[:, s > s.max() * 1e-8]
        self.M = W_pinv @ U @ U.T @ W                   # x -> component to remove
        return self

    def transform(self, X):
        X = X.astype(np.float64)
        return (X - (X - self.mu) @ self.M.T).astype(np.float32)


def concept(df):
    a = (df.age.to_numpy() - 50.0) / 15.0
    return np.c_[a, a ** 2, (df.sex == "Female").to_numpy().astype(float)]


# --------------------------------------------------------------------------- helpers
def load_external(ds, task):
    edf = LOADERS[ds]()
    edf = edf[edf.task_kind == task].copy()
    edf, _ = quality_control(edf)
    edf = edf.reset_index(drop=True)
    # recording-level identity, required: a cache without paths is refused
    E = load_aligned(EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{LAYER}.npz",
                     edf.path).astype(np.float32)
    return edf, E


def train_locked(X, y, groups, seed, erase_df=None):
    """Locked head + calibrator. With `erase_df`, a LEACE eraser is fitted on the head's
    TRAINING rows only (after the split) and applied to everything else; it used to be
    fitted on the whole source cohort, validation rows included. Returns the transform
    to apply to external data as well."""
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed).split(X, y, groups))
    tr, va = ensure_two_class_val(tr, va, np.asarray(y), np.asarray(groups), seed)
    tf = (lambda E: E)
    if erase_df is not None:
        tf = LeaceEraser().fit(X[tr], concept(erase_df.iloc[tr])).transform
    head, vlog = fit_mlp_head(tf(X[tr]), y[tr], tf(X[va]), y[va], seed=seed)
    _, cal = select_best_by_brier(vlog, y[va])
    return head, cal, tf


def score(head, cal, E):
    dev = next(head.parameters()).device
    with torch.no_grad():
        lg = head(torch.tensor(E, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
    return cal.ranking(lg)  # discrimination: strictly increasing map, no ties (RankingMap)


def subject_auroc(sid, y, p):
    g = pd.DataFrame({"s": sid, "y": y, "p": p}).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    return float(auroc(g["y"], g["p"]))


def folds(y, groups, seed, k=5):
    return StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups)


# --------------------------------------------------------------------------- parts
def leakage(X, df, seed):
    """Out-of-fold age R^2 and sex AUROC, linear and non-linear, before and after erasure."""
    sid, age = df.subject_id.to_numpy(), df.age.to_numpy()
    fem = (df.sex == "Female").to_numpy().astype(int)
    out = {}
    for cond in ("original", "erased"):
        preds = {k: np.zeros(len(df)) for k in ("age_lin", "age_nl", "sex_lin", "sex_nl")}
        for tr, te in folds(fem, sid, seed):
            Xtr, Xte = X[tr], X[te]
            if cond == "erased":
                er = LeaceEraser().fit(Xtr, concept(df.iloc[tr]))
                Xtr, Xte = er.transform(Xtr), er.transform(Xte)
            pca = make_pipeline(StandardScaler(), PCA(64, random_state=seed)).fit(Xtr)
            Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
            preds["age_lin"][te] = make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 4, 13))) \
                .fit(Xtr, age[tr]).predict(Xte)
            preds["sex_lin"][te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) \
                .fit(Xtr, fem[tr]).predict_proba(Xte)[:, 1]
            preds["age_nl"][te] = HistGradientBoostingRegressor(random_state=seed).fit(Ptr, age[tr]).predict(Pte)
            preds["sex_nl"][te] = HistGradientBoostingClassifier(random_state=seed) \
                .fit(Ptr, fem[tr]).predict_proba(Pte)[:, 1]
        out[cond] = dict(age_r2_linear=r2_score(age, preds["age_lin"]),
                         age_r2_nonlinear=r2_score(age, preds["age_nl"]),
                         sex_auroc_linear=roc_auc_score(fem, preds["sex_lin"]),
                         sex_auroc_nonlinear=roc_auc_score(fem, preds["sex_nl"]))
    return out


def within(X, df, seed):
    """Participant-level PD AUROC, linear probe as in restricted_permutation.py."""
    sid, y = df.subject_id.to_numpy(), df.label.to_numpy().astype(int)
    res = {}
    for cond in ("original", "erased"):
        oof = np.zeros(len(y))
        for tr, te in folds(y, sid, seed):
            Xtr, Xte = X[tr], X[te]
            if cond == "erased":
                er = LeaceEraser().fit(Xtr, concept(df.iloc[tr]))
                Xtr, Xte = er.transform(Xtr), er.transform(Xte)
            clf = make_pipeline(StandardScaler(), PCA(64, random_state=seed),
                                LogisticRegression(max_iter=2000))
            oof[te] = clf.fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        res[cond] = subject_auroc(sid, y, oof)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    seeds = [int(s) for s in args.seeds.split(",")]

    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))
    base = attach_sex(load_mpower_cohort(cap=3, seed=42))
    n_cohort = len(base)
    base = base.merge(emb, on="path", how="inner")
    assert len(base) <= n_cohort and base.row.is_unique, "join fan-out"
    n_join = base.subject_id.nunique()
    base = base[base.sex.isin(["Female", "Male"])].dropna(subset=["age"]).reset_index(drop=True)
    print(f"mPower cap-3 cohort: {n_join:,} speakers; with age and sex: "
          f"{base.subject_id.nunique():,} speakers, {len(base):,} recordings", flush=True)
    X_all = d["X"][base.row.to_numpy()].astype(np.float32)

    cohorts = {"confounded": lambda s: np.ones(len(base), bool),
               "age-matched": lambda s: base.subject_id.isin(age_match_ids(base, seed=s)).to_numpy()}

    # 1 + 2: leakage and within-mPower, seed 42
    report = {"n_speakers": int(base.subject_id.nunique()), "n_recordings": int(len(base))}
    for cname, maskfn in cohorts.items():
        m = maskfn(42)
        sub, X = base[m].reset_index(drop=True), X_all[m]
        ys = sub.groupby("subject_id").label.first()
        ages = sub.groupby("subject_id").age.first()
        age_only = roc_auc_score(ys, ages.loc[ys.index])
        lk = leakage(X, sub, 42)
        wi = within(X, sub, 42)
        report[cname] = dict(n=int(sub.subject_id.nunique()), age_only_auroc=age_only,
                             leakage=lk, within=wi)
        print(f"\n=== {cname} source ({sub.subject_id.nunique():,} speakers), age alone {age_only:.3f}")
        for cond in ("original", "erased"):
            L = lk[cond]
            print(f"  {cond:<9} age R2 lin {L['age_r2_linear']:+.3f}  non-lin {L['age_r2_nonlinear']:+.3f} | "
                  f"sex AUROC lin {L['sex_auroc_linear']:.3f}  non-lin {L['sex_auroc_nonlinear']:.3f} | "
                  f"PD AUROC within mPower {wi[cond]:.3f}", flush=True)

    # 3: transfer, locked MLP head, five seeds
    ext = {ds: load_external(ds, task) for ds, task in EXTERNALS}
    rows = []
    for cname, maskfn in cohorts.items():
        for seed in seeds:
            m = maskfn(seed)
            sub, X = base[m].reset_index(drop=True), X_all[m]
            y, g = sub.label.to_numpy(), sub.subject_id.to_numpy()
            for cond in ("original", "erased"):
                head, cal, tf = train_locked(X, y, g, seed,
                                             erase_df=sub if cond == "erased" else None)
                for ds, (edf, E) in ext.items():
                    a = subject_auroc(edf.subject_id, edf.label.astype(int), score(head, cal, tf(E)))
                    rows.append(dict(source=cname, seed=seed, cond=cond, target=ds, auroc=round(a, 4)))
            print(f"  [{cname}, seed {seed}] done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(LOGS / "demographic_erasure_transfer.csv", index=False)
    piv = R.groupby(["source", "target", "cond"]).auroc.agg(["mean", "std"]).unstack("cond").round(3)
    print(f"\n{'=' * 80}\nTRANSFER (subject-level AUROC, mean +/- SD over {len(seeds)} seeds)\n{'=' * 80}")
    print(piv.to_string())
    means = R.groupby(["source", "cond"]).auroc.mean().unstack("cond").round(3)
    print("\nmean over the three external datasets:\n" + means.to_string())
    report["transfer_mean"] = {s: means.loc[s].to_dict() for s in means.index}
    (LOGS / "demographic_erasure.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {LOGS / 'demographic_erasure.json'}")


if __name__ == "__main__":
    main()
