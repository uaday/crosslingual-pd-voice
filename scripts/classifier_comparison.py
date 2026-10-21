"""Does the CLASSIFIER choice change our conclusions?  (and: match the field)

Two problems this answers.

1. FAIRNESS. `classical_baseline.py` scores classical features with SVM/RF, while every
   other script scores SSL embeddings with the 256-unit MLP head. The headline
   "classical 0.670 vs SSL 0.803" therefore changes TWO variables at once. Here every
   classifier sees every feature set on identical subject-disjoint folds.

2. COMPARABILITY. The field does not use an MLP head:
     - Klempir 2024 (Sensors)  : Random Forest, XGBoost
     - VoiceFM (Bridge2AI)     : LINEAR probes on frozen embeddings
     - cross-lingual PD work   : SVM
   Linear probing is also the SUPERB convention for evaluating frozen representations,
   because it measures what is IN the embedding rather than what a nonlinear head can
   learn from it. With 525,313 parameters on ~1,100 subjects our MLP can memorise
   (observed: train loss -> 0.009), so a linear probe is arguably the more honest
   instrument for representation-quality claims.

One recording per subject so SSL and classical rows are subject-level and directly
comparable. Age-matched and confounded cohorts both reported.

Run:  .venv\\Scripts\\python.exe -u scripts\\classifier_comparison.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch

from pdvoice.cohort import age_match_ids
from pdvoice.config import EMBEDDINGS, LOGS, MPOWER_DIR
from pdvoice.features.classical import FEATURE_NAMES
from pdvoice.eval.discrimination import bootstrap_ci
from pdvoice.models.head import fit_mlp_head
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL, LAYER = "xls-r-300m", 11
FEATS = FEATURE_NAMES


def sk(name, seed):
    """Field-standard classifiers. HistGradientBoosting stands in for XGBoost."""
    imp = SimpleImputer(strategy="median")
    if name == "logreg (linear probe)":
        return make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    if name == "SVM (RBF)":
        return make_pipeline(imp, StandardScaler(), SVC(probability=True, random_state=seed))
    if name == "Random Forest":
        return make_pipeline(imp, RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1))
    if name == "HistGradBoost (~XGB)":
        return make_pipeline(imp, HistGradientBoostingClassifier(random_state=seed))
    raise ValueError(name)


def cv_scores(X, y, groups, clf_name, seed, n_splits=5):
    """Subject-disjoint CV. One row per subject here, so StratifiedKFold is subject-disjoint."""
    oof = np.zeros(len(y), dtype=float)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        if clf_name == "MLP head (ours)":
            # carve a validation slice out of train, as the real pipeline does
            inner = StratifiedKFold(5, shuffle=True, random_state=seed)
            tr2, va = next(inner.split(X[tr], y[tr]))
            Xi = np.nan_to_num(X, nan=np.nanmedian(X))
            m, _ = fit_mlp_head(Xi[tr][tr2], y[tr][tr2], Xi[tr][va], y[tr][va], seed=seed)
            dev = next(m.parameters()).device
            with torch.no_grad():
                lg = m(torch.tensor(Xi[te], dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
            oof[te] = 1 / (1 + np.exp(-lg))
        else:
            c = sk(clf_name, seed).fit(X[tr], y[tr])
            oof[te] = c.predict_proba(X[te])[:, 1]
    return bootstrap_ci(y, oof), oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    # ---- SSL embeddings, one recording per subject ----
    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"subject_id": d["subject_id"].astype(str), "label": d["label"].astype(int)})
    emb["row"] = np.arange(len(emb))
    one = emb.sample(frac=1.0, random_state=args.seed).drop_duplicates("subject_id")

    # ---- classical features (already subject-level, cap-1) + age ----
    cf = pd.read_csv(LOGS / "classical_features_labeled__cap1.csv")
    demo = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    age = dict(zip(demo.healthCode.astype(str), demo.age))
    one["age"] = one.subject_id.str.replace("mpower_", "", regex=False).map(age)

    merged = one.merge(cf[["subject_id"] + FEATS], on="subject_id", how="inner")
    print(f"subjects with BOTH SSL embedding and classical features: {len(merged):,}")

    X_ssl = d["X"][merged.row.to_numpy()].astype(np.float32)
    X_cls = merged[FEATS].to_numpy(dtype=np.float32)
    X_dem = merged[["age"]].to_numpy(dtype=np.float32)   # demographics-only baseline
    y = merged.label.to_numpy()

    keep = age_match_ids(merged[["subject_id", "label", "age"]]
                         .drop_duplicates("subject_id"), args.seed)
    m_am = merged.subject_id.isin(keep).to_numpy()
    print(f"  confounded cohort : {len(merged):,} subjects "
          f"(PD age {merged[merged.label==1].age.mean():.1f} / HC {merged[merged.label==0].age.mean():.1f})")
    print(f"  age-matched cohort: {int(m_am.sum()):,} subjects "
          f"(PD age {merged[m_am & (merged.label==1)].age.mean():.1f} / "
          f"HC {merged[m_am & (merged.label==0)].age.mean():.1f})\n")

    CLFS = ["logreg (linear probe)", "SVM (RBF)", "Random Forest", "HistGradBoost (~XGB)", "MLP head (ours)"]
    FEATSETS = [("SSL embeddings (2048)", X_ssl), (f"classical ({len(FEATS)})", X_cls), ("age only (1)", X_dem)]

    rows, preds = [], []
    for cohort, mask in [("confounded", np.ones(len(y), bool)), ("age-matched", m_am)]:
        for fname, Xf in FEATSETS:
            for c in CLFS:
                if fname == "age only (1)" and c == "MLP head (ours)":
                    continue  # batch-norm on a single feature is degenerate
                (pt, lo, hi), oof = cv_scores(Xf[mask], y[mask], None, c, args.seed)
                preds.append(pd.DataFrame({
                    "cohort": cohort, "features": fname, "classifier": c,
                    "subject_id": merged.subject_id.to_numpy()[mask],
                    "y": y[mask], "p": oof}))
                rows.append(dict(cohort=cohort, features=fname, classifier=c,
                                 AUROC=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4)))
                print(f"  [{cohort:<11}] {fname:<22} {c:<22} {pt:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
        print()

    R = pd.DataFrame(rows)
    R.to_csv(LOGS / "classifier_comparison.csv", index=False)
    # per-subject predictions, so the objective function can be evaluated at any
    # operating point without re-fitting anything
    pd.concat(preds, ignore_index=True).to_csv(LOGS / "classifier_predictions.csv", index=False)

    pd.set_option("display.width", 200)
    for cohort in ("confounded", "age-matched"):
        print(f"\n{'='*78}\n{cohort.upper()}  ·  AUROC by feature set x classifier\n{'='*78}")
        print(R[R.cohort == cohort].pivot(index="classifier", columns="features",
                                          values="AUROC").to_string())

    print(f"\n{'='*78}\nDOES THE CLASSIFIER CHANGE THE STORY?\n{'='*78}")
    for cohort in ("confounded", "age-matched"):
        sub = R[(R.cohort == cohort) & (R.features != "age only (1)")]
        for f in ("SSL embeddings (2048)", f"classical ({len(FEATS)})"):
            v = sub[sub.features == f].AUROC
            print(f"  {cohort:<12} {f:<22} spread across classifiers: "
                  f"{v.min():.3f} - {v.max():.3f}  (range {v.max()-v.min():.3f})")
    print(f"\nwrote {LOGS / 'classifier_comparison.csv'}")


if __name__ == "__main__":
    main()
