"""Classical-feature baseline (RQ1, Layer 1) — the interpretable comparison to SSL.

Extracts jitter / shimmer / HNR / F0 + 13 MFCCs (Praat via parselmouth) on the SAME
mPower vowel cohort used for the SSL runs, trains SVM + Random Forest with
subject-disjoint 5-fold CV, and reports subject-level AUROC — head-to-head with the
frozen-XLS-R numbers. Classical features are vowel-specific.

Features are computed on the central 2 s of the trimmed vowel, the same segment of the
same recordings the SSL path uses. Cached to CSV with the recording paths.

Run:  python scripts/classical_baseline.py --cap 3
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from pdvoice.config import EMBEDDINGS, SEED
from pdvoice.data.splits import assign_folds
from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
from pdvoice.cohort import age_match_ids, load_mpower_cohort
from pdvoice.features.classical import FEATURE_NAMES, FEATURE_VERSION, classical_features

warnings.filterwarnings("ignore")
# feature list and extractor both live in the package now -- see features/classical.py
FEATS = FEATURE_NAMES


def build_cohort(cap):
    """The shared mPower cohort (load_mpower_cohort): the SAME recordings the SSL arm
    embeds, so fusion and the classifier comparison can join the two arms on the
    recording path. This script used to draw its own recording per speaker, so a join on
    subject_id could pair one recording's Praat features with another's embedding."""
    df = load_mpower_cohort(cap=cap, seed=SEED)
    return df[df["path"].astype(str) != "nan"].reset_index(drop=True)


def extract(df, cap, max_subjects):
    from tqdm import tqdm

    cache = EMBEDDINGS / f"mpower_classical__cap{cap}__{FEATURE_VERSION}.csv"
    if cache.exists() and not max_subjects:
        f = pd.read_csv(cache)
        # identity guard: the cache must hold these exact recordings, in this order, and the
        # current feature set (a stale cache of the right length would silently misalign)
        if (list(f.columns) == ["path", *FEATS]
                and f["path"].tolist() == df["path"].astype(str).tolist()):
            print(f"  cache hit: {cache.name}")
            return f[FEATS]
        print(f"  cache stale ({len(f)} rows x {len(f.columns)} cols) -> re-extracting")
    rows = []
    for p in tqdm(df["path"], unit="rec"):
        try:
            rows.append(classical_features(p))
        except Exception:  # noqa: BLE001
            rows.append({k: np.nan for k in FEATS})
    F = pd.DataFrame(rows, columns=FEATS)
    if not max_subjects:
        cache.parent.mkdir(parents=True, exist_ok=True)
        F.assign(path=df["path"].astype(str).to_numpy())[["path", *FEATS]].to_csv(cache, index=False)
    return F


def cv_auroc(df, F, y, model):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    df = assign_folds(df.reset_index(drop=True), n_splits=5)
    X = F.to_numpy()
    oof = []
    for k in range(5):
        tr = (df["fold"] != k).to_numpy() & (df["fold"] >= 0).to_numpy()
        te = (df["fold"] == k).to_numpy()
        if model == "svm":
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                SVC(probability=True, C=1.0, gamma="scale", random_state=SEED))
        else:
            clf = make_pipeline(SimpleImputer(strategy="median"),
                                RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof.append(pd.DataFrame({"s": df.loc[te, "subject_id"].to_numpy(), "y": y[te], "p": p}))
    g = pd.concat(oof).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    yy, pp = g["y"].to_numpy(), g["p"].to_numpy()
    a, lo, hi = bootstrap_ci(yy, pp, metric=auroc)
    return a, lo, hi, auprc(yy, pp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    print("=== Classical baseline (jitter/shimmer/HNR/F0 + MFCC) vs SSL ===")
    df = build_cohort(args.cap)
    if args.max_subjects:
        keep = (df[df.label.isin([0, 1])].groupby("label")["subject_id"]
                .apply(lambda s: s.drop_duplicates().head(args.max_subjects)).tolist())
        df = df[df["subject_id"].isin(set(keep))].reset_index(drop=True)
    print(f"  cohort: {len(df)} recs / {df.subject_id.nunique()} subj")

    print("  extracting classical features (2 s core) ...")
    F = extract(df, args.cap, args.max_subjects).reset_index(drop=True)
    ok = int(F.notna().any(axis=1).sum())
    print(f"  features: {F.shape}  ({ok} non-empty)  in {(time.time()-t0)/60:.1f} min")

    y = df["label"].astype(int).to_numpy()
    # save labeled features for downstream feature analysis (SHAP / distributions)
    from pdvoice.config import LOGS
    lab_out = df[["subject_id", "path", "label", "age"]].reset_index(drop=True).join(F)
    lab_out.to_csv(LOGS / f"classical_features_labeled__cap{args.cap}.csv", index=False)

    scenarios = {"baseline": np.ones(len(df), bool), "age-matched": df.subject_id.isin(age_match_ids(df)).to_numpy()}
    print(f"\n  {'scenario':<13}{'model':<6}{'AUROC':<22}{'AUPRC':<8}{'PD/HC subj'}")
    SSL_REF = {"baseline": 0.803, "age-matched": 0.621}   # XLS-R L15 within-mPower
    results = {"cap": args.cap, "n_rec": len(df), "SSL_reference": SSL_REF, "scenarios": {}}
    for name, mask in scenarios.items():
        sub, Fm, ym = df[mask].reset_index(drop=True), F[mask].reset_index(drop=True), y[mask]
        npd = sub[sub.label == 1].subject_id.nunique(); nhc = sub[sub.label == 0].subject_id.nunique()
        results["scenarios"][name] = {"n_pd": npd, "n_hc": nhc, "models": {}}
        for mdl in ("svm", "rf"):
            a, lo, hi, ap = cv_auroc(sub, Fm, ym, mdl)
            results["scenarios"][name]["models"][mdl] = {"auroc": a, "auroc_ci": [lo, hi], "auprc": ap}
            print(f"  {name:<13}{mdl:<6}{a:.3f} [{lo:.3f},{hi:.3f}]      {ap:.3f}   {npd}/{nhc}")
        print(f"  {name:<13}{'SSL':<6}{SSL_REF[name]:.3f}  (XLS-R L15, reference)")
    import json
    (LOGS / f"classical_baseline__cap{args.cap}.json").write_text(json.dumps(results, indent=2))
    print(f"\nTotal {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
