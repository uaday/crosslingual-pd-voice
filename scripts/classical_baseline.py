"""Classical-feature baseline (RQ1, Layer 1) — the interpretable comparison to SSL.

Extracts jitter / shimmer / HNR / F0 + 13 MFCCs (Praat via parselmouth) on the SAME
mPower vowel cohort used for the SSL runs, trains SVM + Random Forest with
subject-disjoint 5-fold CV, and reports subject-level AUROC — head-to-head with the
frozen-XLS-R numbers. Classical features are vowel-specific.

Speed: features are computed on the 2 s stable vowel core (same segment the SSL path
uses), which makes Praat ~5x faster than on the full 10 s file. Cached to parquet.

Run:  python scripts/classical_baseline.py --cap 3
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from pdvoice.config import EMBEDDINGS, MPOWER_DIR, SEED
from pdvoice.data.mpower import load_mpower
from pdvoice.data.splits import assign_folds
from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
from pdvoice.preprocess import quality_control
from pdvoice.cohort import age_match_ids
from pdvoice.features.classical import FEATURE_NAMES, classical_features
from pdvoice.utils.audio import load_audio, stable_core, trim_silence

warnings.filterwarnings("ignore")
PRAAT_SR = 44100
# feature list and extractor both live in the package now -- see features/classical.py
FEATS = FEATURE_NAMES


def build_cohort(cap):
    df = load_mpower()
    lab = df[df["label"].isin([0, 1])].sample(frac=1.0, random_state=SEED)
    df = lab[lab.groupby("subject_id").cumcount().to_numpy() < cap].reset_index(drop=True)
    df, _ = quality_control(df)
    df = df[df["path"].astype(str) != "nan"].reset_index(drop=True)
    d = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    age = dict(zip(d["healthCode"].astype(str), d["age"]))
    df["age"] = df["subject_id"].str.replace("mpower_", "", regex=False).map(age)
    return df


def extract(df, cap, max_subjects):
    from tqdm import tqdm

    cache = EMBEDDINGS / f"mpower_classical__cap{cap}.csv"
    if cache.exists() and not max_subjects:
        f = pd.read_csv(cache)
        # length AND columns must match: the feature set changed on 2026-09-06 and a
        # stale 17-column cache would silently be reused for a 42-feature experiment
        if len(f) == len(df) and list(f.columns) == FEATS:
            print(f"  cache hit: {cache.name}")
            return f
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
        F.to_csv(cache, index=False)
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
    lab_out = df[["subject_id", "label", "age"]].reset_index(drop=True).join(F)
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
