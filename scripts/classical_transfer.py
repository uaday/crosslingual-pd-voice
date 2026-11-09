"""Classical-feature transfer: train on mPower vowels, test on IPVS and NeuroVoz vowels.

WHY THIS EXISTS. Every classical-feature number in this project was within-mPower --
`classical_baseline.py`, `classifier_comparison.py` and `fusion.py` all evaluate on the
training corpus only. That left RQ2 with a hole: the thesis says classical features and
SSL embeddings are "statistically indistinguishable", but RQ2 asks about GENERALISATION,
and the classical arm had no generalisation evidence at all. A within-dataset equivalence
cannot support a claim about transfer.

This runs the classical branch through exactly the protocol `layer2_transfer.py` applies
to the SSL branch, so the two are comparable:

    mPower vowel -> 42 Praat descriptors -> SAME MLP head -> calibrate on a
    subject-disjoint validation fold -> LOCK -> score each external corpus once.

Only VOWEL corpora are eligible. The descriptors are vowel-specific (sustained phonation
assumes a steady source), so IPVS and NeuroVoz vowels qualify and MDVR-KCL, which is
speech-only, does not.

Two arms, matching the SSL work: the confounded cohort and the age-matched cohort.

Run:  .venv\\Scripts\\python.exe scripts\\classical_transfer.py --cap 3
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from pdvoice.calibration import select_best_by_brier
from pdvoice.cohort import age_match_ids, load_mpower_cohort
from pdvoice.config import EMBEDDINGS, LOGS, SEED, VAL_FRACTION
from pdvoice.data import LOADERS
from pdvoice.data.splits import assign_folds
from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
from pdvoice.eval.metrics import calibration_error, sensitivity_at_specificity, \
    threshold_metrics, youden_threshold
from pdvoice.features.classical import FEATURE_NAMES, FEATURE_VERSION, classical_features
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control
from pdvoice.data.splits import ensure_two_class_val

VOWEL_EXTERNALS = ("ipvs", "neurovoz")


def extract(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """42 Praat descriptors per recording, cached and identity-guarded.

    The cache is keyed on the recording PATHS, not just the row count. A stale cache with
    the right length but the wrong rows would pair every recording with someone else's
    features and return chance performance without raising.
    """
    from tqdm import tqdm

    cache = EMBEDDINGS / f"classical__{tag}__{FEATURE_VERSION}.csv"
    if cache.exists():
        f = pd.read_csv(cache)
        if (list(f.columns) == ["path"] + list(FEATURE_NAMES)
                and f["path"].tolist() == df["path"].astype(str).tolist()):
            print(f"      cache hit: {cache.name}")
            return f[list(FEATURE_NAMES)]
        print(f"      cache does not match this cohort -> re-extracting")

    rows = []
    for p in tqdm(df["path"], unit="rec", desc=f"      {tag}"):
        try:
            rows.append(classical_features(p))
        except Exception:  # noqa: BLE001 -- a failed file becomes NaN, imputed below
            rows.append({k: np.nan for k in FEATURE_NAMES})
    F = pd.DataFrame(rows, columns=list(FEATURE_NAMES))
    out = F.copy()
    out.insert(0, "path", df["path"].astype(str).to_numpy())
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return F


def subject_level(ids, y, pr) -> pd.DataFrame:
    p, r = pr
    return (pd.DataFrame({"subject_id": ids, "y": y, "p": p, "r": r})
            .groupby("subject_id").agg(y=("y", "first"), p=("p", "mean"), r=("r", "mean"))
            .reset_index())


def evaluate(y, p, thr, r) -> dict:
    """Discrimination (AUROC, AUPRC, sensitivity at 90% specificity) on the ranking score
    r; calibration on the probability p; decisions are p >= thr, where thr is one value
    or one per participant (its own fold's validation threshold, possibly infinite)."""
    lo, hi = bootstrap_ci(y, r)[1:]
    m = {"n_subjects": int(len(y)), "prevalence": round(float(np.mean(y)), 4),
         "auroc": round(float(auroc(y, r)), 4), "auroc_ci": [round(lo, 4), round(hi, 4)],
         "auprc": round(float(auprc(y, r)), 4)}
    m.update({k: round(float(v), 4) for k, v in calibration_error(y, p).items()})
    m.update({k: (round(float(v), 4) if isinstance(v, float) else v)
              for k, v in threshold_metrics(y, p, thr).items()})
    m["sens_at_90spec"] = round(float(sensitivity_at_specificity(y, r)["sensitivity"]), 4)
    # NPV is not emitted by threshold_metrics but is recoverable from the matrix.
    tn, fn = m["tn"], m["fn"]
    m["npv"] = round(tn / (tn + fn), 4) if (tn + fn) else 0.0
    return m


def fit_prep(X):
    """Median imputation and z-scoring with statistics from X (the training rows) only."""
    med = np.nanmedian(X, axis=0)
    Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi.mean(0), Xi.std(0) + 1e-8
    return lambda A: (np.where(np.isnan(A), med, A) - mu) / sd


def fit_locked_head(X, y, groups, seed=SEED):
    """Imputer/scaler, head, calibrator and threshold, all fitted without test data."""
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed)
                  .split(X, y, groups))
    tr, va = ensure_two_class_val(tr, va, np.asarray(y), np.asarray(groups), seed)
    prep = fit_prep(X[tr])
    head, vlog = fit_mlp_head(prep(X[tr]), y[tr], prep(X[va]), y[va], seed=seed)
    cal_name, cal = select_best_by_brier(vlog, y[va])
    thr = youden_threshold(y[va], cal.predict_proba(vlog))
    dev = next(head.parameters()).device

    def score(A):
        with torch.no_grad():
            lg = head(torch.tensor(prep(A), dtype=torch.float32, device=dev)).squeeze(-1)
        lg = lg.cpu().numpy()
        return cal.predict_proba(lg), cal.ranking(lg)

    return score, cal_name, thr, tr


def within_cv(sub, X, y, seed=SEED):
    """5-fold speaker-disjoint CV, the protocol the SSL within-mPower figure uses.

    Before 2026-11-09 the within-mPower figure scored the locked model on the cohort it
    was trained on, which is optimistic. Each fold now fits its own imputer, scaler,
    head, calibrator and threshold on the training folds; the threshold metrics apply
    each fold's own validation threshold to its own test participants.
    """
    dfa = assign_folds(sub.reset_index(drop=True), n_splits=5, seed=seed)
    parts = []
    for k in range(5):
        trv = np.flatnonzero((dfa.fold != k).to_numpy())
        te = np.flatnonzero((dfa.fold == k).to_numpy())
        score, _, thr, _ = fit_locked_head(X[trv], y[trv], dfa.subject_id.to_numpy()[trv], seed)
        p, r = score(X[te])
        parts.append(pd.DataFrame({"subject_id": dfa.subject_id.to_numpy()[te], "y": y[te],
                                   "p": p, "r": r, "thr": thr}))
    g = (pd.concat(parts).groupby("subject_id")
         .agg(y=("y", "first"), p=("p", "mean"), r=("r", "mean"), thr=("thr", "first")))
    # Each participant is decided at its own fold's validation threshold (p >= thr). This
    # used to shift p by the threshold and send the shifted values through calibration,
    # which failed with an infinite (all-negative) Youden threshold.
    m = evaluate(g.y.to_numpy(), g.p.to_numpy(), g.thr.to_numpy(), g.r.to_numpy())
    m["threshold"] = "per fold (validation Youden)"
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    t0 = time.time()

    print("=== Classical-feature transfer: mPower vowel -> IPVS / NeuroVoz vowel ===")

    # ---- 1. training cohort -------------------------------------------------
    mp = load_mpower_cohort(cap=args.cap, seed=SEED)
    mp = mp[mp.task_kind == "vowel"].reset_index(drop=True)
    print(f"[1] mPower: {len(mp):,} recordings / {mp.subject_id.nunique():,} speakers")
    Xmp = extract(mp, f"mpower__cap{args.cap}").to_numpy(dtype=np.float32)
    ymp = mp["label"].astype(int).to_numpy()

    # ---- 2. externals -------------------------------------------------------
    ext = {}
    for ds in VOWEL_EXTERNALS:
        df = LOADERS[ds]()
        df = df[df.task_kind == "vowel"].copy()
        df, _ = quality_control(df)
        df = df.reset_index(drop=True)
        print(f"[2] {ds}: {len(df):,} recordings / {df.subject_id.nunique():,} speakers")
        ext[ds] = (df, extract(df, ds).to_numpy(dtype=np.float32))

    results = {"cap": args.cap, "task": "vowel", "features": len(FEATURE_NAMES),
               "externals": list(VOWEL_EXTERNALS), "scenarios": {}}

    for scen in ("confounded", "age-matched"):
        mask = (np.ones(len(mp), bool) if scen == "confounded"
                else mp.subject_id.isin(age_match_ids(mp)).to_numpy())
        sub, X, y = mp[mask].reset_index(drop=True), Xmp[mask], ymp[mask]
        print(f"\n[3] {scen}: {len(sub):,} recordings / {sub.subject_id.nunique():,} speakers")

        # Locked model: imputation, scaling, head, calibrator and threshold are fitted on
        # mPower training/validation rows only and reused unchanged on every external
        # corpus. Fitting anything on an external set would stop it being a transfer test.
        score, cal_name, thr, tr = fit_locked_head(X, y, sub.subject_id.to_numpy())
        print(f"      head trained on {len(tr):,} recordings "
              f"({sub.subject_id.iloc[tr].nunique():,} speakers); "
              f"calibrator={cal_name}; locked threshold={thr:.3f}")

        within = within_cv(sub, X, y)
        print(f"      within mPower (5-fold CV) AUROC {within['auroc']:.3f} "
              f"{within['auroc_ci']}")

        block = {"calibrator": cal_name, "threshold": round(float(thr), 4),
                 "within_mpower": within, "transfer": {}}
        for ds, (df, A) in ext.items():
            sj = subject_level(df.subject_id, df.label.astype(int).to_numpy(),
                               score(A))
            m = evaluate(sj.y.to_numpy(), sj.p.to_numpy(), thr, sj.r.to_numpy())
            m["transfer_drop_auroc"] = round(within["auroc"] - m["auroc"], 4)
            block["transfer"][ds] = m
            flag = "  <-- INVERTED" if m["auroc_ci"][1] < 0.5 else ""
            print(f"      -> {ds:<10} AUROC {m['auroc']:.3f} {m['auroc_ci']}  "
                  f"drop {m['transfer_drop_auroc']:+.3f}{flag}")
        results["scenarios"][scen] = block

    out = LOGS / f"classical_transfer__cap{args.cap}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out.name}   ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
