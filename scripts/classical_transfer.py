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
from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
from pdvoice.eval.metrics import calibration_error, sensitivity_at_specificity, \
    threshold_metrics, youden_threshold
from pdvoice.features.classical import FEATURE_NAMES, classical_features
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

VOWEL_EXTERNALS = ("ipvs", "neurovoz")


def extract(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """42 Praat descriptors per recording, cached and identity-guarded.

    The cache is keyed on the recording PATHS, not just the row count. A stale cache with
    the right length but the wrong rows would pair every recording with someone else's
    features and return chance performance without raising.
    """
    from tqdm import tqdm

    cache = EMBEDDINGS / f"classical__{tag}.csv"
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


def subject_level(ids, y, p) -> pd.DataFrame:
    return (pd.DataFrame({"subject_id": ids, "y": y, "p": p})
            .groupby("subject_id").agg(y=("y", "first"), p=("p", "mean")).reset_index())


def evaluate(y, p, thr) -> dict:
    lo, hi = bootstrap_ci(y, p)[1:]
    m = {"n_subjects": int(len(y)), "prevalence": round(float(np.mean(y)), 4),
         "auroc": round(float(auroc(y, p)), 4), "auroc_ci": [round(lo, 4), round(hi, 4)],
         "auprc": round(float(auprc(y, p)), 4)}
    m.update({k: round(float(v), 4) for k, v in calibration_error(y, p).items()})
    m.update({k: (round(float(v), 4) if isinstance(v, float) else v)
              for k, v in threshold_metrics(y, p, thr).items()})
    m["sens_at_90spec"] = round(float(sensitivity_at_specificity(y, p)["sensitivity"]), 4)
    # NPV is not emitted by threshold_metrics but is recoverable from the matrix.
    tn, fn = m["tn"], m["fn"]
    m["npv"] = round(tn / (tn + fn), 4) if (tn + fn) else 0.0
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

        # Impute and standardise using TRAINING statistics only, then reuse them on every
        # external corpus. Fitting the scaler on an external set would leak its
        # distribution into the locked model and stop it being a transfer test.
        med = np.nanmedian(X, axis=0)
        def prep(A):
            A = np.where(np.isnan(A), med, A)
            return (A - mu) / sd
        Xi = np.where(np.isnan(X), med, X)
        mu, sd = Xi.mean(0), Xi.std(0) + 1e-8
        Xs = prep(X)

        tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=SEED)
                      .split(Xs, y, sub.subject_id))
        head, vlog = fit_mlp_head(Xs[tr], y[tr], Xs[va], y[va], seed=SEED)
        cal_name, cal = select_best_by_brier(vlog, y[va])
        thr = youden_threshold(y[va], cal.predict_proba(vlog))
        print(f"      head trained on {len(tr):,} recordings "
              f"({sub.subject_id.iloc[tr].nunique():,} speakers); "
              f"calibrator={cal_name}; locked threshold={thr:.3f}")

        dev = next(head.parameters()).device

        def score(A):
            with torch.no_grad():
                lg = head(torch.tensor(A, dtype=torch.float32, device=dev)).squeeze(-1)
            return cal.predict_proba(lg.cpu().numpy())

        s = subject_level(sub.subject_id, y, score(Xs))
        within = evaluate(s.y.to_numpy(), s.p.to_numpy(), thr)
        print(f"      within mPower      AUROC {within['auroc']:.3f} "
              f"{within['auroc_ci']}")

        block = {"calibrator": cal_name, "threshold": round(float(thr), 4),
                 "within_mpower": within, "transfer": {}}
        for ds, (df, A) in ext.items():
            sj = subject_level(df.subject_id, df.label.astype(int).to_numpy(),
                               score(prep(A)))
            m = evaluate(sj.y.to_numpy(), sj.p.to_numpy(), thr)
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
