"""SCALE-UP of the honest age-matched model -- full mPower, multiple seeds, CIs.

`agematched_eval.py` is the pilot: cap-3, one seed, a cohort cache keyed by (cap, seed).
This is the reportable version. It loads the one-off full extraction
(`extract_mpower_full.py`) and then, for each seed, redraws EVERYTHING that a seed
should change:

    cap sampling  ->  age matching  ->  train/val/test folds  ->  head init

so the spread across seeds is an honest estimate of run-to-run variability, not just
weight-init noise. Reports mean +/- SD across seeds alongside per-seed bootstrap CIs.

Also asserts on head convergence: if training stops at `max_epochs` rather than on the
patience criterion, the head is underfitted at this data scale and every number below
is pessimistic (see models/head.py).

Run:
    .venv\\Scripts\\python.exe -u scripts\\scaleup_agematched.py --cap 10 --seeds 42,43,44
    .venv\\Scripts\\python.exe -u scripts\\scaleup_agematched.py --cap 0   --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch

from pdvoice.cohort import age_match_ids, cap_per_subject, subject_age_stats
from pdvoice.calibration import select_best_by_brier
from pdvoice.config import MODELS, EMBEDDINGS, LOGS, MPOWER_DIR, VAL_FRACTION, ensure_dirs
from pdvoice.data import LOADERS
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
from pdvoice.eval.metrics import (calibration_error, sensitivity_at_specificity,
                                  threshold_metrics, youden_threshold)
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control
from sklearn.model_selection import GroupShuffleSplit

MODEL, LAYER = "xls-r-300m", 15
EXTERNALS = [("neurovoz", "vowel"), ("ipvs", "vowel"), ("mdvr_kcl", "speech")]


def load_full():
    """The one-off full extraction: X plus the ids needed to verify alignment."""
    f = EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f}\nRun scripts/extract_mpower_full.py first.")
    d = np.load(f, allow_pickle=True)
    df = pd.DataFrame({"subject_id": d["subject_id"].astype(str),
                       "path": d["path"].astype(str),
                       "label": d["label"].astype(int)})
    demo = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    age = dict(zip(demo["healthCode"].astype(str), demo["age"]))
    df["age"] = df.subject_id.str.replace("mpower_", "", regex=False).map(age)
    return df, d["X"].astype(np.float32)


def block(y, p, thr, r):
    # discrimination on the uncalibrated score r; calibration/threshold on p
    pt, lo, hi = bootstrap_ci(y, r)
    return {"n": int(len(y)), "n_pd": int(y.sum()), "auroc": round(pt, 4),
            "auroc_ci": [round(lo, 4), round(hi, 4)], "auprc": round(auprc(y, r), 4),
            "ece": round(calibration_error(y, p)["ece"], 4),
            "mce": round(calibration_error(y, p)["mce"], 4),
            "sens_at_90spec": round(sensitivity_at_specificity(y, r, 0.90)["sensitivity"], 4),
            **{k: round(v, 4) for k, v in threshold_metrics(y, p, thr).items()
               if k in ("sensitivity", "specificity", "balanced_accuracy", "f1")}}


def run_seed(df, X, cap, seed, conv, save_model=False):
    capped = cap_per_subject(df, cap, seed)
    keep = age_match_ids(capped, seed)
    am = capped[capped.subject_id.isin(keep)]
    idx = am.index.to_numpy()
    # X rows line up with `df` rows, so `idx` must still index into `df`. If anything in
    # the cohort chain resets the index, this pairs recordings with other people's
    # embeddings -- silently, and the only symptom is chance performance.
    assert np.array_equal(df.label.to_numpy()[idx], am.label.to_numpy()), \
        "cohort index no longer aligns with the embedding matrix"
    Xa, y = X[idx], am.label.to_numpy()
    am = am.reset_index(drop=True)

    # ---- within-dataset 5-fold subject-disjoint CV ----
    dfa = assign_folds(am, n_splits=5, seed=seed)
    oof = []
    for k in range(5):
        tr, va, te = train_val_test_indices(dfa, test_fold=k, seed=seed)
        m, vlog = fit_mlp_head(Xa[tr], y[tr], Xa[va], y[va], seed=seed)
        conv.append({"seed": seed, "phase": f"cv{k}", "epochs": m._epochs_run,
                     "early_stopped": m._stopped_early})
        dev = next(m.parameters()).device
        with torch.no_grad():
            tl = m(torch.tensor(Xa[te], dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        _, cal = select_best_by_brier(vlog, y[va])
        # operating threshold from THIS fold's validation split, never from the test
        # predictions it is then used on
        thr_k = youden_threshold(y[va], cal.predict_proba(vlog))
        oof.append(pd.DataFrame({"s": dfa.loc[te, "subject_id"].to_numpy(),
                                 "y": y[te], "p": cal.predict_proba(tl), "r": cal.ranking(tl),
                                 "thr": thr_k}))
    sg = pd.concat(oof).groupby("s").agg(y=("y", "first"), p=("p", "mean"), r=("r", "mean"),
                                         thr=("thr", "first"))
    # Each participant is in one test fold, so p >= thr_fold  <=>  p - thr_fold + 0.5 >= 0.5.
    # (This used to pick one Youden threshold on the pooled out-of-fold predictions and
    # then score those same predictions with it, which flatters sensitivity/specificity.)
    yy, pp, rr = sg.y.to_numpy(), sg.p.to_numpy(), sg.r.to_numpy()
    within = block(yy, pp, 0.5, rr)
    within.update({k: v for k, v in block(yy, pp - sg.thr.to_numpy() + 0.5, 0.5, rr).items()
                   if k in ("sensitivity", "specificity", "balanced_accuracy", "f1")})

    # ---- locked model -> externals ----
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed)
                  .split(Xa, y, am.subject_id))
    head, vlog = fit_mlp_head(Xa[tr], y[tr], Xa[va], y[va], seed=seed)
    conv.append({"seed": seed, "phase": "locked", "epochs": head._epochs_run,
                 "early_stopped": head._stopped_early})
    cal_name, cal = select_best_by_brier(vlog, y[va])
    thr = youden_threshold(y[va], cal.predict_proba(vlog))
    dev = next(head.parameters()).device

    # Save the locked model from the FIRST seed as the project's reference artefact.
    # (Was agematched_eval.py's job; that script is otherwise superseded by this one.)
    if save_model:
        import joblib

        stem = f"agematched__{MODEL}__L{LAYER}"
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), MODELS / f"{stem}__head.pt")
        joblib.dump({"calibrator": cal, "threshold": float(thr), "layer": LAYER,
                     "model": MODEL, "in_dim": int(Xa.shape[1]), "seed": seed,
                     "n_subjects": int(am.subject_id.nunique())},
                    MODELS / f"{stem}__calibrator.joblib")
        print(f"  saved locked model -> {MODELS / (stem + '__head.pt')}", flush=True)

    transfer = {}
    for ds, task in EXTERNALS:
        edf = LOADERS[ds]()
        edf = edf[edf.task_kind == task].copy()
        edf, _ = quality_control(edf); edf = edf.reset_index(drop=True)
        EX = np.load(EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{LAYER}.npz")["X"].astype(np.float32)
        assert len(EX) == len(edf), f"{ds}: {len(EX)} != {len(edf)}"
        with torch.no_grad():
            lg = head(torch.tensor(EX, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        s = pd.DataFrame({"s": edf.subject_id, "y": edf.label.astype(int),
                          "p": cal.predict_proba(lg), "r": cal.ranking(lg)})
        g = s.groupby("s").agg(y=("y", "first"), p=("p", "mean"), r=("r", "mean"))
        transfer[ds] = block(g.y.to_numpy(), g.p.to_numpy(), thr, g.r.to_numpy())

    return {"seed": seed, "n_subjects": int(am.subject_id.nunique()),
            "n_recordings": int(len(am)),
            **{k: subject_age_stats(am)[k]        # per speaker, not per recording
               for k in ("age_pd", "age_hc", "age_gap", "age_level")},
            "calibrator": cal_name, "threshold": round(float(thr), 4),
            "within": within, "transfer": transfer}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=10, help="recordings per subject; 0 = all")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    ensure_dirs()
    seeds = [int(s) for s in args.seeds.split(",")]

    df, X = load_full()
    print(f"=== SCALE-UP | full mPower | cap={args.cap or 'ALL'} | seeds={seeds} ===")
    print(f"  pool: {len(df):,} recordings, {df.subject_id.nunique():,} subjects "
          f"(PD {int((df.drop_duplicates('subject_id').label==1).sum()):,} / "
          f"HC {int((df.drop_duplicates('subject_id').label==0).sum()):,})")
    print(f"  embeddings: {X.shape}\n")

    conv, runs = [], []
    for s in seeds:
        print(f"[seed {s}] ...", flush=True)
        r = run_seed(df, X, args.cap, s, conv, save_model=(s == seeds[0]))
        runs.append(r)
        w = r["within"]
        print(f"  cohort {r['n_subjects']:,} subj ({r['n_recordings']:,} recs), "
              f"age PD {r['age_pd']} / HC {r['age_hc']}")
        print(f"  within {w['auroc']:.3f} {w['auroc_ci']}  | " +
              "  ".join(f"{d} {t['auroc']:.3f}" for d, t in r["transfer"].items()), flush=True)

    # ---------------- aggregate ----------------
    def agg(getter):
        v = np.array([getter(r) for r in runs], float)
        return f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"

    print(f"\n{'='*74}\nACROSS {len(seeds)} SEEDS  (mean +/- SD)\n{'='*74}")
    rows = [dict(target="within mPower", AUROC=agg(lambda r: r["within"]["auroc"]),
                 AUPRC=agg(lambda r: r["within"]["auprc"]),
                 ECE=agg(lambda r: r["within"]["ece"]),
                 sens_at_90spec=agg(lambda r: r["within"]["sens_at_90spec"]))]
    for d, _ in EXTERNALS:
        rows.append(dict(target=f"-> {d}", AUROC=agg(lambda r, d=d: r["transfer"][d]["auroc"]),
                         AUPRC=agg(lambda r, d=d: r["transfer"][d]["auprc"]),
                         ECE=agg(lambda r, d=d: r["transfer"][d]["ece"]),
                         sens_at_90spec=agg(lambda r, d=d: r["transfer"][d]["sens_at_90spec"])))
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))

    cdf = pd.DataFrame(conv)
    hit_cap = int((~cdf.early_stopped).sum())
    print(f"\nHEAD CONVERGENCE: {len(cdf)} fits | median {int(cdf.epochs.median())} epochs | "
          f"{hit_cap} hit the epoch cap")
    if hit_cap:
        print("  ! some fits stopped at max_epochs with val loss still improving -> the head is")
        print("    UNDERFITTED at this scale and these numbers are pessimistic. Raise max_epochs.")
    else:
        print("  OK every fit stopped on the patience criterion -- converged at this scale.")

    stem = f"scaleup_agematched__{MODEL}__L{LAYER}__cap{args.cap}"
    (LOGS / f"{stem}.json").write_text(json.dumps(
        {"cap": args.cap, "seeds": seeds, "runs": runs,
         "convergence": conv, "summary": rows}, indent=2))
    summary.to_csv(LOGS / f"{stem}__summary.csv", index=False)
    print(f"\nwrote {LOGS / (stem + '.json')}")


if __name__ == "__main__":
    main()
