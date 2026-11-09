"""Fusion baseline — SSL embeddings + classical features (RQ1/RQ2, "best of both").

Motivation: on the honest age-matched vowel signal, cheap interpretable features are
statistically competitive with a 300M-parameter transformer (classical_baseline.py).
Does combining them beat either alone — SSL's reach plus the interpretable signal?

All three feature sets go through the SAME MLP head and the same subject-disjoint 5-fold
CV, so the only thing that changes is the features.

WHY THIS SCRIPT WAS REBUILT (2026-09-07)
----------------------------------------
It used to construct its own cap-1 cohort, extract its own embeddings into a private
cache, and align to the classical CSV BY ROW POSITION. That made it the last script with
a bespoke data path, and it disagreed with every other measurement of the same quantity:
SSL-only on the confounded cohort came out at 0.671 here, against 0.724–0.796 from
classifier_comparison.py and 0.79–0.81 from a direct check on the shared cache.

Rather than reconcile two code paths there is now one: `load_mpower_cohort`, the shared
`mpower_full` cache, and a join on **subject_id** rather than on row order. Positional
joins between independently-built frames are exactly the failure this project has been
bitten by twice already, and they never raise -- they just return a plausible wrong number.

Run:  python scripts/fusion.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from pdvoice.cohort import age_match_ids, load_mpower_cohort
from pdvoice.config import EMBEDDINGS, LOGS, SEED
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import (auprc, auroc, bootstrap_ci,
                                         delong_roc_test, mcnemar_test,
                                         paired_bootstrap_auroc)
from pdvoice.eval.metrics import youden_threshold
from pdvoice.features.classical import FEATURE_NAMES
from pdvoice.models.head import fit_mlp_head

MODEL, LAYER, CAP = "xls-r-300m", 11, 1
FEATS = FEATURE_NAMES


def build_cohort():
    """Shared cohort + shared cache; classical features joined on speaker, not position."""
    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))

    df = (load_mpower_cohort(cap=CAP, seed=SEED)
          .merge(emb, on="path", how="inner")
          .reset_index(drop=True))

    cls = pd.read_csv(LOGS / f"classical_features_labeled__cap{CAP}.csv")
    missing = [c for c in ["path", *FEATS] if c not in cls.columns]
    if missing:
        raise SystemExit(
            f"classical CSV is missing {missing[:3]} -- it predates the path-keyed, "
            "full-file-window extraction.\nRe-run: scripts/classical_baseline.py --cap 1")

    # join on the RECORDING, not the speaker: both arms must describe the same audio
    df = df.merge(cls[["path", *FEATS]], on="path", how="inner").reset_index(drop=True)
    assert df.subject_id.is_unique, "cap-1 cohort should hold one recording per speaker"
    X = d["X"][df.row.to_numpy()].astype(np.float32)
    assert len(X) == len(df)
    return df, X


def fit_prep(X, cols):
    """Median-impute and z-score the columns `cols` with statistics from X (training rows
    only); other columns (the SSL embedding) pass through unchanged."""
    if not len(cols):
        return lambda A: A
    C = X[:, cols].astype(float)
    med = np.nanmedian(C, axis=0)
    Ci = np.where(np.isnan(C), med, C)
    mu, sd = Ci.mean(0), Ci.std(0) + 1e-8

    def prep(A):
        A = A.astype(np.float32, copy=True)
        B = A[:, cols].astype(float)
        A[:, cols] = (np.where(np.isnan(B), med, B) - mu) / sd
        return A
    return prep


def within_cv(df, X, y, cols):
    """Subject-disjoint 5-fold CV. Imputation and scaling are fitted on each training fold
    alone (they used to be fitted on the whole cohort before the split, which let test
    participants shape the training inputs). Each fold also records its operating
    threshold, chosen by Youden's J on that fold's VALIDATION predictions, for McNemar."""
    df = assign_folds(df.reset_index(drop=True), n_splits=5)
    oof = []
    for k in range(5):
        tr, va, te = train_val_test_indices(df, test_fold=k)
        prep = fit_prep(X[tr], cols)
        m, _ = fit_mlp_head(prep(X[tr]), y[tr], prep(X[va]), y[va])
        dev = next(m.parameters()).device

        def prob(A):
            with torch.no_grad():
                lg = m(torch.tensor(prep(A), dtype=torch.float32, device=dev)).squeeze(-1)
            return torch.sigmoid(lg).cpu().numpy()
        thr = youden_threshold(y[va], prob(X[va]))
        oof.append(pd.DataFrame({"s": df.loc[te, "subject_id"].to_numpy(), "y": y[te],
                                 "p": prob(X[te]), "thr": thr}))
    g = (pd.concat(oof).groupby("s")
         .agg(y=("y", "first"), p=("p", "mean"), thr=("thr", "first")).reset_index())
    yy, pp = g["y"].to_numpy(), g["p"].to_numpy()
    a, lo, hi = bootstrap_ci(yy, pp, metric=auroc)
    return a, lo, hi, auprc(yy, pp), g


def pairwise_tests(preds):
    """Paired significance tests between every pair of feature sets.

    JOIN ON subject_id, never on row order. The three frames come from separate
    groupby operations and are only guaranteed to contain the same subjects, not to
    list them in the same order -- and a silent misalignment here would produce a
    confident p-value for a comparison that was never made.
    """
    out = {}
    names = list(preds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = preds[a].merge(preds[b], on="s", suffixes=("_a", "_b"))
            assert len(m) == len(preds[a]) == len(preds[b]), \
                f"{a} vs {b}: subject sets differ ({len(m)} matched)"
            assert (m.y_a == m.y_b).all(), f"{a} vs {b}: labels disagree after the join"
            y = m.y_a.to_numpy()
            pa, pb = m.p_a.to_numpy(), m.p_b.to_numpy()
            _, _, p_delong = delong_roc_test(y, pa, pb)
            delta, p_boot = paired_bootstrap_auroc(y, pa, pb)

            # DeLong asks whether one model RANKS better; McNemar asks whether the
            # decisions it actually makes are better. Those come apart -- a model can
            # rank better and decide no better -- so both are reported. Each model is
            # binarised at ITS OWN operating threshold, chosen by Youden's J on each
            # fold's validation split (never on the predictions being tested): comparing
            # decisions at a shared arbitrary cut such as 0.5 would largely measure which
            # model happens to be better centred.
            da = (pa >= m.thr_a.to_numpy()).astype(int)
            db = (pb >= m.thr_b.to_numpy()).astype(int)
            mc = mcnemar_test(y, da, db)

            out[f"{a} vs {b}"] = {"delta": round(float(delta), 4),
                                  "p_delong": round(float(p_delong), 4),
                                  "p_bootstrap": round(float(p_boot), 4),
                                  "p_mcnemar": round(float(mc["p_value"]), 4),
                                  "mcnemar_method": mc["method"],
                                  "mcnemar_n01": mc["n01"],
                                  "mcnemar_n10": mc["n10"],
                                  "n_subjects": int(len(m))}
    return out


def main():
    print(f"=== Fusion: SSL + classical (mPower vowel, cap-1, XLS-R L{LAYER}) ===")
    df, S = build_cohort()

    # classical features stay raw here; within_cv imputes and scales them per fold
    C = df[FEATS].to_numpy(dtype=np.float32)
    F = np.hstack([S, C]).astype(np.float32)
    y = df["label"].astype(int).to_numpy()
    print(f"  cohort {len(df):,} · SSL {S.shape[1]}d · classical {C.shape[1]}d · "
          f"fusion {F.shape[1]}d\n")

    sets = {"SSL-only": S, "classical-only": C, "FUSION": F}
    # which columns of each set are classical (imputed + scaled inside each fold)
    ncls = C.shape[1]
    cols = {"SSL-only": np.array([], int), "classical-only": np.arange(ncls),
            "FUSION": np.arange(S.shape[1], S.shape[1] + ncls)}
    print(f"  {'scenario':<13}{'features':<16}{'AUROC':<22}{'AUPRC'}")
    results = {"cohort": len(df), "dims": {"ssl": S.shape[1], "classical": C.shape[1]},
               "scenarios": {}}
    scenarios = {"baseline": np.ones(len(df), bool),
                 "age-matched": df.subject_id.isin(age_match_ids(df)).to_numpy()}
    for scen, mask in scenarios.items():
        results["scenarios"][scen] = {}
        preds = {}
        for name, X in sets.items():
            a, lo, hi, ap, g = within_cv(df[mask].reset_index(drop=True), X[mask], y[mask],
                                       cols[name])
            results["scenarios"][scen][name] = {"auroc": a, "auroc_ci": [lo, hi], "auprc": ap}
            g.to_csv(LOGS / f"fusion_preds__{scen}__{name.replace('-', '_')}.csv", index=False)
            preds[name] = g
            star = "  <<<" if name == "FUSION" else ""
            print(f"  {scen:<13}{name:<16}{a:.3f} [{lo:.3f},{hi:.3f}]      {ap:.3f}{star}")

        results["scenarios"][scen]["_tests"] = pairwise_tests(preds)
        print(f"  {'':<13}{'pair':<34}{'delta':>8}{'DeLong p':>11}{'boot p':>9}")
        for key, t in results["scenarios"][scen]["_tests"].items():
            verdict = "" if t["p_delong"] < 0.05 else "   not distinguishable"
            print(f"  {'':<13}{key:<34}{t['delta']:+8.4f}{t['p_delong']:>11.3f}"
                  f"{t['p_bootstrap']:>9.3f}{verdict}")
        print()

    (LOGS / f"fusion__xls-r-300m__L{LAYER}.json").write_text(json.dumps(results, indent=2))
    print("wrote fusion json")


if __name__ == "__main__":
    main()
