"""Bridge2AI-Voice: PD vs healthy controls, before and after age matching.

Bridge2AI's patients are ~28 years older than its controls (72.9 vs 45.1), the same
confound as mPower and the one Shukla et al. (JMIR 2026) report on this dataset. This asks
whether the voice signal survives once age carries no information, for each task family.

Pipeline (the same linear probe as restricted_permutation.py, so numbers are comparable):
XLS-R layer-15 mean+std embeddings, PCA inside each training fold, logistic regression,
5-fold participant-disjoint CV, predictions averaged per participant.

Two controls for age:
  1. frequency matching in 5-year bins (pdvoice.cohort.age_match_ids), over five seeds,
     because each seed draws a different matched subset of the larger group
  2. a restricted permutation test on the FULL cohort: labels shuffled only within 5-year
     age bins, so the null keeps the age-diagnosis link. Nobody is discarded.

Run:  .venv\\Scripts\\python.exe -u scripts\\bridge2ai_agematch.py --n-perm 200
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdvoice.cohort import age_match_ids
from pdvoice.config import EMBEDDINGS, LOGS
from pdvoice.data import LOADERS
from pdvoice.data.common_tasks import common_task_mask
from pdvoice.eval.discrimination import bootstrap_ci

MODEL = "xls-r-300m"
LAYER = int(os.environ.get("PDV_LAYER", 15))  # Bridge2AI layer; 8 is chosen inside Bridge2AI (bridge2ai_layer_select.py)
SUFFIX = "" if LAYER == 15 else f"__L{LAYER}"
TASKS = ["vowel", "ddk", "speech"]


def load_task(base, task):
    d = np.load(EMBEDDINGS / f"bridge2ai__{MODEL}__{task}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))
    df = emb.merge(base, on="path", how="inner")
    assert df.row.is_unique
    df = df[df.group.isin(["pd", "control"])].dropna(subset=["age"])
    df = df[common_task_mask(df, "bridge2ai")].reset_index(drop=True)
    df["label"] = df.label.astype(int)
    return df, d["X"][df.row.to_numpy()].astype(np.float32)


def subject_scores(df, X, y, seed):
    """Out-of-fold participant-level probabilities."""
    g = df.subject_id.to_numpy()
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        k = min(64, len(np.unique(g[tr])) - 1)
        clf = make_pipeline(StandardScaler(), PCA(k, random_state=seed),
                            LogisticRegression(max_iter=2000))
        oof[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    s = pd.DataFrame({"s": g, "y": y, "p": oof, "age": df.age.to_numpy()})
    return s.groupby("s").agg(y=("y", "first"), p=("p", "mean"), age=("age", "first"))


def permute_within(y_subj, bins, rng):
    out = y_subj.copy()
    for b in np.unique(bins):
        i = np.flatnonzero(bins == b)
        out[i] = rng.permutation(y_subj[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    seeds = [int(s) for s in args.seeds.split(",")]
    base = LOADERS["bridge2ai"]()

    report = {}
    for task in TASKS:
        df, X = load_task(base, task)
        y = df.label.to_numpy()
        subj = df.drop_duplicates("subject_id").set_index("subject_id")
        print(f"\n=== {task} ===", flush=True)

        # ---- uncorrected cohort
        s = subject_scores(df, X, y, 42)
        a, lo, hi = bootstrap_ci(s.y.to_numpy(), s.p.to_numpy())
        res = {"uncorrected": dict(
            n_pd=int(s.y.sum()), n_hc=int((s.y == 0).sum()),
            age_pd=round(float(s.age[s.y == 1].mean()), 1),
            age_hc=round(float(s.age[s.y == 0].mean()), 1),
            age_only=round(float(roc_auc_score(s.y, s.age)), 3),
            voice=round(float(a), 3), ci=[round(float(lo), 3), round(float(hi), 3)])}
        u = res["uncorrected"]
        print(f"  uncorrected  {u['n_pd']} PD / {u['n_hc']} HC, ages {u['age_pd']} vs {u['age_hc']}"
              f" | age alone {u['age_only']:.3f} | voice {u['voice']:.3f} [{lo:.3f}, {hi:.3f}]",
              flush=True)

        # ---- age-matched, five seeds
        rows = []
        for seed in seeds:
            keep = age_match_ids(df, seed=seed)
            m = df.subject_id.isin(keep).to_numpy()
            sm = subject_scores(df[m].reset_index(drop=True), X[m], y[m], seed)
            a, lo, hi = bootstrap_ci(sm.y.to_numpy(), sm.p.to_numpy())
            rows.append(dict(seed=seed, n_per_group=int(sm.y.sum()),
                             age_gap=float(sm.age[sm.y == 1].mean() - sm.age[sm.y == 0].mean()),
                             age_only=float(roc_auc_score(sm.y, sm.age)),
                             voice=float(a), lo=float(lo), hi=float(hi)))
        R = pd.DataFrame(rows)
        res["age_matched"] = dict(
            n_per_group=int(R.n_per_group.iloc[0]), age_gap=round(R.age_gap.mean(), 2),
            age_only=round(R.age_only.mean(), 3),
            voice_mean=round(R.voice.mean(), 3), voice_sd=round(R.voice.std(), 3),
            ci_seed42=[round(R.lo.iloc[0], 3), round(R.hi.iloc[0], 3)])
        am = res["age_matched"]
        print(f"  age-matched  {am['n_per_group']} PD / {am['n_per_group']} HC, gap {am['age_gap']:+.2f} yr"
              f" | age alone {am['age_only']:.3f} | voice {am['voice_mean']:.3f} +/- {am['voice_sd']:.3f}"
              f" (seed-42 CI {am['ci_seed42']})", flush=True)

        # ---- restricted permutation on the full cohort (labels permuted per participant)
        bins = (subj.age // 5 * 5).astype(int)
        y_subj = subj.label.to_numpy()
        sid = subj.index.to_numpy()
        rng = np.random.default_rng(42)
        null = []
        for i in range(args.n_perm):
            yp = dict(zip(sid, permute_within(y_subj, bins.to_numpy(), rng)))
            y_rec = df.subject_id.map(yp).to_numpy().astype(int)
            sp = subject_scores(df, X, y_rec, 42 + i + 1)
            null.append(roc_auc_score(sp.y, sp.p))
        null = np.asarray(null)
        p = (np.sum(null >= u["voice"]) + 1) / (len(null) + 1)
        res["restricted_permutation"] = dict(
            n_perm=args.n_perm, null_mean=round(float(null.mean()), 3),
            null_sd=round(float(null.std()), 3), excess=round(float(u["voice"] - null.mean()), 3),
            p_value=round(float(p), 4))
        rp = res["restricted_permutation"]
        print(f"  restricted permutation: null {rp['null_mean']:.3f} +/- {rp['null_sd']:.3f}"
              f" -> observed {u['voice']:.3f}, excess {rp['excess']:+.3f}, p = {rp['p_value']}",
              flush=True)
        report[task] = res

    (LOGS / f"bridge2ai_agematch{SUFFIX}.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {LOGS / 'bridge2ai_agematch.json'}")


if __name__ == "__main__":
    main()
