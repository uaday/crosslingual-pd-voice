"""Reproduce the two Bridge2AI shortcut corrections quoted in the thesis.

Every Canadian participant in the PD-vs-control cohort is a patient (there are no
controls), and patients and controls recorded different cohort-specific speech tasks. This
script repeats the within-dataset analysis of unified_experiments.py (same head,
calibration, five seeds, 5-fold participant-disjoint CV) under three cohort definitions:

    all_countries_all_tasks   all patients and the controls, every speech task
    us_all_tasks              US participants only, every speech task
    us_common_tasks           US participants only, speech tasks both groups recorded
                              (the definition used everywhere else in the thesis)

and reports the age-only and country-only baselines. Controls are the loader's healthy
controls (no confirmed diagnosis). Output: outputs/logs/unified/bridge2ai_country_check.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_experiments as U  # noqa: E402
from pdvoice.cohort import age_match_ids  # noqa: E402
from pdvoice.config import EMBEDDINGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402
from pdvoice.data.common_tasks import common_task_mask  # noqa: E402
from pdvoice.preprocess import quality_control  # noqa: E402


def load_raw(task: str):
    """Bridge2AI recordings of patients (either country) and healthy controls, no task filter."""
    base = LOADERS["bridge2ai"]()
    df = base[base.task_kind == task].copy()
    df, _ = quality_control(df)
    df = df[df.group.isin(["pd", "pd_non_us", "control"])].reset_index(drop=True)
    df["label"] = df.group.isin(["pd", "pd_non_us"]).astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    d = np.load(EMBEDDINGS / f"bridge2ai__{U.MODEL}__{task}__Lall.npz", allow_pickle=True)
    row = {p: i for i, p in enumerate(d["path"].astype(str))}
    X = d["X"][[row[p] for p in df.path], U.LAYER].astype(np.float32)
    return df, X


def boot_ci(parts, n=200):
    rng = np.random.default_rng(0)
    vals = []
    for g in parts:
        y, r = g.y.to_numpy(), g.r.to_numpy()
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                vals.append(roc_auc_score(y[i], r[i]))
    return [round(float(v), 3) for v in np.percentile(vals, [2.5, 97.5])]


def evaluate(df, X):
    res = {}
    for cond in ("uncorrected", "agematched"):
        aucs, age_aucs, parts = [], [], []
        for seed in U.SEEDS:
            keep = (df.subject_id.isin(age_match_ids(df, seed=seed)).to_numpy()
                    if cond == "agematched" else np.ones(len(df), bool))
            d, x = df[keep].reset_index(drop=True), X[keep]
            g = U.within_cv(d, x, seed)
            parts.append(g)
            aucs.append(roc_auc_score(g.y, g.r))
            s = d.drop_duplicates("subject_id")
            age_aucs.append(roc_auc_score(s.label, s.age))
        s = df[keep].drop_duplicates("subject_id")
        res[cond] = dict(auroc_mean=round(float(np.mean(aucs)), 3),
                         auroc_sd=round(float(np.std(aucs, ddof=1)), 3), ci95=boot_ci(parts),
                         age_only=round(float(np.mean(age_aucs)), 3),
                         n_pd_hc=[int(s.label.sum()), int((s.label == 0).sum())])
    return res


def main() -> None:
    warnings.simplefilter("ignore")
    out = {}
    for task in ("vowel", "ddk", "speech"):
        df, X = load_raw(task)
        subj = df.drop_duplicates("subject_id")
        out[task] = {
            "country_by_label": pd.crosstab(subj.country.fillna("NA"), subj.label).to_dict(),
            "country_only_auroc": round(float(roc_auc_score(subj.label, subj.country != "USA")), 3)}
        us = (df.country == "USA").to_numpy()
        common = common_task_mask(df, "bridge2ai")
        for name, m in (("all_countries_all_tasks", np.ones(len(df), bool)),
                        ("us_all_tasks", us), ("us_common_tasks", us & common)):
            if task != "speech" and name == "us_common_tasks":
                continue                        # the task filter only changes speech
            r = evaluate(df[m].reset_index(drop=True), X[m])
            out[task][name] = r
            print(f"{task:6s} {name:24s} " + "  ".join(
                f"{c}: {v['auroc_mean']:.3f} {v['ci95']} n={v['n_pd_hc']} age={v['age_only']:.3f}"
                for c, v in r.items()), flush=True)
    p = U.OUT / "bridge2ai_country_check.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
