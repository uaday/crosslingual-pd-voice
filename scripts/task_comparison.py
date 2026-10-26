"""Paired comparison of elicitation tasks with the training source held fixed.

Reads the participant-level scores of the unified experiments (outputs/logs/unified/
scores.csv.gz) and compares connected speech with the sustained vowel and with
diadochokinesis (DDK):

  within    the same dataset, the same participants: out-of-fold scores of the
            within-dataset models for each task (Bridge2AI, NeuroVoz, IPVS)
  transfer  the same source and target datasets: Experiment 2 (speech) against
            Experiment 3 (vowel, DDK), scored on the same target participants

For each comparison and seed only participants scored under both tasks are used, so the
difference is paired. The 95% interval resamples those participants (200 per seed, pooled
over the five seeds), as for every other interval in the thesis.
Output: outputs/logs/unified/task_comparison.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pdvoice.config import LOGS

U = LOGS / "unified"
PAIRS = [("speech", "vowel"), ("speech", "ddk"), ("ddk", "vowel")]


def paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 200) -> dict | None:
    """a, b: participant scores (s, y, r, seed) for two tasks. Difference AUROC(a)-AUROC(b)."""
    diffs, auc_a, auc_b, boots, ns = [], [], [], [], []
    rng = np.random.default_rng(0)
    for seed, ga in a.groupby("seed"):
        gb = b[b.seed == seed]
        m = ga.merge(gb, on="s", suffixes=("_a", "_b"))
        if m.empty or m.y_a.nunique() < 2:
            continue
        y, ra, rb = m.y_a.to_numpy(), m.r_a.to_numpy(), m.r_b.to_numpy()
        auc_a.append(roc_auc_score(y, ra)); auc_b.append(roc_auc_score(y, rb))
        diffs.append(auc_a[-1] - auc_b[-1]); ns.append((int(y.sum()), int((y == 0).sum())))
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                boots.append(roc_auc_score(y[i], ra[i]) - roc_auc_score(y[i], rb[i]))
    if not diffs:
        return None
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(auroc_a=round(float(np.mean(auc_a)), 3), auroc_b=round(float(np.mean(auc_b)), 3),
                diff=round(float(np.mean(diffs)), 3), ci95=[round(float(lo), 3), round(float(hi), 3)],
                a_better_in_seeds=int(sum(d > 0 for d in diffs)), n_seeds=len(diffs),
                n_pd_hc=list(ns[0]))


def main() -> None:
    S = pd.read_csv(U / "scores.csv.gz")
    out = {"within": {}, "transfer": {}}

    # ---- within: same dataset, same participants
    W = S[S.exp == "within"]
    for ds in ("bridge2ai", "neurovoz", "ipvs"):
        for cond in ("uncorrected", "agematched"):
            for ta, tb in PAIRS:
                a = W[(W.source == ds) & (W.source_task == ta) & (W.condition == cond)]
                b = W[(W.source == ds) & (W.source_task == tb) & (W.condition == cond)]
                if a.empty or b.empty:
                    continue
                r = paired(a, b)
                if r:
                    out["within"][f"{ds} | {cond} | {ta} - {tb}"] = r
                    print("within  ", ds, cond, f"{ta}-{tb}", r, flush=True)

    # ---- transfer: same source and target, same target participants
    T = S[S.exp.isin(["exp2", "exp3"])]
    for cond in ("uncorrected", "agematched"):
        for ta, tb in PAIRS:
            A = T[(T.source_task == ta) & (T.condition == cond)]
            B = T[(T.source_task == tb) & (T.condition == cond)]
            for (src, tgt), _ in A.groupby(["source", "target"]):
                for coh in ("released", "agematched"):
                    a = A[(A.source == src) & (A.target == tgt) & (A.target_cohort == coh)]
                    b = B[(B.source == src) & (B.target == tgt) & (B.target_cohort == coh)]
                    if a.empty or b.empty:
                        continue
                    r = paired(a, b)
                    if r:
                        out["transfer"][f"{src}->{tgt} | train {cond} | test {coh} | {ta} - {tb}"] = r
                        print("transfer", src, tgt, cond, coh, f"{ta}-{tb}", r, flush=True)

    (U / "task_comparison.json").write_text(json.dumps(out, indent=2))
    print("wrote", U / "task_comparison.json")


if __name__ == "__main__":
    main()
