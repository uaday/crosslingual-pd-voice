"""How much does each training condition change Experiment 1 transfer, per test set?

Experiment 1's 'mean transfer' averages three test sets (IPVS vowels, NeuroVoz vowels,
MDVR-KCL speech) and hides how unevenly they move. This script compares, for every test set
and on the same test participants, the model trained on an altered mPower cohort with the
model trained on the uncorrected cohort:

    agematched    the age-matched cohort
    sizematched   a random cohort of the same size and class balance, age gap kept: if
                  matching helps only because the cohort is smaller, this does as well
    confident     patients who gave a diagnosis year, controls who gave none

Paired participant bootstrap, as in task_comparison.py.
Output: outputs/logs/unified/exp1_changes.json
"""
from __future__ import annotations

import json

import pandas as pd

from pdvoice.config import LOGS
from task_comparison import paired

U = LOGS / "unified"
TARGETS = [("ipvs", "vowel"), ("neurovoz", "vowel"), ("mdvr_kcl", "speech"),
           ("bridge2ai", "vowel"), ("ipvs", "speech"), ("neurovoz", "speech"),
           ("neurovoz", "ddk"), ("bridge2ai", "speech"), ("bridge2ai", "ddk")]


def main() -> None:
    S = pd.read_csv(U / "scores.csv.gz")
    E = S[(S.exp == "exp1") & (S.target_cohort == "released")]
    out = {}
    for tgt, task in TARGETS:
        base = E[(E.target == tgt) & (E.target_task == task) & (E.condition == "uncorrected")]
        for cond in ("agematched", "sizematched", "confident"):
            alt = E[(E.target == tgt) & (E.target_task == task) & (E.condition == cond)]
            if base.empty or alt.empty:
                continue
            r = paired(alt, base)
            if r:
                out[f"{tgt} {task} | {cond} - uncorrected"] = r
                print(f"{tgt:9s} {task:6s} {cond:11s} {r['auroc_b']:.3f} -> {r['auroc_a']:.3f} "
                      f"diff {r['diff']:+.3f} {r['ci95']}", flush=True)
    (U / "exp1_changes.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
