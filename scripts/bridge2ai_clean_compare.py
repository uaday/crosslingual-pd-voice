"""Before/after comparison of every Bridge2AI result when background noise is removed (spectral gating).

Reads outputs/logs/unified/runs.csv (original audio) and outputs/logs/unified_clean/runs.csv
(noise removed, silent recordings dropped; PDVOICE_B2AI_CLEAN=1 scripts/unified_experiments.py)
and reports, for every row that trains or tests on Bridge2AI, the five-seed mean AUROC
before and after, their difference, and a paired sign count over seeds.

With argument `all`, compares against outputs/logs/unified_clean_all (PDVOICE_CLEAN_ALL=1: IPVS,
NeuroVoz and MDVR-KCL cleaned too) and reports every row that touches a cleaned dataset.

Output: outputs/logs/bridge2ai_clean[_all]_compare.csv and a printed table.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from pdvoice.config import LOGS

# argv[1] = "all" compares against outputs/logs/unified_clean_all (every clinical set cleaned)
TAG = "_all" if sys.argv[1:] == ["all"] else ""

KEY = ["exp", "source", "source_task", "condition", "target", "target_task", "target_cohort"]


def main() -> None:
    a = pd.read_csv(LOGS / "unified" / "runs.csv")
    b = pd.read_csv(LOGS / f"unified_clean{TAG}" / "runs.csv")
    clinical = {"bridge2ai", "ipvs", "neurovoz", "mdvr_kcl"} if TAG else {"bridge2ai"}
    b2ai = lambda r: r[r.source.isin(clinical) | r.target.isin(clinical)]
    m = b2ai(a).merge(b2ai(b), on=KEY + ["seed"], suffixes=("_orig", "_clean"))
    m["d"] = m.auroc_clean - m.auroc_orig
    g = m.groupby(KEY).agg(original=("auroc_orig", "mean"), cleaned=("auroc_clean", "mean"),
                           change=("d", "mean"), up=("d", lambda x: int((x > 0).sum())),
                           seeds=("d", "size"), n_pd=("n_pd_clean", "first"),
                           n_hc=("n_hc_clean", "first")).round(3).reset_index()
    g.to_csv(LOGS / f"bridge2ai_clean{TAG}_compare.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(g.to_string(index=False))
    print("\nlargest absolute change:", g.change.abs().max(),
          "| mean absolute change:", round(g.change.abs().mean(), 3),
          "| rows:", len(g))
    for f in ["bridge2ai_cross_task.json"]:
        try:
            x = json.loads((LOGS / "unified" / f).read_text())["mean"]
            y = json.loads((LOGS / f"unified_clean{TAG}" / f).read_text())["mean"]
            print("\ncross-task (train -> test: original, cleaned)")
            for r0, r1 in zip(x, y):
                print(f"  {r0['train']:6s} -> {r0['test']:6s}  {r0['mean']:.3f}  {r1['mean']:.3f}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
