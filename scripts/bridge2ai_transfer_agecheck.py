"""Is Bridge2AI -> external transfer carried by the TARGET's age structure?

IPVS includes 15 young healthy controls (about 20 years old) next to elderly controls and
patients, and NeuroVoz's groups also differ in age, so a model that responds to age can
score well on those targets without detecting Parkinson's disease. This re-scores the
locked Bridge2AI models (same seeds and protocol as bridge2ai_transfer.py) on
age-comparable target subsets:

    IPVS      elderly healthy controls vs PD only (young controls removed)
    NeuroVoz  frequency-matched on age in 5-year bins
    MDVR-KCL  no age data exists; reported as published only

Run:  .venv\\Scripts\\python.exe -u scripts\\bridge2ai_transfer_agecheck.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bridge2ai_transfer import SUFFIX, auc, bridge2ai  # noqa: E402
from demographic_erasure import load_external, score, train_locked  # noqa: E402

from pdvoice.cohort import age_match_ids  # noqa: E402
from pdvoice.config import LOGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402
from pdvoice.data.demographics import attach_age  # noqa: E402

TARGETS = {"vowel": [("ipvs", "vowel"), ("neurovoz", "vowel")],
           "ddk": [("neurovoz", "ddk")],
           "speech": [("ipvs", "speech"), ("neurovoz", "speech"), ("mdvr_kcl", "speech")]}


def subsets(ds, edf, seed):
    out = {"as published": np.ones(len(edf), bool)}
    if ds == "ipvs":
        out["no young controls"] = ~edf.subject_id.str.contains("_yhc_").to_numpy()
    if ds == "neurovoz":
        a = attach_age(edf, "neurovoz")
        out["age-matched"] = a.subject_id.isin(age_match_ids(a.dropna(subset=["age"]), seed=seed)).to_numpy()
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    seeds = [42, 43, 44, 45, 46]
    base = LOADERS["bridge2ai"]()
    ext = {k: load_external(*k) for v in TARGETS.values() for k in v}
    rows = []
    for task, targets in TARGETS.items():
        src, X = bridge2ai(task, base)
        for cohort in ("uncorrected", "age-matched"):
            for seed in seeds:
                m = (np.ones(len(src), bool) if cohort == "uncorrected"
                     else src.subject_id.isin(age_match_ids(src, seed=seed)).to_numpy())
                s = src[m]
                head, cal, tf = train_locked(X[m], s.label.to_numpy(), s.subject_id.to_numpy(), seed)
                for ds, tt in targets:
                    edf, E = ext[(ds, tt)]
                    p = score(head, cal, tf(E))
                    for name, mask in subsets(ds, edf, seed).items():
                        sub = edf[mask]
                        rows.append(dict(task=task, target=ds, target_subset=name, source=cohort,
                                         seed=seed, n_pd=int(sub[sub.label == 1].subject_id.nunique()),
                                         n_hc=int(sub[sub.label == 0].subject_id.nunique()),
                                         auroc=auc(sub, p[mask])))
            print(f"  {task} [{cohort}] done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(LOGS / f"bridge2ai_transfer_agecheck{SUFFIX}.csv", index=False)
    agg = R.groupby(["task", "target", "target_subset", "source"]).agg(
        n_pd=("n_pd", "median"), n_hc=("n_hc", "median"),
        auroc=("auroc", "mean"), sd=("auroc", "std")).round(3).unstack("source")
    pd.set_option("display.width", 220)
    print("\n" + agg.to_string())


if __name__ == "__main__":
    main()
