"""Cohort facts behind Tables 3.1-3.3: recordings, speakers, class balance, age and sex.

Computed from the same loaders and cohort functions as scripts/unified_experiments.py, so
the dataset tables cannot drift from the cohorts the experiments actually used.

  raw        loader output before quality control and capping (Table 3.1)
  cohort     participants the experiments use, over every task of the dataset: mPower
             capped at three recordings, Bridge2AI restricted to US patients and controls
             and to the common tasks (Tables 3.2 and 3.3)

Run:  .venv\\Scripts\\python.exe scripts\\cohort_summary.py
Writes outputs/logs/cohort_summary.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_experiments import SETS, load  # noqa: E402

from pdvoice.cohort import cap_per_subject  # noqa: E402
from pdvoice.config import LOGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402


def main() -> None:
    warnings.simplefilter("ignore")
    out = {}
    for ds, tasks in SETS.items():
        raw = LOADERS[ds]()
        lab = raw[raw["label"].notna()] if ds == "mpower" else raw
        r = {"raw_recordings": int(len(raw)), "raw_speakers": int(raw.subject_id.nunique()),
             "unlabelled_recordings": int(len(raw) - len(lab)),
             "unlabelled_speakers": int(raw.subject_id.nunique() - lab.subject_id.nunique())}
        parts = []
        for task in tasks:
            df, _ = load(ds, task)
            if ds == "mpower":
                df = cap_per_subject(df, 3, 42)
            parts.append(df[["subject_id", "label", "age", "sex"]
                            + (["country"] if "country" in df else [])])
        sub = pd.concat(parts).drop_duplicates("subject_id")
        if "country" in sub:
            sub = sub[sub.country == "USA"]
        n_pd, n_hc = int((sub.label == 1).sum()), int((sub.label == 0).sum())
        age_pd, age_hc = sub[sub.label == 1].age.mean(), sub[sub.label == 0].age.mean()
        r.update(n_pd=n_pd, n_hc=n_hc, prevalence=n_pd / (n_pd + n_hc),
                 age_pd=None if np.isnan(age_pd) else float(age_pd),
                 age_hc=None if np.isnan(age_hc) else float(age_hc),
                 age_coverage=float(sub.age.notna().mean()),
                 sex_coverage=float(sub.sex.notna().mean()))
        out[ds] = r
        print(ds, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    (LOGS / "cohort_summary.json").write_text(json.dumps(out, indent=2))
    print("wrote", LOGS / "cohort_summary.json")


if __name__ == "__main__":
    main()
