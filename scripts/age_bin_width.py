"""Why five-year age bins? Sensitivity of frequency matching to the bin width.

Frequency matching keeps min(n_PD, n_HC) participants per age bin. Narrow bins remove
more of the age difference but discard more participants; wide bins keep more but leave
age differences inside each bin. This reports, for 2-, 5- and 10-year bins: participants
kept, remaining mean age gap, how well age alone still predicts the label, and the voice
AUROC (linear probe, participant-disjoint 5-fold CV, five matching seeds).

Datasets: mPower vowel (one recording per participant, XLS-R layer 15) and Bridge2AI
connected speech (PD vs healthy controls, layer 15).

Run:  .venv\\Scripts\\python.exe -u scripts\\age_bin_width.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from bridge2ai_agematch import load_task, subject_scores  # noqa: E402

from pdvoice.cohort import age_match_ids, load_mpower_cohort  # noqa: E402
from pdvoice.config import EMBEDDINGS, LOGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
WIDTHS = [2, 5, 10]


def mpower():
    d = np.load(EMBEDDINGS / "mpower_full__xls-r-300m__L15.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))
    df = load_mpower_cohort(cap=1, seed=42).merge(emb, on="path", how="inner")
    df = df.dropna(subset=["age"]).reset_index(drop=True)
    return df, d["X"][df.row.to_numpy()].astype(np.float32)


def run(name, df, X):
    y = df.label.astype(int).to_numpy()
    rows = []
    for w in WIDTHS:
        r = []
        for seed in SEEDS:
            m = df.subject_id.isin(age_match_ids(df, seed=seed, bin_width=w)).to_numpy()
            s = subject_scores(df[m].reset_index(drop=True), X[m], y[m], seed)
            r.append(dict(n_per_group=int(s.y.sum()),
                          gap=float(s.age[s.y == 1].mean() - s.age[s.y == 0].mean()),
                          age_only=float(roc_auc_score(s.y, s.age)),
                          voice=float(roc_auc_score(s.y, s.p))))
        R = pd.DataFrame(r)
        rows.append(dict(dataset=name, bin_width=w, n_per_group=int(R.n_per_group.mean()),
                         age_gap=round(R.gap.mean(), 2), age_only=round(R.age_only.mean(), 3),
                         voice=round(R.voice.mean(), 3), voice_sd=round(R.voice.std(), 3)))
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    warnings.simplefilter("ignore")
    out = run("mPower vowel", *mpower())
    out += run("Bridge2AI speech", *load_task(LOADERS["bridge2ai"](), "speech"))
    (LOGS / "age_bin_width.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
