"""Train on one Bridge2AI task, test on another, same people.

US patients and controls only, age-matched per seed, restricted to participants who
recorded all three task kinds. Folds are drawn over participants, so a participant's
vowel, DDK and speech recordings always fall in the same fold: the test participants are
never seen in training under any task. Same frozen XLS-R layer 11, MLP head and
calibration as scripts/unified_experiments.py; five seeds; participant-level AUROC.

Output: outputs/logs/unified[_clean]/bridge2ai_cross_task.json (aggregates only).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pdvoice.cohort import age_match_ids
from unified_experiments import OUT, SEEDS, fit_locked, load, predict

TASKS = ["vowel", "ddk", "speech"]


def main() -> None:
    data = {t: load("bridge2ai", t) for t in TASKS}
    people = set.intersection(*[set(df.subject_id) for df, _ in data.values()])
    subj = (data["speech"][0].drop_duplicates("subject_id")
            .loc[lambda d: d.subject_id.isin(people), ["subject_id", "label", "age"]])
    out = {"n_people_all_tasks": int(len(subj)), "runs": []}
    for seed in SEEDS:
        am = set(age_match_ids(subj, seed=seed))
        s = subj[subj.subject_id.isin(am)].reset_index(drop=True)
        fold = dict(zip(s.subject_id, np.zeros(len(s), int)))
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed)
                                    .split(s, s.label)):
            for i in te:
                fold[s.subject_id[i]] = k
        for a in TASKS:
            dfa, Xa = data[a]
            ina = dfa.subject_id.isin(am).to_numpy()
            preds = {b: [] for b in TASKS}
            for k in range(5):
                trm = ina & (dfa.subject_id.map(fold) != k).to_numpy()
                head, cal = fit_locked(Xa[trm], dfa.label.to_numpy()[trm],
                                       dfa.subject_id.to_numpy()[trm], seed)
                for b in TASKS:
                    dfb, Xb = data[b]
                    tem = (dfb.subject_id.isin(am) & (dfb.subject_id.map(fold) == k)).to_numpy()
                    r, _ = predict(head, cal, Xb[tem])
                    preds[b].append(pd.DataFrame({"s": dfb.subject_id.to_numpy()[tem],
                                                  "y": dfb.label.to_numpy()[tem], "r": r}))
            for b in TASKS:
                g = pd.concat(preds[b]).groupby("s").agg(y=("y", "first"), r=("r", "mean"))
                out["runs"].append(dict(seed=seed, train=a, test=b, n_pd=int(g.y.sum()),
                                        n_hc=int((g.y == 0).sum()),
                                        auroc=float(roc_auc_score(g.y, g.r))))
                print(seed, a, "->", b, round(out["runs"][-1]["auroc"], 3), flush=True)
    r = pd.DataFrame(out["runs"])
    out["mean"] = (r.groupby(["train", "test"]).auroc.agg(["mean", "std"]).round(3)
                   .reset_index().to_dict("records"))
    (OUT / "bridge2ai_cross_task.json").write_text(json.dumps(out, indent=2))
    print(r.groupby(["train", "test"]).auroc.mean().unstack().round(3).loc[TASKS, TASKS])


if __name__ == "__main__":
    main()
