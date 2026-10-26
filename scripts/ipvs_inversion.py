"""Why does the locked mPower model score BELOW chance on IPVS?

IPVS patients are 20.5 years older than its controls, the same direction as mPower
(27.5 years). A model that had mainly learned "older = Parkinson's" should therefore score
ABOVE chance on IPVS. It scores below 0.5 instead. This script asks what the scores track.

IPVS has three groups: 15 young controls (yhc), 22 elderly controls (ehc) and 28 patients
(pd). The 20.5-year gap is driven by the young controls; patients against elderly controls
is the near-age-matched contrast inside the corpus. A 44.1 kHz recording batch contains
patients only (F-17).

For the locked mPower models of Table 4.8 (layer 11, five seeds, averaged) it reports:
  * AUROC on all participants, on patients vs elderly controls, and on patients vs young
    controls, and the mean score per group
  * Spearman correlation of the score with age, overall and within each class
  * AUROC of age alone on the same participants (what an age-only model would achieve)
  * AUROC after removing every participant who has any 44.1 kHz recording
  * the score difference between patients inside and outside the 44.1 kHz batch

Run:  .venv\\Scripts\\python.exe scripts\\ipvs_inversion.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from pdvoice.config import LOGS
from pdvoice.data.demographics import attach_age
from pdvoice.data.ipvs import load_ipvs

# The same five-seed, layer-11 predictions as Table 4.8 (scripts/unified_experiments.py),
# so every number here agrees with the transfer tables by construction.
SOURCES = {
    "vowel, uncorrected model": dict(condition="uncorrected", target_task="vowel"),
    "speech, uncorrected model": dict(condition="uncorrected", target_task="speech"),
    "vowel, age-matched model": dict(condition="agematched", target_task="vowel"),
    "speech, age-matched model": dict(condition="agematched", target_task="speech"),
}


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def one_run(p: pd.DataFrame) -> dict:
    """Every diagnostic for one set of participant-level scores."""
    y = p.label.to_numpy()
    r = dict(n=len(p), auroc_all=auc(y, p.p))
    pe = p[p.group.isin(["pd", "ehc"])]
    r["auroc_pd_vs_elderly"] = auc(pe.label, pe.p)
    a = p.dropna(subset=["age"])
    r["age_only_auroc"] = auc(a.label, a.age)
    c = a[a.label == 0]
    r["spearman_score_age_controls"] = (float(spearmanr(c.age, c.p)[0]), None)
    no44 = p[~p.any44]
    r["auroc_without_44k_participants"] = auc(no44.label, no44.p)
    pdp = p[p.label == 1]
    r["patients_44k_minus_16k_score"] = float(pdp[pdp.any44].p.mean() - pdp[~pdp.any44].p.mean())
    return r


def main() -> None:
    man = load_ipvs()
    subj = (man.assign(sr44=man.native_sr.fillna(0).astype(int) >= 44100)
               .groupby("subject_id").agg(label=("label", "first"), any44=("sr44", "any"))
               .reset_index())
    subj["group"] = subj.subject_id.str.split("_").str[1]
    subj = attach_age(subj, "ipvs")
    print("participants by group:", subj.group.value_counts().to_dict(),
          "| with age:", int(subj.age.notna().sum()))
    print("mean age by group:", subj.groupby("group").age.mean().round(1).to_dict())
    print("patients in the 44.1 kHz batch:", int(subj[subj.label == 1].any44.sum()),
          "| controls in it:", int(subj[subj.label == 0].any44.sum()), "\n")

    S = pd.read_csv(LOGS / "unified" / "scores.csv.gz")
    S = S[(S.exp == "exp1") & (S.target == "ipvs") & (S.target_cohort == "released")]
    out = {}
    for name, key in SOURCES.items():
        g = S[(S.condition == key["condition"]) & (S.target_task == key["target_task"])]
        per_seed = []
        for _, run in g.groupby("seed"):
            # discrimination uses the ranking score (calibration.RankingMap)
            p = (run.rename(columns={"s": "subject_id"})[["subject_id", "r"]]
                 .rename(columns={"r": "p"}).merge(subj, on="subject_id", how="inner"))
            per_seed.append(one_run(p))
        r = {"n": per_seed[0]["n"], "n_seeds": len(per_seed)}
        for k in per_seed[0]:
            if k == "n":
                continue
            v = [x[k] for x in per_seed]
            r[k] = (round(float(np.mean([t[0] for t in v])), 3), None)                 if isinstance(v[0], tuple) else float(np.mean(v))
        r["per_seed_pd_vs_elderly"] = [round(x["auroc_pd_vs_elderly"], 3) for x in per_seed]
        out[name] = r
        print(f"=== {name}  (n={r['n']}, {len(per_seed)} seeds)")
        for k, v in r.items():
            if k != "n":
                print(f"   {k:<34} {v if not isinstance(v, float) else round(v, 3)}")
        print()

    (LOGS / "ipvs_inversion.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {LOGS / 'ipvs_inversion.json'}")


if __name__ == "__main__":
    main()
