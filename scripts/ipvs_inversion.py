"""Why does the locked mPower model score BELOW chance on IPVS?

IPVS patients are 20.5 years older than its controls, the same direction as mPower
(27.5 years). A model that had mainly learned "older = Parkinson's" should therefore score
ABOVE chance on IPVS. It scores 0.287 (vowel, L11). This script asks what the scores track.

IPVS has three groups: 15 young controls (yhc), 22 elderly controls (ehc) and 28 patients
(pd). The 20.5-year gap is driven by the young controls; patients against elderly controls
is the near-age-matched contrast inside the corpus. A 44.1 kHz recording batch contains
patients only (F-17).

For each saved set of subject-level predictions from a locked mPower model it reports:
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
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

from pdvoice.config import LOGS
from pdvoice.data.demographics import attach_age
from pdvoice.data.ipvs import load_ipvs

PRED_FILES = {
    "vowel L11 (Table: transfer A)": "layer2__xls-r-300m__ipvs__vowel__L11__subject_preds.csv",
    "speech L11 (Table: transfer A)": "layer2__xls-r-300m__ipvs__speech__L11__subject_preds.csv",
    "vowel L15": "layer2__xls-r-300m__ipvs__vowel__L15__subject_preds.csv",
    "vowel L24 (final layer)": "layer2__xls-r-300m__ipvs__vowel__LNone__subject_preds.csv",
    "vowel L15, age-matched model": "agematched__xls-r-300m__L15__ipvs_subject_preds.csv",
}


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


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

    out = {}
    for name, fn in PRED_FILES.items():
        f = LOGS / fn
        if not f.exists():
            print(f"(missing {fn})")
            continue
        p = pd.read_csv(f)[["subject_id", "p"]].merge(subj, on="subject_id", how="inner")
        y = p.label.to_numpy()
        r = dict(n=len(p), auroc_all=auc(y, p.p))
        pe = p[p.group.isin(["pd", "ehc"])]
        py = p[p.group.isin(["pd", "yhc"])]
        r["auroc_pd_vs_elderly"] = auc(pe.label, pe.p)
        r["auroc_pd_vs_young"] = auc(py.label, py.p)
        r["mean_score_by_group"] = p.groupby("group").p.mean().round(3).to_dict()
        a = p.dropna(subset=["age"])
        r["age_only_auroc"] = auc(a.label, a.age)
        rho, pv = spearmanr(a.age, a.p)
        r["spearman_score_age_all"] = (round(float(rho), 3), round(float(pv), 4))
        for cls, lab in ((1, "patients"), (0, "controls")):
            s = a[a.label == cls]
            rho, pv = spearmanr(s.age, s.p)
            r[f"spearman_score_age_{lab}"] = (round(float(rho), 3), round(float(pv), 4))
        no44 = p[~p.any44]
        r["auroc_without_44k_participants"] = auc(no44.label, no44.p)
        r["n_without_44k"] = int(len(no44))
        pdp = p[p.label == 1]
        if pdp.any44.any() and (~pdp.any44).any():
            u = mannwhitneyu(pdp[pdp.any44].p, pdp[~pdp.any44].p)
            r["patients_44k_vs_16k_mean_score"] = (round(float(pdp[pdp.any44].p.mean()), 3),
                                                   round(float(pdp[~pdp.any44].p.mean()), 3),
                                                   round(float(u.pvalue), 4))
        out[name] = r
        print(f"=== {name}  (n={r['n']})")
        for k, v in r.items():
            if k != "n":
                print(f"   {k:<34} {v if not isinstance(v, float) else round(v, 3)}")
        print()

    (LOGS / "ipvs_inversion.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {LOGS / 'ipvs_inversion.json'}")


if __name__ == "__main__":
    main()
