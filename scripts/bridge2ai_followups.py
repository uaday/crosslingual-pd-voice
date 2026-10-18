"""Three follow-up checks on the Bridge2AI results, run before writing them up.

A. Speech split. "Speech" pools read and spontaneous tasks. Which carries the signal?
     read         rainbow / caterpillar passage, CAPE-V sentences, Harvard sentences
     spontaneous  picture description, Cinderella story, story recall
   Within Bridge2AI (uncorrected and age-matched) and locked transfer to the
   age-comparable external subsets used in bridge2ai_transfer_agecheck.py.

B. Age AND sex matching. PD is 42% female and controls 50%; the embeddings encode sex.
   Frequency matching on 5-year age bin x sex (pdvoice.cohort.demographic_match_ids),
   within Bridge2AI and in transfer.

C. Specificity: PD vs other disorders. Bridge2AI's non-PD, non-control participants have
   voice, neurological, mood and respiratory conditions and are closer in age to PD.
     C1  PD vs every other disorder, uncorrected and age-matched
     C2  PD vs each disorder family, uncorrected and age-matched
     C3  confusion: a PD-vs-control model (age-matched) scores the other disorders. Do
         they look like PD? (AUROC PD vs family, and family vs controls, on its scores)

Protocol as elsewhere: XLS-R layer 15 mean+std, linear probe (PCA + logistic regression)
with participant-disjoint 5-fold CV for within-dataset numbers, locked MLP head for
transfer, participant-level AUROC, five seeds.

Run:  .venv\\Scripts\\python.exe -u scripts\\bridge2ai_followups.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from bridge2ai_agematch import subject_scores  # noqa: E402
from bridge2ai_transfer import auc  # noqa: E402
from bridge2ai_transfer_agecheck import subsets  # noqa: E402
from demographic_erasure import load_external, score, train_locked  # noqa: E402

from pdvoice.cohort import age_match_ids, demographic_match_ids  # noqa: E402
from pdvoice.config import BRIDGE2AI_DIR, EMBEDDINGS, LOGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402

MODEL = "xls-r-300m"
LAYER = int(os.environ.get("PDV_LAYER", 15))  # Bridge2AI layer; 8 is chosen inside Bridge2AI (bridge2ai_layer_select.py)
SUFFIX = "" if LAYER == 15 else f"__L{LAYER}"
SEEDS = [42, 43, 44, 45, 46]
READ = {"rainbow-passage", "rainbowpassage", "caterpillar-passage", "cape-v-sentences",
        "harvard-sentences-list"}
SPONT = {"picture-description", "cinderella-story", "story-recall"}
FAMILIES = {
    "neurological (cognitive impairment, ALS)": ["cognitive_impairment", "amyotrophic_lateral_sclerosis"],
    "voice disorders": ["benign_lesions", "laryngeal_dystonia", "muscle_tension_dysphonia",
                        "unilateral_vocal_fold_paralysis", "glottic_insufficiency",
                        "laryngeal_cancer", "laryngitis", "precancerous_lesions"],
    "mood": ["depression", "anxiety", "bipolar_disorder"],
    "respiratory": ["airway_stenosis", "unexplained_chronic_cough", "copd_and_asthma"],
}
SPEECH_TARGETS = [("ipvs", "speech", "no young controls"), ("neurovoz", "speech", "age-matched"),
                  ("mdvr_kcl", "speech", "as published")]
VOWEL_TARGETS = [("ipvs", "vowel", "as published"), ("neurovoz", "vowel", "age-matched")]


def load(task, base, groups):
    d = np.load(EMBEDDINGS / f"bridge2ai__{MODEL}__{task}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))
    df = emb.merge(base, on="path", how="inner")
    df = df[df.group.isin(groups)].dropna(subset=["age"]).reset_index(drop=True)
    return df, d["X"][df.row.to_numpy()].astype(np.float32)


def within(df, X, y, matcher):
    """Uncorrected AUROC, and mean/SD over seeds after matching (matcher(df, seed) -> ids)."""
    s = subject_scores(df, X, y, 42)
    out = dict(n_pos=int(s.y.sum()), n_neg=int((s.y == 0).sum()),
               age_pos=round(float(s.age[s.y == 1].mean()), 1),
               age_neg=round(float(s.age[s.y == 0].mean()), 1),
               age_only=round(float(roc_auc_score(s.y, s.age)), 3),
               voice=round(float(roc_auc_score(s.y, s.p)), 3))
    vals, ages, n = [], [], 0
    for seed in SEEDS:
        m = df.subject_id.isin(matcher(df.assign(label=y), seed)).to_numpy()
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        sm = subject_scores(df[m].reset_index(drop=True), X[m], y[m], seed)
        if min(sm.y.sum(), (sm.y == 0).sum()) < 10:
            continue
        vals.append(roc_auc_score(sm.y, sm.p))
        ages.append(roc_auc_score(sm.y, sm.age))
        n = int(sm.y.sum())
    if vals:
        out.update(matched_n_per_group=n, matched_age_only=round(float(np.mean(ages)), 3),
                   matched_voice=round(float(np.mean(vals)), 3),
                   matched_sd=round(float(np.std(vals, ddof=1)), 3) if len(vals) > 1 else None)
    return out


def transfer(df, X, matcher, targets, ext):
    """Locked head per seed on the matched source; mean AUROC per age-fair target subset."""
    res = {}
    for seed in SEEDS:
        m = df.subject_id.isin(matcher(df, seed)).to_numpy()
        s = df[m]
        head, cal = train_locked(X[m], s.label.to_numpy(), s.subject_id.to_numpy(), seed)
        for ds, task, subset in targets:
            edf, E = ext[(ds, task)]
            mask = subsets(ds, edf, seed)[subset]
            res.setdefault(f"{ds} {task} ({subset})", []).append(
                auc(edf[mask], score(head, cal, E)[mask]))
    return {k: [round(float(np.mean(v)), 3), round(float(np.std(v, ddof=1)), 3)] for k, v in res.items()}


def age_match(df, seed):
    return age_match_ids(df, seed=seed)


def age_sex_match(df, seed):
    return demographic_match_ids(df.dropna(subset=["sex"]), on=("age", "sex"), seed=seed)


def family_ids():
    out = {}
    for fam, files in FAMILIES.items():
        ids = set()
        for f in files:
            ids |= set(pd.read_csv(BRIDGE2AI_DIR / "phenotype" / "diagnosis" / f"{f}.tsv",
                                   sep="\t", dtype=str).participant_id)
        out[fam] = {f"b2ai_{i}" for i in ids}
    return out


def confusion(df, X, fams):
    """Train PD vs control on the age-matched cohort (5-fold, out-of-fold for PD/HC); score
    every 'other' participant with each fold's model and average."""
    rows = []
    for seed in SEEDS:
        pc = df[df.group.isin(["pd", "control"])]
        keep = age_match_ids(pc.assign(label=(pc.group == "pd").astype(int)), seed=seed)
        tr_mask = df.subject_id.isin(keep).to_numpy()
        oth_mask = (df.group == "other").to_numpy()
        T, Xt = df[tr_mask].reset_index(drop=True), X[tr_mask]
        yt = (T.group == "pd").astype(int).to_numpy()
        g = T.subject_id.to_numpy()
        oof = np.zeros(len(T))
        p_oth = np.zeros(oth_mask.sum())
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xt, yt, g):
            k = min(64, len(np.unique(g[tr])) - 1)
            clf = make_pipeline(StandardScaler(), PCA(k, random_state=seed),
                                LogisticRegression(max_iter=2000)).fit(Xt[tr], yt[tr])
            oof[te] = clf.predict_proba(Xt[te])[:, 1]
            p_oth += clf.predict_proba(X[oth_mask])[:, 1] / 5
        sc = pd.concat([pd.DataFrame({"s": g, "grp": T.group.to_numpy(), "p": oof}),
                        pd.DataFrame({"s": df.subject_id[oth_mask].to_numpy(), "grp": "other", "p": p_oth})])
        sc = sc.groupby("s").agg(grp=("grp", "first"), p=("p", "mean"))
        pdp, hc = sc.p[sc.grp == "pd"], sc.p[sc.grp == "control"]
        r = {"seed": seed, "PD vs control": roc_auc_score([1] * len(pdp) + [0] * len(hc), list(pdp) + list(hc))}
        for fam, ids in {"all other disorders": set(sc.index[sc.grp == "other"]), **fams}.items():
            fo = sc.p[(sc.grp == "other") & sc.index.isin(ids)]
            if len(fo) < 10:
                continue
            r[f"PD vs {fam}"] = roc_auc_score([1] * len(pdp) + [0] * len(fo), list(pdp) + list(fo))
            r[f"{fam} vs control"] = roc_auc_score([1] * len(fo) + [0] * len(hc), list(fo) + list(hc))
        rows.append(r)
    R = pd.DataFrame(rows).drop(columns="seed")
    return {k: [round(float(R[k].mean()), 3), round(float(R[k].std()), 3)] for k in R.columns}


def main() -> None:
    warnings.simplefilter("ignore")
    base = LOADERS["bridge2ai"]()
    ext = {(ds, t): load_external(ds, t) for ds, t, _ in SPEECH_TARGETS + VOWEL_TARGETS}
    report = {"A_speech_split": {}, "B_age_sex": {}, "C_specificity": {}}

    # ---- A. speech split
    sp, Xs = load("speech", base, ["pd", "control"])
    for name, codes in (("read", READ), ("spontaneous", SPONT)):
        m = sp.task_code.isin(codes).to_numpy()
        d, X = sp[m].reset_index(drop=True), Xs[m]
        y = d.label.astype(int).to_numpy()
        d["label"] = y
        r = within(d, X, y, age_match)
        r["transfer_age_matched_source"] = transfer(d, X, age_match, SPEECH_TARGETS, ext)
        report["A_speech_split"][name] = r
        print(f"A {name:<12} {json.dumps(r)}", flush=True)

    # ---- B. age and sex matching
    for task in ("vowel", "ddk", "speech"):
        d, X = load(task, base, ["pd", "control"])
        has_sex = d.sex.notna().to_numpy()
        d, X = d[has_sex].reset_index(drop=True), X[has_sex]
        y = d.label.astype(int).to_numpy()
        d["label"] = y
        r = within(d, X, y, age_sex_match)
        sub = d.drop_duplicates("subject_id")
        r["pct_female_pd"] = round(100 * float((sub[sub.label == 1].sex == "Female").mean()), 1)
        r["pct_female_hc"] = round(100 * float((sub[sub.label == 0].sex == "Female").mean()), 1)
        if task in ("vowel", "speech"):
            r["transfer_age_sex_matched_source"] = transfer(
                d, X, age_sex_match, SPEECH_TARGETS if task == "speech" else VOWEL_TARGETS, ext)
        report["B_age_sex"][task] = r
        print(f"B {task:<12} {json.dumps(r)}", flush=True)

    # ---- C. specificity
    fams = family_ids()
    for task in ("vowel", "ddk", "speech"):
        d, X = load(task, base, ["pd", "control", "other"])
        res = {}
        po = d[d.group.isin(["pd", "other"])]
        mask = d.group.isin(["pd", "other"]).to_numpy()
        po = po.reset_index(drop=True)
        y = (po.group == "pd").astype(int).to_numpy()
        res["PD vs all other disorders"] = within(po.assign(label=y), X[mask], y, age_match)
        for fam, ids in fams.items():
            fm = (po.group == "pd") | po.subject_id.isin(ids)
            fm = fm.to_numpy()
            pf = po[fm].reset_index(drop=True)
            yf = (pf.group == "pd").astype(int).to_numpy()
            res[f"PD vs {fam}"] = within(pf.assign(label=yf), X[mask][fm], yf, age_match)
        res["confusion_PD_vs_control_model"] = confusion(d, X, fams)
        report["C_specificity"][task] = res
        print(f"C {task:<12} {json.dumps(res)}", flush=True)

    (LOGS / f"bridge2ai_followups{SUFFIX}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {LOGS / 'bridge2ai_followups.json'}")


if __name__ == "__main__":
    main()
