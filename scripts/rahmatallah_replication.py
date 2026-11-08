"""F-26 follow-up -- can our pipeline reproduce Rahmatallah et al. 2025, and is sex the reason?

Rahmatallah et al. (Sci Rep 15:7337, 2025) report AUC ~0.95 on an mPower cohort aged
50-70, with ONE recording per participant, so their split is participant-disjoint. Our
age-matched mPower result is ~0.6. F-26 used to explain the gap as identity leakage; that
was wrong (corrected 2026-10-04). Their cohort still carries an 18-point sex gap
(35.6% vs 17.1% female) and a 3.5-year age gap, and it was curated by listening.

Their Supplementary Table S1 lists the 398 recordIds, which are our wav filenames, so the
cohort can be rebuilt exactly. Two versions are built:

    exact   their 398 recordings (Table S1)
    rule    their written selection criteria applied to our data, one RANDOM recording
            per participant -- the same cohort definition without curation by ear

and each is evaluated as published, then sex-matched, then age+sex-matched. Matching
draws a random subset, so it is repeated over several seeds.

Two protocols:
    theirs  100 random stratified 70/30 splits, mean AUROC (one row per participant, so
            this is participant-disjoint)
    ours    5-fold StratifiedGroupKFold out-of-fold AUROC with a bootstrap CI

Model: the thesis representation (XLS-R-300M layer 15, mean+std pooled), with a
standardised logistic-regression probe, as in sex_control.py. Demographics-only probes
(age, sex, age+sex) are run on exactly the same splits.

Run:  .venv\\Scripts\\python.exe -u scripts\\rahmatallah_replication.py
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdvoice.cohort import attach_age, attach_sex, demographic_match_ids
from pdvoice.config import EMBEDDINGS, LOGS, MPOWER_DIR
from pdvoice.eval.discrimination import bootstrap_ci

MODEL, LAYER = "xls-r-300m", 15
TABLE_S1 = Path(__file__).resolve().parents[1] / "references" / "rahmatallah2025_tableS1.csv"
# Table S1 of Rahmatallah et al. (2025) lists the mPower recordIds and healthCodes they
# used. It is not redistributed here (it carries participant identifiers): download it from
# the paper's supplementary material and save it at references/rahmatallah2025_tableS1.csv.
N_SPLITS_THEIRS = 100
MATCH_SEEDS = range(10)

# Rahmatallah et al., Methods: conditions excluded "as these conditions affect voice quality".
EXCLUDED_HISTORY = {
    "Depression", "Anxiety", "Asthma", "Chronic Obstructive Pulmonary Disease or Asthma",
    "Schizophrenia or Bipolar Disorder", "Stroke/Transient Ischemic Attack",
}
PD_MED_OK = {"Immediately before Parkinson medication", "Another time"}
HC_MED_OK = {"I don't take Parkinson medications"}


def probe(X: np.ndarray, y: np.ndarray, protocol: str, seed: int = 42) -> tuple[float, float, float]:
    """Return (AUROC, spread, extra) under one of the two protocols.

    theirs -> (mean over 100 splits, SD over splits, nan)
    ours   -> (out-of-fold AUROC, CI low, CI high); one row per participant, so the
              participant groups are simply the row indices.
    """
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    if protocol == "theirs":
        aucs = []
        sss = StratifiedShuffleSplit(N_SPLITS_THEIRS, test_size=0.30, random_state=seed)
        for tr, te in sss.split(X, y):
            m = clf().fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        return float(np.mean(aucs)), float(np.std(aucs)), float("nan")
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            X, y, np.arange(len(y))):
        oof[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    a, lo, hi = bootstrap_ci(y, oof)
    return float(a), float(lo), float(hi)


def load() -> tuple[pd.DataFrame, np.ndarray]:
    """One row per recording that has an embedding, with label, age, sex and record id."""
    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    rid = [re.sub(r"__audio\.wav$", "", Path(str(p)).name) for p in d["path"]]
    df = pd.DataFrame({"recordId": rid, "subject_id": d["subject_id"].astype(str),
                       "label": d["label"].astype(int), "row": np.arange(len(rid))})
    df = attach_sex(attach_age(df))
    meta = pd.read_csv(MPOWER_DIR / "voice_activity_metadata.csv",
                       usecols=["recordId", "medTimepoint"])
    df = df.merge(meta.drop_duplicates("recordId"), on="recordId", how="left")
    demo = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv",
                       usecols=["healthCode", "health-history"])
    hist = {h: set(re.findall(r'"([^"]+)"', str(v)))
            for h, v in zip(demo.healthCode.astype(str), demo["health-history"])}
    df["history"] = df.subject_id.str.replace("mpower_", "", regex=False).map(hist)
    return df, d["X"]


def rule_cohort(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Their written criteria, then one random recording per participant."""
    d = df[df.age.between(50, 70)]
    d = d[~d.history.apply(lambda h: bool(h & EXCLUDED_HISTORY) if isinstance(h, set) else False)]
    # Subjects with conflicting medication answers are excluded outright ...
    med = d.groupby("subject_id").agg(label=("label", "first"),
                                      meds=("medTimepoint", lambda s: set(s.dropna())))
    bad_pd = med[(med.label == 1) & med.meds.apply(lambda m: bool(m & HC_MED_OK))].index
    bad_hc = med[(med.label == 0) & med.meds.apply(lambda m: bool(m - HC_MED_OK))].index
    d = d[~d.subject_id.isin(set(bad_pd) | set(bad_hc))]
    # ... and patients' recordings made just after medication are dropped.
    ok = np.where(d.label == 1, d.medTimepoint.isin(PD_MED_OK), d.medTimepoint.isin(HC_MED_OK))
    d = d[ok]
    return (d.sample(frac=1.0, random_state=seed)
             .drop_duplicates("subject_id").sort_values("subject_id").reset_index(drop=True))


def describe(c: pd.DataFrame) -> dict:
    pdg, hcg = c[c.label == 1], c[c.label == 0]
    return dict(n_pd=len(pdg), n_hc=len(hcg),
                age_pd=round(pdg.age.mean(), 1), age_hc=round(hcg.age.mean(), 1),
                female_pd=round(100 * (pdg.sex == "Female").mean(), 1),
                female_hc=round(100 * (hcg.sex == "Female").mean(), 1))


def evaluate(c: pd.DataFrame, X_all: np.ndarray, protocol: str, seed: int = 42) -> dict:
    c = c.dropna(subset=["age", "sex"]).reset_index(drop=True)
    y = c.label.to_numpy()
    feats = {
        "age": c[["age"]].to_numpy(float),
        "sex": (c[["sex"]] == "Female").to_numpy(float),
        "age_sex": np.c_[c.age.to_numpy(float), (c.sex == "Female").to_numpy(float)],
        "ssl": X_all[c.row.to_numpy()].astype(np.float32),
    }
    out = {}
    for k, X in feats.items():
        a, s1, s2 = probe(X, y, protocol, seed)
        out[k] = a
        if k == "ssl":
            out["ssl_spread"] = (s1, s2)
    return out


def run_variant(name: str, base: pd.DataFrame, X_all: np.ndarray, match: tuple | None,
                rows: list) -> None:
    """Evaluate one cohort, matched over several seeds if ``match`` is given."""
    seeds = MATCH_SEEDS if match else [0]
    per = {"theirs": [], "ours": []}
    descs = []
    for s in seeds:
        c = base.dropna(subset=["age", "sex"])
        if match:
            c = c[c.subject_id.isin(demographic_match_ids(c, on=match, seed=s))]
        c = c.reset_index(drop=True)
        descs.append(describe(c))
        for proto in per:
            per[proto].append(evaluate(c, X_all, proto, seed=42 + s))
    desc = pd.DataFrame(descs).mean().round(1).to_dict()
    row = dict(cohort=name, match="+".join(match) if match else "none", **desc)
    for proto, res in per.items():
        R = pd.DataFrame(res)
        for k in ("age", "sex", "age_sex", "ssl"):
            row[f"{proto}_{k}"] = round(R[k].mean(), 3)
        row[f"{proto}_ssl_sd_over_matchseeds"] = round(R["ssl"].std(ddof=0), 3)
        if not match:
            row[f"{proto}_ssl_spread"] = [round(v, 3) for v in res[0]["ssl_spread"]]
    rows.append(row)
    print(f"--- {name} | match: {row['match']} ---")
    print(f"   PD {row['n_pd']:.0f} / HC {row['n_hc']:.0f}   age {row['age_pd']} vs {row['age_hc']}"
          f"   female {row['female_pd']}% vs {row['female_hc']}%")
    for proto in per:
        print(f"   {proto:<6}  age {row[f'{proto}_age']:.3f}  sex {row[f'{proto}_sex']:.3f}"
              f"  age+sex {row[f'{proto}_age_sex']:.3f}   SSL {row[f'{proto}_ssl']:.3f}")


def main() -> None:
    warnings.simplefilter("ignore")
    df, X_all = load()
    print(f"recordings with an embedding: {len(df):,}\n")

    s1 = pd.read_csv(TABLE_S1, skiprows=1)
    exact = df[df.recordId.isin(set(s1.recordID))].drop_duplicates("subject_id").reset_index(drop=True)
    print(f"Table S1: {len(s1)} recordings; matched to our embeddings: {len(exact)}")
    hc_ok = (exact.subject_id.str.replace("mpower_", "", regex=False)
             .isin(set(s1.healthCode))).all()
    print(f"healthCodes agree with Table S1: {hc_ok}\n")

    rule = rule_cohort(df, seed=0)
    print(f"rule-based cohort before QC-by-ear: {describe(rule)}  "
          f"(paper: 215 PD / 229 HC before curation)\n")

    rows: list[dict] = []
    for name, base in (("exact (Table S1)", exact), ("rule (random recording)", rule)):
        for match in (None, ("sex",), ("age", "sex")):
            run_variant(name, base, X_all, match, rows)
        print()

    R = pd.DataFrame(rows)
    R.to_csv(LOGS / "rahmatallah_replication.csv", index=False)
    (LOGS / "rahmatallah_replication.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {LOGS / 'rahmatallah_replication.csv'}")


if __name__ == "__main__":
    main()
