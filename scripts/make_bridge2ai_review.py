"""Copy a small, labelled sample of Bridge2AI participants for listening and manual checking.

Writes pd-datasets/bridge2ai_review/ (git-ignored, stays under the data use agreement):

    00_README.txt
    all_participants.csv      every participant: group, labels, age, sex, country, diagnoses,
                              PD subtype and stage, number of recordings per task type
    task_coverage.csv         % of each group that recorded each task, with the task's module
    <NN_group>/<participant>/
        labels.txt            that participant's labels and confirmed diagnoses
        <module>__<task>__ses<k>.wav   every recording, named by protocol module and task

Groups use confirmed (gold-standard "Yes") diagnoses only. Run:
    .venv\\Scripts\\python.exe scripts\\make_bridge2ai_review.py --per-group 3
"""
from __future__ import annotations

import argparse
import re
import shutil

import numpy as np
import pandas as pd

from pdvoice.config import BRIDGE2AI_DIR, PD_DATASETS
from pdvoice.data.bridge2ai import (_task_code, bridge2ai_participants, confirmed_ids,
                                    load_bridge2ai)

OUT = PD_DATASETS / "bridge2ai_review"
FAMILIES = {
    "Cognitive_impairment": ["cognitive_impairment"],
    "ALS": ["amyotrophic_lateral_sclerosis"],
    "Voice_disorder": ["benign_lesions", "laryngeal_dystonia", "muscle_tension_dysphonia",
                       "unilateral_vocal_fold_paralysis", "glottic_insufficiency",
                       "laryngeal_cancer", "laryngitis", "precancerous_lesions"],
    "Mood": ["depression", "anxiety", "bipolar_disorder"],
    "Respiratory": ["airway_stenosis", "unexplained_chronic_cough", "copd_and_asthma"],
}
ORDER = ["Parkinsons_USA", "Parkinsons_Canada", "Healthy_control", "Cognitive_impairment",
         "ALS", "Voice_disorder", "Mood", "Respiratory", "Excluded_control_voice_condition",
         "Excluded_ambiguous", "Excluded_control_with_diagnosis"]


def confirmed_diagnoses() -> dict[str, list[str]]:
    """participant -> list of tables in which the gold-standard diagnosis is 'Yes'."""
    out: dict[str, list[str]] = {}
    for f in sorted((BRIDGE2AI_DIR / "phenotype" / "diagnosis").glob("*.tsv")):
        if f.stem == "control":
            continue
        for pid in sorted(confirmed_ids(pd.read_csv(f, sep="\t", dtype=str))):
            out.setdefault(pid, []).append(f.stem)
    return out


def task_modules() -> dict[str, str]:
    """Release task name (run and version tags removed) -> protocol module, from the
    official acoustic_task_cohort column ('generic' = common to every cohort)."""
    a = pd.read_csv(BRIDGE2AI_DIR / "phenotype" / "task" / "acoustic_task.tsv", sep="\t", dtype=str)
    a["base"] = a.acoustic_task_name.map(_task_code)
    m = a.groupby("base").acoustic_task_cohort.agg(lambda x: x.value_counts().index[0]).to_dict()
    m.setdefault("harvard-sentences-list", "generic")
    m.setdefault("rainbowpassage", "generic")
    return m


def base_task(code: str) -> str:
    for prefix in ("diadochokinesis", "respiration-and-cough", "glides"):
        if code.startswith(prefix):
            return prefix
    return {"high-to-low": "glides", "harvard-sentences-list": "harvard-sentences"}.get(code, code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    P = bridge2ai_participants().set_index("participant_id")
    dx = confirmed_diagnoses()
    modules = task_modules()

    def group(pid: str, g: str) -> str | None:
        if g == "pd":
            return "Parkinsons_USA"
        if g == "pd_non_us":
            return "Parkinsons_Canada"
        if g == "control":
            return "Healthy_control"
        if g == "control_voice_condition":
            return "Excluded_control_voice_condition"
        if g == "ambiguous":
            return "Excluded_ambiguous"
        if g == "control_with_diagnosis":
            return "Excluded_control_with_diagnosis"
        for fam, tables in FAMILIES.items():          # first matching family, in this order
            if any(t in tables for t in dx.get(pid, [])):
                return fam
        return None                                    # 'other' without a confirmed diagnosis

    P["review_group"] = [group(pid, g) for pid, g in zip(P.index, P.group)]
    P["confirmed_diagnoses"] = [", ".join(dx.get(pid, [])) or "none" for pid in P.index]
    pdx = pd.read_csv(BRIDGE2AI_DIR / "phenotype" / "diagnosis" / "parkinsons_disease.tsv",
                      sep="\t", dtype=str).drop_duplicates("participant_id").set_index("participant_id")
    P["pd_subtype"] = pdx.diagnosis_parkinsons_disease_subtype.reindex(P.index)

    D = load_bridge2ai(recording_meta=True)
    D["pid"] = D.subject_id.str.replace("b2ai_", "", regex=False)
    D["task"] = D.task_code.map(base_task)
    D["module"] = D.task_code.map(lambda c: modules.get(c, modules.get(base_task(c), "unknown")))
    D["review_group"] = D.pid.map(P.review_group)

    OUT.mkdir(parents=True, exist_ok=True)
    counts = D.groupby(["pid", "task_kind"]).size().unstack(fill_value=0).add_prefix("n_")
    table = P.join(counts).reset_index()
    table["label_pd_vs_healthy"] = table.label
    table[["participant_id", "review_group", "group", "label_pd_vs_healthy", "age", "sex",
           "country", "confirmed_diagnoses", "pd_subtype", "hy",
           *[c for c in table.columns if c.startswith("n_")]]
          ].sort_values(["review_group", "participant_id"]).to_csv(OUT / "all_participants.csv",
                                                                   index=False)

    n = P.review_group.value_counts()
    cov = (D.dropna(subset=["review_group"]).groupby(["task", "module", "review_group"])
           .pid.nunique().unstack(fill_value=0))
    cov = (cov / n[cov.columns] * 100).round(0).astype(int)
    cov[[g for g in ORDER if g in cov.columns]].sort_index(level=["module", "task"]).to_csv(
        OUT / "task_coverage.csv")

    rng = np.random.default_rng(args.seed)
    copied = 0
    for i, g in enumerate(ORDER, 1):
        ids = np.sort(P.index[P.review_group == g].to_numpy())
        ids = [x for x in ids if x in set(D.pid)]
        pick = rng.choice(ids, min(args.per_group, len(ids)), replace=False) if ids else []
        for pid in pick:
            dest = OUT / f"{i:02d}_{g}" / pid
            dest.mkdir(parents=True, exist_ok=True)
            r = P.loc[pid]
            (dest / "labels.txt").write_text(
                f"participant: {pid}\nreview group: {g}\nloader group: {r.group}\n"
                f"PD-vs-healthy label: {r.label}\nage: {r.age}\nsex: {r.sex}\ncountry: {r.country}\n"
                f"confirmed diagnoses: {r.confirmed_diagnoses}\n"
                f"PD subtype: {r.pd_subtype}\nHoehn and Yahr: {r.hy}\n", encoding="utf-8")
            recs = D[D.pid == pid]
            sess = {s: k for k, s in enumerate(sorted(recs.session_id.unique()), 1)}
            for _, rec in recs.iterrows():
                name = f"{rec.module}__{rec.task_raw}__ses{sess[rec.session_id]}.wav"
                shutil.copy2(rec.path, dest / re.sub(r"[^\w.\-]", "_", name))
                copied += 1
        print(f"{i:02d}_{g:34s} {len(ids):4d} available, {len(pick)} copied", flush=True)

    (OUT / "00_README.txt").write_text(
        "Bridge2AI-Voice review sample. DATA USE AGREEMENT: do not move, share or upload.\n\n"
        "Folders: one per group, three randomly chosen participants each (seed 42).\n"
        "File names: <module>__<task>__ses<k>.wav\n"
        "  module 'generic' = Part A, recorded by every cohort\n"
        "  'neurology' / 'voice' / 'mood' / 'respiratory' = cohort-specific Part B tasks\n"
        "labels.txt in each participant folder gives the labels used in the thesis.\n"
        "all_participants.csv lists every participant; task_coverage.csv gives the % of each\n"
        "group that recorded each task.\n\n"
        "Groups use confirmed (gold-standard) diagnoses only. Parkinsons_USA and\n"
        "Healthy_control are the PD-vs-healthy cohort; every other group is used only in the\n"
        "specificity analysis or is excluded.\n", encoding="utf-8")
    print(f"copied {copied} recordings to {OUT}")


if __name__ == "__main__":
    main()
