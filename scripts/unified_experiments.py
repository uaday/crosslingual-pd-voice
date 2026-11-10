"""Every experiment of Chapter 4 under one protocol.

Protocol, identical for every number this script writes:
  * frozen XLS-R-300M, layer 11 (chosen by cross-validation on mPower, the proposal's
    training set; layers 7 to 13 are equivalent inside Bridge2AI);
  * fixed analysis window (central 2 s of a vowel, 5 s windows averaged otherwise), mean+std
    pooling, the caches written by extract_mpower_full.py / layer2_transfer.py /
    extract_bridge2ai_layers.py;
  * the framework head (fit_mlp_head), calibration selected on a participant-disjoint
    validation split and locked; discrimination on the strictly increasing ranking map;
  * five seeds (42-46); participant-level scoring.

Experiments
  WITHIN  every dataset x task, 5-fold participant-disjoint CV, under each condition
  EXP1    train on mPower vowels (as proposed), test on every task of every other dataset
  EXP2    train on the connected speech of each dataset, test on every other dataset's speech
  EXP3    the same for the vowel and diadochokinesis, so that tasks are compared with the
          training source held fixed
  SPEC    Parkinson's disease against other disorders (the only dataset that has them is
          Bridge2AI): direct classifiers, and the PD-vs-control model scoring other groups

Conditions (applied the same way everywhere they are possible)
  uncorrected    data as released
  confident      mPower only: patients with a diagnosis year, controls without
  agematched     training cohort frequency-matched on age in 5-year bins
  sizematched    mPower only: a random sample of the uncorrected cohort with as many patients
                 and controls as the age-matched one (age gap kept), a control for cohort size
  agesexmatched  matched jointly on age and sex (every dataset but MDVR-KCL records sex)
Bridge2AI PD-vs-control uses US participants only (see pdvoice.data.bridge2ai).
Test cohorts: as released, age-matched (where age is recorded), age- and sex-matched (where
sex is recorded).

Outputs (outputs/logs/unified/):
  runs.csv     one row per (experiment, source, condition, seed, target, cohort)
  scores.csv.gz per-participant ranking scores, for seed-pooled bootstrap intervals
Run:  .venv\\Scripts\\python.exe -u scripts\\unified_experiments.py
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from pdvoice.calibration import select_best_by_brier
from pdvoice.cohort import age_match_ids, cap_per_subject, demographic_match_ids
from pdvoice.config import BRIDGE2AI_DIR, EMBEDDINGS, LOGS, MPOWER_DIR, VAL_FRACTION
from pdvoice.data import LOADERS
from pdvoice.data.bridge2ai import confirmed_ids
from pdvoice.data.common_tasks import common_task_mask
from pdvoice.data.demographics import attach_age
from pdvoice.data.splits import (assign_folds, ensure_two_class_val,
                                 train_val_test_indices)
from pdvoice.eval.metrics import calibration_error
from pdvoice.features.ssl import load_aligned
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

MODEL, LAYER, CAP = "xls-r-300m", 11, 3
SEEDS = [42, 43, 44, 45, 46]
# PDVOICE_B2AI_CLEAN=1 reruns everything on Bridge2AI audio with background noise removed (spectral gating) and
# silent recordings dropped (scripts/bridge2ai_noise_audit.py), into outputs/logs/unified_clean/
# PDVOICE_CLEAN_ALL=1 also cleans IPVS, NeuroVoz and MDVR-KCL the same way (mPower is not
# cleaned), into outputs/logs/unified_clean_all/, so transfer compares clean with clean.
CLEAN_ALL = os.environ.get("PDVOICE_CLEAN_ALL") == "1"
CLEAN = CLEAN_ALL or os.environ.get("PDVOICE_B2AI_CLEAN") == "1"
SILENT_DBFS = -55.0
OUT = LOGS / ("unified_clean_all" if CLEAN_ALL else "unified_clean" if CLEAN else "unified")
# PDVOICE_B2AI_SPEECH=rainbow-passage,picture-description keeps only those Bridge2AI speech
# tasks (task-ablation of the connected-speech model), into outputs/logs/unified_speech_<tasks>/
SPEECH_TASKS = [t for t in os.environ.get("PDVOICE_B2AI_SPEECH", "").split(",") if t]
if SPEECH_TASKS:
    OUT = LOGS / ("unified_speech_" + "+".join(t.split("-")[0] for t in SPEECH_TASKS))
SETS = {"mpower": ["vowel"], "ipvs": ["vowel", "speech"], "neurovoz": ["vowel", "ddk", "speech"],
        "mdvr_kcl": ["speech"], "bridge2ai": ["vowel", "ddk", "speech"]}
FAMILIES = {
    "cognitive impairment": ["cognitive_impairment"],
    "neurological (cognitive impairment, ALS)": ["cognitive_impairment",
                                                 "amyotrophic_lateral_sclerosis"],
    "voice disorders": ["benign_lesions", "laryngeal_dystonia", "muscle_tension_dysphonia",
                        "unilateral_vocal_fold_paralysis", "glottic_insufficiency",
                        "laryngeal_cancer", "laryngitis", "precancerous_lesions"],
    "mood": ["depression", "anxiety", "bipolar_disorder"],
    "respiratory": ["airway_stenosis", "unexplained_chronic_cough", "copd_and_asthma"],
}


# ------------------------------------------------------------------------------- data
def _mpower():
    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    df = pd.DataFrame({"subject_id": d["subject_id"].astype(str), "label": d["label"].astype(int)})
    ref = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L15.npz", allow_pickle=True)
    assert np.array_equal(ref["subject_id"].astype(str), df.subject_id.to_numpy()), \
        "mPower layer-11 cache is not row-aligned with the layer-15 cache"
    dem = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    hc = dem["healthCode"].astype(str)
    key = df.subject_id.str.replace("mpower_", "", regex=False)
    df["age"] = key.map(dict(zip(hc, dem["age"])))
    sex = key.map(dict(zip(hc, dem["gender"])))
    df["sex"] = sex.where(sex.isin(["Male", "Female"]))
    df["has_diag_year"] = key.map(dict(zip(hc, dem["diagnosis-year"].notna()))).fillna(False)
    return df, d["X"]


def _bridge2ai(task, groups=("pd", "control")):
    base = LOADERS["bridge2ai"]()
    df = base[base.task_kind == task].copy()
    df, _ = quality_control(df)
    df = df[df.group.isin(groups)].reset_index(drop=True)
    d = np.load(EMBEDDINGS / f"bridge2ai{'_clean' if CLEAN else ''}__{MODEL}__{task}__Lall.npz",
                allow_pickle=True)
    row = {p: i for i, p in enumerate(d["path"].astype(str))}
    if CLEAN:  # background noise removed (scripts/extract_bridge2ai_clean.py); silent recordings dropped
        loud = {p for p, r in zip(d["path"].astype(str), d["rms_db"]) if r >= SILENT_DBFS}
        df = df[df.path.isin(loud)].reset_index(drop=True)
    X = d["X"][[row[p] for p in df.path], LAYER]
    m = common_task_mask(df, "bridge2ai")
    if task == "speech" and SPEECH_TASKS:
        m &= df.task_code.isin(SPEECH_TASKS).to_numpy()
    return df[m].reset_index(drop=True), X[m]


def _clinical(ds, task):
    df = LOADERS[ds]()
    df = df[df.task_kind == task].copy()
    df, _ = quality_control(df)
    df = df.reset_index(drop=True)
    if CLEAN_ALL:  # background noise removed (scripts/extract_bridge2ai_clean.py); silent dropped
        d = np.load(EMBEDDINGS / f"{ds}_clean__{MODEL}__{task}__Lall.npz", allow_pickle=True)
        assert list(d["path"].astype(str)) == df.path.tolist(), f"{ds} {task}: cache order"
        loud = d["rms_db"] >= SILENT_DBFS
        df, X = df[loud].reset_index(drop=True), d["X"][loud, LAYER]
    else:
        # rows are matched to the manifest by recording path; a cache that stores no paths,
        # or lacks a requested recording, is refused (a row count cannot identify recordings)
        X = load_aligned(EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{LAYER}.npz", df.path)
    assert len(X) == len(df)
    m = common_task_mask(df, ds)
    return attach_age(df[m].reset_index(drop=True), ds), X[m]


def load(ds, task):
    if ds == "mpower":
        df, X = _mpower()
    elif ds == "bridge2ai":
        df, X = _bridge2ai(task)
    else:
        df, X = _clinical(ds, task)
    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "sex" not in df:
        df["sex"] = np.nan
    return df.reset_index(drop=True), np.asarray(X, dtype=np.float32)


def has_age(df):
    return df["age"].notna().mean() > 0.8


def has_sex(df):
    return df["sex"].notna().mean() > 0.8


# ------------------------------------------------------------------------- cohorts
def train_conditions(ds, df, seed):
    """Masks over df rows for the training cohort of each condition."""
    if ds == "mpower":
        base = df.index.isin(cap_per_subject(df, CAP, seed).index)
    else:
        base = np.ones(len(df), bool)
    out = {"uncorrected": base}
    sub = df[base]
    if ds == "mpower":
        lab = df.label
        out["confident"] = base & (((lab == 1) & df.has_diag_year) | ((lab == 0) & ~df.has_diag_year)).to_numpy()
    if has_age(sub):
        am = age_match_ids(sub, seed=seed)
        out["agematched"] = base & df.subject_id.isin(am).to_numpy()
        if ds == "mpower":
            # size control: as many patients and controls as the age-matched cohort, drawn at
            # random from the uncorrected cohort, so the age gap is kept and only the size
            # (and the class balance) changes
            s = sub.drop_duplicates("subject_id")
            n_pd = int(s[s.subject_id.isin(am)].label.sum())
            n_hc = int((~s[s.subject_id.isin(am)].label.astype(bool)).sum())
            rng = np.random.default_rng(seed)
            pick = set(rng.choice(np.sort(s.subject_id[s.label == 1].to_numpy()), n_pd, replace=False)) | \
                set(rng.choice(np.sort(s.subject_id[s.label == 0].to_numpy()), n_hc, replace=False))
            out["sizematched"] = base & df.subject_id.isin(pick).to_numpy()
    if has_sex(sub):
        ok = sub.dropna(subset=["age", "sex"])
        out["agesexmatched"] = base & df.subject_id.isin(
            demographic_match_ids(ok, on=("age", "sex"), seed=seed)).to_numpy()
    return out


def test_cohorts(df, seed):
    out = {"released": np.ones(len(df), bool)}
    if has_age(df):
        out["agematched"] = df.subject_id.isin(age_match_ids(df, seed=seed)).to_numpy()
    if has_sex(df):
        ok = df.dropna(subset=["age", "sex"])
        out["agesexmatched"] = df.subject_id.isin(
            demographic_match_ids(ok, on=("age", "sex"), seed=seed)).to_numpy()
    return out


# ---------------------------------------------------------------------- model + scoring
def _ensure_two_class_val(tr, va, y, groups, seed):
    """The shared rule (pdvoice.data.splits.ensure_two_class_val): both training and
    validation hold both classes; a split that already does is returned unchanged, and
    the stratified fallback this script always used is kept whenever it is valid.
    """
    return ensure_two_class_val(tr, va, y, groups, seed)


def fit_locked(X, y, groups, seed):
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed)
                  .split(X, y, groups))
    tr, va = _ensure_two_class_val(tr, va, y, np.asarray(groups), seed)
    head, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va], seed=seed)
    _, cal = select_best_by_brier(vlog, y[va])
    return head, cal


def predict(head, cal, X):
    dev = next(head.parameters()).device
    with torch.no_grad():
        lg = head(torch.tensor(X, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
    return cal.ranking(lg), cal.predict_proba(lg)


def subject_table(df, r, p=None):
    t = pd.DataFrame({"s": df.subject_id.to_numpy(), "y": df.label.to_numpy(), "r": r,
                      "p": r if p is None else p})
    return t.groupby("s").agg(y=("y", "first"), r=("r", "mean"), p=("p", "mean"))


def metrics(g):
    if g.y.nunique() < 2:
        return None
    ce = calibration_error(g.y.to_numpy(), g.p.to_numpy())
    return dict(n_pd=int(g.y.sum()), n_hc=int((g.y == 0).sum()),
                auroc=float(roc_auc_score(g.y, g.r)), ece=float(ce["ece"]),
                brier=float(np.mean((g.p - g.y) ** 2)))


def within_cv(df, X, seed):
    """5-fold participant-disjoint CV of the same head; out-of-fold ranking scores."""
    dfa = assign_folds(df.reset_index(drop=True), n_splits=5, seed=seed)
    y = dfa.label.to_numpy()
    parts = []
    for k in range(5):
        tr, va, te = train_val_test_indices(dfa, test_fold=k, seed=seed)
        tr, va = _ensure_two_class_val(np.asarray(tr), np.asarray(va), y,
                                       dfa.subject_id.to_numpy(), seed)
        head, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va], seed=seed)
        _, cal = select_best_by_brier(vlog, y[va])
        r, p = predict(head, cal, X[te])
        parts.append(pd.DataFrame({"s": dfa.subject_id.to_numpy()[te], "y": y[te], "r": r, "p": p}))
    return pd.concat(parts).groupby("s").agg(y=("y", "first"), r=("r", "mean"), p=("p", "mean"))


# ------------------------------------------------------------------------- experiments
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="within,exp1,exp2,exp3,spec")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--skip-mpower", action="store_true", help="debug: run without mPower")
    args = ap.parse_args()
    SEEDS[:] = [int(x) for x in args.seeds.split(",")]
    which = set(args.only.split(","))
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = {(ds, t): load(ds, t) for ds, ts in SETS.items() for t in ts
            if not (args.skip_mpower and ds == "mpower")}
    for (ds, t), (df, X) in data.items():
        print(f"loaded {ds:9s} {t:6s} {len(df):6,} rec {df.subject_id.nunique():5,} subj "
              f"age {'yes' if has_age(df) else 'no ':3s} sex {'yes' if has_sex(df) else 'no'}",
              flush=True)
    rows, scores = [], []

    def record(exp, src, stask, cond, seed, tgt, ttask, cohort, g):
        m = metrics(g)
        if m is None:
            return
        rows.append(dict(exp=exp, source=src, source_task=stask, condition=cond, seed=seed,
                         target=tgt, target_task=ttask, target_cohort=cohort, **m))
        scores.append(g.reset_index().assign(exp=exp, source=src, source_task=stask,
                                             condition=cond, seed=seed, target=tgt,
                                             target_task=ttask, target_cohort=cohort))

    # ---- WITHIN: every dataset x task x condition; age-only baseline alongside
    if "within" in which:
        for (ds, t), (df, X) in data.items():
            for seed in SEEDS:
                for cond, m in train_conditions(ds, df, seed).items():
                    sub, Xs = df[m].reset_index(drop=True), X[m]
                    if sub[sub.label == 1].subject_id.nunique() < 10 or \
                            sub[sub.label == 0].subject_id.nunique() < 10:
                        continue
                    g = within_cv(sub, Xs, seed)
                    record("within", ds, t, cond, seed, ds, t, cond, g)
                    if has_age(sub):
                        a = sub.groupby("subject_id").agg(y=("label", "first"), age=("age", "first")).dropna()
                        rows.append(dict(exp="within_age_only", source=ds, source_task=t,
                                         condition=cond, seed=seed, target=ds, target_task=t,
                                         target_cohort=cond, n_pd=int(a.y.sum()),
                                         n_hc=int((a.y == 0).sum()),
                                         auroc=float(roc_auc_score(a.y, a.age))))
            print(f"[{(time.time()-t0)/60:5.1f} min] within {ds} {t} done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "runs_within.csv", index=False)

    # ---- EXP1 (vowel, as proposed) and EXP2 (connected speech)
    plans = []
    if "exp1" in which:
        plans.append(("exp1", [("mpower", "vowel")],
                      [k for k in data if k[0] != "mpower"]))
    if "exp2" in which:
        sp = [k for k in data if k[1] == "speech"]
        plans.append(("exp2", sp, sp))
    if "exp3" in which:
        # the same design as Experiment 2 for the vowel and diadochokinesis, so that the
        # task can be compared with the training source held fixed (mPower is excluded:
        # it has vowels only and is Experiment 1's source)
        for task in ("vowel", "ddk"):
            ks = [k for k in data if k[1] == task and k[0] != "mpower"]
            plans.append(("exp3", ks, ks))
    for exp, sources, targets in plans:
        for (src, stask) in sources:
            sdf, SX = data[(src, stask)]
            for seed in SEEDS:
                for cond, m in train_conditions(src, sdf, seed).items():
                    sub = sdf[m]
                    head, cal = fit_locked(SX[m], sub.label.to_numpy(), sub.subject_id.to_numpy(), seed)
                    for (tgt, ttask) in targets:
                        if tgt == src:
                            continue
                        tdf, TX = data[(tgt, ttask)]
                        r, p = predict(head, cal, TX)
                        g_all = subject_table(tdf, r, p)
                        for cohort, cm in test_cohorts(tdf, seed).items():
                            keep = set(tdf.subject_id[cm])
                            record(exp, src, stask, cond, seed, tgt, ttask, cohort,
                                   g_all[g_all.index.isin(keep)])
                print(f"[{(time.time()-t0)/60:5.1f} min] {exp} {src} {stask} seed {seed} done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "runs_partial.csv", index=False)

    # ---- SPEC: Parkinson's disease against other disorders (Bridge2AI)
    if "spec" in which:
        fams = {}
        for fam, files in FAMILIES.items():
            ids = set()
            for f in files:                         # confirmed (gold-standard) diagnoses only
                ids |= confirmed_ids(pd.read_csv(
                    BRIDGE2AI_DIR / "phenotype" / "diagnosis" / f"{f}.tsv", sep="\t", dtype=str))
            fams[fam] = {f"b2ai_{i}" for i in ids}
        confirmed = set().union(*fams.values())
        # Country: every Canadian in the PD-vs-control cohort is a patient, and most of the
        # cognitive-impairment group is Canadian, so every comparison is made within one
        # country (matched on it, or restricted to it).
        for task in SETS["bridge2ai"]:
            df, X = _bridge2ai(task, groups=("pd", "pd_non_us", "control", "other"))
            df = df.copy(); df["age"] = pd.to_numeric(df["age"], errors="coerce")
            X = np.asarray(X, dtype=np.float32)
            pdm, ctl, oth = (df.group == "pd").to_numpy(), (df.group == "control").to_numpy(), \
                (df.group == "other").to_numpy()
            pd_ca = (df.group == "pd_non_us").to_numpy()
            pd_all = pdm | pd_ca
            us = (df.country == "USA").fillna(False).to_numpy(dtype=bool)
            for seed in SEEDS:
                # (a) direct classifiers: PD against each disorder family, matched on age and
                # country
                for fam, ids in {"all other disorders": confirmed, **fams}.items():
                    sel = pd_all | (oth & df.subject_id.isin(ids).to_numpy())
                    sub = df[sel].assign(label=pd_all[sel].astype(int)).reset_index(drop=True)
                    keep = sub.subject_id.isin(demographic_match_ids(
                        sub, on=("age", "country"), seed=seed)).to_numpy()
                    sub, Xs = sub[keep].reset_index(drop=True), X[sel][keep]
                    if min(sub[sub.label == 1].subject_id.nunique(),
                           sub[sub.label == 0].subject_id.nunique()) < 10:
                        continue
                    record("spec_direct", "bridge2ai", task, "agematched", seed, fam, task,
                           "agematched", within_cv(sub, Xs, seed))
                # (b) confusion: the US PD-vs-control model (age-matched) scores the other
                # groups and the Canadian patients, none of whom it saw; groups are compared
                # only within one country
                pc = df[pdm | ctl].assign(label=pdm[pdm | ctl].astype(int))
                keep = pc.subject_id.isin(age_match_ids(pc, seed=seed))
                T = pc[keep.to_numpy()].reset_index(drop=True)
                XT = X[pdm | ctl][keep.to_numpy()]
                oof = within_cv(T, XT, seed)                       # PD and controls, out of fold
                head, cal = fit_locked(XT, T.label.to_numpy(), T.subject_id.to_numpy(), seed)
                unseen = oth | pd_ca
                r_o, _ = predict(head, cal, X[unseen])
                go = pd.DataFrame({"s": df.subject_id[unseen].to_numpy(), "r": r_o}).groupby("s").r.mean()
                ca_ids = set(df.subject_id[~us])
                pdr, hcr = oof.r[oof.y == 1], oof.r[oof.y == 0]
                pdr_ca = go[go.index.isin(set(df.subject_id[pd_ca]))]
                res = {"PD vs control": (pdr, hcr)}
                others = set(df.subject_id[oth])        # never a PD patient on this side
                for fam, ids in {"all other disorders": confirmed, **fams}.items():
                    fo = go[go.index.isin(ids & others)]
                    fo_ca = fo[fo.index.isin(ca_ids)]
                    # both sides scored by the same locked model; the US patients and
                    # controls have out-of-fold scores only, so they are not compared here
                    res[f"PD vs {fam} (Canada)"] = (pdr_ca, fo_ca)
                for name, (pos, neg) in res.items():
                    if len(pos) < 10 or len(neg) < 10:
                        continue
                    rows.append(dict(exp="spec_confusion", source="bridge2ai", source_task=task,
                                     condition="agematched", seed=seed, target=name,
                                     target_task=task, target_cohort="agematched",
                                     n_pd=len(pos), n_hc=len(neg),
                                     auroc=float(roc_auc_score([1] * len(pos) + [0] * len(neg),
                                                               list(pos) + list(neg)))))
            print(f"[{(time.time()-t0)/60:5.1f} min] spec {task} done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / f"runs{'' if which == {'within', 'exp1', 'exp2', 'exp3', 'spec'} else '__' + '_'.join(sorted(which))}.csv",
             index=False)
    if scores:
        pd.concat(scores).to_csv(OUT / f"scores{'' if which == {'within', 'exp1', 'exp2', 'exp3', 'spec'} else '__' + '_'.join(sorted(which))}.csv.gz",
                                 index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min; {len(R)} rows", flush=True)


if __name__ == "__main__":
    main()
