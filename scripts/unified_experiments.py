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
  SPEC    Parkinson's disease against other disorders (the only dataset that has them is
          Bridge2AI): direct classifiers, and the PD-vs-control model scoring other groups

Conditions (applied the same way everywhere they are possible)
  uncorrected    data as released
  confident      mPower only: patients with a diagnosis year, controls without
  agematched     training cohort frequency-matched on age in 5-year bins
  agesexmatched  matched jointly on age and sex (mPower and Bridge2AI record sex)
Test cohorts: as released, age-matched (where age is recorded), age- and sex-matched (where
sex is recorded).

Outputs (outputs/logs/unified/):
  runs.csv     one row per (experiment, source, condition, seed, target, cohort)
  scores.csv.gz per-participant ranking scores, for seed-pooled bootstrap intervals
Run:  .venv\\Scripts\\python.exe -u scripts\\unified_experiments.py
"""
from __future__ import annotations

import argparse
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
from pdvoice.data.demographics import attach_age
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.metrics import calibration_error
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

MODEL, LAYER, CAP = "xls-r-300m", 11, 3
SEEDS = [42, 43, 44, 45, 46]
OUT = LOGS / "unified"
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
    d = np.load(EMBEDDINGS / f"bridge2ai__{MODEL}__{task}__Lall.npz", allow_pickle=True)
    row = {p: i for i, p in enumerate(d["path"].astype(str))}
    X = d["X"][[row[p] for p in df.path], LAYER]
    return df, X


def _clinical(ds, task):
    df = LOADERS[ds]()
    df = df[df.task_kind == task].copy()
    df, _ = quality_control(df)
    df = df.reset_index(drop=True)
    d = np.load(EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{LAYER}.npz", allow_pickle=True)
    if "path" in d:
        assert list(d["path"].astype(str)) == df.path.tolist(), f"{ds} {task}: cache order"
    assert len(d["X"]) == len(df)
    return attach_age(df, ds), d["X"]


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
        out["agematched"] = base & df.subject_id.isin(age_match_ids(sub, seed=seed)).to_numpy()
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
    """Small datasets can leave a validation split with one class, which calibration cannot
    fit. Only then, redraw it from the training part with a stratified participant split;
    every split that already contains both classes is left exactly as it was."""
    if len(np.unique(y[va])) == 2:
        return tr, va
    from sklearn.model_selection import StratifiedGroupKFold
    pool = np.concatenate([tr, va])
    k = max(2, int(round(1 / VAL_FRACTION)))
    a, b = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed)
                .split(pool, y[pool], groups[pool]))
    return pool[a], pool[b]


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
    ap.add_argument("--only", default="within,exp1,exp2,spec")
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
            for f in files:
                ids |= set(pd.read_csv(BRIDGE2AI_DIR / "phenotype" / "diagnosis" / f"{f}.tsv",
                                       sep="\t", dtype=str).participant_id)
            fams[fam] = {f"b2ai_{i}" for i in ids}
        for task in SETS["bridge2ai"]:
            df, X = _bridge2ai(task, groups=("pd", "control", "other"))
            df = df.copy(); df["age"] = pd.to_numeric(df["age"], errors="coerce")
            X = np.asarray(X, dtype=np.float32)
            pdm, ctl, oth = (df.group == "pd").to_numpy(), (df.group == "control").to_numpy(), \
                (df.group == "other").to_numpy()
            for seed in SEEDS:
                # (a) direct classifiers: PD against each disorder family, age-matched
                for fam, ids in {"all other disorders": set(df.subject_id[oth]), **fams}.items():
                    sel = pdm | (oth & df.subject_id.isin(ids).to_numpy())
                    sub = df[sel].assign(label=pdm[sel].astype(int)).reset_index(drop=True)
                    keep = sub.subject_id.isin(age_match_ids(sub, seed=seed)).to_numpy()
                    sub, Xs = sub[keep].reset_index(drop=True), X[sel][keep]
                    if min(sub[sub.label == 1].subject_id.nunique(),
                           sub[sub.label == 0].subject_id.nunique()) < 10:
                        continue
                    record("spec_direct", "bridge2ai", task, "agematched", seed, fam, task,
                           "agematched", within_cv(sub, Xs, seed))
                # (b) confusion: PD-vs-control model (age-matched) scores the other groups
                pc = df[pdm | ctl].assign(label=pdm[pdm | ctl].astype(int))
                keep = pc.subject_id.isin(age_match_ids(pc, seed=seed))
                T = pc[keep.to_numpy()].reset_index(drop=True)
                XT = X[pdm | ctl][keep.to_numpy()]
                oof = within_cv(T, XT, seed)                       # PD and controls, out of fold
                head, cal = fit_locked(XT, T.label.to_numpy(), T.subject_id.to_numpy(), seed)
                r_o, _ = predict(head, cal, X[oth])
                go = pd.DataFrame({"s": df.subject_id[oth].to_numpy(), "r": r_o}).groupby("s").r.mean()
                pdr, hcr = oof.r[oof.y == 1], oof.r[oof.y == 0]
                res = {"PD vs control": (pdr, hcr)}
                for fam, ids in {"all other disorders": set(go.index), **fams}.items():
                    fo = go[go.index.isin(ids)]
                    if len(fo) < 10:
                        continue
                    res[f"PD vs {fam}"] = (pdr, fo)
                    res[f"{fam} vs control"] = (fo, hcr)
                # cognitive impairment matched to controls on age directly
                ci = df[oth & df.subject_id.isin(fams["cognitive impairment"]).to_numpy()]
                cc = pd.concat([ci.assign(label=1), df[ctl].assign(label=0)])
                cm = set(age_match_ids(cc, seed=seed))
                res["cognitive impairment vs control (age-matched to each other)"] = (
                    go[go.index.isin(cm & set(ci.subject_id))],
                    hcr[hcr.index.isin(cm)])
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
    R.to_csv(OUT / f"runs{'' if which == {'within', 'exp1', 'exp2', 'spec'} else '__' + '_'.join(sorted(which))}.csv",
             index=False)
    if scores:
        pd.concat(scores).to_csv(OUT / f"scores{'' if which == {'within', 'exp1', 'exp2', 'spec'} else '__' + '_'.join(sorted(which))}.csv.gz",
                                 index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min; {len(R)} rows", flush=True)


if __name__ == "__main__":
    main()
