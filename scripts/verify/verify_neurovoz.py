"""Verify the within-NeuroVoz speech AUROC=0.95 is genuine PD signal, not a confound.

Checks:
  1. Confound distributions: Sex, Age, recording Date by PD vs HC (subject level).
  2. Label-permutation sanity: shuffle labels (subject-level) -> within-CV AUROC must
     fall to ~0.5 (proves the subject-disjoint CV plumbing does not leak).
  3. Confound-controlled re-runs: age-matched, and sex-stratified within-CV AUROC.
     If the score holds, the 0.95 is not driven by age/sex/batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pdvoice.calibration import select_best_by_brier
from pdvoice.cohort import age_match_ids
from pdvoice.config import EMBEDDINGS, MPOWER_DIR, SEED  # noqa: F401
from pdvoice.config import PD_DATASETS
from pdvoice.data import LOADERS
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auroc
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

MODEL, LAYER = "xls-r-300m", 15
META = PD_DATASETS / "neurovoz_v3" / "data" / "metadata"


def load_speech():
    df = LOADERS["neurovoz"]()
    df = df[df["task_kind"] == "speech"].copy()
    df, _ = quality_control(df); df = df.reset_index(drop=True)
    X = np.load(EMBEDDINGS / f"neurovoz__{MODEL}__speech__L{LAYER}.npz")["X"].astype(np.float32)
    assert X.shape[0] == len(df)
    return df, X


def attach_meta(df):
    hc = pd.read_csv(META / "metadata_hc.csv"); hc["label"] = 0
    pdd = pd.read_csv(META / "metadata_pd.csv"); pdd["label"] = 1
    m = pd.concat([hc, pdd], ignore_index=True)
    m["key"] = m["label"].astype(str) + "_" + m["ID"].astype(float).astype(int).astype(str)
    sex = dict(zip(m["key"], m["Sex"])); age = dict(zip(m["key"], m["Age"]))
    date = dict(zip(m["key"], m["Date "].astype(str)))
    num = df["subject_id"].str.extract(r"neurovoz_(hc|pd)_(\d+)")
    key = df["label"].astype(int).astype(str) + "_" + num[1].astype(int).astype(str)
    out = df.copy()
    out["sex"] = key.map(sex); out["age"] = key.map(age); out["date"] = key.map(date)
    return out


def within_cv_auroc(df, X, y, seed=SEED):
    df = df.reset_index(drop=True)
    dfa = assign_folds(df.assign(label=y), n_splits=5, seed=seed)
    oof = []
    for k in range(5):
        tr, va, te = train_val_test_indices(dfa, test_fold=k)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        m, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va])
        with torch.no_grad():
            dev = next(m.parameters()).device
            tl = m(torch.tensor(X[te], dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        _, cal = select_best_by_brier(vlog, y[va])
        oof.append(pd.DataFrame({"s": dfa.loc[te, "subject_id"].to_numpy(),
                                 "y": y[te], "p": cal.predict_proba(tl)}))
    g = pd.concat(oof).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    return auroc(g["y"], g["p"])


def subj_table(df):
    return df.drop_duplicates("subject_id")[["subject_id", "label", "sex", "age", "date"]]


def main():
    df, X = load_speech()
    df = attach_meta(df)
    y = df["label"].astype(int).to_numpy()
    subj = subj_table(df)

    print("=== NeuroVoz speech confound verification (within-CV AUROC baseline = 0.952) ===\n")

    # 1. confound distributions
    print("[1] Confounds by group (subject level):")
    for lab, name in [(1, "PD"), (0, "HC")]:
        s = subj[subj.label == lab]
        sexc = s["sex"].value_counts(dropna=False).to_dict()
        print(f"  {name}: n={len(s)}  age mean={s['age'].mean():.1f}  "
              f"sex(1/0)={sexc.get(1.0,sexc.get(1,0))}/{sexc.get(0.0,sexc.get(0,0))}  "
              f"dates {s['date'].min()}..{s['date'].max()}")
    # date overlap
    pd_dates, hc_dates = set(subj[subj.label==1]["date"]), set(subj[subj.label==0]["date"])
    print(f"  date overlap: {len(pd_dates & hc_dates)} shared months "
          f"(PD {len(pd_dates)} / HC {len(hc_dates)} distinct)")

    # 2. label-permutation sanity
    rng = np.random.default_rng(SEED)
    sids = subj["subject_id"].to_numpy()
    perm = dict(zip(sids, rng.permutation(subj["label"].to_numpy())))
    y_shuf = df["subject_id"].map(perm).astype(int).to_numpy()
    a_shuf = within_cv_auroc(df, X, y_shuf)
    print(f"\n[2] Label-permutation sanity: shuffled-label AUROC = {a_shuf:.3f}  (expect ~0.5)")

    # 3a. age-matched (5yr bins)
    keep = age_match_ids(subj, seed=SEED)
    mask = df["subject_id"].isin(keep).to_numpy()
    npd = subj[(subj.label==1) & subj.subject_id.isin(keep)].shape[0]
    nhc = subj[(subj.label==0) & subj.subject_id.isin(keep)].shape[0]
    a_age = within_cv_auroc(df[mask], X[mask], y[mask])
    print(f"\n[3a] Age-matched within-CV AUROC = {a_age:.3f}  (PD {npd} / HC {nhc} subj)")

    # 3b. sex-stratified
    for sx, nm in [(1, "male(1)"), (0, "female(0)")]:
        m = (df["sex"] == sx).to_numpy()
        if m.sum() and len(np.unique(y[m])) == 2:
            a = within_cv_auroc(df[m], X[m], y[m])
            ns = subj[(subj.sex==sx)].groupby("label").size().to_dict()
            print(f"[3b] {nm}-only within-CV AUROC = {a:.3f}  (PD {ns.get(1,0)} / HC {ns.get(0,0)} subj)")
    print("\n(If age-matched & sex-stratified stay high and shuffled ~0.5 -> 0.95 is genuine.)")


if __name__ == "__main__":
    main()
