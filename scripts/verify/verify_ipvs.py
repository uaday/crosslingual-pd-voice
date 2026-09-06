"""Verify / explain the within-IPVS speech AUROC = 1.000 (suspected confound).

IPVS controls = 15 YOUNG HC + 22 ELDERLY HC; PD = 28 (elderly). The subject_id tag
(yhc / ehc / pd) identifies the group. Tests:
  1. Label-permutation sanity: shuffled labels -> within-CV AUROC ~0.5 (no plumbing leak).
  2. Drop YOUNG HC: within-CV on elderly-HC vs PD only (age-comparable). If 1.0 falls,
     the young controls were an age shortcut.
  3. Age-proxy: can the embeddings separate YOUNG-HC vs ELDERLY-HC (both healthy)?
     High AUROC => embeddings strongly encode age/maturity (the confound mechanism).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pdvoice.calibration import select_best_by_brier
from pdvoice.config import EMBEDDINGS, SEED
from pdvoice.data import LOADERS
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auroc
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

MODEL, LAYER = "xls-r-300m", 15


def load_speech():
    df = LOADERS["ipvs"]()
    df = df[df["task_kind"] == "speech"].copy()
    df, _ = quality_control(df); df = df.reset_index(drop=True)
    X = np.load(EMBEDDINGS / f"ipvs__{MODEL}__speech__L{LAYER}.npz")["X"].astype(np.float32)
    assert X.shape[0] == len(df), f"{X.shape[0]} != {len(df)}"
    df["grp"] = df["subject_id"].str.extract(r"ipvs_([a-z]+)_")
    return df, X


def within_cv(df, X, y, seed=SEED):
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
        oof.append(pd.DataFrame({"s": dfa.loc[te, "subject_id"].to_numpy(), "y": y[te],
                                 "p": cal.predict_proba(tl)}))
    g = pd.concat(oof).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    return auroc(g["y"], g["p"]), g


def nsub(df, mask):
    s = df[mask].drop_duplicates("subject_id")
    return {g: int((s["grp"] == g).sum()) for g in ["yhc", "ehc", "pd"]}


def main():
    df, X = load_speech()
    y = df["label"].astype(int).to_numpy()
    print("=== IPVS speech verification (within-CV baseline = 1.000) ===")
    print(f"  groups (subjects): {nsub(df, np.ones(len(df), bool))}\n")

    a_all, _ = within_cv(df, X, y)
    print(f"[baseline] all HC (young+elderly) vs PD: AUROC = {a_all:.3f}")

    # 1. permutation
    rng = np.random.default_rng(SEED)
    subj = df.drop_duplicates("subject_id")[["subject_id", "label"]]
    perm = dict(zip(subj["subject_id"], rng.permutation(subj["label"].to_numpy())))
    y_sh = df["subject_id"].map(perm).astype(int).to_numpy()
    a_sh, _ = within_cv(df, X, y_sh)
    print(f"[1 permutation] shuffled-label AUROC = {a_sh:.3f}  (expect ~0.5)")

    # 2. drop young HC -> elderly-HC vs PD
    m = (df["grp"] != "yhc").to_numpy()
    a_eld, _ = within_cv(df[m], X[m], y[m])
    print(f"[2 drop young HC] elderly-HC vs PD: AUROC = {a_eld:.3f}  "
          f"(subjects {nsub(df, m)})")

    # 3. age proxy: young-HC vs elderly-HC (both healthy)
    hc = df["label"] == 0
    y_age = (df.loc[hc, "grp"] == "ehc").astype(int).to_numpy()
    a_age, _ = within_cv(df[hc.to_numpy()], X[hc.to_numpy()], y_age)
    print(f"[3 age proxy] young-HC vs elderly-HC (embeddings): AUROC = {a_age:.3f}  "
          f"(1.0 => embeddings strongly encode age)")

    print("\nReading: if [2] << 1.0 and [3] high => the 1.0 was an age shortcut from young controls.")


if __name__ == "__main__":
    main()
