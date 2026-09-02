"""Label & confound sensitivity check for mPower (answers: is the transfer failure
caused by self-report label noise, or by the PD/HC age confound?).

Reuses cached embeddings (no re-extraction). For three definitions of the training
labels it reports (a) within-mPower 5-fold subject-disjoint AUROC and (b) transfer
AUROC to the externals at the transfer layer:

    baseline   : all self-report labels (as-is)
    confident  : PD only if diagnosis-year present; drop HC who gave a diagnosis-year
    age50      : keep only PD and HC aged >= 50  (removes the young-control shortcut)

Run:  python scripts/label_sensitivity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pdvoice.calibration import select_best_by_brier
from pdvoice.config import EMBEDDINGS, MPOWER_DIR, SEED, VAL_FRACTION
from pdvoice.data import LOADERS
from pdvoice.data.mpower import load_mpower
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auroc
from pdvoice.eval.metrics import youden_threshold
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control
from sklearn.model_selection import GroupShuffleSplit

MODEL = "xls-r-300m"
WITHIN_LAYER = None   # last layer (Layer-1 winner, 0.839)
TRANSFER_LAYER = 15   # best transfer layer so far
CAP = 3
EXTERNALS = [("ipvs", "vowel"), ("neurovoz", "vowel"), ("mdvr_kcl", "speech")]


def load_cache(path, n):
    X = np.load(path)["X"]
    assert X.shape[0] == n, f"{path.name}: {X.shape[0]} != {n} (df/cache misaligned)"
    return X.astype(np.float32)


def mpower_df():
    df = load_mpower()
    lab = df[df["label"].isin([0, 1])].sample(frac=1.0, random_state=SEED)
    df = lab[lab.groupby("subject_id").cumcount().to_numpy() < CAP].reset_index(drop=True)
    df, _ = quality_control(df)
    return df.reset_index(drop=True)


def attach_demographics(df):
    d = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    d["hc"] = d["healthCode"].astype(str)
    age = dict(zip(d["hc"], d["age"]))
    diag = dict(zip(d["hc"], d["diagnosis-year"].notna()))
    hc = df["subject_id"].str.replace("mpower_", "", regex=False)
    df = df.copy()
    df["age"] = hc.map(age)
    df["has_diag_year"] = hc.map(diag).fillna(False)
    return df


def masks(df):
    lab = df["label"].astype(int)
    baseline = pd.Series(True, index=df.index)
    confident = ((lab == 1) & df["has_diag_year"]) | ((lab == 0) & ~df["has_diag_year"])
    age50 = df["age"] >= 50
    return {"baseline": baseline, "confident": confident, "age50": age50}


def within_auroc(dfx, X, y):
    """Subject-level out-of-fold AUROC, 5-fold subject-disjoint."""
    dfx = assign_folds(dfx.reset_index(drop=True), n_splits=5)
    Xr = X  # X already row-aligned to dfx (caller passes matching subset)
    oof = []
    for k in range(5):
        tr, va, te = train_val_test_indices(dfx, test_fold=k)
        m, vlog = fit_mlp_head(Xr[tr], y[tr], Xr[va], y[va])
        with torch.no_grad():
            dev = next(m.parameters()).device
            tl = m(torch.tensor(Xr[te], dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        _, cal = select_best_by_brier(vlog, y[va])
        oof.append(pd.DataFrame({"s": dfx.loc[te, "subject_id"].to_numpy(),
                                 "y": y[te], "p": cal.predict_proba(tl)}))
    s = pd.concat(oof).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    return auroc(s["y"], s["p"])


def transfer_aurocs(dfx, X, y, ext_cache):
    """Train head once on the subset, lock, score each external. Returns {ds: auroc}."""
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr, va = next(gss.split(X, y, dfx["subject_id"]))
    m, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va])
    _, cal = select_best_by_brier(vlog, y[va])
    dev = next(m.parameters()).device

    def score(Xe):
        with torch.no_grad():
            lg = m(torch.tensor(Xe, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        return cal.predict_proba(lg)

    out = {}
    for ds, edf, EX in ext_cache:
        s = pd.DataFrame({"s": edf["subject_id"], "y": edf["label"].astype(int), "p": score(EX)})
        sg = s.groupby("s").agg(y=("y", "first"), p=("p", "mean"))
        out[ds] = auroc(sg["y"], sg["p"])
    return out


def main():
    print("=== mPower label & confound sensitivity (XLS-R) ===")
    df = attach_demographics(mpower_df())
    n = len(df)
    Xlast = load_cache(EMBEDDINGS / f"mpower_pilot__{MODEL}__cap{CAP}__L{WITHIN_LAYER}__seed{SEED}.npz", n)
    Xl15 = load_cache(EMBEDDINGS / f"mpower_pilot__{MODEL}__cap{CAP}__L{TRANSFER_LAYER}__seed{SEED}.npz", n)

    # preload externals at the transfer layer (cached)
    ext_cache = []
    for ds, task in EXTERNALS:
        edf = LOADERS[ds]()
        edf = edf[edf["task_kind"] == task].copy()
        edf, _ = quality_control(edf)
        edf = edf.reset_index(drop=True)
        EX = load_cache(EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{TRANSFER_LAYER}.npz", len(edf))
        ext_cache.append((ds, edf, EX))

    print(f"\n{'scenario':<12}{'PD/HC subj':<16}{'within(last)':<14}"
          f"{'ipvs':<8}{'neurovoz':<10}{'mdvr_kcl':<10}")
    for name, mask in masks(df).items():
        sub = df[mask].reset_index(drop=True)
        idx = df.index[mask].to_numpy()
        y = sub["label"].astype(int).to_numpy()
        npd = sub[sub.label == 1].subject_id.nunique()
        nhc = sub[sub.label == 0].subject_id.nunique()
        if npd < 10 or nhc < 10:
            print(f"{name:<12}{f'{npd}/{nhc}':<16}(too few subjects, skipped)")
            continue
        w = within_auroc(sub, Xlast[idx], y)
        t = transfer_aurocs(sub, Xl15[idx], y, ext_cache)
        print(f"{name:<12}{f'{npd}/{nhc}':<16}{w:<14.3f}"
              f"{t['ipvs']:<8.3f}{t['neurovoz']:<10.3f}{t['mdvr_kcl']:<10.3f}")

    # report the age gap that motivates the age50 row
    print(f"\n  age: PD mean {df[df.label==1].age.mean():.1f}  HC mean {df[df.label==0].age.mean():.1f}")
    print("  (within = 5-fold subject-disjoint on mPower, last layer; externals = transfer at L15)")


if __name__ == "__main__":
    main()
