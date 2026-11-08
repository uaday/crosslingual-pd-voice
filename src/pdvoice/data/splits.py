"""Subject-disjoint cross-validation splits.

The cardinal rule: no participant may appear in more than
one fold. We group by ``subject_id`` and stratify by ``label`` so class balance is
preserved across folds. A validation slice is carved from each training set the same
way, so calibration maps can be fit on val and locked before the test fold is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from pdvoice.config import N_FOLDS, SEED, VAL_FRACTION


def assign_folds(
    df: pd.DataFrame,
    n_splits: int = N_FOLDS,
    seed: int = SEED,
    label_col: str = "label",
    group_col: str = "subject_id",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``fold`` column (test-fold index per row).

    Rows with a missing label get fold ``-1`` (excluded from CV). Uses
    StratifiedGroupKFold so folds are subject-disjoint and label-balanced.
    """
    out = df.copy()
    out["fold"] = -1
    labelled = out[out[label_col].notna()]
    if labelled.empty:
        return out
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = labelled[label_col].astype(int)
    groups = labelled[group_col]
    for fold, (_, test_pos) in enumerate(sgkf.split(labelled, y, groups)):
        out.loc[labelled.index[test_pos], "fold"] = fold
    return out


def train_val_test_indices(
    df: pd.DataFrame,
    test_fold: int,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    label_col: str = "label",
    group_col: str = "subject_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a chosen test fold, return (train_idx, val_idx, test_idx) as index arrays.

    ``df`` must already have a ``fold`` column (see :func:`assign_folds`). The val
    slice is a subject-disjoint sample of the development set (all non-test folds).
    """
    if "fold" not in df.columns:
        raise ValueError("call assign_folds(df) before train_val_test_indices()")
    dev = df[(df["fold"] != test_fold) & (df["fold"] >= 0)]
    test_idx = df.index[df["fold"] == test_fold].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_pos, va_pos = next(gss.split(dev, dev[label_col], dev[group_col]))
    return dev.index[tr_pos].to_numpy(), dev.index[va_pos].to_numpy(), test_idx


def assert_disjoint(df: pd.DataFrame, *idx_groups: np.ndarray, group_col: str = "subject_id") -> None:
    """Raise if any subject appears in more than one of the given index groups."""
    seen: set = set()
    for idx in idx_groups:
        subjects = set(df.loc[idx, group_col])
        overlap = seen & subjects
        if overlap:
            raise AssertionError(f"subject leakage across splits: {sorted(overlap)[:5]}")
        seen |= subjects
