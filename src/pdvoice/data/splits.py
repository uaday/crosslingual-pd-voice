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
    tr_pos, va_pos = ensure_two_class_val(tr_pos, va_pos, dev[label_col].to_numpy(),
                                          dev[group_col].to_numpy(), seed, val_fraction)
    return dev.index[tr_pos].to_numpy(), dev.index[va_pos].to_numpy(), test_idx


def _two_class_split(tr, va, y, groups) -> bool:
    """Both partitions hold both classes and share no participant."""
    return (len(np.unique(y[tr])) == 2 and len(np.unique(y[va])) == 2
            and not set(groups[tr]) & set(groups[va]))


def ensure_two_class_val(tr, va, y, groups, seed: int = SEED,
                         val_fraction: float = VAL_FRACTION):
    """Guarantee both classes in the training AND the validation split.

    A small cohort can leave GroupShuffleSplit's validation slice with one class, which
    calibration cannot be fitted on (a synthetic balanced 20-person cohort did so in 93 of
    500 seed/fold draws), or, with very few participants in one class, leave training
    without that class. The rule, in order:

    1. a split that already holds both classes on both sides is returned unchanged;
    2. otherwise the stratified participant split unified_experiments.py has always used
       is tried, and kept if it is valid (so every earlier result is reproduced);
    3. otherwise a participant-level stratified split is built directly: in each class,
       ``round(val_fraction * n)`` participants (at least one, at most n - 1) go to
       validation, so both sides hold both classes. Every recording of a participant
       stays on one side, and the original row indices are returned.

    Raises ValueError when no valid split exists: one class only, or a class with fewer
    than two participants (training and validation cannot both contain it).
    """
    tr, va = np.asarray(tr), np.asarray(va)
    y, groups = np.asarray(y), np.asarray(groups)
    if _two_class_split(tr, va, y, groups):
        return tr, va
    pool = np.concatenate([tr, va])
    if len(np.unique(y[pool])) < 2:
        raise ValueError("the development rows hold only one class")
    people = pd.DataFrame({"g": groups[pool], "y": y[pool]}).groupby("g", sort=True).y
    if (people.nunique() > 1).any():
        raise ValueError("a participant has recordings with both labels")
    per_class = people.first()
    counts = per_class.value_counts()
    if counts.min() < 2:
        raise ValueError(
            f"a class has {int(counts.min())} participant(s); training and validation "
            "cannot both contain it")

    k = max(2, int(round(1 / val_fraction)))
    try:  # the historical fallback, kept whenever its split is valid
        a, b = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed)
                    .split(pool, y[pool], groups[pool]))
        if _two_class_split(pool[a], pool[b], y, groups):
            return pool[a], pool[b]
    except ValueError:  # fewer participants than folds
        pass

    rng = np.random.default_rng(seed)
    val_people = []
    for _, members in per_class.groupby(per_class):
        ids = members.index.to_numpy()
        n_val = min(max(1, int(round(val_fraction * len(ids)))), len(ids) - 1)
        val_people.extend(rng.permutation(ids)[:n_val])
    in_val = np.isin(groups[pool], np.asarray(val_people, dtype=groups.dtype))
    tr2, va2 = pool[~in_val], pool[in_val]
    assert _two_class_split(tr2, va2, y, groups)
    return tr2, va2


def assert_disjoint(df: pd.DataFrame, *idx_groups: np.ndarray, group_col: str = "subject_id") -> None:
    """Raise if any subject appears in more than one of the given index groups."""
    seen: set = set()
    for idx in idx_groups:
        subjects = set(df.loc[idx, group_col])
        overlap = seen & subjects
        if overlap:
            raise AssertionError(f"subject leakage across splits: {sorted(overlap)[:5]}")
        seen |= subjects
