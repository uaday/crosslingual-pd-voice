"""Standard per-recording manifest schema shared by all dataset loaders.

Every loader returns a pandas DataFrame with (at least) these columns, so the rest
of the pipeline (splits, features, eval) is dataset-agnostic:

    dataset     str    e.g. "mpower", "ipvs", "mdvr_kcl"
    subject_id  str    speaker/participant id  -> the GROUP key for splits
    path        str    absolute path to the audio file
    label       int    1 = PD, 0 = HC, <NA> if unknown
    task_kind   str    one of TASK_KINDS  -> drives the vowel-test / speech-test split
    task_code   str    raw task label from the source (e.g. "VA", "ReadText")
    language    str    "en" | "it" | "es" | "sk"
    native_sr   int    original sample rate (Hz)

Loaders may append extra columns (e.g. hy, updrs) after the standard ones.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

MANIFEST_COLUMNS = [
    "dataset",
    "subject_id",
    "path",
    "label",
    "task_kind",
    "task_code",
    "language",
    "native_sr",
]
TASK_KINDS = {"vowel", "speech", "ddk", "other"}


def new_manifest(rows: Iterable[dict]) -> pd.DataFrame:
    """Build a validated manifest DataFrame from an iterable of row dicts."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    extras = [c for c in df.columns if c not in MANIFEST_COLUMNS]
    return validate_manifest(df[MANIFEST_COLUMNS + extras])


def validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns, task kinds and label values; return a clean copy."""
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    bad_kinds = set(df["task_kind"].dropna().unique()) - TASK_KINDS
    if bad_kinds:
        raise ValueError(f"unknown task_kind(s): {sorted(bad_kinds)} (allowed: {sorted(TASK_KINDS)})")
    labels = set(pd.unique(df["label"].dropna()))
    if not labels <= {0, 1}:
        raise ValueError(f"label must be 0, 1 or NA; got {sorted(labels)}")
    return df.reset_index(drop=True)
