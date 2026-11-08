"""Loader for the mPower voice arm (training set).

Reads the manifest produced by ``pd-datasets/mpower/download_mpower.py`` +
``convert_to_wav.py`` and keeps only the sustained-vowel phonation ("audio"),
dropping the countdown recording. 44.1 kHz, sustained /a/.

LABELS: mPower's PD label is SELF-REPORT and lives in the Demographics Survey
table (keyed by healthCode), which is a separate Synapse download and NOT in the
audio manifest. Pass ``demographics_csv`` (with columns healthCode +
professional-diagnosis) to attach labels; otherwise label is left as <NA>.
mPower diagnoses are self-reported, which is a validity threat.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pdvoice.config import MPOWER_DIR
from pdvoice.data.base import new_manifest


def _local_wav(record_id: str) -> str | None:
    """Reconstruct the converted wav path (robust to the stale absolute paths
    stored in the sample manifest)."""
    p = MPOWER_DIR / "wav" / f"{record_id}__audio.wav"
    return str(p.resolve()) if p.exists() else None


_NOT_A_PATH = {"", "nan", "none", "null", "na", "<na>"}


def _fallback_wav_path(value) -> str | None:
    """Last-resort path from the manifest's ``wavPath`` cell, or None.

    NOTE: the same truthy-string trap as ``_to_binary``. ``str(nan)`` is the
    non-empty string ``"nan"``, so a guard like ``if str(r.get("wavPath", ""))``
    passes for MISSING files and yields 7,410 phantom rows whose path is literally
    "nan" (11.4% of the manifest, and PD-enriched 2.7x).
    Reject the placeholder tokens explicitly, and require the file to exist.
    """
    if value is None or not isinstance(value, str):
        return None
    s = value.strip()
    if s.lower() in _NOT_A_PATH:
        return None
    return s if Path(s).exists() else None


_TRUE_TOKENS = {"true", "1", "1.0", "yes", "pd", "y", "t"}
_FALSE_TOKENS = {"false", "0", "0.0", "no", "hc", "control", "n", "f"}


def _to_binary(value) -> int | None:
    """Map a professional-diagnosis cell to 1 (PD) / 0 (non-PD) / None (declined).

    NOTE: the column arrives as the strings ``"true"``/``"false"`` (with blanks for
    declined). ``astype(bool)`` must NOT be used — every non-empty string is truthy,
    so ``"false"`` would wrongly become 1. Parse the token explicitly instead.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if s in _TRUE_TOKENS:
        return 1
    if s in _FALSE_TOKENS:
        return 0
    return None  # unrecognised / declined -> unlabelled


def _find_diagnosis_column(columns) -> str | None:
    """Pick the PD-status column, preferring 'professional-diagnosis'.

    The Demographics Survey has TWO diagnosis-ish columns: 'diagnosis-year'
    (a year, NOT a label) and 'professional-diagnosis' (the label). A naive
    'diagnosis' substring match grabs 'diagnosis-year' first and yields no labels,
    so match the real column explicitly and never accept a '*-year' column.
    """
    cols = list(columns)

    def norm(c):
        return str(c).lower().replace("_", "-").strip()

    for c in cols:  # exact
        if norm(c) == "professional-diagnosis":
            return c
    for c in cols:  # professional + diagnosis, any order
        if "professional" in norm(c) and "diagnosis" in norm(c):
            return c
    for c in cols:  # any diagnosis column that is not a year
        if "diagnosis" in norm(c) and "year" not in norm(c):
            return c
    return None


def _load_labels(demographics_csv) -> dict:
    """healthCode -> 1/0 from the Demographics Survey (self-report PD)."""
    if demographics_csv is None:
        warnings.warn(
            "mPower labels not attached: no demographics_csv given. Download the "
            "Demographics Survey table (professional-diagnosis by healthCode) and "
            "pass it to load_mpower(demographics_csv=...).",
            stacklevel=2,
        )
        return {}
    demo = pd.read_csv(demographics_csv)
    diag_col = _find_diagnosis_column(demo.columns)
    if diag_col is None or "healthCode" not in demo.columns:
        raise ValueError("demographics_csv needs 'healthCode' and a diagnosis column")
    labels: dict = {}
    for hc, v in zip(demo["healthCode"], demo[diag_col]):
        b = _to_binary(v)
        if b is not None:
            labels[hc] = b
    return labels


def _default_demographics_csv():
    """Locate the Demographics Survey table in mpower/ if present (labels source)."""
    for name in ("Demographics Survey.csv", "demographics_survey.csv", "demographics.csv"):
        p = MPOWER_DIR / name
        if p.exists():
            return p
    return None


def load_mpower(manifest_csv=None, demographics_csv=None):
    """Return a standard manifest DataFrame for the mPower sustained-vowel task.

    If ``demographics_csv`` is not given, auto-discovers ``Demographics Survey.csv``
    in the mpower/ directory (the labels table). Recordings whose participant has no
    label are kept with label ``<NA>`` and excluded from CV by ``assign_folds``.
    """
    manifest_csv = Path(manifest_csv) if manifest_csv else MPOWER_DIR / "audio_file_manifest.csv"
    if demographics_csv is None:
        demographics_csv = _default_demographics_csv()
    df = pd.read_csv(manifest_csv)
    # keep the phonation recording only (column name contains 'audio_audio')
    df = df[df["column"].astype(str).str.contains("audio_audio", na=False)].copy()

    labels = _load_labels(demographics_csv)
    rows = []
    n_missing = 0
    for _, r in df.iterrows():
        record_id, health_code = r["recordId"], r["healthCode"]
        path = _local_wav(record_id) or _fallback_wav_path(r.get("wavPath"))
        if not path:
            n_missing += 1
            continue
        rows.append(
            {
                "dataset": "mpower",
                "subject_id": f"mpower_{health_code}",
                "path": path,
                "label": labels.get(health_code, np.nan),
                "task_kind": "vowel",
                "task_code": "sustained_a",
                "language": "en",
                "native_sr": 44100,
            }
        )
    if n_missing:
        warnings.warn(
            f"mPower: dropped {n_missing:,} manifest rows with no audio file on disk "
            f"({100 * n_missing / max(1, len(df)):.1f}% of the phonation manifest). "
            "These are PD-enriched, so the drop is not class-neutral.",
            stacklevel=2,
        )
    return new_manifest(rows)
