"""Loader for the mPower voice arm (training set).

Reads the manifest produced by ``pd-datasets/mpower/download_mpower.py`` +
``convert_to_wav.py`` and keeps only the sustained-vowel phonation ("audio"),
dropping the countdown recording. 44.1 kHz, sustained /a/.

LABELS: mPower's PD label is SELF-REPORT and lives in the Demographics Survey
table (keyed by healthCode), which is a separate Synapse download and NOT in the
audio manifest. Pass ``demographics_csv`` (with columns healthCode +
professional-diagnosis) to attach labels; otherwise label is left as <NA>.
See PROJECT_CONTEXT.md §3.2 (self-report validity threat).
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
    stored in the sample manifest — see PROJECT_CONTEXT.md §8)."""
    p = MPOWER_DIR / "wav" / f"{record_id}__audio.wav"
    return str(p.resolve()) if p.exists() else None


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
    diag_col = next(
        (c for c in demo.columns if "diagnosis" in c.lower() or "professional" in c.lower()),
        None,
    )
    if diag_col is None or "healthCode" not in demo.columns:
        raise ValueError("demographics_csv needs 'healthCode' and a diagnosis column")
    demo = demo.dropna(subset=[diag_col])
    return {
        hc: int(bool(v)) for hc, v in zip(demo["healthCode"], demo[diag_col].astype(bool))
    }


def load_mpower(manifest_csv=None, demographics_csv=None):
    """Return a standard manifest DataFrame for the mPower sustained-vowel task."""
    manifest_csv = Path(manifest_csv) if manifest_csv else MPOWER_DIR / "audio_file_manifest.csv"
    df = pd.read_csv(manifest_csv)
    # keep the phonation recording only (column name contains 'audio_audio')
    df = df[df["column"].astype(str).str.contains("audio_audio", na=False)].copy()

    labels = _load_labels(demographics_csv)
    rows = []
    for _, r in df.iterrows():
        record_id, health_code = r["recordId"], r["healthCode"]
        path = _local_wav(record_id) or (str(r["wavPath"]) if str(r.get("wavPath", "")) else None)
        if not path:
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
    return new_manifest(rows)
