"""Loader for MDVR-KCL (Mobile Device Voice Recordings, King's College London).

Folder layout (see PROJECT_CONTEXT.md §3.2):
    <root>/ReadText/{HC,PD}/<file>.wav
    <root>/SpontaneousDialogue/{HC,PD}/<file>.wav

File name encodes labels:  ID_status_H&Y_UPDRS-II.5_UPDRS-III.18  e.g. ID02_pd_2_0_1.wav
Both tasks are connected speech (no sustained vowel) — this dataset is the
cross-TASK probe (English -> English, only the task changes). 44.1 kHz.

Robustness notes: the label is taken from the authoritative HC/PD FOLDER, not the
filename (one file is named 'ID22hc_...', missing the status token). The subject id
is the normalised leading 'ID<n>' token, so 'ID22hc' folds back into 'ID22' rather
than becoming a phantom 38th subject (leakage guard).
"""

from __future__ import annotations

import re
from pathlib import Path

from pdvoice.config import MDVR_KCL_DIR
from pdvoice.data.base import new_manifest

_TASKS = ["ReadText", "SpontaneousDialogue"]
_GROUPS = {"HC": 0, "PD": 1}


def _subject_id(token: str) -> str:
    """Leading 'ID<digits>' token, dropping any trailing letters ('ID22hc' -> 'ID22')."""
    m = re.match(r"[A-Za-z]+\d+", token)
    return m.group(0) if m else token


def load_mdvr_kcl(root=MDVR_KCL_DIR):
    """Scan the MDVR-KCL tree and return a standard manifest DataFrame."""
    root = Path(root)
    rows = []
    for task in _TASKS:
        for grp, label in _GROUPS.items():
            d = root / task / grp
            if not d.exists():
                continue
            for wav in d.glob("*.wav"):
                parts = wav.stem.split("_")
                nums = [int(t) for t in parts[1:] if t.isdigit()]  # H&Y, UPDRS-II.5, UPDRS-III.18
                rows.append(
                    {
                        "dataset": "mdvr_kcl",
                        "subject_id": f"kcl_{_subject_id(parts[0])}",
                        "path": str(wav.resolve()),
                        "label": label,  # authoritative: from the HC/PD folder
                        "task_kind": "speech",
                        "task_code": task,
                        "language": "en",
                        "native_sr": 44100,
                        "hy": nums[0] if len(nums) > 0 else None,
                        "updrs2_5": nums[1] if len(nums) > 1 else None,
                        "updrs3_18": nums[2] if len(nums) > 2 else None,
                    }
                )
    return new_manifest(rows)
