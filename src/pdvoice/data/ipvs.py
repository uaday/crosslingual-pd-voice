"""Loader for the Italian Parkinson's Voice and Speech (IPVS / Dimauro 2017) dataset.

Folder layout (see PROJECT_CONTEXT.md §3.2):
    <root>/15 Young Healthy Control/<Speaker>/<CODE...>.wav      -> HC
    <root>/22 Elderly Healthy Control/<Speaker>/<CODE...>.wav    -> HC
    <root>/28 People with Parkinson's disease/<1-5|.../<Speaker>/<CODE...>.wav -> PD

File names begin with a task code: V[AEIOU]=sustained vowel; PR/FB/B/D=reading /
sentence / word tasks (treated as connected speech). 44.1 kHz.
"""

from __future__ import annotations

import re
from pathlib import Path

from pdvoice.config import IPVS_DIR
from pdvoice.data.base import new_manifest

# group folder name -> (label, short tag for a collision-proof subject id)
_GROUPS = {
    "15 Young Healthy Control": (0, "yhc"),
    "22 Elderly Healthy Control": (0, "ehc"),
    "28 People with Parkinson's disease": (1, "pd"),
}


def _task_code(stem: str) -> str:
    """Leading alphabetic prefix of the file name, e.g. 'VA1cda...' -> 'VA'."""
    m = re.match(r"[A-Za-z]+", stem)
    return m.group(0).upper() if m else ""


def _task_kind(code: str) -> str:
    """V + vowel letter => sustained vowel; everything else => connected speech."""
    if len(code) >= 2 and code[0] == "V" and code[1] in "AEIOU":
        return "vowel"
    return "speech"


def load_ipvs(root=IPVS_DIR):
    """Scan the IPVS tree and return a standard manifest DataFrame."""
    root = Path(root)
    rows = []
    for group_name, (label, tag) in _GROUPS.items():
        gdir = root / group_name
        if not gdir.exists():
            continue
        for wav in gdir.rglob("*.wav"):
            # Use the speaker folder's PATH (not just its name): PD patients are
            # nested under range folders (1-5, 6-10, ...) and different patients can
            # share a display name (e.g. two "Roberto R"). The path is unique per
            # person, so this prevents merging distinct subjects (leakage guard).
            speaker_rel = wav.parent.relative_to(gdir)
            sid = re.sub(r"[^A-Za-z0-9]+", "_", str(speaker_rel)).strip("_")
            code = _task_code(wav.stem)
            rows.append(
                {
                    "dataset": "ipvs",
                    "subject_id": f"ipvs_{tag}_{sid}",
                    "path": str(wav.resolve()),
                    "label": label,
                    "task_kind": _task_kind(code),
                    "task_code": code,
                    "language": "it",
                    "native_sr": 44100,
                }
            )
    return new_manifest(rows)
