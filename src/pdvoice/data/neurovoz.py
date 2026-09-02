"""Loader for NeuroVoz (Castilian Spanish external, cross-language).

NeuroVoz v3 layout (Zenodo):
    <root>/data/audios/<STATUS>_<TASK>_<ID>.wav
      STATUS = HC | PD                         -> label 0 / 1 (clinician)
      TASK   = A1/A2/E1/.../U2  sustained vowels  -> task_kind 'vowel'
               PATAKA           DDK /pa-ta-ka/     -> task_kind 'ddk'
               PERRO/FREE/...   words + monologue  -> task_kind 'speech'
      ID     = subject number (unique within a STATUS)
    <root>/data/metadata/metadata_{hc,pd}.csv    per-subject demographics (not needed for labels)

44.1 kHz mono. Subject id includes STATUS so an HC and a PD sharing a number never merge.
"""

from __future__ import annotations

import re
from pathlib import Path

from pdvoice.config import NEUROVOZ_DIR
from pdvoice.data.base import new_manifest

_VOWEL = re.compile(r"^[AEIOU]\d+$")


def _task_kind(task: str) -> str:
    t = task.upper()
    if _VOWEL.match(t):
        return "vowel"
    if t == "PATAKA":
        return "ddk"
    return "speech"


def load_neurovoz(root=NEUROVOZ_DIR):
    """Scan the NeuroVoz audio tree and return a standard manifest DataFrame."""
    root = Path(root)
    audio_dir = root / "data" / "audios"
    if not audio_dir.exists():  # tolerate a flat layout too
        audio_dir = next((p for p in [root / "audios", root] if (p).exists()), root)

    rows = []
    for wav in sorted(audio_dir.rglob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue  # unexpected name -> skip
        status, subject_num = parts[0].upper(), parts[-1]
        task = "_".join(parts[1:-1])
        label = 1 if status == "PD" else 0 if status == "HC" else None
        if label is None:
            continue
        rows.append(
            {
                "dataset": "neurovoz",
                "subject_id": f"neurovoz_{status.lower()}_{subject_num}",
                "path": str(wav.resolve()),
                "label": label,
                "task_kind": _task_kind(task),
                "task_code": task.upper(),
                "language": "es",
                "native_sr": 44100,
            }
        )
    return new_manifest(rows)
