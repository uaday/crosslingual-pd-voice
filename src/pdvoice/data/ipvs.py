"""Loader for the Italian Parkinson's Voice and Speech (IPVS / Dimauro 2017) dataset.

Folder layout (see PROJECT_CONTEXT.md §3.2):
    <root>/15 Young Healthy Control/<Speaker>/<CODE...>.wav      -> HC
    <root>/22 Elderly Healthy Control/<Speaker>/<CODE...>.wav    -> HC
    <root>/28 People with Parkinson's disease/<1-5|.../<Speaker>/<CODE...>.wav -> PD

File names begin with a task code: V[AEIOU]=sustained vowel; PR/FB/B/D=reading /
sentence / word tasks (treated as connected speech).

SAMPLE RATE IS MIXED, and the split is confounded with the label: PD speaker
folders ``1-5`` and ``6-10`` (10 of 28 patients) are 44.1 kHz, while the other 18
patients and ALL 37 controls are 16 kHz. Native rate 44.1 kHz therefore implies PD
with perfect precision -- rate alone scores subject-level AUROC 0.679. The artifact
survives resampling into the SPEECH embeddings (within-PD probe AUROC 0.777) but
not the vowel ones (0.519), because only speech has fricative energy near the 8 kHz
anti-alias rolloff. Read the rate per file; never assume it.
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


def _native_sr(path) -> int | None:
    """True sample rate from the file header (IPVS mixes 16 kHz and 44.1 kHz)."""
    import soundfile as sf

    try:
        return int(sf.info(str(path)).samplerate)
    except Exception:  # noqa: BLE001 — unreadable file; QC drops it downstream
        return None


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
                    "native_sr": _native_sr(wav),
                }
            )
    return new_manifest(rows)
