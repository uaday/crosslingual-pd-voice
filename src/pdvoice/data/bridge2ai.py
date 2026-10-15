"""Loader for Bridge2AI-Voice, adult cohort audio release (English, North America).

Downloaded with scripts/download_bridge2ai.py from Synapse (view syn74341293) into
pd-datasets/bridge2ai/ (config.BRIDGE2AI_DIR). Layout (BIDS-like):

    sub-<pid>/ses-<sid>/audio/sub-<pid>_ses-<sid>_task-<task>.wav  (+ .json sidecar)
    phenotype/{enrollment,demographics,diagnosis,task,...}/<tool>.tsv

All audio is 16 kHz mono 16-bit PCM.

Labels (group column, then label):
    'pd'        in phenotype/diagnosis/parkinsons_disease.tsv with the clinician's
                gold-standard diagnosis = 'Yes'                        -> label 1
    'control'   eligibility.is_control_participant = 'Yes'               -> label 0
    'other'     everyone else: voice, respiratory, neurological and mood disorders.
                Not a healthy control and not PD -> label <NA>. Kept, because this group
                is a disease control for the specificity question (PD vs other disorders).
    'ambiguous' in the PD table but without a gold-standard 'Yes', or flagged as both PD
                and control -> label <NA>.
NOTE: phenotype/diagnosis/control.tsv is NOT the control list (its single column is 'Yes'
for some people); the control flag is in eligibility.tsv.

Task kinds (task_code keeps the release's task name with any run number removed):
    vowel   prolonged-vowel, maximum-phonation-time. The prolonged vowel is elicited as
            "1, 2, 3 aah", so the recording OPENS WITH COUNTED SPEECH; the central-segment
            windowing removes most of it, a whole-file analysis does not.
    ddk     diadochokinesis pa / ta / ka / pataka / buttercup (and v2 puh/tuh/kuh)
    speech  read: rainbow / caterpillar passage, CAPE-V and Harvard sentences;
            spontaneous: picture description, Cinderella story, story recall
    other   glides, loudness, breathing and cough, and the cognitive-linguistic tasks
            (animal fluency, productive vocabulary, random item generation, Stroop)

Extra columns: session_id, task_raw, group, age, sex, hy (Hoehn and Yahr stage, PD only),
and, with ``recording_meta=True``, microphone and input_gain from the JSON sidecars
(recording channel is a known confound; check it before trusting a result).

PRIVACY. Participant ids are study codes, not names, but the data are under a data use
agreement: save aggregates only, never per-participant rows, outside pd-datasets/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from pdvoice.config import BRIDGE2AI_DIR
from pdvoice.data.base import new_manifest

_VOWEL = {"prolonged-vowel", "maximum-phonation-time"}
_DDK = {"diadochokinesis-pa", "diadochokinesis-ta", "diadochokinesis-ka",
        "diadochokinesis-pataka", "diadochokinesis-buttercup",
        "diadochokinesis-v2-puh", "diadochokinesis-v2-tuh", "diadochokinesis-v2-kuh",
        "diadochokinesis-v2-puhtuhkuh", "diadochokinesis-v2-buttercup"}
_SPEECH = {"rainbow-passage", "rainbowpassage", "caterpillar-passage", "cape-v-sentences",
           "harvard-sentences-list", "picture-description", "cinderella-story",
           "story-recall", "free-speech"}
_FILE = re.compile(r"sub-(?P<pid>[^_]+)_ses-(?P<sid>[^_]+)_task-(?P<task>.+)\.wav$")


def _task_code(raw: str) -> str:
    """Drop run numbers and version/option tags: 'maximum-phonation-time-2' ->
    'maximum-phonation-time', 'story-recall-v2' -> 'story-recall',
    'picture-description-option1' -> 'picture-description'."""
    t = re.sub(r"(-\d+)+$", "", raw)
    t = re.sub(r"-option\d+$", "", t)
    return re.sub(r"-v2$", "", t)


def _task_kind(code: str) -> str:
    if code in _VOWEL or code == "maximum-phonation-time-v2":
        return "vowel"
    if code in _DDK:
        return "ddk"
    if code in _SPEECH:
        return "speech"
    return "other"


def _read(root: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(root / "phenotype" / rel, sep="\t", dtype=str)


def bridge2ai_participants(root=BRIDGE2AI_DIR) -> pd.DataFrame:
    """One row per participant: participant_id, group, label, age, sex, hy."""
    root = Path(root)
    elig = _read(root, "enrollment/eligibility.tsv").drop_duplicates("participant_id")
    pdx = _read(root, "diagnosis/parkinsons_disease.tsv").drop_duplicates("participant_id")
    dem = _read(root, "demographics/demographics.tsv").drop_duplicates("participant_id")

    gold = set(pdx.loc[pdx.diagnosis_parkinsons_gold_standard_diagnosis == "Yes", "participant_id"])
    in_pd_table = set(pdx.participant_id)
    control = set(elig.loc[elig.is_control_participant == "Yes", "participant_id"])

    p = elig[["participant_id"]].copy()
    p["group"] = "other"
    p.loc[p.participant_id.isin(control), "group"] = "control"
    p.loc[p.participant_id.isin(gold), "group"] = "pd"
    p.loc[p.participant_id.isin((in_pd_table - gold) | (gold & control)), "group"] = "ambiguous"
    p["label"] = p.group.map({"pd": 1, "control": 0}).astype("Int64")

    d = dem[["participant_id", "age", "sex_at_birth"]].rename(columns={"sex_at_birth": "sex"})
    d["age"] = pd.to_numeric(d.age.str.replace("90 and above", "90", regex=False), errors="coerce")
    p = p.merge(d, on="participant_id", how="left")

    stage = pdx.set_index("participant_id").diagnosis_parkinsons_ds.str.extract(r"Stage (\d)")[0]
    p["hy"] = pd.to_numeric(p.participant_id.map(stage), errors="coerce").astype("Int64")
    return p


def load_bridge2ai(root=BRIDGE2AI_DIR, recording_meta: bool = False) -> pd.DataFrame:
    """Scan the downloaded tree and return a standard manifest DataFrame."""
    root = Path(root)
    if not (root / "phenotype").exists():
        raise FileNotFoundError(f"{root}: no phenotype/ folder. Run scripts/download_bridge2ai.py.")
    people = bridge2ai_participants(root).set_index("participant_id")

    rows = []
    for wav in sorted(root.glob("sub-*/ses-*/audio/*.wav")):
        m = _FILE.match(wav.name)
        if not m:
            continue
        pid, code = m["pid"], _task_code(m["task"])
        info = people.loc[pid] if pid in people.index else None
        row = {
            "dataset": "bridge2ai",
            "subject_id": f"b2ai_{pid}",
            "path": str(wav.resolve()),
            "label": info["label"] if info is not None else pd.NA,
            "task_kind": _task_kind(code),
            "task_code": code,
            "language": "en",
            "native_sr": 16000,
            "session_id": m["sid"],
            "task_raw": m["task"],
            "group": info["group"] if info is not None else "unknown",
            "age": info["age"] if info is not None else pd.NA,
            "sex": info["sex"] if info is not None else pd.NA,
            "hy": info["hy"] if info is not None else pd.NA,
        }
        if recording_meta:
            side = wav.with_suffix(".json")
            meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
            row["microphone"] = meta.get("recording_microphone")
            row["input_gain"] = pd.to_numeric(meta.get("recording_input_gain"), errors="coerce")
        rows.append(row)

    df = new_manifest(rows)
    df["label"] = df["label"].astype("Int64")
    return df
