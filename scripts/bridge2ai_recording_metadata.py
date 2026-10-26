"""What the Bridge2AI release records about how US patients and controls were recorded.

For the US patients and controls the experiments use (scripts/unified_experiments.py), it
reports each participant's main microphone (the one used for most of their recordings),
whether they recorded any revised (v2) task, and whether the recording input gain separates
the groups. Only aggregates are written; no participant-level row leaves this script.

Run:  .venv\\Scripts\\python.exe scripts\\bridge2ai_recording_metadata.py
Writes outputs/logs/bridge2ai_recording_metadata.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_experiments import load  # noqa: E402

from pdvoice.config import LOGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ADAPTER, BUILTIN = "USB-C to 3.5mm Headphone Jack Adapter", "iPad Microphone"


def main() -> None:
    warnings.simplefilter("ignore")
    parts = [load("bridge2ai", t)[0] for t in ("vowel", "ddk", "speech")]
    sub = pd.concat(parts).drop_duplicates("subject_id")
    sub = sub[sub.country == "USA"][["subject_id", "label"]]
    rec = pd.read_csv(ROOT / "pd-datasets" / "bridge2ai" / "phenotype" / "task" / "recording.tsv",
                      sep="\t", low_memory=False)
    rec["subject_id"] = "b2ai_" + rec.participant_id.astype(str).str.zfill(6)
    main_mic = rec.groupby("subject_id").recording_microphone.agg(lambda s: s.mode().iat[0])
    gain = rec.groupby("subject_id").recording_input_gain.mean()
    raw = LOADERS["bridge2ai"]()
    v2 = raw[raw.task_raw.str.contains("v2", na=False)].subject_id.unique()
    sub = sub.assign(mic=sub.subject_id.map(main_mic), gain=sub.subject_id.map(gain),
                     v2=sub.subject_id.isin(v2))
    out = {}
    for lab, name in ((1, "pd"), (0, "control")):
        g = sub[sub.label == lab]
        out[name] = {"n": int(len(g)), "main_adapter": int((g.mic == ADAPTER).sum()),
                     "main_builtin": int((g.mic == BUILTIN).sum()),
                     "share_builtin": float((g.mic == BUILTIN).mean()),
                     "any_v2_task": int(g.v2.sum())}
    ok = sub.dropna(subset=["gain"])
    a = roc_auc_score(ok.label, ok.gain)
    out["input_gain_auroc"] = float(max(a, 1 - a))
    print(json.dumps(out, indent=2))
    (LOGS / "bridge2ai_recording_metadata.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
