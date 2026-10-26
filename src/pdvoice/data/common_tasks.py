"""Connected-speech tasks that patients and controls recorded alike.

A participant-level score averages over whatever tasks a participant recorded, so when the
two groups recorded different tasks the task mix itself predicts the label. In Bridge2AI
the protocol adds cohort-specific tasks (neurology: Cinderella story; voice: CAPE-V
sentences and Caterpillar passage): 85% of patients but 12% of controls recorded the
Cinderella story, and 1% of patients but 62% of controls the CAPE-V and Caterpillar tasks.
NeuroVoz free speech was recorded by 91% of controls and 43% of patients; IPVS tasks D and
FB by every patient and about 58% of controls. Only tasks recorded by (nearly) every
participant in both groups are kept:

    bridge2ai  Rainbow passage, picture description, story recall (the protocol's
               'generic' speech tasks; Harvard sentences are generic but recorded by
               few participants, unevenly)
    neurovoz   every speech task except FREE (the monologue)
    ipvs       B and PR
    mdvr_kcl   both tasks (balanced)

Vowel and diadochokinesis tasks are recorded by everyone and are not filtered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEEP = {
    "bridge2ai": {"rainbow-passage", "rainbowpassage", "picture-description", "story-recall"},
    "ipvs": {"B", "PR"},
}
DROP = {"neurovoz": {"FREE"}}


def common_task_mask(df: pd.DataFrame, dataset: str) -> np.ndarray:
    """True for rows to keep: every non-speech row, and speech rows of a common task."""
    speech = (df["task_kind"] == "speech").to_numpy()
    code = df["task_code"].astype(str)
    if dataset in KEEP:
        ok = code.isin(KEEP[dataset]).to_numpy()
    elif dataset in DROP:
        ok = ~code.isin(DROP[dataset]).to_numpy()
    else:
        ok = np.ones(len(df), bool)
    return ~speech | ok
