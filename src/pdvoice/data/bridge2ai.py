"""Loader for Bridge2AI-Voice (English external; possible future TRAINING set).

NOT downloaded yet — access is pending (DAC resubmission, see PROJECT_CONTEXT.md
§3). When available, implement so it returns the standard manifest:
    dataset='bridge2ai', language='en', label from the clinician participants table,
    task_kind in {'vowel','ddk','speech'} from the multi-task battery
    (sustained vowels, /pa-ta-ka/, read passage, picture description, sentences).
Organise the download under pd-datasets/bridge2ai/ (config.BRIDGE2AI_DIR).
"""

from __future__ import annotations

from pdvoice.config import BRIDGE2AI_DIR


def load_bridge2ai(root=BRIDGE2AI_DIR):
    raise NotImplementedError(
        "Bridge2AI not acquired yet (PhysioNet access pending). See the module "
        "docstring and PROJECT_CONTEXT.md §3 for the expected manifest layout."
    )
