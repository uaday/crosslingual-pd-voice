"""Loader for NeuroVoz (Castilian Spanish external, cross-language).

NOT downloaded yet — Zenodo request-only, access pending (PROJECT_CONTEXT.md §3).
When granted, implement so it returns the standard manifest:
    dataset='neurovoz', language='es', clinician labels,
    task_kind: 'vowel' for sustained /a,e,i,o,u/, 'ddk' for /pa-ta-ka/,
    'speech' for listen-repeat + monologue. Organise under config.NEUROVOZ_DIR.
"""

from __future__ import annotations

from pdvoice.config import NEUROVOZ_DIR


def load_neurovoz(root=NEUROVOZ_DIR):
    raise NotImplementedError(
        "NeuroVoz not acquired yet (Zenodo request pending). See the module "
        "docstring and PROJECT_CONTEXT.md §3 for the expected manifest layout."
    )
