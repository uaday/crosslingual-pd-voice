"""Loader for EWA-DB (Slovak external, cross-language).

NOT downloaded yet — Zenodo request-only, access pending.
EWA-DB is primarily an Alzheimer database (~7 TB) that ALSO contains PD + HC —
download and load ONLY the PD + HC vowel/DDK subset (it will not fit 1 TB otherwise).
When granted, implement so it returns the standard manifest:
    dataset='ewadb', language='sk', clinician labels,
    task_kind: 'vowel' / 'ddk' / 'speech'. Organise under config.EWADB_DIR.
"""

from __future__ import annotations

from pdvoice.config import EWADB_DIR


def load_ewadb(root=EWADB_DIR):
    raise NotImplementedError(
        "EWA-DB not acquired yet (Zenodo request pending). Load ONLY the PD+HC subset. "
        "See the module docstring for the expected layout."
    )
