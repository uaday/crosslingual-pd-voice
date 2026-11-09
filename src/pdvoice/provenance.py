"""Input provenance for derived results.

A derived result (decision curves, difference intervals, ...) is computed from another
saved artefact (a prediction file). If that artefact is regenerated and the derived
result is not, every downstream table and sentence stays internally consistent and
silently wrong: the number check compares the prose with the stale derived file and
passes. (This happened on 2026-11-09: the classifier comparison was rerun, the decision
curves were not.)

`stamp` records the SHA-256 of each input next to the output, in ``<output>.inputs.json``;
`stale` recomputes the hashes and lists any input that has changed since; the number
check of the accompanying thesis calls it for every stamped output.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sidecar(output: Path) -> Path:
    return output.with_name(output.name + ".inputs.json")


def stamp(output: Path, inputs: list[Path]) -> None:
    """Record the hash of every input of `output` (call right after writing it)."""
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent).stdout.strip()
    except OSError:
        rev = ""
    _sidecar(output).write_text(json.dumps(
        {"code_revision": rev,
         "inputs": {str(Path(p).name): {"path": str(p), "sha256": sha256(Path(p))}
                    for p in inputs}}, indent=2))


def stale(output: Path) -> list[str]:
    """Inputs of `output` that changed (or vanished) since it was stamped.

    Returns ["<no provenance record>"] when the output was never stamped.
    """
    side = _sidecar(output)
    if not side.exists():
        return ["<no provenance record>"]
    rec = json.loads(side.read_text())["inputs"]
    bad = []
    for name, v in rec.items():
        p = Path(v["path"])
        if not p.exists() or sha256(p) != v["sha256"]:
            bad.append(name)
    return bad
