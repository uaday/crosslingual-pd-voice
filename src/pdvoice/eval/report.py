"""Model Card + Datasheet writers (PROJECT_CONTEXT.md §4.2, reporting layer).

Minimal Markdown emitters so every model/dataset ships with provenance, intended
use and known limitations. Extend the templates as the project matures.
"""

from __future__ import annotations

from pathlib import Path

from pdvoice.config import CARDS


def _write(path: Path, title: str, meta: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for key, value in meta.items():
        heading = key.replace("_", " ").title()
        if isinstance(value, dict):
            lines.append(f"## {heading}")
            lines += [f"- **{k}**: {v}" for k, v in value.items()]
        elif isinstance(value, (list, tuple)):
            lines.append(f"## {heading}")
            lines += [f"- {v}" for v in value]
        else:
            lines.append(f"- **{heading}**: {value}")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def write_model_card(name: str, meta: dict, out_dir: Path = CARDS) -> Path:
    """Write outputs/cards/model_card_<name>.md. Suggested meta keys: intended_use,
    training_data, metrics, limitations, ethical_considerations."""
    return _write(Path(out_dir) / f"model_card_{name}.md", f"Model Card — {name}", meta)


def write_datasheet(name: str, meta: dict, out_dir: Path = CARDS) -> Path:
    """Write outputs/cards/datasheet_<name>.md. Suggested meta keys: motivation,
    composition, collection_process, labels, known_limitations, access."""
    return _write(Path(out_dir) / f"datasheet_{name}.md", f"Datasheet — {name}", meta)
