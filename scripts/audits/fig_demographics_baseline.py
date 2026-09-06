"""The single most informative figure in the project: what AGE ALONE can do.

Borrowed from the Bridge2AI VoiceFM paper, which includes a demographics-only baseline
"to rule out the possibility that AUROC is driven by demographic confounding rather than
acoustic disease information".

The figure carries the whole confound argument in two panels:

  LEFT   on the raw cohort, one number -- the participant's age, no microphone involved --
         outscores a 315-million-parameter transformer and 42 hand-engineered features.
  RIGHT  after age+sex matching, age falls to chance (which VERIFIES the matching worked)
         and the audio features retain a modest but real signal.

Reads from outputs/logs/classifier_comparison.csv, so it never restates a number by hand.

Run:  .venv\\Scripts\\python.exe -u scripts\\audits\\fig_demographics_baseline.py
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdvoice.config import FIGURES, LOGS  # noqa: E402

WAVE, SPEC, GREY = "#3B6EA5", "#B5651D", "#9A9A94"
ORDER = ["age only (1)", "classical (42)", "SSL embeddings (2048)"]
NICE = {"age only (1)": "AGE ALONE\n(1 number, no audio)",
        "classical (42)": "classical features\n(42)",
        "SSL embeddings (2048)": "XLS-R embeddings\n(2048)"}


def main() -> None:
    src = LOGS / "classifier_comparison.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}\nRun scripts/classifier_comparison.py first.")
    cc = pd.read_csv(src)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cohort, title, note in [
        (axes[0], "confounded", "Raw cohort",
         "PD are ~27 years older than controls"),
        (axes[1], "age-matched", "Age-matched cohort",
         "ages equalised in 5-year bins"),
    ]:
        sub = cc[cc.cohort == cohort]
        feats = [f for f in ORDER if f in set(sub.features)]
        best = [sub[sub.features == f].AUROC.max() for f in feats]
        spread = [(sub[sub.features == f].AUROC.min(),
                   sub[sub.features == f].AUROC.max()) for f in feats]
        colours = [SPEC if f.startswith("age only") else
                   (GREY if f.startswith("classical") else WAVE) for f in feats]

        x = np.arange(len(feats))
        ax.bar(x, best, 0.6, color=colours)
        # the spread across the five classifiers, so no single choice looks decisive
        for xi, (lo, hi) in zip(x, spread):
            ax.plot([xi, xi], [lo, hi], color="#33333388", lw=1.4, zorder=3)
        for xi, v in zip(x, best):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

        ax.axhline(0.5, ls=":", c="k", lw=1)
        ax.text(len(feats) - 0.45, 0.507, "chance", fontsize=8, color="#555", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([NICE.get(f, f) for f in feats], fontsize=9)
        ax.set_ylim(0.40, 1.0)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.text(0.5, 0.965, note, transform=ax.transAxes, ha="center",
                fontsize=8.5, color="#555")
        ax.grid(axis="y", alpha=.25)

    axes[0].set_ylabel("subject-level AUROC")
    fig.suptitle("If a model cannot beat the participant's age with no audio, "
                 "it has not shown a voice biomarker",
                 fontweight="bold", fontsize=12)
    fig.text(0.5, 0.005, "bar = best of five classifiers · vertical line = spread across them",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))

    out = FIGURES / "writeup" / "figF_demographics_baseline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    for cohort in ("confounded", "age-matched"):
        sub = cc[cc.cohort == cohort]
        line = "  ".join(f"{f.split(' (')[0]} {sub[sub.features == f].AUROC.max():.3f}"
                         for f in ORDER if f in set(sub.features))
        print(f"  {cohort:<12} {line}")


if __name__ == "__main__":
    main()
