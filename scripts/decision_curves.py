"""Full decision curves over a range of threshold probabilities (review point 5).

objective_function.py reports net benefit at a single threshold, p_t = 0.10, and picks the
operating point that maximises net benefit after the fact. This script draws the standard
decision curve instead (Vickers & Elkin 2006): at each threshold p_t a participant is
treated as positive when the predicted probability is at least p_t, and

    NB(p_t) = prevalence * sens - (1 - prevalence) * (1 - spec) * p_t / (1 - p_t)

compared with "screen everyone" (sens = 1, spec = 0) and "screen no one" (NB = 0).

PREVALENCE. Out-of-fold predictions come from classifier_comparison.py on two cohorts:
  * the uncorrected cohort (prevalence 0.187): used as they are
  * the age-matched cohort (prevalence 0.5 by construction): its probabilities are
    calibrated to 0.5, so before thresholding they are moved to the screening prevalence
    with the standard prior-shift correction,
        logit p' = logit p + logit(0.187) - logit(0.5),
    and net benefit is computed at prevalence 0.187.

Outputs: outputs/logs/decision_curves.csv, a summary in decision_curves.json, and
thesis/figures/fig_decision_curves.pdf.

Run:  .venv\\Scripts\\python.exe scripts\\decision_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdvoice.config import LOGS

FIG = Path(__file__).resolve().parents[1] / "thesis" / "figures"
TARGET_PREV = 0.187
GRID = np.round(np.arange(0.05, 0.501, 0.01), 3)
MODELS = [("SSL embeddings (2048)", "SVM (RBF)", "SSL, SVM"),
          ("SSL embeddings (2048)", "MLP head (ours)", "SSL, MLP head"),
          ("classical (42)", "Random Forest", "classical, RF"),
          ("age only (1)", "logreg (linear probe)", "age alone")]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def curve(y, p, prev):
    rows = []
    for pt in GRID:
        pred = p >= pt
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        sens = tp / max(1, np.sum(y == 1))
        spec = 1 - fp / max(1, np.sum(y == 0))
        w = pt / (1 - pt)
        rows.append(dict(pt=float(pt), sens=float(sens), spec=float(spec),
                         nb=float(prev * sens - (1 - prev) * (1 - spec) * w),
                         nb_all=float(prev - (1 - prev) * w),
                         flagged=float(np.mean(pred))))
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    global TARGET_PREV
    ap = argparse.ArgumentParser()
    ap.add_argument("--prevalence", type=float, default=TARGET_PREV,
                    help="screening prevalence; other than 0.187 writes "
                         "decision_curves__prev<p>.csv/json and draws no figure")
    TARGET_PREV = ap.parse_args().prevalence
    tag = "" if abs(TARGET_PREV - 0.187) < 1e-9 else f"__prev{TARGET_PREV:g}"
    d = pd.read_csv(LOGS / "classifier_predictions.csv")
    out, summary = [], {}
    for cohort in ("confounded", "age-matched"):
        for feat, clf, name in MODELS:
            s = d[(d.cohort == cohort) & (d.features == feat) & (d.classifier == clf)]
            if s.empty:
                continue
            y, p = s.y.to_numpy().astype(int), s.p.to_numpy().astype(float)
            prev_cohort = y.mean()
            if cohort == "age-matched" or tag:
                # move the probabilities from the cohort prevalence to the screening one
                p = 1 / (1 + np.exp(-(logit(p) + logit(TARGET_PREV) - logit(prev_cohort))))
            c = curve(y, p, TARGET_PREV)
            c["cohort"], c["model"] = cohort, name
            out.append(c)
            best_ref = np.maximum(c.nb_all, 0)
            gain = c.nb - best_ref
            useful = c.loc[gain > 0, "pt"]
            summary[f"{cohort} | {name}"] = dict(
                n=int(len(y)), cohort_prevalence=round(float(prev_cohort), 3),
                max_gain_per_1000=round(float(1000 * gain.max()), 2),
                at_pt=float(c.pt[gain.idxmax()]),
                thresholds_with_gain=(f"{useful.min():.2f}-{useful.max():.2f}"
                                      if len(useful) else "none"),
                n_thresholds_with_gain=int(len(useful)),
                gain_per_1000_at_0_10=round(float(1000 * gain[c.pt == 0.10].iloc[0]), 2),
                flagged_at_0_10=round(float(c.flagged[c.pt == 0.10].iloc[0]), 3))
    D = pd.concat(out)
    D.to_csv(LOGS / f"decision_curves{tag}.csv", index=False)
    (LOGS / f"decision_curves{tag}.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"{k:<40} {v}")
    if tag:
        return

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    styles = {"SSL, SVM": ("#4E79A7", "-"), "SSL, MLP head": ("#4E79A7", "--"),
              "classical, RF": ("#59A14F", "-"), "age alone": ("#E15759", ":")}
    for ax, cohort, title in ((axes[0], "confounded", "Uncorrected cohort"),
                              (axes[1], "age-matched", "Age-matched cohort")):
        sub = D[D.cohort == cohort]
        ref = sub.drop_duplicates("pt")
        ax.plot(ref.pt, ref.nb_all, color="#888888", lw=1.2, label="screen everyone")
        ax.axhline(0, color="#444444", lw=0.8, label="screen no one")
        for name, (col, ls) in styles.items():
            m = sub[sub.model == name]
            if not m.empty:
                ax.plot(m.pt, m.nb, color=col, ls=ls, lw=1.4, label=name)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("threshold probability $p_t$")
        ax.set_ylim(-0.03, 0.2)
        ax.grid(alpha=0.25, ls=":")
    axes[0].set_ylabel("net benefit (prevalence 0.187)")
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_decision_curves.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_decision_curves.png", dpi=200, bbox_inches="tight")
    print(f"wrote {LOGS / 'decision_curves.csv'} and {FIG / 'fig_decision_curves.pdf'}")


if __name__ == "__main__":
    main()
