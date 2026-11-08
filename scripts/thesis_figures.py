"""Generate every data figure the thesis uses, from the artifacts.

Same contract as scripts/thesis_tables.py, and for the same reason: the previous
hand-made figures in outputs/figures/writeup/ drifted from the artifacts they describe
(methods.png shows SSL 0.805 where the artifact now says 0.724). Figures are regenerated
from outputs/logs/ on demand and written to outputs/figures/.

Diagrams -- the pipeline block diagram and the split/lock schematic -- are NOT here.
Those are drawn by hand and saved as PDF in outputs/figures/.

Run:  .venv\\Scripts\\python.exe scripts\\thesis_figures.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdvoice.config import LOGS

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"

# Print-friendly: greyscale-safe ordering, no reliance on colour alone.
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": True, "savefig.bbox": "tight",
})
C_CONF, C_MATCH = "#4E79A7", "#F28E2B"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def _j(n):
    return json.loads((LOGS / n).read_text())


# ------------------------------------------------------------------ figures
def f_protocol() -> None:
    """The headline: same model, two protocols, the ordering reverses."""
    c = pd.read_csv(LOGS / "classifier_comparison.csv")
    b = c.groupby(["cohort", "features"]).AUROC.max()
    sets = [("age only (1)", "Age alone\n(1 feature)"),
            ("classical (42)", "Classical\n(42)"),
            ("SSL embeddings (2048)", "SSL embeddings\n(2048)")]
    conf = [b[("confounded", k)] for k, _ in sets]
    matc = [b[("age-matched", k)] for k, _ in sets]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(sets)); w = 0.38
    ax.bar(x - w/2, conf, w, label="Uncorrected cohort", color=C_CONF)
    ax.bar(x + w/2, matc, w, label="Age-matched cohort", color=C_MATCH)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.text(len(sets) - 0.45, 0.507, "chance", fontsize=7, color="k")
    for xi, (a, m) in enumerate(zip(conf, matc)):
        ax.text(xi - w/2, a + .008, f"{a:.3f}", ha="center", fontsize=7.5)
        ax.text(xi + w/2, m + .008, f"{m:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([s for _, s in sets])
    ax.set_ylabel("Subject-level AUROC"); ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    _save(fig, "fig_protocol_reversal")


def f_depth() -> None:
    """Age contamination as a function of encoder depth."""
    d = pd.read_csv(LOGS / "select_transfer_layer.csv")
    lc = "layer" if "layer" in d.columns else d.columns[0]
    ac = "auroc" if "auroc" in d.columns else d.columns[1]
    d = d.sort_values(lc)
    n_layers = len(d)

    # The confounded curve is SPARSE on purpose. layer1_mpower.py was only ever run at
    # five layers, so only five points exist; drawing a line through them would imply a
    # sweep that was not done. The contrast is the finding, so both are shown and the
    # asymmetry is stated in the caption rather than hidden by interpolation.
    # Both curves now come from the same script, head and cross-validation
    # (select_transfer_layer.py --cohort agematched / confounded), one recording per speaker.
    c = pd.read_csv(LOGS / "select_transfer_layer__confounded.csv").sort_values("layer")
    conf = list(zip(c["layer"], c["auroc"]))

    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.plot(d[lc], d[ac], "-o", ms=3, color=C_MATCH,
            label="Age-matched cohort")
    if conf:
        cx, cy = zip(*conf)
        ax.plot(cx, cy, "s--", ms=3, color=C_CONF, alpha=0.9,
                label="Uncorrected cohort")
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.text(0.2, 0.505, "chance", fontsize=7)

    best = d.loc[d[ac].idxmax()]
    ax.annotate(f"best age-matched:\nlayer {int(best[lc])}, {best[ac]:.3f}",
                (best[lc], best[ac]), textcoords="offset points", xytext=(-18, 12),
                fontsize=7.5)
    last = d.iloc[-1]
    ax.annotate(f"final layer: {last[ac]:.3f}\nrank {n_layers}/{n_layers}",
                (last[lc], last[ac]), textcoords="offset points", xytext=(-70, -4),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.7))
    if conf:
        lx, ly = conf[-1]
        ax.annotate(f"same layer,\nconfounded: {ly:.3f}", (lx, ly),
                    textcoords="offset points", xytext=(-60, -42), fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("Encoder layer"); ax.set_ylabel("Subject-level AUROC")
    ax.set_ylim(0.48, 0.88)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _save(fig, "fig_depth_gradient")


def f_labels_vs_age() -> None:
    """The causal discriminator: the two hypotheses predict opposite directions."""
    R = pd.read_csv(LOGS / "unified" / "runs.csv")
    clin = [("ipvs", "vowel"), ("neurovoz", "vowel"), ("mdvr_kcl", "speech")]

    def within(c):
        return R[(R.exp == "within") & (R.source == "mpower") & (R.condition == c)].auroc.mean()

    def transfer(c):
        g = R[(R.exp == "exp1") & (R.condition == c) & (R.target_cohort == "released")]
        m = np.zeros(len(g), bool)
        for t, tt in clin:
            m |= ((g.target == t) & (g.target_task == tt)).to_numpy()
        return g[m].groupby("seed").auroc.mean().mean()
    b_w, b_t = within("uncorrected"), transfer("uncorrected")
    rows = [("Confident labels only", "confident"), ("Age-matched", "agematched"),
            ("Age- and sex-matched", "agesexmatched")]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(rows)); w = 0.38
    dw = [within(c) - b_w for _, c in rows]
    dt = [transfer(c) - b_t for _, c in rows]
    ax.bar(x - w/2, dw, w, label="Within mPower", color=C_CONF)
    ax.bar(x + w/2, dt, w, label="Mean transfer (IPVS, NeuroVoz, MDVR-KCL)", color=C_MATCH)
    ax.axhline(0, color="k", lw=0.9)
    for xi, (a_, m_) in enumerate(zip(dw, dt)):
        ax.text(xi - w/2, a_ + (.006 if a_ >= 0 else -.018), "<0.001" if abs(a_) < 0.0005 else f"{a_:+.3f}", ha="center", fontsize=7.5)
        ax.text(xi + w/2, m_ + (.006 if m_ >= 0 else -.018), "<0.001" if abs(m_) < 0.0005 else f"{m_:+.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in rows], fontsize=8)
    ax.set_ylabel("Change in AUROC from uncorrected")
    ax.legend(frameon=False, fontsize=7.5)
    _save(fig, "fig_labels_vs_age")

def f_windowing() -> None:
    """A null preprocessing change still swings external results by +-0.29."""
    def g(ds, task, L, w):
        tag = "" if w == "current" else f"__W{w}"
        p = LOGS / f"layer2__xls-r-300m__{ds}__{task}__L{L}{tag}.json"
        return json.loads(p.read_text())["auroc"] if p.exists() else None

    pairs = [("ipvs", "vowel"), ("neurovoz", "vowel"), ("neurovoz", "ddk"),
             ("ipvs", "speech"), ("neurovoz", "speech"), ("mdvr_kcl", "speech")]
    labels, deltas = [], []
    for ds, task in pairs:
        for L in (11, 15):
            a, b = g(ds, task, L, "current"), g(ds, task, L, "full")
            if a is not None and b is not None:
                labels.append(f"{ds}/{task}\nL{L}"); deltas.append(b - a)
    order = np.argsort(deltas)
    deltas = np.array(deltas)[order]; labels = np.array(labels)[order]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    cols = [C_CONF if v < 0 else C_MATCH for v in deltas]
    ax.bar(range(len(deltas)), deltas, color=cols)
    ax.axhline(0, color="k", lw=0.9)
    m = deltas.mean()
    ax.axhline(m, color="k", lw=0.9, ls="--")
    ax.text(len(deltas) - 0.5, m + 0.012, f"mean {m:+.3f}", ha="right", fontsize=7.5)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("Change in AUROC\n(whole recording $-$ fixed window)")
    _save(fig, "fig_windowing_noise")


def f_probes() -> None:
    """What the frozen embedding encodes -- the diagnosis ranks last."""
    d = _j("embedding_probes.json")
    want = {"task (vowel vs speech)": "Elicitation task", "sex": "Speaker sex",
            "corpus identity (3-way)": "Dataset identity", "language (3-way)": "Language",
            "age, young(<35) vs older(>=55)": "Age band",
            "** PD diagnosis **": "Parkinson's diagnosis",
            "recording device (4-way)": "Recording device"}
    seen, rows = set(), []
    for r in d:
        a = r["attribute"]
        if a in want and a not in seen:
            seen.add(a); rows.append((want[a], r["value"], r["chance"]))
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    y = np.arange(len(rows))
    cols = ["#C0392B" if n.startswith("Parkinson") else C_CONF for n, _, _ in rows]
    ax.barh(y, [v for _, v, _ in rows], color=cols)
    for i, (_, v, ch) in enumerate(rows):
        ax.plot([ch, ch], [i - .4, i + .4], color="k", lw=1.2)
        ax.text(v + .012, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in rows], fontsize=8)
    ax.set_xlabel("Probe performance (AUROC, or accuracy for multi-class)")
    ax.set_xlim(0, 1.12)
    ax.text(0.99, -0.9, "vertical bars mark chance", fontsize=6.5, ha="right")
    _save(fig, "fig_embedding_probes")


def f_transfer_b() -> None:
    """Experiment 2 as two 4 x 4 matrices: uncorrected, and age-matched training and test."""
    R = pd.read_csv(LOGS / "unified" / "runs.csv")
    names = {"bridge2ai": "Bridge2AI", "neurovoz": "NeuroVoz", "ipvs": "IPVS",
             "mdvr_kcl": "MDVR-KCL"}
    order = ["bridge2ai", "neurovoz", "ipvs", "mdvr_kcl"]
    W = R[(R.exp == "within") & (R.source_task == "speech")]
    E = R[R.exp == "exp2"]
    panels = [("Uncorrected", "uncorrected", "released"), ("Age-matched", "agematched", "agematched")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    fig.set_layout_engine("constrained")
    for ax, (title, cond, coh) in zip(axes, panels):
        M = np.full((4, 4), np.nan)
        for i, src in enumerate(order):
            for j, tgt in enumerate(order):
                if src == tgt:
                    v = W[(W.source == src) & (W.condition == cond)].auroc
                else:
                    g = E[(E.source == src) & (E.target == tgt) & (E.condition == cond)]
                    gc = g[g.target_cohort == coh]
                    v = (gc if not gc.empty else g[g.target_cohort == "released"]).auroc
                if len(v):
                    M[i, j] = v.mean()
        im = ax.imshow(M, cmap="YlGnBu", vmin=0.2, vmax=1.0, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels([names[o] for o in order], fontsize=7, rotation=20)
        ax.set_yticks(range(4)); ax.set_yticklabels([names[o] for o in order], fontsize=7)
        ax.set_title(title, fontsize=9); ax.set_xlabel("Tested on", fontsize=8)
        for i in range(4):
            for j in range(4):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if M[i, j] > 0.8 else "black",
                            fontweight="bold" if i == j else "normal")
                else:
                    ax.text(j, i, "--", ha="center", va="center", fontsize=7.5)
        ax.grid(False)
    axes[0].set_ylabel("Trained on", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="AUROC")
    _save(fig, "fig_transfer_matrix")


def f_agegap() -> None:
    """Age distributions that motivate the whole analysis."""
    import warnings as _w
    _w.simplefilter("ignore")
    from pdvoice.cohort import attach_age, age_match_ids
    from pdvoice.data.mpower import load_mpower
    df = attach_age(load_mpower())
    df = df[df.label.isin([0, 1])].drop_duplicates("subject_id").dropna(subset=["age"])
    keep = age_match_ids(df)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, sub, title in ((axes[0], df, "Before matching"),
                           (axes[1], df[df.subject_id.isin(keep)], "After age matching")):
        for lab, name, col in ((0, "Control", C_CONF), (1, "Parkinson's", C_MATCH)):
            a = sub[sub.label == lab].age
            ax.hist(a, bins=np.arange(15, 95, 5), alpha=0.65, label=name, color=col)
        pd_m = sub[sub.label == 1].age.mean(); hc_m = sub[sub.label == 0].age.mean()
        ax.set_title(f"{title}\nmean gap {pd_m - hc_m:+.1f} yr", fontsize=9)
        ax.set_xlabel("Age (years)")
    axes[0].set_ylabel("Speakers"); axes[0].legend(frameon=False, fontsize=7.5)
    _save(fig, "fig_age_gap")


def f_experiment_map() -> None:
    """Which dataset trains, which tests, and under which design: one grid for the reader."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    names = ["mPower", "Bridge2AI", "NeuroVoz", "IPVS", "MDVR-KCL"]
    sub = ["English, vowel\nself-report", "English, vowel/DDK/speech\nclinician",
           "Spanish\nclinician", "Italian\nclinician", "English\nclinician"]
    cells = {  # (row = trained on, col = tested on) -> (code, colour)
        (0, 0): ("W", "#DDDDDD"), (0, 1): ("1", "#4E79A7"), (0, 2): ("1", "#4E79A7"),
        (0, 3): ("1", "#4E79A7"), (0, 4): ("1", "#4E79A7"),
        (1, 1): ("W+S", "#DDDDDD"), (1, 2): ("2", "#F28E2B"),
        (1, 3): ("2", "#F28E2B"), (1, 4): ("2", "#F28E2B"),
        (2, 1): ("2", "#F28E2B"), (2, 2): ("W", "#DDDDDD"), (2, 3): ("2", "#F28E2B"), (2, 4): ("2", "#F28E2B"),
        (3, 1): ("2", "#F28E2B"), (3, 2): ("2", "#F28E2B"), (3, 3): ("W", "#DDDDDD"), (3, 4): ("2", "#F28E2B"),
        (4, 1): ("2", "#F28E2B"), (4, 2): ("2", "#F28E2B"), (4, 3): ("2", "#F28E2B"), (4, 4): ("W", "#DDDDDD"),
    }
    fig = plt.figure(figsize=(7.4, 6.6))
    fig.set_layout_engine(None)

    # --- top: the locked pipeline every off-diagonal cell uses
    top = fig.add_axes([0.02, 0.80, 0.96, 0.18]); top.axis("off")
    top.set_xlim(0, 100); top.set_ylim(0, 20)
    steps = ["Source dataset\n(row)", "Frozen XLS-R\n(one fixed layer)", "MLP head +\ncalibration",
             "Lock model\nand threshold", "Target dataset\n(column)"]
    xs = [1, 21, 41, 61, 81]
    for i, (x, s) in enumerate(zip(xs, steps)):
        top.add_patch(FancyBboxPatch((x, 5), 17, 11, boxstyle="round,pad=0.4",
                                     fc="#F2F2F2" if i not in (3,) else "#FCE8D5", ec="#555555", lw=0.8))
        top.text(x + 8.5, 10.5, s, ha="center", va="center", fontsize=8)
        if i < 4:
            top.add_patch(FancyArrowPatch((x + 17.6, 10.5), (xs[i + 1] - 0.6, 10.5),
                                          arrowstyle="-|>", mutation_scale=10, color="#333333"))
    top.text(50, 1.0, "Training data never reach the target; nothing is re-fitted after locking.",
             ha="center", fontsize=7.5, style="italic", color="#333333")

    # --- grid: rows trained on, columns tested on
    ax = fig.add_axes([0.25, 0.20, 0.73, 0.55])
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.invert_yaxis(); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    for r in range(5):
        for c in range(5):
            code, col = cells.get((r, c), ("", "white"))
            ax.add_patch(plt.Rectangle((c + 0.04, r + 0.04), 0.92, 0.92, fc=col,
                                       ec="#BBBBBB", lw=0.6))
            if code:
                ax.text(c + 0.5, r + 0.5, code, ha="center", va="center", fontsize=11,
                        fontweight="bold", color="black" if col == "#DDDDDD" else "white")
    for c, n in enumerate(names):
        ax.text(c + 0.5, -0.12, n, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    lab = fig.add_axes([0.0, 0.20, 0.25, 0.55]); lab.axis("off")
    lab.set_xlim(0, 1); lab.set_ylim(0, 5); lab.invert_yaxis()
    for r, (n, s) in enumerate(zip(names, sub)):
        lab.text(0.97, r + 0.38, n, ha="right", va="center", fontsize=8.5, fontweight="bold")
        lab.text(0.97, r + 0.68, s, ha="right", va="center", fontsize=6.5, color="#444444")
    lab.text(0.10, 2.5, "Trained on", rotation=90, ha="center", va="center", fontsize=9)
    lab.text(0.97, -0.12, "Tested on:", ha="right", va="bottom", fontsize=9, style="italic")

    # --- legend
    leg = fig.add_axes([0.02, 0.0, 0.96, 0.18]); leg.axis("off")
    leg.set_xlim(0, 100); leg.set_ylim(0, 18)
    items = [("1", "#4E79A7", "Experiment 1: trained on mPower sustained vowels, applied to every task of each other dataset"),
             ("2", "#F28E2B", "Experiment 2: trained on one dataset's connected speech, applied to each other dataset's speech"),
             ("W", "#DDDDDD", "Within one dataset: participant-disjoint cross-validation; S = PD against other disorders")]
    for i, (code, col, txt) in enumerate(items):
        y = 16 - i * 3.0
        leg.add_patch(plt.Rectangle((1, y - 1.1), 3.2, 2.2, fc=col, ec="#888888", lw=0.5))
        leg.text(2.6, y, code, ha="center", va="center", fontsize=7, fontweight="bold",
                 color="black" if col == "#DDDDDD" else "white")
        leg.text(5.5, y, txt, va="center", fontsize=7.5)
    leg.text(1, 0.6, "Every cell: layer 11, locked head, five seeds; run uncorrected, age-matched and "
             "age- and sex-matched where the data allow.", fontsize=7.5, style="italic")
    _save(fig, "fig4_0_experiment_map")


def f_confound_diagram() -> None:
    """How mPower recruitment ties age to the diagnosis, so a model can score by detecting age."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig = plt.figure(figsize=(7.2, 3.3))
    fig.set_layout_engine(None)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 46)
    red = "#C0392B"

    def box(x, y, w, h, txt, fc="#F2F2F2", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc=fc,
                                    ec="#555555", lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5,
                fontweight="bold" if bold else "normal")

    def arrow(p, q, col="#333333", style="-|>", ls="-", lw=1.1):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=col,
                                     lw=lw, linestyle=ls))

    box(1, 17, 16, 12, "mPower\nrecruitment", bold=True)
    box(30, 31, 15, 9, "Age", fc="#FCE8D5", bold=True)
    box(30, 6, 15, 9, "Diagnosis", fc="#FCE8D5", bold=True)
    box(60, 18, 14, 10, "Voice\nrecording")
    box(84, 18, 15, 10, "Classifier\n(predicts PD)")

    arrow((17.6, 26), (29.4, 34))
    arrow((17.6, 20), (29.4, 11))
    ax.text(20, 36.5, "young volunteers\njoin the app", fontsize=7, ha="center")
    ax.text(20, 6.0, "older patients\nenrol deliberately", fontsize=7, ha="center")

    arrow((37.5, 30.4), (37.5, 15.6), col=red, style="<|-|>", lw=1.4)
    ax.text(39.3, 23, "27.5-year gap:\nPD 60.0,\ncontrols 32.6", fontsize=7.2, color=red,
            ha="left", va="center")

    arrow((45.6, 35), (59.4, 25.5), lw=1.6)
    ax.text(53.5, 36.5, "strong: age changes\nevery voice", fontsize=7, ha="center")
    arrow((45.6, 11), (59.4, 20.5), ls="--")
    ax.text(53.5, 6.0, "weak: the disease\nsignal we want", fontsize=7, ha="center")

    arrow((74.6, 23), (83.4, 23), lw=1.4)

    R = pd.read_csv(LOGS / "unified" / "runs.csv")
    home = R[(R.source == "mpower") & (R.condition == "uncorrected")]
    age_only = home[home.exp == "within_age_only"].auroc.mean()
    voice = home[home.exp == "within"].auroc.mean()
    ax.text(50, 0.5, "Through the gap, a model can predict the diagnosis by reading age from the "
            f"voice: age alone scores {age_only:.3f}, the voice model {voice:.3f}.",
            ha="center", fontsize=7.5, style="italic")
    _save(fig, "fig4_0c_confound_diagram")


def _box(ax, x, y, w, h, txt, fc, fs=8.5, bold_first=None, ec="#555555", lw=0.8):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, p, q, ls="-", col="#3A3F4B", lw=1.3):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=col, lw=lw,
                                 linestyle=ls))


def f_score_distributions() -> None:
    """Participant scores of patients and controls in four settings (one seed)."""
    S = pd.read_csv(LOGS / "unified" / "scores.csv.gz")
    panels = [("(a) mPower, as released", dict(exp="within", source="mpower",
                                               condition="uncorrected")),
              ("(b) mPower, age-matched", dict(exp="within", source="mpower",
                                               condition="agematched")),
              ("(c) Bridge2AI speech, age-matched (US)",
               dict(exp="within", source="bridge2ai", source_task="speech",
                    condition="agematched")),
              ("(d) mPower model on IPVS vowels", dict(exp="exp1", condition="uncorrected",
                                                       target="ipvs", target_task="vowel",
                                                       target_cohort="released"))]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
    bins = np.linspace(0, 1, 21)
    from sklearn.metrics import roc_auc_score
    for ax, (title, key) in zip(axes.ravel(), panels):
        m = np.ones(len(S), bool)
        for k, v in key.items():
            m &= (S[k] == v).to_numpy()
        g = S[m & (S.seed == 42).to_numpy()]
        for lab, col, name in ((0, "#4E79A7", "controls"), (1, "#E15759", "patients")):
            x = g.r[g.y == lab]
            ax.hist(x, bins=bins, density=True, alpha=0.55, color=col,
                    label=f"{name} (n={len(x)})")
        ax.set_title(f"{title}\nAUROC {roc_auc_score(g.y, g.r):.2f} (seed 42)", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False)
        ax.tick_params(labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel("model score (ranking map)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("density", fontsize=8)
    _save(fig, "fig4_score_distributions")


TENS = {10: "ten", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}


def f_preprocessing() -> None:
    """The four pre-processing stages and what each discards (Figure 3.1)."""
    fig = plt.figure(figsize=(7.8, 4.6))
    fig.set_layout_engine(None)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    green, orange = "#E2F2E0", "#FCE4CC"
    pause = json.loads((LOGS / "pause_information.json").read_text())
    keep = {ds: {int(k): 100 * (1 - v) for k, v in g["pause_frac"].items()}
            for ds, g in pause["per_group"].items()}
    timing_ipvs = next(r["auroc"] for r in pause["results"] if r["dataset"] == "ipvs")
    win = json.loads((LOGS / "speech_windows.json").read_text())
    stages = [("Resample\n16 kHz mono", "loses nothing:\nall files already mono"),
              ("Remove non-speech\nvowel: trim 30 dB\nspeech: Silero VAD",
               f"loses pauses, hesitation,\nspeech rate (worth {timing_ipvs:.3f}\non IPVS alone)"),
              ("Fixed window\nvowel: 2 s core\nspeech: 5 s, 50% overlap",
               "loses long-range\nvariation within\nan utterance"),
              ("Normalise\n$-$23 dBFS (EBU R128)",
               "loses absolute loudness\n(unusable on an\nuncalibrated phone)")]
    w, gap = 21, 4.3
    for i, (top, bottom) in enumerate(stages):
        x = 2 + i * (w + gap)
        _box(ax, x, 74, w, 17, top, green, fs=7.6, lw=1.6 if i == 1 else 0.8)
        _box(ax, x, 45, w, 17, bottom, orange, fs=7.2)
        _arrow(ax, (x + w / 2, 73.4), (x + w / 2, 62.8))
        if i < 3:
            _arrow(ax, (x + w + 0.5, 82.5), (x + w + gap - 0.5, 82.5))
    w_nv, w_ip, w_md = (win[d]["mean_windows"] for d in ("neurovoz", "ipvs", "mdvr_kcl"))
    notes = [f"Stage 2 is not class-neutral. On IPVS it retains {keep['ipvs'][1]:.1f}% of a patient "
             f"recording against {keep['ipvs'][0]:.1f}% of a control; the direction\nreverses on "
             f"NeuroVoz ({keep['neurovoz'][1]:.1f}% vs {keep['neurovoz'][0]:.1f}%), "
             "where most speech is short sentences repeated after the examiner.",
             "Stage 4 is provably inert. Attenuating by 60 dB and renormalising changes F0, HNR, "
             "jitter and shimmer by 0.00%.",
             f"Windows per recording differ about {TENS[int(round(w_md / w_nv, -1))]}fold: NeuroVoz {w_nv:.1f}, "
             f"IPVS {w_ip:.1f}, MDVR-KCL {w_md:.1f}. These are not comparable quantities of material."]
    for k, t in enumerate(notes):
        ax.text(2, 33 - k * 11, t, fontsize=7.2, color="#555555", va="top")
    _save(fig, "fig3_1_preprocessing")


def f_pipeline() -> None:
    """The proposed framework, stage by stage (Figure 4.2)."""
    fig = plt.figure(figsize=(7.6, 6.8))
    fig.set_layout_engine(None)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(7, 101)
    blue, green, yellow, grey, pink, purple = ("#DCEBFA", "#E2F2E0", "#FFF2B0", "#ECECEC",
                                               "#F7DADF", "#DCC7EE")
    X, W, H = 33.5, 37, 7.2
    rows = [
        (92, "Voice recording: sustained vowel or connected speech", blue),
        (80, "Pre-processing: 16 kHz mono, silence trim (vowels) or\nvoice-activity detection "
             "(speech), 2 s vowel core or 5 s\nspeech windows, $-$23 dBFS", green),
        (68, "FROZEN ENCODER: XLS-R-300M, layer 11\n315,438,720 weights, never updated", yellow),
        (57, "Mean + standard-deviation pooling over time\n$\\rightarrow$ 2048 numbers", grey),
        (46, "Classifier head: 2048 $\\rightarrow$ 256 $\\rightarrow$ 1\n525,313 trained weights "
             "(0.17%)", pink),
        (33, "Calibration: temperature, Platt or isotonic, chosen\nby Brier score; Platt ranking "
             "map for AUROC;\nYouden threshold. Fitted on VALIDATION only", pink),
        (21, "LOCK: nothing is re-fitted after this line", purple),
        (10, "Evaluation: scored per PARTICIPANT, not per recording", purple),
    ]
    for y, txt, fc in rows:
        _box(ax, X, y, W, H if "\n" not in txt else H + 1.5 * txt.count("\n") - 1.0, txt, fc,
             fs=7.3)
    centers = []
    for y, txt, _ in rows:
        h = H if "\n" not in txt else H + 1.5 * txt.count("\n") - 1.0
        centers.append((y, y + h))
    for (y0, top0), (y1, top1) in zip(centers[:-1], centers[1:]):
        _arrow(ax, (X + W / 2, y0 - 0.4), (X + W / 2, top1 + 0.6))
    # classical branch
    _box(ax, 1, 55, 29, 10, "CLASSICAL BRANCH:\n42 Praat descriptors: jitter,\nshimmer, HNR, "
         "F0, formants, MFCCs", green, fs=6.8)
    _arrow(ax, (X - 0.4, 84.5), (15.5, 65.6), ls="--")
    _arrow(ax, (15.5, 54.6), (X - 0.4, 50), ls="--")
    ax.text(15.5, 50.5, "alternative input,\nor combined with\nthe embedding", fontsize=6.6,
            ha="center", va="top", color="#333333")
    # side notes
    notes = [(72, "Why frozen: about 900 patients.\nFine-tuning 315 million weights\nwould fit "
                  "the training dataset,\nnot the disease."),
             (59, "Why two branches: jitter is a\n~10 ms period perturbation, finer\nthan the "
                  "encoder's 20 ms frame."),
             (24.5, "Why lock: re-choosing the\nthreshold on each external dataset\nwould tune "
                    "the model on its own\ntest data.")]
    for y, t in notes:
        ax.text(73.8, y, t, fontsize=6.6, color="#555555", va="center")
    _save(fig, "fig4_1_pipeline")


def f_split_lock() -> None:
    """Which quantity is fitted on which partition (Figure 4.3)."""
    fig = plt.figure(figsize=(7.2, 5.6))
    fig.set_layout_engine(None)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    blue, green, yellow, grey, pink, purple = ("#DCEBFA", "#E2F2E0", "#FFF2B0", "#ECECEC",
                                               "#F7DADF", "#DCC7EE")
    _box(ax, 34, 86, 32, 8, "Training dataset, e.g. mPower:\n12,053 recordings / 4,809 participants",
         blue, fs=8)
    _box(ax, 4, 70, 24, 7, "TRAIN\nparticipant-disjoint", green, fs=8)
    _box(ax, 37, 70, 26, 7, "VALIDATION 15%\nparticipant-disjoint", yellow, fs=8)
    _box(ax, 72, 70, 24, 7, "TEST: touched once", grey, fs=8)
    for x in (16, 50, 84):
        _arrow(ax, (50, 85.6), (x, 77.6), lw=1.0)
    _box(ax, 4, 59, 24, 6, "head weights", pink, fs=8)
    for i, t in enumerate(["early-stopping epoch", "calibrator (best Brier)\n+ ranking map",
                           "operating threshold (Youden)"]):
        _box(ax, 37, 59 - i * 8.5, 26, 6, t, pink, fs=7.6)
    _arrow(ax, (16, 69.6), (16, 65.6), lw=1.0)
    _arrow(ax, (50, 69.6), (50, 65.6), lw=1.0)
    for i in range(2):
        _arrow(ax, (50, 58.6 - i * 8.5), (50, 57.1 - i * 8.5 - 0.5), lw=1.0)
    _box(ax, 4, 27, 92, 7, "LOCK: weights + calibrator + ranking map + threshold frozen before "
         "any external dataset is opened", purple, fs=8.5)
    _arrow(ax, (16, 58.6), (16, 34.6), lw=1.0)
    _arrow(ax, (50, 41.6), (50, 34.6), lw=1.0)
    ax.text(4, 24, "Not fitted on TRAIN: the head has\nalready seen it and is overconfident "
                   "there.\nNot fitted on TEST: that would use\nthe answers to set the "
                   "instrument.", fontsize=6.8, color="#555555", va="top")
    _box(ax, 24, 3, 52, 9, "Every other dataset (NeuroVoz, IPVS, MDVR-KCL,\nBridge2AI): scored by "
         "the locked model, never fitted", grey, fs=8)
    _arrow(ax, (50, 26.6), (50, 12.6), lw=1.0)
    _save(fig, "fig4_2_split_lock")


def main() -> None:
    warnings.simplefilter("ignore")
    f_protocol(); f_depth(); f_labels_vs_age(); f_windowing()
    f_probes(); f_transfer_b(); f_agegap(); f_experiment_map(); f_confound_diagram()
    f_preprocessing(); f_pipeline(); f_split_lock(); f_score_distributions()
    # f_two_designs() is no longer called: fig4_3_two_designs.pdf is now the author's
    # hand-made version, and regenerating it here would overwrite that.
    n = len(list(OUT.glob("*.pdf")))
    print(f"\n{n} figures written to {OUT}")


if __name__ == "__main__":
    main()
