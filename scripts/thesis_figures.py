"""Generate every data figure the thesis uses, from the artifacts.

Same contract as scripts/thesis_tables.py, and for the same reason: the previous
hand-made figures in outputs/figures/writeup/ drifted from the artifacts they describe
(methods.png shows SSL 0.805 where the artifact now says 0.724). Figures are regenerated
from outputs/logs/ on demand and written to thesis/figures/.

Diagrams -- the pipeline block diagram and the split/lock schematic -- are NOT here.
Those are drawn by hand and saved as PDF in thesis/figures/.

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

OUT = Path(__file__).resolve().parents[1] / "thesis" / "figures"

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
    ax.bar(x - w/2, conf, w, label="Field protocol (confounded)", color=C_CONF)
    ax.bar(x + w/2, matc, w, label="Controlled protocol (age-matched)", color=C_MATCH)
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
                label="Confounded cohort")
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
    import re
    d = pd.DataFrame(_j("label_sensitivity.json")["summary"])
    f = lambda v: float(re.match(r"([\d.]+)", v).group(1))
    base = d[d.scenario == "baseline"].iloc[0]
    rows = [("Confident labels only", d[d.scenario == "confident"].iloc[0]),
            ("Age-matched", d[d.scenario == "agematched"].iloc[0])]
    tg = ["ipvs", "neurovoz", "mdvr_kcl"]
    b_w, b_t = f(base["within"]), np.mean([f(base[t]) for t in tg])

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(rows)); w = 0.38
    dw = [f(r["within"]) - b_w for _, r in rows]
    dt = [np.mean([f(r[t]) for t in tg]) - b_t for _, r in rows]
    ax.bar(x - w/2, dw, w, label="Within mPower", color=C_CONF)
    ax.bar(x + w/2, dt, w, label="Mean external transfer", color=C_MATCH)
    ax.axhline(0, color="k", lw=0.9)
    for xi, (a, m) in enumerate(zip(dw, dt)):
        ax.text(xi - w/2, a + (.006 if a >= 0 else -.018), f"{a:+.3f}",
                ha="center", fontsize=7.5)
        ax.text(xi + w/2, m + (.006 if m >= 0 else -.018), f"{m:+.3f}",
                ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in rows])
    ax.set_ylabel("Change in AUROC from baseline")
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
    """Speech-trained transfer, as a matrix."""
    d = _j("speech_agematch.json")
    b = pd.DataFrame(d["runs"]); b = b[b.cohort == "full"]
    tg = [c for c in b.columns if c.startswith("-> ")]
    g = b.groupby("source")[["within"] + tg].mean(numeric_only=True)
    names = {"ipvs": "IPVS", "neurovoz": "NeuroVoz", "mdvr_kcl": "MDVR-KCL"}
    srcs = ["mdvr_kcl", "neurovoz", "ipvs"]
    M = np.full((len(srcs), len(tg) + 1), np.nan)
    for i, s in enumerate(srcs):
        M[i, 0] = g.loc[s, "within"]
        for j, t in enumerate(tg):
            M[i, j + 1] = g.loc[s, t]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(tg) + 1))
    ax.set_xticklabels(["Within"] + [names[t[3:]] for t in tg], fontsize=8)
    ax.set_yticks(range(len(srcs))); ax.set_yticklabels([names[s] for s in srcs], fontsize=8)
    ax.set_xlabel("Tested on"); ax.set_ylabel("Trained on")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.82 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="AUROC")
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


def f_two_designs() -> None:
    """Figure 4.3: the two transfer designs side by side, drawn from the logged results.

    Replaces the hand-exported Excalidraw diagram, which was never exported; the numbers
    in the boxes are read from the same logs as Tables 4.5 to 4.7.
    """
    from matplotlib.patches import FancyBboxPatch

    ta = []
    for ds, task in [("neurovoz", "vowel"), ("neurovoz", "ddk"), ("mdvr_kcl", "speech"),
                     ("ipvs", "vowel"), ("ipvs", "speech"), ("neurovoz", "speech")]:
        ta.append(_j(f"layer2__xls-r-300m__{ds}__{task}__L11.json")["auroc"])
    sp = pd.DataFrame(_j("speech_agematch.json")["runs"])
    tg = [c for c in sp.columns if c.startswith("-> ")]

    def span(cohort):
        v = sp[sp.cohort == cohort].groupby("source")[tg].mean(numeric_only=True).to_numpy()
        v = v[~np.isnan(v)]
        return v.min(), v.max()

    fl, fh = span("full")
    ml, mh = span("age-matched")
    fills = {"data": "#DCE6F2", "train": "#F6DDE4", "target": "#EDEDED",
             "bad": "#F4CCCC", "mixed": "#FFF2CC"}

    with plt.rc_context({"axes.grid": False, "figure.autolayout": False}):
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10.6)
        ax.axis("off")

        def box(x, y, text, kind, w=4.0, h=1.45):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.18",
                                        fc=fills[kind], ec="#444444", lw=0.9))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5,
                    linespacing=1.35)

        def arrow(x, y0, y1):
            ax.annotate("", xy=(x, y1), xytext=(x, y0),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#444444"))

        for x0, title in ((0.3, "Design A: the proposed system"),
                          (5.7, "Design B: a separate observation")):
            ax.text(x0 + 2.0, 10.25, title, ha="center", va="center", fontsize=10.5,
                    fontweight="bold")
        ys = [8.35, 6.05, 3.75, 1.45]
        a = ["mPower sustained vowels\nself-reported labels",
             "train the head, then lock\ncalibration and threshold",
             "NeuroVoz, IPVS, MDVR-KCL\nvowel, DDK and speech",
             f"AUROC {min(ta):.3f} to {max(ta):.3f}\nthree of six runs inverted"]
        b = ["one clinical dataset\nconnected speech\nclinician labels",
             "a fresh head on the\nsame frozen encoder",
             "the other clinical datasets\nconnected speech",
             f"unmatched {fl:.3f} to {fh:.3f}\nage-matched {ml:.3f} to {mh:.3f}"]
        kinds_a = ["data", "train", "target", "bad"]
        kinds_b = ["data", "train", "target", "mixed"]
        for x, texts, kinds in ((0.3, a, kinds_a), (5.7, b, kinds_b)):
            for y, t, k in zip(ys, texts, kinds):
                box(x, y, t, k)
            for y0, y1 in zip(ys[:-1], ys[1:]):
                arrow(x + 2.0, y0, y1 + 1.45)
        ax.text(5.0, 0.55,
                "The encoder is the same frozen model in both designs. Design B trains a new head "
                "from random\ninitialisation, on a different task, dataset and label source, so it "
                "does not repair design A.",
                ha="center", va="center", fontsize=8.5, style="italic", color="#333333")
        _save(fig, "fig4_3_two_designs")


def f_experiment_map() -> None:
    """Which dataset trains, which tests, and under which design: one grid for the reader."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    names = ["mPower", "Bridge2AI", "NeuroVoz", "IPVS", "MDVR-KCL"]
    sub = ["English, vowel\nself-report", "English, vowel/DDK/speech\nclinician",
           "Spanish\nclinician", "Italian\nclinician", "English\nclinician"]
    cells = {  # (row = trained on, col = tested on) -> (code, colour)
        (0, 0): ("W", "#DDDDDD"), (0, 1): ("D", "#B07AA1"), (0, 2): ("A", "#4E79A7"),
        (0, 3): ("A", "#4E79A7"), (0, 4): ("A", "#4E79A7"),
        (1, 0): ("C", "#59A14F"), (1, 1): ("W+S", "#DDDDDD"), (1, 2): ("C", "#59A14F"),
        (1, 3): ("C", "#59A14F"), (1, 4): ("C", "#59A14F"),
        (2, 2): ("W", "#DDDDDD"), (2, 3): ("B", "#F28E2B"), (2, 4): ("B", "#F28E2B"),
        (3, 2): ("B", "#F28E2B"), (3, 3): ("W", "#DDDDDD"), (3, 4): ("B", "#F28E2B"),
        (4, 2): ("B", "#F28E2B"), (4, 3): ("B", "#F28E2B"), (4, 4): ("W", "#DDDDDD"),
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
    items = [("A", "#4E79A7", "Design A: the proposed model, trained on mPower vowels, applied to each clinical dataset"),
             ("B", "#F28E2B", "Design B: a head trained on one clinical dataset's speech, applied to the other two"),
             ("C", "#59A14F", "Bridge2AI as source: vowel, DDK and speech heads applied to the other datasets"),
             ("D", "#B07AA1", "Bridge2AI as target: the mPower vowel model applied to Bridge2AI"),
             ("W", "#DDDDDD", "Within one dataset: participant-disjoint cross-validation; S = PD against other disorders")]
    for i, (code, col, txt) in enumerate(items):
        y = 16 - i * 3.0
        leg.add_patch(plt.Rectangle((1, y - 1.1), 3.2, 2.2, fc=col, ec="#888888", lw=0.5))
        leg.text(2.6, y, code, ha="center", va="center", fontsize=7, fontweight="bold",
                 color="black" if col == "#DDDDDD" else "white")
        leg.text(5.5, y, txt, va="center", fontsize=7.5)
    leg.text(1, 0.6, "Every cell is run on the data as published and again with age controlled "
             "where age is recorded. Design A: one run per configuration, five seeds for the age "
             "analysis; B, C, D, S: five seeds.", fontsize=7.5, style="italic")
    _save(fig, "fig4_0_experiment_map")


def main() -> None:
    warnings.simplefilter("ignore")
    f_protocol(); f_depth(); f_labels_vs_age(); f_windowing()
    f_probes(); f_transfer_b(); f_agegap(); f_experiment_map()
    # f_two_designs() is no longer called: fig4_3_two_designs.pdf is now the author's
    # hand-made version, and regenerating it here would overwrite that.
    n = len(list(OUT.glob("*.pdf")))
    print(f"\n{n} figures written to {OUT}")


if __name__ == "__main__":
    main()
