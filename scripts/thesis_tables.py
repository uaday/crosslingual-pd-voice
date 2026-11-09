"""Generate every LaTeX table the thesis uses, from the artifacts.

Nothing here is typed by hand. The stale-figure problem this project already hit --
`outputs/figures/writeup/methods.png` still shows SSL 0.805 when the artifact says 0.724 --
happens because a figure or table is made once and the numbers move underneath it. Every
table is therefore regenerated from `outputs/logs/` on demand, and the thesis inputs
the result rather than containing numbers of its own.

Output: outputs/tables/*.tex, one file per table, booktabs only (the Unitec template loads
booktabs and both sample theses use it with no vertical rules).

Run:  .venv\\Scripts\\python.exe scripts\\thesis_tables.py
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pdvoice.config import LOGS

OUT = Path(__file__).resolve().parents[1] / "outputs" / "tables"


def _w(name: str, body: str, caption: str, label: str, note: str = "") -> None:
    """Write one table file. `body` is the tabular block only."""
    OUT.mkdir(parents=True, exist_ok=True)
    tex = ["\\begin{table}[htbp]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}", body.rstrip()]
    if note:
        tex.append(f"\\\\[0.2cm]\n\\begin{{minipage}}{{0.92\\linewidth}}"
                   f"\\footnotesize {note}\\end{{minipage}}")
    tex.append("\\end{table}")
    (OUT / name).write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  {name}")


def _tab(header: list[str], rows: list[list[str]], align: str) -> str:
    # An "L" column (defined in Main.tex as a ragged-right X column) makes this a
    # tabularx at exactly the text width, so the table cannot run into the margin.
    env, width = ("tabularx", "{\\linewidth}") if "L" in align else ("tabular", "")
    out = [f"\\begin{{{env}}}{width}{{{align}}}", "\\toprule",
           " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", f"\\end{{{env}}}"]
    return "\n".join(out)


def _j(name):
    return json.loads((LOGS / name).read_text())


# --------------------------------------------------------------- Chapter 3
NAMES = {"mpower": "mPower", "neurovoz": "NeuroVoz", "ipvs": "IPVS", "mdvr_kcl": "MDVR-KCL",
         "bridge2ai": "Bridge2AI"}


def _cohort():
    """Cohort facts computed by scripts/cohort_summary.py from the experiment loaders."""
    return _j("cohort_summary.json")


def t_datasets() -> None:
    c = _cohort()
    meta = {"mpower": ("English", "vowel", "self-report", "Training"),
            "neurovoz": ("Spanish", "vowel, speech, DDK", "clinician", "Test; training in Exp.~2"),
            "ipvs": ("Italian", "vowel, speech", "clinician", "Test; training in Exp.~2"),
            "mdvr_kcl": ("English", "speech", "clinician", "Test; training in Exp.~2"),
            "bridge2ai": ("English", "vowel, speech, DDK", "clinician", "Training and test")}
    rows = []
    for ds, (lang, tasks, labels, role) in meta.items():
        rows.append([NAMES[ds], lang, f"{c[ds]['raw_recordings']:,}",
                     f"{c[ds]['raw_speakers']:,}", tasks, labels, role])
    m = c["mpower"]
    _w("t3_1_datasets.tex",
       "{\\small\\setlength{\\tabcolsep}{4pt}\n"
       + _tab(["Dataset", "Language", "Recordings", "Speakers", "Tasks", "Labels",
               "Role"], rows, "llrrLlL") + "\n}",
       "Datasets used in this thesis. Counts are taken from the loaders before quality "
       "control and capping.",
       "tab:datasets",
       f"The mPower counts include {m['unlabelled_recordings']:,} recordings from "
       f"{m['unlabelled_speakers']:,} participants with no diagnosis label, which are "
       f"excluded, leaving {m['raw_recordings'] - m['unlabelled_recordings']:,} recordings from "
       f"{m['raw_speakers'] - m['unlabelled_speakers']:,} participants. "
       "Of the 767 Bridge2AI participants, 98 have a clinician-confirmed "
       "diagnosis of Parkinson's disease (38 recruited in the USA, 60 in Canada) and 118 are "
       "healthy controls, all recruited in the USA; patients are compared with controls "
       "within the USA only (Section~\\ref{subsec:bridge2ai}). The rest have other "
       "voice, neurological, mood or respiratory conditions and are used only in the "
       "specificity analysis of Section~\\ref{subsec:specificity}. Three English datasets "
       "are used, so that a change of language can be separated from a change of dataset.")


def t_demographics() -> None:
    c = _cohort()
    rows = []
    for ds in ("mpower", "ipvs", "neurovoz", "mdvr_kcl", "bridge2ai"):
        r = c[ds]
        name = NAMES[ds] + (" (USA)" if ds == "bridge2ai" else "")
        if r["age_pd"] is None:
            rows.append([name, "--", "--", "--", "\\textbf{0\\%}", "--"])
            continue
        gap = r["age_pd"] - r["age_hc"]
        g = f"{gap:+.1f}"
        rows.append([name, f"{r['age_pd']:.1f}", f"{r['age_hc']:.1f}",
                     f"\\textbf{{{g}}}" if gap > 15 else g,
                     f"{100 * r['age_coverage']:.1f}\\%".replace("100.0", "100"),
                     f"{100 * r['sex_coverage']:.1f}\\%".replace("100.0", "100")])
    ip = c["ipvs"]
    _w("t3_2_demographics.tex",
       _tab(["Dataset", "PD age", "HC age", "Gap (yr)", "Age coverage", "Sex coverage"],
            rows, "lrrrrr"),
       "Subject-level age and sex coverage by dataset.",
       "tab:demographics",
       "Participants with at least one usable recording, as used in the experiments. "
       f"The external datasets are confounded as well: IPVS carries a "
       f"{ip['age_pd'] - ip['age_hc']:.1f}-year gap of its "
       "own. MDVR-KCL has no age data at all and therefore cannot be age-corrected under "
       "any analysis. Bridge2AI repeats the mPower gap. Every dataset except MDVR-KCL "
       "records participant sex.")


def t_prevalence() -> None:
    c = _cohort()
    rows = []
    for ds in ("mpower", "neurovoz", "ipvs", "mdvr_kcl", "bridge2ai"):
        r = c[ds]
        name = {"mpower": "mPower (3 per participant)",
                "bridge2ai": "Bridge2AI (USA)"}.get(ds, NAMES[ds])
        rows.append([name, f"{r['n_pd']:,}", f"{r['n_hc']:,}", f"{r['prevalence']:.3f}"])
    small = [c[d]["prevalence"] for d in ("neurovoz", "ipvs", "mdvr_kcl")]
    _w("t3_3_prevalence.tex",
       _tab(["Dataset", "PD", "HC", "Prevalence"], rows, "lrrr"),
       "Class balance at subject level. Prevalence is the chance level for AUPRC.",
       "tab:prevalence",
       "Prevalence is the chance level for AUPRC, so an AUPRC is read only beside it.")


def t_baselines() -> None:
    c = pd.read_csv(LOGS / "classifier_comparison.csv")
    best = c.groupby(["cohort", "features"]).AUROC.max()
    order = [("age only (1)", "Age alone (1 feature)"),
             ("classical (42)", "Classical acoustic (42)"),
             ("SSL embeddings (2048)", "SSL embeddings (2048)")]
    rows = []
    for key, label in order:
        conf, am = best[("confounded", key)], best[("age-matched", key)]
        b = (lambda v: f"\\textbf{{{v:.3f}}}")
        rows.append([label,
                     b(conf) if key.startswith("age") else f"{conf:.3f}",
                     b(am) if key.startswith("SSL") else f"{am:.3f}"])
    _w("t3_4_baselines.tex",
       _tab(["Feature set", "Uncorrected cohort", "Age-matched cohort"], rows, "lrr"),
       "Subject-level AUROC on mPower by feature set, best of five classifiers, under the "
       "two evaluation protocols.",
       "tab:baselines",
       "On the uncorrected cohort a single demographic variable outperforms a "
       "315-million-parameter speech encoder. After age matching the demographic baseline "
       "falls to chance and the voice model does not. Taking the best of five classifiers "
       "lifts age alone slightly above chance, to 0.531; the single-classifier baseline used "
       "in the rest of the thesis is 0.505.")


# --------------------------------------------------------------- Chapter 4
ENCODER_CITE = {
    "xls-r-300m": " \\cite{babu2022xlsr}",
    "hubert-large": " \\cite{hsu2021hubert}",
    "wav2vec2-base": " \\cite{baevski2020}",
    "wavlm-large": " \\cite{chen2022wavlm}",
    "whisper-small": " \\cite{radford2023whisper}",
}


def t_encoders_confounded() -> None:
    rows = []
    for stem, label, cfgs in [
        ("xls-r-300m__cap3__LNone", "XLS-R-300M \\cite{babu2022xlsr}", "5"),
        ("hubert-large__cap3__L12", "HuBERT-large \\cite{hsu2021hubert}", "2"),
        ("wav2vec2-base__cap3__LNone", "wav2vec2-base \\cite{baevski2020}", "\\textbf{1}"),
        ("wavlm-large__cap3__L12", "WavLM-large \\cite{chen2022wavlm}", "2"),
    ]:
        d = _j(f"layer1_mpower__{stem}.json")
        lo, hi = d["auroc_ci"]
        bold = label.startswith("XLS-R")
        a = f"\\textbf{{{d['auroc']:.3f}}}" if bold else f"{d['auroc']:.3f}"
        rows.append([label, a, f"[{lo:.3f}, {hi:.3f}]", cfgs])
    _w("t4_1_encoders_confounded.tex",
       _tab(["Encoder", "AUROC", "95\\% CI", "Configurations tried"], rows, "lrrr"),
       "Encoder comparison on the uncorrected cohort, as originally conducted.",
       "tab:encoders-confounded",
       "Configurations tried: layer and pooling settings searched per encoder. The protocol's "
       "two defects, an unequal search budget and a confounded cohort, are discussed in the "
       "text.")


def t_encoders_matched() -> None:
    d = _j("encoder_layer_sweep.json")["best"]
    rows = []
    for r in sorted(d, key=lambda x: -x["best_layer_auroc"]):
        used = r["model"] == "xls-r-300m"
        name = f"{r['model']}" + ENCODER_CITE.get(r["model"], "") + (" $^{*}$" if used else "")
        rows.append([name, str(r["best_layer"]), f"{r['best_layer_auroc']:.3f}",
                     f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
                     f"{r['last_layer_auroc']:.3f}",
                     f"+{r['best_layer_auroc'] - r['last_layer_auroc']:.3f}"])
    _w("t4_2_encoders_matched.tex",
       _tab(["Encoder", "Best layer", "AUROC", "95\\% CI", "Final layer", "Gain"],
            rows, "lrrrrr"),
       "Encoder comparison on the age-matched cohort with every layer of every encoder "
       "searched under an equal budget, read by a linear probe.",
       "tab:encoders-matched",
       "Every confidence interval overlaps every other. The nominal leader is the "
       "smallest model at its embedding layer, before any transformer block is applied. "
       "Every encoder peaks in an early or middle layer; the final layer is below the "
       "encoder's median layer for four of the five, and the worst for Whisper. "
       "$^{*}$ the encoder retained for the remaining experiments, for its multilingual "
       "pre-training rather than for benchmark performance.")


def t_protocol() -> None:
    c = pd.read_csv(LOGS / "classifier_comparison.csv")
    best = c.groupby(["cohort", "features"]).AUROC.max()
    rows = [
        ["Age alone (1 feature)", f"\\textbf{{{best[('confounded','age only (1)')]:.3f}}}",
         f"{best[('age-matched','age only (1)')]:.3f}"],
        ["Classical acoustic (42)", f"{best[('confounded','classical (42)')]:.3f}",
         f"{best[('age-matched','classical (42)')]:.3f}"],
        ["SSL embeddings (2048)", f"{best[('confounded','SSL embeddings (2048)')]:.3f}",
         f"\\textbf{{{best[('age-matched','SSL embeddings (2048)')]:.3f}}}"],
        ["\\midrule Within-to-transfer gap", "\\textbf{32.1 pp}", "\\textbf{1.2 pp}"],
    ]
    _w("t4_3_protocol.tex",
       _tab(["", "Uncorrected cohort", "Age-matched cohort"],
            rows, "lrr"),
       "The same model, the same audio and the same splits, evaluated under two protocols.",
       "tab:protocol",
       "Under the protocol the field uses, a single demographic variable beats the speech "
       "model. Under a protocol that controls for demographic composition, the ordering "
       "reverses. Only the evaluation protocol differs between the two columns.")


def t_significance() -> None:
    f = _j("fusion__xls-r-300m__L11.json")
    dc = _j("difference_cis.json")
    keys = [("SSL-only vs classical-only", "SSL vs classical"),
            ("SSL-only vs FUSION", "SSL vs combined"),
            ("classical-only vs FUSION", "Classical vs combined")]
    rows = []
    for coh, label in (("baseline", "Uncorrected"), ("age-matched", "Age-matched")):
        t = f["scenarios"][coh]["_tests"]
        for k, name in keys:
            v = t[k]
            fmt = lambda x: f"{x:.4f}"
            hd = dc["holm"][coh][k]["p_delong"]["holm"]
            # One convention throughout: first-named minus second-named. difference_cis
            # stores two of the three as FUSION minus the other, so those are negated
            # (with the interval endpoints swapped). The uncorrected delta from fusion.py
            # is already first minus second; it used to be printed as abs(), losing the sign.
            dk, sgn = {"SSL-only vs classical-only": ("SSL_only - classical_only", 1),
                       "SSL-only vs FUSION": ("FUSION - SSL_only", -1),
                       "classical-only vs FUSION": ("FUSION - classical_only", -1)}[k]
            if coh == "age-matched":
                fs = dc["feature_sets"][dk]
                lo, hi = sorted(sgn * x for x in fs["ci95"])
                diff = f"{sgn * fs['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            else:
                diff = f"{v['delta']:+.3f}"
            rows.append([label, name, diff, fmt(v["p_delong"]), f"{hd:.3f}",
                         f"{v['p_bootstrap']:.3f}", fmt(v["p_mcnemar"])])
    _w("t4_4_significance.tex",
       "{\\footnotesize\\setlength{\\tabcolsep}{2pt}\n"
       + _tab(["Cohort", "Comparison", "$\\Delta$ AUROC", "DeLong", "Holm",
               "Bootstrap", "McNemar"], rows, "llcrrrr") + "\n}",
       "Paired significance tests between feature sets, on the same subjects and splits.",
       "tab:significance",
       "Framework head, one recording per participant, layer 11, one seed. "
       "$\\Delta$ AUROC is the first-named model minus the second. "
       "The last four columns are $p$ values. "
       "DeLong and the paired bootstrap compare ranking; McNemar compares the decisions "
       "made at each model's operating threshold. Holm: DeLong $p$ adjusted within each "
       "cohort's three comparisons. On the age-matched cohort the difference column gives "
       "the paired-bootstrap 95\\% interval of the AUROC difference (1,000 resamples). With "
       "the confound removed, the self-supervised branch still beats the classical one, by "
       "about 0.05, and combining the two adds nothing.")


def t_transfer_a() -> None:
    def g(ds, task, layer, w="", target=""):
        tag = ("" if not w else f"__W{w}") + target
        p = LOGS / f"layer2__xls-r-300m__{ds}__{task}__L{layer}{tag}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return d["auroc"], d["auroc_ci"], d["n_subjects"]

    rows = []
    for ds, task, label in [("neurovoz", "vowel", "NeuroVoz / vowel"),
                            ("neurovoz", "ddk", "NeuroVoz / DDK"),
                            ("mdvr_kcl", "speech", "MDVR-KCL / speech"),
                            ("ipvs", "vowel", "IPVS / vowel"),
                            ("ipvs", "speech", "IPVS / speech"),
                            ("neurovoz", "speech", "NeuroVoz / speech")]:
        r = g(ds, task, 11)
        if not r:
            continue
        a, (lo, hi), n = r
        inv = "\\textbf{yes}" if hi < 0.5 else "--"
        rows.append([label, str(n), f"{a:.3f}", f"[{lo:.3f}, {hi:.3f}]", inv])
    # Bridge2AI, the fourth test set: as released (patients ~29 years older than controls,
    # as in mPower) and on an age-matched subset scored by the same locked predictions.
    for task, tname in (("vowel", "vowel"), ("ddk", "DDK"), ("speech", "speech")):
        for target, suffix in (("", ""), ("__agematched-target", ", age-matched")):
            r = g("bridge2ai", task, 11, target=target)
            if not r:
                continue
            a, (lo, hi), n = r
            inv = "\\textbf{yes}" if hi < 0.5 else "--"
            rows.append([f"Bridge2AI / {tname}{suffix}", str(n), f"{a:.3f}",
                         f"[{lo:.3f}, {hi:.3f}]", inv])
    ceiling = _j("layer1_mpower__xls-r-300m__cap3__L11.json")["auroc"]
    allruns = [json.loads(f.read_text()) for f in LOGS.glob("layer2__xls-r-300m__*.json")
               if "agematched-target" not in f.name]
    n_inv = sum(r["auroc_ci"][1] < 0.5 for r in allruns)
    n_inv_l11 = sum(1 for r in rows if "yes" in r[-1] and "age-matched" not in r[0])
    n_pairs = sum(1 for r in rows if "age-matched" not in r[0])
    _w("t4_5_transfer_a.tex",
       _tab(["Target dataset / task", "$n$", "AUROC", "95\\% CI", "Inverted"], rows,
            "lrrrc"),
       "Design A: transfer of the locked mPower-trained model to each external dataset, at "
       "layer 11.",
       "tab:transfer-a",
       f"Single run. The within-mPower ceiling at this layer is {ceiling:.3f}. "
       "\\emph{Inverted} marks results whose entire confidence interval lies below 0.500, "
       f"meaning the model ranks patients beneath controls: {n_inv_l11} of the {n_pairs} "
       "dataset and task pairs at this layer. Bridge2AI rows marked age-matched rescore the "
       "same locked predictions on patients and controls matched in five-year age bins; "
       "the other three datasets are scored as released. "
       f"Across all {len(allruns)} runs of this design (layers 11, 15 and 24, two windows for "
       f"the three clinical datasets, with and without adaptation), {n_inv} are inverted, "
       "most of them on connected speech.")


def t_transfer_b() -> None:
    d = _j("speech_agematch.json")
    b = pd.DataFrame(d["runs"])
    b = b[b.cohort == "full"]
    tg = [c for c in b.columns if c.startswith("-> ")]
    g = b.groupby("source")[["within"] + tg].mean(numeric_only=True)
    names = {"ipvs": "IPVS", "neurovoz": "NeuroVoz", "mdvr_kcl": "MDVR-KCL"}
    rows = []
    for src in ("mdvr_kcl", "neurovoz", "ipvs"):
        r = g.loc[src]
        cells = [f"{r[t]:.3f}" if pd.notna(r[t]) else "--" for t in tg]
        rows.append([names[src], f"{r['within']:.3f}"] + cells)
    _w("t4_6_transfer_b.tex",
       _tab(["Trained on", "Within"] + [f"$\\rightarrow$ {names[t[3:]]}" for t in tg],
            rows, "lrrrr"),
       "Transfer between external datasets when a new head is trained on connected speech, "
       "averaged over five seeds.",
       "tab:transfer-b",
       "Design B. The encoder is the same frozen model used throughout; only the head is "
       f"retrained. Every direction transfers, at {np.nanmin(g[tg].to_numpy()):.3f}--"
       f"{np.nanmax(g[tg].to_numpy()):.3f}, against within-dataset scores of "
       f"{g['within'].min():.3f}--{g['within'].max():.3f}.")


def t_agecontrol() -> None:
    d = _j("speech_agematch.json")
    df = pd.DataFrame(d["runs"])
    g = df.groupby(["source", "cohort"]).mean(numeric_only=True)
    names = {"ipvs": "IPVS", "neurovoz": "NeuroVoz", "mdvr_kcl": "MDVR-KCL"}
    rows = []
    for src in ("neurovoz", "ipvs"):
        for t in [c for c in df.columns if c.startswith("-> ")]:
            try:
                full = g.loc[(src, "full"), t]
                am = g.loc[(src, "age-matched"), t]
                sr = g.loc[(src, "size-matched random"), t]
            except KeyError:
                continue
            if pd.isna(full):
                continue
            rows.append([f"{names[src]} $\\rightarrow$ {names[t[3:]]}",
                         f"{full:.3f}", f"{am:.3f}", f"{sr:.3f}",
                         f"\\textbf{{{am - sr:+.3f}}}"])
    _w("t4_7_agecontrol.tex",
       "{\\small\n"
       + _tab(["Direction", "Full", "Age-matched", "Size-matched random",
               "Due to age"], rows,
              "Lrrrr") + "\n}",
       "Three-arm control separating the effect of age matching from the effect of the "
       "sample size it removes.",
       "tab:agecontrol",
       "Age matching also halves the cohort, so a drop could be caused by the matching or "
       "by the lost data. The third arm removes the same number of speakers at random and "
       "leaves the age gap intact, so only the difference between the second and third "
       "columns is attributable to age. Age matching \\emph{reduces} speech transfer, "
       "which is the opposite of its effect on the vowel model.")


def t_deviation() -> None:
    rows = [
        ["Klemp\\'i\\v{r} 2024 \\cite{klempir2024}", "IPVS 0.99",
         "Italian $\\rightarrow$ English 0.72 / 0.78", "+21 to +27"],
        ["Klemp\\'i\\v{r} 2024 \\cite{klempir2024}", "MDVR-KCL 0.80",
         "English $\\rightarrow$ Italian 0.90 / 0.98", "$-$10 to $-$18"],
        ["Kopar et al.\\ 2026 \\cite{kopar2026}", "--", "change of language, balanced accuracy",
         "+16"],
        *_thesis_deviation_rows(),
    ]
    _w("t4_8_deviation.tex",
       "{\\small\n"
       + _tab(["Study", "Within", "Transfer", "Deviation (pp)"], rows,
              "P{3.3cm}P{2.1cm}LP{2.1cm}") + "\n}",
       "Cross-dataset deviation reported in the literature, and in this thesis.",
       "tab:deviation",
       "Where transfer degrades, the published drops are roughly 16 to 27 percentage points, "
       "and one direction in Klemp\\'i\\v{r}'s work improves rather than degrades. Experiment 2 "
       "sits below that band on average and reproduces the direction-dependence. The age-matched arm falls far below the "
       "published band because the \\emph{home} benchmark was made honest, not because "
       "transfer improved. Kopar et al.\\ report balanced accuracy, not AUROC, so their "
       "figure is on a different scale from the others. The Experiment 2 row leaves out IPVS "
       "as a training source, because its within-dataset score is a recording artefact "
       "(Section~\\ref{sec:within}).")


def _thesis_deviation_rows():
    """This thesis's rows, from the unified layer-11 experiments."""
    R = pd.read_csv(LOGS / "unified" / "runs.csv")
    clin = [("ipvs", "vowel"), ("neurovoz", "vowel"), ("mdvr_kcl", "speech")]

    def e1(cond):
        w = R[(R.exp == "within") & (R.source == "mpower") & (R.condition == cond)].auroc.mean()
        g = R[(R.exp == "exp1") & (R.condition == cond) & (R.target_cohort == "released")]
        m = np.zeros(len(g), bool)
        for t, tt in clin:
            m |= ((g.target == t) & (g.target_task == tt)).to_numpy()
        return w, g[m].groupby("seed").auroc.mean().mean()
    rows = []
    for cond, name in (("uncorrected", "\\midrule \\textbf{This thesis}, Experiment 1 (uncorrected)"),
                       ("agematched", "\\textbf{This thesis}, Experiment 1 (age-matched)")):
        w, t = e1(cond)
        rows.append([name, f"{w:.3f}", f"{t:.3f}", f"\\textbf{{{100 * (w - t):+.1f}}}"])
    # IPVS is left out as a training source: its within-dataset score (1.000) is a recording
    # artefact, not a measurement of voice (Section 4.3), so a deviation from it means nothing
    W = R[(R.exp == "within") & (R.source_task == "speech") & (R.condition == "uncorrected")
          & (R.source != "ipvs")].groupby("source").auroc.mean()
    E = R[(R.exp == "exp2") & (R.condition == "uncorrected") & (R.target_cohort == "released")
          & (R.source != "ipvs")].groupby(["source", "target"]).auroc.mean()
    dev = [W[s_] - v for (s_, _), v in E.items()]
    rows.append(["\\textbf{This thesis}, Experiment 2 (uncorrected)",
                 f"{W.min():.3f}--{W.max():.3f}", f"{E.min():.3f}--{E.max():.3f}",
                 f"\\textbf{{{100 * np.mean(dev):+.1f}}} ({100 * min(dev):+.1f} to "
                 f"{100 * max(dev):+.1f})"])
    return rows


def t_probes() -> None:
    d = _j("embedding_probes.json")
    seen, rows = set(), []
    order = ["task (vowel vs speech)", "sex", "corpus identity (3-way)",
             "language (3-way)", "age, young(<35) vs older(>=55)", "** PD diagnosis **",
             "recording device (4-way)"]
    label = {"task (vowel vs speech)": "Elicitation task", "sex": "Speaker sex",
             "corpus identity (3-way)": "Dataset identity",
             "language (3-way)": "Language",
             "age, young(<35) vs older(>=55)": "Age band",
             "** PD diagnosis **": "\\textbf{Parkinson's diagnosis}",
             "recording device (4-way)": "Recording device"}
    for key in order:
        for r in d:
            if r["attribute"] == key and key not in seen:
                seen.add(key)
                v = f"\\textbf{{{r['value']:.3f}}}" if "PD" in key else f"{r['value']:.3f}"
                rows.append([label[key], r["metric"], v, f"{r['chance']:.2f}"])
                break
    _w("t4_9_probes.tex",
       _tab(["Attribute", "Measure", "Value", "Chance"], rows, "llrr"),
       "Linear probes applied to the frozen embeddings, subject-disjoint.",
       "tab:probes",
       "The diagnosis is measured on the uncorrected cohort. Chance: 0.5 for AUROC; for "
       "accuracy, the share "
       "of the most common class. Language and dataset identity share one label here, "
       "because each pooled dataset has one language. Recording device is not recovered "
       "(below the majority rate), so the probes do not detect everything.")


def t_windowing() -> None:
    def g(ds, task, L, w):
        tag = "" if w == "current" else f"__W{w}"
        p = LOGS / f"layer2__xls-r-300m__{ds}__{task}__L{L}{tag}.json"
        return json.loads(p.read_text())["auroc"] if p.exists() else None

    pairs = [("ipvs", "vowel"), ("neurovoz", "vowel"), ("neurovoz", "ddk"),
             ("ipvs", "speech"), ("neurovoz", "speech"), ("mdvr_kcl", "speech")]
    names = {"ipvs": "IPVS", "neurovoz": "NeuroVoz", "mdvr_kcl": "MDVR-KCL"}
    rows, deltas = [], []
    for ds, task in pairs:
        cells = []
        for L in (11, 15):
            a, b = g(ds, task, L, "current"), g(ds, task, L, "full")
            if a is None or b is None:
                cells += [f"{a:.3f}" if a else "--", f"{b:.3f}" if b else "--", "--"]
            else:
                deltas.append(b - a)
                cells += [f"{a:.3f}", f"{b:.3f}", f"{b - a:+.3f}"]
        rows.append([f"{names[ds]} / {task}"] + cells)
    l1c = _j("layer1_mpower__xls-r-300m__cap3__L15.json")
    l1f = _j("layer1_mpower__xls-r-300m__cap3__L15__Wfull.json")
    w15c, w15f = l1c["auroc"], l1f["auroc"]
    w15c_ci = f"[{l1c['auroc_ci'][0]:.3f}, {l1c['auroc_ci'][1]:.3f}]"
    w15f_ci = f"[{l1f['auroc_ci'][0]:.3f}, {l1f['auroc_ci'][1]:.3f}]"
    _w("t4_10_windowing.tex",
       "{\\small\\setlength{\\tabcolsep}{4pt}\n"
       + _tab(["Dataset / task", "L11 core", "L11 full", "$\\Delta$",
               "L15 core", "L15 full", "$\\Delta$"], rows, "lrrrrrr") + "\n}",
       "Effect of replacing the fixed analysis window with the whole recording, at two "
       "layers.",
       "tab:windowing",
       f"Across the {len(deltas)} comparisons the mean change is "
       f"{np.mean(deltas):+.3f} (standard deviation {np.std(deltas, ddof=1):.3f}, range "
       f"{min(deltas):+.3f} to {max(deltas):+.3f}). Within mPower, at layer 15, the same "
       f"change moves AUROC from {w15c:.3f} {w15c_ci} to {w15f:.3f} {w15f_ci}. A change "
       "with no systematic external effect therefore moves individual external results by "
       f"up to {max(abs(min(deltas)), abs(max(deltas))):.2f}, on cohorts of 37 to 111 "
       "speakers.")


def t_robustness() -> None:
    """One-line summary of every model and protocol variant, against age matching."""
    R = pd.read_csv(LOGS / "unified" / "runs.csv")

    def mean_transfer(cond):
        g = R[(R.exp == "exp1") & (R.condition == cond) & (R.target_cohort == "released")
              & (((R.target.isin(["ipvs", "neurovoz"])) & (R.target_task == "vowel"))
                 | (R.target == "mdvr_kcl"))]
        return g.groupby("target").auroc.mean().mean()

    def within(cond):
        return R[(R.exp == "within") & (R.source == "mpower")
                 & (R.condition == cond)].auroc.mean()

    wu, wa, tu, ta = within("uncorrected"), within("agematched"), \
        mean_transfer("uncorrected"), mean_transfer("agematched")
    sweep = [r["best_layer_auroc"] for r in _j("encoder_layer_sweep.json")["best"]]
    dc = _j("difference_cis.json")
    enc = [v["delta"] for v in dc["encoders"].values()]
    cc = pd.read_csv(LOGS / "classifier_comparison.csv")
    ssl = cc[(cc.cohort == "age-matched") & (cc.features == "SSL embeddings (2048)")].AUROC
    fs = dc["feature_sets"]

    def ci(d, sign=1):
        lo, hi = sorted(sign * x for x in d["ci95"])
        return f"{sign * d['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    ad = {r["scenario"]: r for r in _j("label_sensitivity__L11_adapt_centroid.json")["summary"]}
    ad_t = np.mean([float(ad["agematched"][k].split()[0]) for k in ("ipvs", "neurovoz",
                                                                     "mdvr_kcl")])
    er = _j("demographic_erasure.json")
    l1c = _j("layer1_mpower__xls-r-300m__cap3__L11.json")["auroc"]
    l1f = _j("layer1_mpower__xls-r-300m__cap3__L11__Wfull.json")["auroc"]
    wd = []
    for ds, task in [("ipvs", "vowel"), ("neurovoz", "vowel"), ("neurovoz", "ddk"),
                     ("ipvs", "speech"), ("neurovoz", "speech"), ("mdvr_kcl", "speech")]:
        for L in (11, 15):
            a = LOGS / f"layer2__xls-r-300m__{ds}__{task}__L{L}.json"
            b = LOGS / f"layer2__xls-r-300m__{ds}__{task}__L{L}__Wfull.json"
            if a.exists() and b.exists():
                wd.append(json.loads(b.read_text())["auroc"] - json.loads(a.read_text())["auroc"])
    rows = [
        ["Age matching (reference)", f"{wu:.3f} $\\rightarrow$ {wa:.3f}",
         f"{tu:.3f} $\\rightarrow$ {ta:.3f}", "\\ref{tab:u-exp1}"],
        ["Encoder (5, best layer each)", f"{min(sweep):.3f} to {max(sweep):.3f}; "
         f"vs XLS-R {min(enc):+.3f} to {max(enc):+.3f}", "--", "\\ref{tab:encoders-matched}"],
        ["Classifier (5, embeddings)", f"{ssl.min():.3f} to {ssl.max():.3f}", "--",
         "\\ref{tab:appA-classifiers}"],
        ["Embeddings vs classical", ci(fs["SSL_only - classical_only"]),
         "both invert on IPVS", "\\ref{tab:significance}, \\ref{tab:classical-transfer}"],
        ["Embeddings + classical", ci(fs["FUSION - SSL_only"]), "--", "\\ref{tab:significance}"],
        ["Centroid adaptation", "--", f"{ta:.3f} $\\rightarrow$ {ad_t:.3f}",
         "Section~\\ref{subsec:variants}"],
        ["Erasure of age and sex$^{a}$",
         f"{er['confounded']['within']['original']:.3f} $\\rightarrow$ "
         f"{er['confounded']['within']['erased']:.3f}",
         f"{er['transfer_mean']['confounded']['original']:.3f} $\\rightarrow$ "
         f"{er['transfer_mean']['confounded']['erased']:.3f}", "Section~\\ref{subsec:diagnosis}"],
        ["Whole recording as window$^{a}$", f"{l1c:.3f} $\\rightarrow$ {l1f:.3f}",
         f"mean {np.mean(wd):+.3f} ({len(wd)} tests)", "\\ref{tab:windowing}"],
    ]
    _w("t4_12_robustness.tex",
       "{\\small\\setlength{\\tabcolsep}{3pt}\n"
       + _tab(["Change", "Within mPower", "Transfer", "Detail"], rows, "lLll") + "\n}",
       "Every change to the model or protocol tested, against the effect of matching on age.",
       "tab:robustness",
       "Within mPower: age-matched cohort unless marked; the first row and rows marked $^{a}$ "
       "start from the uncorrected cohort. Transfer: mean AUROC on IPVS vowels, NeuroVoz "
       "vowels and MDVR-KCL speech for the model trained on mPower, except the window row, "
       "which gives the mean change over every external dataset and task tested. Bracketed "
       "values are paired-bootstrap 95\\% intervals of the difference. The erasure and window "
       "rows use their own set-ups (Table~\\ref{tab:protocols}). No change to the model moves "
       "transfer as much as matching on age, and none makes the model transfer.")


def t_classical_transfer() -> None:
    """Both feature families, both cohorts, the same two vowel datasets."""
    cl = _j("classical_transfer__cap3.json")
    R = pd.read_csv(LOGS / "unified" / "runs.csv")
    cond = {"confounded": "uncorrected", "age-matched": "agematched"}

    def ssl(scen, ds=None):
        if ds is None:
            g = R[(R.exp == "within") & (R.source == "mpower") & (R.condition == cond[scen])]
        else:
            g = R[(R.exp == "exp1") & (R.condition == cond[scen]) & (R.target == ds)
                  & (R.target_task == "vowel") & (R.target_cohort == "released")]
        v = g.auroc
        cell = f"{v.mean():.3f}"
        return "\\textbf{" + cell + "}" if ds is not None and v.max() < 0.5 else cell

    rows = []
    for scen, label in (("confounded", "Uncorrected"), ("age-matched", "Age-matched")):
        b_ = cl["scenarios"][scen]
        rows.append([label, "within mPower", f"{b_['within_mpower']['auroc']:.3f}", ssl(scen)])
        for ds, name in (("ipvs", "IPVS"), ("neurovoz", "NeuroVoz")):
            m = b_["transfer"][ds]
            a_ = f"{m['auroc']:.3f}"
            if m["auroc_ci"][1] < 0.5:
                a_ = "\\textbf{" + a_ + "}"
            rows.append([label, "$\\rightarrow$ " + f"{name} ($n={m['n_subjects']}$)", a_,
                         ssl(scen, ds)])
    _w("t4_11_classical_transfer.tex",
       _tab(["Cohort", "Evaluated on", "Classical (42)", "SSL (2048)"], rows, "llrr"),
       "Transfer of both feature families through the identical locked protocol, on the "
       "two vowel datasets.",
       "tab:classical-transfer",
       "The classical branch uses the same head, the same calibrator selection and the "
       "same locked threshold as the self-supervised branch; imputation and scaling are "
       "fitted on mPower training participants alone and reused unchanged. Within mPower: "
       "five-fold participant-disjoint cross-validation for both. Classical: one run; SSL: layer 11, "
       "mean over five seeds (Table~\\ref{tab:u-exp1}). Values in bold are below 0.500 in "
       "the whole confidence interval (classical) or in every seed (SSL). Neither family "
       "transfers, and the IPVS inversion occurs for both, so it is not a property of the "
       "encoder. Only vowel datasets are eligible, because the descriptors assume "
       "sustained phonation.")


def t_master() -> None:
    """One place for every headline number, with the settings that produced it.

    Several headline figures come from different runs (layer, cohort, classifier, seeds),
    and quoting them side by side without saying so made them look inconsistent. Every
    chapter now cites this table when it quotes one of these figures.
    """
    ls = {r["scenario"]: r for r in _j("label_sensitivity.json")["summary"]}
    ls_runs = _j("label_sensitivity.json")["runs"]
    asx = {r["scenario"]: r for r in _j("label_sensitivity__agesex.json")["summary"]}
    ad_runs = _j("label_sensitivity__adapt_centroid.json")["runs"]
    rp = _j("restricted_permutation__fast__n1000.json")
    dc = _j("decision_curves.json")["age-matched | SSL, SVM"]
    b2 = _j(f"bridge2ai_agematch__L{B2L}.json")
    b2c = _j(f"bridge2ai_review_checks__L{B2L}.json")

    def mean_t(runs, sc):
        return float(np.mean([np.nanmean([r[k] for k in ("transfer_ipvs", "transfer_neurovoz", "transfer_mdvr_kcl") if k in r])
                              for r in runs[sc]]))

    def gap(sc):
        rs = ls_runs[sc]
        w = np.mean([r["within"] for r in rs])
        t = np.mean([np.nanmean([r[k] for k in ("transfer_ipvs", "transfer_neurovoz", "transfer_mdvr_kcl") if k in r])
                     for r in rs])
        return 100 * (w - t)

    def pm(s):
        return s.replace("+/-", "$\\pm$")

    def ext(sc):
        r = ls[sc]
        return f"{r['ipvs'].split()[0]} / {r['neurovoz'].split()[0]} / {r['mdvr_kcl'].split()[0]}"

    cc = pd.read_csv(LOGS / "classifier_comparison.csv")
    best = cc.groupby(["cohort", "features"]).AUROC.max()
    mlp = cc[cc.classifier.str.startswith("MLP")].set_index(["cohort", "features"]).AUROC
    lr = cc[cc.classifier.str.startswith("logreg")].set_index(["cohort", "features"]).AUROC
    sel = _j("select_transfer_layer.json")
    sweep = {r["model"]: r for r in _j("encoder_layer_sweep.json")["best"]}
    span = [r["best_layer_auroc"] for r in sweep.values()]
    l11 = _j("layer1_mpower__xls-r-300m__cap3__L11.json")["auroc"]
    ta = []
    for ds, task in [("neurovoz", "vowel"), ("neurovoz", "ddk"), ("mdvr_kcl", "speech"),
                     ("ipvs", "vowel"), ("ipvs", "speech"), ("neurovoz", "speech")]:
        f = LOGS / f"layer2__xls-r-300m__{ds}__{task}__L11.json"
        if f.exists():
            ta.append(json.loads(f.read_text())["auroc"])
    sp = pd.DataFrame(_j("speech_agematch.json")["runs"])
    tg = [c for c in sp.columns if c.startswith("-> ")]
    def span_of(cohort):
        g = sp[sp.cohort == cohort].groupby("source")[tg].mean(numeric_only=True)
        v = g.to_numpy().ravel()
        v = v[~np.isnan(v)]
        return f"{v.min():.3f}--{v.max():.3f}"
    ob = [r for r in _j("objective_function.json")["rows"]
          if r["cohort"] == "age-matched" and r["features"].startswith("SSL")
          and r["classifier"].startswith("SVM")][0]

    S = "\\textsuperscript{a}"
    rows = [
        ["\\multicolumn{6}{l}{\\emph{Design A, five-seed run: uncorrected cohort (897 PD / 3{,}912 HC) "
         "and age-matched (562 / 562)}} \\\\[-0.2em]%", "", "", "", "", ""],
        ["Within mPower, uncorrected", pm(ls["baseline"]["within"]), "15", "MLP head", "5",
         "\\S\\ref{subsec:diagnosis}"],
        ["mPower $\\rightarrow$ IPVS / NeuroVoz / MDVR-KCL, uncorrected", ext("baseline"), "15",
         "MLP head", "5", "\\S\\ref{subsec:diagnosis}"],
        ["Within mPower, age-matched", pm(ls["agematched"]["within"]), "15", "MLP head", "5",
         "\\S\\ref{subsec:diagnosis}"],
        ["mPower $\\rightarrow$ IPVS / NeuroVoz / MDVR-KCL, age-matched" + S,
         ext("agematched"), "15", "MLP head", "5", "\\S\\ref{subsec:diagnosis}"],
        ["mPower $\\rightarrow$ Bridge2AI vowel, uncorrected: as released / age-matched test",
         f"{ls['baseline']['bridge2ai'].split()[0]} / {ls['baseline']['bridge2ai_matched'].split()[0]}",
         "15", "MLP head", "5", "\\S\\ref{subsec:diagnosis}"],
        ["mPower $\\rightarrow$ Bridge2AI vowel, age-matched: as released / age-matched test",
         f"{ls['agematched']['bridge2ai'].split()[0]} / {ls['agematched']['bridge2ai_matched'].split()[0]}",
         "15", "MLP head", "5", "\\S\\ref{subsec:diagnosis}"],
        ["Within minus mean transfer, uncorrected $\\rightarrow$ age-matched",
         f"{gap('baseline'):.1f} $\\rightarrow$ {gap('agematched'):.1f} pp", "15",
         "MLP head", "5", "\\S\\ref{subsec:diagnosis}"],
        ["Within mPower, age- and sex-matched (560 / 560)", pm(asx["agesexmatched"]["within"]),
         "15", "MLP head", "5", "\\S\\ref{subsec:diagnosis}"],
        ["$\\rightarrow$ IPVS / NeuroVoz / MDVR-KCL, age- and sex-matched",
         f"{asx['agesexmatched']['ipvs'].split()[0]} / "
         f"{asx['agesexmatched']['neurovoz'].split()[0]} / "
         f"{asx['agesexmatched']['mdvr_kcl'].split()[0]}", "15", "MLP head", "5",
         "\\S\\ref{subsec:diagnosis}"],
        ["Mean transfer, age-matched, with centroid adaptation",
         f"{mean_t(ls_runs, 'agematched'):.3f} $\\rightarrow$ "
         f"{mean_t(ad_runs, 'agematched'):.3f}", "15", "MLP head", "5",
         "\\S\\ref{subsec:variants}"],
        [f"Restricted permutation ({rp['n_perm']} permutations): observed vs within-age-bin null",
         f"{rp['observed']:.3f} vs {rp['within_age_bins']['null_mean']:.3f} "
         f"($p = {rp['within_age_bins']['p_value']:.3f}$)", "15",
         "PCA + logistic", "--", "\\S\\ref{subsec:diagnosis}"],
        ["\\midrule \\multicolumn{6}{l}{\\emph{Classifier comparison, one recording per "
         "participant (uncorrected $\\rightarrow$ age-matched)}} \\\\[-0.2em]%", "", "", "", "", ""],
        ["Age alone", f"{best[('confounded', 'age only (1)')]:.3f} $\\rightarrow$ "
         f"{best[('age-matched', 'age only (1)')]:.3f}", "--",
         f"best of 5 (logistic: {lr[('age-matched', 'age only (1)')]:.3f})", "1",
         "Table~\\ref{tab:baselines}"],
        ["SSL embeddings", f"{best[('confounded', 'SSL embeddings (2048)')]:.3f} $\\rightarrow$ "
         f"{best[('age-matched', 'SSL embeddings (2048)')]:.3f}", "15",
         f"best of 5 (MLP: {mlp[('confounded', 'SSL embeddings (2048)')]:.3f} $\\rightarrow$ "
         f"{mlp[('age-matched', 'SSL embeddings (2048)')]:.3f})", "1", "Table~\\ref{tab:baselines}"],
        ["Classical acoustic (42)", f"{best[('confounded', 'classical (42)')]:.3f} $\\rightarrow$ "
         f"{best[('age-matched', 'classical (42)')]:.3f}", "--", "best of 5", "1",
         "Table~\\ref{tab:baselines}"],
        ["\\midrule \\multicolumn{6}{l}{\\emph{Layer and encoder selection (age-matched cohort)}}"
         " \\\\[-0.2em]%", "", "", "", "", ""],
        ["XLS-R layer chosen by cross-validation", f"L{sel['best_layer']}: {sel['best_auroc']:.3f}",
         "all 25", "MLP head", "1", "\\S\\ref{sec:encoder}"],
        ["Five encoders, best layer each (XLS-R: L"
         f"{sweep['xls-r-300m']['best_layer']})", f"{min(span):.3f}--{max(span):.3f}", "all",
         "linear probe", "1", "Table~\\ref{tab:encoders-matched}"],
        ["\\midrule \\multicolumn{6}{l}{\\emph{Design A: locked transfer at layer 11, on the "
         "cross-validated plateau (uncorrected cohort)}} \\\\[-0.2em]%", "", "", "", "", ""],
        ["Within mPower", f"{l11:.3f}", "11", "MLP head", "1", "\\S\\ref{sec:encoder}"],
        ["mPower $\\rightarrow$ six external dataset/task pairs", f"{min(ta):.3f}--{max(ta):.3f}",
         "11", "MLP head", "1", "Table~\\ref{tab:transfer-a}"],
        ["\\midrule \\multicolumn{6}{l}{\\emph{Design B: a new head trained on connected speech "
         "of one clinical dataset}} \\\\[-0.2em]%", "", "", "", "", ""],
        ["Transfer between clinical datasets, uncorrected", span_of("full"), "15", "MLP head", "5",
         "Table~\\ref{tab:transfer-b}"],
        ["Transfer between clinical datasets, age-matched", span_of("age-matched"), "15",
         "MLP head", "5", "Table~\\ref{tab:agecontrol}"],
        ["\\midrule \\multicolumn{6}{l}{\\emph{Clinical utility}} \\\\[-0.2em]%",
         "", "", "", "", ""],
        ["Largest decision-curve gain over both defaults, age-matched\\textsuperscript{b}",
         f"{dc['max_gain_per_1000']:.1f} per 1{{,}}000 at $p_t = {dc['at_pt']:.2f}$", "15",
         "SVM", "1", "Figure~\\ref{fig:dca}"],
        ["\\midrule \\multicolumn{6}{l}{\\emph{Designs C and S: Bridge2AI (98 PD / 120 HC), layer "
         "chosen inside Bridge2AI}} \\\\[-0.2em]%", "", "", "", "", ""],
        ["Age alone, as recruited", f"{b2['vowel']['uncorrected']['age_only']:.3f}", "--",
         "logistic", "--", "Table~\\ref{tab:b2ai-within}"],
        ["Within, age-matched: speech / DDK / vowel",
         " / ".join(f"{b2[t]['age_matched']['voice_mean']:.3f}" for t in ("speech", "ddk", "vowel")),
         "8", "linear probe", "5", "Table~\\ref{tab:b2ai-within}"],
        ["Speech $\\rightarrow$ IPVS / NeuroVoz / MDVR-KCL (age-comparable)",
         " / ".join(f"{b2c['A5_transfer'][k]['auroc']:.3f}" for k in
                    ("ipvs (no young controls)", "neurovoz (age-matched)", "mdvr_kcl (as published)")),
         "8", "MLP head", "5", "Table~\\ref{tab:b2ai-transfer}"],
        ["Speech model: cognitive impairment vs controls / PD vs cognitive impairment",
         f"{b2c['A1']['ci vs control']['auroc']:.3f} / {b2c['A1']['pd vs ci']['auroc']:.3f}",
         "8", "linear probe", "5", "Table~\\ref{tab:confusion}"],
    ]
    # rows that open a block carry their own line ending; drop the padding cells
    body_rows = []
    for r in rows:
        if r[0].rstrip().endswith("%"):
            body_rows.append(r[0][:-1])
        else:
            body_rows.append(" & ".join(r) + " \\\\")
    header = ("\\begin{tabularx}{\\linewidth}{LP{2.3cm}cP{1.9cm}cl}\n\\toprule\n"
              "\\textbf{Quantity} & \\textbf{AUROC} & \\textbf{Layer} & \\textbf{Classifier} & "
              "\\textbf{Seeds} & \\textbf{Where} \\\\\n\\midrule\n")
    body = header + "\n".join(body_rows) + "\n\\bottomrule\n\\end{tabularx}"
    _w("t4_0_master.tex", "{\\scriptsize\\setlength{\\tabcolsep}{3pt}"
       "\\renewcommand{\\arraystretch}{0.95}\n" + body + "\n}",
       "Headline figures of this thesis and the settings that produced each. Figures quoted "
       "elsewhere refer to this table.",
       "tab:master",
       "Appendix~\\ref{app:tables} gives the per-seed and per-run values. "
       "Uncorrected: the mPower cohort as released, capped at three recordings per "
       "participant (897 with a diagnosis, 3{,}912 without; prevalence 0.187). Age-matched: "
       "frequency-matched in 5-year bins (562 per group). Within-dataset and transfer figures "
       "of the five-seed run are both at layer 15. Bridge2AI uses layer 8, chosen by "
       "cross-validation inside Bridge2AI. \\textsuperscript{a}MDVR-KCL records no age, so "
       "only the training side is matched for that dataset. \\textsuperscript{b}Standard "
       "decision curve: positive when the prior-shifted probability reaches $p_t$; net "
       "benefit at prevalence 0.187.")


def t_binwidth() -> None:
    r = pd.DataFrame(_j("age_bin_width.json"))
    rows = [[d.dataset, f"{d.bin_width}", f"{d.n_per_group:,}", f"{d.age_gap:+.2f}",
             f"{d.age_only:.3f}", f"{d.voice:.3f} $\\pm$ {d.voice_sd:.3f}"]
            for d in r.itertuples()]
    _w("t3_5_binwidth.tex",
       "{\\small\\setlength{\\tabcolsep}{4pt}\n"
       + _tab(["Dataset", "Bin (yr)", "Per group", "Age gap (yr)", "Age alone", "Voice"],
              rows, "lrrrcc") + "\n}",
       "Sensitivity of age matching to the width of the age bins.",
       "tab:binwidth",
       "Means over five matching seeds. Age alone is the AUROC of age as the only "
       "predictor after matching (0.5 means age carries no information about the label). "
       "Voice is the participant-level AUROC of a linear probe on XLS-R layer 11. mPower: "
       "sustained vowel, one recording per participant; Bridge2AI: connected speech.")


# --------------------------------------------------------------- Bridge2AI (Chapter 4)
def _pm(m, s):
    return f"{m:.3f} $\\pm$ {s:.3f}"


def _ci(d):
    return f"{d['auroc']:.3f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"


B2L = 11   # Bridge2AI layer, chosen inside Bridge2AI (bridge2ai_layer_select.py; review A2)


def t_b2ai_within() -> None:
    r = _j(f"bridge2ai_agematch__L{B2L}.json")
    ci = _j(f"bridge2ai_review_checks__L{B2L}.json")["A5_within"]
    rows = []
    for task, name in (("vowel", "Sustained vowel"), ("ddk", "Diadochokinesis"),
                       ("speech", "Connected speech")):
        u, a, p = r[task]["uncorrected"], r[task]["age_matched"], r[task]["restricted_permutation"]
        rows.append([name, f"{u['n_pd']} / {u['n_hc']}", f"{u['age_only']:.3f}",
                     f"{u['voice']:.3f}", f"{a['n_per_group']}", f"{a['age_only']:.3f}",
                     _ci(ci[task]), f"+{p['excess']:.3f} ({p['p_value']:.3f})"])
    body = ("\\begin{tabular}{lccccccc}\n\\toprule\n"
            " & \\multicolumn{3}{c}{\\textbf{Uncorrected}} & "
            "\\multicolumn{4}{c}{\\textbf{Age-matched}} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n"
            "\\textbf{Task} & \\textbf{PD / HC} & \\textbf{Age} & \\textbf{Voice} & "
            "\\textbf{Pairs} & \\textbf{Age} & \\textbf{Voice [95\\% CI]} & "
            "\\textbf{Above null ($p$)} \\\\\n\\midrule\n"
            + "\n".join(" & ".join(r) + " \\\\" for r in rows)
            + "\n\\bottomrule\n\\end{tabular}")
    _w("t4_20_b2ai_within.tex",
       "{\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n" + body + "\n}",
       "Design C, within Bridge2AI: Parkinson's disease against healthy controls before "
       "(columns 2 to 4) and after (columns 5 to 7) frequency matching on age in five-year "
       "bins.",
       "tab:b2ai-within",
       f"Participant-level AUROC; linear probe on XLS-R layer {B2L}, the layer chosen by "
       "cross-validation inside Bridge2AI (Section~\\ref{subsec:bridge2ai-results}); "
       "participant-disjoint 5-fold cross-validation. Matched values are the mean over five "
       "matching seeds with a participant bootstrap interval (1,000 resamples per seed, "
       "pooled). The last column is the margin over a restricted permutation null in which "
       "labels are shuffled only within five-year age bins (1,000 permutations, full cohort); "
       "0.001 is the smallest attainable $p$.")


def t_b2ai_transfer() -> None:
    a = pd.read_csv(LOGS / f"bridge2ai_transfer_agecheck__L{B2L}.csv")
    t = pd.read_csv(LOGS / "bridge2ai_transfer.csv")          # mPower rows, layer 15
    ci = _j(f"bridge2ai_review_checks__L{B2L}.json")["A5_transfer"]
    plan = [("speech", "ipvs", "no young controls", "IPVS speech, elderly controls only",
             "ipvs (no young controls)"),
            ("speech", "neurovoz", "age-matched", "NeuroVoz speech, age-matched",
             "neurovoz (age-matched)"),
            ("speech", "mdvr_kcl", "as published", "MDVR-KCL speech (no age data)",
             "mdvr_kcl (as published)"),
            ("vowel", "ipvs", "as published", "IPVS vowel (age-balanced)", None),
            ("vowel", "neurovoz", "age-matched", "NeuroVoz vowel, age-matched", None),
            ("ddk", "neurovoz", "age-matched", "NeuroVoz DDK, age-matched", None)]
    names = {"speech": "Speech", "vowel": "Vowel", "ddk": "DDK"}
    rows = []
    for task, tgt, sub, label, cikey in plan:
        g = a[(a.task == task) & (a.target == tgt) & (a.target_subset == sub)]
        n = g.iloc[0]
        unc = _pm(*g[g.source == "uncorrected"].auroc.agg(["mean", "std"]))
        mat = _ci(ci[cikey]) if cikey else _pm(*g[g.source == "age-matched"].auroc.agg(["mean", "std"]))
        rows.append([f"C: Bridge2AI {names[task]}", label, f"{int(n.n_pd)} / {int(n.n_hc)}", unc, mat])
    for direction, tgt, label, src in (
            ("bridge2ai->", "mpower", "mPower vowel, age-matched", "C: Bridge2AI Vowel\\textsuperscript{a}"),):
        g = t[(t.direction == direction) & (t.task == "vowel") & (t.target == tgt)
              & (t.target_cohort == "age-matched")]
        cell = {s: _pm(*g[g.source == s].auroc.agg(["mean", "std"])) for s in ("uncorrected", "age-matched")}
        rows.append([src, label, "--", cell["uncorrected"], cell["age-matched"]])
    _w("t4_21_b2ai_transfer.tex",
       "{\\small\\setlength{\\tabcolsep}{4pt}\n"
       + _tab(["Design: trained on", "Tested on", "PD / HC",
               "\\shortstack{Source\\\\uncorrected}", "\\shortstack{Source\\\\age-matched}"],
              rows, "P{3.1cm}Lccc") + "\n}",
       "Design C: locked transfer with Bridge2AI as the training dataset, scored on "
       "age-comparable test sets.",
       "tab:b2ai-transfer",
       f"Participant-level AUROC over five seeds; locked MLP head, layer {B2L}. Speech rows "
       "with an age-matched source give the mean and a participant bootstrap interval "
       "(1,000 resamples per seed, pooled); other cells give the mean and the standard "
       "deviation across seeds. The IPVS speech test set excludes the fifteen young "
       "controls; the IPVS vowel set contains none. \\textsuperscript{a}Layer 15, because "
       "mPower embeddings exist only at layers 15 and 24; mPower rows use the age-matched "
       "cohort of Section~\\ref{subsec:diagnosis}.")


def t_b2ai_checks() -> None:
    f = _j(f"bridge2ai_followups__L{B2L}.json")
    rows = []

    def tr(d, key):
        return [_pm(*d[k]) if k in d else "--" for k in key]
    sp = ["ipvs speech (no young controls)", "neurovoz speech (age-matched)",
          "mdvr_kcl speech (as published)"]
    for name, k in (("Read speech, age-matched", "read"), ("Spontaneous speech, age-matched", "spontaneous")):
        d = f["A_speech_split"][k]
        rows.append([name, str(d["matched_n_per_group"]), _pm(d["matched_voice"], d["matched_sd"]),
                     *tr(d["transfer_age_matched_source"], sp)])
    vw = ["ipvs vowel (as published)", "neurovoz vowel (age-matched)"]
    for name, k in (("Speech, age and sex matched", "speech"), ("Vowel, age and sex matched", "vowel"),
                    ("DDK, age and sex matched", "ddk")):
        d = f["B_age_sex"][k]
        t = d.get("transfer_age_sex_matched_source", {})
        cells = tr(t, sp) if k == "speech" else (tr(t, vw) + ["--"] if k == "vowel" else ["--"] * 3)
        rows.append([name, str(d["matched_n_per_group"]), _pm(d["matched_voice"], d["matched_sd"]), *cells])
    _w("t4_22_b2ai_checks.tex",
       "{\\small\\setlength{\\tabcolsep}{3pt}\n"
       + _tab(["Bridge2AI source", "Pairs", "Within Bridge2AI", "IPVS", "NeuroVoz", "MDVR-KCL"],
              rows, "Lcccc c") + "\n}",
       "Design C checks: read against spontaneous speech, and matching on age and sex "
       "together.",
       "tab:b2ai-checks",
       f"Participant-level AUROC, mean and standard deviation over five seeds; layer {B2L}. "
       "External columns are locked transfer from the matched Bridge2AI cohort to the "
       "age-comparable test sets of Table~\\ref{tab:b2ai-transfer} (vowel rows: IPVS vowel "
       "and NeuroVoz vowel, age-matched). DDK has no external target in these columns.")


def t_specificity() -> None:
    c = _j(f"bridge2ai_followups__L{B2L}.json")["C_specificity"]
    fam = [("all other disorders", "All other disorders"),
           ("neurological (cognitive impairment, ALS)", "Neurological (cognitive impairment)"),
           ("voice disorders", "Voice disorders"),
           ("respiratory", "Respiratory"), ("mood", "Mood")]
    rows = []
    for key, name in fam:
        d = {t: c[t][f"PD vs {key}"] for t in ("vowel", "ddk", "speech")}
        s = d["speech"]
        rows.append([name, str(s["n_neg"]), f"{s['age_neg']:.1f}", str(s["matched_n_per_group"]),
                     *[_pm(d[t]["matched_voice"], d[t]["matched_sd"]) for t in ("vowel", "ddk", "speech")]])
    _w("t4_23_specificity.tex",
       "{\\small\\setlength{\\tabcolsep}{3pt}\n"
       + _tab(["PD against", "$n$", "Age", "Pairs", "Vowel", "DDK", "Speech"], rows,
              "Lccc ccc") + "\n}",
       "Specificity test (S): separating Parkinson's disease from other disorders within "
       "Bridge2AI, after matching on age.",
       "tab:specificity",
       f"A classifier trained directly on each contrast; layer {B2L}; participant-level "
       "AUROC, mean and standard deviation over five matching seeds. $n$ and age describe "
       "the comparison group on the speech task. The "
       "mood group yields few matched pairs, so its figures are unreliable.")

    conf = {t: c[t]["confusion_PD_vs_control_model"] for t in ("vowel", "ddk", "speech")}
    a1 = _j(f"bridge2ai_review_checks__L{B2L}.json")["A1"]
    order = [("PD vs control", "PD against controls"),
             ("neurological (cognitive impairment, ALS) vs control", "Cognitive impairment against controls"),
             ("PD vs neurological (cognitive impairment, ALS)", "PD against cognitive impairment"),
             ("voice disorders vs control", "Voice disorders against controls"),
             ("PD vs voice disorders", "PD against voice disorders"),
             ("all other disorders vs control", "All other disorders against controls"),
             ("PD vs all other disorders", "PD against all other disorders")]
    rows = [[name, *[_pm(*conf[t][k]) for t in ("vowel", "ddk", "speech")]] for k, name in order]
    rows.append(["\\midrule \\multicolumn{4}{l}{\\emph{Each pair matched on age directly "
                 "(speech model)}} \\\\[-0.2em]%", "", "", ""])
    for k, name in (("ci vs control", "Cognitive impairment against controls"),
                    ("pd vs ci", "PD against cognitive impairment")):
        v = a1[k]
        rows.append([f"{name} ({v['pairs']} pairs; age alone {v['age_only']:.3f})", "--", "--",
                     _pm(v["auroc"], v["seed_sd"])])
    body = []
    for r in rows:
        body.append(r[0][:-1] if r[0].rstrip().endswith("%") else " & ".join(r) + " \\\\")
    tab = ("\\begin{tabularx}{\\linewidth}{Lccc}\n\\toprule\n\\textbf{Comparison scored by the PD "
           "model} & \\textbf{Vowel} & \\textbf{DDK} & \\textbf{Speech} \\\\\n\\midrule\n"
           + "\n".join(body) + "\n\\bottomrule\n\\end{tabularx}")
    _w("t4_24_confusion.tex", "{\\small\\setlength{\\tabcolsep}{4pt}\n" + tab + "\n}",
       "Specificity test (S): how a model trained only to separate Parkinson's disease from "
       "age-matched healthy controls scores participants with other disorders.",
       "tab:confusion",
       f"Layer {B2L}; participant-level AUROC on the model's predicted probability of "
       "Parkinson's disease, mean and standard deviation over five seeds. In the upper block "
       "the controls are those matched to the patients (mean age about 69) and the other "
       "groups are as recruited (cognitive impairment: mean age 74.8). In the lower block each "
       "comparison is itself frequency-matched on age, which leaves age alone at chance. "
       "Patients and controls are scored out of fold; participants with other disorders were "
       "never seen in training. A value near 0.5 for PD against a disorder means the model "
       "cannot tell them apart.")


def _b2ai_calibration_unified() -> dict:
    """Calibration and decision curves of the Bridge2AI speech model, from the participant
    scores of the unified experiments (the same runs as every other table): within
    Bridge2AI (age-matched, out-of-fold) and locked on age-matched NeuroVoz. Medians over the
    five seeds, as in bridge2ai_review_checks.py, whose calib() and dca() are reused."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bridge2ai_review_checks import calib, dca
    S = pd.read_csv(LOGS / "unified" / "scores.csv.gz")
    sel = {"within_bridge2ai": (S.exp == "within") & (S.target == "bridge2ai")
           & (S.source_task == "speech") & (S.condition == "agematched"),
           "neurovoz_age_matched": (S.exp == "exp2") & (S.source == "bridge2ai")
           & (S.condition == "agematched") & (S.target == "neurovoz")
           & (S.target_cohort == "agematched")}
    out = {}
    for key, m in sel.items():
        cal_rows, dca_rows = [], []
        for _, g in S[m].groupby("seed"):
            cal_rows.append(calib(g.y.to_numpy(), g.p.to_numpy()))
            dca_rows.append(dca(g.y.to_numpy(), g.p.to_numpy()))
        med = {k: float(np.median([r[k] for r in rows]))
               for rows in (cal_rows, dca_rows) for k in rows[0]
               if not isinstance(rows[0][k], list) and rows[0][k] is not None}
        out[key] = {**med, "per_seed_dca": dca_rows}
    return out


def t_b2ai_calibration() -> None:
    a4 = _b2ai_calibration_unified()
    rows = []
    for key, name in (("within_bridge2ai", "Within Bridge2AI (5-fold CV)"),
                      ("neurovoz_age_matched", "Locked, on NeuroVoz")):
        d = a4[key]
        pos = [s["positive_range"] for s in d["per_seed_dca"] if s["positive_range"]]
        lo, hi = (float(np.median([p[0] for p in pos])),
                  float(np.median([p[1] for p in pos]))) if pos else (None, None)
        rows.append([name, f"{d['ece']:.2f}", f"{d['brier']:.2f}", f"{d['slope']:.2f}",
                     f"{d['intercept']:.2f}", f"{d['max_gain_per_1000']:.0f} at {d['at_threshold']:.2f}",
                     f"{lo:.2f}--{hi:.2f}" if pos else "--"])
    _w("t4_25_b2ai_calibration.tex",
       "{\\small\\setlength{\\tabcolsep}{3pt}\n"
       + _tab(["Evaluated", "ECE", "Brier", "Slope", "Intercept",
               "\\shortstack{Largest gain\\\\per 1,000 ($p_t$)}", "\\shortstack{Positive\\\\for $p_t$}"],
              rows, "Lcccccc") + "\n}",
       "Calibration and clinical utility of the Bridge2AI connected-speech model trained in Experiment 2.",
       "tab:b2ai-calibration",
       f"Layer {B2L}; framework head with calibration locked on validation data; the same "
       "runs as Tables~\\ref{tab:u-within} and~\\ref{tab:u-exp2}; medians over five seeds. Slope and intercept from logistic recalibration (ideal 1 and 0). Decision "
       "curves use probabilities prior-shifted to a prevalence of 0.187, as for mPower "
       "(Section~\\ref{subsec:utility}); net benefit is the gain over the better of screening "
       "everyone and screening no one. The last column gives the threshold range over which the "
       "gain is positive (medians across seeds of its lower and upper ends).")


def main() -> None:
    warnings.simplefilter("ignore")
    print("Chapter 3:")
    t_datasets(); t_demographics(); t_prevalence(); t_baselines(); t_binwidth()
    print("Chapter 4:")
    t_encoders_confounded(); t_encoders_matched(); t_significance()
    # t_protocol() is not called: its table duplicated Table 3.4 (t3_4_baselines) and was
    # never included in the thesis.
    # Transfer, within-dataset, specificity and master tables come from the unified layer-11
    # experiments (scripts/unified_tables.py).
    t_deviation(); t_probes(); t_windowing(); t_classical_transfer(); t_b2ai_calibration()
    t_robustness()
    n = len(list(OUT.glob("*.tex")))
    print(f"\n{n} tables written to {OUT}")


if __name__ == "__main__":
    main()
