"""Generate Appendix A (evaluation tables) and Appendix B (code samples) from the project.

Appendix A is built from outputs/logs/, so its numbers are the numbers the analyses wrote.
Appendix B prints functions straight from src/pdvoice/ (located by name with the ast module),
so the code shown is the code that ran. Re-run after any analysis or code change.

    thesis/appendix_a_tables.tex
    thesis/appendix_b_code.tex

Run:  .venv\\Scripts\\python.exe scripts\\thesis_appendix.py
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from pdvoice.config import LOGS

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "thesis"
SRC = ROOT / "src" / "pdvoice"
NAMES = {"ipvs": "IPVS", "neurovoz": "NeuroVoz", "mdvr_kcl": "MDVR-KCL"}
TASKS = {"vowel": "vowel", "speech": "speech", "ddk": "DDK"}


def _j(name):
    return json.loads((LOGS / name).read_text())


def longtable(cols, header, rows, caption, label, note=""):
    out = ["{\\small\\setlength{\\tabcolsep}{3pt}", f"\\begin{{longtable}}{{{cols}}}",
           f"\\caption{{{caption}}}\\label{{{label}}} \\\\", "\\toprule",
           " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule",
           "\\endfirsthead",
           f"\\multicolumn{{{len(header)}}}{{l}}{{\\emph{{Table \\thetable\\ continued}}}} \\\\",
           "\\toprule", " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule",
           "\\endhead", "\\bottomrule", "\\endlastfoot"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\end{longtable}", "}"]
    if note:
        out.insert(0, "")
        out.append(f"{{\\footnotesize {note}\\par}}\\medskip")
    return "\n".join(out)


# ------------------------------------------------------------------ Appendix A
def a_transfer():
    rows = []
    for f in sorted(LOGS.glob("layer2__xls-r-300m__*.json")):
        m = re.match(r"layer2__xls-r-300m__(\w+?)__(\w+?)__L(\w+?)(__Wfull)?(__adapt-[\w-]+)?\.json",
                     f.name)
        if not m:
            continue
        ds, task, layer, wfull, adapt = m.groups()
        d = json.loads(f.read_text())
        lo, hi = d["auroc_ci"]
        layer = "24" if layer == "None" else layer
        cfg = f"L{layer}, {'whole file' if wfull else 'fixed window'}"
        if adapt:
            cfg += ", " + {"__adapt-hc": "HC centroid",
                           "__adapt-hc-standardize": "HC standardise"}[adapt]
        a = f"{d['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"
        if hi < 0.5:
            a = f"\\textbf{{{a}}}"
        s90 = d["sensitivity_at_90_spec"]
        s90 = s90["sensitivity"] if isinstance(s90, dict) else s90
        rows.append([f"{NAMES[ds]} / {TASKS[task]}", cfg, f"{d['n_pd']}/{d['n_hc']}", a,
                     f"{d['auprc']:.3f}", f"{d['ece']:.3f}", f"{d['brier']:.3f}",
                     f"{s90:.3f}"])
    n_inv = sum(r[3].startswith("\\textbf") for r in rows)
    return longtable(
        "p{2.4cm}p{3.0cm}rp{2.9cm}rrrr",
        ["Dataset / task", "Configuration", "PD/HC", "AUROC [95\\% CI]", "AUPRC", "ECE", "Brier",
         "Sens."],
        rows,
        f"All {len(rows)} locked transfer runs of design A. The model is trained on mPower "
        "sustained vowels (uncorrected cohort, three recordings per participant) and applied "
        "unchanged to each external dataset.",
        "tab:appA-transfer",
        f"Bold: the whole confidence interval lies below 0.5 ({n_inv} of {len(rows)} runs). "
        "Sens.: sensitivity at a specificity of at least 0.90. L24 is the final layer. HC "
        "centroid and HC standardise are the healthy-control adaptations of "
        "Section~\\ref{subsec:variants}.")


def a_layers():
    sel = pd.read_csv(LOGS / "select_transfer_layer.csv").set_index("layer")
    sw = pd.read_csv(LOGS / "encoder_layer_sweep__per_layer.csv")
    order = ["wav2vec2-base", "hubert-large", "wavlm-large", "xls-r-300m", "whisper-small"]
    piv = sw.pivot(index="layer", columns="model", values="auroc")
    rows = []
    for L in range(0, int(max(piv.index.max(), sel.index.max())) + 1):
        cells = [f"{sel.loc[L, 'auroc']:.3f}" if L in sel.index else "--"]
        for m in order:
            v = piv.loc[L, m] if (L in piv.index and m in piv.columns) else np.nan
            cells.append("--" if pd.isna(v) else f"{v:.3f}")
        rows.append([str(L)] + cells)
    best = {m: int(piv[m].idxmax()) for m in order if m in piv.columns}
    note = ("Age-matched mPower cohort, one seed. The first column uses the framework's MLP "
            "head, as in Section~\\ref{sec:encoder}; the other five use a linear probe so that "
            "every encoder is read the same way. Best layer per column: XLS-R (MLP) "
            f"{int(sel['auroc'].idxmax())}; " + "; ".join(f"{m} {b}" for m, b in best.items())
            + ". Layer 0 is the convolutional feature encoder's output.")
    return longtable("r" + "r" * 6,
                     ["Layer", "XLS-R (MLP)", "wav2vec2", "HuBERT", "WavLM", "XLS-R", "Whisper"],
                     rows, "Participant-level AUROC at every encoder layer.", "tab:appA-layers",
                     note)


def a_seeds():
    blocks = [("label_sensitivity.json", "Label and age interventions"),
              ("label_sensitivity__agesex.json", "Age and sex matching"),
              ("label_sensitivity__adapt_centroid.json", "With HC centroid adaptation"),
              ("label_sensitivity__adapt_standardize.json",
               "With HC standardise adaptation")]
    names = {"baseline": "uncorrected", "confident": "confident labels", "age50": "aged 50+",
             "agematched": "age-matched", "agesexmatched": "age- and sex-matched"}
    rows, seen = [], set()
    for fn, title in blocks:
        d = _j(fn)
        rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}}", "", "", "", "", ""])
        for sc, runs in d["runs"].items():
            key = (sc, d.get("adapt", "none"))
            if fn != "label_sensitivity.json" and sc == "baseline" and d.get("adapt", "none") == "none":
                continue                       # identical to the first block
            if key in seen and d.get("adapt", "none") == "none":
                continue
            seen.add(key)
            for r in runs:
                rows.append([names.get(sc, sc), str(r["seed"]), f"{r['within']:.3f}",
                             f"{r.get('transfer_ipvs', float('nan')):.3f}",
                             f"{r.get('transfer_neurovoz', float('nan')):.3f}",
                             f"{r.get('transfer_mdvr_kcl', float('nan')):.3f}"])
    rows = [r if not r[0].startswith("\\multicolumn") else [r[0]] for r in rows]
    body = []
    for r in rows:
        body.append(r[0] if len(r) == 1 else " & ".join(r))
    return longtable("llrrrr", ["Cohort", "Seed", "Within", "IPVS", "NeuroVoz", "MDVR-KCL"],
                     [[b] if "multicolumn" in b else b.split(" & ") for b in body],
                     "Per-seed results of the five-seed experiments behind "
                     "Table~\\ref{tab:master}: within mPower at the final layer and transfer at "
                     "layer 15, MLP head.", "tab:appA-seeds")


def a_speech():
    d = pd.DataFrame(_j("speech_agematch.json")["runs"])
    tg = [c for c in d.columns if c.startswith("-> ")]
    rows = []
    for src in ("mdvr_kcl", "neurovoz", "ipvs"):
        for t in tg:
            cells = []
            for coh in ("full", "age-matched", "size-matched random"):
                v = d[(d.source == src) & (d.cohort == coh)][t].dropna()
                cells.append(f"{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}" if len(v) else "--")
            if all(c == "--" for c in cells):
                continue
            rows.append([f"{NAMES[src]} $\\rightarrow$ {NAMES[t[3:]]}"] + cells)
    return longtable("p{4.2cm}p{3.0cm}p{3.0cm}p{3.0cm}",
                     ["Direction", "Unmatched", "Age-matched", "Size-matched random"], rows,
                     "Connected-speech transfer of design B by direction, mean $\\pm$ SD over "
                     "five seeds, layer 15.", "tab:appA-speech",
                     "MDVR-KCL has no age data, so it cannot be age-matched as a training dataset.")


def a_classifiers():
    c = pd.read_csv(LOGS / "classifier_comparison.csv")
    rows = [[r.cohort, r.features, r.classifier, f"{r.AUROC:.3f}", f"[{r.lo:.3f}, {r.hi:.3f}]"]
            for r in c.itertuples()]
    return longtable("llp{3.6cm}rr", ["Cohort", "Features", "Classifier", "AUROC", "95\\% CI"],
                     rows, "All classifier and feature combinations, one recording per "
                     "participant, layer 15 for the embeddings.", "tab:appA-classifiers")


def a_dca():
    d = pd.read_csv(LOGS / "decision_curves.csv")
    pts = [0.05, 0.10, 0.15, 0.19, 0.25, 0.30, 0.40, 0.50]
    rows = []
    for coh in ("confounded", "age-matched"):
        for model in ("SSL, SVM", "SSL, MLP head", "classical, RF", "age alone"):
            s = d[(d.cohort == coh) & (d.model == model)].set_index("pt")
            if s.empty:
                continue
            rows.append([("uncorrected" if coh == "confounded" else "age-matched"), model]
                        + [f"{1000 * (s.loc[p, 'nb'] - max(s.loc[p, 'nb_all'], 0)):+.1f}"
                           for p in pts])
    return longtable("ll" + "r" * len(pts),
                     ["Cohort", "Model"] + [f"{p:.2f}" for p in pts], rows,
                     "Decision-curve gain over the better of screening everyone and screening "
                     "no one, in cases per 1{,}000, at selected threshold probabilities.",
                     "tab:appA-dca",
                     "Prevalence 0.187; age-matched predictions prior-shifted from 0.5. "
                     "Figure~\\ref{fig:dca} shows the full curves.")


def a_ipvs():
    inv = _j("ipvs_inversion.json")
    rows = []
    for name, r in inv.items():
        rows.append([name.replace("(Table: transfer A)", "").strip(), str(r["n"]),
                     f"{r['auroc_all']:.3f}", f"{r['auroc_pd_vs_elderly']:.3f}",
                     f"{r['age_only_auroc']:.3f}", f"{r['auroc_without_44k_participants']:.3f}",
                     f"{r['spearman_score_age_controls'][0]:+.2f}"])
    t1 = longtable("p{4.6cm}rrrrrr",
                   ["Locked mPower model", "$n$", "AUROC", "PD vs EHC", "Age only", "No 44.1k",
                    "$\\rho$ (HC)"], rows,
                   "Diagnosis of the IPVS inversions (Section~\\ref{subsec:designA}).",
                   "tab:appA-ipvs-inv",
                   "PD vs EHC: patients against elderly controls only. No 44.1k: without the "
                   "participants recorded at 44.1~kHz. $\\rho$ (HC): Spearman correlation of the "
                   "model's score with age among controls.")
    ce = _j("ipvs_ceiling.json")
    label = {"noise_hf_ratio_db": "noise, share above 4 kHz (dB)",
             "noise_centroid_hz": "noise, spectral centroid (Hz)",
             "noise_floor_db": "noise floor (dB)", "duration_s": "duration (s)",
             "rms_db": "overall level (dB)", "peak": "peak amplitude",
             "dc_offset": "DC offset", "clipped_frac": "clipped fraction"}
    rows = []
    for k, lab in label.items():
        cells = [lab]
        for task in ("all", "vowel", "speech"):
            r = ce[task]["features"][k]
            cells.append(f"{r['auroc']:.3f}")
        r = ce["all"]["features"][k]
        cells += [f"{r.get('pd_mean', float('nan')):.4g}", f"{r.get('hc_mean', float('nan')):.4g}"]
        rows.append(cells)
    t2 = longtable("p{5.0cm}rrrrr", ["Recording property", "All", "Vowel", "Speech",
                                     "PD mean", "EHC mean"], rows,
                   "Recording properties that separate IPVS patients (18) from elderly controls "
                   "(22), all at 16 kHz (Section~\\ref{subsec:defects}).", "tab:appA-ipvs-ceiling",
                   "AUROC of each property on its own, oriented so that 0.5 is chance. Noise "
                   "measures use the quietest 10\\% of 25~ms frames in each file.")
    return t1 + "\n\n" + t2


def appendix_a():
    parts = [
        "\\chapter{Evaluation Tables}", "\\label{app:tables}", "",
        "This appendix gives the full results behind the summary tables of "
        "Chapters~\\ref{chap:existing} and~\\ref{chap:proposed}. Every table is generated "
        "from the logs that the analyses wrote, so the values match the text exactly. "
        "Table~\\ref{tab:master} states which experiment each headline figure comes from.", "",
        "\\section{Locked Transfer Runs}", a_transfer(), "",
        "\\section{Layer Sweeps}", a_layers(), "",
        "\\section{Five-Seed Experiments}", a_seeds(), "",
        "\\section{Connected-Speech Transfer}", a_speech(), "",
        "\\section{Classifier Comparison}", a_classifiers(), "",
        "\\section{Decision Curves}", a_dca(), "",
        "\\section{IPVS Diagnostics}", a_ipvs(), "",
    ]
    (THESIS / "appendix_a_tables.tex").write_text("\n".join(parts), encoding="utf-8")


# ------------------------------------------------------------------ Appendix B
ASCII = {"—": "--", "–": "-", "→": "->", "←": "<-", "≤": "<=",
         "≥": ">=", "≈": "~", "×": "x", "−": "-", "‘": "'",
         "’": "'", "“": '"', "”": '"', "…": "...", "±": "+/-",
         "ρ": "rho", "α": "alpha", "²": "^2"}


def to_ascii(s: str) -> str:
    for k, v in ASCII.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s


def source_of(rel: str, names: list[str]) -> str:
    path = SRC / rel
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            seg = "\n".join(text.splitlines()[start: node.end_lineno])
            out.append(seg)
    missing = set(names) - {n.name for n in tree.body
                            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if missing:
        raise SystemExit(f"{rel}: not found {sorted(missing)}")
    return to_ascii("\n\n".join(out))


SAMPLES = [
    ("Pre-processing", "Silence trimming and voice-activity detection applied to every "
     "recording (Section~\\ref{sec:preproc}).",
     "preprocess.py", ["voice_activity_trim", "preprocess_clip", "quality_control"]),
    ("Embedding extraction", "Frozen encoder, chosen layer, mean and standard deviation pooling "
     "(Equation~\\ref{eq:pooling}).",
     "features/ssl.py", ["pool", "_embed_wave"]),
    ("Cohort construction", "Recording cap and frequency matching on age, or on age and sex "
     "(Equation~\\ref{eq:agematch}).",
     "cohort.py", ["cap_per_subject", "demographic_match_ids"]),
    ("Participant-disjoint splits", "Folds that never place one participant in two partitions.",
     "data/splits.py", ["assign_folds", "train_val_test_indices", "assert_disjoint"]),
    ("Classifier head", "The 256-unit head of Equation~\\ref{eq:head} and its training loop.",
     "models/head.py", ["build_head", "fit_mlp_head"]),
    ("Calibration", "The three calibration maps and selection by validation Brier score "
     "(Section~\\ref{sec:calibration}).",
     "calibration.py", ["TemperatureScaler", "PlattScaler", "IsotonicCalibrator",
                        "select_best_by_brier"]),
    ("Evaluation", "DeLong's and McNemar's tests, calibration error and net benefit "
     "(Section~\\ref{sec:measures}).",
     "eval/discrimination.py", ["delong_roc_test", "mcnemar_test"]),
    ("", "", "eval/metrics.py", ["calibration_error", "sensitivity_at_specificity"]),
    ("", "", "eval/dca.py", ["net_benefit"]),
]


def appendix_b():
    parts = ["\\chapter{Code Samples}", "\\label{app:code}", "",
             "This appendix reproduces the central functions of the pipeline, taken directly "
             "from the project's source package \\texttt{pdvoice}, in the order a recording "
             "passes through them. Imports and module-level constants are omitted. The "
             "complete code, including the analysis scripts named in the text, is in the "
             "project repository.", ""]
    for title, desc, rel, names in SAMPLES:
        if title:
            parts += [f"\\section{{{title}}}", desc, ""]
        parts += [f"\\noindent\\texttt{{src/pdvoice/{rel.replace('_', chr(92) + '_')}}}",
                  "\\begin{lstlisting}", source_of(rel, names), "\\end{lstlisting}", ""]
    (THESIS / "appendix_b_code.tex").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    appendix_a()
    appendix_b()
    print("wrote thesis/appendix_a_tables.tex and thesis/appendix_b_code.tex")


if __name__ == "__main__":
    main()
