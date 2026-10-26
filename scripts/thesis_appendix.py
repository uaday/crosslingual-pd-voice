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
NAMES = {"ipvs": "IPVS", "neurovoz": "NeuroVoz", "mdvr_kcl": "MDVR-KCL",
         "bridge2ai": "Bridge2AI"}
TASKS = {"vowel": "vowel", "speech": "speech", "ddk": "DDK"}


def _j(name):
    return json.loads((LOGS / name).read_text())


def longtable(cols, header, rows, caption, label, note="", size="footnotesize"):
    out = [f"{{\\{size}\\setlength{{\\tabcolsep}}{{{'2.5pt' if size == 'scriptsize' else '3pt'}}}", f"\\begin{{longtable}}{{{cols}}}",
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
        m = re.match(r"layer2__xls-r-300m__(\w+?)__(\w+?)__L(\w+?)(__Wfull)?"
                     r"(__adapt-[\w-]+|__agematched-target)?\.json", f.name)
        if not m:
            continue
        ds, task, layer, wfull, adapt = m.groups()
        d = json.loads(f.read_text())
        lo, hi = d["auroc_ci"]
        layer = "24" if layer == "None" else layer
        cfg = f"L{layer}, {'whole' if wfull else 'fixed'}"
        if adapt:
            cfg += ", " + {"__adapt-hc-centroid": "HC centroid",
                           "__adapt-hc-standardize": "HC standardise",
                           "__agematched-target": "matched test"}[adapt]
        a, c = f"{d['auroc']:.3f}", f"[{lo:.3f}, {hi:.3f}]"
        if hi < 0.5:
            a, c = f"\\textbf{{{a}}}", f"\\textbf{{{c}}}"
        s90 = d["sensitivity_at_90_spec"]
        s90 = s90["sensitivity"] if isinstance(s90, dict) else s90
        rows.append([f"{NAMES[ds]} {TASKS[task]}", cfg, f"{d['n_pd']}/{d['n_hc']}", a, c,
                     f"{d['auprc']:.3f}", f"{d['ece']:.3f}", f"{d['brier']:.3f}",
                     f"{s90:.3f}"])
    runs = [r for r in rows if "matched test" not in r[1]]
    n_inv = sum(r[3].startswith("\\textbf") for r in runs)
    return longtable(
        "llrrcrrrr",
        ["Test set", "Configuration", "PD/HC", "AUROC", "95\\% CI", "AUPRC", "ECE", "Brier",
         "Sens."],
        rows,
        f"All {len(runs)} single-run configurations of Experiment 1 across layers, windows and adaptation. The model is trained on mPower "
        "sustained vowels (uncorrected cohort, three recordings per participant) and applied "
        "unchanged to each external dataset.",
        "tab:appA-transfer",
        f"Bold: the whole confidence interval lies below 0.5 ({n_inv} of {len(runs)} runs). "
        "Configuration: encoder layer (L24 is the final layer), then the fixed analysis "
        "window or the whole file. Sens.: sensitivity at a specificity of at least 0.90. HC "
        "centroid and HC standardise are the healthy-control adaptations of "
        "Section~\\ref{subsec:variants}. Rows marked matched test rescore the same "
        "locked Bridge2AI predictions on 30 or 31 US patients and as many controls matched on age; they are "
        "not counted as separate runs. Bridge2AI was embedded with the fixed window only.",
        size="scriptsize")


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
            "head, as in Section~\\ref{subsec:layer}; the other five use a linear probe so that "
            "every encoder is read the same way. Best layer per column: XLS-R (MLP) "
            f"{int(sel['auroc'].idxmax())}; " + "; ".join(f"{m} {b}" for m, b in best.items())
            + ". Layer 0 is the convolutional feature encoder's output.")
    return longtable("r" + "r" * 6,
                     ["Layer", "XLS-R (MLP)", "wav2vec2", "HuBERT", "WavLM", "XLS-R", "Whisper"],
                     rows, "Participant-level AUROC at every encoder layer.", "tab:appA-layers",
                     note)


def a_seeds():
    """Per-seed results of Experiment 1 (unified layer-11 protocol)."""
    import pandas as pd
    R = pd.read_csv(LOGS / "unified" / "runs.csv")
    W = R[(R.exp == "within") & (R.source == "mpower")]
    E = R[R.exp == "exp1"]
    cols = [("ipvs", "vowel", "released"), ("neurovoz", "vowel", "released"),
            ("mdvr_kcl", "speech", "released"), ("bridge2ai", "vowel", "released"),
            ("bridge2ai", "vowel", "agematched")]
    names = {"uncorrected": "uncorrected", "confident": "confident labels",
             "agematched": "age-matched", "agesexmatched": "age- and sex-matched"}
    rows = []
    for cond in ("uncorrected", "confident", "agematched", "agesexmatched"):
        for seed in sorted(W.seed.unique()):
            w = W[(W.condition == cond) & (W.seed == seed)].auroc
            cells = []
            for t, tt, coh in cols:
                v = E[(E.condition == cond) & (E.seed == seed) & (E.target == t)
                      & (E.target_task == tt) & (E.target_cohort == coh)].auroc
                cells.append(f"{v.iloc[0]:.3f}" if len(v) else "--")
            rows.append([names[cond], str(seed), f"{w.iloc[0]:.3f}" if len(w) else "--", *cells])
    return longtable("llrrrrrr", ["Cohort", "Seed", "Within", "IPVS", "NeuroVoz", "MDVR-KCL",
                                  "B2AI", "B2AI matched"], rows,
                     "Per-seed results of Experiment 1 behind Table~\\ref{tab:master}: layer 11, "
                     "the framework head. Within: mPower cross-validation. IPVS and NeuroVoz: "
                     "sustained vowels; MDVR-KCL: connected speech; B2AI: Bridge2AI vowels as "
                     "released; B2AI matched: the same predictions on an age-matched Bridge2AI "
                     "subset.", "tab:appA-seeds")


def a_classifiers():
    c = pd.read_csv(LOGS / "classifier_comparison.csv")
    rows = [[r.cohort, r.features, r.classifier.replace("~", "$\\sim$"), f"{r.AUROC:.3f}", f"[{r.lo:.3f}, {r.hi:.3f}]"]
            for r in c.itertuples()]
    return longtable("llp{3.6cm}rr", ["Cohort", "Features", "Classifier", "AUROC", "95\\% CI"],
                     rows, "All classifier and feature combinations, one recording per "
                     "participant, layer 11 for the embeddings.", "tab:appA-classifiers")


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
                   "The locked Experiment 1 models of Table~\\ref{tab:u-exp1}: layer 11, mean over "
                   "five seeds. "
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
        cells += [f"{r.get('pd_mean', float('nan')) + 0.0:.4g}".replace('-0', '0') if abs(r.get('pd_mean', 0)) < 5e-5 else f"{r.get('pd_mean', float('nan')):.4g}",
                  f"{r.get('hc_mean', float('nan')):.4g}"]
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
        "\\section{Experiment 1 Across Layers and Windows}", a_transfer(), "",
        "\\section{Layer Sweeps}", a_layers(), "",
        "\\section{Experiment 1 by Seed}", a_seeds(), "",
        "\\section{Classifier Comparison}", a_classifiers(), "",
        "\\section{Feature Sets, Classical Transfer and the Analysis Window}",
        "These three tables give the detail behind the robustness summary of "
        "Section~\\ref{subsec:variants} (Table~\\ref{tab:robustness}).", "",
        "\\input{tables/t4_4_significance.tex}",
        "\\input{tables/t4_11_classical_transfer.tex}",
        "\\input{tables/t4_10_windowing.tex}",
        # the later tables are longtables, which print in place; flush these floats first
        # so the appendix tables appear in numbered order
        "\\clearpage", "",
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


# Only the functions that carry the thesis's methodological guarantees are reproduced:
# demographic matching, participant-disjoint splits, the head and the locked calibration.
# Everything else is in the repository named below.
SAMPLES = [
    ("Cohort construction", "Recording cap and frequency matching on age, or on age and sex "
     "(Equation~\\ref{eq:agematch}).",
     "cohort.py", ["cap_per_subject", "demographic_match_ids"]),
    ("Participant-disjoint splits", "Folds that never place one participant in two partitions.",
     "data/splits.py", ["assign_folds"]),
    ("Classifier head", "The 256-unit head of Equation~\\ref{eq:head} and its training loop.",
     "models/head.py", ["build_head", "fit_mlp_head"]),
    ("Calibration", "Selection of the calibration map by validation Brier score, and the "
     "strictly increasing ranking map used for AUROC, AUPRC and sensitivity "
     "(Section~\\ref{sec:calibration}).",
     "calibration.py", ["RankingMap", "select_best_by_brier"]),
]


def short_docstrings(code: str) -> str:
    """Keep only the first line of each docstring; the full text is in the repository."""
    import re
    return re.sub(r'("""[^\n]*?)\n(?:.*?\n)*?\s*"""',
                  lambda m: m.group(1).rstrip() + '"""' if not m.group(1).endswith('"""') else m.group(0),
                  code, flags=re.S)


def appendix_b():
    parts = ["\\chapter{Code Samples}", "\\label{app:code}", "",
             "This appendix reproduces the functions that carry the protocol's guarantees: "
             "demographic matching, participant-disjoint splits, the classifier head and the "
             "locked calibration, taken directly from the project's source package "
             "\\texttt{pdvoice}. Imports, module-level constants and the longer docstrings are omitted. The "
             "complete code, including the analysis scripts named in the text, is in the "
             "project repository, "
             "\\texttt{github.com/uaday/pd-research}, which is private and available on "
             "request; its file \\texttt{REPRODUCE.md} gives the software versions, the "
             "encoder revisions and the steps to re-run every experiment.", ""]
    for title, desc, rel, names in SAMPLES:
        if title:
            parts += [f"\\section{{{title}}}", desc, ""]
        # \needspace stops a listing from starting in the last few lines of a page; the
        # listing itself breaks across pages normally.
        path = f"src/pdvoice/{rel}".replace("_", "\\_")
        parts += ["\\needspace{8\\baselineskip}",
                  f"\\begin{{lstlisting}}[title={{\\texttt{{{path}}}}}]",
                  short_docstrings(source_of(rel, names)), "\\end{lstlisting}", ""]
    (THESIS / "appendix_b_code.tex").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    appendix_a()
    appendix_b()
    print("wrote thesis/appendix_a_tables.tex and thesis/appendix_b_code.tex")


if __name__ == "__main__":
    main()
