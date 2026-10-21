"""Tables for Chapter 4 from the unified layer-11 experiments (scripts/unified_experiments.py).

Writes thesis/tables/u_*.tex. Every cell is the mean over five seeds; where a cell is a
headline result, the participant bootstrap interval pooled over seeds is given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pdvoice.config import LOGS

U = LOGS / "unified"
OUT = __import__("pathlib").Path(__file__).resolve().parents[1] / "thesis" / "tables"
NAME = {"mpower": "mPower", "ipvs": "IPVS", "neurovoz": "NeuroVoz", "mdvr_kcl": "MDVR-KCL",
        "bridge2ai": "Bridge2AI"}
TASK = {"vowel": "vowel", "ddk": "DDK", "speech": "speech"}
COND = {"uncorrected": "Uncorrected", "confident": "Confident", "agematched": "Age-matched",
        "agesexmatched": "Age, sex"}
ORDER_DS = ["mpower", "bridge2ai", "neurovoz", "ipvs", "mdvr_kcl"]


def load():
    R = pd.read_csv(U / "runs.csv")
    S = pd.read_csv(U / "scores.csv.gz")
    return R, S


def ms(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "--"
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}"


def pooled_ci(S, **key):
    """Participant bootstrap, 200 resamples per seed, pooled over seeds."""
    m = np.ones(len(S), bool)
    for k, v in key.items():
        m &= (S[k] == v).to_numpy()
    rng = np.random.default_rng(0)
    vals = []
    for _, g in S[m].groupby("seed"):
        y, r = g.y.to_numpy(), g.r.to_numpy()
        for _ in range(200):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                vals.append(roc_auc_score(y[i], r[i]))
    if not vals:
        return ""
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"[{lo:.2f}, {hi:.2f}]"


def write(name, body, caption, label, note=""):
    tex = ["\\begin{table}[htbp]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}", body]
    if note:
        tex += ["\\\\[0.2cm]", "\\begin{minipage}{0.95\\linewidth}\\footnotesize " + note
                + "\\end{minipage}"]
    tex.append("\\end{table}")
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("wrote", name)


def tab(header, rows, align, size="\\small", sep="4pt"):
    out = [f"{{{size}\\setlength{{\\tabcolsep}}{{{sep}}}", f"\\begin{{tabular}}{{{align}}}",
           "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "}"]
    return "\n".join(out)


# ------------------------------------------------------------------------------ tables
def t_within(R):
    W = R[R.exp == "within"]
    A = R[R.exp == "within_age_only"]
    rows = []
    for ds in ORDER_DS:
        for t in ("vowel", "ddk", "speech"):
            w = W[(W.source == ds) & (W.source_task == t)]
            if w.empty:
                continue
            a = A[(A.source == ds) & (A.source_task == t)]

            def cell(df, c):
                v = df[df.condition == c].auroc
                return "--" if v.empty else f"{v.mean():.3f}"
            n = w[w.condition == "uncorrected"]
            npd, nhc = int(n.n_pd.iloc[0]), int(n.n_hc.iloc[0])
            nm = w[w.condition == "agematched"]
            rows.append([f"{NAME[ds]} / {TASK[t]}", f"{npd} / {nhc}", cell(a, "uncorrected"),
                         cell(w, "uncorrected"),
                         f"{int(nm.n_pd.mean())}" if not nm.empty else "--",
                         cell(a, "agematched"), cell(w, "agematched"), cell(w, "agesexmatched")])
    body = tab(["", "", "\\multicolumn{2}{c}{\\textbf{Uncorrected}}",
                "\\multicolumn{3}{c}{\\textbf{Age-matched}}", "\\textbf{Age, sex}"],
               [["\\textbf{Dataset / task}", "\\textbf{PD / HC}", "\\textbf{Age}",
                 "\\textbf{Voice}", "\\textbf{Pairs}", "\\textbf{Age}", "\\textbf{Voice}",
                 "\\textbf{Voice}"], "\\midrule", *rows], "lccccccc", "\\small", "4pt")
    body = body.replace("\\midrule\n\\textbf{Dataset", "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n\\textbf{Dataset", 1)
    write("u_within", body,
          "Within-dataset performance of the framework for every dataset and task, before and "
          "after matching on age.",
          "tab:u-within",
          "Participant-level AUROC, mean over five seeds; 5-fold participant-disjoint "
          "cross-validation of the framework head at layer 11. Age: AUROC of age alone. "
          "Pairs: participants per group after frequency matching in five-year bins. Age, "
          "sex: matched jointly on age and sex. -- : the dataset does not record the variable.")


def t_exp(R, S, exp, conds, caption, label, note):
    E = R[R.exp == exp]
    rows = []
    for src in [d for d in ORDER_DS if d in set(E.source)]:
        for (tgt, tt), g in E[E.source == src].groupby(["target", "target_task"], sort=False):
            for coh in ("released", "agematched"):
                gc = g[g.target_cohort == coh]
                if gc.empty:
                    continue
                cells = [ms(gc[gc.condition == c].auroc) if not gc[gc.condition == c].empty else "--"
                         for c in conds]
                lab = f"{NAME[tgt]} / {TASK[tt]}" + ("" if coh == "released" else ", matched")
                n = gc.iloc[0]
                rows.append(([f"{NAME[src]}"] if exp == "exp2" else []) +
                            [lab, f"{int(gc.n_pd.mean())} / {int(gc.n_hc.mean())}", *cells])
    head = (["\\textbf{Trained on}"] if exp == "exp2" else []) + \
        ["\\textbf{Tested on}", "\\textbf{PD / HC}"] + [f"\\textbf{{{COND[c]}}}" for c in conds]
    body = tab(head, rows, ("l" if exp == "exp2" else "") + "lc" + "c" * len(conds),
               "\\footnotesize", "2pt")
    write(f"u_{exp}", body, caption, label, note)


def t_spec(R):
    D = R[R.exp == "spec_direct"]
    C = R[R.exp == "spec_confusion"]
    fams = ["voice disorders", "respiratory", "mood", "cognitive impairment", "all other disorders"]
    rows = [r"\multicolumn{4}{l}{\emph{(a) A classifier trained to separate PD from each group, age-matched}} \\"]
    for f in fams:
        rows.append([f"PD against {f}"] + [ms(D[(D.target == f) & (D.source_task == t)].auroc)
                                           for t in ("vowel", "ddk", "speech")])
    rows.append(r"\midrule\multicolumn{4}{l}{\emph{(b) The PD-against-control model scoring participants it never saw}} \\")
    for name, lab in (("PD vs control", "PD against controls"),
                      ("cognitive impairment vs control", "Cognitive impairment against controls"),
                      ("cognitive impairment vs control (age-matched to each other)",
                       r"\quad the same, matched to the controls on age"),
                      ("PD vs cognitive impairment", "PD against cognitive impairment"),
                      ("PD vs voice disorders", "PD against voice disorders"),
                      ("PD vs all other disorders", "PD against all other disorders")):
        rows.append([lab] + [ms(C[(C.target == name) & (C.source_task == t)].auroc)
                             for t in ("vowel", "ddk", "speech")])
    body = tab(["", r"\textbf{Vowel}", r"\textbf{DDK}", r"\textbf{Speech}"], rows, "lccc",
               r"\small", "4pt")
    write("u_spec", body,
          "Specificity: Parkinson's disease against other disorders in Bridge2AI.",
          "tab:u-spec",
          r"Participant-level AUROC, mean $\pm$ standard deviation over five seeds; layer 11, the "
          "framework head. (a) participant-disjoint 5-fold cross-validation after matching each "
          "pair of groups on age. (b) a model trained on age-matched patients and healthy controls; "
          "patients and controls are scored out of fold, and every other participant is scored by "
          "a model that never saw them. A value near 0.5 for PD against a disorder means the model "
          "cannot tell the two apart.")


def t_master(R):
    """Headline figures, all from the layer-11 protocol, with where each is reported."""
    import json
    W = R[R.exp == "within"]
    A = R[R.exp == "within_age_only"]
    E1 = R[R.exp == "exp1"]
    E2 = R[R.exp == "exp2"]
    C = R[R.exp == "spec_confusion"]
    clin = [("ipvs", "vowel"), ("neurovoz", "vowel"), ("mdvr_kcl", "speech")]

    def w(ds, t, c):
        return W[(W.source == ds) & (W.source_task == t) & (W.condition == c)].auroc.mean()

    def a(ds, t, c):
        return A[(A.source == ds) & (A.source_task == t) & (A.condition == c)].auroc.mean()

    def mt(c):
        g = E1[(E1.condition == c) & (E1.target_cohort == "released")]
        m = np.zeros(len(g), bool)
        for t, tt in clin:
            m |= ((g.target == t) & (g.target_task == tt)).to_numpy()
        return g[m].groupby("seed").auroc.mean().mean()

    def e1(tgt, tt, c, coh):
        return E1[(E1.target == tgt) & (E1.target_task == tt) & (E1.condition == c)
                  & (E1.target_cohort == coh)].auroc.mean()

    def e2(src, tgt, c, coh):
        g = E2[(E2.source == src) & (E2.target == tgt) & (E2.condition == c)]
        gc = g[g.target_cohort == coh]
        return (gc if not gc.empty else g[g.target_cohort == "released"]).auroc.mean()

    def c(name):
        return C[(C.target == name) & (C.source_task == "speech")].auroc.mean()
    rp = json.loads((LOGS / "restricted_permutation__L11__fast__n1000.json").read_text())
    f = lambda x: f"{x:.3f}"
    sec = lambda lab: f"\\S\\ref{{{lab}}}"
    rows = [
        r"\multicolumn{3}{l}{\emph{Within each dataset (uncorrected $\rightarrow$ age-matched)}} \\",
        ["mPower vowel: age alone", f"{f(a('mpower','vowel','uncorrected'))} $\\rightarrow$ {f(a('mpower','vowel','agematched'))}", sec("sec:within")],
        ["mPower vowel: voice", f"{f(w('mpower','vowel','uncorrected'))} $\\rightarrow$ {f(w('mpower','vowel','agematched'))}", sec("sec:within")],
        ["Bridge2AI vowel: age alone", f"{f(a('bridge2ai','vowel','uncorrected'))} $\\rightarrow$ {f(a('bridge2ai','vowel','agematched'))}", sec("sec:within")],
        ["Bridge2AI, same speakers, age-matched: speech / DDK / vowel",
         f"{f(w('bridge2ai','speech','agematched'))} / {f(w('bridge2ai','ddk','agematched'))} / {f(w('bridge2ai','vowel','agematched'))}", sec("sec:within")],
        r"\midrule\multicolumn{3}{l}{\emph{Experiment 1: trained on mPower vowels}} \\",
        ["Within mPower / mean transfer, uncorrected", f"{f(w('mpower','vowel','uncorrected'))} / {f(mt('uncorrected'))}", sec("subsec:diagnosis")],
        ["Within mPower / mean transfer, confident labels", f"{f(w('mpower','vowel','confident'))} / {f(mt('confident'))}", sec("subsec:diagnosis")],
        ["Within mPower / mean transfer, age-matched", f"{f(w('mpower','vowel','agematched'))} / {f(mt('agematched'))}", sec("subsec:diagnosis")],
        ["Within mPower / mean transfer, age- and sex-matched", f"{f(w('mpower','vowel','agesexmatched'))} / {f(mt('agesexmatched'))}", sec("subsec:diagnosis")],
        ["Gap, uncorrected $\\rightarrow$ age-matched (points)",
         f"{100*(w('mpower','vowel','uncorrected')-mt('uncorrected')):.1f} $\\rightarrow$ {100*(w('mpower','vowel','agematched')-mt('agematched')):.1f}", sec("subsec:diagnosis")],
        ["Bridge2AI vowel, as released / age-matched test",
         f"{f(e1('bridge2ai','vowel','uncorrected','released'))} / {f(e1('bridge2ai','vowel','uncorrected','agematched'))}", sec("subsec:designA")],
        ["Restricted permutation: observed / age-preserving null",
         f"{rp['observed']:.3f} / {rp['within_age_bins']['null_mean']:.3f}", sec("subsec:diagnosis")],
        r"\midrule\multicolumn{3}{l}{\emph{Experiment 2: trained on connected speech (age-matched training and test)}} \\",
        ["Bridge2AI $\\rightarrow$ IPVS / NeuroVoz / MDVR-KCL",
         " / ".join(f(e2('bridge2ai', t, 'agematched', 'agematched')) for t in ("ipvs", "neurovoz", "mdvr_kcl")), sec("sec:exp2")],
        ["NeuroVoz / IPVS / MDVR-KCL$^{a}$ $\\rightarrow$ Bridge2AI",
         " / ".join(f(e2(s_, 'bridge2ai', 'agematched' if s_ != 'mdvr_kcl' else 'uncorrected', 'agematched'))
                    for s_ in ("neurovoz", "ipvs", "mdvr_kcl")), sec("sec:exp2")],
        r"\midrule\multicolumn{3}{l}{\emph{Specificity (Bridge2AI speech model, age-matched)}} \\",
        ["PD / cognitive impairment, each against controls",
         f"{f(c('PD vs control'))} / {f(c('cognitive impairment vs control (age-matched to each other)'))}", sec("sec:specificity")],
        ["PD against cognitive impairment", f(c('PD vs cognitive impairment')), sec("sec:specificity")],
    ]
    body = tab([r"\textbf{Quantity}", r"\textbf{AUROC}", r"\textbf{Where}"], rows, "llc",
               r"\footnotesize", "4pt")
    write("t4_0_master", body,
          "Headline figures of this thesis. Every figure comes from the same protocol: layer 11, "
          "the framework head with locked calibration, five seeds.",
          "tab:master",
          "Mean over five seeds; participant-level AUROC. Mean transfer: the average over IPVS "
          "vowels, NeuroVoz vowels and MDVR-KCL speech, the three clinical test sets. "
          "Appendix~\\ref{app:tables} gives every seed. $^{a}$MDVR-KCL records no age, so it is "
          "trained on as released.")


def main():
    R, S = load()
    t_within(R)
    t_spec(R)
    t_master(R)
    t_exp(R, S, "exp1", ["uncorrected", "confident", "agematched", "agesexmatched"],
          "Experiment 1: the framework trained on mPower sustained vowels and applied, locked, "
          "to every task of every other dataset.",
          "tab:u-exp1",
          "Participant-level AUROC, mean $\\pm$ standard deviation over five seeds; layer 11. "
          "Columns: the mPower training cohort. Rows marked matched score the same locked "
          "predictions on test participants matched on age in five-year bins.")
    t_exp(R, S, "exp2", ["uncorrected", "agematched", "agesexmatched"],
          "Experiment 2: the framework trained on the connected speech of one dataset and "
          "applied, locked, to the connected speech of every other dataset.",
          "tab:u-exp2",
          "Participant-level AUROC, mean $\\pm$ standard deviation over five seeds; layer 11. "
          "Columns: the training cohort. MDVR-KCL records no age, so it is trained on as "
          "released only. Rows marked matched score the same predictions on test "
          "participants matched on age.")


if __name__ == "__main__":
    main()
