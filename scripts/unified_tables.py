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
def t_within(R, S):
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
                         cell(a, "agematched"), cell(w, "agematched"),
                         pooled_ci(S, exp="within", source=ds, source_task=t,
                                   condition="agematched") if not nm.empty else "--",
                         cell(w, "agesexmatched")])
    body = tab(["", "", "\\multicolumn{2}{c}{\\textbf{Uncorrected}}",
                "\\multicolumn{4}{c}{\\textbf{Age-matched}}", "\\textbf{Age, sex}"],
               [["\\textbf{Dataset / task}", "\\textbf{PD / HC}", "\\textbf{Age}",
                 "\\textbf{Voice}", "\\textbf{Pairs}", "\\textbf{Age}", "\\textbf{Voice}",
                 "\\textbf{95\\% CI}", "\\textbf{Voice}"], "\\midrule", *rows], "lcccccccc",
               "\\small", "3pt")
    body = body.replace("\\midrule\n\\textbf{Dataset", "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\n\\textbf{Dataset", 1)
    write("u_within", body,
          "Within-dataset performance of the framework for every dataset and task, before and "
          "after matching on age.",
          "tab:u-within",
          "Participant-level AUROC, mean over five seeds; 5-fold participant-disjoint "
          "cross-validation of the framework head at layer 11. 95\\% CI: participant "
          "bootstrap of the age-matched voice result, 200 resamples per seed, pooled over "
          "seeds. Age: AUROC of age alone. "
          "Pairs: participants per group after frequency matching in five-year bins. Age, "
          "sex: matched jointly on age and sex. -- : the dataset does not record the variable.")


def t_exp(R, S, exp, conds, caption, label, note, ci_cond=None):
    E = R[R.exp == exp]
    src_col = exp in ("exp2", "exp3")              # several source datasets: name each one
    rows = []
    for src in [d for d in ORDER_DS if d in set(E.source)]:
        for (tgt, tt), g in E[E.source == src].groupby(["target", "target_task"], sort=False):
            for coh in ("released", "agematched"):
                gc = g[g.target_cohort == coh]
                if gc.empty:
                    continue
                cells = [ms(gc[gc.condition == c].auroc) if not gc[gc.condition == c].empty else "--"
                         for c in conds]
                if ci_cond:
                    ci = (pooled_ci(S, exp=exp, source=src, condition=ci_cond, target=tgt,
                                    target_task=tt, target_cohort=coh)
                          if not gc[gc.condition == ci_cond].empty else "")
                    cells.insert(conds.index(ci_cond) + 1, ci or "--")
                lab = f"{NAME[tgt]} / {TASK[tt]}" + ("" if coh == "released" else ", matched")
                n = gc.iloc[0]
                rows.append(([f"{NAME[src]}"] if src_col else []) +
                            [lab, f"{int(gc.n_pd.mean())} / {int(gc.n_hc.mean())}", *cells])
    head = (["\\textbf{Trained on}"] if src_col else []) + \
        ["\\textbf{Tested on}", "\\textbf{PD / HC}"] + [f"\\textbf{{{COND[c]}}}" for c in conds]
    if ci_cond:
        head.insert(head.index(f"\\textbf{{{COND[ci_cond]}}}") + 1, "\\textbf{95\\% CI}")
    body = tab(head, rows, ("l" if src_col else "") + "lc" + "c" * (len(head) - 2 - src_col),
               "\\scriptsize" if ci_cond else "\\footnotesize", "2pt")
    write(f"u_{exp}", body, caption, label, note)


def t_spec(R):
    D = R[R.exp == "spec_direct"]
    C = R[R.exp == "spec_confusion"]
    fams = ["voice disorders", "respiratory", "mood", "cognitive impairment", "all other disorders"]
    rows = [r"\multicolumn{4}{l}{\emph{(a) Trained to separate PD from each group, matched on age and country}} \\"]
    for f in fams:
        rows.append([f"PD against {f}"] + [ms(D[(D.target == f) & (D.source_task == t)].auroc)
                                           for t in ("vowel", "ddk", "speech")])
    rows.append(r"\midrule\multicolumn{4}{l}{\emph{(b) The US PD-against-control model scoring unseen Canadians}} \\")
    rows.append(["US PD against US controls (out of fold)"]
                + [ms(C[(C.target == "PD vs control") & (C.source_task == t)].auroc)
                   for t in ("vowel", "ddk", "speech")])
    for name, lab in (("PD vs cognitive impairment (Canada)", "Canadian PD against cognitive impairment"),
                      ("PD vs all other disorders (Canada)", "Canadian PD against all other disorders")):
        rows.append([lab] + [ms(C[(C.target == name) & (C.source_task == t)].auroc)
                             for t in ("vowel", "ddk", "speech")])
    body = tab(["", r"\textbf{Vowel}", r"\textbf{DDK}", r"\textbf{Speech}"], rows, "lccc",
               r"\small", "4pt")
    write("u_spec", body,
          "Specificity: Parkinson's disease against other disorders in Bridge2AI.",
          "tab:u-spec",
          r"Participant-level AUROC, mean $\pm$ standard deviation over five seeds; layer 11, the "
          "framework head. Every Canadian participant in the PD-against-control cohort is a "
          "patient and most participants with cognitive impairment are Canadian, so every "
          "comparison is made within one country. (a) participant-disjoint 5-fold "
          "cross-validation after matching each pair of groups on age and country. (b) a model "
          "trained on age-matched US patients and controls, then applied to the Canadian "
          "patients and the Canadian participants with other disorders, none of whom it saw; "
          "both groups in a row are scored by the same locked model. A value near 0.5 for PD "
          "against a disorder means the model cannot tell the two apart.")


def t_exp1_changes():
    """Per-test-set change in Experiment 1 transfer for each training condition."""
    import json
    p = LOGS / "unified" / "exp1_changes.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    rows = []
    for tgt, task in [("ipvs", "vowel"), ("neurovoz", "vowel"), ("bridge2ai", "vowel"),
                      ("mdvr_kcl", "speech"), ("bridge2ai", "speech"), ("ipvs", "speech"),
                      ("neurovoz", "speech"), ("bridge2ai", "ddk"), ("neurovoz", "ddk")]:
        cells = []
        for cond in ("agematched", "sizematched", "confident"):
            r = d.get(f"{tgt} {task} | {cond} - uncorrected")
            lo, hi = r["ci95"]
            cells.append(f"{r['diff']:+.3f} [{lo:+.2f}, {hi:+.2f}]")
        base = d[f"{tgt} {task} | agematched - uncorrected"]["auroc_b"]
        rows.append([f"{NAME[tgt]} / {TASK[task]}", f"{base:.3f}", *cells])
    body = tab([r"\textbf{Test set}", r"\textbf{Uncorrected}", r"\textbf{Age-matched}",
                r"\textbf{Same size}", r"\textbf{Confident}"], rows, "lcccc",
               r"\scriptsize", "2pt")
    write("u_exp1_changes", body,
          "Change in Experiment 1 transfer from the uncorrected training cohort, per test set.",
          "tab:u-exp1-changes",
          "Same test participants for both models; AUROC difference with a paired participant "
          "bootstrap 95\\% interval (200 resamples per seed, five seeds). Same size: a random "
          "sample of the uncorrected cohort with as many patients and controls as the "
          "age-matched one, so the age gap is kept and only the size changes. Test sets as "
          "released.")


def t_task_compare(src="task_comparison.json", name="u_task_compare",
                   caption=None, label="tab:u-task-compare", extra=""):
    """Speech against the vowel and DDK on the same participants (and, for the equal-audio
    control, the same amount of audio), from task_comparison.py or equal_audio.py output."""
    import json
    p = LOGS / "unified" / src
    if not p.exists():
        return
    d = json.loads(p.read_text())
    tl = {"speech": "Speech", "vowel": "Vowel", "ddk": "DDK"}

    def row(label, r, other):
        lo, hi = r["ci95"]
        return [label, f"{r['auroc_a']:.3f}", f"{r['auroc_b']:.3f} ({tl[other]})",
                f"{r['diff']:+.3f} [{lo:+.2f}, {hi:+.2f}]",
                f"{r['a_better_in_seeds']}/{r['n_seeds']}", f"{r['n_pd_hc'][0]} / {r['n_pd_hc'][1]}"]

    rows = [r"\multicolumn{6}{l}{\emph{(a) Within each dataset, age-matched, the same participants}} \\"]
    for ds in ("bridge2ai", "neurovoz", "ipvs"):
        for other in ("vowel", "ddk"):
            r = d["within"].get(f"{ds} | agematched | speech - {other}")
            if r:
                rows.append(row(NAME[ds], r, other))
    rows.append(r"\midrule\multicolumn{6}{l}{\emph{(b) Transfer, trained age-matched, the same target participants}} \\")
    for key, r in d["transfer"].items():
        pair, train, test, tasks = [x.strip() for x in key.split("|")]
        if train != "train agematched" or not tasks.startswith("speech"):
            continue
        src, tgt = pair.split("->")
        other = tasks.split(" - ")[1]
        if test == "test released" and f"{pair} | {train} | test agematched | {tasks}" in d["transfer"]:
            continue                               # prefer the age-matched test cohort
        lab = f"{NAME[src]} $\\rightarrow$ {NAME[tgt]}" + (", matched" if test == "test agematched" else "")
        rows.append(row(lab, r, other))
    # sign test across the transfer directions: is speech ahead of the vowel more often than
    # chance, without relying on any single direction's unadjusted interval?
    from scipy.stats import binomtest
    vd = [r["diff"] for k, r in d["transfer"].items()
          if "train agematched | test agematched" in k and k.endswith("speech - vowel")]
    ahead = sum(x > 0 for x in vd)
    sign = binomtest(ahead, len(vd), 0.5, alternative="greater").pvalue
    extra = (f" Across the {len(vd)} transfer directions speech is ahead of the vowel in "
             f"{ahead} (one-sided sign test $p = {sign:.3f}$); the per-direction intervals are "
             "not adjusted for multiplicity." + extra)
    body = tab([r"\textbf{Comparison}", r"\textbf{Speech}", r"\textbf{Other task}",
                r"\textbf{Difference [95\% CI]}", r"\textbf{Ahead}", r"\textbf{PD / HC}"],
               rows, "lccccc", r"\scriptsize", "2pt")
    write(name, body,
          caption or "Connected speech against the vowel and diadochokinesis with the training "
          "source and the participants held fixed.",
          label,
          "Participant-level AUROC, mean over five seeds; only participants scored under both "
          "tasks. Difference: speech minus the other task, with a participant-bootstrap 95\\% "
          "interval (200 resamples per seed, pooled). Ahead: seeds in which speech "
          "scores higher. MDVR-KCL has no vowel or DDK recordings." + extra)


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

    def d(name):
        D = R[R.exp == "spec_direct"]
        return D[(D.target == name) & (D.source_task == "speech")].auroc.mean()
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
        r"\midrule\multicolumn{3}{l}{\emph{Specificity (Bridge2AI speech, within one country)}} \\",
        ["PD against cognitive impairment / voice disorders, direct classifier",
         f"{f(d('cognitive impairment'))} / {f(d('voice disorders'))}", sec("sec:specificity")],
        ["Canadian PD against Canadian cognitive impairment, US model",
         f(c('PD vs cognitive impairment (Canada)')), sec("sec:specificity")],
    ]
    body = tab([r"\textbf{Quantity}", r"\textbf{AUROC}", r"\textbf{Where}"], rows, "llc",
               r"\footnotesize", "4pt")
    write("t4_0_master", body,
          "Headline figures of this thesis. Every figure comes from the same protocol (layer 11, "
          "the framework head with locked calibration, five seeds) except the restricted "
          "permutation test, whose set-up is given in Table~\\ref{tab:protocols}.",
          "tab:master",
          "Mean over five seeds; participant-level AUROC. Mean transfer: the average over IPVS "
          "vowels, NeuroVoz vowels and MDVR-KCL speech, the three clinical test sets. "
          "Table~\\ref{tab:appA-seeds} gives every seed. $^{a}$MDVR-KCL records no age, so it is "
          "trained on as released.")


def main():
    R, S = load()
    t_within(R, S)
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
          "participants matched on age. 95\\% CI: participant bootstrap of the age-matched "
          "column, 200 resamples per seed, pooled over seeds.", ci_cond="agematched")
    t_exp(R, S, "exp3", ["uncorrected", "agematched", "agesexmatched"],
          "Same-source control of Experiment 2: the framework trained on the sustained vowel or the diadochokinetic "
          "task of one dataset and applied, locked, to the same task in every other dataset.",
          "tab:u-exp3",
          "The design of Experiment 2 with the task changed, so that tasks can be compared with "
          "the training source held fixed. Participant-level AUROC, mean $\\pm$ standard "
          "deviation over five seeds; layer 11. 95\\% CI: participant bootstrap of the "
          "age-matched column.", ci_cond="agematched")
    t_task_compare()
    t_exp1_changes()
    t_task_compare("equal_audio_comparison.json", "u_equal_audio",
                   "The comparison of Table~\\ref{tab:u-task-compare} with the same amount of "
                   "audio for every task: three two-second segments per participant.",
                   "tab:u-equal-audio",
                   " Each participant contributes three 2~s segments per task (the vowel's stable "
                   "core, or 2~s cuts of the voice-active diadochokinesis and speech); participants "
                   "with fewer are left out of that task.")


if __name__ == "__main__":
    main()
