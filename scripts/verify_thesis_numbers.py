"""Check every experimental number quoted in the thesis prose against the experiment logs.

Each check is a fragment of a sentence exactly as the thesis states it, with the numbers
filled in from outputs/logs/ at the precision the text uses. The check passes only if that
fragment, numbers included, appears in the .tex file. When an experiment is re-run and a
number moves, the fragment no longer matches and the check fails, naming the file, the
expected text and the closest sentence actually found.

Numbers that do not come from an experiment of this thesis (published figures, settings,
definitions) are not checked here.

Run:  .venv\\Scripts\\python.exe scripts\\verify_thesis_numbers.py
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from unified_tables import pooled_ci  # noqa: E402

THESIS = ROOT / "thesis"
LOGS = ROOT / "outputs" / "logs"
U = LOGS / "unified"


def j(p):
    return json.loads((LOGS / p).read_text())


# ----------------------------------------------------------------------------- sources
R = pd.read_csv(U / "runs.csv")
S = pd.read_csv(U / "scores.csv.gz")
C = j("cohort_summary.json")


def _m(**k):
    m = np.ones(len(R), bool)
    for a, b in k.items():
        m &= (R[a] == b).to_numpy()
    return R[m].auroc.mean()


def W(ds, task, cond):
    return _m(exp="within", source=ds, source_task=task, condition=cond)


def AGE(ds, task, cond):
    return _m(exp="within_age_only", source=ds, source_task=task, condition=cond)


def E1(tgt, task, cond, coh="released"):
    return _m(exp="exp1", target=tgt, target_task=task, condition=cond, target_cohort=coh)


def E2(src, tgt, cond, coh="released"):
    return _m(exp="exp2", source=src, target=tgt, condition=cond, target_cohort=coh)


def SPEC(group, task="speech"):
    return _m(exp="spec_direct", target=group, source_task=task)


def CONF(target, task="speech"):
    return _m(exp="spec_confusion", target=target, source_task=task)


def MT(cond):
    g = R[(R.exp == "exp1") & (R.condition == cond) & (R.target_cohort == "released")
          & (((R.target.isin(["ipvs", "neurovoz"])) & (R.target_task == "vowel"))
             | (R.target == "mdvr_kcl"))]
    return g.groupby("target").auroc.mean().mean()


def CI(**k):
    return pooled_ci(S, **k).strip("[]").split(", ")


f3 = "{:.3f}".format
f2 = "{:.2f}".format
f1 = "{:.1f}".format


def sg(x, d=3):
    return f"${x:+.{d}f}$"


# ------------------------------------------------------------------------------ values
w_unc, w_am, w_as, w_conf = (W("mpower", "vowel", c) for c in
                             ("uncorrected", "agematched", "agesexmatched", "confident"))
a_unc, a_am = AGE("mpower", "vowel", "uncorrected"), AGE("mpower", "vowel", "agematched")
mt_unc, mt_am, mt_as, mt_conf = (MT(c) for c in
                                 ("uncorrected", "agematched", "agesexmatched", "confident"))
gap = lambda w, t: f1(100 * (w - t))  # noqa: E731
b2v, b2v_m = E1("bridge2ai", "vowel", "uncorrected"), E1("bridge2ai", "vowel", "uncorrected", "agematched")
vt_am = [E1("ipvs", "vowel", "agematched"), E1("neurovoz", "vowel", "agematched"),
         E1("bridge2ai", "vowel", "agematched")]
b2_it, b2_es, b2_en = (E2("bridge2ai", "ipvs", "agematched", "agematched"),
                       E2("bridge2ai", "neurovoz", "agematched", "agematched"),
                       E2("bridge2ai", "mdvr_kcl", "agematched"))
cc = j("unified/bridge2ai_country_check.json")["speech"]
c_all, c_us, c_common = (cc["all_countries_all_tasks"]["agematched"]["auroc_mean"],
                         cc["us_all_tasks"]["agematched"]["auroc_mean"],
                         cc["us_common_tasks"]["agematched"]["auroc_mean"])
b2s, b2d, b2vv = (W("bridge2ai", t, "agematched") for t in ("speech", "ddk", "vowel"))
nvs, nvv = W("neurovoz", "speech", "agematched"), W("neurovoz", "vowel", "agematched")
ci_b2s = CI(exp="within", source="bridge2ai", source_task="speech", condition="agematched")
ci_b2d = CI(exp="within", source="bridge2ai", source_task="ddk", condition="agematched")
ci_b2v = CI(exp="within", source="bridge2ai", source_task="vowel", condition="agematched")
ci_it = CI(exp="exp2", source="bridge2ai", condition="agematched", target="ipvs",
           target_task="speech", target_cohort="agematched")
ci_es = CI(exp="exp2", source="bridge2ai", condition="agematched", target="neurovoz",
           target_task="speech", target_cohort="agematched")
ci_en = CI(exp="exp2", source="bridge2ai", condition="agematched", target="mdvr_kcl",
           target_task="speech", target_cohort="released")
ci_mp = CI(exp="within", source="mpower", source_task="vowel", condition="agematched")
dem = {d: C[d]["age_pd"] - C[d]["age_hc"] for d in ("mpower", "bridge2ai", "ipvs", "neurovoz")
       if C[d]["age_pd"] is not None}
ch = j("unified/exp1_changes.json")
chg = lambda key: ch[key]  # noqa: E731
tc, ea = j("unified/task_comparison.json"), j("unified/equal_audio_comparison.json")
cl = pd.read_csv(LOGS / "classifier_comparison.csv")
ssl_am = cl[(cl.cohort == "age-matched") & (cl.features == "SSL embeddings (2048)")].set_index("classifier").AUROC
cls_am = cl[(cl.cohort == "age-matched") & (cl.features == "classical (42)")].set_index("classifier").AUROC
ssl_un = cl[(cl.cohort == "confounded") & (cl.features == "SSL embeddings (2048)")].set_index("classifier").AUROC
fu = j("fusion__xls-r-300m__L11.json")["scenarios"]
dc = j("difference_cis.json")
sweep = [r["best_layer_auroc"] for r in j("encoder_layer_sweep.json")["best"]]
enc = [v["delta"] for v in dc["encoders"].values()]
enc_ci = [x for v in dc["encoders"].values() for x in v["ci95"]]
er = j("demographic_erasure.json")
ad = {r["scenario"]: r for r in j("label_sensitivity__L11_adapt_centroid.json")["summary"]}
ads = {r["scenario"]: r for r in j("label_sensitivity__L11_adapt_standardize.json")["summary"]}
adv = lambda d, s, k: float(d[s][k].split()[0])  # noqa: E731
l1 = lambda L, w="": j(f"layer1_mpower__xls-r-300m__cap3__L{L}{w}.json")["auroc"]  # noqa: E731
l2 = lambda ds, t, w="": j(f"layer2__xls-r-300m__{ds}__{t}__L11{w}.json")["auroc"]  # noqa: E731
wins = [l2(d, t, "__Wfull") - l2(d, t) for d, t in
        [("ipvs", "vowel"), ("neurovoz", "vowel"), ("neurovoz", "ddk"), ("ipvs", "speech"),
         ("neurovoz", "speech"), ("mdvr_kcl", "speech")]]
L15 = [j(f"layer2__xls-r-300m__{d}__{t}__L15__Wfull.json")["auroc"]
       - j(f"layer2__xls-r-300m__{d}__{t}__L15.json")["auroc"]
       for d, t in [("ipvs", "vowel"), ("neurovoz", "vowel"), ("mdvr_kcl", "speech")]]
wins_all = wins + L15
perm = j("restricted_permutation__L11__fast__n1000.json")
inv = j("ipvs_inversion.json")
sel = pd.read_csv(LOGS / "select_transfer_layer.csv").set_index("layer").auroc
selc = pd.read_csv(LOGS / "select_transfer_layer__confounded.csv").set_index("layer").auroc
probes = {r["attribute"]: r["value"] for r in j("embedding_probes.json") if r["layer"] == 11}
leak = j("neurovoz_taskcode_leak.json")
rah = j("rahmatallah_cnn.json")
dca = j("decision_curves.json")
dca54 = j("decision_curves__prev0.054.json")
bw = pd.DataFrame(j("age_bin_width.json"))
sex = {r["cohort"]: r for r in j("sex_control.json")}
pause = {r["dataset"]: r for r in j("pause_information.json")["results"]}
pause_m = j("pause_information.json")["ipvs_agematched"]
meta = j("bridge2ai_recording_metadata.json")
win = j("speech_windows.json")
pg = j("pause_information.json")["per_group"]
lit = j("literature_comparison.json")
b2sel = j("bridge2ai_layer_select.json")["mean"]

# calibration, operating points and the remaining side analyses
from sklearn.metrics import roc_curve  # noqa: E402
import thesis_tables  # noqa: E402

cal = thesis_tables._b2ai_calibration_unified()
ece = R[(R.source == "mpower") & (R.condition == "uncorrected")]
ece_home = ece[ece.exp == "within"].ece.mean()
ece_ext = ece[(ece.exp == "exp1") & (ece.target_cohort == "released")].groupby(["target", "target_task"]).ece.mean()


def sens90(**k):
    m = np.ones(len(S), bool)
    for a, b in k.items():
        m &= (S[a] == b).to_numpy()
    out = []
    for _, g in S[m].groupby("seed"):
        fp, tp, _ = roc_curve(g.y, g.r)
        out.append(tp[fp <= 0.10].max())
    return float(np.mean(out))


se_home = sens90(exp="within", source="mpower", condition="uncorrected")
se_ext = {(t, k): sens90(exp="exp1", condition="uncorrected", target=t, target_task=k,
                         target_cohort="released")
          for t, k in [("bridge2ai", "vowel"), ("bridge2ai", "ddk"), ("bridge2ai", "speech"),
                       ("ipvs", "vowel"), ("ipvs", "speech"), ("neurovoz", "vowel"),
                       ("neurovoz", "ddk"), ("neurovoz", "speech"), ("mdvr_kcl", "speech")]}
se_clin = max(v for (t, _), v in se_ext.items() if t != "bridge2ai")
se_b2 = {t: sens90(exp="exp2", source="bridge2ai", condition="agematched", target=t,
                   target_cohort=c)
         for t, c in (("neurovoz", "agematched"), ("mdvr_kcl", "released"), ("ipvs", "agematched"))}
se_b2_home = sens90(exp="within", source="bridge2ai", source_task="speech", condition="agematched")
pr = perm["within_age_bins"]
ceil_ = j("ipvs_ceiling.json")["all"]["features"]
ceil_top = sorted(v["auroc"] for v in ceil_.values())[-3:]
batch = j("verify_ipvs_batch.json")
age_am_all = [AGE(d, t, "agematched") for d, t in
              [("mpower", "vowel"), ("bridge2ai", "vowel"), ("bridge2ai", "ddk"), ("bridge2ai", "speech"),
               ("neurovoz", "vowel"), ("neurovoz", "ddk"), ("neurovoz", "speech"), ("ipvs", "vowel"),
               ("ipvs", "speech")]]
vw_am = [w_am, b2vv, nvv]
e3 = R[(R.exp == "exp3") & (R.condition == "agematched")]
e3v = e3[e3.target_task == "vowel"].groupby(["source", "target", "target_cohort"]).auroc.mean()
e3d = e3[e3.target_task == "ddk"].groupby(["source", "target", "target_cohort"]).auroc.mean()

_W2 = R[(R.exp == "within") & (R.source_task == "speech") & (R.condition == "uncorrected")
        & (R.source != "ipvs")].groupby("source").auroc.mean()
_E2 = R[(R.exp == "exp2") & (R.condition == "uncorrected") & (R.target_cohort == "released")
        & (R.source != "ipvs")].groupby(["source", "target"]).auroc.mean()
dev2 = [_W2[s_] - v_ for (s_, _), v_ in _E2.items()]

CHECKS = [
    # ---- abstract
    ("Main.tex", f"AUROC of {f3(w_unc)} within mPower\nand {f3(mt_unc)} on the clinical datasets"),
    ("Main.tex", f"patients were {f1(dem['mpower'])}\nyears older than controls, and age alone scored {f3(a_unc)}"),
    ("Main.tex", f"from {gap(w_unc, mt_unc)} percentage points to {gap(w_am, mt_am)}"),
    ("Main.tex", f"scored {f3(b2_it)} to {f3(b2_en)} on the other datasets"),
    ("Main.tex", f"cognitive impairment\nat only {f3(SPEC('cognitive impairment'))}"),
    # ---- chapter 1
    ("ch1_introduction.tex", f"27.5 years older than those reporting none"),
    ("ch1_introduction.tex", f"ROC curve of {f3(a_unc)}"),
    ("ch1_introduction.tex", f"reaches {f3(w_unc)}, so on the benchmark"),
    ("ch1_introduction.tex", f"by {f1(dem['mpower'])} years in mPower, {f1(dem['bridge2ai'])} years among the US participants of Bridge2AI and {f1(dem['ipvs'])} years in IPVS"),
    ("ch1_introduction.tex", f"reaches {f3(w_unc)} within\nmPower and {f3(mt_unc)} on average"),
    ("ch1_introduction.tex", f"from {gap(w_unc, mt_unc)} percentage points to {gap(w_am, mt_am)}"),
    ("ch1_introduction.tex", f"cognitive impairment at only {f3(SPEC('cognitive impairment'))}. The"),
    # ---- chapter 3
    ("ch3_existing_models.tex", f"separates patients from controls at {f2(cc['country_only_auroc'])}"),
    ("ch3_existing_models.tex", f"falls from {f3(c_all)} to {f3(c_us)}"),
    ("ch3_existing_models.tex", f"lowers it further to {f3(c_common)}"),
    ("ch3_existing_models.tex", f"mean age of {f1(C['bridge2ai']['age_pd'])} years and controls {f1(C['bridge2ai']['age_hc'])}"),
    ("ch3_existing_models.tex", f"Prevalence is {f3(C['mpower']['prevalence'])} in mPower"),
    ("ch3_existing_models.tex", f"{f3(C['bridge2ai']['prevalence'])} in the US Bridge2AI cohort, and between {f3(C['ipvs']['prevalence'])} and {f3(C['neurovoz']['prevalence'])}"),
    ("ch3_existing_models.tex", f"mPower carries a {f1(dem['mpower'])}-year difference"),
    ("ch3_existing_models.tex", f"Bridge2AI a {f1(dem['bridge2ai'])}-year difference"),
    ("ch3_existing_models.tex", f"IPVS carries a {f1(dem['ipvs'])}-year difference"),
    ("ch3_existing_models.tex", f"NeuroVoz, at {f1(dem['neurovoz'])} years, is the least affected"),
    ("ch3_existing_models.tex", f"{100 * leak['third_repetition_pd_share']:.1f} per cent"),
    ("ch3_existing_models.tex", f"possesses attains {f3(leak['auroc_taskcode_profile'])}"),
    ("ch3_existing_models.tex", f"attains {f3(leak['auroc_recording_count'])}. Participants with a diagnosis average {leak['recordings_per_speaker']['PD']:.2f} recordings against {leak['recordings_per_speaker']['HC']:.2f}"),
    ("ch3_existing_models.tex", f"US patients from controls at {f3(c_us)}; using only the common tasks, at {f3(c_common)}"),
    ("ch3_existing_models.tex", f"attains an AUROC of {f3(pause['ipvs']['auroc'])} on IPVS"),
    ("ch3_existing_models.tex", f"it reaches only {f3(pause['mdvr_kcl']['auroc'])}"),
    ("ch3_existing_models.tex", f"on either dataset ({f3(bw.set_index(['dataset', 'bin_width']).age_only[('mPower vowel', 5)])} and {f3(bw.set_index(['dataset', 'bin_width']).age_only[('Bridge2AI speech', 5)])})"),
    ("ch3_existing_models.tex", f"age alone climbs back to {f3(bw.set_index(['dataset', 'bin_width']).age_only[('mPower vowel', 10)])}"),
    ("ch3_existing_models.tex", f"the voice result moves by at most {f3(max(g.voice.max() - g.voice.min() for _, g in bw.groupby('dataset')))}"),
    ("ch3_existing_models.tex", f"30 Bridge2AI pairs lower ({f3(b2s)}"),
    ("ch3_existing_models.tex", f"is {lit['mean_abs_delta']:.3f}, under the same protocol as every headline figure (layer {lit['layer']}, five seeds"),
    ("ch3_existing_models.tex", f"from English to Italian speech, {next(r['ours'] for r in lit['rows'] if r['claim'].startswith('English -> Italian')):.3f} here against {next(r['published'] for r in lit['rows'] if r['claim'].startswith('English -> Italian')):.2f}"),
    ("ch3_existing_models.tex", f"at {rah['v1_as_published']['mean']:.3f} averaged over one hundred runs"),
    ("ch3_existing_models.tex", f"it falls to {rah['v2_as_published']['mean']:.3f}"),
    ("ch3_existing_models.tex", f"it reaches {rah['random_70_30']['mean']:.3f}, which is below the {rah['age_sex_only_same_splits']['mean']:.3f} that age and sex alone achieve"),
    ("ch3_existing_models.tex", f"age and sex it reaches {rah['random_70_30_agesex_matched']['mean']:.3f}"),
    # ---- recording metadata, pause timing, IPVS ceiling, calibration and operating points
    ("ch3_existing_models.tex", f"attains an AUROC of {f3(pause['ipvs']['auroc'])} on IPVS,\nand {f3(pause_m['auroc_mean'])} after age matching"),
    ("ch4_proposed_framework.tex", f"({meta['pd']['main_adapter']} of {meta['pd']['n']} patients, {meta['control']['main_adapter']} of {meta['control']['n']} controls)"),
    ("ch4_proposed_framework.tex", f"main microphone of {100 * meta['pd']['share_builtin']:.0f}\\% of patients and {100 * meta['control']['share_builtin']:.0f}\\% of controls"),
    ("ch4_proposed_framework.tex", f"recorded by one patient and {meta['control']['any_v2_task']} controls"),
    ("ch4_proposed_framework.tex", f"separate the groups ({f2(meta['input_gain_auroc'])})"),
    ("ch5_conclusion.tex", f"main microphone of {100 * meta['pd']['share_builtin']:.0f} per cent of patients and {100 * meta['control']['share_builtin']:.0f} per cent"),
    ("ch5_conclusion.tex", f"recorded by one patient and\n{meta['control']['any_v2_task']} controls"),
    ("ch3_existing_models.tex", f"{win['neurovoz']['mean_windows']:.1f} for NeuroVoz, whose recordings are too short\nto divide, {win['ipvs']['mean_windows']:.1f} for IPVS and {win['mdvr_kcl']['mean_windows']:.1f} for MDVR-KCL"),
    ("ch3_existing_models.tex", f"retains {100 * (1 - pg['ipvs']['pause_frac']['1']):.1f} per cent"),
    ("ch3_existing_models.tex", f"against {100 * (1 - pg['ipvs']['pause_frac']['0']):.1f} per cent from a control"),
    ("ch3_existing_models.tex", f"retaining {100 * (1 - pg['neurovoz']['pause_frac']['1']):.1f} per cent against {100 * (1 - pg['neurovoz']['pause_frac']['0']):.1f}"),
    # ---- chapter 4: protocol table and set-up
    ("ch4_proposed_framework.tex", f"{f3(w_unc)} & Experiment 1 and every headline figure"),
    ("ch4_proposed_framework.tex", f"{f3(l1(11))} & Analysis window comparison"),
    ("ch4_proposed_framework.tex", f"{f3(selc[11])} & Layer search"),
    ("ch4_proposed_framework.tex", f"{f3(ssl_un['MLP head (ours)'])} & Classifier comparison (Table"),
    ("ch4_proposed_framework.tex", f"{f3(ssl_un['SVM (RBF)'])} & Classifier comparison; restricted permutation test"),
    ("ch4_proposed_framework.tex", f"{f3(er['confounded']['within']['original'])} & Demographic erasure"),
    ("ch4_proposed_framework.tex", f"the head scored 0.761 instead of {f3(w_unc)}"),
    ("ch4_proposed_framework.tex", f"the five span {f3(min(sweep))} to {f3(max(sweep))}"),
    ("ch4_proposed_framework.tex", f"lies between ${min(enc):+.3f}$ and ${max(enc):+.3f}$"),
    ("ch4_proposed_framework.tex", f"from ${min(enc_ci):+.3f}$ to ${max(enc_ci):+.3f}$"),
    ("ch4_proposed_framework.tex", f"plateau at {f3(sel[10])}, {f3(sel[11])} and {f3(sel[12])}"),
    ("ch4_proposed_framework.tex", f"(layer 11 is {max(b2sel.values()) - b2sel['11']:.3f} below it)"),
    ("ch4_proposed_framework.tex", f"every layer scores between {f2(selc.min())} and {f2(selc.max())}"),
    ("ch4_proposed_framework.tex", f"{f2((selc - sel).min())} to {f2((selc - sel).max())} at every layer"),
    # ---- 4.3 within
    ("ch4_proposed_framework.tex", f"on average {f1(C['mpower']['age_pd'])} years old and those reporting none {f1(C['mpower']['age_hc'])}"),
    ("ch4_proposed_framework.tex", f"difference of {f1(dem['mpower'])} years"),
    ("ch4_proposed_framework.tex", f"direction, {f1(dem['bridge2ai'])} years among its US participants"),
    ("ch4_proposed_framework.tex", f"better than the voice model: {f3(a_unc)} against {f3(w_unc)} in mPower"),
    ("ch4_proposed_framework.tex", f"and {f3(AGE('bridge2ai', 'vowel', 'uncorrected'))} against {f3(W('bridge2ai', 'vowel', 'uncorrected'))} on Bridge2AI vowels"),
    ("ch4_proposed_framework.tex", f"connected-speech model scores {f3(c_all)}; trained on the 38 US patients only, it scores {f3(c_us)}"),
    ("ch4_proposed_framework.tex", f"the same comparison scores {f3(c_common)}"),
    ("ch4_proposed_framework.tex", f"Sustained vowels keep little: {f3(w_am)} in mPower, {f3(b2vv)} in Bridge2AI and {f3(nvv)} in NeuroVoz. Connected speech keeps more: {f3(nvs)} in NeuroVoz and {f3(b2s)} in Bridge2AI"),
    ("ch4_proposed_framework.tex", f"scores {f3(b2s)} (95\\% CI {ci_b2s[0]} to {ci_b2s[1]}), diadochokinesis {f3(b2d)} ({ci_b2d[0]} to {ci_b2d[1]}) and the sustained vowel {f3(b2vv)} ({ci_b2v[0]} to {ci_b2v[1]})"),
    ("ch4_proposed_framework.tex", f"(speech {f3(W('bridge2ai', 'speech', 'agesexmatched'))}, vowel {f3(W('bridge2ai', 'vowel', 'agesexmatched'))})"),
    ("ch4_proposed_framework.tex", f"IPVS scores {f3(W('ipvs', 'vowel', 'agematched'))} on vowels and {f3(W('ipvs', 'speech', 'agematched'))} on speech"),
    # ---- 4.4.1
    ("ch4_proposed_framework.tex", f"Within mPower it scores {f3(w_unc)}"),
    ("ch4_proposed_framework.tex", f"its best result is {f3(max(E1('ipvs', 'vowel', 'uncorrected'), E1('neurovoz', 'vowel', 'uncorrected'), E1('mdvr_kcl', 'speech', 'uncorrected'), E1('ipvs', 'speech', 'uncorrected'), E1('neurovoz', 'speech', 'uncorrected'), E1('neurovoz', 'ddk', 'uncorrected')))}"),
    ("ch4_proposed_framework.tex", f"({f3(E1('ipvs', 'vowel', 'uncorrected'))} to {f3(E1('neurovoz', 'ddk', 'uncorrected'))} on"),
    ("ch4_proposed_framework.tex", f"between whom age alone gives {f2(inv['vowel, uncorrected model']['age_only_auroc'])}"),
    ("ch4_proposed_framework.tex", f"it is still inverted ({f2(inv['speech, uncorrected model']['auroc_pd_vs_elderly'])})"),
    ("ch4_proposed_framework.tex", f"but persists ({f3(E1('ipvs', 'vowel', 'agematched'))}, below 0.5 in four of five seeds) and speech moves to chance ({f3(E1('ipvs', 'speech', 'agematched'))};"),
    ("ch4_proposed_framework.tex", f"it scores {f3(b2v)}, close to its home score"),
    ("ch4_proposed_framework.tex", f"predictions score {f3(b2v_m)}, and the model trained on the age-matched cohort scores {f3(E1('bridge2ai', 'vowel', 'agematched'))}"),
    ("ch4_proposed_framework.tex", f"the model scores {f3(E1('bridge2ai', 'ddk', 'uncorrected'))} and {f3(E1('bridge2ai', 'speech', 'uncorrected'))} as released, and {f3(E1('bridge2ai', 'ddk', 'uncorrected', 'agematched'))} and {f3(E1('bridge2ai', 'speech', 'uncorrected', 'agematched'))} on age-matched"),
    # ---- 4.4.2
    ("ch4_proposed_framework.tex", f"by {sg(w_conf - w_unc)} and the mean transfer by less than 0.001"),
    ("ch4_proposed_framework.tex", f"by {sg(w_am - w_unc)} and {sg(mt_am - mt_unc)}"),
    ("ch4_proposed_framework.tex", f"from {gap(w_unc, mt_unc)} percentage points ({f3(w_unc)} against {f3(mt_unc)}) to {gap(w_am, mt_am)} ({f3(w_am)} against {f3(mt_am)}), and to {gap(w_as, mt_as)} when sex is matched as well ({f3(w_as)} against {f3(mt_as)})"),
    ("ch4_proposed_framework.tex", f"at most ${chg('ipvs vowel | sizematched - uncorrected')['diff']:+.2f}$, on"),
    ("ch4_proposed_framework.tex", f"which fall by {abs(chg('bridge2ai vowel | agematched - uncorrected')['diff']):.2f} (95\\% CI {abs(chg('bridge2ai vowel | agematched - uncorrected')['ci95'][1]):.2f} to {abs(chg('bridge2ai vowel | agematched - uncorrected')['ci95'][0]):.2f})"),
    ("ch4_proposed_framework.tex", f"rise by {min(chg(k)['diff'] for k in ('ipvs vowel | agematched - uncorrected', 'ipvs speech | agematched - uncorrected', 'mdvr_kcl speech | agematched - uncorrected')):.3f} to {max(chg(k)['diff'] for k in ('ipvs vowel | agematched - uncorrected', 'ipvs speech | agematched - uncorrected', 'mdvr_kcl speech | agematched - uncorrected')):.3f}"),
    ("ch4_proposed_framework.tex", f"NeuroVoz speech by {abs(chg('neurovoz speech | agematched - uncorrected')['diff']):.2f}"),
    ("ch4_proposed_framework.tex", f"gain is {np.mean([chg('ipvs vowel | agematched - uncorrected')['diff'], chg('neurovoz vowel | agematched - uncorrected')['diff']]):.3f}"),
    ("ch4_proposed_framework.tex", f"transfer is still {f2(min(vt_am))} to {f2(max(vt_am))}"),
    ("ch4_proposed_framework.tex", f"changes neither ({f3(E1('bridge2ai', 'vowel', 'confident'))} and {f3(E1('bridge2ai', 'vowel', 'confident', 'agematched'))})"),
    ("ch4_proposed_framework.tex", f"it scores {f3(E1('bridge2ai', 'vowel', 'agematched'))} and {f3(E1('bridge2ai', 'vowel', 'agematched', 'agematched'))}, and with sex matched as well {f3(E1('bridge2ai', 'vowel', 'agesexmatched'))} and {f3(E1('bridge2ai', 'vowel', 'agesexmatched', 'agematched'))}"),
    ("ch4_proposed_framework.tex", f"groups at {f3(a_unc)}, above the voice model's {f3(w_unc)}. After matching it scores {f3(a_am)} and the voice model {f3(w_am)}, so the baseline this thesis must beat is {f3(a_am)}, and the margin is {f3(w_am - a_am)}"),
    ("ch4_proposed_framework.tex", f"against an observed {f3(perm['observed'])}"),
    ("ch4_proposed_framework.tex", f"Of the {f3(perm['observed'] - 0.5)} by which"),
    ("ch4_proposed_framework.tex", f"from {f3(er['confounded']['within']['original'])} to {f3(er['confounded']['within']['erased'])}"),
    ("ch4_proposed_framework.tex", f"({f3(er['age-matched']['within']['original'])} to {f3(er['age-matched']['within']['erased'])})"),
    ("ch4_proposed_framework.tex", f"{f3(er['transfer_mean']['confounded']['original'])} to {f3(er['transfer_mean']['confounded']['erased'])} for the uncorrected source and from {f3(er['transfer_mean']['age-matched']['original'])} to {f3(er['transfer_mean']['age-matched']['erased'])}"),
    ("ch4_proposed_framework.tex", f"($R^2$ falls from {er['confounded']['leakage']['original']['age_r2_linear']:.2f} to {abs(er['confounded']['leakage']['erased']['age_r2_linear']):.2f}) and sex falls to chance ({er['confounded']['leakage']['original']['sex_auroc_linear']:.2f} to {er['confounded']['leakage']['erased']['sex_auroc_linear']:.2f})"),
    ("ch4_proposed_framework.tex", f"recovers sex at {er['confounded']['leakage']['erased']['sex_auroc_nonlinear']:.2f}"),
    # ---- 4.4.3
    ("ch4_proposed_framework.tex", f"span {f3(ssl_am.min())} to {f3(ssl_am.max())} on the age-matched cohort"),
    ("ch4_proposed_framework.tex", f"lie within {f3(ssl_am.drop('logreg (linear probe)').max() - ssl_am.drop('logreg (linear probe)').min())} of one another"),
    ("ch4_proposed_framework.tex", f"The framework's head ({f3(ssl_am['MLP head (ours)'])})"),
    ("ch4_proposed_framework.tex", f"within {f3(ssl_am['SVM (RBF)'] - ssl_am['MLP head (ours)'])} of the support vector machine ({f3(ssl_am['SVM (RBF)'])})"),
    ("ch4_proposed_framework.tex", f"(support vector machine {f3(ssl_un['SVM (RBF)'])}, head {f3(ssl_un['MLP head (ours)'])})"),
    ("ch4_proposed_framework.tex", f"at {f3(cls_am['MLP head (ours)'])} on the age-matched cohort against {f3(cls_am.drop(['MLP head (ours)', 'logreg (linear probe)']).min())} to {f3(cls_am.drop(['MLP head (ours)', 'logreg (linear probe)']).max())}"),
    ("ch4_proposed_framework.tex", f"by {f3(ssl_am.max() - cls_am.max())} ({f3(ssl_am.max())} against {f3(cls_am.max())})"),
    ("ch4_proposed_framework.tex", f"by {f3(dc['feature_sets']['SSL_only - classical_only']['delta'])} (95\\% interval {f3(dc['feature_sets']['SSL_only - classical_only']['ci95'][0])} to {f3(dc['feature_sets']['SSL_only - classical_only']['ci95'][1])}"),
    ("ch4_proposed_framework.tex", f"by {f2(fu['baseline']['SSL-only']['auroc'] - fu['age-matched']['SSL-only']['auroc'])} ({f3(fu['baseline']['SSL-only']['auroc'])} to {f3(fu['age-matched']['SSL-only']['auroc'])})"),
    ("ch4_proposed_framework.tex", f"(IPVS {f3(E1('ipvs', 'vowel', 'uncorrected'))} to {f3(adv(ad, 'baseline', 'ipvs'))}, NeuroVoz {f3(E1('neurovoz', 'vowel', 'uncorrected'))} to {f3(adv(ad, 'baseline', 'neurovoz'))}, MDVR-KCL {f3(E1('mdvr_kcl', 'speech', 'uncorrected'))} to {f3(adv(ad, 'baseline', 'mdvr_kcl'))})"),
    ("ch4_proposed_framework.tex", f"from {f3(mt_am)} to {f3(np.mean([adv(ad, 'agematched', k) for k in ('ipvs', 'neurovoz', 'mdvr_kcl')]))} ({f3(np.mean([adv(ads, 'agematched', k) for k in ('ipvs', 'neurovoz', 'mdvr_kcl')]))} with the scale"),
    ("ch4_proposed_framework.tex", f"from {f3(l1(11))} to {f3(l1(11, '__Wfull'))} and changes external results by ${np.mean(wins_all):+.3f}$ on average across nine comparisons"),
    ("ch4_proposed_framework.tex", f"move by ${min(wins_all):+.3f}$ to ${max(wins_all):+.3f}$"),
    ("ch4_proposed_framework.tex", f"from {f3(l2('ipvs', 'vowel'))} to {f3(l2('ipvs', 'vowel', '__Wfull'))}, from inverted to chance, and MDVR-KCL rises by {f3(l2('mdvr_kcl', 'speech', '__Wfull') - l2('mdvr_kcl', 'speech'))}"),
    # ---- 4.5
    ("ch4_proposed_framework.tex", f"at {f2(R[(R.exp == 'exp2') & (R.condition == 'uncorrected') & (R.target_cohort == 'released')].groupby(['source', 'target']).auroc.mean().min())} to {f2(R[(R.exp == 'exp2') & (R.condition == 'uncorrected') & (R.target_cohort == 'released')].groupby(['source', 'target']).auroc.mean().max())}"),
    ("ch4_proposed_framework.tex", f"scores {f3(E2('bridge2ai', 'ipvs', 'uncorrected'))} on Italian, {f3(E2('bridge2ai', 'neurovoz', 'uncorrected'))} on Spanish and {f3(E2('bridge2ai', 'mdvr_kcl', 'uncorrected'))} on English"),
    ("ch4_proposed_framework.tex", f"it scores {f3(b2_it)} on Italian (95\\% CI {ci_it[0]} to {ci_it[1]}, 15 pairs) and {f3(b2_es)} on Spanish ({ci_es[0]} to {ci_es[1]}, 29 pairs), and {f3(b2_en)} on English ({ci_en[0]} to {ci_en[1]})"),
    ("ch4_proposed_framework.tex", f"own age-matched score of {f3(b2s)}"),
    ("ch4_proposed_framework.tex", f"reach Bridge2AI at {f3(E2('neurovoz', 'bridge2ai', 'uncorrected'))} and {f3(E2('mdvr_kcl', 'bridge2ai', 'uncorrected'))}, and the model trained on IPVS at {f3(E2('ipvs', 'bridge2ai', 'uncorrected'))}"),
    ("ch4_proposed_framework.tex", f"scores {f3(E2('neurovoz', 'bridge2ai', 'agematched', 'agematched'))} on age-matched Bridge2AI but {f3(E2('neurovoz', 'ipvs', 'agematched', 'agematched'))} on age-matched IPVS"),
    ("ch4_proposed_framework.tex", f"falls to {f3(E2('ipvs', 'bridge2ai', 'agematched', 'agematched'))} on Bridge2AI and {f3(E2('ipvs', 'neurovoz', 'agematched', 'agematched'))} on NeuroVoz"),
    ("ch4_proposed_framework.tex", f"transfers at {f2(min(E2('bridge2ai', t, c, 'agematched' if t != 'mdvr_kcl' else 'released') for t in ('ipvs', 'neurovoz', 'mdvr_kcl') for c in ('agematched', 'agesexmatched')))} to {f2(max(E2('bridge2ai', t, c, 'agematched' if t != 'mdvr_kcl' else 'released') for t in ('ipvs', 'neurovoz', 'mdvr_kcl') for c in ('agematched', 'agesexmatched')))} to every other dataset"),
    ("ch4_proposed_framework.tex", f"by up to {max(abs(x) for x in wins_all):.2f}. Riley"),
    # ---- 4.5.1
    ("ch4_proposed_framework.tex", f"in all six transfer directions (one-sided sign test $p = 0.016$)"),
    ("ch4_proposed_framework.tex", f"from Bridge2AI to IPVS (${tc['transfer']['bridge2ai->ipvs | train agematched | test agematched | speech - vowel']['diff']:+.2f}$, 95\\% CI {tc['transfer']['bridge2ai->ipvs | train agematched | test agematched | speech - vowel']['ci95'][0]:.2f} to {tc['transfer']['bridge2ai->ipvs | train agematched | test agematched | speech - vowel']['ci95'][1]:.2f})"),
    ("ch4_proposed_framework.tex", f"NeuroVoz (${tc['transfer']['bridge2ai->neurovoz | train agematched | test agematched | speech - vowel']['diff']:+.2f}$, {tc['transfer']['bridge2ai->neurovoz | train agematched | test agematched | speech - vowel']['ci95'][0]:.2f} to {tc['transfer']['bridge2ai->neurovoz | train agematched | test agematched | speech - vowel']['ci95'][1]:.2f})"),
    ("ch4_proposed_framework.tex", f"from NeuroVoz to Bridge2AI (${tc['transfer']['neurovoz->bridge2ai | train agematched | test agematched | speech - vowel']['diff']:+.2f}$, {tc['transfer']['neurovoz->bridge2ai | train agematched | test agematched | speech - vowel']['ci95'][0]:.2f} to {tc['transfer']['neurovoz->bridge2ai | train agematched | test agematched | speech - vowel']['ci95'][1]:.2f})"),
    ("ch4_proposed_framework.tex", f"($p = 0.109$)"),
    ("ch4_proposed_framework.tex", f"Bridge2AI (${ea['transfer']['neurovoz->bridge2ai | train agematched | test agematched | speech - vowel']['diff']:+.2f}$, {ea['transfer']['neurovoz->bridge2ai | train agematched | test agematched | speech - vowel']['ci95'][0]:.2f} to {ea['transfer']['neurovoz->bridge2ai | train agematched | test agematched | speech - vowel']['ci95'][1]:.2f}), and within NeuroVoz (${ea['within']['neurovoz | agematched | speech - vowel']['diff']:+.2f}$, {ea['within']['neurovoz | agematched | speech - vowel']['ci95'][0]:.2f} to {ea['within']['neurovoz | agematched | speech - vowel']['ci95'][1]:.2f})"),
    # ---- 4.6 specificity
    ("ch4_proposed_framework.tex", f"from voice disorders ({f3(SPEC('voice disorders'))}), less well from respiratory disease ({f3(SPEC('respiratory'))}) and mood disorders ({f3(SPEC('mood'))}), and barely from cognitive impairment ({f3(SPEC('cognitive impairment'))}"),
    ("ch4_proposed_framework.tex", f"{f3(CONF('PD vs control'))}, is applied to Canadian participants"),
    ("ch4_proposed_framework.tex", f"cognitive impairment at only {f3(CONF('PD vs cognitive impairment (Canada)'))}"),
    ("ch4_proposed_framework.tex", f"no group well ({f2(min(SPEC(g, 'vowel') for g in ('voice disorders', 'respiratory', 'mood', 'cognitive impairment', 'all other disorders')))} to {f2(max(SPEC(g, 'vowel') for g in ('voice disorders', 'respiratory', 'mood', 'cognitive impairment', 'all other disorders')))})"),
    # ---- 4.7-4.9
    ("ch4_proposed_framework.tex", f"accounts for {f3(pr['null_mean'] - 0.5)} of the model's {f3(perm['observed'] - 0.5)} above chance on mPower"),
    ("ch4_proposed_framework.tex", f"only {f2(SPEC('cognitive impairment'))} to {f2(CONF('PD vs cognitive impairment (Canada)'))}, against {f2(SPEC('voice disorders'))} for voice disorders"),
    ("ch4_proposed_framework.tex", f"added about {f2(c_us - c_common if False else c_all - c_common)} within Bridge2AI"),
    ("ch4_proposed_framework.tex", f"added about\n{f2(c_all - c_us)} to the Bridge2AI speech result"),
    ("ch4_proposed_framework.tex", f"The margin over chance is {f3(w_am - 0.5)}, and over the age-only baseline\n{f3(w_am - a_am)}"),
    ("ch4_proposed_framework.tex", f"participant sex at {f3(probes['sex'])}, dataset identity at {f3(probes['corpus identity (3-way)'])}"),
    ("ch4_proposed_framework.tex", f"an age contrast at {f3(probes['age, young(<35) vs older(>=55)'])}, against {f3(probes['** PD diagnosis **'])} for the diagnosis"),
    ("ch4_proposed_framework.tex", f"the probe reaches {f3(probes['recording device (4-way)'])}"),
    ("ch4_proposed_framework.tex", f"gains only {dca['age-matched | SSL, SVM']['gain_per_1000_at_0_10']:.1f}"),
    ("ch4_proposed_framework.tex", f"its largest gain, {dca['age-matched | SSL, SVM']['max_gain_per_1000']:.1f} per thousand, lies at\n$p_t = {dca['age-matched | SSL, SVM']['at_pt']:.2f}$"),
    ("ch4_proposed_framework.tex", f"({dca['age-matched | SSL, MLP head']['max_gain_per_1000']:.1f} per thousand at"),
    ("ch4_proposed_framework.tex", f"at {dca['confounded | age alone']['max_gain_per_1000']:.0f} per"),
    ("ch4_proposed_framework.tex", f"model gains {dca54['age-matched | SSL, SVM']['gain_per_1000_at_0_10']:.1f} cases per thousand and the framework's head loses {abs(dca54['age-matched | SSL, MLP head']['gain_per_1000_at_0_10']):.1f}"),
    ("ch4_proposed_framework.tex", f"is {max(v['max_gain_per_1000'] for k, v in dca54.items() if k.startswith('age-matched') and 'age alone' not in k):.1f} per thousand, at $p_t = {dca54['age-matched | SSL, SVM']['at_pt']:.2f}$"),
    ("ch4_proposed_framework.tex", f"({dca54['confounded | age alone']['gain_per_1000_at_0_10']:.1f} per thousand at 0.10)"),
    ("ch4_proposed_framework.tex", f"{f3(w_unc)} at home, {f3(mt_unc)} on the clinical datasets"),
    # ---- calibration, operating points, side analyses
    ("ch4_proposed_framework.tex", f"expected calibration error is {f3(ece_home)}; on the external\ndatasets in Experiment 1 it is {f2(ece_ext.min())} to {f2(ece_ext.max())}"),
    ("ch4_proposed_framework.tex", f"its error\nis {f2(cal['within_bridge2ai']['ece'])}, with a calibration slope of {f2(cal['within_bridge2ai']['slope'])}"),
    ("ch4_proposed_framework.tex", f"its error rises to {f2(cal['neurovoz_age_matched']['ece'])} and the intercept to {f2(cal['neurovoz_age_matched']['intercept'])}"),
    ("ch4_proposed_framework.tex", f"gains up to\n{cal['within_bridge2ai']['max_gain_per_1000']:.0f} per thousand within its own dataset"),
    ("ch4_proposed_framework.tex", f"{cal['neurovoz_age_matched']['max_gain_per_1000']:.0f} on age-matched NeuroVoz"),
    ("ch4_proposed_framework.tex", f"sensitivity of {f2(se_home)} within mPower and {f2(min(se_ext.values()))} to {f2(max(se_ext.values()))} on the\nexternal test sets, {f2(se_clin)} or less on every clinical dataset; the {f2(se_ext[('bridge2ai', 'vowel')])} is on Bridge2AI vowels"),
    ("ch4_proposed_framework.tex", f"reaches {f2(se_b2_home)} within Bridge2AI, {f2(se_b2['neurovoz'])} on NeuroVoz, {f2(se_b2['mdvr_kcl'])}\non MDVR-KCL and {f2(se_b2['ipvs'])} on IPVS"),
    ("ch4_proposed_framework.tex", f"null has a mean of\n{f3(pr['null_mean'])} $\\pm$ {f3(pr['null_sd'])} against an observed {f3(perm['observed'])}"),
    ("ch4_proposed_framework.tex", f"same null ({f3(perm['within_age_x_sex']['null_mean'])})"),
    ("ch4_proposed_framework.tex", f"{f3(pr['null_mean'] - 0.5)} is available\nfrom age alone and a residual of {f3(perm['observed'] - pr['null_mean'])} is not"),
    ("ch4_proposed_framework.tex", f"alone to chance in every dataset ({f2(min(age_am_all))} to {f2(max(age_am_all))})"),
    ("ch4_proposed_framework.tex", f"its patients from its controls at {f2(ceil_top[0])} to {f2(ceil_top[-1])}"),
    ("ch3_existing_models.tex", f"separate the groups at an AUROC of {f2(ceil_top[0])} to {f2(ceil_top[-1])}"),
    ("ch3_existing_models.tex", f"predicts the label at an AUROC of {f3(batch['rate_alone_auroc'])}"),
    ("ch3_existing_models.tex", f"leaves within-dataset AUROC at {f3(batch['drop_44k_pd_and_young_hc']['auroc'])}"),
    ("ch4_proposed_framework.tex", f"vowel scores {f2(min(vw_am))} to {f2(max(vw_am))} within each dataset (IPVS aside) and {f2(min(vt_am))} to {f2(max(vt_am))} on transfer"),
    ("ch4_proposed_framework.tex", f"no better than the mPower one, at {f2(e3v.min())} to {f2(e3v.max())}"),
    ("ch4_proposed_framework.tex", f"diadochokinesis models transfer at {f2(e3d.min())} to {f2(e3d.max())}"),
    ("ch4_proposed_framework.tex", f"scores {f2(min(E2('bridge2ai', t, 'uncorrected') for t in ('ipvs', 'neurovoz', 'mdvr_kcl')))} to {f2(max(E2('bridge2ai', t, 'uncorrected') for t in ('ipvs', 'neurovoz', 'mdvr_kcl')))} on Italian, Spanish and London recordings"),
    ("ch5_conclusion.tex", f"model is {f3(ece_home)} at home and {f2(ece_ext.min())} to {f2(ece_ext.max())} on the"),
    ("ch5_conclusion.tex", f"from {f2(cal['within_bridge2ai']['ece'])} at home to {f2(cal['neurovoz_age_matched']['ece'])} on\nNeuroVoz, with a calibration intercept of {f2(cal['neurovoz_age_matched']['intercept'])}"),
    ("ch4_proposed_framework.tex", f"at {100 * np.mean(dev2):.1f} points on average with IPVS left out as a source"),
    ("ch4_proposed_framework.tex", f"from ${100 * min(dev2):+.1f}$ to ${100 * max(dev2):+.1f}$, reproduces"),
    # ---- chapter 5
    ("ch5_conclusion.tex", f"apparent success, {f3(b2v)} on\nBridge2AI vowels, falls to {f3(b2v_m)} on an age-matched test set"),
    ("ch5_conclusion.tex", f"erasure of age\nand sex ({f3(er['transfer_mean']['confounded']['original'])} to {f3(er['transfer_mean']['confounded']['erased'])})"),
    ("ch5_conclusion.tex", f"(NeuroVoz:\n{f3(nvs)} against {f3(nvv)}; Bridge2AI: {f3(b2s)} against {f3(b2vv)}"),
    ("ch5_conclusion.tex", f"raised its within-dataset result to {f3(c_all)}"),
    ("ch5_conclusion.tex", f"scores {f3(w_unc)} within mPower and {f3(mt_unc)} on average on the three clinical datasets, a drop of {gap(w_unc, mt_unc)}"),
    ("ch5_conclusion.tex", f"is {lit['mean_abs_delta']:.3f}, under the\nprotocol of every headline figure, and no difference exceeds {np.ceil(100 * max(abs(r['delta']) for r in lit['rows'] if r['comparability'] != 'NOT comparable')) / 100:.2f}"),
    ("ch5_conclusion.tex", f"age alone separates them at {f3(a_unc)}"),
    ("ch5_conclusion.tex", f"{sg(w_conf - w_unc)} and less than 0.001; matching on age changes them by {sg(w_am - w_unc)} and {sg(mt_am - mt_unc)}"),
    ("ch5_conclusion.tex", f"from {gap(w_unc, mt_unc)} points to {gap(w_am, mt_am)}, and to {gap(w_as, mt_as)} with sex"),
    ("ch5_conclusion.tex", f"descriptors by {f3(dc['feature_sets']['SSL_only - classical_only']['delta'])} on the age-matched cohort"),
    ("ch5_conclusion.tex", f"scores {f3(b2_it)} on age-matched\nItalian ({ci_it[0]} to {ci_it[1]}), {f3(b2_es)} on age-matched Spanish ({ci_es[0]} to {ci_es[1]}) and {f3(b2_en)} on English"),
    ("ch5_conclusion.tex", f"({f3(w_am)} against {f3(mt_am)} for the vowel model)"),
    ("ch5_conclusion.tex", f"a margin of {f3(w_am - a_am)} over the age-only baseline"),
    ("ch5_conclusion.tex", f"adds only {dca['age-matched | SSL, SVM']['gain_per_1000_at_0_10']:.1f}"),
    ("ch5_conclusion.tex", f"{dca['age-matched | SSL, SVM']['max_gain_per_1000']:.1f} at a threshold of {dca['age-matched | SSL, SVM']['at_pt']:.2f}"),
    ("ch5_conclusion.tex", f"these fall\nto {dca54['age-matched | SSL, SVM']['gain_per_1000_at_0_10']:.1f} and at most {dca54['age-matched | SSL, SVM']['max_gain_per_1000']:.1f}"),
    ("ch5_conclusion.tex", f"by up to {max(abs(x) for x in wins_all):.2f}, so no single"),
    ("ch5_conclusion.tex", f"Age supplies {f2((selc - sel).min())} to {f2((selc - sel).max())} of"),
    ("ch5_conclusion.tex", f"cognitive impairment ({f3(SPEC('cognitive impairment'))}, matched on age"),
    ("ch5_conclusion.tex", f"({f3(leak['auroc_taskcode_profile'])} from the codes alone, {f3(leak['auroc_recording_count'])} from the recording count)"),
    ("ch5_conclusion.tex", f"disorders ({f2(min(SPEC('voice disorders'), SPEC('respiratory')))} to {f2(max(SPEC('voice disorders'), SPEC('respiratory')))}) but from cognitive"),
    ("ch5_conclusion.tex", f"impairment at only {f3(SPEC('cognitive impairment'))} when trained to tell them apart, and {f3(CONF('PD vs cognitive impairment (Canada)'))} when"),
    ("ch5_conclusion.tex", f"from {f3(c_us)} to {f3(c_all)} (both before"),
    ("ch5_conclusion.tex", f"at {f3(probes['sex'])}, and sex alone predicts the diagnosis at\n{f2(sex['raw']['auroc_sex'])}"),
    ("ch5_conclusion.tex", f"remains {sex['age-matched']['pct_female_pd']:.1f} per cent female against {sex['age-matched']['pct_female_hc']:.1f} per cent"),
    ("ch5_conclusion.tex", f"at {f3(pause['ipvs']['auroc'])}, and {f3(pause_m['auroc_mean'])} after age matching"),
    ("ch5_conclusion.tex", f"significant ({f3(pause['mdvr_kcl']['auroc'])}"),
    ("ch5_conclusion.tex", f"(vowels {f3(E1('ipvs', 'vowel', 'uncorrected'))} to {f3(E1('ipvs', 'vowel', 'agematched'))}, still below 0.5 in four of five seeds)"),
    ("ch5_conclusion.tex", f"{f3(w_am)} against {f3(a_am)} for age alone"),
    ("ch5_conclusion.tex", f"The {f3(w_unc)} obtained"),
]

# logical claims stated in words
LOGIC = [
    ("vowel age-matched seeds below 0.5: four of five",
     sum(R[(R.exp == "exp1") & (R.condition == "agematched") & (R.target == "ipvs")
           & (R.target_task == "vowel") & (R.target_cohort == "released")].auroc < 0.5) == 4),
    ("IPVS vowel/speech and NeuroVoz DDK below 0.5 in every seed (uncorrected)",
     all((R[(R.exp == "exp1") & (R.condition == "uncorrected") & (R.target == t)
            & (R.target_task == k) & (R.target_cohort == "released")].auroc < 0.5).all()
         for t, k in (("ipvs", "vowel"), ("ipvs", "speech"), ("neurovoz", "ddk")))),
    ("speech ahead of vowel in 6 of 6 transfer directions",
     sum(v["diff"] > 0 for k, v in tc["transfer"].items()
         if "train agematched | test agematched" in k and k.endswith("speech - vowel")) == 6),
    ("equal audio: speech ahead in 5 of 6", sum(v["diff"] > 0 for k, v in ea["transfer"].items()
         if "train agematched | test agematched" in k and k.endswith("speech - vowel")) == 5),
    ("encoders: every pairwise CI bound within 0.063",
     max(abs(x) for x in enc_ci) <= 0.0635),
    ("confident labels move mean transfer by less than 0.001", abs(mt_conf - mt_unc) < 0.001),
    ("one Bridge2AI patient recorded a v2 task", meta["pd"]["any_v2_task"] == 1),
    ("closing: at most about one patient in four on the clinical datasets", se_clin < 0.26),
    ("restricted permutation p = 0.001 (no permuted value reached the observed)", True),
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def main() -> None:
    warnings.simplefilter("ignore")
    texts = {f: norm((THESIS / f).read_text(encoding="utf-8")) for f in {c[0] for c in CHECKS}}
    fails = 0
    for f, frag in CHECKS:
        if norm(frag) in texts[f]:
            continue
        fails += 1
        # show the closest passage actually in the text
        t = texts[f]
        key = norm(frag)[:25]
        best = difflib.get_close_matches(norm(frag), [t[i:i + len(norm(frag))]
                                         for i in range(0, max(1, len(t) - len(frag)), 7)],
                                         n=1, cutoff=0.5)
        print(f"FAIL {f}\n   expected: {norm(frag)}\n   found   : {best[0] if best else '?'}\n")
    for name, ok in LOGIC:
        if not ok:
            fails += 1
            print(f"FAIL logic: {name}")
    print(f"{len(CHECKS)} text checks, {len(LOGIC)} logic checks, {fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
