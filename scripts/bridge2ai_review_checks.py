"""Bridge2AI review checks A1, A4 and A5, at the layer chosen inside Bridge2AI.

A5  Uncertainty. The +/- reported so far is the spread across matching seeds. Here each
    headline figure gets a participant-level bootstrap (1,000 resamples per seed), pooled
    across the five seeds, so the interval reflects both the participants and the draw.
    Rows: within Bridge2AI after age matching (vowel, DDK, speech; linear probe), and the
    locked speech model (trained on age-matched Bridge2AI) on IPVS without young controls,
    age-matched NeuroVoz and MDVR-KCL.

A1  Specificity with direct age control. The PD-vs-control speech model (age-matched,
    out-of-fold) scores participants with cognitive impairment. Before, the comparison
    "cognitive impairment vs controls" used the controls matched to PD (mean ~69) against a
    cognitive-impairment group of mean 74.8. Here cognitive impairment is frequency-matched
    on age to the healthy controls directly (5-year bins), as is PD to cognitive impairment,
    and the age-only AUROC of each comparison is reported to show the age control worked.

A4  Calibration and clinical utility for the speech model: within Bridge2AI (5-fold CV of
    the framework's own MLP head with locked calibration, age-matched cohort) and the locked
    model on age-matched NeuroVoz. ECE, Brier, calibration slope and intercept (logistic
    recalibration), and the decision curve. Probabilities are prior-shifted to a prevalence
    of 0.187 (as in the thesis's mPower decision curves) before the decision curve.

Layer: PDV_LAYER (default 8, chosen by bridge2ai_layer_select.py).
Run:  .venv\\Scripts\\python.exe -u scripts\\bridge2ai_review_checks.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PDV_LAYER", "8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from bridge2ai_agematch import subject_scores  # noqa: E402
from bridge2ai_followups import family_ids, load  # noqa: E402
from bridge2ai_transfer import SUFFIX, LAYER  # noqa: E402
from bridge2ai_transfer_agecheck import subsets  # noqa: E402
from demographic_erasure import load_external  # noqa: E402
from unified_experiments import _ensure_two_class_val  # noqa: E402

from pdvoice.calibration import select_best_by_brier  # noqa: E402
from pdvoice.cohort import age_match_ids  # noqa: E402
from pdvoice.config import LOGS, VAL_FRACTION  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402
from pdvoice.eval.dca import net_benefit  # noqa: E402
from pdvoice.eval.metrics import calibration_error  # noqa: E402
from pdvoice.models.head import fit_mlp_head  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 1000
PREV = 0.187
THRESHOLDS = tuple(i / 100 for i in range(5, 51))


def boot_aucs(y, s, seed):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = []
    for _ in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        out.append(roc_auc_score(y[i], s[i]))
    return out


def summarise(points, boots):
    b = np.concatenate(boots)
    return {"auroc": round(float(np.mean(points)), 3),
            "ci95": [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)],
            "seed_sd": round(float(np.std(points, ddof=1)), 3)}


def mlp_logits(head, X):
    dev = next(head.parameters()).device
    with torch.no_grad():
        return head(torch.tensor(X, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()


def locked(X, y, g, seed):
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed).split(X, y, g))
    tr, va = _ensure_two_class_val(tr, va, np.asarray(y), np.asarray(g), seed)
    head, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va], seed=seed)
    _, cal = select_best_by_brier(vlog, y[va])
    return head, cal


def by_subject(sid, y, *cols):
    d = pd.DataFrame({"s": sid, "y": y, **{f"c{i}": c for i, c in enumerate(cols)}})
    return d.groupby("s").agg(y=("y", "first"), **{f"c{i}": (f"c{i}", "mean") for i in range(len(cols))})


def prior_shift(p, prev_from, prev_to):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    odds = p / (1 - p) * (prev_to / (1 - prev_to)) / (prev_from / (1 - prev_from))
    return odds / (1 + odds)


def calib(y, p):
    # isotonic outputs exact 0 and 1; clip so the logistic recalibration stays finite
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    lp = np.log(p / (1 - p))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    intercept = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    ce = calibration_error(y, p)
    return {"ece": round(float(ce["ece"]), 3), "brier": round(float(brier_score_loss(y, p)), 3),
            "slope": round(float(slope), 2), "intercept": round(float(intercept), 2)}


def dca(y, p):
    """Decision curve at prevalence PREV: probabilities prior-shifted, net benefit reweighted."""
    y = np.asarray(y)
    ps = prior_shift(p, y.mean(), PREV)
    rows = []
    for pt in THRESHOLDS:
        w = pt / (1 - pt)
        pred = ps >= pt
        sens = (pred & (y == 1)).sum() / (y == 1).sum()
        fpr = (pred & (y == 0)).sum() / (y == 0).sum()
        rows.append({"threshold": pt, "model": sens * PREV - fpr * (1 - PREV) * w,
                     "treat_all": PREV - (1 - PREV) * w})
    d = pd.DataFrame(rows)
    d["gain_per_1000"] = 1000 * (d.model - np.maximum(d.treat_all, 0))
    best = d.loc[d.gain_per_1000.idxmax()]
    pos = d[d.gain_per_1000 > 0].threshold
    return {"max_gain_per_1000": round(float(best.gain_per_1000), 1),
            "at_threshold": round(float(best.threshold), 2),
            "positive_range": [round(float(pos.min()), 2), round(float(pos.max()), 2)] if len(pos) else None}


def main() -> None:
    warnings.simplefilter("ignore")
    print(f"layer {LAYER}", flush=True)
    base = LOADERS["bridge2ai"]()
    report = {"layer": LAYER, "A5_within": {}, "A5_transfer": {}, "A1": {}, "A4": {}}

    # ---- A5 within Bridge2AI (linear probe, age-matched)
    for task in ("vowel", "ddk", "speech"):
        d, X = load(task, base, ["pd", "control"])
        y = d.label.astype(int).to_numpy(); d["label"] = y
        pts, bs = [], []
        for seed in SEEDS:
            m = d.subject_id.isin(age_match_ids(d, seed=seed)).to_numpy()
            s = subject_scores(d[m].reset_index(drop=True), X[m], y[m], seed)
            pts.append(roc_auc_score(s.y, s.p)); bs.append(boot_aucs(s.y, s.p, seed))
        report["A5_within"][task] = summarise(pts, bs)
        print("A5 within", task, report["A5_within"][task], flush=True)

    # ---- A5 transfer + A4 external (locked MLP head on age-matched Bridge2AI speech)
    sp, Xs = load("speech", base, ["pd", "control"])
    ys = sp.label.astype(int).to_numpy(); sp["label"] = ys
    targets = [("ipvs", "speech", "no young controls"), ("neurovoz", "speech", "age-matched"),
               ("mdvr_kcl", "speech", "as published")]
    ext = {(ds, t): load_external(ds, t) for ds, t, _ in targets}
    pts, bs, ext_pred = {}, {}, {}
    for seed in SEEDS:
        m = sp.subject_id.isin(age_match_ids(sp, seed=seed)).to_numpy()
        head, cal = locked(Xs[m], ys[m], sp.subject_id.to_numpy()[m], seed)
        for ds, t, sub in targets:
            edf, E = ext[(ds, t)]
            mask = subsets(ds, edf, seed)[sub]
            lg = mlp_logits(head, E)[mask]
            g = by_subject(edf.subject_id[mask], edf.label[mask].astype(int), cal.ranking(lg), cal.predict_proba(lg))
            pts.setdefault(ds, []).append(roc_auc_score(g.y, g.c0))
            bs.setdefault(ds, []).append(boot_aucs(g.y, g.c0, seed))
            if ds == "neurovoz":
                ext_pred.setdefault("y", []).append(g.y.to_numpy()); ext_pred.setdefault("p", []).append(g.c1.to_numpy())
    for ds, t, sub in targets:
        report["A5_transfer"][f"{ds} ({sub})"] = summarise(pts[ds], bs[ds])
        print("A5 transfer", ds, report["A5_transfer"][f"{ds} ({sub})"], flush=True)

    # ---- A4 within: 5-fold CV of the framework head, locked calibration per fold
    cal_rows, dca_rows = [], []
    for seed in SEEDS:
        m = sp.subject_id.isin(age_match_ids(sp, seed=seed)).to_numpy()
        d, X, y = sp[m].reset_index(drop=True), Xs[m], ys[m]
        g = d.subject_id.to_numpy()
        p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            head, cal = locked(X[tr], y[tr], g[tr], seed)
            p[te] = cal.predict_proba(mlp_logits(head, X[te]))
        s = by_subject(g, y, p)
        cal_rows.append(calib(s.y.to_numpy(), s.c0.to_numpy()))
        dca_rows.append(dca(s.y.to_numpy(), s.c0.to_numpy()))
    ext_cal = [calib(y, p) for y, p in zip(ext_pred["y"], ext_pred["p"])]
    ext_dca = [dca(y, p) for y, p in zip(ext_pred["y"], ext_pred["p"])]

    def mean_rows(rows):
        keys = [k for k in rows[0] if not isinstance(rows[0][k], list) and rows[0][k] is not None]
        # median across seeds: one seed's recalibration can be unstable on a small cohort
        return {k: round(float(np.median([r[k] for r in rows])), 3) for k in keys}
    report["A4"] = {"prevalence_for_dca": PREV,
                    "within_bridge2ai": {**mean_rows(cal_rows), **mean_rows(dca_rows),
                                         "per_seed_dca": dca_rows},
                    "neurovoz_age_matched": {**mean_rows(ext_cal), **mean_rows(ext_dca),
                                             "per_seed_dca": ext_dca}}
    print("A4", json.dumps({k: v for k, v in report["A4"].items()}, default=str)[:900], flush=True)

    # ---- A1 specificity with direct age matching
    d, X = load("speech", base, ["pd", "control", "other"])
    ci_ids = family_ids()["neurological (cognitive impairment, ALS)"]
    d["grp"] = np.where(d.group == "pd", "pd", np.where(d.group == "control", "control",
                        np.where(d.subject_id.isin(ci_ids), "ci", "other")))
    res = {"ci vs control": [], "pd vs ci": [], "pd vs control": []}
    ages = {"ci vs control": [], "pd vs ci": [], "pd vs control": []}
    npairs = {"ci vs control": [], "pd vs ci": [], "pd vs control": []}
    for seed in SEEDS:
        pc = d[d.grp.isin(["pd", "control"])]
        keep = age_match_ids(pc.assign(label=(pc.grp == "pd").astype(int)), seed=seed)
        tr_mask = d.subject_id.isin(keep).to_numpy()
        T, XT = d[tr_mask].reset_index(drop=True), X[tr_mask]
        yT, gT = (T.grp == "pd").astype(int).to_numpy(), T.subject_id.to_numpy()
        score = pd.Series(0.0, index=np.arange(len(d)))
        oof = np.zeros(len(T)); rest = np.zeros((~tr_mask).sum())
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(XT, yT, gT):
            k = min(64, len(np.unique(gT[tr])) - 1)
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            clf = make_pipeline(StandardScaler(), PCA(k, random_state=seed),
                                LogisticRegression(max_iter=2000)).fit(XT[tr], yT[tr])
            oof[te] = clf.predict_proba(XT[te])[:, 1]
            rest += clf.predict_proba(X[~tr_mask])[:, 1] / 5
        score[np.flatnonzero(tr_mask)] = oof
        score[np.flatnonzero(~tr_mask)] = rest
        S = pd.DataFrame({"s": d.subject_id, "grp": d.grp, "age": d.age, "p": score.to_numpy()}) \
              .groupby("s").agg(grp=("grp", "first"), age=("age", "first"), p=("p", "mean")).reset_index()
        for name, pos, neg in (("ci vs control", "ci", "control"), ("pd vs ci", "pd", "ci"),
                               ("pd vs control", "pd", "control")):
            sub = S[S.grp.isin([pos, neg])].assign(subject_id=lambda x: x.s, label=lambda x: (x.grp == pos).astype(int))
            mk = sub.subject_id.isin(age_match_ids(sub, seed=seed))
            sub = sub[mk]
            res[name].append(roc_auc_score(sub.label, sub.p))
            ages[name].append(roc_auc_score(sub.label, sub.age))
            npairs[name].append(int(sub.label.sum()))
    for name in res:
        report["A1"][name] = {"pairs": int(np.median(npairs[name])),
                              "auroc": round(float(np.mean(res[name])), 3),
                              "seed_sd": round(float(np.std(res[name], ddof=1)), 3),
                              "age_only": round(float(np.mean(ages[name])), 3)}
        print("A1", name, report["A1"][name], flush=True)

    (LOGS / f"bridge2ai_review_checks{SUFFIX}.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote bridge2ai_review_checks{SUFFIX}.json")


if __name__ == "__main__":
    main()
