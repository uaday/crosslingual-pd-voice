"""What are we actually optimising?  (supervisor ask, 2026-09-05)

    "It would be good to have a kind of objective function... whether it is accepted
     for us to have more false positives than false negatives... because it is more
     important for us to identify the Parkinson patients, even if sometimes we go with
     the [false alarm]. A justification should be provided."

Until now every model in this project was selected by AUROC. AUROC is threshold-free:
it says how well a model RANKS people, and says nothing about which mistakes we are
willing to make. Two models with the same AUROC can behave completely differently at
the operating point a clinic would actually use.

THE OBJECTIVE, STATED
---------------------
This is a SCREENING tool, not a diagnostic one. A positive result sends someone to a
neurologist; a negative one sends them home. So the two errors are not symmetric:

    false negative  a patient is told they are fine and leaves the pathway
    false positive  a healthy person attends one unnecessary appointment

We therefore accept many false positives to avoid one false negative. Decision Curve
Analysis makes that trade explicit through a single number, the threshold probability
`pt`: acting when P(PD) >= pt means you are indifferent between acting and not acting
at pt, which is exactly the statement

    (harm of one false positive) / (harm of one false negative) = pt / (1 - pt)

so at pt you would accept (1 - pt)/pt false positives to catch one true case:

    pt = 0.20  ->  4 FP per TP        conservative referral
    pt = 0.10  ->  9 FP per TP        our default for screening
    pt = 0.05  -> 19 FP per TP        aggressive, catch nearly everyone

We adopt **pt = 0.10** and state it rather than leaving it implicit. Net benefit is then

    NB = TP/n - (FP/n) * pt/(1 - pt)

measured in "true cases found per person screened, after paying for the false alarms",
and directly comparable against screening everyone and screening no one.

WHAT THIS SCRIPT DOES
---------------------
Scores every feature set x classifier at that objective instead of at AUROC, reports the
confusion matrix each choice implies, and checks the question that matters: does
selecting on the objective pick a DIFFERENT model than selecting on AUROC?

Run:  .venv\\Scripts\\python.exe -u scripts\\objective_function.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from pdvoice.config import LOGS
from pdvoice.eval.dca import net_benefit
from pdvoice.eval.metrics import threshold_metrics

PT_DEFAULT = 0.10


def operating_point(y, p, pt):
    """The threshold that maximises net benefit at `pt`, and what it costs."""
    w = pt / (1 - pt)
    n = len(y)
    best = None
    for thr in np.unique(np.round(p, 4)):
        pred = p >= thr
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        nb = tp / n - fp / n * w
        if best is None or nb > best["net_benefit"]:
            best = {"threshold": float(thr), "net_benefit": float(nb)}
    m = threshold_metrics(y, p, best["threshold"])
    fp_per_tp = m["fp"] / m["tp"] if m["tp"] else np.inf
    return {**best,
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "ppv": round(m["ppv"], 4),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "fp_per_tp": round(float(fp_per_tp), 2)}


def net_benefit_at_prevalence(sens, spec, pt, prevalence):
    """Net benefit rewritten in terms of sensitivity, specificity and a STATED prevalence.

    NB = prevalence*sens - (1 - prevalence)*(1 - spec)*pt/(1 - pt)

    This matters here and the plain version is misleading. Net benefit depends on how
    common the disease is, and our age-matched cohort is deliberately 50/50 -- an
    artefact of the confound control, not a real screening population. At 50%
    prevalence and pt=0.10, "screen everyone" is already near-optimal, so every model
    looks worthless no matter how well it discriminates.

    The 1:1 design is correct for AUROC (it removes the confound) and wrong for utility
    (it fabricates prevalence). So discrimination is measured on the matched cohort and
    utility is re-expressed at a prevalence someone would actually screen at.
    """
    w = pt / (1 - pt)
    return prevalence * sens - (1 - prevalence) * (1 - spec) * w


def sens_at_spec_floor(y, p, floor=0.70):
    """Alternative objective: highest sensitivity while specificity stays >= floor."""
    best = {"sensitivity": 0.0, "specificity": 1.0, "threshold": 1.0}
    for thr in np.unique(np.round(p, 4)):
        m = threshold_metrics(y, p, float(thr))
        if m["specificity"] >= floor and m["sensitivity"] > best["sensitivity"]:
            best = {"sensitivity": round(m["sensitivity"], 4),
                    "specificity": round(m["specificity"], 4), "threshold": float(thr)}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=float, default=PT_DEFAULT,
                    help="threshold probability; 0.10 means 9 false positives per true case")
    ap.add_argument("--spec-floor", type=float, default=0.70)
    ap.add_argument("--target-prevalence", type=float, default=None,
                    help="prevalence to express utility at; default = the CONFOUNDED "
                         "cohort's natural rate, since the age-matched cohort is 50/50 "
                         "by construction and would fabricate it")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    src = LOGS / "classifier_predictions.csv"
    if not src.exists():
        raise SystemExit(
            f"missing {src}\nRun scripts/classifier_comparison.py --save-preds first.")
    P = pd.read_csv(src)

    # natural prevalence = the unmatched cohort. The age-matched cohort is 1:1 by
    # design, which is right for discrimination and wrong for utility.
    nat = P[P.cohort == "confounded"].drop_duplicates("subject_id")
    target_prev = args.target_prevalence or float(nat.y.mean())

    w = args.pt / (1 - args.pt)
    print(f"=== Objective: maximise net benefit at pt = {args.pt} ===")
    print(f"    utility re-expressed at prevalence = {target_prev:.3f} "
          f"(the unmatched cohort's natural rate)")
    print(f"    pt = {args.pt}  =>  willing to accept {1/w:.0f} false positives per true case")
    print(f"    NB = TP/n - (FP/n) x {w:.3f}\n")

    rows = []
    for (cohort, feats, clf), g in P.groupby(["cohort", "features", "classifier"]):
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        op = operating_point(y, p, args.pt)
        sf = sens_at_spec_floor(y, p, args.spec_floor)
        nb = net_benefit(y, p, thresholds=[args.pt]).iloc[0]
        from pdvoice.eval.discrimination import auroc
        prev = float(np.mean(y))
        nb_adj = net_benefit_at_prevalence(op["sensitivity"], op["specificity"],
                                          args.pt, target_prev)
        nb_all_adj = net_benefit_at_prevalence(1.0, 0.0, args.pt, target_prev)
        rows.append({
            "cohort": cohort, "features": feats, "classifier": clf,
            "cohort_prevalence": round(prev, 4),
            "NB_at_target_prev": round(nb_adj, 5),
            "NB_all_at_target_prev": round(nb_all_adj, 5),
            "beats_screen_all": bool(nb_adj > nb_all_adj),
            "AUROC": round(auroc(y, p), 4),
            "net_benefit": round(op["net_benefit"], 5),
            "NB_treat_all": round(float(nb["treat_all"]), 5),
            "sens": op["sensitivity"], "spec": op["specificity"], "ppv": op["ppv"],
            "TP": op["tp"], "FP": op["fp"], "FN": op["fn"], "TN": op["tn"],
            "FP_per_TP": op["fp_per_tp"],
            f"sens@spec>={args.spec_floor}": sf["sensitivity"],
        })
    R = pd.DataFrame(rows)
    R.to_csv(LOGS / "objective_function.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    for cohort in R.cohort.unique():
        sub = R[R.cohort == cohort].sort_values("net_benefit", ascending=False)
        print(f"{'='*110}\n{cohort.upper()}  ·  ranked by NET BENEFIT at pt={args.pt}\n{'='*110}")
        print(sub.drop(columns=["cohort"]).to_string(index=False))
        print()

    print(f"{'='*110}\nDOES THE OBJECTIVE PICK A DIFFERENT MODEL THAN AUROC?\n{'='*110}")
    for cohort in R.cohort.unique():
        sub = R[R.cohort == cohort]
        by_auroc = sub.loc[sub.AUROC.idxmax()]
        by_nb = sub.loc[sub.NB_at_target_prev.idxmax()]
        same = (by_auroc.features, by_auroc.classifier) == (by_nb.features, by_nb.classifier)
        print(f"  {cohort}:")
        print(f"    best by AUROC       : {by_auroc.features} + {by_auroc.classifier}"
              f"   (AUROC {by_auroc.AUROC:.3f}, NB {by_auroc.net_benefit:+.5f})")
        print(f"    best by NET BENEFIT : {by_nb.features} + {by_nb.classifier}"
              f"   (AUROC {by_nb.AUROC:.3f}, NB@prev {by_nb.NB_at_target_prev:+.5f})")
        print(f"    -> {'SAME model' if same else 'DIFFERENT model -- selecting on AUROC would have chosen wrong'}")
        print(f"    -> at prevalence {target_prev:.3f}, screening everyone scores "
              f"{by_nb.NB_all_at_target_prev:+.5f}")
        print(f"    -> beats screening everyone: "
              f"{'YES' if by_nb.beats_screen_all else 'NO -- not clinically useful at this pt'}")
        gain = by_nb.NB_at_target_prev - by_nb.NB_all_at_target_prev
        print(f"    -> net gain {gain:+.5f} true cases per person screened "
              f"= {gain * 1000:+.1f} per 1,000 screened\n")

    (LOGS / "objective_function.json").write_text(json.dumps(
        {"pt": args.pt, "fp_per_tp_accepted": round(1 / w, 2),
         "spec_floor": args.spec_floor, "rows": rows}, indent=2))
    print(f"wrote {LOGS / 'objective_function.csv'}")


if __name__ == "__main__":
    main()
