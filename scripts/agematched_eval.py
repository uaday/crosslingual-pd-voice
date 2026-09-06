"""Age-matched honest model — full evaluation (Layer 1 + Layer 2).

Motivation (see scripts/label_sensitivity.py): mPower's headline AUROC is inflated by
a PD/HC age confound (PD ~60, HC ~33). Here we frequency-match the PD and HC AGE
distributions per 5-year bin (a proper confound control, not a blunt cutoff), then run
the whole pipeline on the confound-controlled cohort:

    Layer 1  : 5-fold subject-disjoint CV within age-matched mPower  (within-dataset)
    Layer 2  : train head once on age-matched mPower, lock calibration+threshold,
               apply UNCHANGED to IPVS / NeuroVoz / MDVR-KCL         (transfer)

All at the transfer-optimal layer (XLS-R L15). Reports AUROC/AUPRC (+CI), calibration
(Brier/ECE/MCE), DCA, and threshold metrics; saves metrics JSON, subject preds, a
figure, and the locked Layer-2 head+calibrator to outputs/models/.

Run:  python scripts/agematched_eval.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from pdvoice.cohort import age_match_ids
from pdvoice.calibration import select_best_by_brier
from pdvoice.config import EMBEDDINGS, FIGURES, LOGS, MODELS, MPOWER_DIR, SEED, VAL_FRACTION, ensure_dirs
from pdvoice.data import LOADERS
from pdvoice.data.mpower import load_mpower
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.dca import net_benefit
from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
from pdvoice.eval.metrics import (calibration_error, sensitivity_at_specificity,
                                  threshold_metrics, youden_threshold)
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

MODEL, LAYER, CAP = "xls-r-300m", 15, 3
EXTERNALS = [("ipvs", "vowel"), ("neurovoz", "vowel"), ("mdvr_kcl", "speech")]


def load_cache(path, n):
    X = np.load(path)["X"]
    assert X.shape[0] == n, f"{path.name}: {X.shape[0]} != {n} (misaligned)"
    return X.astype(np.float32)


def mpower_df():
    df = load_mpower()
    lab = df[df["label"].isin([0, 1])].sample(frac=1.0, random_state=SEED)
    df = lab[lab.groupby("subject_id").cumcount().to_numpy() < CAP].reset_index(drop=True)
    df, _ = quality_control(df)
    df = df.reset_index(drop=True)
    d = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    age = dict(zip(d["healthCode"].astype(str), d["age"]))
    df["age"] = df["subject_id"].str.replace("mpower_", "", regex=False).map(age)
    return df


def metrics_block(y, p, thr):
    auc, alo, ahi = bootstrap_ci(y, p, metric=auroc)
    ap, plo, phi = bootstrap_ci(y, p, metric=auprc)
    ce = calibration_error(y, p)
    return {
        "n": int(len(y)), "n_pd": int((y == 1).sum()), "n_hc": int((y == 0).sum()),
        "auroc": auc, "auroc_ci": [alo, ahi], "auprc": ap, "auprc_ci": [plo, phi],
        "brier": float(brier_score_loss(y, p)), "ece": ce["ece"], "mce": ce["mce"],
        "at_threshold": threshold_metrics(y, p, thr),
        "sensitivity_at_90_spec": sensitivity_at_specificity(y, p, 0.90),
    }


def within_cv(df, X, y):
    """5-fold subject-disjoint CV -> pooled out-of-fold subject-level (y, p)."""
    df = assign_folds(df.reset_index(drop=True), n_splits=5)
    oof = []
    for k in range(5):
        tr, va, te = train_val_test_indices(df, test_fold=k)
        m, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va])
        with torch.no_grad():
            dev = next(m.parameters()).device
            tl = m(torch.tensor(X[te], dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        _, cal = select_best_by_brier(vlog, y[va])
        oof.append(pd.DataFrame({"s": df.loc[te, "subject_id"].to_numpy(),
                                 "y": y[te], "p": cal.predict_proba(tl)}))
    s = pd.concat(oof).groupby("s").agg(y=("y", "first"), p=("p", "mean")).reset_index()
    return s


def train_locked(df, X, y):
    """Train head once, lock calibrator+threshold on a subject-disjoint val slice."""
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr, va = next(gss.split(X, y, df["subject_id"]))
    m, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va])
    cal_name, cal = select_best_by_brier(vlog, y[va])
    thr = youden_threshold(y[va], cal.predict_proba(vlog))
    return m, cal, cal_name, thr


def make_figure(within_s, transfer, thr, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # ROC: within + each external
    for lab, s in [("mPower (within)", within_s)] + [(d, t["subj"]) for d, t in transfer.items()]:
        fpr, tpr, _ = roc_curve(s["y"], s["p"])
        ax[0].plot(fpr, tpr, label=f"{lab} ({auroc(s['y'], s['p']):.2f})")
    ax[0].plot([0, 1], [0, 1], "--", c="gray")
    ax[0].set(title="ROC (age-matched)", xlabel="1 - specificity", ylabel="sensitivity"); ax[0].legend(fontsize=8)
    # Reliability (within)
    p, y = within_s["p"].to_numpy(), within_s["y"].to_numpy(); o = np.argsort(p)
    bx = [p[o][b].mean() for b in np.array_split(np.arange(len(p)), 10) if len(b)]
    by = [y[o][b].mean() for b in np.array_split(np.arange(len(p)), 10) if len(b)]
    ax[1].plot([0, 1], [0, 1], "--", c="gray"); ax[1].plot(bx, by, "o-")
    ax[1].set(title="Reliability (within)", xlabel="predicted", ylabel="observed")
    # DCA (within)
    dca = net_benefit(within_s["y"].to_numpy(), within_s["p"].to_numpy())
    ax[2].plot(dca["threshold"], dca["model"], label="model")
    ax[2].plot(dca["threshold"], dca["treat_all"], "--", label="treat all")
    ax[2].plot(dca["threshold"], dca["treat_none"], ":", c="k", label="treat none")
    ax[2].set(title="Decision curve (within)", xlabel="threshold", ylabel="net benefit")
    ax[2].legend(fontsize=8); ax[2].set_ylim(bottom=min(0, dca["model"].min()))
    fig.tight_layout(); fig.savefig(outpath, dpi=120); plt.close(fig)


def main():
    ensure_dirs()
    print("=== Age-matched honest model | XLS-R L15 ===")
    df = mpower_df()
    keep = age_match_ids(df)
    am = df[df["subject_id"].isin(keep)].reset_index(drop=True)
    idx = df.index[df["subject_id"].isin(keep)].to_numpy()
    y = am["label"].astype(int).to_numpy()
    print(f"  full mPower: {df.subject_id.nunique()} subj (PD age {df[df.label==1].age.mean():.1f}, "
          f"HC age {df[df.label==0].age.mean():.1f})")
    print(f"  age-matched: {am.subject_id.nunique()} subj "
          f"(PD {am[am.label==1].subject_id.nunique()} age {am[am.label==1].age.mean():.1f}, "
          f"HC {am[am.label==0].subject_id.nunique()} age {am[am.label==0].age.mean():.1f})")

    X = load_cache(EMBEDDINGS / f"mpower_pilot__{MODEL}__cap{CAP}__L{LAYER}__seed{SEED}.npz", len(df))[idx]

    # ---- Layer 1: within-dataset CV ----
    print("[L1] within-dataset 5-fold CV ...")
    within_s = within_cv(am, X, y)
    thr_cv = youden_threshold(within_s["y"], within_s["p"])
    m_within = metrics_block(within_s["y"].to_numpy(), within_s["p"].to_numpy(), thr_cv)

    # ---- Layer 2: locked transfer ----
    print("[L2] train locked head; transfer to externals ...")
    head, cal, cal_name, thr = train_locked(am, X, y)
    dev = next(head.parameters()).device

    def score(Xe):
        with torch.no_grad():
            lg = head(torch.tensor(Xe, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        return cal.predict_proba(lg)

    transfer = {}
    for ds, task in EXTERNALS:
        edf = LOADERS[ds]()
        edf = edf[edf["task_kind"] == task].copy()
        edf, _ = quality_control(edf); edf = edf.reset_index(drop=True)
        EX = load_cache(EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{LAYER}.npz", len(edf))
        s = pd.DataFrame({"subject_id": edf["subject_id"], "y": edf["label"].astype(int), "p": score(EX)})
        sg = s.groupby("subject_id").agg(y=("y", "first"), p=("p", "mean")).reset_index()
        mm = metrics_block(sg["y"].to_numpy(), sg["p"].to_numpy(), thr)
        transfer[ds] = {"metrics": mm, "subj": sg, "language": str(edf["language"].iloc[0])}

    # ---- save artifacts ----
    stem = f"agematched__{MODEL}__L{LAYER}"
    out = {"within_mpower": m_within, "locked_threshold": thr, "calibrator": cal_name,
           "transfer": {d: t["metrics"] | {"language": t["language"]} for d, t in transfer.items()}}
    (LOGS / f"{stem}.json").write_text(json.dumps(out, indent=2))
    within_s.to_csv(LOGS / f"{stem}__within_subject_preds.csv", index=False)
    for d, t in transfer.items():
        t["subj"].to_csv(LOGS / f"{stem}__{d}_subject_preds.csv", index=False)
    make_figure(within_s, transfer, thr, FIGURES / f"{stem}.png")
    torch.save(head.state_dict(), MODELS / f"{stem}__head.pt")
    import joblib
    joblib.dump({"calibrator": cal, "threshold": thr, "layer": LAYER, "model": MODEL,
                 "in_dim": X.shape[1]}, MODELS / f"{stem}__calibrator.joblib")

    # ---- summary ----
    print("\n===================== AGE-MATCHED HONEST MODEL =====================")
    w = m_within
    print(f"  WITHIN mPower (5-fold): AUROC {w['auroc']:.3f} [{w['auroc_ci'][0]:.3f},{w['auroc_ci'][1]:.3f}]  "
          f"AUPRC {w['auprc']:.3f}  ECE {w['ece']:.3f}  sens@90spec {w['sensitivity_at_90_spec']['sensitivity']:.3f}")
    print(f"  {'TRANSFER':<12}{'lang':<6}{'subj':<7}{'AUROC':<20}{'AUPRC':<8}{'ECE'}")
    for d, t in transfer.items():
        mm = t["metrics"]; ci = f"[{mm['auroc_ci'][0]:.3f},{mm['auroc_ci'][1]:.3f}]"
        print(f"  {d:<12}{t['language']:<6}{mm['n']:<7}{mm['auroc']:.3f} {ci:<14}{mm['auprc']:.3f}   {mm['ece']:.3f}")
    print("====================================================================")
    print(f"  artifacts: {LOGS/(stem+'.json')} | model: {MODELS/(stem+'__head.pt')}")


if __name__ == "__main__":
    main()
