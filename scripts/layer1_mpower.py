"""Layer 1 (RQ1) — within-dataset mPower baseline, capped pilot.

Pipeline (PROJECT_CONTEXT.md §4.5, Layer 1):
    load mPower + labels -> cap recordings/subject -> QC -> 5-fold subject-disjoint
    CV -> frozen SSL embedding (cached) -> MLP head -> calibrate on val (LOCK) ->
    out-of-fold predictions -> aggregate to SUBJECT level -> AUROC/AUPRC (+95% CI),
    calibration (ECE/MCE/Brier), Decision Curve Analysis + figures.

Why capped: mPower recording counts are wildly imbalanced per subject (PD ~40 recs,
HC ~6). Capping recordings/subject keeps a few prolific PD patients from dominating
and makes a first real result affordable. Evaluation is at the subject level (the
clinically meaningful unit); every subject sits in exactly one test fold, so subject
aggregation of out-of-fold predictions is leakage-free.

Run:  python scripts/layer1_mpower.py --cap 3 --model wav2vec2-base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pdvoice.config import EMBEDDINGS, FIGURES, LOGS, SEED, ensure_dirs
from pdvoice.data.mpower import load_mpower
from pdvoice.data.splits import assign_folds, assert_disjoint, train_val_test_indices
from pdvoice.preprocess import quality_control


# ----------------------------------------------------------------------------- data
def cap_per_subject(df: pd.DataFrame, cap: int, seed: int = SEED) -> pd.DataFrame:
    """Keep up to `cap` labelled recordings per subject (seeded, subject-aware).

    Shuffle-then-take-head-per-subject via cumcount — avoids groupby.apply, whose
    grouping-column handling changed in pandas 3.0.
    """
    lab = df[df["label"].isin([0, 1])].sample(frac=1.0, random_state=seed)
    rank = lab.groupby("subject_id").cumcount()
    return lab[rank.to_numpy() < cap].reset_index(drop=True)


# ------------------------------------------------------------------------ embeddings
def extract_embeddings(df: pd.DataFrame, model_key: str, layer, seed: int, cap: int):
    """Embed every row of df (order-aligned). Cached to a subset-specific npz."""
    from pdvoice.features.ssl import DEFAULT_MODELS, _embed_wave, load_encoder
    from pdvoice.preprocess import chunks, preprocess_clip

    cache = EMBEDDINGS / f"mpower_pilot__{model_key}__cap{cap}__L{layer}__seed{seed}.npz"
    if cache.exists():
        d = np.load(cache)
        if d["X"].shape[0] == len(df):
            print(f"      cache hit: {cache.name}  X={d['X'].shape}")
            return d["X"]
        print("      cache size mismatch -> re-extracting")

    from tqdm import tqdm

    model_id = DEFAULT_MODELS.get(model_key, model_key)
    model, extractor, device = load_encoder(model_id)
    print(f"      encoder {model_key} on {device}; embedding {len(df):,} recordings")

    vecs = []
    t0 = time.time()
    for row in tqdm(df.itertuples(index=False), total=len(df), unit="rec"):
        wav = preprocess_clip(row.path, row.task_kind)
        if row.task_kind == "vowel":
            v = _embed_wave(wav, model, extractor, device, layer, "mean_std")
        else:
            v = np.mean([_embed_wave(c, model, extractor, device, layer, "mean_std")
                         for c in chunks(wav)], axis=0)
        vecs.append(v.astype(np.float16))
    X = np.vstack(vecs).astype(np.float16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X)
    print(f"      done in {(time.time() - t0)/60:.1f} min -> cached {cache.name}")
    return X


# --------------------------------------------------------------------------------- CV
def run_cv(df: pd.DataFrame, X: np.ndarray, n_folds: int):
    """5-fold subject-disjoint CV. Returns per-recording OOF frame (calibrated p)."""
    from pdvoice.calibration import select_best_by_brier
    from pdvoice.models.head import fit_mlp_head

    import torch

    df = assign_folds(df, n_splits=n_folds)
    oof = []
    chosen_cals = []
    for k in range(n_folds):
        tr, va, te = train_val_test_indices(df, test_fold=k)
        assert_disjoint(df, tr, va, te)
        ytr, yva, yte = (df.loc[i, "label"].astype(int).to_numpy() for i in (tr, va, te))

        model, val_logits = fit_mlp_head(X[tr], ytr, X[va], yva)
        with torch.no_grad():
            dev = next(model.parameters()).device
            te_logits = model(torch.tensor(np.asarray(X[te]), dtype=torch.float32,
                                           device=dev)).squeeze(-1).cpu().numpy()

        name, cal = select_best_by_brier(val_logits, yva)     # LOCK on val
        chosen_cals.append(name)
        te_p = cal.predict_proba(te_logits)

        oof.append(pd.DataFrame({
            "subject_id": df.loc[te, "subject_id"].to_numpy(),
            "y": yte, "p": te_p, "fold": k,
        }))
        print(f"      fold {k}: train={len(tr)} val={len(va)} test={len(te)} "
              f"calibrator={name}")
    print(f"      calibrators chosen across folds: {chosen_cals}")
    return pd.concat(oof, ignore_index=True)


# ------------------------------------------------------------------------- evaluation
def evaluate_subject_level(oof: pd.DataFrame, outstem: str, model_key: str) -> dict:
    from pdvoice.eval.dca import net_benefit
    from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
    from pdvoice.eval.metrics import (
        calibration_error, sensitivity_at_specificity, threshold_metrics, youden_threshold,
    )
    from sklearn.metrics import brier_score_loss

    subj = oof.groupby("subject_id").agg(y=("y", "first"), p=("p", "mean"),
                                         n_rec=("p", "size")).reset_index()
    subj.to_csv(LOGS / f"{outstem}__subject_preds.csv", index=False)
    y, p = subj["y"].to_numpy(), subj["p"].to_numpy()

    auc, auc_lo, auc_hi = bootstrap_ci(y, p, metric=auroc)
    ap, ap_lo, ap_hi = bootstrap_ci(y, p, metric=auprc)
    thr = youden_threshold(y, p)   # NOTE: on OOF -> mildly optimistic; pilot only
    tm = threshold_metrics(y, p, thr)
    sat90 = sensitivity_at_specificity(y, p, 0.90)
    cal = calibration_error(y, p)

    metrics = {
        "model": model_key,
        "n_subjects": int(len(subj)),
        "n_pd": int((y == 1).sum()), "n_hc": int((y == 0).sum()),
        "auroc": auc, "auroc_ci": [auc_lo, auc_hi],
        "auprc": ap, "auprc_ci": [ap_lo, ap_hi],
        "brier": float(brier_score_loss(y, p)),
        "ece": cal["ece"], "mce": cal["mce"],
        "operating_point_youden": tm,
        "sensitivity_at_90_spec": sat90,
    }

    _make_figures(y, p, net_benefit(y, p), outstem, model_key)
    return metrics


def _make_figures(y, p, dca_df, outstem, model_key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="gray")
    ax[0].set(title=f"ROC ({model_key})", xlabel="1 - specificity", ylabel="sensitivity")

    order = np.argsort(p)
    bins = np.array_split(np.argsort(p), 10)
    bx = [p[np.argsort(p)][b].mean() if len(b) else np.nan for b in np.array_split(np.arange(len(p)), 10)]
    by = [y[np.argsort(p)][b].mean() if len(b) else np.nan for b in np.array_split(np.arange(len(p)), 10)]
    ax[1].plot([0, 1], [0, 1], "--", c="gray"); ax[1].plot(bx, by, "o-")
    ax[1].set(title="Reliability", xlabel="mean predicted", ylabel="observed freq")

    ax[2].plot(dca_df["threshold"], dca_df["model"], label="model")
    ax[2].plot(dca_df["threshold"], dca_df["treat_all"], "--", label="treat all")
    ax[2].plot(dca_df["threshold"], dca_df["treat_none"], ":", label="treat none")
    ax[2].set(title="Decision Curve", xlabel="threshold prob", ylabel="net benefit")
    ax[2].legend(); ax[2].set_ylim(bottom=min(0, dca_df["model"].min()))

    fig.tight_layout()
    out = FIGURES / f"{outstem}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"      figures -> {out}")


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3, help="max labelled recordings/subject")
    ap.add_argument("--model", default="wav2vec2-base")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="debug: keep only this many subjects per class (fast dry-run)")
    args = ap.parse_args()

    ensure_dirs()
    t0 = time.time()
    print(f"=== Layer 1 mPower pilot | model={args.model} cap={args.cap} ===")

    print("[1/5] Load mPower + labels ...")
    df = load_mpower()

    if args.max_subjects:
        keep = (df[df.label.isin([0, 1])].groupby("label")["subject_id"]
                .apply(lambda s: s.drop_duplicates().head(args.max_subjects)).tolist())
        df = df[df["subject_id"].isin(set(keep))].copy()
        print(f"      [debug] limited to {df['subject_id'].nunique()} subjects")

    print(f"[2/5] Cap {args.cap} rec/subject + QC ...")
    df = cap_per_subject(df, args.cap)
    df, dropped = quality_control(df)
    df = df.reset_index(drop=True)
    print(f"      {len(df):,} recordings across {df['subject_id'].nunique():,} subjects "
          f"(PD subj={df[df.label==1].subject_id.nunique():,}  "
          f"HC subj={df[df.label==0].subject_id.nunique():,}; dropped {len(dropped)})")

    print("[3/5] Extract frozen embeddings ...")
    X = extract_embeddings(df, args.model, args.layer, SEED, args.cap)

    print(f"[4/5] {args.folds}-fold subject-disjoint CV + calibration ...")
    oof = run_cv(df, X, args.folds)
    outstem = f"layer1_mpower__{args.model}__cap{args.cap}__L{args.layer}"
    oof.to_csv(LOGS / f"{outstem}__oof_preds.csv", index=False)

    print("[5/5] Subject-level evaluation ...")
    metrics = evaluate_subject_level(oof, outstem, args.model)
    metrics["layer"] = args.layer

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{outstem}.json").write_text(json.dumps(metrics, indent=2))

    print("\n===================== RESULT (subject level) =====================")
    print(f"  model={args.model}  layer={args.layer}")
    print(f"  subjects={metrics['n_subjects']} (PD={metrics['n_pd']} HC={metrics['n_hc']})")
    print(f"  AUROC = {metrics['auroc']:.3f}  95% CI [{metrics['auroc_ci'][0]:.3f}, {metrics['auroc_ci'][1]:.3f}]")
    print(f"  AUPRC = {metrics['auprc']:.3f}  95% CI [{metrics['auprc_ci'][0]:.3f}, {metrics['auprc_ci'][1]:.3f}]")
    print(f"  Brier = {metrics['brier']:.3f}  ECE = {metrics['ece']:.3f}  MCE = {metrics['mce']:.3f}")
    op = metrics["operating_point_youden"]
    print(f"  @Youden: sens={op['sensitivity']:.3f} spec={op['specificity']:.3f} "
          f"bal-acc={op['balanced_accuracy']:.3f}")
    s90 = metrics["sensitivity_at_90_spec"]
    print(f"  sens@90%spec = {s90['sensitivity']:.3f}")
    print(f"  metrics -> {LOGS / (outstem + '.json')}")
    print(f"==================================================================")
    print(f"Total {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
