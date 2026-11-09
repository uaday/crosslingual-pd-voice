"""Generalised cross-dataset transfer: train on ANY dataset+task, test on others.

HISTORICAL. Not the route behind the thesis numbers: those come from
scripts/unified_experiments.py (see REPRODUCE.md). Kept for exploration; it now applies
the same cohort rules (shared mPower cohort, binary labels, common speech tasks) and
refuses embedding caches that cannot be matched to the manifest by recording path.

Dataset-agnostic version of layer2_transfer.py. Trains the
frozen-XLS-R + head on a chosen source (subject-disjoint train/val, locked calibration),
reports the within-source 5-fold ceiling, then applies the locked model to a list of
target (dataset:task) pairs. Embeddings are cached per (dataset, task, layer); mPower
reuses the cap-3 Layer-1 cache.

Examples:
  # reverse transfer, VOWEL: train NeuroVoz -> test mPower + IPVS (all vowel)
  python scripts/transfer_any.py --train neurovoz:vowel --tests mpower:vowel,ipvs:vowel
  # SPEECH: train NeuroVoz speech -> test IPVS + MDVR-KCL speech
  python scripts/transfer_any.py --train neurovoz:speech --tests ipvs:speech,mdvr_kcl:speech
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from pdvoice.calibration import select_best_by_brier
from pdvoice.config import EMBEDDINGS, LOGS, SEED, VAL_FRACTION, ensure_dirs
from pdvoice.data import LOADERS
from pdvoice.cohort import load_mpower_cohort
from pdvoice.data.common_tasks import common_task_mask
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
from pdvoice.eval.metrics import calibration_error, sensitivity_at_specificity
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control
from pdvoice.data.splits import ensure_two_class_val

MODEL, CAP = "xls-r-300m", 3


def embed(df, layer, cache_path):
    """Cached embeddings, row-aligned to ``df`` by recording PATH.

    Caches carry the path of every row. On load, the rows are matched to the manifest by
    path (and reordered if the order differs); a cache that lacks paths, or lacks any of
    the manifest's recordings, is refused. Speaker ids alone, which this function used to
    check, cannot tell two recordings by the same speaker apart.
    """
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        if "path" not in d.files:
            raise SystemExit(
                f"{cache_path.name} has no recording paths, so its rows cannot be matched "
                "to the manifest. Delete it and re-extract.")
        row = {q: i for i, q in enumerate(d["path"].astype(str))}
        missing = [q for q in df["path"].astype(str) if q not in row]
        if missing:
            raise SystemExit(
                f"{cache_path.name}: {len(missing)} manifest recordings are not in the cache "
                f"(e.g. {missing[0]}). Delete it and re-extract.")
        return d["X"][[row[q] for q in df["path"].astype(str)]].astype(np.float32)
    from tqdm import tqdm

    from pdvoice.features.ssl import DEFAULT_MODELS, _embed_wave, load_encoder
    from pdvoice.preprocess import chunks, preprocess_clip

    model, extractor, device = load_encoder(DEFAULT_MODELS.get(MODEL, MODEL))
    print(f"      extracting {len(df):,} recs on {device} -> {cache_path.name}")
    vecs = []
    for row in tqdm(df.itertuples(index=False), total=len(df), unit="rec"):
        wav = preprocess_clip(row.path, row.task_kind)
        if row.task_kind == "vowel":
            v = _embed_wave(wav, model, extractor, device, layer, "mean_std")
        else:
            v = np.mean([_embed_wave(c, model, extractor, device, layer, "mean_std")
                         for c in chunks(wav)], axis=0)
        vecs.append(v.astype(np.float16))
    X = np.vstack(vecs).astype(np.float16)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X,
                        subject_id=df["subject_id"].to_numpy().astype(str),
                        path=df["path"].to_numpy().astype(str))
    return X.astype(np.float32)


def get_set(ds, task, layer, all_groups=False):
    """Manifest and embeddings for one dataset and task.

    Binary evaluation (the default) keeps PD and control rows only, and for connected
    speech only the tasks both groups recorded (common_task_mask). ``all_groups=True``
    is the separate extraction mode that also keeps Bridge2AI's other-disorder rows.
    """
    if ds == "mpower":  # the shared cohort: labelled, QC before the cap, with age
        df = load_mpower_cohort(cap=CAP, seed=SEED)
        df = df[df["task_kind"] == task].reset_index(drop=True)
        cache = EMBEDDINGS / f"mpower_cohort__{MODEL}__{task}__cap{CAP}__L{layer}__seed{SEED}.npz"
    else:
        df = LOADERS[ds]()
        df = df[df["task_kind"] == task].copy()
        df, _ = quality_control(df); df = df.reset_index(drop=True)
        cache = EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{layer}.npz"
    X = embed(df, layer, cache)
    if all_groups:
        return df, X
    keep = df["label"].isin([0, 1]).to_numpy() & common_task_mask(df, ds)
    df = df[keep].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df, X[keep]


def subj_metrics(df, pr, thr):
    """pr = (calibrated probability, ranking score from calibration.RankingMap). Discrimination uses the
    ranking score (calibration.RankingMap); calibration measures use the probability."""
    p, r = pr
    s = pd.DataFrame({"s": df["subject_id"], "y": df["label"].astype(int), "p": p, "r": r})
    g = s.groupby("s").agg(y=("y", "first"), p=("p", "mean"), r=("r", "mean"))
    y, p, r = g["y"].to_numpy(), g["p"].to_numpy(), g["r"].to_numpy()
    auc, alo, ahi = bootstrap_ci(y, r, metric=auroc)
    return {"n": int(len(y)), "n_pd": int((y == 1).sum()), "n_hc": int((y == 0).sum()),
            "auroc": auc, "auroc_ci": [alo, ahi], "auprc": auprc(y, r),
            "brier": float(brier_score_loss(y, p)), "ece": calibration_error(y, p)["ece"],
            "sens_at_90spec": sensitivity_at_specificity(y, r, 0.90)["sensitivity"]}


def within_cv(df, X, y):
    df = assign_folds(df.reset_index(drop=True), n_splits=5)
    oof = []
    for k in range(5):
        tr, va, te = train_val_test_indices(df, test_fold=k)
        m, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va])
        with torch.no_grad():
            dev = next(m.parameters()).device
            tl = m(torch.tensor(X[te], dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        _, cal = select_best_by_brier(vlog, y[va])
        oof.append(pd.DataFrame({"subject_id": df.loc[te, "subject_id"].to_numpy(),
                                 "label": y[te], "p": cal.predict_proba(tl),
                                 "r": cal.ranking(tl)}))
    o = pd.concat(oof)
    return subj_metrics(o, (o["p"].to_numpy(), o["r"].to_numpy()), 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="dataset:task, e.g. neurovoz:vowel")
    ap.add_argument("--tests", required=True, help="comma list of dataset:task")
    ap.add_argument("--layer", type=int, default=15)
    args = ap.parse_args()
    tr_ds, tr_task = args.train.split(":")
    tests = [t.split(":") for t in args.tests.split(",") if t]

    ensure_dirs()
    print(f"=== transfer | TRAIN {args.train} | L{args.layer} ===")
    print(f"[train] {tr_ds}:{tr_task} ...")
    tdf, Xtr = get_set(tr_ds, tr_task, args.layer)
    ytr = tdf["label"].astype(int).to_numpy()
    print(f"  {tdf.subject_id.nunique()} subj (PD {tdf[tdf.label==1].subject_id.nunique()}, "
          f"HC {tdf[tdf.label==0].subject_id.nunique()}), {len(tdf)} recs")

    print("[within] 5-fold CV ceiling ...")
    within = within_cv(tdf, Xtr, ytr)

    print("[lock] train head + calibration ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr, va = next(gss.split(Xtr, ytr, tdf["subject_id"]))
    tr, va = ensure_two_class_val(tr, va, ytr, tdf["subject_id"].to_numpy(), SEED)
    head, vlog = fit_mlp_head(Xtr[tr], ytr[tr], Xtr[va], ytr[va])
    _, cal = select_best_by_brier(vlog, ytr[va])
    dev = next(head.parameters()).device

    def score(X):
        with torch.no_grad():
            lg = head(torch.tensor(X, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        return cal.predict_proba(lg), cal.ranking(lg)

    results = {"train": args.train, "layer": args.layer, "within": within, "transfer": {}}
    for ds, task in tests:
        print(f"[test] {ds}:{task} ...")
        edf, EX = get_set(ds, task, args.layer)
        results["transfer"][f"{ds}:{task}"] = subj_metrics(edf, score(EX), 0.5) | \
            {"language": str(edf["language"].iloc[0])}

    stem = f"transferany__{MODEL}__train-{tr_ds}-{tr_task}__L{args.layer}"
    (LOGS / f"{stem}.json").write_text(json.dumps(results, indent=2))

    print(f"\n===================== TRANSFER (train={args.train}) =====================")
    w = within
    print(f"  WITHIN {tr_ds}:{tr_task} (5-fold): AUROC {w['auroc']:.3f} "
          f"[{w['auroc_ci'][0]:.3f},{w['auroc_ci'][1]:.3f}]  AUPRC {w['auprc']:.3f}  ECE {w['ece']:.3f}")
    print(f"  {'TEST':<16}{'lang':<6}{'subj':<7}{'AUROC':<20}{'AUPRC':<8}{'ECE'}")
    for k, m in results["transfer"].items():
        ci = f"[{m['auroc_ci'][0]:.3f},{m['auroc_ci'][1]:.3f}]"
        print(f"  {k:<16}{m['language']:<6}{m['n']:<7}{m['auroc']:.3f} {ci:<14}{m['auprc']:.3f}   {m['ece']:.3f}")
    print("=========================================================================")


if __name__ == "__main__":
    main()
