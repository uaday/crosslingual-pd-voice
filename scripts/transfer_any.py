"""Generalised cross-dataset transfer: train on ANY dataset+task, test on others.

Dataset-agnostic version of layer2_transfer.py (PROJECT_CONTEXT.md §3.4). Trains the
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
from pdvoice.data.mpower import load_mpower
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
from pdvoice.eval.metrics import calibration_error, sensitivity_at_specificity
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

MODEL, CAP = "xls-r-300m", 3


def embed(df, layer, cache_path):
    """Cached embeddings, row-aligned to ``df``.

    The cache stores the speaker ids alongside X and CHECKS them on load. It used to
    store X alone and rely on the caller rebuilding the manifest in exactly the same
    order (filter task, then apply QC, in that order). Row count alone does not catch a
    reordering, so a stale or differently-built manifest silently paired recordings with
    other people's embeddings -- which raises nothing and simply returns a plausible
    wrong number. Older caches have no ids; those fall back to the length check with a
    warning rather than being silently trusted.
    """
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        if d["X"].shape[0] == len(df):
            if "subject_id" in d.files:
                cached = d["subject_id"].astype(str)
                if not np.array_equal(cached, df["subject_id"].to_numpy().astype(str)):
                    raise SystemExit(
                        f"{cache_path.name}: speaker ids do not match the manifest.\n"
                        "The cohort was built differently from when this cache was written. "
                        "Delete the cache and re-extract rather than trusting it.")
            else:
                print(f"      NOTE {cache_path.name} predates id-checking; "
                      "verified on row count only")
            return d["X"].astype(np.float32)
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


def get_set(ds, task, layer):
    if ds == "mpower":
        df = load_mpower()
        lab = df[df["label"].isin([0, 1])].sample(frac=1.0, random_state=SEED)
        df = lab[lab.groupby("subject_id").cumcount().to_numpy() < CAP].reset_index(drop=True)
        df, _ = quality_control(df); df = df.reset_index(drop=True)
        cache = EMBEDDINGS / f"mpower_pilot__{MODEL}__cap{CAP}__L{layer}__seed{SEED}.npz"
    else:
        df = LOADERS[ds]()
        df = df[df["task_kind"] == task].copy()
        df, _ = quality_control(df); df = df.reset_index(drop=True)
        cache = EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{layer}.npz"
    return df, embed(df, layer, cache)


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
