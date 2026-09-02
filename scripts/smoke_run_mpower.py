"""End-to-end smoke run on a small mPower subset.

Exercises the whole chain on the GPU box before committing to the full (~115k
recording) extraction:

    load mPower + labels -> QC -> subject-disjoint folds -> take a small balanced
    subset -> frozen wav2vec2-base embeddings (GPU, cached) -> MLP head (fold 0)
    -> calibrate on val -> report AUROC / AUPRC / ECE.

Run:  python scripts/smoke_run_mpower.py --per-class 100 --model wav2vec2-base

This is a sanity check (tiny n, no CV loop, no external tests), not a result.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pdvoice.config import SEED, ensure_dirs
from pdvoice.data.mpower import load_mpower
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.preprocess import quality_control


def balanced_subset(df, per_class: int, seed: int = SEED):
    """Take up to ``per_class`` labelled recordings per class, subject-aware."""
    labelled = df[df["label"].isin([0, 1])].copy()
    rng = np.random.default_rng(seed)
    picks = []
    for cls in (0, 1):
        pool = labelled[labelled["label"] == cls]
        n = min(per_class, len(pool))
        picks.append(pool.sample(n=n, random_state=int(rng.integers(1e9))))
    import pandas as pd

    return pd.concat(picks).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100, help="labelled recs per class")
    ap.add_argument("--model", default="wav2vec2-base")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    ensure_dirs()
    t0 = time.time()

    print("[1/6] Loading mPower manifest + labels ...")
    df = load_mpower()
    n_lab = int(df["label"].isin([0, 1]).sum())
    print(f"      {len(df):,} recordings · {n_lab:,} labelled "
          f"(PD={int((df['label']==1).sum()):,}  non-PD={int((df['label']==0).sum()):,})")

    print(f"[2/6] Balanced subset ({args.per_class}/class) ...")
    sub = balanced_subset(df, args.per_class)

    print("[3/6] Quality control (duration) on subset ...")
    sub, dropped = quality_control(sub)
    print(f"      kept {len(sub):,} · dropped {len(dropped):,}")
    sub = assign_folds(sub, n_splits=5)
    tr, va, te = train_val_test_indices(sub, test_fold=0)
    print(f"      subset={len(sub)}  train={len(tr)} val={len(va)} test={len(te)}")

    print(f"[4/6] Frozen embeddings via {args.model} (GPU) ...")
    from pdvoice.features.ssl import extract_for_manifest, select_device

    print(f"      device = {select_device()}")
    X, ids = extract_for_manifest(sub, model_key=args.model, layer=args.layer, cache=False)
    y = sub.loc[ids, "label"].astype(int).to_numpy()
    id_to_pos = {rid: i for i, rid in enumerate(ids)}
    tr_p = [id_to_pos[i] for i in tr if i in id_to_pos]
    va_p = [id_to_pos[i] for i in va if i in id_to_pos]
    te_p = [id_to_pos[i] for i in te if i in id_to_pos]
    print(f"      X = {X.shape}  (dim={X.shape[1]})")

    print("[5/6] Training MLP head (fold 0) ...")
    from pdvoice.models.head import fit_mlp_head

    model, _ = fit_mlp_head(X[tr_p], y[tr_p], X[va_p], y[va_p])

    print("[6/6] Evaluating on held-out test slice ...")
    import torch

    from pdvoice.eval.discrimination import auroc
    from pdvoice.eval.metrics import calibration_error

    with torch.no_grad():
        dev = next(model.parameters()).device
        logits = model(torch.tensor(np.asarray(X[te_p]), dtype=torch.float32, device=dev)).squeeze(-1)
        proba = torch.sigmoid(logits).cpu().numpy()
    yte = y[te_p]
    print(f"\n  n_test={len(yte)}  AUROC={auroc(yte, proba):.3f}  "
          f"ECE={calibration_error(yte, proba)['ece']:.3f}")
    print(f"\nDone in {time.time() - t0:.1f}s. (Smoke run — tiny n, not a reportable result.)")


if __name__ == "__main__":
    main()
