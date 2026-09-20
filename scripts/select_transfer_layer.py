"""Choose the encoder layer for transfer WITHOUT looking at any external dataset.

THE PROBLEM THIS FIXES. Layer 15 became the transfer layer because it scored best on
NeuroVoz, IPVS and MDVR-KCL -- the very datasets the transfer results are reported on.
Selecting a setting on the test data biases every number produced with it, and the
project's own protocol says settings are fixed before an evaluation dataset is opened.
That sentence is not currently true.

THE FIX. Score every layer by subject-disjoint cross-validation on mPower alone, using
the same head the transfer pipeline uses, and take the winner. No external data is
loaded by this script at all -- that is the whole point, and it is why the script
deliberately has no code path that could read one.

The cache is one recording per speaker on the age-matched cohort, so the selection is
also made on the honest cohort rather than the confounded one.

Run:  .venv\\Scripts\\python.exe -u scripts\\select_transfer_layer.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from pdvoice.config import EMBEDDINGS, LOGS
from pdvoice.eval.discrimination import auroc, bootstrap_ci, paired_bootstrap_auroc
from pdvoice.models.head import fit_mlp_head

CACHE = "layersweep__xls-r-300m__agematched.npz"
CURRENT_LAYER = 15        # what the pipeline uses today, for comparison


def cv_auroc(X, y, groups, seed, n_splits=5):
    """Subject-disjoint out-of-fold AUROC for one layer."""
    oof = np.zeros(len(y), dtype=float)
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True,
                                       random_state=seed).split(X, y, groups):
        # an inner split for early stopping, still subject-disjoint
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
        itr, iva = next(inner.split(X[tr], y[tr], groups[tr]))
        model, _ = fit_mlp_head(X[tr][itr], y[tr][itr], X[tr][iva], y[tr][iva], seed=seed)
        with torch.no_grad():
            dev = next(model.parameters()).device
            lg = model(torch.tensor(X[te], dtype=torch.float32, device=dev)).squeeze(-1)
            oof[te] = torch.sigmoid(lg).cpu().numpy()
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    f = EMBEDDINGS / CACHE
    if not f.exists():
        raise SystemExit(f"missing {f}\nRun: scripts/extract_all_layers.py --model xls-r-300m")

    d = np.load(f, allow_pickle=True)
    X_all = d["X"].astype(np.float32)          # (recordings, layers, dim)
    y = d["label"].astype(int)
    groups = d["subject_id"].astype(str)
    n_layers = X_all.shape[1]

    print("=== Selecting the transfer layer on mPower ONLY ===")
    print(f"  cache   {CACHE}")
    print(f"  cohort  {len(y):,} recordings / {len(set(groups)):,} speakers "
          f"(PD {int((y == 1).sum())} / HC {int((y == 0).sum())}), age-matched")
    print(f"  layers  {n_layers}, scored by subject-disjoint 5-fold CV")
    print("  NO external dataset is read by this script.\n")

    rows, oofs = [], {}
    for layer in range(n_layers):
        oof = cv_auroc(X_all[:, layer, :], y, groups, args.seed)
        g = (pd.DataFrame({"s": groups, "y": y, "p": oof})
               .groupby("s").agg(y=("y", "first"), p=("p", "mean")))
        a, lo, hi = bootstrap_ci(g.y.to_numpy(), g.p.to_numpy())
        rows.append(dict(layer=layer, auroc=round(a, 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
        oofs[layer] = (g.y.to_numpy(), g.p.to_numpy())
        print(f"  layer {layer:>2}  AUROC {a:.4f}  [{lo:.4f}, {hi:.4f}]", flush=True)

    R = pd.DataFrame(rows)
    best = int(R.loc[R.auroc.idxmax(), "layer"])
    best_auroc = float(R.auroc.max())
    cur_auroc = float(R.loc[R.layer == CURRENT_LAYER, "auroc"].iloc[0])

    print(f"\n{'=' * 78}\nRESULT\n{'=' * 78}")
    print(f"  best layer by CV on mPower : {best}  (AUROC {best_auroc:.4f})")
    print(f"  layer the pipeline uses    : {CURRENT_LAYER}  (AUROC {cur_auroc:.4f})")

    # Is the difference real, or is the ranking just noise? Same speakers, paired.
    ya, pa = oofs[best]
    yb, pb = oofs[CURRENT_LAYER]
    assert np.array_equal(ya, yb), "layer folds disagree on labels"
    delta, p_paired = paired_bootstrap_auroc(ya, pa, pb)
    print(f"  difference                 : {delta:+.4f}  paired bootstrap p = {p_paired:.3f}")

    if best == CURRENT_LAYER:
        verdict = ("The layer already in use is also the one training-data CV picks. "
                   "The transfer results stand as they are, and the selection can now "
                   "be described as made without reference to the test sets.")
    elif p_paired >= 0.05:
        verdict = (f"CV prefers layer {best}, but the difference from layer "
                   f"{CURRENT_LAYER} is not distinguishable from noise (p = {p_paired:.3f}). "
                   "Either is defensible; report the selection rule, not the winner.")
    else:
        verdict = (f"CV prefers layer {best} over layer {CURRENT_LAYER} and the difference "
                   f"is significant (p = {p_paired:.3f}). The transfer experiments should "
                   f"be re-run on layer {best}.")
    print(f"\n  {verdict}")

    LOGS.mkdir(parents=True, exist_ok=True)
    R.to_csv(LOGS / "select_transfer_layer.csv", index=False)
    (LOGS / "select_transfer_layer.json").write_text(json.dumps({
        "cache": CACHE, "seed": args.seed, "n_layers": n_layers,
        "n_recordings": int(len(y)), "n_speakers": int(len(set(groups))),
        "best_layer": best, "best_auroc": round(best_auroc, 4),
        "current_layer": CURRENT_LAYER, "current_auroc": round(cur_auroc, 4),
        "delta": round(float(delta), 4), "p_paired_bootstrap": round(float(p_paired), 4),
        "selected_without_test_data": True,
        "verdict": verdict, "per_layer": rows,
    }, indent=2))
    print(f"\nwrote {LOGS / 'select_transfer_layer.json'}")


if __name__ == "__main__":
    main()
