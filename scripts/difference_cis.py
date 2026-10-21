"""Review items A7 and A6 (Holm): how large could the differences between models be?

"Statistically indistinguishable" (p > 0.6) is an absence of evidence, not evidence of no
difference. This reports, on the age-matched mPower cohort, the paired-bootstrap 95% CI of
the AUROC difference (1,000 participant resamples, same participants on both sides) for:

  encoders     each encoder at its best layer against XLS-R at its best layer
               (linear probe, as in encoder_layer_sweep.py, recomputed at that layer)
  feature sets SSL, classical and their fusion (fusion_preds__age-matched__*.csv)
  classifiers  every pair of the five classifiers on the SSL embeddings
               (classifier_predictions.csv)

It then applies the Holm correction within each family of the significance tests in
Table 4.6 (fusion__xls-r-300m__L11.json): DeLong, paired bootstrap and McNemar, separately
for the uncorrected and the age-matched cohort.

Run:  .venv\\Scripts\\python.exe -u scripts\\difference_cis.py
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from encoder_layer_sweep import linear_probe, load  # noqa: E402

from pdvoice.config import LOGS  # noqa: E402

N = 1000


def diff_ci(y, a, b, seed=42):
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(N):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    obs = roc_auc_score(y, a) - roc_auc_score(y, b)
    return {"delta": round(float(obs), 3), "ci95": [round(float(np.percentile(d, 2.5)), 3),
                                                     round(float(np.percentile(d, 97.5)), 3)]}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main() -> None:
    warnings.simplefilter("ignore")
    out = {"encoders": {}, "feature_sets": {}, "classifiers": {}, "holm": {}}

    # ---- encoders, best layer each, same participants
    best = {r["model"]: r["best_layer"] for r in json.loads((LOGS / "encoder_layer_sweep.json").read_text())["best"]}
    oofs, ref_y = {}, None
    for model, layer in best.items():
        got = load(model)
        if got is None:
            continue
        X, sids, y, _ = got
        _, oof = linear_probe(X[:, layer, :], y, sids, 42)
        oofs[model] = (sids, y, oof)
    ref_sids, ref_y, ref = oofs["xls-r-300m"]
    for model, (sids, y, oof) in oofs.items():
        if model == "xls-r-300m":
            continue
        assert np.array_equal(sids, ref_sids), "encoder caches disagree on participants"
        out["encoders"][f"{model} - xls-r-300m"] = diff_ci(y, oof, ref)
    print("encoders", out["encoders"], flush=True)

    # ---- feature sets
    fp = {k: pd.read_csv(LOGS / f"fusion_preds__age-matched__{k}.csv").set_index("s")
          for k in ("SSL_only", "classical_only", "FUSION")}
    idx = fp["SSL_only"].index
    for a, b in (("SSL_only", "classical_only"), ("FUSION", "SSL_only"), ("FUSION", "classical_only")):
        out["feature_sets"][f"{a} - {b}"] = diff_ci(fp[a].loc[idx, "y"], fp[a].loc[idx, "p"], fp[b].loc[idx, "p"])
    print("feature sets", out["feature_sets"], flush=True)

    # ---- classifiers on SSL embeddings
    c = pd.read_csv(LOGS / "classifier_predictions.csv")
    c = c[(c.cohort == "age-matched") & (c.features == "SSL embeddings (2048)")]
    piv = c.pivot_table(index="subject_id", columns="classifier", values="p")
    yy = c.groupby("subject_id").y.first().loc[piv.index]
    for a, b in itertools.combinations(piv.columns, 2):
        out["classifiers"][f"{a} - {b}"] = diff_ci(yy, piv[a], piv[b])
    widest = max(max(abs(v["ci95"][0]), abs(v["ci95"][1])) for v in out["classifiers"].values())
    out["classifiers_max_abs_ci_bound"] = round(widest, 3)
    print("classifiers widest CI bound", widest, flush=True)

    # ---- Holm within each test family of Table 4.6
    f = json.loads((LOGS / "fusion__xls-r-300m__L11.json").read_text())
    for scen, block in f["scenarios"].items():
        tests = block["_tests"]
        names = list(tests)
        for key in ("p_delong", "p_bootstrap", "p_mcnemar"):
            ps = [tests[n].get(key) for n in names]
            if any(p is None for p in ps):
                continue
            adj = holm(ps)
            for n, p, a in zip(names, ps, adj):
                out["holm"].setdefault(scen, {}).setdefault(n, {})[key] = {"raw": p, "holm": round(float(a), 4)}
    print("holm", json.dumps(out["holm"])[:600], flush=True)

    (LOGS / "difference_cis.json").write_text(json.dumps(out, indent=2))
    print("wrote difference_cis.json")


if __name__ == "__main__":
    main()
