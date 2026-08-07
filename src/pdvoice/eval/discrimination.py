"""Discrimination metrics + significance tests (PROJECT_CONTEXT.md §4.4).

AUROC (primary) and AUPRC, with 95% bootstrap CIs. For comparing two models on the
same test set: a paired bootstrap (robust default) and DeLong's test for AUROC.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pdvoice.config import N_BOOTSTRAP, SEED


def auroc(y, p) -> float:
    return float(roc_auc_score(y, p))


def auprc(y, p) -> float:
    return float(average_precision_score(y, p))


def bootstrap_ci(y, p, metric=auroc, n: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = 0.05):
    """Stratified bootstrap CI for a threshold-free metric. Returns (point, lo, hi)."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    stats = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, p)), float(lo), float(hi)


def paired_bootstrap_auroc(y, p1, p2, n: int = N_BOOTSTRAP, seed: int = SEED):
    """Paired bootstrap test that AUROC(p1) != AUROC(p2) on the same data.

    Returns (delta = auroc1 - auroc2, two-sided p-value). Robust default; use
    alongside DeLong.
    """
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    deltas = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    deltas = np.asarray(deltas)
    observed = auroc(y, p1) - auroc(y, p2)
    p_value = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(observed), float(min(p_value, 1.0))


# --- DeLong's test (adapted from Sun & Xu, 2014). Verify against a reference
#     implementation on a known case before reporting. ---

def _midrank(x):
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong_roc_test(y, p1, p2):
    """DeLong two-sided test for two correlated AUROCs. Returns (auc1, auc2, p_value)."""
    from scipy import stats

    y = np.asarray(y)
    order = (-y).argsort(kind="mergesort")
    m = int(np.sum(y == 1))
    preds = np.vstack((np.asarray(p1), np.asarray(p2)))[:, order]
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]

    tx = np.vstack([_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n

    lvec = np.array([[1.0, -1.0]])
    var = float(lvec @ cov @ lvec.T)
    z = (aucs[0] - aucs[1]) / np.sqrt(var + 1e-12)
    return float(aucs[0]), float(aucs[1]), float(2.0 * stats.norm.sf(abs(z)))
