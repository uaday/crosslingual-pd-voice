"""Threshold-dependent metrics, operating-point selection, and calibration error.

Operating threshold = Youden's J on the validation fold; also report sensitivity at
a fixed 90% specificity. ECE/MCE quantify calibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from pdvoice.config import ECE_BINS


def youden_threshold(y, p) -> float:
    """Threshold maximising Youden's J (= sensitivity + specificity - 1)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def threshold_metrics(y, p, threshold) -> dict:
    """Confusion-matrix-derived metrics at an operating threshold.

    ``threshold`` is one value or one per participant (for example each cross-validation
    fold's own validation threshold). Decisions are ``p >= threshold`` directly, so an
    infinite threshold (Youden's all-negative operating point) is handled exactly.
    """
    y = np.asarray(y)
    thr = np.asarray(threshold, dtype=float)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    return {
        "threshold": float(thr) if thr.ndim == 0 else "per participant",
        "accuracy": (tp + tn) / len(y),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "f1": f1,
        "balanced_accuracy": 0.5 * (sens + spec),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def sensitivity_at_specificity(y, p, target_spec: float = 0.90) -> dict:
    """Highest sensitivity achievable while specificity >= target."""
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    ok = np.where(spec >= target_spec)[0]
    if len(ok) == 0:
        return {"sensitivity": 0.0, "specificity": 1.0, "threshold": 1.0}
    i = ok[int(np.argmax(tpr[ok]))]
    return {"sensitivity": float(tpr[i]), "specificity": float(spec[i]), "threshold": float(thr[i])}


def calibration_error(y, p, n_bins: int = ECE_BINS) -> dict:
    """Expected and Maximum Calibration Error over equal-frequency bins, ties kept together.

    Participants are sorted by predicted probability and cut into ``n_bins`` bins of
    (nearly) equal size, except that a cut never separates participants with the SAME
    probability: each cut moves forward to the end of its run of ties. Before 2026-11-09
    the cuts fell at fixed row positions, so a run of identical probabilities could be
    split across bins with different label mixes, and the error then depended on the
    arbitrary order within the tie. With every participant at one probability the
    result is now simply |observed rate - predicted probability|.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    y, p = y[order], p[order]
    n = len(p)
    cuts = []
    # start from the same boundaries as np.array_split, so untied data bin exactly as before
    for c in np.cumsum([len(b) for b in np.array_split(np.arange(n), n_bins)])[:-1]:
        c = int(c)
        while 0 < c < n and p[c - 1] == p[c]:
            c += 1
        if 0 < c < n and (not cuts or c > cuts[-1]):
            cuts.append(c)
    ece, mce = 0.0, 0.0
    for b in np.split(np.arange(n), cuts):
        if len(b) == 0:
            continue
        gap = abs(y[b].mean() - p[b].mean())
        ece += (len(b) / len(p)) * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}
