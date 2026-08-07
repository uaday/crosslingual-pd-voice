"""Threshold-dependent metrics, operating-point selection, and calibration error.

Operating threshold = Youden's J on the validation fold; also report sensitivity at
a fixed 90% specificity (PROJECT_CONTEXT.md §4.4). ECE/MCE quantify calibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from pdvoice.config import ECE_BINS


def youden_threshold(y, p) -> float:
    """Threshold maximising Youden's J (= sensitivity + specificity - 1)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def threshold_metrics(y, p, threshold: float) -> dict:
    """Confusion-matrix-derived metrics at a fixed operating threshold."""
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    return {
        "threshold": float(threshold),
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
    """Expected and Maximum Calibration Error over equal-frequency bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    y, p = y[order], p[order]
    ece, mce = 0.0, 0.0
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b) == 0:
            continue
        gap = abs(y[b].mean() - p[b].mean())
        ece += (len(b) / len(p)) * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}
