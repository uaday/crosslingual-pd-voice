"""Probability calibration — fit on val, LOCK before test (PROJECT_CONTEXT.md §4.3).

Three candidate maps (temperature / Platt / isotonic); pick the best by validation
Brier, then freeze it and apply unchanged to the test fold and to each external set.
Each calibrator exposes ``fit(scores, y)`` and ``predict_proba(scores)``, where
``scores`` are raw logits (temperature/Platt) or the model's raw logits (isotonic
converts internally).
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """Single-parameter temperature scaling on logits."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits, y):
        from scipy.optimize import minimize_scalar

        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y, dtype=float)

        def nll(t: float) -> float:
            p = np.clip(_sigmoid(logits / max(t, 1e-3)), 1e-7, 1 - 1e-7)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        self.temperature = float(minimize_scalar(nll, bounds=(1e-2, 100), method="bounded").x)
        return self

    def predict_proba(self, logits):
        return _sigmoid(np.asarray(logits, dtype=float) / self.temperature)


class PlattScaler:
    """Logistic regression on the logit (Platt scaling)."""

    def __init__(self) -> None:
        self.lr = LogisticRegression()

    def fit(self, logits, y):
        self.lr.fit(np.asarray(logits, dtype=float).reshape(-1, 1), np.asarray(y))
        return self

    def predict_proba(self, logits):
        return self.lr.predict_proba(np.asarray(logits, dtype=float).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression on the sigmoid of the logit."""

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, logits, y):
        self.iso.fit(_sigmoid(np.asarray(logits, dtype=float)), np.asarray(y))
        return self

    def predict_proba(self, logits):
        return self.iso.predict(_sigmoid(np.asarray(logits, dtype=float)))


CALIBRATORS = {
    "temperature": TemperatureScaler,
    "platt": PlattScaler,
    "isotonic": IsotonicCalibrator,
}


def select_best_by_brier(val_logits, y_val):
    """Fit all candidates on the validation fold and return (name, fitted_calibrator).

    Selection is by validation Brier score, as specified in the proposal.
    """
    y_val = np.asarray(y_val)
    best_name, best_cal, best_brier = None, None, float("inf")
    for name, cls in CALIBRATORS.items():
        cal = cls().fit(val_logits, y_val)
        brier = brier_score_loss(y_val, cal.predict_proba(val_logits))
        if brier < best_brier:
            best_name, best_cal, best_brier = name, cal, brier
    return best_name, best_cal
