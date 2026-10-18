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


class RankingMap:
    """Strictly increasing map from logit to a common probability scale, for DISCRIMINATION.

    Calibration must not change the ranking, but the isotonic map (which wins the
    validation-Brier selection almost every time) is a step function that clips scores
    outside its fitted range. Applied to a shifted external dataset it collapses distinct
    scores into ties, and in the worst case gives every participant 0.5, so an AUROC computed
    on calibrated probabilities can be forced to exactly 0.5. The raw logit has no ties, but
    when out-of-fold predictions from several fold models are pooled, each model's logits sit
    on their own scale. Platt scaling fitted on the same validation fold fixes both: it is
    strictly increasing (no ties, ranking within a model unchanged) and puts every fold on a
    common probability scale. If the fitted slope is not positive, the map falls back to the
    sigmoid so that the ranking is never reversed.

    AUROC, AUPRC and sensitivity at fixed specificity use this score; the selected
    calibrator's probabilities are used for calibration, Brier, threshold and decision-curve
    measures.
    """

    def fit(self, logits, y):
        z = np.asarray(logits, dtype=float).reshape(-1, 1)
        y = np.asarray(y)
        self.a, self.b = 1.0, 0.0
        if len(np.unique(y)) < 2:   # a small validation fold can hold one class only
            return self
        lr = LogisticRegression().fit(z, y)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if a > 0:
            self.a, self.b = a, b
        return self

    def __call__(self, logits) -> np.ndarray:
        return _sigmoid(self.a * np.asarray(logits, dtype=float) + self.b)


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
    best_cal.ranking = RankingMap().fit(val_logits, y_val)  # see RankingMap
    return best_name, best_cal
