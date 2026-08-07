"""Smoke tests — verify the scaffold imports and the pure-Python logic works.

These do NOT require the audio datasets (except the two loader tests, which skip if
the data is absent). Run with `pytest -q`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdvoice import config
from pdvoice.calibration import select_best_by_brier
from pdvoice.data.base import MANIFEST_COLUMNS, validate_manifest
from pdvoice.data.splits import assign_folds, assert_disjoint, train_val_test_indices
from pdvoice.eval.dca import net_benefit
from pdvoice.eval.discrimination import auroc, bootstrap_ci, delong_roc_test, paired_bootstrap_auroc
from pdvoice.eval.metrics import calibration_error, threshold_metrics, youden_threshold


def _toy_manifest(n_subjects: int = 40, per_subject: int = 5) -> pd.DataFrame:
    rows = []
    for s in range(n_subjects):
        label = s % 2  # balanced across subjects
        for _ in range(per_subject):
            rows.append(
                {
                    "dataset": "toy",
                    "subject_id": f"s{s}",
                    "path": f"/tmp/s{s}.wav",
                    "label": label,
                    "task_kind": "vowel",
                    "task_code": "sustained_a",
                    "language": "en",
                    "native_sr": 44100,
                }
            )
    return validate_manifest(pd.DataFrame(rows))


def _toy_scores(seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(100), np.zeros(100)].astype(int)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.2, size=len(y)), 0, 1)
    return y, p


def test_config_paths_and_dirs():
    assert config.PROJECT_ROOT.exists()
    config.ensure_dirs()
    assert config.EMBEDDINGS.exists()


def test_manifest_columns():
    df = _toy_manifest()
    assert list(df.columns[: len(MANIFEST_COLUMNS)]) == MANIFEST_COLUMNS
    assert set(df["label"].unique()) <= {0, 1}


def test_subject_disjoint_folds():
    df = assign_folds(_toy_manifest(), n_splits=5)
    assert set(df["fold"].unique()) == {0, 1, 2, 3, 4}
    # no subject spans two folds
    per_subject_folds = df.groupby("subject_id")["fold"].nunique()
    assert (per_subject_folds == 1).all()


def test_train_val_test_split_is_disjoint():
    df = assign_folds(_toy_manifest(), n_splits=5)
    tr, va, te = train_val_test_indices(df, test_fold=0)
    assert len(tr) and len(va) and len(te)
    assert_disjoint(df, tr, va, te)  # raises on leakage


def test_discrimination_and_significance():
    y, p = _toy_scores()
    point, lo, hi = bootstrap_ci(y, p, metric=auroc, n=200)
    assert 0.5 < point <= 1.0 and lo <= point <= hi
    delta, pval = paired_bootstrap_auroc(y, p, p * 0.9 + 0.05, n=200)
    assert 0.0 <= pval <= 1.0
    a1, a2, pv = delong_roc_test(y, p, p * 0.9 + 0.05)
    assert 0.0 <= pv <= 1.0 and 0.5 < a1 <= 1.0


def test_threshold_and_calibration_metrics():
    y, p = _toy_scores()
    thr = youden_threshold(y, p)
    m = threshold_metrics(y, p, thr)
    assert 0.0 <= m["sensitivity"] <= 1.0 and 0.0 <= m["specificity"] <= 1.0
    ce = calibration_error(y, p)
    assert ce["ece"] >= 0 and ce["mce"] >= 0


def test_calibration_selection():
    y, p = _toy_scores()
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    name, cal = select_best_by_brier(logits, y)
    assert name in {"temperature", "platt", "isotonic"}
    proba = cal.predict_proba(logits)
    assert proba.shape == y.shape and ((0 <= proba) & (proba <= 1)).all()


def test_dca_net_benefit():
    y, p = _toy_scores()
    df = net_benefit(y, p)
    assert {"threshold", "model", "treat_all", "treat_none"} <= set(df.columns)
    assert (df["treat_none"] == 0).all()


@pytest.mark.skipif(not config.IPVS_DIR.exists(), reason="IPVS data not present")
def test_ipvs_loader():
    from pdvoice.data.ipvs import load_ipvs

    df = load_ipvs()
    assert len(df) > 0
    assert set(df["task_kind"].unique()) <= {"vowel", "speech", "ddk", "other"}
    assert set(df["label"].unique()) <= {0, 1}


@pytest.mark.skipif(not config.MDVR_KCL_DIR.exists(), reason="MDVR-KCL data not present")
def test_mdvr_kcl_loader():
    from pdvoice.data.mdvr_kcl import load_mdvr_kcl

    df = load_mdvr_kcl()
    assert len(df) > 0
    assert (df["task_kind"] == "speech").all()  # KCL is speech-only
