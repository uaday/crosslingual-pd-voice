"""Regression tests for the failure modes found by the 2026-11-09 code audit.

Each test pins one defect that previously passed silently: tied-probability ECE binning,
one-class validation splits, stale derived results, cache identity, the IPVS demographic
key, and the level dependence of the classical features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------- ECE (audit F9)
def _old_ece(y, p, n_bins=10):
    o = np.argsort(p)
    y, p = y[o], p[o]
    return sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean())
               for b in np.array_split(np.arange(len(p)), n_bins) if len(b))


def test_ece_keeps_tied_probabilities_in_one_bin():
    from pdvoice.eval.metrics import calibration_error
    y = np.array([1] * 12 + [0] * 12, float)
    p = np.full(24, 0.557971)
    # one probability, observed rate 0.5: the only honest gap is |0.5 - 0.557971|
    assert calibration_error(y, p)["ece"] == pytest.approx(0.057971)


def test_ece_unchanged_when_there_are_no_ties():
    from pdvoice.eval.metrics import calibration_error
    rng = np.random.default_rng(0)
    for n in (7, 24, 111, 1124):
        p = rng.random(n)
        y = (rng.random(n) < p).astype(float)
        assert calibration_error(y, p)["ece"] == pytest.approx(_old_ece(y, p), abs=1e-12)


# ------------------------------------------------- validation splits (audit, sec. 4)
def test_validation_split_always_has_both_classes():
    from pdvoice.data.splits import assign_folds, train_val_test_indices
    for seed in range(60):
        df = pd.DataFrame({"subject_id": [f"s{i}" for i in range(20)],
                           "label": [0] * 10 + [1] * 10})
        df = assign_folds(df, 5, seed)
        for k in range(5):
            tr, va, te = train_val_test_indices(df, k, seed=seed)
            assert df.loc[va, "label"].nunique() == 2
            assert df.loc[tr, "label"].nunique() == 2
            assert not set(df.loc[tr, "subject_id"]) & set(df.loc[va, "subject_id"])


# ------------------------------------------------------ provenance (audit F1)
def test_provenance_detects_a_changed_input(tmp_path):
    from pdvoice.provenance import stale, stamp
    inp, out = tmp_path / "preds.csv", tmp_path / "curves.json"
    inp.write_text("a,b\n1,2\n")
    out.write_text("{}")
    assert stale(out) == ["<no provenance record>"]
    stamp(out, [inp])
    assert stale(out) == []
    inp.write_text("a,b\n1,3\n")            # predictions regenerated, curves not
    assert stale(out) == ["preds.csv"]


# ------------------------------------------------------- cache identity (review 2)
def test_ssl_cache_returns_only_the_requested_recordings(tmp_path, monkeypatch):
    import pdvoice.features.ssl as ssl
    monkeypatch.setattr(ssl, "EMBEDDINGS", tmp_path)
    np.savez_compressed(tmp_path / "toy__wav2vec2-base__mean_std__LNone.npz",
                        X=np.array([[1.0], [2.0]]), ids=np.array([0, 1]),
                        path=np.array(["a.wav", "b.wav"]))
    df = pd.DataFrame({"dataset": ["toy"], "path": ["b.wav"], "task_kind": ["vowel"]},
                      index=[7])
    X, ids = ssl.extract_for_manifest(df)
    assert X.ravel().tolist() == [2.0] and ids.tolist() == [7]


# ----------------------------------------------------- IPVS demographics (audit F5)
def test_ipvs_name_key_keeps_the_group():
    from pdvoice.data.demographics import _name_key
    # a control and a patient with the same short name must not share a key
    assert _name_key("ipvs_ehc_ANNA_B") != _name_key("ipvs_pd_17_28_Anna_B")
    # batch folders and case still normalise away within a group
    assert _name_key("ipvs_pd_17_28_Anna_B") == _name_key("ipvs_pd_Anna_B")


def test_conflicting_demographic_rows_are_rejected():
    from pdvoice.data.demographics import _resolve
    t = pd.DataFrame({"k": ["a", "a", "b", "c", "c"], "age": [70, 71, 60, 50, 50],
                      "sex": ["M", "M", "F", "M", "M"]})
    r, bad = _resolve(t, "k")
    assert bad == ["a"]
    assert np.isnan(r.loc["a", "age"]) and r.loc["b", "age"] == 60 and r.loc["c", "age"] == 50


# ---------------------------------------------- classical features vs level (audit F4)
def test_only_mfcc0_depends_on_recording_level(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("parselmouth")
    from pdvoice.features.classical import FEATURE_NAMES, classical_features
    sr = 44_100
    t = np.arange(int(3 * sr)) / sr
    f0 = 120 * (1 + 0.01 * np.sin(2 * np.pi * 4 * t))           # a little vibrato
    phase = 2 * np.pi * np.cumsum(f0) / sr
    wav = sum(np.sin(k * phase) / k for k in range(1, 12))
    wav = 0.3 * wav / np.abs(wav).max()
    feats = {}
    for gain_db in (0, -20):
        f = tmp_path / f"v{gain_db}.wav"
        # float samples: 16-bit quantisation of a noiseless tone at low level would
        # itself perturb HNR and the formants, which is not what this test is about
        sf.write(f, wav * 10 ** (gain_db / 20), sr, subtype="FLOAT")
        feats[gain_db] = classical_features(str(f))
    moved = [k for k in FEATURE_NAMES
             if np.isfinite(feats[0][k]) and abs(feats[0][k] - feats[-20][k])
             > 1e-3 * max(1.0, abs(feats[0][k]))]
    assert moved == ["mfcc_0"], moved
