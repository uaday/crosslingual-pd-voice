"""Regression tests for the 2026-11-10 follow-up review (R1-R9).

Each test exercises the failure mode itself: a three-value trainer contract, one-class
training or validation splits, a failed step in the rerun chain, an infinite Youden
threshold, caches matched by speaker instead of recording, an incomplete provenance
record, absent demographic metadata and eager baseline construction.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


# ------------------------------------------------ R1: train_locked returns three values
LEGACY_CALLERS = ("bridge2ai_transfer.py", "bridge2ai_transfer_agecheck.py",
                  "bridge2ai_followups.py", "demographic_erasure.py")


@pytest.mark.parametrize("name", LEGACY_CALLERS)
def test_train_locked_callers_unpack_three_values_and_transform_inputs(name):
    tree = ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "train_locked"]
    assert calls, f"{name}: no train_locked call found"
    for a in calls:
        assert isinstance(a.targets[0], ast.Tuple) and len(a.targets[0].elts) == 3, \
            f"{name}:{a.lineno} unpacks train_locked into {ast.unparse(a.targets[0])}"
    # every scoring call passes its features through the returned transform
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "score" and len(n.args) == 3:
            feats = n.args[2]
            assert isinstance(feats, ast.Call) and getattr(feats.func, "id", "") == "tf", \
                f"{name}:{n.lineno} scores {ast.unparse(feats)} without the transform"


def test_train_locked_transform_identity_and_erased():
    import demographic_erasure as de
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)],
                       "age": rng.uniform(30, 80, n),
                       "sex": rng.choice(["Male", "Female"], n)})
    y = (np.arange(n) % 2).astype(int)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    X[:, 0] += df.age.to_numpy() / 20
    _, _, tf = de.train_locked(X, y, df.subject_id.to_numpy(), 0)
    assert np.array_equal(tf(X), X)                         # default: identity
    _, _, tf = de.train_locked(X, y, df.subject_id.to_numpy(), 0, erase_df=df)
    Z = tf(X)
    assert Z.shape == X.shape and not np.allclose(Z, X)     # erased: a real transform
    # the erased features carry (almost) no linear age signal
    a = (df.age - df.age.mean()).to_numpy()
    assert abs(np.corrcoef(Z[:, 0], a)[0, 1]) < 0.05


# --------------------------------------------------- R2: two-class training AND validation
def _check(tr, va, y, g, te=()):
    assert len(np.unique(y[tr])) == 2 and len(np.unique(y[va])) == 2
    assert not set(g[tr]) & set(g[va])
    assert not (set(g[tr]) | set(g[va])) & set(g[np.asarray(te, int)])


def test_two_minority_participants_split_into_both_sides():
    from sklearn.model_selection import GroupShuffleSplit

    from pdvoice.data.splits import ensure_two_class_val
    y = np.r_[np.zeros(18, int), np.ones(2, int)]
    g = np.arange(20)
    for seed in range(100):
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(g, y, g))
        tr, va = ensure_two_class_val(tr, va, y, g, seed=seed)
        _check(tr, va, y, g)


def test_repeated_uneven_recordings_keep_participants_whole():
    from sklearn.model_selection import GroupShuffleSplit

    from pdvoice.data.splits import ensure_two_class_val
    rng = np.random.default_rng(1)
    people = [(f"p{i}", int(i < 3)) for i in range(15)]     # 3 PD, 12 controls
    rows = [(s, lab) for s, lab in people for _ in range(rng.integers(1, 6))]
    g = np.array([s for s, _ in rows])
    y = np.array([lab for _, lab in rows])
    idx = np.arange(len(y)) + 1000                           # original row indices kept
    yy, gg = np.zeros(idx.max() + 1, int), np.empty(idx.max() + 1, object)
    yy[idx], gg[idx] = y, g
    for seed in range(100):
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(idx, y, g))
        tr, va = ensure_two_class_val(idx[a], idx[b], yy, gg, seed=seed)
        _check(tr, va, yy, gg)
        assert sorted(np.r_[tr, va]) == sorted(idx)


def test_valid_split_is_returned_unchanged():
    from pdvoice.data.splits import ensure_two_class_val
    y = np.array([0, 1, 0, 1, 0, 1])
    g = np.arange(6)
    tr, va = np.array([0, 1, 2, 3]), np.array([4, 5])
    t2, v2 = ensure_two_class_val(tr, va, y, g)
    assert np.array_equal(t2, tr) and np.array_equal(v2, va)


def test_one_minority_participant_is_impossible():
    from pdvoice.data.splits import ensure_two_class_val
    y = np.r_[np.zeros(10, int), np.ones(3, int)]
    g = np.r_[np.arange(10), [99, 99, 99]]                   # one PD participant, 3 rows
    with pytest.raises(ValueError, match="1 participant"):
        ensure_two_class_val(np.arange(10), np.arange(10, 13), y, g)


def test_train_val_test_indices_imbalanced_and_disjoint_from_test():
    from pdvoice.data.splits import assign_folds, train_val_test_indices
    for seed in range(30):
        df = pd.DataFrame({"subject_id": [f"s{i}" for i in range(40)],
                           "label": [1] * 10 + [0] * 30})
        df = assign_folds(df, 5, seed)
        for k in range(5):
            tr, va, te = train_val_test_indices(df, k, seed=seed)
            _check(tr, va, df.label.to_numpy(), df.subject_id.to_numpy(), te)


# ------------------------------------------------------- R3: the rerun chain stops on failure
def _driver(tmp_path, monkeypatch, codes):
    import run_audit_fixes as raf
    monkeypatch.setattr(raf, "LOG", tmp_path)
    monkeypatch.setattr(raf, "STEPS", [("first", ["a.py"], {}), ("second", ["b.py"], {})])
    ran = []

    def fake_call(cmd, **kw):
        ran.append(cmd[-1])
        return codes[len(ran) - 1]
    monkeypatch.setattr(raf.subprocess, "call", fake_call)
    return raf, ran


def test_failed_step_stops_the_chain(tmp_path, monkeypatch):
    raf, ran = _driver(tmp_path, monkeypatch, [7, 0])
    assert raf.main([]) == 7
    assert ran == ["a.py"]                                   # the dependent step never ran
    log = (tmp_path / "chain.log").read_text()
    assert "CHAIN DONE" not in log and "CHAIN FAILED at first" in log


def test_successful_chain_and_skips(tmp_path, monkeypatch):
    raf, ran = _driver(tmp_path, monkeypatch, [0, 0])
    assert raf.main([]) == 0 and ran == ["a.py", "b.py"]
    assert "CHAIN DONE" in (tmp_path / "chain.log").read_text()
    raf, ran = _driver(tmp_path, monkeypatch, [0])
    assert raf.main(["--skip", "first"]) == 0 and ran == ["b.py"]
    with pytest.raises(SystemExit):
        raf.main(["--skip", "no_such_step"])


# ----------------------------------------------------- R5: infinite Youden threshold
def test_threshold_metrics_accepts_infinite_and_per_participant_thresholds():
    from pdvoice.eval.metrics import threshold_metrics, youden_threshold
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.5)
    thr = youden_threshold(y, p)
    assert np.isinf(thr)
    m = threshold_metrics(y, p, thr)                         # all negative
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (0, 0, 2, 2)
    m = threshold_metrics(y, np.array([0.2, 0.6, 0.4, 0.9]), np.array([0.5, 0.5, 0.3, np.inf]))
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["threshold"] == "per participant"


def test_classical_within_cv_survives_an_infinite_fold_threshold(monkeypatch):
    import classical_transfer as ct
    n = 50
    sub = pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)],
                        "label": (np.arange(n) % 2).astype(int)})
    y = sub.label.to_numpy()
    X = np.arange(n, dtype=np.float32)[:, None]
    calls = []

    def fake_fit(Xtr, ytr, groups, seed):                    # fold 0: all-negative point
        k = len(calls)
        calls.append(k)
        thr = np.inf if k == 0 else 0.5

        def score(A):
            r = A[:, 0] / n
            return np.full(len(A), 0.5) if k == 0 else r, r
        return score, "platt", thr, None
    monkeypatch.setattr(ct, "fit_locked_head", fake_fit)
    m = ct.within_cv(sub, X, y)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == n
    assert all(np.isfinite(m[k]) for k in ("auroc", "ece", "brier") if k in m)
    assert m["threshold"] == "per fold (validation Youden)"


# ------------------------------------------------ R6: recording identity, not speakers
def _cache(path, paths, X, sids=None):
    kw = {"X": X}
    if paths is not None:
        kw["path"] = np.array(paths)
    if sids is not None:
        kw["subject_id"] = np.array(sids)
    np.savez(path, **kw)


def test_load_aligned_reorders_by_path_and_rejects_unidentified(tmp_path):
    from pdvoice.features.ssl import load_aligned
    f = tmp_path / "c.npz"
    _cache(f, ["a.wav", "b.wav"], np.array([[1.0], [2.0]]))
    assert load_aligned(f, ["b.wav", "a.wav"])[:, 0].tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="not in the cache"):
        load_aligned(f, ["a.wav", "c.wav"])
    _cache(f, None, np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="no recording paths"):
        load_aligned(f, ["a.wav", "b.wav"])


def test_transfer_any_cache_needs_matching_recordings_not_speakers(tmp_path):
    import transfer_any as ta
    df = pd.DataFrame({"subject_id": ["s1", "s1"], "path": ["different1.wav", "different2.wav"]})
    f = tmp_path / "c.npz"
    _cache(f, ["old1.wav", "old2.wav"], np.zeros((2, 3)), sids=["s1", "s1"])
    with pytest.raises(SystemExit, match="not in the cache"):
        ta.embed(df, 15, f)
    _cache(f, None, np.zeros((2, 3)), sids=["s1", "s1"])
    with pytest.raises(SystemExit, match="no recording paths"):
        ta.embed(df, 15, f)
    _cache(f, ["different2.wav", "different1.wav"], np.array([[2.0] * 3, [1.0] * 3]))
    assert ta.embed(df, 15, f)[:, 0].tolist() == [1.0, 2.0]


def test_transfer_any_binary_evaluation_drops_unknown_diagnoses(tmp_path, monkeypatch):
    import transfer_any as ta
    df = pd.DataFrame({"subject_id": ["a", "b", "c"], "path": ["a.wav", "b.wav", "c.wav"],
                       "task_kind": "vowel", "task_code": "a", "label": [1, 0, np.nan],
                       "group": ["pd", "control", "other"]})
    monkeypatch.setitem(ta.LOADERS, "bridge2ai", lambda: df)
    monkeypatch.setattr(ta, "quality_control", lambda d: (d, None))
    monkeypatch.setattr(ta, "EMBEDDINGS", tmp_path)
    _cache(tmp_path / "bridge2ai__xls-r-300m__vowel__L8.npz", ["a.wav", "b.wav", "c.wav"],
           np.array([[1.0], [2.0], [3.0]]))
    d, X = ta.get_set("bridge2ai", "vowel", 8)
    assert d.subject_id.tolist() == ["a", "b"] and d.label.dtype.kind == "i"
    assert X[:, 0].tolist() == [1.0, 2.0]
    d, X = ta.get_set("bridge2ai", "vowel", 8, all_groups=True)   # specificity analyses
    assert len(d) == 3 and len(X) == 3


# ----------------------------------------- R7: difference intervals record encoder inputs
def test_difference_cis_stamps_every_encoder_input():
    src = (SCRIPTS / "difference_cis.py").read_text(encoding="utf-8")
    call = src[src.index('stamp(LOGS / "difference_cis.json"'):]
    call = call[:call.index("])") + 2]
    for needed in ("encoder_layer_sweep.json", "layersweep__", "classifier_predictions.csv",
                   "fusion__xls-r-300m__L11.json", "fusion_preds__age-matched__"):
        assert needed in call, needed


def test_changed_encoder_cache_makes_the_result_stale(tmp_path):
    from pdvoice.provenance import stale, stamp
    out, sweep, cache = tmp_path / "d.json", tmp_path / "sweep.json", tmp_path / "ls.npz"
    out.write_text("{}"); sweep.write_text("{}"); np.savez(cache, X=np.zeros(2))
    stamp(out, [sweep, cache])
    assert stale(out) == []
    np.savez(cache, X=np.ones(2))
    assert stale(out) == ["ls.npz"]


# ------------------------------------------------ R8: absent demographic metadata
@pytest.mark.parametrize("fn", ["neurovoz_ages", "ipvs_ages"])
def test_absent_metadata_gives_the_empty_schema(tmp_path, fn):
    import pdvoice.data.demographics as d
    for root in (tmp_path / "missing", tmp_path):
        t = getattr(d, fn)(root)
        assert t.empty and t.columns.tolist() == ["subject_id", "age", "sex"]


def test_attach_age_without_metadata_leaves_age_unknown(monkeypatch, tmp_path):
    import pdvoice.data.demographics as d
    monkeypatch.setattr(d, "neurovoz_ages", lambda: d.pd.DataFrame(columns=d._AGE_COLUMNS))
    m = pd.DataFrame({"subject_id": ["neurovoz_hc_0001"], "label": [0]})
    out = d.attach_age(m, "neurovoz")
    assert out.age.isna().all() and out.sex.isna().all()


# ---------------------------------------------------------- R9: only the chosen baseline
def test_sklearn_baseline_builds_only_the_selected_estimator():
    from pdvoice.models.baselines import sklearn_baseline
    assert sklearn_baseline("svm", C=2)[-1].C == 2
    assert sklearn_baseline("rf", max_depth=3)[-1].max_depth == 3
    assert sklearn_baseline("gbt", learning_rate=0.2)[-1].learning_rate == 0.2
    with pytest.raises(ValueError, match="unknown baseline"):
        sklearn_baseline("knn")
