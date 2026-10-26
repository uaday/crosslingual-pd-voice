"""What does the 2048-dim feature vector actually encode?  (supervisor ask, 2026-09-05)

    "When you apply them, when you get the feature vector out of it -- what does it
     mean? How can it be [interpreted] and then how can it be used for different
     training mechanisms? This is the main component, main part of your thesis, and
     that is the development that you need to do, not just the implementation and
     reporting some numbers."

An embedding is 2048 numbers with no names. The honest way to say what they mean is to
ask what can be READ BACK OUT of them. For each attribute we train a probe on the frozen
embeddings and report how well it recovers that attribute. A high score means the
information is present and available to any downstream classifier -- wanted or not.

This doubles as confound documentation, and it completes an argument the rest of the
project only half-makes:

    age alone, no audio, predicts PD at 0.922   (scripts/classifier_comparison.py)
    embeddings encode age at ...                 <- measured here
    => the embeddings inherit the confound, and a PD head trained on them
       cannot help but exploit it

DESIGN NOTES
------------
* Probes are subject-disjoint (GroupKFold on speaker), like every other result here.
* Demographic probes run on HEALTHY CONTROLS ONLY wherever the attribute correlates
  with disease. Probing age across both groups would partly recover the label, which
  would tell us nothing about the embedding.
* Language and dataset probes hold the TASK constant (sustained vowel), otherwise a
  "language" probe is really a task probe.
* Two layers are compared -- L15 (used for transfer) and L24 (best within mPower) --
  because "which layer is more invariant to nuisance" is a design question we have
  never actually answered.

Run:  .venv\\Scripts\\python.exe -u scripts\\embedding_probes.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from pdvoice.config import EMBEDDINGS, LOGS, MPOWER_DIR
from pdvoice.data import LOADERS
from pdvoice.preprocess import quality_control
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL = "xls-r-300m"
LAYERS = [11, 24]


def load_mpower_layer(layer):
    f = EMBEDDINGS / f"mpower_full__{MODEL}__L{layer}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f}\nRun: scripts/extract_mpower_full.py --layer {layer}")
    d = np.load(f, allow_pickle=True)
    return d["X"].astype(np.float32), d["subject_id"].astype(str), d["label"].astype(int)


def mpower_meta():
    d = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    d["hc"] = d["healthCode"].astype(str)
    return d.drop_duplicates("hc").set_index("hc")


def probe_binary(X, y, groups, n_splits=5):
    """AUROC of recovering a binary attribute from the embeddings, subject-disjoint."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < n_splits:
        return None
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def probe_multiclass(X, y, groups, n_splits=5):
    """Accuracy of recovering a categorical attribute; chance = majority-class share.

    1/n_classes understates chance when the classes are unbalanced (600 mPower vowels
    against a few hundred from each clinical dataset), so the always-majority rate is used.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2:
        return None, None
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float(accuracy_score(y, pred)), float(pd.Series(y).value_counts(normalize=True).iloc[0])


def probe_continuous(X, y, groups, n_splits=5):
    """R^2 and MAE of recovering a continuous attribute (age, in years).

    The penalty must be CHOSEN, not guessed. With 2048 features a weak ridge overfits
    hard: at alpha=10 this probe returned R2 = -0.73, i.e. worse than predicting the
    mean, which reads like "age is not encoded" when it actually means "the probe was
    misconfigured". RidgeCV picks alpha per fold; at a well-chosen penalty the same data
    gives R2 = +0.17.
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    X, y, groups = X[ok], y[ok], np.asarray(groups)[ok]
    if len(y) < 50:
        return None, None
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=np.logspace(1, 6, 12)))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return float(r2_score(y, oof)), float(mean_absolute_error(y, oof))


def one_per_subject(sids, seed=42):
    """Row indices keeping one recording per speaker, so heavy users cannot dominate."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sids))
    seen, keep = set(), []
    for i in order:
        if sids[i] not in seen:
            seen.add(sids[i])
            keep.append(i)
    return np.sort(np.array(keep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    meta = mpower_meta()
    results = []

    for layer in LAYERS:
        X, sids, labels = load_mpower_layer(layer)
        keep = one_per_subject(sids, args.seed)
        X, sids, labels = X[keep], sids[keep], labels[keep]
        key = pd.Series(sids).str.replace("mpower_", "", regex=False)
        age = key.map(meta["age"]).to_numpy(dtype=float)
        gender = key.map(meta["gender"]).astype(str).to_numpy()
        phone = key.map(meta["phoneInfo"]).astype(str).to_numpy()

        hc = labels == 0            # controls only: age/sex must not smuggle in the label
        print(f"\n=== LAYER {layer} · mPower · {len(X):,} speakers "
              f"({int(hc.sum()):,} controls) ===", flush=True)

        r2, mae = probe_continuous(X[hc], age[hc], sids[hc])
        results.append(dict(layer=layer, attribute="age (years)", cohort="mPower controls",
                            metric="R2", value=round(r2, 4), extra=f"MAE {mae:.1f} yr",
                            chance=0.0))
        print(f"  age            R2 = {r2:.3f}   MAE {mae:.1f} years")

        for lo, hi in [(35, 55)]:
            m_ = np.isfinite(age) & hc & ((age < lo) | (age >= hi))
            y_ = (age[m_] >= hi).astype(int)
            a_age = probe_binary(X[m_], y_, sids[m_])
            results.append(dict(layer=layer, attribute=f"age, young(<{lo}) vs older(>={hi})",
                                cohort="mPower controls", metric="AUROC",
                                value=round(a_age, 4), extra=f"n={len(y_)}", chance=0.5))
            print(f"  age (binary)   AUROC = {a_age:.3f}   young<{lo} vs older>={hi}, n={len(y_)}")

        g = pd.Series(gender[hc])
        gm = g.isin(["Male", "Female"]).to_numpy()
        a = probe_binary(X[hc][gm], (g[gm] == "Female").astype(int).to_numpy(), sids[hc][gm])
        results.append(dict(layer=layer, attribute="sex", cohort="mPower controls",
                            metric="AUROC", value=round(a, 4), extra="", chance=0.5))
        print(f"  sex            AUROC = {a:.3f}")

        ph = pd.Series(phone[hc])
        top = ph.value_counts().head(4).index.tolist()
        pm = ph.isin(top).to_numpy()
        acc, ch = probe_multiclass(X[hc][pm], ph[pm].to_numpy(), sids[hc][pm])
        results.append(dict(layer=layer, attribute=f"recording device ({len(top)}-way)",
                            cohort="mPower controls", metric="accuracy",
                            value=round(acc, 4), extra=f"chance {ch:.2f}", chance=ch))
        print(f"  device model   accuracy = {acc:.3f}   (chance {ch:.2f})")

        a_pd = probe_binary(X, labels, sids)
        results.append(dict(layer=layer, attribute="** PD diagnosis **", cohort="mPower all",
                            metric="AUROC", value=round(a_pd, 4),
                            extra="confounded cohort", chance=0.5))
        print(f"  PD (reference) AUROC = {a_pd:.3f}   <- what we actually want")

    # ---------------- cross-dataset: language / corpus / task -------------------
    print(f"\n=== CROSS-DATASET (task held constant: sustained vowel) ===", flush=True)
    Xs, langs, dsets, sids_all = [], [], [], []
    Xm, sm, _ = load_mpower_layer(11)
    k = one_per_subject(sm, args.seed)[:600]
    Xs.append(Xm[k]); langs += ["en"] * len(k); dsets += ["mpower"] * len(k)
    sids_all += list(sm[k])
    for ds, lang in [("neurovoz", "es"), ("ipvs", "it")]:
        edf = LOADERS[ds]()
        edf = edf[edf.task_kind == "vowel"].copy()
        edf, _ = quality_control(edf); edf = edf.reset_index(drop=True)
        E = np.load(EMBEDDINGS / f"{ds}__{MODEL}__vowel__L11.npz")["X"].astype(np.float32)
        assert len(E) == len(edf)
        Xs.append(E); langs += [lang] * len(E); dsets += [ds] * len(E)
        sids_all += list(edf.subject_id)
    Xc = np.vstack(Xs)
    acc, ch = probe_multiclass(Xc, np.array(langs), np.array(sids_all))
    results.append(dict(layer=11, attribute="language (3-way)", cohort="pooled vowel",
                        metric="accuracy", value=round(acc, 4),
                        extra=f"chance {ch:.2f}", chance=ch))
    print(f"  language       accuracy = {acc:.3f}   (chance {ch:.2f})")
    acc2, ch2 = probe_multiclass(Xc, np.array(dsets), np.array(sids_all))
    results.append(dict(layer=11, attribute="corpus identity (3-way)", cohort="pooled vowel",
                        metric="accuracy", value=round(acc2, 4),
                        extra=f"chance {ch2:.2f}", chance=ch2))
    print(f"  corpus         accuracy = {acc2:.3f}   (chance {ch2:.2f})")

    # task, within one corpus and one language
    edf = LOADERS["neurovoz"]()
    edf = edf[edf.task_kind.isin(["vowel", "speech"])].copy()
    parts = []
    for task in ("vowel", "speech"):
        sub = edf[edf.task_kind == task].copy()
        sub, _ = quality_control(sub); sub = sub.reset_index(drop=True)
        E = np.load(EMBEDDINGS / f"neurovoz__{MODEL}__{task}__L11.npz")["X"].astype(np.float32)
        assert len(E) == len(sub)
        parts.append((E, sub))
    Xt = np.vstack([p[0] for p in parts])
    yt = np.r_[np.zeros(len(parts[0][0])), np.ones(len(parts[1][0]))].astype(int)
    gt = np.r_[parts[0][1].subject_id.to_numpy(), parts[1][1].subject_id.to_numpy()]
    a_task = probe_binary(Xt, yt, gt)
    results.append(dict(layer=11, attribute="task (vowel vs speech)", cohort="NeuroVoz",
                        metric="AUROC", value=round(a_task, 4), extra="", chance=0.5))
    print(f"  task           AUROC = {a_task:.3f}")

    # ---------------- the table that answers the question ----------------------
    R = pd.DataFrame(results)
    R["above_chance"] = (R["value"] - R["chance"]).round(4)
    R.to_csv(LOGS / "embedding_probes.csv", index=False)
    (LOGS / "embedding_probes.json").write_text(json.dumps(results, indent=2))

    pd.set_option("display.width", 200)
    print(f"\n{'='*96}\nWHAT THE EMBEDDING ENCODES, ranked by how far above chance\n{'='*96}")
    print(R.sort_values("above_chance", ascending=False)
           [["layer", "attribute", "cohort", "metric", "value", "chance", "above_chance", "extra"]]
           .to_string(index=False))

    print(f"\n{'='*96}\nWHICH LAYER IS MORE INVARIANT TO NUISANCE?\n{'='*96}")
    piv = R[R.layer.isin(LAYERS)].pivot_table(index="attribute", columns="layer", values="value")
    if piv.shape[1] == 2:
        piv["L24 - L11"] = (piv[24] - piv[11]).round(4)
        print(piv.round(4).to_string())
    print(f"\nwrote {LOGS / 'embedding_probes.csv'}")


if __name__ == "__main__":
    main()
