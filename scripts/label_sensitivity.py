"""Is the transfer failure caused by LABEL NOISE or by the AGE CONFOUND?  (multi-seed)

This is the evidence for the project's central claim, so it needs error bars. The
single-seed version was misleading: when the cohort was corrected (the phantom-row fix,
4,799 -> 4,809 speakers) its headline figure moved from IPVS 0.356 to 0.555, and the
"below chance" inversion jumped to a different dataset. The conclusions held, but the
numbers had no stated uncertainty and one of them was quoted heavily.

THE ARGUMENT. Two candidate explanations for why a locked mPower model fails abroad:

    label noise   mPower's PD label is self-reported, not clinician-confirmed
    age confound  mPower patients are ~27 years older than its controls

They make opposite predictions, which is what makes this a real test rather than a
description. If labels are the problem, cleaning them should RAISE within-dataset
performance and improve transfer. If age is the problem, removing it should LOWER
within-dataset performance (the model loses a shortcut) while IMPROVING transfer.

Scenarios:
    baseline    all self-report labels, as-is
    confident   PD only if a diagnosis year is present; drop HC who gave one
    age50       keep only speakers aged >= 50 (a blunt cutoff)
    agematched  frequency-matched PD/HC age per 5-year bin (the proper control)

Per seed we redraw the capped recordings, the CV folds, the train/val split and the
head initialisation, so the spread is run-to-run variability rather than weight-init
noise alone.

Run:
    .venv\\Scripts\\python.exe -u scripts\\label_sensitivity.py --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch

from pdvoice.calibration import select_best_by_brier
from pdvoice.cohort import age_match_ids, cap_per_subject
from pdvoice.config import EMBEDDINGS, LOGS, MPOWER_DIR, VAL_FRACTION
from pdvoice.data import LOADERS
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auroc
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control
from sklearn.model_selection import GroupShuffleSplit

MODEL = "xls-r-300m"
WITHIN_LAYER = 24     # last layer -- best within mPower (chosen on within-CV, not on transfer)
TRANSFER_LAYER = 15   # best transfer layer
CAP = 3
EXTERNALS = [("ipvs", "vowel"), ("neurovoz", "vowel"), ("mdvr_kcl", "speech")]


def load_full(layer):
    f = EMBEDDINGS / f"mpower_full__{MODEL}__L{layer}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f}\nRun: scripts/extract_mpower_full.py --layer {layer}")
    d = np.load(f, allow_pickle=True)
    return d["X"].astype(np.float32), d["subject_id"].astype(str), d["label"].astype(int)


def build_frame():
    """One row per recording, aligned to the full-cache row order, with demographics."""
    X15, sids, labels = load_full(TRANSFER_LAYER)
    df = pd.DataFrame({"subject_id": sids, "label": labels})
    d = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    hc = d["healthCode"].astype(str)
    age = dict(zip(hc, d["age"]))
    has_year = dict(zip(hc, d["diagnosis-year"].notna()))
    key = df["subject_id"].str.replace("mpower_", "", regex=False)
    df["age"] = key.map(age)
    df["has_diag_year"] = key.map(has_year).fillna(False)
    return df


def scenario_masks(df, seed):
    lab = df["label"].astype(int)
    keep = age_match_ids(df, seed=seed)
    return {
        "baseline": pd.Series(True, index=df.index),
        "confident": ((lab == 1) & df["has_diag_year"]) | ((lab == 0) & ~df["has_diag_year"]),
        "age50": df["age"] >= 50,
        "agematched": df["subject_id"].isin(keep),
    }


def within_auroc(sub, X, y, seed):
    dfa = assign_folds(sub.reset_index(drop=True), n_splits=5, seed=seed)
    oof = []
    for k in range(5):
        tr, va, te = train_val_test_indices(dfa, test_fold=k, seed=seed)
        m, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va], seed=seed)
        with torch.no_grad():
            dev = next(m.parameters()).device
            tl = m(torch.tensor(X[te], dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        _, cal = select_best_by_brier(vlog, y[va])
        oof.append(pd.DataFrame({"s": dfa.loc[te, "subject_id"].to_numpy(),
                                 "y": y[te], "p": cal.predict_proba(tl)}))
    g = pd.concat(oof).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    return auroc(g["y"], g["p"])


def transfer_aurocs(sub, X, y, ext_cache, seed):
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed)
                  .split(X, y, sub["subject_id"]))
    m, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va], seed=seed)
    _, cal = select_best_by_brier(vlog, y[va])
    dev = next(m.parameters()).device
    out = {}
    for ds, edf, EX in ext_cache:
        with torch.no_grad():
            lg = m(torch.tensor(EX, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        s = pd.DataFrame({"s": edf["subject_id"], "y": edf["label"].astype(int),
                          "p": cal.predict_proba(lg)})
        g = s.groupby("s").agg(y=("y", "first"), p=("p", "mean"))
        out[ds] = auroc(g["y"], g["p"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    seeds = [int(s) for s in args.seeds.split(",")]

    df = build_frame()
    Xw, _, _ = load_full(WITHIN_LAYER)
    Xt, _, _ = load_full(TRANSFER_LAYER)
    assert len(Xw) == len(Xt) == len(df), "the two full caches disagree on row count"

    ext_cache = []
    for ds, task in EXTERNALS:
        edf = LOADERS[ds]()
        edf = edf[edf["task_kind"] == task].copy()
        edf, _ = quality_control(edf)
        edf = edf.reset_index(drop=True)
        EX = np.load(EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{TRANSFER_LAYER}.npz")["X"].astype(np.float32)
        assert len(EX) == len(edf), f"{ds}: {len(EX)} != {len(edf)}"
        ext_cache.append((ds, edf, EX))

    print(f"=== mPower label vs age sensitivity | {len(seeds)} seeds ===")
    print(f"  pool {len(df):,} recordings / {df.subject_id.nunique():,} speakers")
    print(f"  within = layer {WITHIN_LAYER} (last) · transfer = L{TRANSFER_LAYER} · cap-{CAP}")
    print(f"  age: PD {df[df.label==1].age.mean():.1f}  HC {df[df.label==0].age.mean():.1f}"
          f"  gap {df[df.label==1].age.mean()-df[df.label==0].age.mean():+.1f} yr\n")

    runs = {}
    for seed in seeds:
        capped = cap_per_subject(df, CAP, seed)
        for name, mask in scenario_masks(capped, seed).items():
            sub = capped[mask]
            idx = sub.index.to_numpy()
            assert np.array_equal(df.label.to_numpy()[idx], sub.label.to_numpy()), \
                "cohort index no longer aligns with the embedding matrix"
            y = sub["label"].astype(int).to_numpy()
            npd = sub[sub.label == 1].subject_id.nunique()
            nhc = sub[sub.label == 0].subject_id.nunique()
            if npd < 10 or nhc < 10:
                continue
            w = within_auroc(sub, Xw[idx], y, seed)
            t = transfer_aurocs(sub, Xt[idx], y, ext_cache, seed)
            runs.setdefault(name, []).append(
                {"seed": seed, "n_pd": int(npd), "n_hc": int(nhc),
                 "within": round(float(w), 4),
                 **{f"transfer_{k}": round(float(v), 4) for k, v in t.items()}})
            print(f"  [seed {seed}] {name:<11} {npd:>4}/{nhc:<5} within {w:.3f}  "
                  + "  ".join(f"{k} {v:.3f}" for k, v in t.items()), flush=True)

    def ms(name, key):
        v = np.array([r[key] for r in runs[name]], float)
        return f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"

    print(f"\n{'='*88}\nACROSS {len(seeds)} SEEDS (mean +/- SD)\n{'='*88}")
    rows = [{"scenario": n, "PD/HC": f"{runs[n][0]['n_pd']}/{runs[n][0]['n_hc']}",
             "within": ms(n, "within"),
             "ipvs": ms(n, "transfer_ipvs"), "neurovoz": ms(n, "transfer_neurovoz"),
             "mdvr_kcl": ms(n, "transfer_mdvr_kcl")} for n in runs]
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    print(f"\n{'='*88}\nREADING\n{'='*88}")
    b = np.array([r["within"] for r in runs["baseline"]]).mean()
    for n in ("confident", "age50", "agematched"):
        if n not in runs:
            continue
        w = np.array([r["within"] for r in runs[n]]).mean()
        gains = [np.array([r[f"transfer_{d}"] for r in runs[n]]).mean()
                 - np.array([r[f"transfer_{d}"] for r in runs["baseline"]]).mean()
                 for d, _ in EXTERNALS]
        print(f"  {n:<11} within {w-b:+.3f} vs baseline | mean transfer change {np.mean(gains):+.3f}")
    print("\n  A confound shows up as: within DOWN and transfer UP.")
    print("  Label noise would show up as: within UP and transfer UP.")

    LOGS.mkdir(parents=True, exist_ok=True)
    out = {"model": MODEL, "cap": CAP, "within_layer": WITHIN_LAYER,
           "transfer_layer": TRANSFER_LAYER, "seeds": seeds,
           "age_pd": round(float(df[df.label == 1].age.mean()), 2),
           "age_hc": round(float(df[df.label == 0].age.mean()), 2),
           "runs": runs, "summary": rows}
    (LOGS / "label_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {LOGS / 'label_sensitivity.json'}")


if __name__ == "__main__":
    main()
