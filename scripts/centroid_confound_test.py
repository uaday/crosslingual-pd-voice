"""I1 -- why did the published cross-lingual fix fail for us?

The cross-lingual dysarthria literature proposes a cheap domain adaptation:

    x~ = x - mu_source_HC + mu_target_HC

with centroids computed from HEALTHY CONTROLS ONLY, so that the shift captures
language/corpus variation rather than pathology. Reported results are strong: F1 from
0.48 to 0.74 and sensitivity from 0.35 to 0.93, with language identity in the embedding
dropping from 96% to chance.

Our F-5 records it as a FAILURE: IPVS 0.356 -> 0.16, NeuroVoz +0.00, MDVR +0.05.

THE HYPOTHESIS
--------------
Their cohorts are age-matched (60-67 years across all three languages). Ours is not:
mPower controls average 32.6 years against patients at 60.0. The centroid is built from
controls only, so when the source controls are ~33 and the target controls are ~64,

    mu_target_HC - mu_source_HC

encodes an AGE difference at least as much as a LANGUAGE difference. Applying it adds a
~30-year age offset to every source vector. The fix should therefore not merely fail --
it should actively hurt, which is what we observed.

If that is right, the same shift applied to an AGE-MATCHED source should behave the way
the paper reports. That is a testable claim, and it converts a negative result into a
contribution: the published method silently assumes demographically comparable cohorts.

Run:  .venv\\Scripts\\python.exe -u scripts\\centroid_confound_test.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch

from pdvoice.calibration import select_best_by_brier
from pdvoice.cohort import (age_match_ids, attach_sex, cap_per_subject,
                            demographic_match_ids, load_mpower_cohort,
                            subject_age_stats)
from pdvoice.config import EMBEDDINGS, LOGS, VAL_FRACTION
from pdvoice.data import LOADERS
from pdvoice.eval.discrimination import auroc
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control
from pdvoice.data.splits import ensure_two_class_val
from sklearn.model_selection import GroupShuffleSplit

MODEL, LAYER = "xls-r-300m", 15
EXTERNALS = [("neurovoz", "vowel"), ("ipvs", "vowel"), ("mdvr_kcl", "speech")]


def load_external(ds, task):
    edf = LOADERS[ds]()
    edf = edf[edf.task_kind == task].copy()
    edf, _ = quality_control(edf)
    edf = edf.reset_index(drop=True)
    E = np.load(EMBEDDINGS / f"{ds}__{MODEL}__{task}__L{LAYER}.npz")["X"].astype(np.float32)
    assert len(E) == len(edf), f"{ds}: {len(E)} != {len(edf)}"
    return edf, E


def train_locked(X, y, groups, seed):
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed).split(X, y, groups))
    tr, va = ensure_two_class_val(tr, va, np.asarray(y), np.asarray(groups), seed)
    head, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va], seed=seed)
    _, cal = select_best_by_brier(vlog, y[va])
    return head, cal


def score(head, cal, E):
    dev = next(head.parameters()).device
    with torch.no_grad():
        lg = head(torch.tensor(E, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
    return cal.ranking(lg)  # discrimination: strictly increasing map, no ties (RankingMap)


def subject_auroc(edf, p):
    g = pd.DataFrame({"s": edf.subject_id, "y": edf.label.astype(int), "p": p}) \
          .groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    return auroc(g["y"], g["p"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    seeds = [int(s) for s in args.seeds.split(",")]

    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))
    base = load_mpower_cohort(cap=3, seed=42)
    n_cohort = len(base)
    base = base.merge(emb, on="path", how="inner")
    # Join on PATH, which is unique on both sides. This used to join on subject_id --
    # non-unique on BOTH frames, because the cohort is recording-level (<=3 rows per
    # speaker) and the cache is recording-level and uncapped. That is a cartesian
    # product per speaker: 12,053 cohort rows became 163,670, every one of the 55,480
    # cached recordings entered (each ~3x, one speaker contributing 1,065 rows), and
    # cap=3 was silently undone. It never raised; it returned a plausible number.
    assert len(base) <= n_cohort, (
        f"fan-out: {n_cohort:,} cohort rows became {len(base):,} after the join")
    assert base.row.is_unique, "an embedding row was selected more than once"

    ext = {ds: load_external(ds, task) for ds, task in EXTERNALS}

    cohorts = {
        "confounded source": lambda s: np.ones(len(base), bool),
        "age-matched source": lambda s: base.subject_id.isin(age_match_ids(base, seed=s)).to_numpy(),
    }

    rows = []
    for cname, maskfn in cohorts.items():
        for seed in seeds:
            m = maskfn(seed)
            sub = base[m]
            X = d["X"][sub.row.to_numpy()].astype(np.float32)
            y = sub.label.to_numpy()
            hc_mu = X[y == 0].mean(axis=0)          # source healthy-control centroid
            head, cal = train_locked(X, y, sub.subject_id.to_numpy(), seed)

            _ages = subject_age_stats(sub)   # per speaker, not per recording
            age_pd, age_hc = _ages["age_pd"], _ages["age_hc"]

            for ds, (edf, E) in ext.items():
                plain = subject_auroc(edf, score(head, cal, E))
                hc_mask = (edf.label == 0).to_numpy()
                if hc_mask.sum() < 5:
                    continue
                tgt_mu = E[hc_mask].mean(axis=0)     # target healthy-control centroid
                shifted = E - tgt_mu + hc_mu         # x~ = x - mu_target_HC + mu_source_HC
                adapted = subject_auroc(edf, score(head, cal, shifted))
                rows.append(dict(source=cname, seed=seed, target=ds,
                                 source_age_gap=round(float(age_pd - age_hc), 1),
                                 plain=round(plain, 4), adapted=round(adapted, 4),
                                 gain=round(adapted - plain, 4)))
            print(f"  [{cname}, seed {seed}] source age gap "
                  f"{age_pd - age_hc:+.1f} yr, n={sub.subject_id.nunique():,}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(LOGS / "centroid_confound_test.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n{'=' * 92}\nDOES THE CENTROID SHIFT HELP?  (mean +/- SD over seeds)\n{'=' * 92}")
    agg = R.groupby(["source", "target"]).agg(
        age_gap=("source_age_gap", "mean"),
        plain=("plain", "mean"), plain_sd=("plain", "std"),
        adapted=("adapted", "mean"), adapted_sd=("adapted", "std"),
        gain=("gain", "mean"), gain_sd=("gain", "std")).round(3)
    print(agg.to_string())

    print(f"\n{'=' * 92}\nREADING\n{'=' * 92}")
    for cname in R.source.unique():
        g = R[R.source == cname]
        print(f"  {cname:<20} source age gap {g.source_age_gap.mean():+.1f} yr"
              f"   mean adaptation gain {g.gain.mean():+.4f}")
    conf = R[R.source == "confounded source"].gain.mean()
    matched = R[R.source == "age-matched source"].gain.mean()
    print(f"\n  difference in what the shift buys: {matched - conf:+.4f}")
    if matched > 0 >= conf:
        print("  -> The published fix WORKS on an age-matched source and FAILS on a")
        print("     confounded one. HC-only centroids encode the demographic difference,")
        print("     not just the language one, so the method silently assumes comparable")
        print("     cohorts. That is a contribution, not a failed replication.")
    elif matched > conf:
        print("  -> The shift helps more from an age-matched source, in the predicted")
        print("     direction, but it does not fully rescue the method.")
    else:
        print("  -> The hypothesis is NOT supported: matching the source does not make the")
        print("     shift work. The failure has some other cause and F-5 stands as a plain")
        print("     negative result.")

    (LOGS / "centroid_confound_test.json").write_text(
        json.dumps({"rows": rows,
                    "mean_gain_confounded": round(float(conf), 4),
                    "mean_gain_agematched": round(float(matched), 4)}, indent=2))
    print(f"\nwrote {LOGS / 'centroid_confound_test.csv'}")


if __name__ == "__main__":
    main()
