"""Does the connected-speech transfer result survive age matching?

WHY THIS EXISTS. The vowel results in this project are age-matched, sex-controlled and
five-seeded. The speech transfer matrix -- the project's main POSITIVE claim, 0.70-0.85
in all directions -- is none of those: `transfer_any.py` applies no age control at all.
So the strongest claim rests on results held to a weaker standard than the ones they are
contrasted against, which is precisely the criticism this thesis makes of other people.

The gaps are not small:

    NeuroVoz   PD 71.1 / HC 64.0   gap  +7.1 yr
    IPVS       PD 67.1 / HC 46.6   gap +20.5 yr   <-- the 15 young controls recorded
                                                      speech but not vowels
    MDVR-KCL   no age data exists, anywhere in the release

So MDVR-KCL cannot be matched and is reported separately rather than quietly dropped or
silently included.

WHAT IS COMPARED. For every source corpus, the same experiment is run twice: once on the
full cohort and once on an age-matched one, with everything else held fixed. The question
is not whether the age-matched score is high, it is whether it MOVES.

Run:  .venv\\Scripts\\python.exe -u scripts\\speech_agematch.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from pdvoice.calibration import select_best_by_brier
from pdvoice.cohort import age_match_ids
from pdvoice.config import EMBEDDINGS, LOGS, SEED, VAL_FRACTION
from pdvoice.data import LOADERS
from pdvoice.data.demographics import age_gap, attach_age
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auroc, bootstrap_ci
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

MODEL, LAYER, TASK = "xls-r-300m", 15, "speech"
CORPORA = ["neurovoz", "ipvs", "mdvr_kcl"]


def load(ds):
    """Manifest + cached embeddings + age, aligned. Embeddings are positional, so the
    manifest must be rebuilt exactly as it was when the cache was written: filter the
    task, then quality control, in that order."""
    df = LOADERS[ds]()
    df = df[df.task_kind == TASK].copy()
    df, _ = quality_control(df)
    df = df.reset_index(drop=True)
    f = EMBEDDINGS / f"{ds}__{MODEL}__{TASK}__L{LAYER}.npz"
    X = np.load(f)["X"].astype(np.float32)
    assert len(X) == len(df), f"{ds}: {len(X)} embeddings vs {len(df)} rows"
    return attach_age(df, ds), X


def subject_auroc(df, p):
    g = (pd.DataFrame({"s": df.subject_id.to_numpy(), "y": df.label.astype(int).to_numpy(),
                       "p": p}).groupby("s").agg(y=("y", "first"), p=("p", "mean")))
    return bootstrap_ci(g.y.to_numpy(), g.p.to_numpy())


def train_locked(X, y, groups, seed):
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION,
                                    random_state=seed).split(X, y, groups))
    head, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va], seed=seed)
    _, cal = select_best_by_brier(vlog, y[va])
    return head, cal


def score(head, cal, X):
    dev = next(head.parameters()).device
    with torch.no_grad():
        lg = head(torch.tensor(X, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
    return cal.predict_proba(lg)


def within_cv(df, X, y, seed):
    dfa = assign_folds(df.reset_index(drop=True), n_splits=5, seed=seed)
    oof = np.zeros(len(y))
    for k in range(5):
        tr, va, te = train_val_test_indices(dfa, test_fold=k, seed=seed)
        m, _ = fit_mlp_head(X[tr], y[tr], X[va], y[va], seed=seed)
        with torch.no_grad():
            dev = next(m.parameters()).device
            oof[te] = torch.sigmoid(
                m(torch.tensor(X[te], dtype=torch.float32, device=dev)).squeeze(-1)
            ).cpu().numpy()
    return subject_auroc(dfa, oof)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    seeds = [int(s) for s in args.seeds.split(",")]

    data = {ds: load(ds) for ds in CORPORA}

    print("=== Does connected-speech transfer survive age matching? ===\n")
    print("AGE GAPS (one row per speaker)")
    gaps = {}
    for ds, (df, _) in data.items():
        g = age_gap(df)
        gaps[ds] = g
        n = df.subject_id.nunique()
        if g["n_with_age"] == 0:
            print(f"  {ds:<10} {n:>3} speakers  --  NO AGE DATA, cannot be matched")
        else:
            print(f"  {ds:<10} {n:>3} speakers, {g['n_with_age']} with age  |  "
                  f"PD {g['age_pd']} / HC {g['age_hc']}  |  gap {g['age_gap']:+.1f} yr")
    print()

    rows = []
    for src in CORPORA:
        df_s, X_s = data[src]
        # Age matching also HALVES the cohort, so a drop could be the matching or could
        # just be the lost data. The third arm removes the same number of speakers at
        # random, keeping the age gap intact. Only the difference between arms two and
        # three is attributable to age.
        n_matched = None
        for cohort in ("full", "age-matched", "size-matched random"):
            if cohort == "age-matched":
                sub = df_s.dropna(subset=["age"])
                if sub.empty:
                    continue                       # MDVR-KCL: nothing to match on
                keep = age_match_ids(sub, seed=SEED)
                mask = df_s.subject_id.isin(keep).to_numpy()
                n_matched = df_s[mask].subject_id.nunique()
            elif cohort == "size-matched random":
                if n_matched is None:
                    continue
                ids = np.sort(df_s.subject_id.unique())
                rng = np.random.default_rng(SEED)
                keep = set(rng.choice(ids, n_matched, replace=False).tolist())
                mask = df_s.subject_id.isin(keep).to_numpy()
            else:
                mask = np.ones(len(df_s), bool)

            d, X = df_s[mask].reset_index(drop=True), X_s[mask]
            y = d.label.astype(int).to_numpy()
            if d.subject_id.nunique() < 20 or len(np.unique(y)) < 2:
                print(f"  [{src} {cohort}] too few speakers after matching, skipped")
                continue

            g = age_gap(d)
            n_sp = d.subject_id.nunique()
            for seed in seeds:
                w, wlo, whi = within_cv(d, X, y, seed)
                head, cal = train_locked(X, y, d.subject_id.to_numpy(), seed)
                rec = dict(source=src, cohort=cohort, seed=seed, n_speakers=n_sp,
                           source_age_gap=g.get("age_gap"), within=round(w, 4))
                for tgt in CORPORA:
                    if tgt == src:
                        continue
                    df_t, X_t = data[tgt]
                    a, _, _ = subject_auroc(df_t, score(head, cal, X_t))
                    rec[f"-> {tgt}"] = round(a, 4)
                rows.append(rec)
            print(f"  [{src:<9} {cohort:<11}] {n_sp:>3} speakers, gap "
                  f"{g.get('age_gap', float('nan')):+.1f} yr", flush=True)

    R = pd.DataFrame(rows)
    num = [c for c in R.columns if c == "within" or c.startswith("-> ")]
    agg = R.groupby(["source", "cohort"])[num].agg(["mean", "std"]).round(3)

    print(f"\n{'=' * 92}\nSPEECH TRANSFER, FULL vs AGE-MATCHED SOURCE "
          f"(mean +/- SD over {len(seeds)} seeds)\n{'=' * 92}")
    print(agg.to_string())

    print(f"\n{'=' * 92}\nREADING\n{'=' * 92}")
    for src in CORPORA:
        s = R[R.source == src]
        if s.cohort.nunique() < 2:
            print(f"  {src:<10} age-matched arm not possible (no age data)")
            continue
        f_ = s[s.cohort == "full"]
        m_ = s[s.cohort == "age-matched"]
        r_ = s[s.cohort == "size-matched random"]
        tcols = [c for c in num if c.startswith("-> ")]

        def delta(a, b):
            return (b.within.mean() - a.within.mean(),
                    np.nanmean([b[c].mean() - a[c].mean() for c in tcols]))

        dw_m, dt_m = delta(f_, m_)
        print(f"  {src:<10} full {int(f_.n_speakers.iloc[0])} speakers "
              f"-> {int(m_.n_speakers.iloc[0])}")
        print(f"             age-matched vs full      within {dw_m:+.3f}   transfer {dt_m:+.3f}")
        if len(r_):
            dw_r, dt_r = delta(f_, r_)
            print(f"             RANDOM same size vs full within {dw_r:+.3f}   "
                  f"transfer {dt_r:+.3f}   <- the cost of losing data alone")
            print(f"             ATTRIBUTABLE TO AGE      within {dw_m - dw_r:+.3f}   "
                  f"transfer {dt_m - dt_r:+.3f}")
    print("\n  A confound shows up the same way it did for vowels: within DOWN, transfer UP.")
    print("  If both barely move, the speech result was not resting on age.")

    LOGS.mkdir(parents=True, exist_ok=True)
    R.to_csv(LOGS / "speech_agematch.csv", index=False)
    (LOGS / "speech_agematch.json").write_text(json.dumps({
        "model": MODEL, "layer": LAYER, "task": TASK, "seeds": seeds,
        "age_gaps": gaps, "runs": R.to_dict("records"),
    }, indent=2, default=str))
    print(f"\nwrote {LOGS / 'speech_agematch.json'}")


if __name__ == "__main__":
    main()
