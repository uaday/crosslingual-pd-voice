"""I3 -- the vowel transfer confounds LANGUAGE with PHONEME. Separate them.

mPower records one thing: a sustained /a/. NeuroVoz and IPVS record all five vowels.
Every experiment in this project filters on `task_kind` and never on `task_code`, so
"mPower -> NeuroVoz vowel transfer" actually trains on /a/ and tests on /a,e,i,o,u/
pooled. Different vowels have entirely different formant structure, so part of whatever
drop we attribute to LANGUAGE may simply be PHONEME mismatch.

The fix is free: the external caches are row-aligned to their manifests, so /a/ can be
selected by row without re-extracting anything. NeuroVoz /a/ is task_code A1/A2/A3,
IPVS /a/ is VA.

If /a/-only transfer is clearly better than all-vowels transfer, then some of the
reported cross-language gap was never linguistic, and the honest comparison is the
matched-phoneme one.

Run:  .venv\\Scripts\\python.exe -u scripts\\vowel_phoneme_control.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch

from pdvoice.calibration import select_best_by_brier
from pdvoice.cohort import age_match_ids, load_mpower_cohort
from pdvoice.config import EMBEDDINGS, LOGS, VAL_FRACTION
from pdvoice.data import LOADERS
from pdvoice.eval.discrimination import auroc, bootstrap_ci
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control
from sklearn.model_selection import GroupShuffleSplit

MODEL, LAYER = "xls-r-300m", 15
# which task codes are the /a/ vowel in each corpus
A_ONLY = {"neurovoz": lambda c: c.str.match(r"^A\d"), "ipvs": lambda c: c.eq("VA")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    seeds = [int(s) for s in args.seeds.split(",")]

    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"subject_id": d["subject_id"].astype(str)})
    emb["row"] = np.arange(len(emb))
    base = load_mpower_cohort(cap=3, seed=42).merge(emb, on="subject_id", how="inner")

    # externals, with a mask marking the /a/ rows
    ext = {}
    for ds in A_ONLY:
        edf = LOADERS[ds]()
        edf = edf[edf.task_kind == "vowel"].copy()
        edf, _ = quality_control(edf)
        edf = edf.reset_index(drop=True)
        E = np.load(EMBEDDINGS / f"{ds}__{MODEL}__vowel__L{LAYER}.npz")["X"].astype(np.float32)
        assert len(E) == len(edf), f"{ds}: {len(E)} != {len(edf)}"
        m_a = A_ONLY[ds](edf.task_code.astype(str)).to_numpy()
        ext[ds] = (edf, E, m_a)
        codes = sorted(edf.task_code.unique())
        print(f"{ds}: {len(edf):,} vowel recordings, codes {codes}")
        print(f"   /a/ only -> {int(m_a.sum()):,} recordings "
              f"({edf[m_a].subject_id.nunique()} speakers)")
    print()

    rows = []
    for seed in seeds:
        keep = age_match_ids(base, seed=seed)
        sub = base[base.subject_id.isin(keep)]
        X = d["X"][sub.row.to_numpy()].astype(np.float32)
        y = sub.label.to_numpy()
        tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed)
                      .split(X, y, sub.subject_id))
        head, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va], seed=seed)
        _, cal = select_best_by_brier(vlog, y[va])
        dev = next(head.parameters()).device

        def score(E):
            with torch.no_grad():
                lg = head(torch.tensor(E, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
            return cal.ranking(lg)  # discrimination: strictly increasing map, no ties (RankingMap)

        for ds, (edf, E, m_a) in ext.items():
            p = score(E)
            # A CONFOUND IN THIS COMPARISON: "all vowels" averages about 5x more
            # recordings per speaker than "/a/ only" (IPVS: 10 vs 2), and averaging more
            # recordings gives a cleaner subject-level estimate regardless of phoneme.
            # The extra conditions equalise that by keeping exactly ONE recording per
            # speaker in both, so the only thing that differs is which phoneme was spoken.
            rng = np.random.default_rng(seed)

            def one_each(mask):
                order = rng.permutation(np.where(mask)[0])
                seen, keep_i = set(), []
                for i in order:
                    sid = edf.subject_id.iloc[i]
                    if sid not in seen:
                        seen.add(sid)
                        keep_i.append(i)
                out = np.zeros(len(edf), bool)
                out[keep_i] = True
                return out

            conditions = [("all vowels", np.ones(len(edf), bool)),
                          ("/a/ only", m_a),
                          ("all vowels, 1 per speaker", one_each(np.ones(len(edf), bool))),
                          ("/a/ only, 1 per speaker", one_each(m_a))]
            for label, m in conditions:
                g = pd.DataFrame({"s": edf.subject_id[m], "y": edf.label[m].astype(int),
                                  "p": p[m]}).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
                rows.append(dict(seed=seed, target=ds, subset=label,
                                 n_speakers=len(g), auroc=round(auroc(g.y, g.p), 4)))

    R = pd.DataFrame(rows)
    R.to_csv(LOGS / "vowel_phoneme_control.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"{'=' * 84}\nmPower (/a/-trained) -> external vowels, {len(seeds)} seeds\n{'=' * 84}")
    agg = R.groupby(["target", "subset"]).auroc.agg(["mean", "std", "count"]).round(4)
    print(agg.to_string())

    print(f"\n{'=' * 84}\nDOES MATCHING THE PHONEME CHANGE THE PICTURE?\n{'=' * 84}")
    for ds in R.target.unique():
        a = R[(R.target == ds) & (R.subset == "all vowels")].auroc
        b = R[(R.target == ds) & (R.subset == "/a/ only")].auroc
        print(f"  {ds:<10} all vowels {a.mean():.3f} +/- {a.std():.3f}   "
              f"/a/ only {b.mean():.3f} +/- {b.std():.3f}   delta {b.mean() - a.mean():+.3f}")
    print()
    print("  with recordings-per-speaker EQUALISED (1 each, so only the phoneme differs):")
    for ds in R.target.unique():
        a1 = R[(R.target == ds) & (R.subset == "all vowels, 1 per speaker")].auroc
        b1 = R[(R.target == ds) & (R.subset == "/a/ only, 1 per speaker")].auroc
        print(f"    {ds:<10} all vowels {a1.mean():.3f}   /a/ only {b1.mean():.3f}   "
              f"delta {b1.mean() - a1.mean():+.3f}")

    overall = (R[R.subset == "/a/ only, 1 per speaker"].auroc.mean()
               - R[R.subset == "all vowels, 1 per speaker"].auroc.mean())
    print(f"\n  mean change from matching the phoneme: {overall:+.4f}")
    if abs(overall) < 0.02:
        print("  -> Phoneme mismatch is NOT a meaningful part of the cross-language gap.")
        print("     Pooling all five vowels is defensible; state it and move on.")
    elif overall > 0:
        print("  -> Some of the reported cross-language gap was PHONEME mismatch, not")
        print("     language. The matched-phoneme number is the honest comparison.")
    else:
        print("  -> Restricting to /a/ makes transfer WORSE, which pooling was masking.")

    (LOGS / "vowel_phoneme_control.json").write_text(
        json.dumps({"rows": rows, "mean_delta": round(float(overall), 4)}, indent=2))
    print(f"\nwrote {LOGS / 'vowel_phoneme_control.csv'}")


if __name__ == "__main__":
    main()
