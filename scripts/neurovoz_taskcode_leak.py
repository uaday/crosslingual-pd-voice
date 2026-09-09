"""Do NeuroVoz's vowel task codes leak the diagnosis?  (a corpus-artefact audit)

NeuroVoz names every vowel file `<STATUS>_<TASK>_<ID>.wav`, where TASK is a vowel letter
plus a repetition number: A1/A2/A3 are the first, second and third sustained /a/. The
protocol is five vowels x three repetitions = 15 recordings per speaker.

The delivered data is not that. The THIRD repetition survives almost only for patients --
A3 is 29 PD / 3 HC, E3 24/2, I3 26/1, O3 28/2 -- while U2 runs the other way and U3 has a
single recording in the whole corpus. Controls average ~8.6 vowel recordings, patients
~10.7.

So the label is partly recoverable from the FILE NAMES, with no audio at all. This script
measures how much, three ways:

    task-code profile   which codes a speaker has, scored by each code's PD share
    recording count     nothing but how many vowel files the speaker contributed
    per-code imbalance  the PD share of each individual code

This is the same class of problem as the IPVS 44.1 kHz batch (F-16): a corpus artefact that
correlates with the label, invisible to a permutation test because it is a property of the
data rather than of the plumbing. It matters for any WITHIN-NeuroVoz vowel result -- ours
(`transfer_any.py --train neurovoz-vowel`, 0.610) and the published ones alike.

Run:  .venv\\Scripts\\python.exe -u scripts\\neurovoz_taskcode_leak.py
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from pdvoice.config import LOGS
from pdvoice.data.neurovoz import load_neurovoz
from pdvoice.eval.discrimination import auroc, bootstrap_ci


def main() -> None:
    warnings.simplefilter("ignore")
    nv = load_neurovoz()
    v = nv[nv.task_kind == "vowel"].copy()

    # ---- the per-code table ---------------------------------------------------
    tab = (v.pivot_table(index="task_code", columns="label", values="path", aggfunc="count")
             .fillna(0).astype(int))
    tab.columns = ["HC", "PD"]
    tab = tab[["PD", "HC"]]
    tab["total"] = tab.PD + tab.HC
    tab["pd_share"] = (tab.PD / tab.total).round(3)

    print("=" * 78)
    print("NEUROVOZ VOWEL TASK CODES — letter = vowel, digit = which repetition")
    print("=" * 78)
    print(tab.to_string())

    # ---- leak 1: the task-code profile ----------------------------------------
    # Score each recording by its code's PD share, then average per speaker. This is the
    # most a model could read off the code alone; it uses no audio.
    share = v.groupby("task_code").label.mean()
    v["p_code"] = v.task_code.map(share)
    g = v.groupby("subject_id").agg(y=("label", "first"), p=("p_code", "mean"),
                                    n=("path", "size"))
    # bootstrap_ci returns (point, lo, hi) -- take the point from it too, so the
    # reported AUROC and its interval can never come from different computations.
    a_code, lo_c, hi_c = bootstrap_ci(g.y.to_numpy(), g.p.to_numpy())

    # ---- leak 2: the recording count, and nothing else ------------------------
    a_count, lo_n, hi_n = bootstrap_ci(g.y.to_numpy(), g.n.to_numpy().astype(float))

    print("\n" + "=" * 78)
    print("HOW MUCH OF THE DIAGNOSIS IS IN THE METADATA?  (subject level, no audio)")
    print("=" * 78)
    print(f"  task-code profile      AUROC {a_code:.3f}  95% CI [{lo_c:.3f}, {hi_c:.3f}]")
    print(f"  recording count alone  AUROC {a_count:.3f}  95% CI [{lo_n:.3f}, {hi_n:.3f}]")
    print(f"  recording-level, code as the only feature: AUROC {auroc(v.label, v.p_code):.3f}")

    counts = g.groupby("y").n.agg(["count", "mean", "min", "max"]).round(2)
    counts.index = ["HC", "PD"]
    print("\n  vowel recordings per speaker:")
    print(counts.to_string())

    third = tab.loc[[c for c in tab.index if c.endswith("3")]]
    print(f"\n  third repetitions only: {int(third.PD.sum())} PD vs {int(third.HC.sum())} HC"
          f"  ({100 * third.PD.sum() / third.total.sum():.0f}% patients)")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print("  A speaker's task codes predict Parkinson's better than the audio model does")
    print("  on the same corpus. The third repetition of each vowel was kept mostly for")
    print("  patients, so 'which takes exist' is close to a label.")
    print("\n  CONSEQUENCE: never train within NeuroVoz on pooled vowels without balancing")
    print("  by task code. Transfer INTO NeuroVoz is unaffected -- a locked external model")
    print("  never sees the code -- and the /a/-only control confirms it is not inflating")
    print("  those numbers.")

    LOGS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(LOGS / "neurovoz_taskcode_leak__per_code.csv")
    (LOGS / "neurovoz_taskcode_leak.json").write_text(json.dumps({
        "auroc_taskcode_profile": round(float(a_code), 4),
        "ci_taskcode_profile": [round(float(lo_c), 4), round(float(hi_c), 4)],
        "auroc_recording_count": round(float(a_count), 4),
        "ci_recording_count": [round(float(lo_n), 4), round(float(hi_n), 4)],
        "auroc_recording_level": round(float(auroc(v.label, v.p_code)), 4),
        "n_speakers": int(g.shape[0]),
        "n_pd": int((g.y == 1).sum()), "n_hc": int((g.y == 0).sum()),
        "recordings_per_speaker": {"HC": float(counts.loc["HC", "mean"]),
                                   "PD": float(counts.loc["PD", "mean"])},
        "third_repetition_pd_share": round(float(third.PD.sum() / third.total.sum()), 4),
        "per_code": tab.reset_index().to_dict("records"),
    }, indent=2))
    print(f"\nwrote {LOGS / 'neurovoz_taskcode_leak.json'}")


if __name__ == "__main__":
    main()
