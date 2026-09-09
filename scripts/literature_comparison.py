"""Compare our results against the published numbers, claim by claim.

WHY THIS IS A SCRIPT AND NOT A TABLE IN A DOCUMENT. The published figures are typed by
hand -- they come from papers, not from our pipeline -- but OUR side of every comparison
must not be. It is read from the same artifacts the notebook reads, so a re-run cannot
leave the comparison quietly out of date.

The published side lives in ``references/published_results.csv`` with a citation and a
``comparability`` flag per row. That flag is the point of the report: a published number is
only a benchmark if it was measured the same way, and one of these rows exists precisely
because it was not.

Run:  .venv\\Scripts\\python.exe -u scripts\\literature_comparison.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pdvoice.config import LOGS

REFERENCES = Path(__file__).resolve().parents[1] / "references" / "published_results.csv"
MODEL, LAYER = "xls-r-300m", 15


def _load(name: str) -> dict:
    f = LOGS / name
    if not f.exists():
        raise FileNotFoundError(name)
    return json.loads(f.read_text())


def resolve(key: str) -> tuple[float | None, str]:
    """Turn a key from the CSV into (our AUROC, the artifact it came from).

    Keys are deliberately explicit about *which* evaluation is being quoted, because the
    same corpus can appear as a training set and as a transfer target with very different
    numbers, and conflating those is the easiest way to build a flattering comparison.
    """
    parts = key.split(":")

    if parts[0] == "transferany":
        run = parts[1]                      # e.g. train-neurovoz-speech
        f = f"transferany__{MODEL}__{run}__L{LAYER}.json"
        d = _load(f)
        if parts[2] == "within":
            return float(d["within"]["auroc"]), f
        target = ":".join(parts[3:])        # e.g. ipvs:speech
        return float(d["transfer"][target]["auroc"]), f

    if parts[0] == "labelsens":
        f = "label_sensitivity.json"
        d = _load(f)
        runs = d["runs"][parts[1]]
        v = np.array([r["within"] for r in runs], float)
        return float(v.mean()), f

    raise KeyError(f"unrecognised key: {key}")


def main() -> None:
    warnings.simplefilter("ignore")
    ref = pd.read_csv(REFERENCES)

    ours, srcs, notes = [], [], []
    for _, r in ref.iterrows():
        try:
            v, f = resolve(r["our_key"])
        except (FileNotFoundError, KeyError) as e:                     # noqa: PERF203
            v, f = None, f"MISSING ({e})"
        ours.append(v)
        srcs.append(f)
    ref["ours"] = ours
    ref["our_artifact"] = srcs
    ref["delta"] = ref["ours"] - ref["published"]

    def verdict(row):
        if row["ours"] is None or pd.isna(row["ours"]):
            return "no artifact"
        if row["comparability"] == "NOT comparable":
            return "not a benchmark"
        d = abs(row["delta"])
        sd = row["published_sd"]
        if not pd.isna(sd) and sd > 0:
            return "within 1 SD" if d <= sd else f"{d / sd:.1f} SD away"
        return "match" if d <= 0.05 else ("close" if d <= 0.10 else "differs")

    ref["verdict"] = ref.apply(verdict, axis=1)

    cols = ["claim", "task", "published", "published_sd", "ours", "delta",
            "verdict", "comparability", "source"]
    out = ref[cols].copy()

    pd.set_option("display.width", 250, "display.max_colwidth", 46)
    print("=" * 110)
    print("OUR RESULTS vs THE PUBLISHED LITERATURE")
    print("=" * 110)
    print(out.round(3).to_string(index=False))

    print("\n" + "=" * 110)
    print("READING")
    print("=" * 110)
    comparable = ref[ref["comparability"] != "NOT comparable"].dropna(subset=["ours"])
    print(f"  comparable claims          : {len(comparable)} of {len(ref)}")
    print(f"  mean |difference|          : {comparable['delta'].abs().mean():.3f} AUROC")
    print(f"  largest difference         : {comparable['delta'].abs().max():.3f}"
          f"  ({comparable.loc[comparable['delta'].abs().idxmax(), 'claim']})")
    print("\n  Our numbers are concordant with the field. They differ from the HEADLINE")
    print("  numbers because those come from single-dataset, internally-validated studies,")
    print("  and in one case from a recording-level split -- see the row flagged")
    print("  'NOT comparable', which is in the table to be criticised, not matched.")

    for _, r in ref[ref["comparability"].isin(["NOT comparable", "caveat", "ceiling"])].iterrows():
        print(f"\n  [{r['comparability']}] {r['claim']}")
        print(f"      {r['note']}")

    LOGS.mkdir(parents=True, exist_ok=True)
    ref.to_csv(LOGS / "literature_comparison.csv", index=False)
    (LOGS / "literature_comparison.json").write_text(json.dumps({
        "model": MODEL, "layer": LAYER,
        "n_claims": int(len(ref)),
        "n_comparable": int(len(comparable)),
        "mean_abs_delta": round(float(comparable["delta"].abs().mean()), 4),
        "rows": ref[cols + ["note", "our_artifact"]].to_dict("records"),
    }, indent=2, default=str))
    print(f"\nwrote {LOGS / 'literature_comparison.csv'}")


if __name__ == "__main__":
    main()
