"""Recompute stored ECE/MCE with the tie-aware binning (2026-11-09), without retraining.

`eval.metrics.calibration_error` used to cut the sorted predictions at fixed row positions,
so identical probabilities could be split across bins with different label mixes. The fix
keeps ties in one bin. Every stored ECE that has its participant-level predictions saved
next to it is recomputed here:

  * unified experiment runs   outputs/logs/unified*/runs*.csv  <- scores*.csv.gz
  * layer-1 / layer-2 results outputs/logs/layer[12]_*.json    <- *__subject_preds.csv

Safety check: for every group the OLD binning is applied first and must reproduce the
stored value (to 1e-9). Only then is the value replaced. Groups without saved predictions
(age-only and disorder-confusion rows) or that fail the check are left unchanged and
counted in the report.

Run:  python scripts/recompute_ece.py            (dry run: report only)
      python scripts/recompute_ece.py --write    (rewrite the stored values)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pdvoice.config import LOGS
from pdvoice.eval.metrics import ECE_BINS, calibration_error

KEYS = ["exp", "source", "source_task", "condition", "seed", "target", "target_task",
        "target_cohort"]


def old_ece(y, p, n_bins=ECE_BINS):
    """The pre-2026-11-09 definition, kept only to validate the stored values."""
    o = np.argsort(p)
    y, p = np.asarray(y, float)[o], np.asarray(p, float)[o]
    ece, mce = 0.0, 0.0
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b):
            gap = abs(y[b].mean() - p[b].mean())
            ece += len(b) / len(p) * gap
            mce = max(mce, gap)
    return ece, mce


def unified(write: bool) -> list[dict]:
    report = []
    for d in sorted(LOGS.glob("unified*")):
        if not d.is_dir():
            continue
        for runs_f in sorted(d.glob("runs*.csv")):
            stem = runs_f.stem.replace("runs", "scores", 1)
            if runs_f.stem in ("runs_partial", "runs_within"):
                stem = "scores"
            sc_f = d / f"{stem}.csv.gz"
            if not sc_f.exists():
                continue
            R = pd.read_csv(runs_f, float_precision="round_trip")
            S = pd.read_csv(sc_f, float_precision="round_trip")
            groups = {k: g for k, g in S.groupby(KEYS, dropna=False)}
            n_new = n_skip = n_bad = 0
            max_change = 0.0
            for i, row in R.iterrows():
                g = groups.get(tuple(row[k] for k in KEYS))
                if g is None or "ece" not in R.columns or pd.isna(row["ece"]):
                    n_skip += 1
                    continue
                o, _ = old_ece(g.y.to_numpy(), g.p.to_numpy())
                if abs(o - row["ece"]) > 1e-9:
                    n_bad += 1
                    continue
                new = calibration_error(g.y.to_numpy(), g.p.to_numpy())["ece"]
                max_change = max(max_change, abs(new - row["ece"]))
                R.at[i, "ece"] = new
                n_new += 1
            if write and n_new:
                R.to_csv(runs_f, index=False)
            report.append({"file": str(runs_f.relative_to(LOGS)), "recomputed": n_new,
                           "no_predictions": n_skip, "failed_check": n_bad,
                           "max_abs_change": round(max_change, 4)})
    return report


def layer_jsons(write: bool) -> list[dict]:
    report = []
    for f in sorted(LOGS.glob("layer[12]_*.json")):
        pf = f.with_name(f.stem + "__subject_preds.csv")
        if not pf.exists():
            continue
        d = json.loads(f.read_text())
        if "ece" not in d:
            continue
        P = pd.read_csv(pf, float_precision="round_trip")
        o, om = old_ece(P.y.to_numpy(), P.p.to_numpy())
        # stored JSON values are rounded in some scripts, so compare at their precision
        ok = abs(o - d["ece"]) < 5e-4
        entry = {"file": f.name, "old": d["ece"], "check": ok}
        if ok:
            c = calibration_error(P.y.to_numpy(), P.p.to_numpy())
            entry["new"] = round(c["ece"], 4)
            if write:
                d["ece"], d["mce"] = round(c["ece"], 4), round(c["mce"], 4)
                f.write_text(json.dumps(d, indent=2))
        report.append(entry)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    u = unified(a.write)
    for r in u:
        print(r)
    L = layer_jsons(a.write)
    changed = [r for r in L if r.get("check") and abs(r["new"] - r["old"]) >= 5e-4]
    print(f"\nlayer JSONs: {len(L)} with predictions, {sum(r['check'] for r in L)} pass the "
          f"check, {len(changed)} change at 3 dp")
    for r in changed:
        print(f"  {r['file']}: {r['old']} -> {r['new']}")
    for r in L:
        if not r["check"]:
            print(f"  CHECK FAILED (left unchanged): {r['file']}")
    if not a.write:
        print("\n(dry run; pass --write to update the files)")


if __name__ == "__main__":
    main()
