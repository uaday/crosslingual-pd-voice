"""End-to-end DATA INVENTORY: what we actually have, before any modelling.

Answers, per dataset: how many recordings/subjects, PD vs HC at both levels, which
tasks, the TRUE native sample rate / bit depth / channels (not the declared one),
and duration statistics. Writes a tidy CSV plus a per-recording probe cache so the
figures + notebook can reuse it without re-probing 115k files.

Usage:
    .venv\\Scripts\\python.exe -u scripts\\data_inventory.py [--mpower-sample 4000]
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
import soundfile as sf

from pdvoice.config import LOGS
from pdvoice.data.ipvs import load_ipvs
from pdvoice.data.mdvr_kcl import load_mdvr_kcl
from pdvoice.data.mpower import load_mpower
from pdvoice.data.neurovoz import load_neurovoz

LOADERS = {
    "mpower": load_mpower,
    "neurovoz": load_neurovoz,
    "ipvs": load_ipvs,
    "mdvr_kcl": load_mdvr_kcl,
}


def probe(paths: list[str]) -> pd.DataFrame:
    """Read true header info (no decode) for a list of audio files."""
    rows = []
    for p in paths:
        try:
            i = sf.info(p)
            rows.append(
                {
                    "path": p,
                    "true_sr": i.samplerate,
                    "subtype": i.subtype,
                    "channels": i.channels,
                    "duration_s": i.frames / float(i.samplerate),
                }
            )
        except Exception as e:  # noqa: BLE001 — unreadable file is itself a finding
            rows.append({"path": p, "true_sr": np.nan, "subtype": f"ERROR:{type(e).__name__}",
                         "channels": np.nan, "duration_s": np.nan})
    # one probe row per path: the manifest is the left side of every merge below, so a
    # duplicated probe row would multiply manifest rows instead of annotating them.
    return pd.DataFrame(rows).drop_duplicates(subset="path")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpower-sample", type=int, default=4000,
                    help="probe only N random mPower files (0 = all 115k, slow)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    manifests, probes = {}, {}
    for name, loader in LOADERS.items():
        print(f"[{name}] loading manifest ...", flush=True)
        try:
            m = loader()
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] LOADER FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        manifests[name] = m
        paths = list(m["path"])
        if name == "mpower" and args.mpower_sample and len(paths) > args.mpower_sample:
            paths = random.sample(paths, args.mpower_sample)
            print(f"[{name}] probing random {len(paths):,} of {len(m):,} files", flush=True)
        else:
            print(f"[{name}] probing all {len(paths):,} files", flush=True)
        probes[name] = probe(paths)

    # ---------------- table 1: dataset-level inventory ----------------
    rows = []
    for name, m in manifests.items():
        pr = probes[name]
        joined = m.merge(pr, on="path", how="inner")
        sub = m.groupby("subject_id")["label"].first()
        rows.append({
            "dataset": name,
            "language": m["language"].mode().iat[0] if len(m) else "-",
            "recordings": len(m),
            "subjects": m["subject_id"].nunique(),
            "subj_PD": int((sub == 1).sum()),
            "subj_HC": int((sub == 0).sum()),
            "subj_unlabelled": int(sub.isna().sum()),
            "rec_PD": int((m["label"] == 1).sum()),
            "rec_HC": int((m["label"] == 0).sum()),
            "recs_per_subj": round(len(m) / max(1, m["subject_id"].nunique()), 1),
            "tasks": ",".join(sorted(m["task_kind"].dropna().unique())),
            "declared_sr": ",".join(str(int(v)) for v in sorted(m["native_sr"].dropna().unique())),
            "TRUE_sr": ",".join(f"{int(k)}x{v}" for k, v in
                                joined["true_sr"].value_counts().sort_index().items()),
            "bit_depth": ",".join(sorted(joined["subtype"].dropna().unique())),
            "channels": ",".join(str(int(v)) for v in sorted(joined["channels"].dropna().unique())),
            "dur_median_s": round(float(joined["duration_s"].median()), 2),
            "dur_p05_s": round(float(joined["duration_s"].quantile(0.05)), 2),
            "dur_p95_s": round(float(joined["duration_s"].quantile(0.95)), 2),
            "dur_total_h": round(float(joined["duration_s"].sum()) / 3600, 2),
            "probed_n": len(joined),
        })
    inv = pd.DataFrame(rows)

    # ---------------- table 2: task-level breakdown ----------------
    trows = []
    for name, m in manifests.items():
        j = m.merge(probes[name], on="path", how="inner")
        for (task, code), g in j.groupby(["task_kind", "task_code"], dropna=False):
            trows.append({
                "dataset": name, "task_kind": task, "task_code": code,
                "recordings": len(g), "subjects": g["subject_id"].nunique(),
                "rec_PD": int((g["label"] == 1).sum()), "rec_HC": int((g["label"] == 0).sum()),
                "dur_median_s": round(float(g["duration_s"].median()), 2),
                "dur_total_min": round(float(g["duration_s"].sum()) / 60, 1),
            })
    tasks = pd.DataFrame(trows).sort_values(["dataset", "task_kind", "task_code"])

    # ---------------- table 3: declared vs TRUE sample rate ----------------
    mrows = []
    for name, m in manifests.items():
        j = m.merge(probes[name], on="path", how="inner")
        bad = j[j["native_sr"] != j["true_sr"]]
        mrows.append({
            "dataset": name, "probed": len(j), "sr_mismatch": len(bad),
            "pct_wrong": round(100 * len(bad) / max(1, len(j)), 1),
            "declared": ",".join(str(int(v)) for v in sorted(j["native_sr"].dropna().unique())),
            "true": ",".join(str(int(v)) for v in sorted(j["true_sr"].dropna().unique())),
        })
    mism = pd.DataFrame(mrows)

    # ---------------- table 4: does acquisition correlate with label? ----------------
    crows = []
    for name, m in manifests.items():
        j = m.merge(probes[name], on="path", how="inner")
        s = j.groupby("subject_id").agg(label=("label", "first"), sr=("true_sr", "max"),
                                        subtype=("subtype", "first"))
        s = s.dropna(subset=["label"])
        for field in ["sr", "subtype"]:
            if s[field].nunique() < 2:
                continue
            ct = pd.crosstab(s[field], s["label"])
            pure = int(sum(1 for _, r in ct.iterrows() if (r == 0).any()))
            crows.append({"dataset": name, "field": field, "n_levels": s[field].nunique(),
                          "levels_pure_one_class": pure,
                          "crosstab": ct.to_dict()})
    conf = pd.DataFrame(crows)

    pd.set_option("display.width", 250, "display.max_columns", 60)
    for title, t in [("DATASET INVENTORY", inv), ("TASK BREAKDOWN", tasks),
                     ("DECLARED vs TRUE SAMPLE RATE", mism),
                     ("ACQUISITION x LABEL (subject level)", conf)]:
        print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
        print(t.to_string(index=False) if len(t) else "  (nothing to report)")

    LOGS.mkdir(parents=True, exist_ok=True)
    inv.to_csv(LOGS / "inventory__datasets.csv", index=False)
    tasks.to_csv(LOGS / "inventory__tasks.csv", index=False)
    mism.to_csv(LOGS / "inventory__sr_mismatch.csv", index=False)
    for name, pr in probes.items():
        pr.to_csv(LOGS / f"inventory__probe__{name}.csv", index=False)
    print(f"\nwrote inventory__*.csv to {LOGS}")


if __name__ == "__main__":
    main()
