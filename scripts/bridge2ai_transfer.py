"""Cross-dataset transfer with Bridge2AI-Voice (English, clinician-confirmed PD).

Same protocol as the rest of the thesis: XLS-R layer-15 mean+std embeddings, MLP head on a
participant-disjoint train/validation split, calibration chosen by validation Brier score,
model LOCKED, then applied unchanged to each target. Participant-level AUROC, five seeds.

Directions
  out of Bridge2AI  (source uncorrected or age-matched, PD vs healthy controls)
      vowel  -> IPVS, NeuroVoz, mPower (vowel)
      ddk    -> NeuroVoz
      speech -> IPVS, NeuroVoz, MDVR-KCL
  into Bridge2AI    (source mPower vowel, uncorrected or age-matched)
      vowel  -> Bridge2AI vowel, PD vs healthy controls

Targets are scored as published (their own age structure is part of what transfers);
the Bridge2AI and mPower targets are additionally scored on their age-matched subsets.

Run:  .venv\\Scripts\\python.exe -u scripts\\bridge2ai_transfer.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from demographic_erasure import load_external, score, subject_auroc, train_locked  # noqa: E402

from pdvoice.cohort import age_match_ids, load_mpower_cohort  # noqa: E402
from pdvoice.config import EMBEDDINGS, LOGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402

MODEL = "xls-r-300m"
LAYER = int(os.environ.get("PDV_LAYER", 15))  # Bridge2AI layer; 8 is chosen inside Bridge2AI (bridge2ai_layer_select.py)
SUFFIX = "" if LAYER == 15 else f"__L{LAYER}"
TARGETS = {"vowel": [("ipvs", "vowel"), ("neurovoz", "vowel"), ("mpower", "vowel")],
           "ddk": [("neurovoz", "ddk")],
           "speech": [("ipvs", "speech"), ("neurovoz", "speech"), ("mdvr_kcl", "speech")]}


def bridge2ai(task, base):
    d = np.load(EMBEDDINGS / f"bridge2ai__{MODEL}__{task}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))
    df = emb.merge(base, on="path", how="inner")
    df = df[df.group.isin(["pd", "control"])].dropna(subset=["age"]).reset_index(drop=True)
    df["label"] = df.label.astype(int)
    return df, d["X"][df.row.to_numpy()].astype(np.float32)


def mpower():
    d = np.load(EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz", allow_pickle=True)
    emb = pd.DataFrame({"path": d["path"].astype(str)})
    emb["row"] = np.arange(len(emb))
    df = load_mpower_cohort(cap=3, seed=42).merge(emb, on="path", how="inner")
    assert df.row.is_unique
    df = df.dropna(subset=["age"]).reset_index(drop=True)
    return df, d["X"][df.row.to_numpy()].astype(np.float32)


def auc(df, p):
    return subject_auroc(df.subject_id.to_numpy(), df.label.astype(int).to_numpy(), p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    seeds = [int(s) for s in args.seeds.split(",")]

    base = LOADERS["bridge2ai"]()
    # mPower embeddings exist only at layers 15 and 24; at any other Bridge2AI layer the
    # mPower rows are skipped (a head trained at one layer cannot score another).
    have_mp = (EMBEDDINGS / f"mpower_full__{MODEL}__L{LAYER}.npz").exists()
    mp_df, mp_X = mpower() if have_mp else (None, None)
    ext = {k: load_external(*k) for task in TARGETS for k in TARGETS[task] if k[0] != "mpower"}
    rows = []

    # ---- out of Bridge2AI
    for task, targets in TARGETS.items():
        src, X = bridge2ai(task, base)
        for cohort in ("uncorrected", "age-matched"):
            for seed in seeds:
                m = (np.ones(len(src), bool) if cohort == "uncorrected"
                     else src.subject_id.isin(age_match_ids(src, seed=seed)).to_numpy())
                s = src[m]
                head, cal = train_locked(X[m], s.label.to_numpy(), s.subject_id.to_numpy(), seed)
                for ds, ttask in targets:
                    if ds == "mpower" and not have_mp:
                        continue
                    if ds == "mpower":
                        tm = mp_df.subject_id.isin(age_match_ids(mp_df, seed=seed)).to_numpy()
                        p = score(head, cal, mp_X)
                        rows.append(dict(direction="bridge2ai->", task=task, source=cohort, seed=seed,
                                         target="mpower", target_cohort="as published",
                                         auroc=auc(mp_df, p)))
                        rows.append(dict(direction="bridge2ai->", task=task, source=cohort, seed=seed,
                                         target="mpower", target_cohort="age-matched",
                                         auroc=auc(mp_df[tm], p[tm])))
                        continue
                    edf, E = ext[(ds, ttask)]
                    rows.append(dict(direction="bridge2ai->", task=task, source=cohort, seed=seed,
                                     target=ds, target_cohort="as published",
                                     auroc=auc(edf, score(head, cal, E))))
            print(f"  bridge2ai {task} [{cohort}] done", flush=True)

    # ---- into Bridge2AI (vowel), trained on mPower
    tgt, TX = bridge2ai("vowel", base)
    for cohort in (("uncorrected", "age-matched") if have_mp else ()):
        for seed in seeds:
            m = (np.ones(len(mp_df), bool) if cohort == "uncorrected"
                 else mp_df.subject_id.isin(age_match_ids(mp_df, seed=seed)).to_numpy())
            s = mp_df[m]
            head, cal = train_locked(mp_X[m], s.label.to_numpy(), s.subject_id.to_numpy(), seed)
            p = score(head, cal, TX)
            tm = tgt.subject_id.isin(age_match_ids(tgt, seed=seed)).to_numpy()
            for tc, mask in (("as published", np.ones(len(tgt), bool)), ("age-matched", tm)):
                rows.append(dict(direction="mpower->", task="vowel", source=cohort, seed=seed,
                                 target="bridge2ai", target_cohort=tc, auroc=auc(tgt[mask], p[mask])))
        print(f"  mpower -> bridge2ai [{cohort}] done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(LOGS / f"bridge2ai_transfer{SUFFIX}.csv", index=False)
    agg = (R.groupby(["direction", "task", "target", "target_cohort", "source"]).auroc
             .agg(["mean", "std"]).round(3).unstack("source"))
    pd.set_option("display.width", 200)
    print("\n" + agg.to_string())
    (LOGS / f"bridge2ai_transfer{SUFFIX}.json").write_text(json.dumps(
        R.groupby(["direction", "task", "target", "target_cohort", "source"]).auroc.mean()
         .round(4).reset_index().to_dict("records"), indent=2))
    print(f"\nwrote {LOGS / 'bridge2ai_transfer.csv'}")


if __name__ == "__main__":
    main()
