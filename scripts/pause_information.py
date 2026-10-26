"""How much Parkinson's signal is in the SILENCE we throw away?

THE OBJECTION THIS ANSWERS. Our pipeline runs a voice-activity detector over connected
speech and keeps only the speech regions -- on one MDVR-KCL file that discards 58% of the
recording. Parkinson's affects fluency: patients pause more often and for longer, and
hesitate mid-utterance. So the obvious criticism is that we delete a real symptom before
the model ever sees the audio.

The criticism is correct in principle. This script measures how much it costs, using only
the TIMING of speech and silence -- never the acoustic content:

    pause_frac      share of the recording that is not speech
    n_pauses_per_s  how often speech stops
    mean_pause_s    typical pause length
    longest_pause_s the worst hesitation
    speech_s        how much speech there is
    total_s         how long the recording is

If a classifier on those six numbers alone separates PD from HC, that is signal the
current pipeline is discarding, and the thesis has to say so.

Run:  .venv\\Scripts\\python.exe -u scripts\\pause_information.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdvoice.config import LOGS, SAMPLE_RATE, SEED
from pdvoice.data import LOADERS
from pdvoice.eval.discrimination import auroc, bootstrap_ci
from pdvoice.preprocess import quality_control
from pdvoice.utils.audio import load_audio

CORPORA = ["neurovoz", "ipvs", "mdvr_kcl"]
FEATURES = ["pause_frac", "n_pauses_per_s", "mean_pause_s",
            "longest_pause_s", "speech_s", "total_s"]


def timing(path, model):
    """Speech/silence timing for one recording. Returns None if VAD finds no speech."""
    import torch
    from silero_vad import get_speech_timestamps

    wav = load_audio(path, sr=SAMPLE_RATE)
    total = len(wav) / SAMPLE_RATE
    ts = get_speech_timestamps(torch.from_numpy(wav), model, sampling_rate=SAMPLE_RATE)
    if not ts:
        return None

    speech = sum(t["end"] - t["start"] for t in ts) / SAMPLE_RATE
    # gaps BETWEEN speech regions, plus leading and trailing silence
    gaps = [(ts[i + 1]["start"] - ts[i]["end"]) / SAMPLE_RATE for i in range(len(ts) - 1)]
    gaps += [ts[0]["start"] / SAMPLE_RATE, (len(wav) - ts[-1]["end"]) / SAMPLE_RATE]
    gaps = [g for g in gaps if g > 0.05]        # ignore anything under 50 ms

    return {
        "total_s": total,
        "speech_s": speech,
        "pause_frac": 1.0 - speech / total if total else np.nan,
        "n_pauses_per_s": len(gaps) / total if total else np.nan,
        "mean_pause_s": float(np.mean(gaps)) if gaps else 0.0,
        "longest_pause_s": float(np.max(gaps)) if gaps else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap recordings per corpus (debug)")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    from silero_vad import load_silero_vad
    model = load_silero_vad()

    rows = []
    for ds in CORPORA:
        df = LOADERS[ds]()
        df = df[df.task_kind == "speech"].copy()
        df, _ = quality_control(df)
        if args.limit:
            df = df.head(args.limit)
        print(f"{ds}: {len(df)} recordings ...", flush=True)
        for i, r in enumerate(df.itertuples()):
            t = timing(r.path, model)
            if t is None:
                continue
            rows.append(dict(dataset=ds, subject_id=r.subject_id,
                             label=int(r.label), **t))
            if (i + 1) % 400 == 0:
                print(f"   {i + 1}/{len(df)}", flush=True)

    R = pd.DataFrame(rows)
    # one row per speaker: a speaker's typical timing, not their noisiest recording
    S = R.groupby(["dataset", "subject_id"]).agg(
        label=("label", "first"), **{f: (f, "mean") for f in FEATURES}).reset_index()

    print(f"\n{'=' * 84}\nSPEECH vs SILENCE, per speaker\n{'=' * 84}")
    for ds in CORPORA:
        s = S[S.dataset == ds]
        if s.empty:
            continue
        g = s.groupby("label")[["total_s", "speech_s", "pause_frac", "mean_pause_s"]].mean()
        print(f"\n  {ds}  ({len(s)} speakers)")
        for lab, name in [(0, "healthy"), (1, "patient")]:
            if lab in g.index:
                r = g.loc[lab]
                print(f"    {name:<8} recording {r.total_s:6.1f} s  |  speech {r.speech_s:6.1f} s"
                      f"  |  pause {100 * r.pause_frac:4.1f}%  |  mean pause {r.mean_pause_s:.2f} s")
        if {0, 1} <= set(g.index):
            d = g.loc[1, "pause_frac"] - g.loc[0, "pause_frac"]
            print(f"    -> patients pause {100 * d:+.1f} percentage points "
                  f"{'MORE' if d > 0 else 'LESS'} of the time")

    print(f"\n{'=' * 84}\nCAN TIMING ALONE PREDICT PARKINSON'S?  (no acoustic content at all)\n{'=' * 84}")
    out_rows = []
    for ds in CORPORA + ["ALL"]:
        s = S if ds == "ALL" else S[S.dataset == ds]
        if len(s) < 25 or s.label.nunique() < 2:
            continue
        y = s.label.to_numpy()
        groups = s.subject_id.to_numpy()

        singles = {}
        for f in FEATURES:
            v = s[f].to_numpy()
            a = auroc(y, v)
            singles[f] = round(max(a, 1 - a), 3)     # direction-free: how informative is it

        X = s[FEATURES].to_numpy()
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True,
                                           random_state=SEED).split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        a, lo, hi = bootstrap_ci(y, oof)

        print(f"\n  {ds:<10} n={len(s)}   all six timing features: "
              f"AUROC {a:.3f}  [{lo:.3f}, {hi:.3f}]")
        best = sorted(singles.items(), key=lambda kv: -kv[1])[:3]
        print("             strongest single features: "
              + ",  ".join(f"{k} {v:.3f}" for k, v in best))
        out_rows.append(dict(dataset=ds, n=len(s), auroc=round(a, 4),
                             ci=[round(lo, 4), round(hi, 4)], singles=singles))

    # IPVS again on age-matched speakers (five matching seeds), since its young controls
    # pause differently for reasons other than the disease
    from pdvoice.cohort import age_match_ids
    from pdvoice.data.demographics import attach_age
    ip = attach_age(S[S.dataset == "ipvs"].copy(), "ipvs")
    matched = []
    for seed in range(42, 47):
        s = ip[ip.subject_id.isin(age_match_ids(ip.dropna(subset=["age"]), seed=seed))]
        y, X, groups = s.label.to_numpy(), s[FEATURES].to_numpy(), s.subject_id.to_numpy()
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True,
                                           random_state=SEED).split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            oof[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        matched.append(auroc(y, oof))
    ipvs_matched = {"auroc_mean": round(float(np.mean(matched)), 4),
                    "per_seed": [round(float(a), 4) for a in matched],
                    "pairs": int(len(s) // 2)}
    print(f"\n  ipvs, age-matched (5 seeds): AUROC {np.mean(matched):.3f}")

    print(f"\n{'=' * 84}\nREADING\n{'=' * 84}")
    print("  These numbers come ONLY from when the speaker was and was not talking.")
    print("  No pitch, no formants, no spectrum -- just timing.")
    print("  Whatever is here is signal the VAD stage currently deletes before the")
    print("  encoder ever sees the audio, and it should be reported as a limitation")
    print("  or recovered as an extra feature block.")

    LOGS.mkdir(parents=True, exist_ok=True)
    S.drop(columns=["subject_id"]).to_csv(LOGS / "pause_information__per_speaker.csv",
                                          index=False)
    (LOGS / "pause_information.json").write_text(json.dumps({
        "features": FEATURES, "n_recordings": int(len(R)), "n_speakers": int(len(S)),
        "results": out_rows,
        "ipvs_agematched": ipvs_matched,
        "per_group": {ds: S[S.dataset == ds].groupby("label")[FEATURES].mean()
                      .round(3).to_dict() for ds in CORPORA},
    }, indent=2, default=str))
    print(f"\nwrote {LOGS / 'pause_information.json'}")


if __name__ == "__main__":
    main()
