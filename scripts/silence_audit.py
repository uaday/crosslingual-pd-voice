"""How many recordings contain no usable voice, and does that correlate with the label?

QC currently keeps any file whose HEADER says >= 1.0 s; it never opens the audio. A
10 s file of pure silence or room noise therefore passes, gets a 2 s "stable core"
taken out of the middle, and is embedded and voted on as if it were phonation.

This measures, per recording: full-file RMS and peak in dBFS, the fraction of the
file below a silence floor, and what survives librosa's silence trim. Then it splits
the result by label, because a class-correlated dropout is a confound, not just noise.

Usage:
    .venv\\Scripts\\python.exe -u scripts\\silence_audit.py [--mpower-sample 4000]
"""

from __future__ import annotations

import argparse
import random
import warnings

import numpy as np
import pandas as pd
import soundfile as sf

from pdvoice.config import LOGS
from pdvoice.data.ipvs import load_ipvs
from pdvoice.data.mdvr_kcl import load_mdvr_kcl
from pdvoice.data.mpower import load_mpower
from pdvoice.data.neurovoz import load_neurovoz

LOADERS = {"mpower": load_mpower, "neurovoz": load_neurovoz,
           "ipvs": load_ipvs, "mdvr_kcl": load_mdvr_kcl}

DIGITAL_SILENCE = -np.inf   # peak == 0, literally all zeros
NEAR_SILENT_DB = -60.0      # below this is inaudible without cranking the volume
VERY_QUIET_DB = -50.0
FRAME_MS = 25


def _db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def measure(path: str) -> dict:
    """Full-file level statistics, read at native rate (no resample, no decode cost)."""
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001
        return {"path": path, "error": type(e).__name__}
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.size == 0:
        return {"path": path, "error": "empty"}

    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(np.square(wav))))
    n = max(1, int(sr * FRAME_MS / 1000))
    n_frames = wav.size // n
    if n_frames:
        frames = wav[: n_frames * n].reshape(n_frames, n)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
        quiet_frac = float(np.mean(frame_db < VERY_QUIET_DB))
    else:
        quiet_frac = 1.0
    return {
        "path": path, "error": None, "sr": sr, "duration_s": wav.size / sr,
        "peak_dbfs": _db(peak), "rms_dbfs": _db(rms),
        "quiet_frame_frac": quiet_frac, "all_zero": bool(peak == 0.0),
    }


def surviving_seconds(path: str) -> float:
    """Seconds of audio left after librosa's silence trim -- what the encoder would see."""
    import librosa

    try:
        wav, _ = librosa.load(path, sr=16000, mono=True)
        trimmed, _ = librosa.effects.trim(wav, top_db=30.0)
        return float(len(trimmed) / 16000)
    except Exception:  # noqa: BLE001
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpower-sample", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    warnings.simplefilter("ignore")

    frames = []
    for name, loader in LOADERS.items():
        m = loader()
        if name == "mpower" and args.mpower_sample and len(m) > args.mpower_sample:
            m = m.sample(args.mpower_sample, random_state=args.seed)
            note = f"random {len(m):,}"
        else:
            note = f"all {len(m):,}"
        print(f"[{name}] measuring {note} recordings ...", flush=True)
        meas = pd.DataFrame([measure(p) for p in m["path"]])
        j = m.merge(meas, on="path", how="left")
        j["dataset"] = name
        frames.append(j)
    df = pd.concat(frames, ignore_index=True)

    df["is_silent"] = df["all_zero"].fillna(False) | (df["rms_dbfs"] < NEAR_SILENT_DB)
    df["is_very_quiet"] = df["rms_dbfs"] < VERY_QUIET_DB

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(f"\n{'=' * 96}\nLEVEL SUMMARY BY DATASET\n{'=' * 96}")
    summ = df.groupby("dataset").agg(
        recordings=("path", "size"),
        unreadable=("error", lambda s: int(s.notna().sum())),
        all_zero=("all_zero", "sum"),
        silent_lt_60dB=("is_silent", "sum"),
        very_quiet_lt_50dB=("is_very_quiet", "sum"),
        median_rms_dbfs=("rms_dbfs", "median"),
        p05_rms_dbfs=("rms_dbfs", lambda s: s.quantile(0.05)),
        median_quiet_frac=("quiet_frame_frac", "median"),
    ).round(2)
    print(summ.to_string())

    print(f"\n{'=' * 96}\nSILENT / NEAR-SILENT BY LABEL  (is the dropout class-correlated?)\n{'=' * 96}")
    lab = df[df["label"].isin([0, 1])]
    by = lab.groupby(["dataset", "label"]).agg(
        recordings=("path", "size"),
        silent=("is_silent", "sum"),
        silent_rate=("is_silent", "mean"),
        very_quiet=("is_very_quiet", "sum"),
        very_quiet_rate=("is_very_quiet", "mean"),
    )
    by["silent_rate"] = (100 * by["silent_rate"]).round(2)
    by["very_quiet_rate"] = (100 * by["very_quiet_rate"]).round(2)
    print(by.to_string())

    print(f"\n{'=' * 96}\nSUBJECTS WHOSE RECORDINGS ARE *ALL* SILENT\n{'=' * 96}")
    s = lab.groupby(["dataset", "subject_id"]).agg(label=("label", "first"),
                                                   n=("path", "size"), silent=("is_silent", "sum"))
    s["all_silent"] = s["n"] == s["silent"]
    tot = s.groupby(["dataset", "label"]).agg(subjects=("n", "size"), all_silent=("all_silent", "sum"))
    print(tot.to_string() if len(tot) else "  none")

    worst = df[df["is_silent"]].nsmallest(12, "rms_dbfs")
    if len(worst):
        print(f"\n{'=' * 96}\nWHAT SURVIVES PREPROCESSING FOR THE QUIETEST FILES\n{'=' * 96}")
        worst = worst.copy()
        worst["survives_s"] = [surviving_seconds(p) for p in worst["path"]]
        print(worst[["dataset", "label", "task_kind", "duration_s", "rms_dbfs",
                     "peak_dbfs", "quiet_frame_frac", "survives_s"]].round(3).to_string(index=False))

    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / "silence_audit.csv"
    df.drop(columns=[c for c in ("wavPath",) if c in df.columns]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
