"""Audit the Bridge2AI recordings the thesis uses for silence, noise and tonal interference.

Two sources, aggregated per group (US patients vs controls) and task kind:

1. The consortium's own quality table (audio_quality_metrics.tsv): proportion silent,
   clipping, two SNR estimates, amplitude-modulation depth, and their boolean checks.
2. Our own tonal-noise detector: a narrow spectral peak that persists through most frames of
   a recording (a hum, whine or electronic tone), which speech does not produce.

Only aggregates are written (outputs/logs/bridge2ai_noise_audit.json); per-recording flags
stay on university storage (pd-datasets/bridge2ai_review/bridge2ai_noise_flags.csv, git-ignored).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pdvoice.config import BRIDGE2AI_DIR, LOGS
from pdvoice.data.bridge2ai import load_bridge2ai
from pdvoice.data.common_tasks import common_task_mask
from pdvoice.preprocess import quality_control

CACHE = Path(BRIDGE2AI_DIR).parent / "bridge2ai_review"  # git-ignored, stays on university storage
SR = 16000


def _norm(task: str) -> str:
    return re.sub(r"[()]", "", str(task)).lower()


def used_manifest() -> pd.DataFrame:
    """The recordings the thesis analyses: US patients and controls, vowel/DDK/common speech."""
    m = load_bridge2ai()
    m = m[m.group.isin(["pd", "control"]) & m.task_kind.isin(["vowel", "ddk", "speech"])]
    m, _ = quality_control(m)
    m = m[common_task_mask(m, "bridge2ai")].reset_index(drop=True)
    m["pid"] = m.subject_id.str.replace("b2ai_", "", regex=False)
    m["key"] = m.pid + "|" + m.session_id + "|" + m.task_raw.map(_norm)
    return m


def consortium_metrics(m: pd.DataFrame) -> pd.DataFrame:
    q = pd.read_csv(Path(BRIDGE2AI_DIR) / "audio_quality_metrics.tsv", sep="\t", dtype=str)
    q["key"] = q.participant_id + "|" + q.session_id + "|" + q.task_name.map(_norm)
    q = q.drop_duplicates("key").set_index("key")
    out = m[["key"]].join(q, on="key")
    for c in [c for c in q.columns if c.endswith("_metric")]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in [c for c in q.columns if c.endswith("_check")]:
        out[c] = out[c].map({"True": True, "False": False})
    return out.drop(columns=["key", "participant_id", "session_id", "task_name"])


def tonal_noise(path: str) -> dict:
    """Background noise measured in the pauses, where the voice cannot be the cause.

    Pause frames are the quietest 20% of frames. In them we record the noise floor, the
    energy contrast between speech and pauses (a simple SNR), and the strongest narrowband
    peak: for each frequency bin, the fraction of pause frames in which it stands >= 12 dB
    above its spectral neighbourhood. A hum or electronic tone persists through the pauses;
    the voice's pitch and harmonics do not."""
    import librosa
    from scipy.ndimage import median_filter
    y, _ = librosa.load(path, sr=SR, mono=True)
    empty = dict(rms_db=-120.0, buzz50_db=0.0, buzz60_db=0.0, floor_db=-120.0, snr_db=0.0, tone_hz=np.nan,
                 tone_persist=0.0, tone_db=0.0)
    if len(y) < SR // 2 or np.max(np.abs(y)) == 0:
        return empty
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    e = 10 * np.log10(S.sum(axis=0) + 1e-12)
    quiet = e <= np.percentile(e, 20)
    L = 10 * np.log10(S[:, quiet] + 1e-12)
    base = median_filter(L, size=(31, 1), mode="nearest")
    persist = ((L - base) >= 12).mean(axis=1)
    persist[:3] = 0  # ignore DC
    k = int(np.argmax(persist))
    # mains buzz: a comb of peaks at the harmonics of 50 or 60 Hz in the pauses, measured at
    # a fine resolution (n_fft 16384, ~1 Hz), as the mean height of the harmonics above the
    # mid-points between them, over 60 Hz to 4 kHz
    S2 = np.abs(librosa.stft(y, n_fft=16384, hop_length=2048)) ** 2
    e2 = S2.sum(axis=0)
    P = 10 * np.log10(S2[:, e2 <= np.percentile(e2, 20)].mean(axis=1) + 1e-12)
    hz = np.arange(len(P)) * SR / 16384
    def comb(f0):
        h = np.arange(1, int(4000 // f0) + 1) * f0
        on = [P[np.abs(hz - x) <= 2].max() for x in h]
        off = [np.median(P[np.abs(hz - x - f0 / 2) <= 2]) for x in h]
        return float(np.median(np.array(on) - np.array(off)))
    buzz50, buzz60 = comb(50.0), comb(60.0)
    return dict(rms_db=float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)),
                buzz50_db=buzz50, buzz60_db=buzz60,
                floor_db=float(np.median(e[quiet])),
                snr_db=float(np.percentile(e, 95) - np.median(e[quiet])),
                tone_hz=float(k * SR / 2048), tone_persist=float(persist[k]),
                tone_db=float(np.median((L - base)[k])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tonal", action="store_true", help="also run the tonal-noise detector")
    args = ap.parse_args()
    m = used_manifest()
    q = consortium_metrics(m)
    flags = pd.concat([m[["pid", "group", "task_kind", "task_code", "path", "duration_s"]], q], axis=1)
    print("recordings:", len(flags), "matched to consortium table:",
          int(flags.proportion_silent_metric.notna().sum()))
    if args.tonal:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=8, verbose=0)(delayed(tonal_noise)(p) for p in flags.path)
        flags = pd.concat([flags, pd.DataFrame(res)], axis=1)
    CACHE.mkdir(parents=True, exist_ok=True)
    flags.drop(columns=["path"]).to_csv(CACHE / "bridge2ai_noise_flags.csv", index=False)

    checks = [c for c in flags.columns if c.endswith("_check")]
    summary = {}
    for (g, t), d in flags.groupby(["group", "task_kind"]):
        s = {"recordings": len(d), "participants": int(d.pid.nunique())}
        for c in checks:
            s[c] = int(d[c].fillna(False).sum())
        s["any_check"] = int(d[checks].fillna(False).any(axis=1).sum())
        s["participants_any_check"] = int(d[d[checks].fillna(False).any(axis=1)].pid.nunique())
        for c in ["proportion_silent_metric", "spectral_gating_snr_metric", "peak_snr_from_spectral_metric"]:
            s[c + "_median"] = round(float(d[c].median()), 3)
        if "tone_persist" in d:
            s["tone_persist_ge_0.5"] = int((d.tone_persist >= 0.5).sum())
            s["participants_tone"] = int(d[d.tone_persist >= 0.5].pid.nunique())
            buzz = d.buzz60_db >= 5  # comb of 60 Hz mains harmonics at least 5 dB above the gaps
            s["buzz60_ge_5"] = int(buzz.sum())
            s["participants_buzz60"] = int(d[buzz].pid.nunique())
            s["silent_rms_lt_-55"] = int((d.rms_db < -55).sum())
        summary[f"{g}|{t}"] = s
    (LOGS / "bridge2ai_noise_audit.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main()
