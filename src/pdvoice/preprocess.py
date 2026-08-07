"""Raw audio -> analysis-ready clips + quality-controlled manifest.

Pipeline (PROJECT_CONTEXT.md §4.1): resample 16 kHz mono -> VAD -> QC.
Sustained vowels take a stable central core; connected speech is chunked and the
per-chunk embeddings are averaged downstream (see features/ssl.py).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from pdvoice.config import SAMPLE_RATE
from pdvoice.utils.audio import duration_seconds, load_audio, stable_core, trim_silence

MIN_DURATION_S = 1.0
VOWEL_CORE_S = 2.0
CHUNK_S = 5.0
CHUNK_OVERLAP = 0.5


def voice_activity_trim(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Strip non-speech regions from connected speech.

    Tries silero-vad; falls back to an energy-based trim if it isn't available.
    """
    try:
        import torch
        from silero_vad import collect_chunks, get_speech_timestamps, load_silero_vad

        model = load_silero_vad()
        t = torch.from_numpy(wav)
        ts = get_speech_timestamps(t, model, sampling_rate=sr)
        if ts:
            return collect_chunks(ts, t).numpy()
    except Exception:  # noqa: BLE001 — VAD is best-effort; never fail preprocessing on it
        pass
    return trim_silence(wav)


def preprocess_clip(path, task_kind: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load + clean one recording according to its task kind."""
    wav = load_audio(path, sr=sr)
    if task_kind == "vowel":
        return stable_core(trim_silence(wav), sr=sr, seconds=VOWEL_CORE_S)
    return voice_activity_trim(wav, sr=sr)


def chunks(wav: np.ndarray, sr: int = SAMPLE_RATE, seconds: float = CHUNK_S,
           overlap: float = CHUNK_OVERLAP) -> Iterator[np.ndarray]:
    """Yield fixed-length windows for connected speech (chunk-and-average)."""
    n = int(seconds * sr)
    if len(wav) <= n:
        yield wav
        return
    hop = max(1, int(n * (1 - overlap)))
    for start in range(0, len(wav) - n + 1, hop):
        yield wav[start : start + n]


def quality_control(df: pd.DataFrame, min_duration_s: float = MIN_DURATION_S):
    """Split a manifest into (kept, dropped) by basic QC. Adds a ``duration_s`` column."""
    durations = []
    for p in df["path"]:
        try:
            durations.append(duration_seconds(p))
        except Exception:  # noqa: BLE001 — unreadable file -> duration 0 -> dropped
            durations.append(0.0)
    out = df.copy()
    out["duration_s"] = durations
    keep = out["duration_s"] >= min_duration_s
    return out[keep].reset_index(drop=True), out[~keep].reset_index(drop=True)
