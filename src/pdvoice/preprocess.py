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
from pdvoice.utils.audio import (duration_seconds, load_audio, normalise_amplitude,
                                 stable_core, trim_silence)

MIN_DURATION_S = 1.0
VOWEL_CORE_S = 2.0
CHUNK_S = 5.0
CHUNK_OVERLAP = 0.5
NORMALISE_AMPLITUDE = True   # explicit stage; see preprocess_clip.__doc__
TARGET_DBFS = -23.0          # EBU R128 reference level


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


def preprocess_clip(path, task_kind: str, sr: int = SAMPLE_RATE,
                    normalise: bool = NORMALISE_AMPLITUDE) -> np.ndarray:
    """Load + clean one recording according to its task kind.

    Stages: decode to 16 kHz mono -> remove non-speech -> take a fixed-length window
    -> normalise amplitude.

    AMPLITUDE NORMALISATION is applied explicitly here rather than left implicit. The
    SSL feature extractor already zero-means and unit-variances every waveform, so for
    the XLS-R path this stage is a no-op in effect -- but leaving it implicit meant the
    pipeline had an undocumented step that a reader could not see, and it would silently
    disappear behind any model whose front end does not normalise. Making it explicit
    also means the level the encoder actually receives is a property of OUR pipeline,
    not of a library default.

    It changes no measurement: normalisation is a scalar multiply, and
    `scripts/audits/amplitude_vs_harmonics.py` shows F0, HNR, jitter and shimmer are
    unchanged to 0.00% across a 60 dB range.
    """
    wav = load_audio(path, sr=sr)
    if task_kind == "vowel":
        clip = stable_core(trim_silence(wav), sr=sr, seconds=VOWEL_CORE_S)
    else:
        clip = voice_activity_trim(wav, sr=sr)
    return normalise_amplitude(clip) if normalise else clip


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


WINDOW_MODES = ("current", "full", "chunk-drift")


def window_clip(path, task_kind: str, mode: str = "current", sr: int = SAMPLE_RATE,
                normalise: bool = NORMALISE_AMPLITUDE) -> list[np.ndarray]:
    """Return the list of waveforms the encoder should see for ONE recording.

    The windowing decision was never an experiment -- the 2 s vowel core and the 5 s
    speech chunk were inherited defaults. This makes the decision a parameter so it can
    be measured. Three modes:

    ``current``      vowel -> 2 s stable core; speech -> 5 s chunks at 50% overlap.
                     Byte-for-byte what `preprocess_clip` + `chunks` already produce,
                     so this mode reproduces every existing number.
    ``full``         one window per recording: the whole trimmed vowel, or the whole
                     VAD'd speech. Costs no more than ``current`` -- one long forward
                     is cheaper than 12-48 short ones -- and stays inside VRAM: the
                     longest recording in the corpus (250 s) peaks at 7.3 GiB.
    ``chunk-drift``  IDENTICAL audio to ``current``. The difference is downstream: the
                     between-window std is kept instead of discarded (see
                     `pool_windows`). Listed here so the three modes read as one axis.

    NOTE ON NORMALISATION ORDER. The vowel path normalises AFTER taking the core, the
    speech path normalises BEFORE chunking. That asymmetry is inherited from
    `preprocess_clip`, and it is reproduced rather than tidied: normalisation is a
    scalar multiply, so it changes no measurement, but changing WHERE it happens would
    stop `current` matching the caches every published number in this repo was built on.
    """
    if mode not in WINDOW_MODES:
        raise ValueError(f"unknown window mode: {mode!r} (expected one of {WINDOW_MODES})")

    wav = load_audio(path, sr=sr)
    if task_kind == "vowel":
        clean = trim_silence(wav)
        if mode != "full":
            clean = stable_core(clean, sr=sr, seconds=VOWEL_CORE_S)
        return [normalise_amplitude(clean) if normalise else clean]

    clean = voice_activity_trim(wav, sr=sr)
    if normalise:
        clean = normalise_amplitude(clean)
    return [clean] if mode == "full" else list(chunks(clean, sr=sr))


def pool_windows(means: np.ndarray, stds: np.ndarray, mode: str = "current") -> np.ndarray:
    """Collapse per-window statistics into one vector per recording.

    ``means`` and ``stds`` are (n_windows, ...) stacks of the per-window temporal mean
    and temporal std of the hidden states.

    Averaging per-window stds is NOT the std of the whole recording. By the law of total
    variance the total splits into the variance WITHIN windows plus the variance BETWEEN
    window means, and ``current`` keeps only the first term -- measured on real files,
    its std runs 0.89-0.92 of the full-pass std, and the discarded between-window term is
    0.10-0.13 of the within-window one. That term is the long-range drift: loudness decay
    across a sentence, fatigue over a monologue, exactly the monopitch/monoloudness
    effects a reviewer will ask about. ``chunk-drift`` appends it instead of dropping it.

    With a single window there is no between-window variation to measure, so the drift
    block is zeros. That is honest rather than useful: ``chunk-drift`` is only meaningful
    for chunked connected speech.
    """
    mean, std = means.mean(axis=0), stds.mean(axis=0)
    if mode != "chunk-drift":
        return np.concatenate([mean, std], axis=-1)
    drift = means.std(axis=0) if len(means) > 1 else np.zeros_like(mean)
    return np.concatenate([mean, std, drift], axis=-1)


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
