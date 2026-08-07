"""Audio loading and lightweight signal helpers.

Heavy deps (librosa / soundfile) are imported inside the functions so that merely
importing the package stays cheap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pdvoice.config import SAMPLE_RATE


def load_audio(path: str | Path, sr: int = SAMPLE_RATE, mono: bool = True) -> np.ndarray:
    """Load an audio file as a float32 waveform, resampled to ``sr`` (mono by default)."""
    import librosa

    wav, _ = librosa.load(str(path), sr=sr, mono=mono)
    return wav.astype(np.float32)


def duration_seconds(path: str | Path) -> float:
    """Return the duration of an audio file in seconds (no full decode)."""
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def rms_dbfs(wav: np.ndarray) -> float:
    """Root-mean-square level in dBFS. Rough loudness / silence indicator."""
    rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def trim_silence(wav: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Energy-based leading/trailing silence trim (good enough for sustained vowels)."""
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed


def stable_core(wav: np.ndarray, sr: int = SAMPLE_RATE, seconds: float = 2.0) -> np.ndarray:
    """Take a stable central segment of ``seconds`` (for sustained-vowel phonation).

    Returns the whole clip if it is shorter than the requested window.
    """
    n = int(seconds * sr)
    if len(wav) <= n:
        return wav
    start = (len(wav) - n) // 2
    return wav[start : start + n]
