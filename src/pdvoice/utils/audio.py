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


def normalise_amplitude(wav: np.ndarray, target_dbfs: float = -23.0,
                        peak_ceiling: float = 0.98) -> np.ndarray:
    """Scale a waveform to a common RMS level. Loudness only — nothing else changes.

    Recording level varies enormously between speakers (microphone, distance, gain) and
    carries no information about the voice. Two recordings in this corpus differ by
    ~20 dB with no clinical difference between them.

    THIS IS A PURE SCALAR MULTIPLY, so every measure the analysis actually uses is
    unaffected: `scripts/audits/amplitude_vs_harmonics.py` attenuates real recordings by
    60 dB, renormalises, and re-measures F0, HNR, jitter and shimmer -- all change by
    0.00%. That is the point. Loudness is not information; harmonic structure is.

    -23 dBFS is the EBU R128 broadcast reference. The peak ceiling prevents a quiet
    recording with one loud transient from clipping when it is scaled up.
    """
    rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-12))
    if rms <= 1e-9:
        return wav                                   # digital silence: nothing to scale
    gain = 10.0 ** (target_dbfs / 20.0) / rms
    peak = float(np.max(np.abs(wav)))
    if peak * gain > peak_ceiling:
        gain = peak_ceiling / max(peak, 1e-12)
    return (wav * gain).astype(np.float32)


def encoder_applied_gain_db(wav: np.ndarray) -> float:
    """How much gain the SSL feature extractor's zero-mean/unit-variance step applies.

    For the methods chapter: the encoder normalises every waveform whether we do or not,
    so it never sees the loudness differences above. Measured across this corpus the
    implicit gain is roughly +17 dB on a normal recording and +55 to +70 dB on a very
    quiet one -- which is also why discarding audio for being quiet is the wrong test.

    The extractor scales to unit variance, i.e. to 0 dBFS, so the gain it applies is just
    the negative of the input level.
    """
    return -rms_dbfs(wav)


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
