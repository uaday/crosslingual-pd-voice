"""Classical hand-engineered voice features — the interpretable parallel arm.

Every feature here maps to a speech measure a clinician recognises, unlike a dimension
of a 2048-vector. That is the point of this arm: SHAP over these means something.

WHAT CHANGED (2026-09-06) AND WHY
---------------------------------
The original set was jitter, shimmer, HNR, F0-mean and 13 MFCC *means* — 17 numbers.
Two problems, both found by comparing against the Bridge2AI VoiceFM study:

1. It emphasised the wrong family. VoiceFM decomposed 88 eGeMAPS features by effect
   size and found the largest PD-vs-control separations in **articulatory** features
   (formant amplitudes, MFCC variability), while the traditional **phonatory** markers
   -- jitter, shimmer, HNR -- showed small effects, "consistently uninformative across
   both cohorts". Our set was almost entirely the family that does not carry the signal.

2. It threw away variability. `mfcc.mean(axis=1)` collapses the time axis, discarding
   exactly the MFCC variability VoiceFM identifies as one of the two dominant families.
   And F0 had a mean but no standard deviation -- so **monopitch**, the single
   most-cited prosodic marker of Parkinson's, was not represented at all.

A third fix: the MFCC window. librosa defaults to n_fft=2048, hop=512, which at 16 kHz
is a **128 ms** window -- a music-analysis default that smears across many glottal
cycles. Speech analysis uses 25 ms / 10 ms, which is what we now pass explicitly.

ANALYSIS WINDOW
---------------
Features are computed on the **same 2 s stable core the SSL path uses**: the whole file
is decoded and trimmed, and its central 2 s are kept, exactly as in
``preprocess.preprocess_clip``. The two arms therefore describe the same seconds (to
within a trim frame, since the trim runs at 44.1 kHz here and 16 kHz there), and the
comparison between them is about representations rather than about which audio was
analysed. Before 2026-11-09 only the first 4 s were decoded, which put this window near
second 2 of a 10 s mPower file and the SSL window near second 5.

AMPLITUDE
---------
No loudness normalisation is applied, and none is needed: every measure here is
amplitude-invariant by construction. `scripts/audits/amplitude_vs_harmonics.py` verifies
this empirically -- attenuating by 60 dB and renormalising changes F0, HNR, jitter and
shimmer by 0.00%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdvoice.config import SAMPLE_RATE
from pdvoice.utils.audio import stable_core, trim_silence

PRAAT_SR = 44_100          # Praat measures periods; keep the native resolution
CORE_SECONDS = 2.0         # matches preprocess.VOWEL_CORE_S
MFCC_N_FFT = 400           # 25 ms at 16 kHz  (librosa's 2048 default = 128 ms)
MFCC_HOP = 160             # 10 ms at 16 kHz
N_MFCC = 13
MAX_FORMANT_HZ = 5500      # Praat's convention for a female voice; see caveat below


def feature_names() -> list[str]:
    """The canonical feature order. Import this instead of re-typing the list."""
    return (
        # --- phonatory: vocal-fold stability -------------------------------
        ["jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq3", "hnr"]
        # --- prosodic: pitch and its variability ---------------------------
        # f0_cv is f0_sd / f0_mean, which cancels most of the sex difference in
        # absolute pitch -- useful because sex is not controlled in our cohorts.
        + ["f0_mean", "f0_sd", "f0_cv"]
        # --- articulatory: vocal-tract shape and movement -------------------
        + ["f1_mean", "f2_mean", "f3_mean", "f1_sd", "f2_sd",
           "f1_bandwidth", "f2_bandwidth", "f2_f1_ratio"]
        # --- spectral envelope, mean AND variability ------------------------
        + [f"mfcc_{i}" for i in range(N_MFCC)]
        + [f"mfcc_{i}_sd" for i in range(N_MFCC)]
    )


FEATURE_NAMES = feature_names()

# Part of every feature-cache file name. Bump it whenever the extraction itself changes
# (window, settings), so a cache keyed only on recording paths cannot be reused silently.
# "core2s" = central 2 s of the whole trimmed file (2026-11-09), as in preprocess_clip.
FEATURE_VERSION = "core2s"


def _praat_sound(path):
    """The 2 s stable core at Praat's sample rate, as a parselmouth Sound."""
    import librosa
    import parselmouth

    # Decode the WHOLE file, trim, then take the central 2 s: the same steps, in the same
    # order, as preprocess.preprocess_clip, so both arms analyse the same seconds. (An
    # earlier version decoded only the first 4 s, which centred this window on about
    # second 2 of a 10 s mPower file while the SSL window sat at about second 5.)
    y, _ = librosa.load(str(path), sr=PRAAT_SR, mono=True)
    y = stable_core(trim_silence(y), sr=PRAAT_SR, seconds=CORE_SECONDS).astype("float64")
    if len(y) < PRAAT_SR // 2:
        raise ValueError("less than 0.5 s of usable audio")
    return parselmouth.Sound(y, sampling_frequency=PRAAT_SR), y


def classical_features(path: str) -> dict:
    """All features for one recording, keyed by :data:`FEATURE_NAMES`."""
    import librosa
    from parselmouth.praat import call

    snd, y = _praat_sound(path)
    f: dict[str, float] = {}

    # ---- phonatory ------------------------------------------------------------
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    f["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3)
    f["jitter_rap"] = call(pp, "Get jitter (rap)", 0, 0, 1e-4, 0.02, 1.3)
    f["shimmer_local"] = call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6)
    f["shimmer_apq3"] = call([snd, pp], "Get shimmer (apq3)", 0, 0, 1e-4, 0.02, 1.3, 1.6)
    f["hnr"] = call(call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0)

    # ---- prosodic: pitch ------------------------------------------------------
    pitch = snd.to_pitch()
    f["f0_mean"] = call(pitch, "Get mean", 0, 0, "Hertz")
    f["f0_sd"] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f["f0_cv"] = f["f0_sd"] / f["f0_mean"] if f["f0_mean"] else np.nan

    # ---- articulatory: formants ----------------------------------------------
    fm = call(snd, "To Formant (burg)", 0.0, 5, MAX_FORMANT_HZ, 0.025, 50)
    for n in (1, 2, 3):
        f[f"f{n}_mean"] = call(fm, "Get mean", n, 0, 0, "Hertz")
    for n in (1, 2):
        f[f"f{n}_sd"] = call(fm, "Get standard deviation", n, 0, 0, "Hertz")
        f[f"f{n}_bandwidth"] = call(fm, "Get quantile of bandwidth", n, 0, 0, "Hertz", 0.5)
    f["f2_f1_ratio"] = (f["f2_mean"] / f["f1_mean"]
                        if f["f1_mean"] and np.isfinite(f["f1_mean"]) else np.nan)

    # ---- spectral envelope: MFCC mean AND standard deviation -------------------
    y16 = librosa.resample(y.astype("float32"), orig_sr=PRAAT_SR, target_sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y16, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                               n_fft=MFCC_N_FFT, hop_length=MFCC_HOP)
    for i in range(N_MFCC):
        f[f"mfcc_{i}"] = float(mfcc[i].mean())
        f[f"mfcc_{i}_sd"] = float(mfcc[i].std())

    return {k: float(f[k]) for k in FEATURE_NAMES}


def extract_for_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Features for every row, row-aligned to ``df``. Failures become all-NaN rows."""
    records = []
    for idx, r in df.iterrows():
        try:
            feats = classical_features(r["path"])
        except Exception as e:  # noqa: BLE001 — one bad file must not abort the run
            feats = dict.fromkeys(FEATURE_NAMES, np.nan)
            feats["error"] = str(e)[:120]
        feats["_row"] = idx
        records.append(feats)
    return pd.DataFrame(records).set_index("_row")
