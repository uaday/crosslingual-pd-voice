"""Classical hand-engineered voice features (the interpretable parallel head).

Jitter, shimmer, HNR, F0 and MFCCs via Praat (parselmouth) + librosa. These are
designed for SUSTAINED PHONATION and are most meaningful on the vowel task; they
are not reliable on connected speech (PROJECT_CONTEXT.md §3.2). SHAP is applied to
this head because each feature maps to an interpretable speech measure.

NOTE: the exact Praat call signatures below follow common practice but should be
sanity-checked against a few known recordings before results are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdvoice.config import SAMPLE_RATE


def classical_features(path: str, n_mfcc: int = 13) -> dict:
    """Return a dict of classical acoustic features for one recording.

    Praat features use the file's native sample rate (they need high resolution);
    MFCCs are computed at 16 kHz.
    """
    import librosa
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(str(path))
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = call(
        [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pitch = snd.to_pitch()
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")

    feats = {
        "jitter_local": float(jitter_local),
        "shimmer_local": float(shimmer_local),
        "hnr": float(hnr),
        "f0_mean": float(f0_mean),
    }

    y, sr = librosa.load(str(path), sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    feats.update({f"mfcc_{i}": float(v) for i, v in enumerate(mfcc)})
    return feats


def extract_for_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Compute classical features for every row; return a (row-aligned) feature frame."""
    records = []
    for idx, r in df.iterrows():
        try:
            feats = classical_features(r["path"])
        except Exception as e:  # noqa: BLE001 — log + skip a bad file rather than abort
            feats = {"error": str(e)}
        feats["_row"] = idx
        records.append(feats)
    return pd.DataFrame(records).set_index("_row")
