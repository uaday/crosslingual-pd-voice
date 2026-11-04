"""How much spectral gating removes from the recordings that were checked by listening.

The listening set (pd-datasets/bridge2ai_review/listening_before_after/, git-ignored, never
shared) holds the Bridge2AI recordings with the strongest 60 Hz mains hum, the five worst
US patients and five worst controls (three connected-speech and two other recordings per
group, one per participant), plus two connected-speech recordings with no hum as a
reference. Each was compared by ear before and after cleaning.

For each recording this measures, with pdvoice.preprocess.spectral_gate:
  pause noise   change in energy over the quietest 20% of frames (noise only)
  loud speech   change in energy over the loudest 20% of frames (voice)

Output: outputs/logs/denoise_listening_check.json (aggregates only, no identifiers).
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from pdvoice.config import BRIDGE2AI_DIR, LOGS
from pdvoice.preprocess import spectral_gate

PAIRS = Path(BRIDGE2AI_DIR).parent / "bridge2ai_review" / "listening_before_after"


def change_db(x, y):
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) ** 2
    T = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    e = S.sum(axis=0)
    q, v = e <= np.percentile(e, 20), e >= np.percentile(e, 80)
    return (10 * np.log10(T[:, q].sum() / S[:, q].sum()),
            10 * np.log10(T[:, v].sum() / S[:, v].sum()))


def main() -> None:
    res = {"hum": [], "reference": []}
    for f in sorted(PAIRS.glob("*_A_original.wav")):
        x, _ = sf.read(f, dtype="float32")
        pause, loud = change_db(x, spectral_gate(x))
        res["reference" if "clean-reference" in f.name else "hum"].append((pause, loud))
    out = {}
    for k, v in res.items():
        p, s = np.array(v).T
        out[k] = {"n": len(v), "pause_noise_db_min": round(float(p.min()), 1),
                  "pause_noise_db_max": round(float(p.max()), 1),
                  "loud_speech_db_median": round(float(np.median(s)), 2),
                  "loud_speech_db_worst": round(float(s.min()), 2),
                  "loud_speech_within_0.2db": int((s > -0.2).sum())}
    (LOGS / "denoise_listening_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
