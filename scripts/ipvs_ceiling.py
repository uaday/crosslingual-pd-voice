"""Why does a head trained within IPVS reach AUROC 1.000? (review point 6)

The ceiling survives removing the patient-only 44.1 kHz batch and the young controls
(verify/verify_ipvs_batch.py): 18 patients against 22 elderly controls of similar age, all at
16 kHz, are still separated perfectly. This script asks whether properties of the RECORDING,
which carry no information about the voice, already separate those two groups.

Per recording (connected speech and vowels, 16 kHz only, patients vs elderly controls):
  * file format: subtype (bit depth), channels, duration
  * level: RMS of the whole file, peak, fraction of clipped samples, DC offset
  * noise floor: RMS and spectral centroid of the quietest 10% of 25 ms frames
    (silence and room noise, not speech)
  * high-frequency energy above 4 kHz in the quietest frames
Each is averaged per participant and scored on its own as a classifier (AUROC, oriented
so that 0.5 is chance and 1.0 is perfect in either direction).

Run:  .venv\\Scripts\\python.exe scripts\\ipvs_ceiling.py
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score

from pdvoice.config import LOGS
from pdvoice.data.ipvs import load_ipvs


def rec_features(path: str) -> dict:
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x[:, 0]
    n = int(0.025 * sr)
    frames = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)
    quiet = frames[rms <= np.quantile(rms, 0.10)]
    spec = np.abs(np.fft.rfft(quiet * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    p = spec.mean(axis=0) + 1e-20
    return dict(
        subtype_16bit=float(info.subtype == "PCM_16"), channels=float(info.channels),
        duration_s=len(x) / sr, rms_db=20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12),
        peak=float(np.max(np.abs(x))), clipped_frac=float(np.mean(np.abs(x) >= 0.999)),
        dc_offset=float(np.mean(x)),
        noise_floor_db=20 * np.log10(np.sqrt(np.mean(quiet ** 2)) + 1e-12),
        noise_centroid_hz=float(np.sum(freqs * p) / np.sum(p)),
        noise_hf_ratio_db=10 * np.log10(p[freqs >= 4000].sum() / p.sum()),
    )


def oriented_auc(y, v):
    a = roc_auc_score(y, v)
    return max(a, 1 - a), ("higher in PD" if a >= 0.5 else "higher in controls")


def main() -> None:
    warnings.simplefilter("ignore")
    man = load_ipvs()
    man["group"] = man.subject_id.str.split("_").str[1]
    sr44 = man.groupby("subject_id").native_sr.transform(lambda s: (s.fillna(0) >= 44100).any())
    m = man[man.group.isin(["pd", "ehc"]) & ~sr44].copy()
    print(f"recordings {len(m)} | participants: "
          f"{m.groupby('group').subject_id.nunique().to_dict()}")

    feats = pd.DataFrame([rec_features(p) for p in m.path])
    feats["subject_id"], feats["label"], feats["task"] = (m.subject_id.to_numpy(),
                                                         m.label.to_numpy(), m.task_kind.to_numpy())
    out = {}
    for task in ("all", "speech", "vowel"):
        f = feats if task == "all" else feats[feats.task == task]
        s = f.drop(columns="task").groupby("subject_id").mean()
        y = s.pop("label").astype(int).to_numpy()
        res = {}
        for c in s.columns:
            if s[c].nunique() < 2:
                res[c] = dict(auroc=0.5, direction="constant")
                continue
            a, d = oriented_auc(y, s[c].to_numpy())
            res[c] = dict(auroc=round(float(a), 3), direction=d,
                          pd_mean=round(float(s[c][y == 1].mean()), 4),
                          hc_mean=round(float(s[c][y == 0].mean()), 4))
        out[task] = dict(n_pd=int(y.sum()), n_hc=int((y == 0).sum()), features=res)
        print(f"\n=== {task}: {int(y.sum())} PD vs {int((y == 0).sum())} elderly HC")
        for c, r in sorted(res.items(), key=lambda kv: -kv[1]["auroc"]):
            print(f"   {c:<20} AUROC {r['auroc']:.3f}  {r.get('direction', '')}"
                  + (f"  (PD {r['pd_mean']}, HC {r['hc_mean']})" if 'pd_mean' in r else ""))
    (LOGS / "ipvs_ceiling.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {LOGS / 'ipvs_ceiling.json'}")


if __name__ == "__main__":
    main()
