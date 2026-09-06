"""Is a QUIET recording an UNINFORMATIVE recording?  (Supervisor question, 2026-09-05)

The proposed QC guard discarded recordings below an RMS floor. The objection:

    "Amplitude can be easily improved -- just multiply the signal by a gain. Some
     voice samples have very less amplitude but they are rich in terms of the
     harmonics that they carry, and those harmonics are important for analysis.
     Energy is only one thing."

That is testable, so we test it rather than asserting either way.

  EXPERIMENT A (controlled) -- take normal-level recordings, attenuate them by a
    known amount, renormalise, and re-measure the harmonic descriptors. If the
    descriptors survive, amplitude carries no information the analysis needs and
    an RMS floor discards good data.

  EXPERIMENT B (real) -- take the genuinely quiet recordings found by
    silence_audit.py, normalise them, and ask whether they are harmonically rich
    (keep them) or actually empty (a *harmonic* criterion would drop them).

Descriptors are the ones the classical arm already uses -- F0, HNR, jitter,
shimmer -- plus spectral flatness (tonal vs noise-like) and a count of detectable
harmonics.

Run:  .venv\\Scripts\\python.exe -u scripts\\amplitude_vs_harmonics.py --n 60
"""

from __future__ import annotations

import argparse
import warnings

import librosa
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdvoice.config import FIGURES, LOGS, SAMPLE_RATE  # noqa: E402
from pdvoice.data.mpower import load_mpower  # noqa: E402
from pdvoice.preprocess import VOWEL_CORE_S, quality_control  # noqa: E402
from pdvoice.utils.audio import load_audio, rms_dbfs, stable_core, trim_silence  # noqa: E402

WAVE, SPEC = "#3B6EA5", "#B5651D"
ATTENUATIONS_DB = [0, -20, -40, -60]


def peak_normalise(w, target_peak=0.95):
    p = float(np.max(np.abs(w)))
    return w if p == 0 else (w * (target_peak / p)).astype(np.float32)


def rms_normalise(w, target_dbfs=-23.0):
    r = float(np.sqrt(np.mean(w ** 2)))
    if r == 0:
        return w
    return (w * (10 ** (target_dbfs / 20) / r)).astype(np.float32)


def harmonic_descriptors(w, sr=SAMPLE_RATE) -> dict:
    """Voice-quality descriptors. All are amplitude-INVARIANT in principle."""
    import parselmouth
    from parselmouth.praat import call

    out = {"rms_dbfs": rms_dbfs(w), "peak_dbfs": 20 * np.log10(max(float(np.max(np.abs(w))), 1e-12))}
    try:
        snd = parselmouth.Sound(w.astype(np.float64), sampling_frequency=sr)
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        out["jitter"] = call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3)
        out["shimmer"] = call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6)
        h = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        out["hnr"] = call(h, "Get mean", 0, 0)
        pitch = snd.to_pitch()
        out["f0"] = call(pitch, "Get mean", 0, 0, "Hertz")
        out["voiced_frac"] = float(np.mean(pitch.selected_array["frequency"] > 0))
    except Exception as e:  # noqa: BLE001
        out |= {"jitter": np.nan, "shimmer": np.nan, "hnr": np.nan,
                "f0": np.nan, "voiced_frac": np.nan, "praat_error": type(e).__name__}

    # spectral flatness: ~0 = tonal/harmonic, ~1 = noise-like
    S = np.abs(librosa.stft(w, n_fft=1024, hop_length=160)) ** 2
    out["spectral_flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=np.sqrt(S))))

    # count spectral peaks above the local noise floor = detectable harmonics
    spec = S.mean(axis=1)
    db = 10 * np.log10(np.maximum(spec, 1e-20))
    db -= db.max()
    med = np.median(db)
    peaks = ((db[1:-1] > db[:-2]) & (db[1:-1] > db[2:]) & (db[1:-1] > med + 6)).sum()
    out["n_harmonic_peaks"] = int(peaks)
    return out


def prep(path):
    return stable_core(trim_silence(load_audio(path)), sr=SAMPLE_RATE, seconds=VOWEL_CORE_S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="recordings for experiment A")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    mp = load_mpower()
    mp = mp[mp.label.isin([0, 1])]
    mp = mp[mp.groupby("subject_id").cumcount() < 1]
    mp, _ = quality_control(mp); mp = mp.reset_index(drop=True)

    # ---------- levels, to separate "normal" from "quiet" ----------
    rng = np.random.default_rng(args.seed)
    probe = mp.sample(min(6000, len(mp)), random_state=args.seed)
    levels = []
    for p in probe.path:
        try:
            w = prep(p)
            levels.append((p, rms_dbfs(w)))
        except Exception:  # noqa: BLE001
            pass
    lv = pd.DataFrame(levels, columns=["path", "rms_dbfs"])
    quiet = lv[lv.rms_dbfs < -45].path.tolist()
    normal = lv[(lv.rms_dbfs > -30) & (lv.rms_dbfs < -15)].path.tolist()
    print(f"probed {len(lv):,} recordings | quiet (<-45 dBFS): {len(quiet)} | "
          f"normal (-30..-15 dBFS): {len(normal):,}")

    # ================= EXPERIMENT A: attenuate then renormalise =================
    print("\n[A] attenuate normal recordings, renormalise, re-measure ...", flush=True)
    picks = rng.choice(normal, min(args.n, len(normal)), replace=False)
    rows = []
    for p in picks:
        w0 = prep(p)
        for att in ATTENUATIONS_DB:
            w = (w0 * (10 ** (att / 20))).astype(np.float32)
            d_raw = harmonic_descriptors(w)
            d_nrm = harmonic_descriptors(rms_normalise(w))
            rows.append({"path": p, "atten_db": att, "stage": "as-is", **d_raw})
            rows.append({"path": p, "atten_db": att, "stage": "normalised", **d_nrm})
    A = pd.DataFrame(rows)
    A.to_csv(LOGS / "amplitude_vs_harmonics__experimentA.csv", index=False)

    cols = ["f0", "hnr", "jitter", "shimmer", "spectral_flatness", "n_harmonic_peaks", "voiced_frac"]
    print("\n  Descriptor means by attenuation (AFTER RMS normalisation):")
    tblN = A[A.stage == "normalised"].groupby("atten_db")[cols].mean().round(4)
    print(tblN.to_string())
    print("\n  Same, WITHOUT normalisation (what a naive pipeline would see):")
    tblR = A[A.stage == "as-is"].groupby("atten_db")[cols].mean().round(4)
    print(tblR.to_string())

    base = tblN.loc[0]
    drift = ((tblN - base).abs() / base.abs().replace(0, np.nan) * 100).round(2)
    print("\n  % change vs 0 dB, after normalisation (0% => amplitude carries no information):")
    print(drift.to_string())

    # ================= EXPERIMENT B: the genuinely quiet recordings =================
    print(f"\n[B] the {len(quiet)} genuinely quiet recordings, normalised ...", flush=True)
    rowsB = []
    for p in quiet:
        w = prep(p)
        rowsB.append({"path": p, "group": "quiet(<-45dB)", **harmonic_descriptors(rms_normalise(w))})
    for p in rng.choice(normal, min(len(quiet) * 3, len(normal)), replace=False):
        w = prep(p)
        rowsB.append({"path": p, "group": "normal", **harmonic_descriptors(rms_normalise(w))})
    B = pd.DataFrame(rowsB)
    B.to_csv(LOGS / "amplitude_vs_harmonics__experimentB.csv", index=False)
    print("\n  After normalisation, are the quiet ones harmonically poorer?")
    print(B.groupby("group")[cols].agg(["mean", "std"]).round(3).to_string())

    # ================= FIGURE =================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ex = picks[0]
    w0 = prep(ex)
    for att, c in zip(ATTENUATIONS_DB, ["#222", WAVE, SPEC, "#999"]):
        w = w0 * (10 ** (att / 20))
        S = np.abs(librosa.stft(rms_normalise(w), n_fft=1024, hop_length=160)) ** 2
        db = 10 * np.log10(np.maximum(S.mean(1), 1e-20)); db -= db.max()
        axes[0].plot(librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=1024), db,
                     color=c, lw=1, label=f"{att} dB then normalised")
    axes[0].set_xscale("log"); axes[0].set_xlim(60, 8000); axes[0].set_ylim(-80, 5)
    axes[0].set_xlabel("frequency (Hz)"); axes[0].set_ylabel("relative power (dB)")
    axes[0].set_title("A · attenuate then normalise:\nthe harmonic structure is unchanged",
                      fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=7)

    for ax, m, lbl in [(axes[1], "hnr", "HNR (dB)"), (axes[2], "n_harmonic_peaks", "detectable harmonic peaks")]:
        d = A[A.stage == "normalised"]
        ax.boxplot([d[d.atten_db == a][m].dropna() for a in ATTENUATIONS_DB])
        ax.set_xticks(range(1, len(ATTENUATIONS_DB) + 1))
        ax.set_xticklabels([f"{a} dB" for a in ATTENUATIONS_DB])
        ax.set_ylabel(lbl); ax.set_xlabel("attenuation applied before normalising")
        ax.set_title(f"A · {lbl} survives attenuation", fontsize=10, fontweight="bold")
    fig.suptitle("Does low amplitude mean low information?  Attenuate -> normalise -> re-measure",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94))
    out = FIGURES / "writeup" / "figE_amplitude_vs_harmonics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}")
    print(f"wrote {LOGS / 'amplitude_vs_harmonics__experimentA.csv'}")


if __name__ == "__main__":
    main()
