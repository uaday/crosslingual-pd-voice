"""Figures for the DATA-UNDERSTANDING deliverable.

A. Four views of the same sound (waveform / spectrum / spectrogram / mel), PD vs HC,
   for each dataset+task. These are the four representations from the HF audio course.
B. Preprocessing stages, drawn on the actual waveform (raw -> 16 kHz -> trim/VAD -> core).
C. Recording-duration distributions per dataset (why we chunk / take a core).
D. Vowel vs connected speech, side by side -- why the task matters more than the model.

Usage:
    .venv\\Scripts\\python.exe -u scripts\\voice_fundamentals_figs.py
"""

from __future__ import annotations

import argparse

import librosa
import librosa.display
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdvoice.config import FIGURES, SAMPLE_RATE  # noqa: E402
from pdvoice.data.ipvs import load_ipvs  # noqa: E402
from pdvoice.data.mdvr_kcl import load_mdvr_kcl  # noqa: E402
from pdvoice.data.neurovoz import load_neurovoz  # noqa: E402
from pdvoice.preprocess import CHUNK_S, VOWEL_CORE_S, voice_activity_trim  # noqa: E402
from pdvoice.utils.audio import load_audio, stable_core, trim_silence  # noqa: E402

OUT = FIGURES / "writeup"
LOADERS = {"neurovoz": load_neurovoz, "ipvs": load_ipvs, "mdvr_kcl": load_mdvr_kcl}
N_FFT, HOP, N_MELS, FMAX = 1024, 160, 80, 8000   # speech-appropriate (64 ms / 10 ms)


def four_views(ax_col, wav, sr, title):
    """Draw waveform / spectrum / spectrogram / mel into a column of 4 axes."""
    a_wave, a_spec, a_stft, a_mel = ax_col

    librosa.display.waveshow(wav, sr=sr, ax=a_wave, color="#3b6ea5")
    a_wave.set_title(title, fontsize=11, fontweight="bold")
    a_wave.set_ylabel("amplitude\n[-1, 1]")
    a_wave.set_xlabel("")

    # single-window spectrum, taken from the centre of the clip
    n = min(4096, len(wav))
    seg = wav[max(0, len(wav) // 2 - n // 2):][:n]
    win = np.hanning(len(seg))
    dft = np.fft.rfft(seg * win)
    amp_db = librosa.amplitude_to_db(np.abs(dft), ref=np.max)
    freq = librosa.fft_frequencies(sr=sr, n_fft=len(seg))
    a_spec.plot(freq, amp_db, lw=0.8, color="#b5651d")
    a_spec.set_xscale("log")
    a_spec.set_xlim(50, sr / 2)
    a_spec.set_ylim(-80, 5)
    a_spec.set_ylabel("amplitude\n(dB)")
    a_spec.set_xlabel("frequency (Hz)")

    D = librosa.stft(wav, n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="hz",
                             ax=a_stft, cmap="magma")
    a_stft.set_ylim(0, FMAX)
    a_stft.set_ylabel("frequency\n(Hz, linear)")
    a_stft.set_xlabel("")

    M = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmax=FMAX)
    M_db = librosa.power_to_db(M, ref=np.max)
    librosa.display.specshow(M_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel",
                             fmax=FMAX, ax=a_mel, cmap="magma")
    a_mel.set_ylabel("mel\n(perceptual)")
    a_mel.set_xlabel("time (s)")


def fig_a(manifests, seed):
    """Four views, PD vs HC, per dataset+task."""
    for ds, m in manifests.items():
        for task in sorted(m["task_kind"].dropna().unique()):
            if task not in ("vowel", "speech"):
                continue
            pdr = m[(m.task_kind == task) & (m.label == 1)]
            hcr = m[(m.task_kind == task) & (m.label == 0)]
            if not len(pdr) or not len(hcr):
                continue
            fig, axes = plt.subplots(4, 2, figsize=(14, 11))
            for col, (row, who) in enumerate([
                (hcr.sample(1, random_state=seed).iloc[0], "HC (control)"),
                (pdr.sample(1, random_state=seed).iloc[0], "PD (Parkinson's)"),
            ]):
                wav = load_audio(row["path"])
                wav = (stable_core(trim_silence(wav), seconds=VOWEL_CORE_S)
                       if task == "vowel" else voice_activity_trim(wav)[:int(5 * SAMPLE_RATE)])
                four_views([axes[r][col] for r in range(4)], wav, SAMPLE_RATE,
                           f"{who}\n{ds} · {task} · {row['subject_id']}")
            fig.suptitle(f"Four views of the same sound — {ds} · {task} "
                         f"(16 kHz, after preprocessing)", fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            p = OUT / f"figA_fourviews__{ds}__{task}.png"
            fig.savefig(p, dpi=140); plt.close(fig)
            print("wrote", p, flush=True)


def fig_b(manifests, seed):
    """Preprocessing stages drawn on the real waveform."""
    for ds, m in manifests.items():
        for task in ("vowel", "speech"):
            sel = m[(m.task_kind == task) & (m.label == 1)]
            if not len(sel):
                continue
            row = sel.sample(1, random_state=seed).iloc[0]
            info = sf.info(row["path"])
            raw_native, sr_n = sf.read(row["path"], dtype="float32")
            if raw_native.ndim > 1:
                raw_native = raw_native.mean(axis=1)
            wav = load_audio(row["path"])
            if task == "vowel":
                s2 = trim_silence(wav); s3 = stable_core(s2, seconds=VOWEL_CORE_S)
                lbl2, lbl3 = "3. trim_silence (top_db=30)", f"4. stable_core ({VOWEL_CORE_S}s)"
            else:
                s2 = voice_activity_trim(wav); s3 = s2[:int(CHUNK_S * SAMPLE_RATE)]
                lbl2, lbl3 = "3. silero VAD", f"4. first {CHUNK_S}s chunk"
            stages = [
                (raw_native, sr_n, f"1. raw file — {info.samplerate} Hz, {info.subtype}"),
                (wav, SAMPLE_RATE, "2. load_audio — 16 kHz mono float32"),
                (s2, SAMPLE_RATE, lbl2),
                (s3, SAMPLE_RATE, lbl3),
            ]
            fig, axes = plt.subplots(len(stages), 1, figsize=(13, 9))
            for ax, (w, sr, lab) in zip(axes, stages):
                librosa.display.waveshow(w, sr=sr, ax=ax, color="#3b6ea5")
                ax.set_title(f"{lab}   →   {len(w):,} samples, {len(w)/sr:.2f} s",
                             fontsize=10, loc="left")
                ax.set_ylabel("amp")
            axes[-1].set_xlabel("time (s)")
            fig.suptitle(f"Preprocessing stages — {ds} · {task} · PD · {row['subject_id']}",
                         fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            p = OUT / f"figB_stages__{ds}__{task}.png"
            fig.savefig(p, dpi=140); plt.close(fig)
            print("wrote", p, flush=True)


def fig_c(manifests):
    """Duration distributions -- motivates the 2 s core / 5 s chunking."""
    fig, axes = plt.subplots(1, len(manifests), figsize=(5 * len(manifests), 4), squeeze=False)
    for ax, (ds, m) in zip(axes[0], manifests.items()):
        for task, colour in [("vowel", "#3b6ea5"), ("speech", "#b5651d")]:
            sel = m[m.task_kind == task]
            if not len(sel):
                continue
            durs = []
            for p in sel["path"]:
                try:
                    i = sf.info(p); durs.append(i.frames / i.samplerate)
                except Exception:  # noqa: BLE001
                    pass
            if durs:
                ax.hist(durs, bins=40, alpha=0.6, label=f"{task} (n={len(durs)})", color=colour)
        ax.axvline(VOWEL_CORE_S, ls="--", c="k", lw=1)
        ax.axvline(CHUNK_S, ls=":", c="k", lw=1)
        ax.set_title(ds); ax.set_xlabel("recording duration (s)"); ax.legend(fontsize=8)
    axes[0][0].set_ylabel("recordings")
    fig.suptitle("Recording durations (dashed = 2 s vowel core, dotted = 5 s speech chunk)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "figC_durations.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name, loader in LOADERS.items():
        try:
            manifests[name] = loader()
            print(f"[{name}] {len(manifests[name]):,} recordings", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] loader failed: {e}", flush=True)
    fig_a(manifests, args.seed)
    fig_b(manifests, args.seed)
    fig_c(manifests)
    print("\nall figures ->", OUT)


if __name__ == "__main__":
    main()
