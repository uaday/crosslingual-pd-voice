"""GROUP-level spectra: PD vs HC averaged over many speakers, with confidence bands.

The four-views figures in voice_fundamentals_figs.py are single recordings — correct
for showing what a spectrogram IS, useless for claiming a PD/HC difference. This is
the figure that can support such a claim.

You cannot average spectrograms across speakers (no time alignment between different
people). The standard analogue is the Long-Term Average Spectrum: average power over
time within a recording, then over recordings within a speaker, then over speakers.

Two design decisions that decide whether the result means anything:

  1. AGE-MATCHED cohort. mPower controls are ~27 years younger than its patients, and
     age changes the voice. On the raw cohort any difference found here is mostly the
     age confound, not disease.
  2. Each recording's spectrum is normalised to unit total power before averaging, so
     we compare spectral SHAPE, not recording gain. Microphone and distance vary far
     more between speakers than disease does.

Run:  .venv\\Scripts\\python.exe -u scripts\\group_spectra.py --per-group 150
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

from pdvoice.config import FIGURES, LOGS, MPOWER_DIR, SAMPLE_RATE  # noqa: E402
from pdvoice.data.mpower import load_mpower  # noqa: E402
from pdvoice.data.neurovoz import load_neurovoz  # noqa: E402
from pdvoice.preprocess import quality_control  # noqa: E402
from pdvoice.utils.audio import load_audio, stable_core, trim_silence  # noqa: E402

N_FFT, HOP = 1024, 160
WAVE, SPEC = "#3B6EA5", "#B5651D"


def ltas(path, seconds=2.0):
    """Long-term average spectrum of one recording, normalised to unit total power."""
    w = stable_core(trim_silence(load_audio(path)), sr=SAMPLE_RATE, seconds=seconds)
    if len(w) < N_FFT:
        return None
    S = np.abs(librosa.stft(w, n_fft=N_FFT, hop_length=HOP)) ** 2
    p = S.mean(axis=1)                      # average power per frequency bin over time
    tot = p.sum()
    if not np.isfinite(tot) or tot <= 0:
        return None
    return p / tot                          # shape only — removes gain differences


def age_match(subj, seed, bin_width=5):
    rng = np.random.default_rng(seed)
    s = subj.dropna(subset=["age"]).assign(bin=lambda d: (d.age // bin_width * bin_width).astype(int))
    keep = []
    for _, g in s.groupby("bin"):
        p_, h_ = g[g.label == 1].subject_id.to_numpy(), g[g.label == 0].subject_id.to_numpy()
        k = min(len(p_), len(h_))
        if k:
            keep += rng.choice(p_, k, replace=False).tolist() + rng.choice(h_, k, replace=False).tolist()
    return set(keep)


def collect(df, per_group, seed):
    """One recording per subject -> LTAS; returns {label: (n_subjects, array)}."""
    rng = np.random.default_rng(seed)
    out = {}
    for lab in (0, 1):
        ids = df[df.label == lab].subject_id.unique()
        if len(ids) > per_group:
            ids = rng.choice(ids, per_group, replace=False)
        rows = []
        for sid in ids:
            p = df[df.subject_id == sid].path.iloc[0]
            v = ltas(p)
            if v is not None:
                rows.append(v)
        out[lab] = np.vstack(rows)
        print(f"    label {lab}: {len(rows)} speakers", flush=True)
    return out


def panel(ax, res, freqs, title, note):
    for lab, colour, name in [(0, WAVE, "HC"), (1, SPEC, "PD")]:
        A = 10 * np.log10(np.maximum(res[lab], 1e-12))
        m, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
        ax.plot(freqs, m, color=colour, lw=1.4, label=f"{name} (n={len(A)})")
        ax.fill_between(freqs, m - 1.96 * se, m + 1.96 * se, color=colour, alpha=.25, lw=0)
    ax.set_xscale("log"); ax.set_xlim(60, 8000)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("relative power (dB)")
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.legend(fontsize=8)
    ax.text(.02, .03, note, transform=ax.transAxes, fontsize=7.5, color="#666")


def diff_panel(ax, res, freqs, title):
    A0 = 10 * np.log10(np.maximum(res[0], 1e-12))
    A1 = 10 * np.log10(np.maximum(res[1], 1e-12))
    d = A1.mean(0) - A0.mean(0)
    se = np.sqrt(A1.var(0, ddof=1) / len(A1) + A0.var(0, ddof=1) / len(A0))
    ax.plot(freqs, d, color="#333", lw=1.3)
    ax.fill_between(freqs, d - 1.96 * se, d + 1.96 * se, color="#888", alpha=.3, lw=0)
    ax.axhline(0, color="k", lw=.8, ls="--")
    ax.set_xscale("log"); ax.set_xlim(60, 8000)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PD − HC (dB)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    sig = np.abs(d) > 1.96 * se
    ax.text(.02, .93, f"{100*sig.mean():.0f}% of bins differ at 95%",
            transform=ax.transAxes, fontsize=8, color="#333")
    return d, se, sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

    # ---------- mPower: raw vs age-matched, to show the confound directly ----------
    mp = load_mpower()
    mp = mp[mp.label.isin([0, 1])]
    mp = mp[mp.groupby("subject_id").cumcount() < 1]
    mp, _ = quality_control(mp); mp = mp.reset_index(drop=True)
    demo = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    age = dict(zip(demo.healthCode.astype(str), demo.age))
    mp["age"] = mp.subject_id.str.replace("mpower_", "", regex=False).map(age)

    subj = mp.drop_duplicates("subject_id")[["subject_id", "label", "age"]]
    keep = age_match(subj, args.seed)
    mp_am = mp[mp.subject_id.isin(keep)]
    print(f"mPower raw       : {subj.subject_id.nunique():,} subjects "
          f"(PD age {subj[subj.label==1].age.mean():.1f}, HC age {subj[subj.label==0].age.mean():.1f})")
    print(f"mPower age-match : {mp_am.subject_id.nunique():,} subjects "
          f"(PD age {mp_am[mp_am.label==1].age.mean():.1f}, HC age {mp_am[mp_am.label==0].age.mean():.1f})")

    print("  [raw cohort] computing LTAS ...", flush=True)
    res_raw = collect(mp, args.per_group, args.seed)
    print("  [age-matched] computing LTAS ...", flush=True)
    res_am = collect(mp_am, args.per_group, args.seed)

    # ---------- NeuroVoz: clinician labels, small age gap ----------
    nv = load_neurovoz()
    nv = nv[(nv.task_kind == "vowel") & (nv.task_code.str.startswith("A"))]
    nv, _ = quality_control(nv); nv = nv.reset_index(drop=True)
    print(f"NeuroVoz /a/     : {nv.subject_id.nunique()} subjects")
    print("  computing LTAS ...", flush=True)
    res_nv = collect(nv, args.per_group, args.seed)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    panel(axes[0][0], res_raw, freqs, "mPower /a/ — RAW cohort",
          "PD ~27 yrs older than HC:\nany gap here is mostly AGE")
    panel(axes[0][1], res_am, freqs, "mPower /a/ — AGE-MATCHED",
          "5-yr bins, ages equal:\nthis is the disease-only comparison")
    panel(axes[0][2], res_nv, freqs, "NeuroVoz /a/ — clinician labels",
          "Spanish, ~7-yr age gap,\nclinical microphone")
    d1, _, _ = diff_panel(axes[1][0], res_raw, freqs, "difference — RAW")
    d2, _, _ = diff_panel(axes[1][1], res_am, freqs, "difference — AGE-MATCHED")
    d3, _, _ = diff_panel(axes[1][2], res_nv, freqs, "difference — NeuroVoz")
    fig.suptitle("Long-Term Average Spectrum, PD vs HC — averaged over speakers, "
                 "shaded band = 95% CI of the mean", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    out = FIGURES / "writeup" / "figD_group_spectra.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}")

    pd.DataFrame({"freq_hz": freqs, "raw_PD_minus_HC_dB": d1,
                  "agematched_PD_minus_HC_dB": d2, "neurovoz_PD_minus_HC_dB": d3}
                 ).to_csv(LOGS / "group_spectra.csv", index=False)
    print(f"wrote {LOGS / 'group_spectra.csv'}")


if __name__ == "__main__":
    main()
