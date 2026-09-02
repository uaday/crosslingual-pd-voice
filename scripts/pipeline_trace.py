"""END-TO-END PIPELINE TRACE: one recording, every stage, input -> output.

For a chosen example (or one per dataset/task) this walks the real pipeline and
records at every stage: what goes in, what comes out, the shape, dtype, units,
value range and size on disk/in memory. This is the "show me the process" artefact
-- nothing here is re-implemented, every stage calls the same code the experiments
call.

Usage:
    .venv\\Scripts\\python.exe -u scripts\\pipeline_trace.py
    .venv\\Scripts\\python.exe -u scripts\\pipeline_trace.py --dataset ipvs --task speech
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import soundfile as sf

from pdvoice.config import LOGS, SAMPLE_RATE
from pdvoice.data.ipvs import load_ipvs
from pdvoice.data.mdvr_kcl import load_mdvr_kcl
from pdvoice.data.mpower import load_mpower
from pdvoice.data.neurovoz import load_neurovoz
from pdvoice.preprocess import CHUNK_S, VOWEL_CORE_S, chunks, voice_activity_trim
from pdvoice.utils.audio import load_audio, rms_dbfs, stable_core, trim_silence

LOADERS = {"mpower": load_mpower, "neurovoz": load_neurovoz,
           "ipvs": load_ipvs, "mdvr_kcl": load_mdvr_kcl}
LAYER = 15
MODEL_ID = "facebook/wav2vec2-xls-r-300m"


def describe(x: np.ndarray, sr: int | None = SAMPLE_RATE) -> dict:
    """Shape / dtype / range / duration / memory of an array stage."""
    d = {
        "shape": str(tuple(x.shape)),
        "dtype": str(x.dtype),
        "n_values": int(x.size),
        "min": round(float(np.min(x)), 4),
        "max": round(float(np.max(x)), 4),
        "mean": round(float(np.mean(x)), 4),
        "std": round(float(np.std(x)), 4),
        "bytes": int(x.nbytes),
    }
    if sr and x.ndim == 1:
        d["seconds"] = round(x.size / sr, 3)
        d["rms_dBFS"] = round(rms_dbfs(x), 1)
    return d


def trace_one(row, model, extractor, device) -> pd.DataFrame:
    """Walk one recording through every pipeline stage."""
    import torch

    path, task = row["path"], row["task_kind"]
    info = sf.info(path)
    stages = []

    def add(step, what_in, what_out, desc, note=""):
        stages.append({"step": step, "input": what_in, "output": what_out,
                       "note": note, **desc})

    # --- 0. raw file on disk -------------------------------------------------
    raw_bytes = int(info.frames * info.channels * (24 if "24" in info.subtype else 16) / 8)
    add("0. raw file", f"{path.split(chr(92))[-1]}", f"{info.subtype} PCM",
        {"shape": f"({info.frames},)", "dtype": info.subtype, "n_values": info.frames,
         "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan,
         "bytes": raw_bytes, "seconds": round(info.frames / info.samplerate, 3),
         "rms_dBFS": np.nan},
        f"native {info.samplerate} Hz, {info.channels} ch")

    # --- 1. decode + resample to 16 kHz mono float32 -------------------------
    wav = load_audio(path)
    add("1. load_audio", f"{info.samplerate} Hz {info.subtype}", "16 kHz mono float32",
        describe(wav), f"librosa.load(sr=16000); anti-aliased; values in [-1,1]")

    # --- 2. silence handling (task dependent) --------------------------------
    if task == "vowel":
        trimmed = trim_silence(wav)
        add("2. trim_silence", "16 kHz waveform", "leading/trailing silence removed",
            describe(trimmed), "librosa.effects.trim(top_db=30)")
        clip = stable_core(trimmed, seconds=VOWEL_CORE_S)
        add("3. stable_core", "trimmed waveform", f"central {VOWEL_CORE_S}s core",
            describe(clip), "geometric centre of the phonation")
        clip_list = [clip]
    else:
        trimmed = voice_activity_trim(wav)
        add("2. VAD", "16 kHz waveform", "non-speech regions removed",
            describe(trimmed), "silero-vad (energy-trim fallback)")
        clip_list = list(chunks(trimmed, seconds=CHUNK_S))
        add("3. chunks", "VAD'd waveform", f"{len(clip_list)} x {CHUNK_S}s windows",
            describe(np.asarray(clip_list[0])),
            f"50% overlap; {len(clip_list)} chunks -> embeddings averaged later")

    # --- 4. feature extractor normalisation ----------------------------------
    inputs = extractor(clip_list[0], sampling_rate=SAMPLE_RATE, return_tensors="pt")
    iv = inputs.input_values.numpy()[0]
    add("4. feature extractor", "raw 16 kHz clip", "zero-mean / unit-var waveform",
        describe(iv), "Wav2Vec2FeatureExtractor — NO mel spectrogram; XLS-R eats waveform")

    # --- 5. frozen encoder forward -------------------------------------------
    with torch.no_grad():
        out = model(inputs.input_values.to(device), output_hidden_states=True)
    hs = out.hidden_states[LAYER].squeeze(0).float().cpu().numpy()
    frame_ms = 1000 * len(clip_list[0]) / SAMPLE_RATE / hs.shape[0]
    add("5. XLS-R L15 (frozen)", f"waveform {iv.shape}", f"hidden states (T x D)",
        describe(hs, sr=None),
        f"24-layer transformer, taking layer {LAYER}; ~{frame_ms:.1f} ms per frame "
        f"({hs.shape[0]} frames x {hs.shape[1]} dims)")

    # --- 6. temporal pooling --------------------------------------------------
    vec = np.concatenate([hs.mean(axis=0), hs.std(axis=0)])
    add("6. mean+std pooling", f"(T={hs.shape[0]}, D={hs.shape[1]})", "fixed 2048-dim vector",
        describe(vec, sr=None),
        "mean = average timbre; std = temporal variability (tremor/instability proxy)")

    # --- 7. cache as float16 ---------------------------------------------------
    vec16 = vec.astype(np.float16)
    add("7. float16 cache", "float32 2048-dim", "float16 .npz on disk",
        describe(vec16, sr=None),
        f"halves storage; {vec.nbytes} -> {vec16.nbytes} bytes per recording")

    df = pd.DataFrame(stages)
    df.insert(0, "task", task)
    df.insert(0, "label", "PD" if row["label"] == 1 else "HC")
    df.insert(0, "dataset", row["dataset"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--task", default=None, help="restrict to one task_kind")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from pdvoice.features.ssl import load_encoder
    print(f"loading {MODEL_ID} ...", flush=True)
    model, extractor, device = load_encoder(MODEL_ID)
    print(f"  device={device}", flush=True)

    picks = []
    for name, loader in LOADERS.items():
        if args.dataset and name != args.dataset:
            continue
        m = loader()
        for task in sorted(m["task_kind"].dropna().unique()):
            if args.task and task != args.task:
                continue
            for lab in (1, 0):
                sel = m[(m["task_kind"] == task) & (m["label"] == lab)]
                if len(sel):
                    picks.append(sel.sample(1, random_state=args.seed).iloc[0])

    out = []
    for row in picks:
        print(f"\ntracing {row['dataset']}/{row['task_kind']}/"
              f"{'PD' if row['label'] == 1 else 'HC'} ...", flush=True)
        try:
            out.append(trace_one(row, model, extractor, device))
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

    if not out:
        print("nothing traced"); return
    allt = pd.concat(out, ignore_index=True)
    pd.set_option("display.width", 260, "display.max_columns", 40,
                  "display.max_colwidth", 46)
    for (ds, task, lab), g in allt.groupby(["dataset", "task", "label"], sort=False):
        print(f"\n{'=' * 118}\n{ds}  |  task={task}  |  {lab}\n{'=' * 118}")
        print(g[["step", "input", "output", "shape", "dtype", "seconds",
                 "rms_dBFS", "bytes", "note"]].to_string(index=False))

    LOGS.mkdir(parents=True, exist_ok=True)
    allt.to_csv(LOGS / "pipeline_trace.csv", index=False)
    print(f"\nwrote {LOGS / 'pipeline_trace.csv'}")


if __name__ == "__main__":
    main()
