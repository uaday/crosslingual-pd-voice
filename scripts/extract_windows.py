"""Extract EVERY layer under a CHOSEN WINDOW MODE, in one pass.

Why this exists. Two decisions in the pipeline were never experiments, only inherited
defaults, and both are now known to be questionable:

  * **The layer.** `select_transfer_layer.py` scores all 25 XLS-R layers by
    subject-disjoint CV on mPower alone. Layer 15 -- the one every transfer number in
    this repo was built on, and which was picked by looking at the *test* sets -- ranks
    13th of 25. Layer 11 wins.
  * **The window.** Sustained vowels are cut to a 2 s core, discarding **80%** of every
    10 s mPower recording. Connected speech is cut into 5 s chunks whose embeddings are
    averaged, which by the law of total variance throws away the between-chunk variance
    -- the long-range drift, measured at 0.10-0.13 of the within-chunk std.

Re-extracting once per question would mean paying for audio decoding, VAD and silence
trimming three times over. `output_hidden_states=True` returns every layer from a single
forward pass, so the layer axis is FREE once the audio has been read. This script pays
the audio cost once per window mode and emits all layers, turning two open questions
into one extraction each.

`--window current` is bit-identical to `preprocess_clip` + `chunks`, verified against
them directly, so it reproduces existing caches rather than quietly replacing them.

Output: (n_recordings, n_layers, dim) float16, plus the recording paths (identity guard),
subject ids and labels. Feed it to `slice_windows.py` to materialise the single-layer
caches that `layer2_transfer.py` reads.

Run:
    .venv\\Scripts\\python.exe -u scripts\\extract_windows.py --dataset mpower --task vowel --cap 3 --window full
    .venv\\Scripts\\python.exe -u scripts\\extract_windows.py --dataset ipvs --task speech --window chunk-drift
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from pdvoice.cohort import load_mpower_cohort
from pdvoice.config import EMBEDDINGS, SEED
from pdvoice.data import LOADERS
from pdvoice.preprocess import WINDOW_MODES, pool_windows, quality_control, window_clip

MODELS = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wavlm-large": "microsoft/wavlm-large",
    "hubert-large": "facebook/hubert-large-ll60k",
    "xls-r-300m": "facebook/wav2vec2-xls-r-300m",
}


def cache_path(dataset: str, model: str, task: str, cap, window: str):
    cap_tag = f"__cap{cap}" if cap else ""
    return EMBEDDINGS / f"windows__{dataset}__{model}__{task}{cap_tag}__W{window}.npz"


def build_cohort(dataset: str, task: str, cap: int | None) -> pd.DataFrame:
    """The recordings to extract, in a FIXED order that the cache is keyed on.

    mPower goes through `load_mpower_cohort` so that "the cohort" means the same thing
    here as everywhere else -- in particular QC before capping, which is the ordering
    that keeps the cap from being spent on files QC then deletes (a loss that was not
    class-neutral). Externals mirror `layer2_transfer.py`: filter to the task, then QC.
    """
    if dataset == "mpower":
        df = load_mpower_cohort(cap=cap, seed=SEED)
    else:
        df = LOADERS[dataset]()
        df, _ = quality_control(df)
    df = df[df["task_kind"] == task].copy().reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no '{task}' recordings in {dataset}")
    return df


def load_model(key: str):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(MODELS[key])
    model = AutoModel.from_pretrained(MODELS[key]).to(device).eval()
    return model, extractor, device


def _forward(wav, model, extractor, device, sr):
    """Per-layer (mean, std) over time for one window. Returns two (n_layers, D) arrays."""
    import torch

    inp = extractor(wav, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model(inp.input_values.to(device), output_hidden_states=True)
    means, stds = [], []
    for h in out.hidden_states:
        a = h.squeeze(0).float().cpu().numpy()
        means.append(a.mean(axis=0))
        stds.append(a.std(axis=0))
    return np.vstack(means), np.vstack(stds)


def embed_recording(windows, model, extractor, device, sr, window_mode):
    """All layers for one recording, pooled across its windows.

    A long `full` window is the one place this can run out of VRAM, and a bare OOM would
    kill a multi-thousand-file run near the end. Retry that single recording on the CPU
    instead: it costs seconds, and it keeps the failure from being a silently missing row
    whose absence would quietly change the cohort the results are computed on.
    """
    import torch

    try:
        stats = [_forward(w, model, extractor, device, sr) for w in windows]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        cpu = model.to("cpu")
        stats = [_forward(w, cpu, extractor, "cpu", sr) for w in windows]
        model.to(device)
        print(f"      OOM on a {sum(len(w) for w in windows) / sr:.0f}s recording "
              f"-> recomputed on CPU", flush=True)
    means = np.stack([m for m, _ in stats])
    stds = np.stack([s for _, s in stats])
    return pool_windows(means, stds, window_mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--task", default="vowel", choices=["vowel", "speech", "ddk"])
    ap.add_argument("--cap", type=int, default=None, help="mPower recordings per speaker")
    ap.add_argument("--window", default="current", choices=list(WINDOW_MODES))
    ap.add_argument("--model", default="xls-r-300m", choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    from pdvoice.config import SAMPLE_RATE

    df = build_cohort(args.dataset, args.task, args.cap)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    out = cache_path(args.dataset, args.model, args.task, args.cap, args.window)

    # Resume guard. A length check is not enough: the cohort's row ORDER and even its row
    # SET can change while the count stays put, and X is consumed positionally, so a
    # length-only check hands back embeddings paired with the wrong recordings.
    if out.exists():
        d = np.load(out, allow_pickle=True)
        if "path" in d and list(d["path"].astype(str)) == df.path.tolist():
            print(f"already present and matches this cohort: {out.name}  X={d['X'].shape}")
            return
        print(f"{out.name} exists but does not match this cohort -> re-extracting")

    n_sub = df.subject_id.nunique()
    print(f"cohort   : {len(df):,} recordings / {n_sub:,} speakers "
          f"(PD {int((df.label == 1).sum()):,} / HC {int((df.label == 0).sum()):,})")
    print(f"window   : {args.window}")
    model, extractor, device = load_model(args.model)
    print(f"encoder  : {args.model} on {device}; all layers from one forward pass",
          flush=True)

    X, ok, n_win = None, np.zeros(len(df), bool), np.zeros(len(df), np.int32)
    failures, t0 = [], time.time()
    for i, r in enumerate(df.itertuples()):
        try:
            windows = window_clip(r.path, r.task_kind, args.window, sr=SAMPLE_RATE)
            v = embed_recording(windows, model, extractor, device, SAMPLE_RATE, args.window)
            if X is None:
                X = np.zeros((len(df), *v.shape), dtype=np.float16)
                print(f"  {v.shape[0]} layers x {v.shape[1]} dims per recording")
            X[i], ok[i], n_win[i] = v.astype(np.float16), True, len(windows)
        except Exception as e:  # noqa: BLE001 — record it, never drop it silently
            failures.append((r.path, f"{type(e).__name__}: {e}"))
            print(f"  [{i}] FAILED {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 250 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1:>6}/{len(df)}  {rate:4.1f} rec/s  "
                  f"eta {(len(df)-i-1)/rate/60:5.1f} min", flush=True)

    w = n_win[ok]
    print(f"\nextracted {int(ok.sum()):,}/{len(df):,} in {(time.time()-t0)/60:.1f} min")
    print(f"windows per recording: mean {w.mean():.1f}  median {np.median(w):.0f}  "
          f"max {w.max()}  (total forwards {int(w.sum()):,})")
    if failures:
        print(f"{len(failures)} FAILED — these recordings are absent from the cache:")
        for p, msg in failures[:10]:
            print(f"  {p}  {msg}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X[ok],
             path=df.loc[ok, "path"].to_numpy().astype(str),
             subject_id=df.loc[ok, "subject_id"].to_numpy().astype(str),
             label=df.loc[ok, "label"].to_numpy().astype(np.int8),
             n_windows=n_win[ok], window=np.array(args.window),
             model=np.array(args.model), task=np.array(args.task))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
