"""Extract EVERY layer of an encoder in one pass, for a fixed comparison cohort.

Why this exists. The Layer-1 sweep gave XLS-R four layers and wav2vec2-base exactly one
-- its last, which is the single configuration we have direct evidence is the worst
choice for that model family (moving to a mid layer gained +0.099 for WavLM and +0.093
for HuBERT). "XLS-R wins" was therefore partly a statement about search effort. Worse,
the sweep ran on the CONFOUNDED cohort, so it may partly have been measuring which
encoder best encodes age.

`output_hidden_states=True` returns all layers from a single forward pass, so an equal
budget for every encoder costs no more than an unequal one. Cohort is fixed and
age-matched, one recording per speaker, so every encoder sees identical audio.

Output: (n_recordings, n_layers, dim) float16, plus the speaker ids and labels.

Run:
    .venv\\Scripts\\python.exe -u scripts\\extract_all_layers.py --model xls-r-300m
    .venv\\Scripts\\python.exe -u scripts\\extract_all_layers.py --model whisper-small
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from pdvoice.cohort import age_match_ids, load_mpower_cohort, subject_age_stats
from pdvoice.config import EMBEDDINGS, SAMPLE_RATE
from pdvoice.preprocess import preprocess_clip

MODELS = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wavlm-large": "microsoft/wavlm-large",
    "hubert-large": "facebook/hubert-large-ll60k",
    "xls-r-300m": "facebook/wav2vec2-xls-r-300m",
    "whisper-small": "openai/whisper-small",
}
WHISPER_FRAMES_PER_SEC = 50.0   # 30 s of audio -> 1500 encoder frames


def build_cohort(seed=42):
    """Age-matched, one recording per speaker. Identical for every encoder."""
    df = load_mpower_cohort(cap=1, seed=seed)
    keep = age_match_ids(df, seed=seed)
    return df[df.subject_id.isin(keep)].reset_index(drop=True)


def load_model(key):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    mid = MODELS[key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(mid)
    if key.startswith("whisper"):
        from transformers import WhisperModel
        model = WhisperModel.from_pretrained(mid).encoder.to(device).eval()
    else:
        model = AutoModel.from_pretrained(mid).to(device).eval()
    return model, extractor, device


def embed_all_layers(wav, model, extractor, device, is_whisper):
    """(n_layers, dim) -- mean+std pooled per layer, from one forward pass."""
    import torch

    if is_whisper:
        # Whisper pads every input to 30 s. Pooling over all 1500 frames would average
        # mostly padding, so keep only the frames the real audio occupies.
        n_frames = max(1, int(round(len(wav) / SAMPLE_RATE * WHISPER_FRAMES_PER_SEC)))
        inp = extractor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            out = model(inp.input_features.to(device), output_hidden_states=True)
        hs = [h.squeeze(0)[:n_frames] for h in out.hidden_states]
    else:
        inp = extractor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            out = model(inp.input_values.to(device), output_hidden_states=True)
        hs = [h.squeeze(0) for h in out.hidden_states]

    vecs = []
    for h in hs:
        a = h.float().cpu().numpy()
        vecs.append(np.concatenate([a.mean(0), a.std(0)]))
    return np.vstack(vecs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    out = EMBEDDINGS / f"layersweep__{args.model}__agematched.npz"
    if out.exists():
        d = np.load(out, allow_pickle=True)
        print(f"already present: {out.name}  X={d['X'].shape}")
        return

    df = build_cohort(args.seed)
    print(f"cohort: {len(df):,} recordings / {df.subject_id.nunique():,} speakers "
          f"(PD {int((df.label == 1).sum()):,} / HC {int((df.label == 0).sum()):,})")
    _a = subject_age_stats(df)   # per speaker, not per recording
    print(f"        age PD {_a['age_pd']:.1f} / HC {_a['age_hc']:.1f} "
          f"(gap {_a['age_gap']:+.1f} yr)")

    model, extractor, device = load_model(args.model)
    is_whisper = args.model.startswith("whisper")
    print(f"{args.model} on {device}; extracting ALL layers in one pass ...", flush=True)

    X, ok, t0 = None, np.zeros(len(df), bool), time.time()
    for i, r in enumerate(df.itertuples()):
        try:
            wav = preprocess_clip(r.path, r.task_kind)
            v = embed_all_layers(wav, model, extractor, device, is_whisper)
            if X is None:
                X = np.zeros((len(df), *v.shape), dtype=np.float16)
                print(f"  {v.shape[0]} layers x {v.shape[1]} dims per recording")
            X[i] = v.astype(np.float16)
            ok[i] = True
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] FAILED {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"  {i+1:>5}/{len(df)}  {(i+1)/el:4.1f} rec/s  eta {(len(df)-i-1)/((i+1)/el)/60:4.1f} min",
                  flush=True)

    print(f"\nextracted {int(ok.sum()):,}/{len(df):,} in {(time.time()-t0)/60:.1f} min")
    np.savez_compressed(out, X=X[ok],
                        subject_id=df.loc[ok, "subject_id"].to_numpy().astype(str),
                        label=df.loc[ok, "label"].to_numpy().astype(np.int8),
                        age=df.loc[ok, "age"].to_numpy().astype(np.float32))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
