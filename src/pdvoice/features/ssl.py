"""Frozen self-supervised speech embeddings — the default encoder path.

Run each encoder once in inference mode, pool the hidden states to a fixed-length
vector, and cache to disk (float16). The shallow head then trains on the cache.
This fits the 5 GB vGPU and matches the field's dominant recipe. wav2vec2 / WavLM / HuBERT / XLS-R share this code path; Whisper uses a
different (encoder-decoder + log-mel) API and is a TODO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdvoice.config import EMBEDDINGS, SAMPLE_RATE

# short key -> HuggingFace model id
DEFAULT_MODELS = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wavlm-large": "microsoft/wavlm-large",
    "hubert-large": "facebook/hubert-large-ll60k",
    "xls-r-300m": "facebook/wav2vec2-xls-r-300m",
}


# Exact Hugging Face commits used for every cached embedding (REPRODUCE.md). Passing them
# to from_pretrained pins the weights: without it a fresh download takes whatever the
# model's main branch holds that day.
REVISIONS = {
    "facebook/wav2vec2-xls-r-300m": "1a640f32ac3e39899438a2931f9924c02f080a54",
    "facebook/wav2vec2-base": "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
    "facebook/hubert-large-ll60k": "ff022d095678a2995f3c49bab18a96a9e553f782",
    "microsoft/wavlm-large": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
    "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
}


def select_device(device: str | None = None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pool(hidden: np.ndarray, method: str = "mean_std") -> np.ndarray:
    """Temporal pooling of a (T, D) hidden-state matrix to a fixed vector."""
    if method == "mean":
        return hidden.mean(axis=0)
    if method == "std":
        return hidden.std(axis=0)
    if method == "mean_std":
        return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)])
    raise ValueError(f"unknown pooling: {method}")


def load_encoder(model_id: str, device: str | None = None):
    from transformers import AutoFeatureExtractor, AutoModel

    device = select_device(device)
    rev = REVISIONS.get(model_id)
    extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=rev)
    model = AutoModel.from_pretrained(model_id, revision=rev).to(device).eval()
    return model, extractor, device


def _embed_wave(wav, model, extractor, device, layer, pooling, sr=SAMPLE_RATE) -> np.ndarray:
    import torch

    inputs = extractor(wav, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values.to(device), output_hidden_states=True)
    hs = out.hidden_states[layer] if layer is not None else out.last_hidden_state
    return pool(hs.squeeze(0).float().cpu().numpy(), pooling)


def extract_for_manifest(
    df: pd.DataFrame,
    model_key: str = "wav2vec2-base",
    layer: int | None = None,
    pooling: str = "mean_std",
    device: str | None = None,
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one pooled embedding per recording. Returns (X, row_index).

    Connected-speech clips are chunked and their embeddings averaged. Results are
    cached under data/embeddings/ keyed by dataset/model/pooling/layer.
    """
    from pdvoice.preprocess import chunks, preprocess_clip

    if df.empty:
        return np.empty((0, 0), dtype=np.float16), np.array([])

    dataset = str(df["dataset"].iloc[0])
    cache_path = EMBEDDINGS / f"{dataset}__{model_key}__{pooling}__L{layer}.npz"
    paths = df["path"].astype(str).to_numpy()
    if cache and cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        # Identity guard: the cache is keyed only by dataset/model/pooling/layer, so it
        # must also hold exactly these recordings. Rows are returned in df's order; a
        # cache that lacks any requested recording (or predates the stored paths) is
        # rebuilt instead of being returned as-is.
        if "path" in d:
            row = {p: i for i, p in enumerate(d["path"].astype(str))}
            if all(p in row for p in paths):
                take = np.array([row[p] for p in paths], dtype=int)
                return d["X"][take], df.index.to_numpy()

    model_id = DEFAULT_MODELS.get(model_key, model_key)
    model, extractor, device = load_encoder(model_id, device)

    vectors, ids = [], []
    for idx, r in df.iterrows():
        wav = preprocess_clip(r["path"], r["task_kind"])
        if r["task_kind"] == "vowel":
            vec = _embed_wave(wav, model, extractor, device, layer, pooling)
        else:  # chunk-and-average for connected speech
            chunk_vecs = [_embed_wave(c, model, extractor, device, layer, pooling) for c in chunks(wav)]
            vec = np.mean(chunk_vecs, axis=0)
        vectors.append(vec)
        ids.append(idx)

    X = np.vstack(vectors).astype(np.float16)
    ids = np.asarray(ids)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, ids=ids, path=paths)
    return X, ids


def load_aligned(cache_path, paths, key="X"):
    """Rows of a cached embedding file, matched to ``paths`` by recording path.

    The cache must store a ``path`` array. Rows are returned in the order of ``paths``
    (reordered if needed). A cache without paths, or one missing any requested recording,
    raises ValueError: a row count or speaker sequence cannot tell two recordings apart.
    """
    d = np.load(cache_path, allow_pickle=True)
    if "path" not in d.files:
        raise ValueError(f"{cache_path} has no recording paths; re-extract it")
    row = {q: i for i, q in enumerate(d["path"].astype(str))}
    want = [str(q) for q in paths]
    missing = [q for q in want if q not in row]
    if missing:
        raise ValueError(f"{cache_path}: {len(missing)} requested recordings "
                         f"are not in the cache (e.g. {missing[0]})")
    return d[key][[row[q] for q in want]]
