"""Extract XLS-R embeddings for EVERY labelled mPower recording, once.

The pilot caches were keyed by (cap, seed), so changing either meant re-extracting.
This extracts the whole labelled set a single time; the scale-up then caps,
age-matches and folds *in memory* by indexing into this matrix, which makes an
arbitrary number of seeds cheap.

Unlike the pilot caches this stores **ids alongside X** (subject_id + path), so
alignment is verifiable instead of positional. See scripts/scaleup_agematched.py.

Run:  .venv\\Scripts\\python.exe -u scripts\\extract_mpower_full.py
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from pdvoice.config import EMBEDDINGS, SAMPLE_RATE
from pdvoice.data.mpower import load_mpower
from pdvoice.preprocess import preprocess_clip, quality_control

MODEL_ID, MODEL_KEY, LAYER = "facebook/wav2vec2-xls-r-300m", "xls-r-300m", 15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N recordings")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    out = EMBEDDINGS / f"mpower_full__{MODEL_KEY}__L{args.layer}.npz"
    if out.exists():
        d = np.load(out, allow_pickle=True)
        print(f"cache already present: {out.name}  X={d['X'].shape}")
        return

    df = load_mpower()
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    df, dropped = quality_control(df)
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"labelled + QC'd recordings : {len(df):,}  (QC dropped {len(dropped):,})")
    print(f"subjects                   : {df.subject_id.nunique():,}")

    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from pdvoice.features.ssl import REVISIONS
    rev = REVISIONS.get(MODEL_ID)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=rev)
    model = AutoModel.from_pretrained(MODEL_ID, revision=rev).to(device).eval()
    print(f"encoder on {device}; extracting layer {args.layer} ...", flush=True)

    vecs = np.zeros((len(df), 2048), dtype=np.float16)
    ok = np.zeros(len(df), dtype=bool)
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        try:
            wav = preprocess_clip(r.path, r.task_kind)
            inputs = extractor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                hs = model(inputs.input_values.to(device),
                           output_hidden_states=True).hidden_states[args.layer]
            h = hs.squeeze(0).float().cpu().numpy()
            vecs[i] = np.concatenate([h.mean(0), h.std(0)]).astype(np.float16)
            ok[i] = True
        except Exception as e:  # noqa: BLE001 — log and continue; a dead file must not kill 15 h of work
            print(f"  [{i}] FAILED {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"  {i+1:>6,}/{len(df):,}  {rate:5.1f} rec/s  "
                  f"elapsed {el/60:5.1f} min  eta {(len(df)-i-1)/rate/60:5.1f} min", flush=True)

    print(f"\nextracted {int(ok.sum()):,} / {len(df):,}  ({int((~ok).sum()):,} failed)")
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=vecs[ok],
        subject_id=df.loc[ok, "subject_id"].to_numpy().astype(str),
        path=df.loc[ok, "path"].to_numpy().astype(str),
        label=df.loc[ok, "label"].to_numpy().astype(np.int8),
    )
    print(f"wrote {out}  ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
