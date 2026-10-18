"""Extract every XLS-R layer for Bridge2AI, so its layer can be chosen inside Bridge2AI.

Review item A2. Layer 15 was first chosen on the IPVS and NeuroVoz test results
(Section 4.2.2), and the Bridge2AI models are tested on those same datasets. To remove that
dependence, the layer for each Bridge2AI task is chosen by cross-validation inside
Bridge2AI only (scripts/bridge2ai_layer_select.py). That needs all 25 layers.

The manifest and pre-processing are exactly those of transfer_any.get_set: task filter,
quality control, central 2 s for vowels, 5 s windows averaged for other tasks, mean+std
pooling. One forward pass per clip returns all layers.

Output: data/embeddings/bridge2ai__xls-r-300m__<task>__Lall.npz with X (n, 25, 2048)
float16, subject_id and path. Layer 15 is checked against the existing single-layer cache.

Run:  .venv\\Scripts\\python.exe -u scripts\\extract_bridge2ai_layers.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_all_layers import embed_all_layers  # noqa: E402

from pdvoice.config import EMBEDDINGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402
from pdvoice.features.ssl import DEFAULT_MODELS, load_encoder  # noqa: E402
from pdvoice.preprocess import chunks, preprocess_clip, quality_control  # noqa: E402

MODEL = "xls-r-300m"


def main() -> None:
    warnings.simplefilter("ignore")
    model, extractor, device = load_encoder(DEFAULT_MODELS[MODEL])
    base = LOADERS["bridge2ai"]()
    for task in ("vowel", "ddk", "speech"):
        out = EMBEDDINGS / f"bridge2ai__{MODEL}__{task}__Lall.npz"
        if out.exists():
            print(f"already present: {out.name}", flush=True)
            continue
        df = base[base.task_kind == task].copy()
        df, _ = quality_control(df)
        df = df.reset_index(drop=True)
        print(f"{task}: {len(df):,} recordings on {device}", flush=True)
        X, t0 = None, time.time()
        for i, r in enumerate(df.itertuples(index=False)):
            wav = preprocess_clip(r.path, r.task_kind)
            if task == "vowel":
                v = embed_all_layers(wav, model, extractor, device, False)
            else:
                v = np.mean([embed_all_layers(c, model, extractor, device, False)
                             for c in chunks(wav)], axis=0)
            if X is None:
                X = np.zeros((len(df), *v.shape), dtype=np.float16)
            X[i] = v.astype(np.float16)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {task} {i + 1:,}/{len(df):,}  eta {(len(df) - i - 1) / ((i + 1) / el) / 60:.1f} min",
                      flush=True)
        np.savez_compressed(out, X=X, subject_id=df.subject_id.to_numpy().astype(str),
                            path=df.path.to_numpy().astype(str))
        ref = np.load(EMBEDDINGS / f"bridge2ai__{MODEL}__{task}__L15.npz", allow_pickle=True)
        same_rows = np.array_equal(ref["path"].astype(str), df.path.to_numpy().astype(str))
        diff = float(np.abs(ref["X"].astype(np.float32) - X[:, 15].astype(np.float32)).max())
        print(f"DONE {task}: {X.shape} in {(time.time() - t0) / 60:.1f} min; layer-15 check: "
              f"rows match {same_rows}, max abs diff {diff:.4f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
