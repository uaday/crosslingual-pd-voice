"""Re-extract Bridge2AI XLS-R embeddings after removing background noise.

Some Bridge2AI recordings carry electrical buzz, whine and hiss
(scripts/bridge2ai_noise_audit.py). This script applies stationary spectral gating
(pdvoice.preprocess.spectral_gate) to every recording before the usual preprocessing,
and records each recording's level so that silent recordings can be dropped downstream.

Everything else is identical to scripts/extract_bridge2ai_layers.py: same manifest, quality
control, central 2 s for vowels, 5 s windows averaged for other tasks, mean+std pooling.

The clinical sets (IPVS, NeuroVoz, MDVR-KCL) can be cleaned the same way so that transfer
compares clean with clean; the same code without the noise step reproduces their existing
layer-11 caches exactly (cosine 1.0).

Output: data/embeddings/<dataset>_clean__xls-r-300m__<task>__Lall.npz with X (n, 25, 2048)
float16, subject_id, path (the original path, so rows align with the uncleaned cache) and
rms_db (level of the raw recording, dBFS).

Run:  .venv\\Scripts\\python.exe -u scripts\\extract_bridge2ai_clean.py [dataset ...]
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_all_layers import embed_all_layers  # noqa: E402

import pdvoice.preprocess as pre  # noqa: E402
from pdvoice.config import EMBEDDINGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402
from pdvoice.features.ssl import DEFAULT_MODELS, load_encoder  # noqa: E402
from pdvoice.utils.audio import load_audio  # noqa: E402

MODEL = "xls-r-300m"
TASKS = {"bridge2ai": ["vowel", "ddk", "speech"], "ipvs": ["vowel", "speech"],
         "neurovoz": ["vowel", "ddk", "speech"], "mdvr_kcl": ["speech"]}


def main() -> None:
    warnings.simplefilter("ignore")
    model, extractor, device = load_encoder(DEFAULT_MODELS[MODEL])
    levels = {}

    def load_clean(path, sr=pre.SAMPLE_RATE):
        wav = load_audio(path, sr=sr)
        levels[str(path)] = float(20 * np.log10(np.sqrt(np.mean(wav.astype(np.float64) ** 2)) + 1e-12))
        return pre.spectral_gate(wav)

    pre.load_audio = load_clean  # preprocess_clip resolves load_audio from this module
    for ds, task in [(ds, t) for ds in (sys.argv[1:] or ["bridge2ai"]) for t in TASKS[ds]]:
        out = EMBEDDINGS / f"{ds}_clean__{MODEL}__{task}__Lall.npz"
        if out.exists():
            print(f"already present: {out.name}", flush=True)
            continue
        base = LOADERS[ds]()
        df = base[base.task_kind == task].copy()
        df, _ = pre.quality_control(df)
        df = df.reset_index(drop=True)
        print(f"{ds} {task}: {len(df):,} recordings on {device}", flush=True)
        X, t0 = None, time.time()
        for i, r in enumerate(df.itertuples(index=False)):
            try:
                wav = pre.preprocess_clip(r.path, r.task_kind)
                if task == "vowel":
                    v = embed_all_layers(wav, model, extractor, device, False)
                else:
                    v = np.mean([embed_all_layers(c, model, extractor, device, False)
                                 for c in pre.chunks(wav)], axis=0)
            except Exception:  # noqa: BLE001 - an empty recording cannot be embedded
                v = np.full((25, 2048), np.nan)
            if X is None:
                X = np.zeros((len(df), *v.shape), dtype=np.float16)
            X[i] = v.astype(np.float16)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {task} {i + 1:,}/{len(df):,}  eta {(len(df) - i - 1) / ((i + 1) / el) / 60:.1f} min",
                      flush=True)
        rms = np.array([levels.get(str(p), -120.0) for p in df.path])
        np.savez_compressed(out, X=X, subject_id=df.subject_id.to_numpy().astype(str),
                            path=df.path.to_numpy().astype(str), rms_db=rms)
        print(f"DONE {ds} {task}: {X.shape} in {(time.time() - t0) / 60:.1f} min; "
              f"silent (< -55 dBFS): {int((rms < -55).sum())}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
