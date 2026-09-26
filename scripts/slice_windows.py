"""Materialise single-layer caches from an all-layer window sweep.

`extract_windows.py` writes one (n, n_layers, dim) file per window mode. The downstream
scripts -- `layer2_transfer.py`, `layer1_mpower.py` -- each read a flat (n, dim) cache for
ONE layer. This slices the former into the latter, so choosing a different layer costs a
disk read instead of another pass over the audio.

The emitted file carries the `path` array, because that is what the consumers' identity
guard checks: X is consumed positionally, so a cache whose rows are in a different order
than the cohort silently pairs embeddings with the wrong recordings.

Naming. Under `--window current` the legacy names are emitted, since that mode is
bit-identical to the old pipeline -- but an existing file is never overwritten without
`--force`, because those caches are what every published number in this repo was computed
from and a silent replacement would be unauditable. Other window modes get a `__W<mode>`
suffix and cannot collide.

Run:
    .venv\\Scripts\\python.exe scripts\\slice_windows.py --dataset ipvs --task speech --window full --layers 11,15
"""

from __future__ import annotations

import argparse

import numpy as np

from pdvoice.config import EMBEDDINGS, SEED
from pdvoice.preprocess import WINDOW_MODES

from extract_windows import MODELS, cache_path  # noqa: E402  (same directory)


def target_name(dataset, model, task, cap, window, layer):
    """Where a consumer script will look for this layer's flat cache."""
    suffix = "" if window == "current" else f"__W{window}"
    if dataset == "mpower":
        # layer2_transfer.py / layer1_mpower.py build this exact stem
        return EMBEDDINGS / f"mpower_pilot__{model}__cap{cap}__L{layer}__seed{SEED}{suffix}.npz"
    return EMBEDDINGS / f"{dataset}__{model}__{task}__L{layer}{suffix}.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", default="vowel")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--window", default="current", choices=list(WINDOW_MODES))
    ap.add_argument("--model", default="xls-r-300m", choices=sorted(MODELS))
    ap.add_argument("--layers", default="11,15",
                    help="comma-separated layer indices, or 'all'")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing flat cache (refused by default)")
    args = ap.parse_args()

    src = cache_path(args.dataset, args.model, args.task, args.cap, args.window)
    if not src.exists():
        raise SystemExit(f"missing {src}\nRun: scripts/extract_windows.py --dataset "
                         f"{args.dataset} --task {args.task} --window {args.window}")

    d = np.load(src, allow_pickle=True)
    X, paths = d["X"], d["path"].astype(str)
    n, n_layers, dim = X.shape
    print(f"{src.name}: {n:,} recordings x {n_layers} layers x {dim} dims "
          f"(window={str(d['window'])})")

    layers = range(n_layers) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    for layer in layers:
        if not 0 <= layer < n_layers:
            raise SystemExit(f"layer {layer} out of range (0..{n_layers - 1})")
        out = target_name(args.dataset, args.model, args.task, args.cap, args.window, layer)
        if out.exists() and not args.force:
            print(f"  L{layer:<2} SKIP (exists, use --force): {out.name}")
            continue
        np.savez_compressed(out, X=X[:, layer, :], path=paths)
        print(f"  L{layer:<2} -> {out.name}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
