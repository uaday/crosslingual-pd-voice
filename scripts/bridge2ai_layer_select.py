"""Choose the XLS-R layer for Bridge2AI using Bridge2AI alone (review item A2).

Layer 15 was originally chosen on the IPVS and NeuroVoz test results, and the Bridge2AI
models are later tested on those datasets. This picks the layer without looking at any
other dataset: for every layer 0 to 24, a linear probe (PCA + logistic regression) is
scored by participant-disjoint 5-fold cross-validation on the age-matched Bridge2AI
cohort (PD vs healthy controls), over five matching seeds, for each task. The layer with
the highest mean AUROC across the three tasks is selected. Matching first means the choice
cannot reward a layer for encoding age.

Needs data/embeddings/bridge2ai__xls-r-300m__<task>__Lall.npz
(scripts/extract_bridge2ai_layers.py). Writes outputs/logs/bridge2ai_layer_select.json and,
for the chosen layer, single-layer caches bridge2ai__xls-r-300m__<task>__L<k>.npz.

Run:  .venv\\Scripts\\python.exe -u scripts\\bridge2ai_layer_select.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from bridge2ai_agematch import subject_scores  # noqa: E402

from pdvoice.cohort import age_match_ids  # noqa: E402
from pdvoice.config import EMBEDDINGS, LOGS  # noqa: E402
from pdvoice.data import LOADERS  # noqa: E402

MODEL = "xls-r-300m"
SEEDS = [42, 43, 44, 45, 46]
TASKS = ["vowel", "ddk", "speech"]


def main() -> None:
    warnings.simplefilter("ignore")
    base = LOADERS["bridge2ai"]()
    grid = {}
    for task in TASKS:
        d = np.load(EMBEDDINGS / f"bridge2ai__{MODEL}__{task}__Lall.npz", allow_pickle=True)
        emb = pd.DataFrame({"path": d["path"].astype(str)})
        emb["row"] = np.arange(len(emb))
        df = emb.merge(base, on="path", how="inner")
        df = df[df.group.isin(["pd", "control"])].dropna(subset=["age"]).reset_index(drop=True)
        df["label"] = df.label.astype(int)
        y = df.label.to_numpy()
        XA = d["X"][df.row.to_numpy()]
        masks = [df.subject_id.isin(age_match_ids(df, seed=s)).to_numpy() for s in SEEDS]
        for layer in range(XA.shape[1]):
            X = XA[:, layer].astype(np.float32)
            v = [roc_auc_score(*subject_scores(df[m].reset_index(drop=True), X[m], y[m], s)[["y", "p"]].T.to_numpy())
                 for s, m in zip(SEEDS, masks)]
            grid.setdefault(task, {})[layer] = round(float(np.mean(v)), 4)
            print(f"  {task:<6} L{layer:<2} {np.mean(v):.3f} +/- {np.std(v, ddof=1):.3f}", flush=True)

    G = pd.DataFrame(grid)
    G["mean"] = G.mean(axis=1)
    best = int(G["mean"].idxmax())
    print("\n" + G.round(3).to_string())
    print(f"\nselected layer {best} (mean {G.loc[best, 'mean']:.3f}); layer 15 mean "
          f"{G.loc[15, 'mean']:.3f}, rank {int(G['mean'].rank(ascending=False)[15])} of {len(G)}")
    (LOGS / "bridge2ai_layer_select.json").write_text(json.dumps(
        {"selected": best, "grid": {t: grid[t] for t in TASKS},
         "mean": {int(k): round(float(v), 4) for k, v in G["mean"].items()}}, indent=2))

    for task in TASKS:
        out = EMBEDDINGS / f"bridge2ai__{MODEL}__{task}__L{best}.npz"
        if not out.exists():
            d = np.load(EMBEDDINGS / f"bridge2ai__{MODEL}__{task}__Lall.npz", allow_pickle=True)
            np.savez_compressed(out, X=d["X"][:, best], subject_id=d["subject_id"], path=d["path"])
            print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
