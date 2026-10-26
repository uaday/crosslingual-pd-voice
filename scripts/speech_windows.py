"""How many analysis windows the encoder sees per connected-speech recording.

Runs the pipeline's own window_clip ("current" mode: voice-activity detection, then 5 s
windows at 50% overlap) on exactly the speech recordings the experiments use (quality
control and the common-task filter), and reports the mean and median per dataset.

Run:  .venv\\Scripts\\python.exe scripts\\speech_windows.py
Writes outputs/logs/speech_windows.json
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from pdvoice.config import LOGS
from pdvoice.data import LOADERS
from pdvoice.data.common_tasks import common_task_mask
from pdvoice.preprocess import quality_control, window_clip


def main() -> None:
    warnings.simplefilter("ignore")
    out = {}
    for ds in ("neurovoz", "ipvs", "mdvr_kcl", "bridge2ai"):
        df = LOADERS[ds]()
        df = df[df.task_kind == "speech"].copy()
        df, _ = quality_control(df)
        if ds == "bridge2ai":
            df = df[df.group.isin(["pd", "control"]) & (df.country == "USA")]
        df = df[common_task_mask(df.reset_index(drop=True), ds).to_numpy()
                if hasattr(common_task_mask(df.reset_index(drop=True), ds), "to_numpy")
                else common_task_mask(df.reset_index(drop=True), ds)]
        n = [len(window_clip(p, "speech")) for p in df.path]
        out[ds] = {"recordings": len(n), "mean_windows": float(np.mean(n)),
                   "median_windows": float(np.median(n))}
        print(ds, out[ds], flush=True)
    (LOGS / "speech_windows.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
