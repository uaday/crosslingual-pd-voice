"""Review item A8: the depth gradient under one protocol for both cohorts.

Waits until the GPU is free of the layer-8 extraction, extracts all XLS-R layers for the
confounded mPower cohort (one recording per speaker), then scores every layer on both
cohorts with the same head and cross-validation (select_transfer_layer.py).

Run:  .venv\Scripts\python.exe -u scripts\run_depth_gradient.py
"""

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
WAIT = ROOT / "data" / "embeddings" / "mdvr_kcl__xls-r-300m__speech__L8.npz"
RUNS = [["extract_all_layers.py", "--model", "xls-r-300m", "--cohort", "confounded"],
        ["select_transfer_layer.py", "--cohort", "agematched"],
        ["select_transfer_layer.py", "--cohort", "confounded"]]

while not WAIT.exists():
    time.sleep(20)
time.sleep(30)
log = ROOT / "outputs" / "logs" / "depth_gradient_chain.log"
for cmd in RUNS:
    with open(log, "a") as fh:
        fh.write(f"START {' '.join(cmd)} {time.strftime('%H:%M')}\n")
    with open(ROOT / "outputs" / "logs" / f"depth_{cmd[0]}_{cmd[-1]}.log", "w", encoding="utf-8") as out:
        rc = subprocess.call([PY, "-u", str(ROOT / "scripts" / cmd[0]), *cmd[1:]],
                             cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    with open(log, "a") as fh:
        fh.write(f"END {cmd[0]} {cmd[-1]} rc={rc} {time.strftime('%H:%M')}\n")
with open(log, "a") as fh:
    fh.write("CHAIN DONE\n")
