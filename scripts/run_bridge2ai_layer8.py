"""Run the Bridge2AI analyses at the layer chosen inside Bridge2AI (layer 8; review A2).

Waits until the layer-8 caches of the external datasets exist, then runs each analysis
with PDV_LAYER=8. Outputs carry the suffix __L8; the layer-15 results are kept.

Run:  .venv\Scripts\python.exe -u scripts\run_bridge2ai_layer8.py
"""

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
NEED = ROOT / "data" / "embeddings" / "mdvr_kcl__xls-r-300m__speech__L8.npz"
RUNS = [["bridge2ai_agematch.py", "--n-perm", "1000"], ["bridge2ai_transfer.py"],
        ["bridge2ai_transfer_agecheck.py"], ["bridge2ai_followups.py"],
        ["bridge2ai_review_checks.py"]]

while not NEED.exists():
    time.sleep(20)
time.sleep(30)
env = {**os.environ, "PDV_LAYER": "8"}
log = ROOT / "outputs" / "logs" / "b2ai_L8_chain.log"
for cmd in RUNS:
    with open(log, "a") as fh:
        fh.write(f"START {' '.join(cmd)} {time.strftime('%H:%M')}\n")
    with open(ROOT / "outputs" / "logs" / f"b2ai_L8_{cmd[0]}.log", "w", encoding="utf-8") as out:
        rc = subprocess.call([PY, "-u", str(ROOT / "scripts" / cmd[0]), *cmd[1:]],
                             cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    with open(log, "a") as fh:
        fh.write(f"END {cmd[0]} rc={rc} {time.strftime('%H:%M')}\n")
with open(log, "a") as fh:
    fh.write("CHAIN DONE\n")
