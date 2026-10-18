"""Review A3: the five-seed experiments with within-dataset and transfer at layer 15.

Writes label_sensitivity.json and label_sensitivity__agesex.json, which the thesis tables
and the labels-vs-age figure read, at a single layer.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
for extra in ([], ["--scenarios", "baseline,agematched,agesexmatched", "--tag", "agesex"]):
    with open(ROOT / "outputs" / "logs" / f"samelayer_{'agesex' if extra else 'main'}.log", "w") as fh:
        rc = subprocess.call([PY, "-u", str(ROOT / "scripts" / "label_sensitivity.py"),
                              "--within-layer", "15", "--transfer-layer", "15", *extra],
                             cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    with open(ROOT / "outputs" / "logs" / "samelayer_chain.log", "a") as fh:
        fh.write(f"{'agesex' if extra else 'main'} rc={rc}\n")
with open(ROOT / "outputs" / "logs" / "samelayer_chain.log", "a") as fh:
    fh.write("CHAIN DONE\n")
