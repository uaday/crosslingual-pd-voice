"""Re-run every analysis that reads connected speech or the Bridge2AI cohort, after
restricting speech to the tasks both groups recorded (pdvoice.data.common_tasks), the
Bridge2AI disease groups to confirmed diagnoses, and the Bridge2AI controls to those with
no confirmed diagnosis. One log per step in outputs/logs/rerun_common_tasks/.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "outputs" / "logs" / "rerun_common_tasks"
LOG.mkdir(parents=True, exist_ok=True)
L11 = {"PDV_LAYER": "11"}

STEPS = [
    ("unified", ["scripts/unified_experiments.py"], {}),
    ("b2ai_review_checks", ["scripts/bridge2ai_review_checks.py"], L11),
    ("age_bin_width", ["scripts/age_bin_width.py"], L11),
    ("b2ai_layer_select", ["scripts/bridge2ai_layer_select.py"], {}),
]
for ds in ("ipvs", "neurovoz"):
    STEPS += [(f"l2_{ds}_speech_L11", ["scripts/layer2_transfer.py", "--externals", ds,
                                       "--task", "speech", "--layer", "11"], {}),
              (f"l2_{ds}_speech_L11_Wfull", ["scripts/layer2_transfer.py", "--externals", ds,
                                             "--task", "speech", "--layer", "11",
                                             "--window", "full"], {}),
              (f"l2_{ds}_speech_L15_Wfull", ["scripts/layer2_transfer.py", "--externals", ds,
                                             "--task", "speech", "--layer", "15",
                                             "--window", "full"], {})]
for task in ("vowel", "ddk", "speech"):
    for layer in ("11", "15", None):
        STEPS.append((f"l2_b2ai_{task}_L{layer}", ["scripts/layer2_transfer.py", "--externals",
                      "bridge2ai", "--task", task] + (["--layer", layer] if layer else []), {}))
for task in ("vowel", "speech"):
    for adapt in ("hc-centroid", "hc-standardize"):
        STEPS.append((f"l2_b2ai_{task}_{adapt}", ["scripts/layer2_transfer.py", "--externals",
                      "bridge2ai", "--task", task, "--adapt", adapt], {}))

with open(LOG / "chain.log", "w") as chain:
    for name, args, env in STEPS:
        t0 = time.time()
        with open(LOG / f"{name}.log", "w") as fh:
            rc = subprocess.call([PY, "-u", "-W", "ignore", *args], cwd=ROOT, stdout=fh,
                                 stderr=subprocess.STDOUT, env={**os.environ, **env})
        chain.write(f"{name} rc={rc} {(time.time() - t0) / 60:.1f} min\n")
        chain.flush()
    chain.write("CHAIN DONE\n")
