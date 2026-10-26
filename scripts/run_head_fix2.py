"""Second part of the head-fix re-run: the analyses that read the outputs of run_head_fix.py
(difference CIs read the fusion and classifier predictions), the within-mPower window runs
that Table 4.10 and the master table read, and decision curves at a population prevalence.
Logs go to outputs/logs/rerun_head_fix/ next to the first part.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "outputs" / "logs" / "rerun_head_fix"

STEPS = [("difference_cis", ["scripts/difference_cis.py"]),
         ("dca_prev0054", ["scripts/decision_curves.py", "--prevalence", "0.054"])]
for layer in ("11", "15"):
    for window in ("current", "full"):
        STEPS.append((f"l1_L{layer}_{window}",
                      ["scripts/layer1_mpower.py", "--model", "xls-r-300m", "--cap", "3",
                       "--layer", layer, "--window", window]))

with open(LOG / "chain2.log", "w") as chain:
    for name, args in STEPS:
        t0 = time.time()
        with open(LOG / f"{name}.log", "w") as fh:
            rc = subprocess.call([PY, "-u", "-W", "ignore", *args], cwd=ROOT, stdout=fh,
                                 stderr=subprocess.STDOUT, env=dict(os.environ))
        chain.write(f"{name} rc={rc} {(time.time() - t0) / 60:.1f} min\n")
        chain.flush()
    chain.write("CHAIN DONE\n")
