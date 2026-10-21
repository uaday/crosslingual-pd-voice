"""Design A on Bridge2AI: the same locked runs as for the other three test sets.

The locked mPower head is deterministic for a given seed (re-running IPVS vowel at layer 11
reproduces 0.29708 exactly), so adding Bridge2AI to the existing runs scores it with the
same model that produced the IPVS, NeuroVoz and MDVR-KCL results.

Only the fixed-window configurations are run: Bridge2AI was embedded with the current
windowing only (scripts/extract_bridge2ai_layers.py).

Run:  .venv\\Scripts\\python.exe -u scripts\\run_designA_bridge2ai.py
"""
import subprocess
import sys

PY = sys.executable
RUNS = [
    ["--layer", "11", "--task", "ddk"],
    ["--layer", "11", "--task", "speech"],
    ["--layer", "15", "--task", "vowel"],
    ["--layer", "15", "--task", "ddk"],
    ["--layer", "15", "--task", "speech"],
    ["--task", "vowel"],                                  # final layer (24)
    ["--task", "ddk"],
    ["--task", "speech"],
    ["--task", "vowel", "--adapt", "hc-centroid"],
    ["--task", "vowel", "--adapt", "hc-standardize"],
    ["--task", "speech", "--adapt", "hc-centroid"],
    ["--task", "speech", "--adapt", "hc-standardize"],
]
for r in RUNS:
    cmd = [PY, "-u", "scripts/layer2_transfer.py", "--externals", "bridge2ai", *r]
    print(">>>", " ".join(r), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    tail = [l for l in out.stdout.splitlines() if "bridge2ai" in l or "age-matched target" in l]
    print("\n".join(tail[-3:]), flush=True)
    if out.returncode:
        print(out.stderr[-2000:], flush=True)
print("ALL DONE", flush=True)
