"""Re-run every analysis that trains the framework head, after the head's training limit was
raised from 200 steps (patience 20) to 1,000 (patience 50) so that it converges on the full
uncorrected mPower cohort. Order: the unified experiments and the analyses built on their
scores first, then the side analyses, then every per-layer transfer run that Appendix A
reads. One log per step in outputs/logs/rerun_head_fix/.
"""
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "outputs" / "logs" / "rerun_head_fix"
LOG.mkdir(parents=True, exist_ok=True)
L11 = {"PDV_LAYER": "11"}

STEPS = [
    ("unified", ["scripts/unified_experiments.py"], {}),
    ("task_comparison", ["scripts/task_comparison.py"], {}),
    ("exp1_changes", ["scripts/exp1_changes.py"], {"PYTHONPATH": "scripts"}),
    ("country_check", ["scripts/bridge2ai_country_check.py"], {}),
    ("equal_audio", ["scripts/equal_audio.py"], {}),
    ("classifier_comparison", ["scripts/classifier_comparison.py"], {}),
    ("decision_curves", ["scripts/decision_curves.py"], {}),
    ("fusion", ["scripts/fusion.py"], {}),
    ("classical_transfer", ["scripts/classical_transfer.py", "--cap", "3"], {}),
    ("demographic_erasure", ["scripts/demographic_erasure.py"], L11),
    ("adapt_centroid", ["scripts/label_sensitivity.py", "--within-layer", "11", "--transfer-layer",
                        "11", "--scenarios", "baseline,agematched", "--adapt", "hc-centroid",
                        "--tag", "L11_adapt_centroid"], {}),
    ("adapt_standardize", ["scripts/label_sensitivity.py", "--within-layer", "11",
                           "--transfer-layer", "11", "--scenarios", "baseline,agematched",
                           "--adapt", "hc-standardize", "--tag", "L11_adapt_standardize"], {}),
]
# every per-layer transfer run Appendix A reads, rebuilt with the same arguments
pat = re.compile(r"layer2__xls-r-300m__(\w+?)__(\w+?)__L(\w+?)(__Wfull)?(__adapt-[\w-]+)?\.json$")
for f in sorted((ROOT / "outputs" / "logs").glob("layer2__xls-r-300m__*.json")):
    m = pat.match(f.name)
    if not m or "agematched-target" in f.name:
        continue
    ds, task, layer, wfull, adapt = m.groups()
    args = ["scripts/layer2_transfer.py", "--externals", ds, "--task", task]
    if layer != "None":
        args += ["--layer", layer]
    if wfull:
        args += ["--window", "full"]
    if adapt:
        args += ["--adapt", adapt.replace("__adapt-", "")]
    STEPS.append((f"l2_{f.stem.replace('layer2__xls-r-300m__', '')}", args, {}))

with open(LOG / "chain.log", "w") as chain:
    for name, args, env in STEPS:
        t0 = time.time()
        with open(LOG / f"{name}.log", "w") as fh:
            rc = subprocess.call([PY, "-u", "-W", "ignore", *args], cwd=ROOT, stdout=fh,
                                 stderr=subprocess.STDOUT, env={**os.environ, **env})
        chain.write(f"{name} rc={rc} {(time.time() - t0) / 60:.1f} min\n")
        chain.flush()
    chain.write("CHAIN DONE\n")
