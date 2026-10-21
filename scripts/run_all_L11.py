"""Re-run every experiment of the thesis at layer 11, in order, with one log per step.

Waits for the mPower layer-11 cache (scripts/extract_mpower_full.py --layer 11), then runs
the unified experiments and the side analyses that the thesis quotes.

Run detached:  .venv\\Scripts\\python.exe -u scripts\\run_all_L11.py
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "outputs" / "logs" / "rerun_L11"
LOG.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data" / "embeddings" / "mpower_full__xls-r-300m__L11.npz"

STEPS = [
    ("unified", ["scripts/unified_experiments.py"], {}),
    ("restricted_permutation", ["scripts/restricted_permutation.py", "--fast", "--n-perm", "1000"], {}),
    ("classifier_comparison", ["scripts/classifier_comparison.py"], {}),
    ("fusion", ["scripts/fusion.py"], {}),
    ("objective_function", ["scripts/objective_function.py"], {}),
    ("decision_curves", ["scripts/decision_curves.py"], {}),
    ("embedding_probes", ["scripts/embedding_probes.py"], {}),
    ("age_bin_width", ["scripts/age_bin_width.py"], {"PDV_LAYER": "11"}),
    ("demographic_erasure", ["scripts/demographic_erasure.py"], {"PDV_LAYER": "11"}),
    ("adapt_centroid", ["scripts/label_sensitivity.py", "--within-layer", "11", "--transfer-layer",
                        "11", "--scenarios", "baseline,agematched", "--adapt", "hc-centroid",
                        "--tag", "L11_adapt_centroid"], {}),
    ("adapt_standardize", ["scripts/label_sensitivity.py", "--within-layer", "11", "--transfer-layer",
                           "11", "--scenarios", "baseline,agematched", "--adapt", "hc-standardize",
                           "--tag", "L11_adapt_standardize"], {}),
    ("b2ai_review_checks", ["scripts/bridge2ai_review_checks.py"], {"PDV_LAYER": "11"}),
]

while not CACHE.exists():
    time.sleep(30)
time.sleep(30)                      # let the writer finish closing the file
with open(LOG / "chain.log", "a") as chain:
    for name, args, env in STEPS:
        t0 = time.time()
        with open(LOG / f"{name}.log", "w") as fh:
            rc = subprocess.call([PY, "-u", *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                 env={**os.environ, **env})
        chain.write(f"{name} rc={rc} {(time.time() - t0) / 60:.1f} min\n")
        chain.flush()
    chain.write("CHAIN DONE\n")
