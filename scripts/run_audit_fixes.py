"""Re-run every analysis affected by the 2026-11-09 code audit, in dependency order.

Changes being propagated:
  * tie-aware ECE binning (eval.metrics.calibration_error)          -> everything with ECE
  * IPVS demographic key keeps the group; conflicting rows rejected -> every IPVS age use
  * recording-path joins (restricted_permutation, sex_control, vowel_phoneme_control)
  * LEACE eraser fitted after the split (demographic_erasure, transfer branch)
  * classifier_comparison head imputation from the head's own training rows
    -> decision curves (both prevalences) and difference intervals, which read its output

Order: the predictions first, then everything derived from them, so nothing derived is
left older than its inputs (pdvoice.provenance.stale checks this).
One log per step in outputs/logs/rerun_audit/.

Run:  .venv\\Scripts\\python.exe scripts\\run_audit_fixes.py [--skip classifier_comparison]
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "outputs" / "logs" / "rerun_audit"
LOG.mkdir(parents=True, exist_ok=True)

STEPS = [
    ("classifier_comparison", ["scripts/classifier_comparison.py"], {}),
    ("decision_curves", ["scripts/decision_curves.py"], {}),
    ("decision_curves_0054", ["scripts/decision_curves.py", "--prevalence", "0.054"], {}),
    ("unified", ["scripts/unified_experiments.py"], {}),
    ("unified_clean", ["scripts/unified_experiments.py"], {"PDVOICE_B2AI_CLEAN": "1"}),
    ("unified_clean_all", ["scripts/unified_experiments.py"], {"PDVOICE_CLEAN_ALL": "1"}),
    ("task_comparison", ["scripts/task_comparison.py"], {}),
    ("exp1_changes", ["scripts/exp1_changes.py"], {"PYTHONPATH": "scripts"}),
    ("equal_audio", ["scripts/equal_audio.py"], {}),
    ("ipvs_inversion", ["scripts/ipvs_inversion.py"], {}),
    ("pause_information", ["scripts/pause_information.py"], {}),
    ("speech_agematch", ["scripts/speech_agematch.py", "--seeds", "42,43,44,45,46"], {}),
    ("restricted_permutation", ["scripts/restricted_permutation.py", "--fast",
                                "--n-perm", "1000"], {}),
    ("sex_control", ["scripts/sex_control.py"], {}),
    ("demographic_erasure", ["scripts/demographic_erasure.py"], {"PDV_LAYER": "11"}),
    ("vowel_phoneme_control", ["scripts/vowel_phoneme_control.py"], {}),
    # difference_cis reads classifier_predictions and the fusion predictions
    ("difference_cis", ["scripts/difference_cis.py"], {}),
]


def main(argv=None) -> int:
    """Run the chain; stop at the first failed step and return a nonzero status.

    Every step depends on the ones before it (predictions first, then what is derived
    from them), so a failed step must not be followed by steps that would read its
    older outputs. Before 2026-11-10 the return code was logged but not acted on.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma-separated step names already done")
    skip = set(filter(None, ap.parse_args(argv).skip.split(",")))
    unknown = skip - {name for name, _, _ in STEPS}
    if unknown:
        raise SystemExit(f"unknown step(s) in --skip: {sorted(unknown)}")
    with open(LOG / "chain.log", "a") as chain:
        for name, args, env in STEPS:
            if name in skip:
                chain.write(f"{name} skipped\n")
                continue
            t0 = time.time()
            with open(LOG / f"{name}.log", "w") as fh:
                rc = subprocess.call([PY, "-u", "-W", "ignore", *args], cwd=ROOT, stdout=fh,
                                     stderr=subprocess.STDOUT, env={**os.environ, **env})
            chain.write(f"{name} rc={rc} {(time.time() - t0) / 60:.1f} min\n")
            chain.flush()
            if rc != 0:
                chain.write(f"CHAIN FAILED at {name} (rc={rc}); later steps not run\n")
                print(f"step {name} failed (rc={rc}); see {LOG / (name + '.log')}")
                return rc
        chain.write("CHAIN DONE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
