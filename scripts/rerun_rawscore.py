"""Re-run every experiment whose AUROC was computed on calibrated probabilities.

Discrimination is now computed on the uncalibrated head score
(pdvoice.calibration.RankingMap, attached to each calibrator as cal.ranking); see it for why. This driver repeats each
configuration that produced a result in the thesis, with the same arguments, in dependency
order (the within-mPower ceilings that layer2_transfer.py reads come first). The previous
results are kept in outputs/logs_before_rawscore/ for comparison.

Run:  .venv\\Scripts\\python.exe -u scripts\\rerun_rawscore.py [--from N]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
S5 = "42,43,44,45,46"

RUNS = [
    # within-mPower ceilings (layer1), all configurations used by tables and by layer2
    *[["layer1_mpower.py", "--model", "xls-r-300m", *(["--layer", str(l)] if l else []), *w]
      for l, w in [(9, []), (11, []), (12, []), (15, []), (None, []),
                   (11, ["--window", "full"]), (15, ["--window", "full"])]],
    ["layer1_mpower.py", "--model", "hubert-large", "--layer", "12"],
    ["layer1_mpower.py", "--model", "hubert-large"],
    ["layer1_mpower.py", "--model", "wavlm-large", "--layer", "12"],
    ["layer1_mpower.py", "--model", "wavlm-large"],
    ["layer1_mpower.py", "--model", "wav2vec2-base"],
    # Design A (layer2): vowel
    ["layer2_transfer.py", "--layer", "11", "--externals", "ipvs,neurovoz", "--task", "vowel"],
    ["layer2_transfer.py", "--layer", "15", "--externals", "ipvs,neurovoz", "--task", "vowel"],
    ["layer2_transfer.py", "--externals", "ipvs,neurovoz", "--task", "vowel"],
    ["layer2_transfer.py", "--layer", "11", "--externals", "ipvs,neurovoz", "--task", "vowel", "--window", "full"],
    ["layer2_transfer.py", "--layer", "15", "--externals", "ipvs,neurovoz", "--task", "vowel", "--window", "full"],
    ["layer2_transfer.py", "--externals", "ipvs,neurovoz", "--task", "vowel", "--adapt", "hc-centroid"],
    ["layer2_transfer.py", "--externals", "ipvs,neurovoz", "--task", "vowel", "--adapt", "hc-standardize"],
    # Design A: speech
    ["layer2_transfer.py", "--layer", "11", "--externals", "ipvs,neurovoz,mdvr_kcl", "--task", "speech"],
    ["layer2_transfer.py", "--layer", "11", "--externals", "ipvs,neurovoz,mdvr_kcl", "--task", "speech", "--window", "full"],
    ["layer2_transfer.py", "--layer", "15", "--externals", "ipvs,neurovoz,mdvr_kcl", "--task", "speech", "--window", "full"],
    ["layer2_transfer.py", "--layer", "15", "--externals", "mdvr_kcl", "--task", "speech"],
    ["layer2_transfer.py", "--externals", "mdvr_kcl", "--task", "speech", "--adapt", "hc-centroid"],
    ["layer2_transfer.py", "--externals", "mdvr_kcl", "--task", "speech", "--adapt", "hc-standardize"],
    # Design A: DDK
    ["layer2_transfer.py", "--layer", "11", "--externals", "neurovoz", "--task", "ddk"],
    ["layer2_transfer.py", "--layer", "11", "--externals", "neurovoz", "--task", "ddk", "--window", "full"],
    ["layer2_transfer.py", "--layer", "15", "--externals", "neurovoz", "--task", "ddk", "--window", "full"],
    # Design B and the reverse-vowel runs
    ["transfer_any.py", "--train", "ipvs:speech", "--tests", "neurovoz:speech,mdvr_kcl:speech"],
    ["transfer_any.py", "--train", "mdvr_kcl:speech", "--tests", "neurovoz:speech,ipvs:speech"],
    ["transfer_any.py", "--train", "neurovoz:speech", "--tests", "ipvs:speech,mdvr_kcl:speech"],
    ["transfer_any.py", "--train", "ipvs:vowel", "--tests", "neurovoz:vowel,mpower:vowel"],
    ["transfer_any.py", "--train", "neurovoz:vowel", "--tests", "mpower:vowel,ipvs:vowel"],
    ["speech_agematch.py", "--seeds", S5],
    # five-seed label-noise vs age experiments
    ["label_sensitivity.py"],
    ["label_sensitivity.py", "--scenarios", "baseline,agematched,agesexmatched", "--tag", "agesex"],
    ["label_sensitivity.py", "--scenarios", "baseline,agematched", "--adapt", "hc-centroid", "--tag", "adapt_centroid"],
    ["label_sensitivity.py", "--scenarios", "baseline,agematched", "--adapt", "hc-standardize", "--tag", "adapt_standardize"],
    # review item A3: within-dataset and transfer at the same layer
    ["label_sensitivity.py", "--scenarios", "baseline,agematched,agesexmatched",
     "--within-layer", "15", "--transfer-layer", "15", "--tag", "samelayer15"],
    # other locked-head experiments
    ["classical_transfer.py", "--cap", "3"],
    ["centroid_confound_test.py", "--seeds", "42,43,44"],
    ["scaleup_agematched.py", "--cap", "0", "--seeds", S5],
    ["scaleup_agematched.py", "--cap", "10", "--seeds", S5],
    ["vowel_phoneme_control.py", "--seeds", "42,43,44"],
    ["demographic_erasure.py"],
    ["bridge2ai_transfer.py"],
    ["bridge2ai_transfer_agecheck.py"],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    args = ap.parse_args()
    logdir = ROOT / "outputs" / "logs" / "rerun_rawscore"
    logdir.mkdir(parents=True, exist_ok=True)
    for i, cmd in enumerate(RUNS):
        if i < args.start:
            continue
        name = f"{i:02d}_" + "_".join(c.strip("-").replace(":", "-").replace(",", "+")
                                      for c in cmd).replace(".py", "")[:120]
        t0 = time.time()
        print(f"[{i + 1}/{len(RUNS)}] {' '.join(cmd)}", flush=True)
        with open(logdir / f"{name}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call([PY, "-u", str(ROOT / "scripts" / cmd[0]), *cmd[1:]],
                                 cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        print(f"      {'ok' if rc == 0 else f'FAILED rc={rc}'} in {(time.time() - t0) / 60:.1f} min",
              flush=True)
        if rc != 0:
            print(f"      see {logdir / name}.log", flush=True)
    print("RERUN DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
