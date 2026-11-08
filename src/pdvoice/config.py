"""Central paths and constants for the pdvoice project.

Import this everywhere instead of hard-coding paths. All working directories are
git-ignored and created on demand via :func:`ensure_dirs`.
"""

from __future__ import annotations

from pathlib import Path

# Project root = two levels up from this file (src/pdvoice/config.py -> project/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- raw datasets (git-ignored) ---
PD_DATASETS = PROJECT_ROOT / "pd-datasets"
MPOWER_DIR = PD_DATASETS / "mpower"
IPVS_DIR = PD_DATASETS / "Italian Parkinson's Voice and speech"
MDVR_KCL_DIR = PD_DATASETS / "26-29_09_2017_KCL"
BRIDGE2AI_DIR = PD_DATASETS / "bridge2ai"   # when downloaded
NEUROVOZ_DIR = PD_DATASETS / "neurovoz_v3"  # NeuroVoz v3 (Zenodo), audios under data/audios/
EWADB_DIR = PD_DATASETS / "ewadb"           # when access granted

# --- working cache + outputs (git-ignored; created at runtime) ---
DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"          # resampled 16 kHz wavs, per-dataset manifests
EMBEDDINGS = DATA / "embeddings"    # cached frozen SSL embeddings (.npy / .parquet)
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
LOGS = OUTPUTS / "logs"
CARDS = OUTPUTS / "cards"           # Model Cards + Datasheets
CONFIGS = PROJECT_ROOT / "configs"

# --- signal / split constants ---
SAMPLE_RATE = 16_000        # SSL encoders expect 16 kHz mono
SEED = 42
N_FOLDS = 5                 # subject-disjoint GroupKFold on the training set
VAL_FRACTION = 0.15         # subject-disjoint val carved from each train fold
LABEL_PD, LABEL_HC = 1, 0

# --- evaluation constants ---
ECE_BINS = 10
N_BOOTSTRAP = 1_000
DCA_THRESHOLDS = tuple(i / 100 for i in range(5, 31))  # 0.05 ... 0.30 screening band

# --- dataset language codes ---
LANGUAGE = {
    "mpower": "en",
    "bridge2ai": "en",
    "mdvr_kcl": "en",
    "ipvs": "it",
    "neurovoz": "es",
    "ewadb": "sk",
}

WORKING_DIRS = [RAW, INTERIM, EMBEDDINGS, FIGURES, MODELS, LOGS, CARDS]


def ensure_dirs() -> None:
    """Create all git-ignored working directories if they do not exist."""
    for d in WORKING_DIRS:
        d.mkdir(parents=True, exist_ok=True)
