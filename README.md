# crosslingual-pd-voice

Code for the Master's thesis *Voice-Based Parkinson's Disease Detection with Cross-Dataset
and Multilingual Evaluation* (Sudipta Ghosh, Unitec Institute of Technology, 2026;
principal supervisor Dr Hamid Sharifzadeh).

The thesis asks whether a Parkinson's disease (PD) detector trained on one voice dataset
still works on another dataset, recorded in another language, and what it is actually
detecting. It trains a small classifier on frozen XLS-R-300M speech embeddings and tests it
across five datasets in English, Spanish and Italian. The main findings are:

- Much of the accuracy reported for voice-based PD detection comes from **age**: patients
  are older than controls, and the embeddings encode age. Patients and controls must be
  matched on age before a score can be read as a disease signal.
- In Bridge2AI-Voice, **recruitment country** is a second confound: every Canadian
  participant in the PD-versus-control cohort is a patient, so PD-versus-control analyses
  use US participants only.
- **Connected speech** carries more of the PD signal than a sustained vowel, and it
  transfers between datasets and languages after age matching.

## What is in this repository

```
src/pdvoice/        the package: dataset loaders, preprocessing, encoders, classifier
                    head, calibration and evaluation
scripts/            one script per experiment (REPRODUCE.md maps results to scripts)
configs/            default settings, encoder list and evaluation settings
notebooks/          50_data_and_results.ipynb reads the result files and shows them
tests/              a smoke test
env/                requirements-gpu.txt
pd-datasets/mpower/ download and conversion scripts for mPower
references/         published results used in the literature comparison
```

No audio, embeddings or participant-level results are included. The datasets are under
data-use agreements; see below for how to obtain each one.

## Setup

Python 3.10 or later. The thesis used Python 3.12 on Windows, with an NVIDIA L40S GPU for
embedding extraction; everything after extraction runs on a CPU.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python -m pip install -r env/requirements-gpu.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest -q tests
```

On Linux or macOS use `.venv/bin/python`. Keep `transformers==4.46.3` and `numpy<2.0`:
newer `transformers` refuses to load the encoders' `.bin` weights with torch 2.5. Installing
`silero-vad` can replace `torchaudio` with a CPU build; re-run the first install line if
that happens.

## Data

Each dataset goes in its own folder under `pd-datasets/` (paths are set in
`src/pdvoice/config.py`).

| Dataset | Language | Folder | How to obtain |
|---|---|---|---|
| mPower | English | `pd-datasets/mpower/` | Synapse study syn4993293 (registered account, accept the conditions of use); `pd-datasets/mpower/download_mpower.py` |
| Bridge2AI-Voice (adult) | English | `pd-datasets/bridge2ai/` | Controlled access through the consortium's Data Access Compliance Office (DACO@b2ai-voice.org); `scripts/download_bridge2ai.py` |
| NeuroVoz | Spanish | `pd-datasets/neurovoz_v3/` | Public release (Mendes-Laureano et al., *Scientific Data*, 2024) |
| IPVS | Italian | `pd-datasets/Italian Parkinson's Voice and speech/` | Public release (Dimauro et al., *IEEE Access*, 2017) |
| MDVR-KCL | English | `pd-datasets/26-29_09_2017_KCL/` | Zenodo, doi:10.5281/zenodo.2867215 |

`scripts/download_bridge2ai.py` reads a Synapse personal access token from
`~/.synapseConfig`. Never commit the token, and keep Bridge2AI data on the storage your
data-use agreement approves.

## Running the experiments

All commands run from the repository root.

1. **Embeddings.** `scripts/extract_mpower_full.py` for mPower,
   `scripts/layer2_transfer.py` for the external datasets and
   `scripts/extract_bridge2ai_layers.py` for Bridge2AI. Caches are written to
   `data/embeddings/`.
2. **Experiments.** `scripts/unified_experiments.py` runs every Chapter 4 experiment under
   one protocol (XLS-R-300M layer 11, five seeds, participant-disjoint cross-validation, age
   matching in 5-year bins) and writes `outputs/logs/unified/`.
3. **Tables and figures.** `scripts/unified_tables.py`, `scripts/thesis_tables.py` and
   `scripts/thesis_figures.py` write LaTeX tables to `outputs/tables/` and figures to
   `outputs/figures/`.

`unified_experiments.py` takes three optional environment variables:

| Variable | Effect | Output folder |
|---|---|---|
| `PDVOICE_B2AI_CLEAN=1` | spectral-gating denoising on Bridge2AI (run `scripts/extract_bridge2ai_clean.py` first) | `outputs/logs/unified_clean/` |
| `PDVOICE_CLEAN_ALL=1` | the same denoising on every clinical dataset | `outputs/logs/unified_clean_all/` |
| `PDVOICE_B2AI_SPEECH=<tasks>` | restricts the Bridge2AI speech model to the named tasks, comma-separated | `outputs/logs/unified_speech_<tasks>/` |

[REPRODUCE.md](REPRODUCE.md) lists, for every result in the thesis, the script that
produces it.

Training the classifier head on a GPU is not bit-for-bit reproducible; repeated runs move
AUROC by about 0.01.

## Licence

MIT; see [LICENSE](LICENSE). The datasets keep their own licences and data-use agreements.
