# Reproducing the thesis results

This file lists, for every result in the thesis, the script that produces it, how to run it,
and where its output is written. Every table and data figure in `thesis/` is generated from
those outputs by `scripts/thesis_tables.py` and `scripts/thesis_figures.py`; no number in a
table is typed by hand.

## 1. Environment

- Python 3.12, a CUDA GPU for embedding extraction (an NVIDIA L40S was used). Everything
  after extraction runs on a CPU.
- Create a virtual environment in the project root and install the package and its
  dependencies:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install transformers==4.46.3 librosa soundfile audioread praat-parselmouth silero-vad statsmodels matplotlib tqdm "keras>=3.6" "ml-dtypes<0.5" synapseclient pytest
```

- Versions used for the thesis: torch 2.5.1+cu121, transformers 4.46.3, scikit-learn 1.9.0,
  numpy 1.26.4 (must stay below 2.0), pandas 3.0.5, scipy 1.17.1, synapseclient 4.14.0.
- Check the installation: `.venv\Scripts\python.exe -m pytest -q tests`.

All commands below are run from the project root with
`.venv\Scripts\python.exe -u scripts\<script>.py`.

## 2. Data

The data are not in this repository. Each dataset goes in its own folder under
`pd-datasets/` (paths are set in `src/pdvoice/config.py`).

| Dataset | Folder | How to obtain |
|---|---|---|
| mPower | `pd-datasets/mpower/` | Synapse, study syn4993293; registered account and acceptance of the study's conditions of use |
| NeuroVoz | `pd-datasets/neurovoz_v3/` | Public release (Mendes-Laureano et al., Scientific Data 2024) |
| IPVS | `pd-datasets/Italian Parkinson's Voice and speech/` | Public release (Dimauro et al., IEEE Access 2017) |
| MDVR-KCL | `pd-datasets/26-29_09_2017_KCL/` | Zenodo, doi:10.5281/zenodo.2867215 |
| Bridge2AI-Voice (adult) | `pd-datasets/bridge2ai/` | Controlled access only: application and Data Transfer and Use Agreement through the consortium's Data Access Compliance Office (DACO@b2ai-voice.org). See thesis Appendices C and D. |

Bridge2AI is downloaded with `download_bridge2ai.py`, which needs a Synapse personal access
token (View and Download scopes) written to `~/.synapseConfig` as

```
[authentication]
authtoken = <token>
```

Write the file with a text editor: the interactive `synapse config` prompt corrupts pasted
tokens on Windows. The script checks every file against its published MD5 and can be re-run
to resume. Bridge2AI data must stay on approved storage and must not be shared beyond the
approved personnel.

## 3. Pipeline settings

Fixed for every experiment (the thesis section "Where Each Setting Comes From" lists each with its source):
16 kHz mono; vowels trimmed at 30 dB below peak and cut to the central 2 s; connected speech
passed through the Silero voice-activity detector and cut into 5 s windows with 50 per cent
overlap; RMS level -23 dBFS; recordings under 1 s or without readable voice removed; at most
3 recordings per mPower participant, chosen after quality control; XLS-R-300M embeddings,
mean and standard deviation pooled (2048 dimensions); participant-disjoint 5-fold
cross-validation with a 15 per cent validation split; age matching in 5-year bins; five
seeds (42 to 46); 1,000 bootstrap resamples.

## 4. Order of execution

1. **Embeddings.** `extract_mpower_full.py` (mPower, all layers needed), `extract_all_layers.py`
   and `extract_windows.py` (layer sweeps and window variants), `slice_windows.py`.
   External datasets and Bridge2AI are extracted on first use by `transfer_any.get_set`
   (for Bridge2AI: `get_set("bridge2ai", task, 15)` for `vowel`, `ddk` and `speech`).
   Caches go to `data/embeddings/` and store speaker ids, which are checked on load.
2. **Experiments** in the table below. Results go to `outputs/logs/`.
3. **Tables and figures:** `thesis_tables.py`, `thesis_figures.py`, `thesis_appendix.py`.

## 5. Where each result comes from

### Chapter 3

| Result | Script |
|---|---|
| Dataset, demographic and prevalence tables | `thesis_tables.py` (counts from the loaders in `src/pdvoice/data/`) |
| NeuroVoz task-code leak | `neurovoz_taskcode_leak.py` |
| IPVS recording batch and noise floor | `verify/verify_ipvs_batch.py`, `ipvs_ceiling.py` |
| Information in the removed pauses | `pause_information.py` |
| Age-bin width sensitivity | `age_bin_width.py` |
| Classical baseline | `classical_baseline.py` |
| Re-examination of Rahmatallah et al. 2025 | `rahmatallah_replication.py`, `rahmatallah_cnn.py` |
| Comparison with published figures | `literature_comparison.py` |

### Chapter 4

| Result | Script |
|---|---|
| Encoder and layer selection | `encoder_layer_sweep.py`, `select_transfer_layer.py` |
| Calibration, locking and Design A transfer | `layer2_transfer.py` |
| Age baselines, classifier families | `classifier_comparison.py` |
| Label noise against age (five seeds), age and sex matching, domain adaptation on the matched cohort | `label_sensitivity.py --scenarios baseline,agematched,agesexmatched` and `--adapt hc-centroid` / `--adapt hc-standardize` |
| Restricted permutation test (mPower) | `restricted_permutation.py --n-perm 100` |
| Age and sex erasure (LEACE) | `demographic_erasure.py` |
| Sex control | `sex_control.py` |
| Centroid correction | `centroid_confound_test.py` |
| Classical branch under transfer | `classical_transfer.py` |
| Fusion of classical and SSL features | `fusion.py` |
| Design B (speech between clinical datasets) | `transfer_any.py --train <dataset>:speech --tests ...` |
| Speech transfer under age matching (three-arm) | `speech_agematch.py` |
| Whole recording against fixed windows | `extract_windows.py`, then the windowing grid via `transfer_any.py` |
| IPVS inversion | `ipvs_inversion.py` |
| Vowel phoneme control | `vowel_phoneme_control.py` |
| Bridge2AI within-dataset, before and after age matching | `bridge2ai_agematch.py --n-perm 200` |
| Bridge2AI transfer (Designs C and D) | `bridge2ai_transfer.py`, then `bridge2ai_transfer_agecheck.py` |
| Read against spontaneous speech, age and sex matching, PD against other disorders | `bridge2ai_followups.py` |
| Embedding probes (what the vector encodes) | `embedding_probes.py` |
| Calibration and objective function | `objective_function.py` |
| Decision curves | `decision_curves.py` |
| Scaled age-matched model | `scaleup_agematched.py` |

## 6. Notes

- Bridge2AI groups: Parkinson's disease is the gold-standard diagnosis field; controls are
  `eligibility.is_control_participant`, excluding the 11 controls the assessor judged to
  have a voice condition (`control.tsv`, `diagnosis_c_ac = Yes`). See
  `src/pdvoice/data/bridge2ai.py`.
- The first-session check and the recording-setup checks quoted in Section 4.4 (sessions,
  microphone, input gain, protocol version) use `load_bridge2ai(recording_meta=True)`.
- `RESEARCH_STATE.md` records every finding with the script and numbers behind it.
