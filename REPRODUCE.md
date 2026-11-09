# Reproducing the thesis results

This file lists, for every result in the thesis, the script that produces it, how to run it,
and where its output is written. Every table and data figure in the thesis is generated from
those outputs by `scripts/unified_tables.py`, `scripts/thesis_tables.py` and
`scripts/thesis_figures.py` (written to `outputs/tables/` and `outputs/figures/`); no number
in a table is typed by hand.

## 1. Environment

- Python 3.12.10 on Windows 10, with a university GPU server (NVIDIA L40S, 24 GB virtual
  GPU, 2 CPU cores) for embedding extraction. Everything after extraction runs on a CPU.
- Training the classifier head on a GPU is not bit-for-bit reproducible; repeated runs
  move AUROC by about 0.01.

### Model checkpoints (Hugging Face Hub, exact revisions used)

| Encoder | Model ID | Revision |
|---|---|---|
| XLS-R-300M | `facebook/wav2vec2-xls-r-300m` | `1a640f32ac3e39899438a2931f9924c02f080a54` |
| wav2vec 2.0 base | `facebook/wav2vec2-base` | `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8` |
| HuBERT-large | `facebook/hubert-large-ll60k` | `ff022d095678a2995f3c49bab18a96a9e553f782` |
| WavLM-large | `microsoft/wavlm-large` | `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` |
| Whisper-small | `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` |

Voice-activity detection: `silero-vad` 6.2.1. Acoustic features: `praat-parselmouth` 0.4.7.
- Create a virtual environment in the project root and install the package and its
  dependencies:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r env\requirements-gpu.txt
.venv\Scripts\python.exe -m pip install -e .
```

  (`make env` runs the same commands; on Linux/macOS use `.venv/bin/python`.)
- `env/requirements-gpu.txt` pins the exact versions used for the thesis (Python 3.12,
  torch 2.5.1+cu121, transformers 4.46.3, scikit-learn 1.9.0, numpy 1.26.4, pandas 3.0.5,
  scipy 1.17.1). Keep scikit-learn at 1.9.0: its participant splits differ between
  versions, so another version gives different numbers.
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
| Bridge2AI-Voice (adult) | `pd-datasets/bridge2ai/` | Controlled access only: application and Data Transfer and Use Agreement through the consortium's Data Access Compliance Office (DACO@b2ai-voice.org). |

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
   (for Bridge2AI: `get_set("bridge2ai", task, 15, all_groups=True)` for `vowel`, `ddk`
   and `speech`; extraction always covers every group). Caches go to `data/embeddings/`
   and store the path of every recording; every loader matches rows to the manifest by
   path and refuses a cache without paths.
2. **Experiments.** `unified_experiments.py` runs every Chapter 4 experiment under one
   protocol and writes `outputs/logs/unified/`; the scripts in the tables below produce the
   remaining results. Results go to `outputs/logs/`. To rerun the analyses changed by the
   2026-11-09 audit in dependency order, use `run_audit_fixes.py`, which stops at the
   first failed step.
3. **Tables and figures:** `unified_tables.py`, `thesis_tables.py`, `thesis_figures.py`.

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
| Design B (speech between clinical datasets) | `unified_experiments.py` (Experiment 2) |
| Speech transfer under age matching (three-arm) | `speech_agematch.py` |
| Whole recording against fixed windows | `extract_windows.py`, then the windowing grid via `transfer_any.py` |
| IPVS inversion | `ipvs_inversion.py` |
| Vowel phoneme control | `vowel_phoneme_control.py` |
| Bridge2AI within-dataset, before and after age matching | `bridge2ai_agematch.py --n-perm 200` |
| Bridge2AI transfer (Designs C and D) | `unified_experiments.py` (Experiments 1 and 2) |
| Parkinson's disease against other voice disorders (Bridge2AI) | `unified_experiments.py` (specificity part) |
| Embedding probes (what the vector encodes) | `embedding_probes.py` |
| Calibration and objective function | `objective_function.py` |
| Decision curves | `decision_curves.py` |

### Historical scripts (not behind any thesis number)

`transfer_any.py` (the earlier any-to-any transfer, single seed, layer 15),
`scaleup_agematched.py` (the scaled age-matched model) and the earlier Bridge2AI scripts
`bridge2ai_transfer.py`, `bridge2ai_transfer_agecheck.py` and `bridge2ai_followups.py`
(single layer, set by `PDV_LAYER`) were superseded by `unified_experiments.py`. They still run, with the same cohort and cache-identity rules,
but their outputs are not read by the thesis tables or the number check.

## 6. Notes

- Bridge2AI groups: Parkinson's disease is the gold-standard diagnosis field; controls are
  `eligibility.is_control_participant`, excluding the 11 controls the assessor judged to
  have a voice condition (`control.tsv`, `diagnosis_c_ac = Yes`). See
  `src/pdvoice/data/bridge2ai.py`.
- The first-session check and the recording-setup checks quoted in Section 4.4 (sessions,
  microphone, input gain, protocol version) use `load_bridge2ai(recording_meta=True)`.
