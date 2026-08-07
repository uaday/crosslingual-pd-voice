# pd-research — Voice-Based Parkinson's Disease Detection

Cross-dataset and multilingual evaluation of a **voice-only** Parkinson's disease (PD)
detector. Train on one English dataset (mPower), then apply the **locked model without
retraining** to external datasets across languages and tasks, with **calibration and
Decision Curve Analysis**. Master's thesis by Sudipta Ghosh (supervisor: Hamid Sharifzadeh).

> **Full project context, methodology, datasets, decisions log →** [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)
> Load that file first — this README is only the code map + quickstart.

## Approach in one line
`audio → 16 kHz → VAD → frozen SSL embedding (cached) → shallow head → calibrate (lock on val) → evaluate (discrimination · calibration · DCA)`, run **per-dataset, never pooled**, with **subject-disjoint** splits.

## Quickstart
```bash
conda env create -f env/environment.yml
conda activate pdvoice
pip install -e .
make kernel            # register the Jupyter kernel
pytest -q              # smoke test
```
See [`env/SETUP.md`](env/SETUP.md) for Mac (MPS) vs Windows/L40S (CUDA cu121) notes.

## Repository layout
```
src/pdvoice/            importable package (the real logic lives here)
  config.py             paths, seed, sample rate, folds, DCA thresholds
  data/                 per-dataset loaders → a standard manifest DataFrame + subject-disjoint splits
  preprocess.py         resample 16 kHz · VAD · quality control
  features/             ssl.py (frozen encoder embeddings, cached) · classical.py (jitter/shimmer/HNR/MFCC)
  models/               head.py (256-unit MLP) · baselines.py (SVM/RF/GBT, 2D-CNN)
  calibration.py        Platt / isotonic / temperature — fit on val, lock before test
  eval/                 discrimination (AUROC/AUPRC/DeLong) · metrics · dca · report (Model Card/Datasheet)
  utils/                seed, audio helpers
configs/                default.yaml · encoders.yaml · eval.yaml
notebooks/              00_explore → 10_preprocess → 20_features → 30_layer1 → 40_layer2 → 50_calibration → 60_dca_transfer → 70_layer3
data/                   (git-ignored) working cache: raw/ interim/ embeddings/
outputs/                (git-ignored) figures/ models/ logs/ cards/
pd-datasets/            (git-ignored) raw datasets — see PROJECT_CONTEXT.md §3
tests/                  pytest smoke tests
```

## Datasets (see PROJECT_CONTEXT.md §3 for details + access status)
mPower (train, English) · Bridge2AI (English) · MDVR-KCL (English, cross-task) · IPVS (Italian) · NeuroVoz (Spanish, pending) · EWA-DB (Slovak, pending).
**Datasets are never committed** (data-use agreements + participant identifiers).

## Status
Scaffold + environment in place. Implementation follows the TODO list in `PROJECT_CONTEXT.md` §9.
Module stubs raise `NotImplementedError` with a docstring describing the intended behaviour.
