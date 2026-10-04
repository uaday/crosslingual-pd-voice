"""F-26, part 2 -- re-run Rahmatallah et al. 2025's own CNN on their own mPower recordings.

The published mPower AUC (~0.94-0.95, mel spectrograms) could not be reproduced with our
representation on the same 365 recordings (0.686; rahmatallah_replication.py). Their code
is public (github.com/uams-tri/PD-Voice, Apache 2.0) and its history explains why:

    V1 (2024-10, the code behind the paper): neither Keras generator passes ``subset=``,
        so the model is trained on ALL images and evaluated on the SAME images; and the
        AUC is computed from argmax hard labels, not probabilities.
    V2 (2026-09-23): both fixed, "a data leakage issue identified independently by
        Lorenzo Moni and Angela Kinnen". Their 2023 paper, built on the same pipeline,
        received an Author Correction (doi:10.1038/s41598-026-61935-3): AUC 0.97 -> 0.83.
        The 2025 mPower paper has no correction on Crossref as of 2026-10-05.

This script re-implements their pipeline faithfully and runs it under four protocols:

    v1_as_published   train on all, test on all, AUC from hard labels (their V1 code)
    v1_prob_auc       same leak, AUC from probabilities (isolates the leak from the AUC bug)
    v2_as_published   their corrected code: Keras' fixed validation_split, softmax, prob AUC
    random_70_30      what the paper describes: 100 random stratified 70/30 splits
    + random_70_30 on the age+sex-matched cohort

FAITHFUL TO THEIR CODE (Create_MelSpectrograms_mPowerdataset.R, Inception notebook):
  * tuneR::normalize (centre, scale to max |x| = 1); decimate by taking every 5th sample
    with NO anti-alias filter (fs 44.1 kHz -> 8.82 kHz); crop x[16500:(len-10000)] (R, 1-based)
  * trim_ends(w=512, thr=0.5): centre of the longest run of 512-sample blocks whose energy
    exceeds 0.5; skip if shorter than 1.5 s; keep 1.5 s centred there
  * mel: n_fft 1024, win 512, hop 51 (R gets 512*0.1 = 51.2), f 0-4000 Hz, 256 mels, Hann,
    power 2, then log2; drawn with image() and 256 viridis colours over the finite range
    into a 2x2 in, 300 dpi (600 x 600 px) JPEG at quality 100
  * InceptionV3, ImageNet weights, include_top=False, pooling='avg', input 600x600x3,
    pixels rescaled by 1/255 only (no ImageNet preprocessing, as in their code), frozen
  * head: BatchNorm -> Dense 1024 relu -> Dense 1024 relu -> Dropout 0.2 -> Dense 2
    softmax as the paper states (the V1 notebook's sigmoid is discussed in main),
    categorical cross-entropy, Adam 1e-3, batch 4, 10 epochs

DEPARTURES (none changes what is learned):
  * Keras 3 on the PyTorch backend instead of TensorFlow 2 -- same InceptionV3 graph and
    the same ImageNet weights file. The base is frozen and there is no augmentation, so its
    pooled features are computed once and only the head is retrained per run, which is
    mathematically equivalent to their loop and ~100x faster.
  * spectrograms rendered with matplotlib instead of R's image(); same matrix, colour map,
    colour scaling and size.
  * steps_per_epoch = ceil(n/4) rather than floor(n/4) -- at most one extra small batch.

Requires `keras>=3.6` with KERAS_BACKEND=torch (set below).
Run:  .venv\\Scripts\\python.exe -u scripts\\rahmatallah_cnn.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc as sk_auc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdvoice.cohort import attach_age, attach_sex, demographic_match_ids
from pdvoice.config import LOGS, MPOWER_DIR

ROOT = Path(__file__).resolve().parents[1]
TABLE_S1 = ROOT / "references" / "rahmatallah2025_tableS1.csv"
WORK = ROOT / "outputs" / "rahmatallah_cnn"
IMG_DIR = WORK / "mel"
FEATS = WORK / "inceptionv3_features.npz"
N_RUNS = 100
DS, TIME_LIMIT = 5, 1.5


# ---------------------------------------------------------------- spectrograms (their R)
def trim_ends(x: np.ndarray, w: int = 512, thr: float = 0.5) -> tuple[int, int]:
    """Port of their R ``trim_ends``; returns 1-based (st, en) exactly as R does."""
    n = len(x)
    nseg = n // w
    ste = (x[: nseg * w].reshape(nseg, w) ** 2).sum(axis=1)
    check = np.repeat((ste > thr).astype(int), w)
    # rle
    change = np.flatnonzero(np.diff(check)) + 1
    starts = np.r_[0, change]
    lengths = np.diff(np.r_[starts, len(check)])
    values = check[starts]
    if len(lengths) > 1:
        indmax = int(np.argmax(lengths * values)) + 1          # which.max, 1-based
        if indmax > 1:
            st = int(lengths[: indmax - 1].sum()) + 1
            en = int(lengths[:indmax].sum())
        else:
            st, en = 1, int(lengths[0])
    else:
        st, en = 1, n
    return st, en


def mel_image(wav: Path, out: Path) -> str:
    """Their mPower mel-spectrogram pipeline. Returns 'ok' or the reason it was skipped."""
    x, fs = sf.read(str(wav), dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    x = x - x.mean()                                   # tuneR::normalize: centre ...
    x = x / np.max(np.abs(x))                          # ... and rescale to level 1
    fs = fs / DS
    x = x[::DS]                                        # x[seq(1, length(x), by=ds)]
    x = x[16499: len(x) - 10000]                       # x[16500:(length(x)-10000)]
    if len(x) < 512:
        return "too short after fixed crop"
    st, en = trim_ends(x)
    mp = int(np.floor((st + en) / 2))
    half = int(fs * TIME_LIMIT / 2)                    # 6615 samples
    if (en - st) < fs * TIME_LIMIT:
        return "voiced run < 1.5 s"
    seg = x[mp - half - 1: mp + half]                  # x[(mp-half):(mp+half)], 1-based
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(fs), n_fft=1024, win_length=512, hop_length=51, f_min=0.0,
        f_max=4000.0, n_mels=256, window_fn=torch.hann_window, power=2.0,
        normalized=False)(torch.tensor(seg, dtype=torch.float32))
    with np.errstate(divide="ignore"):
        z = np.log2(mel.numpy()).T                     # (time, mel), as $t()
    z = np.where(np.isfinite(z), z, np.nan)            # image() leaves -Inf cells blank
    cmap = matplotlib.colormaps["viridis"].resampled(256).copy()
    cmap.set_bad("white")
    fig = plt.figure(figsize=(2, 2), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(z.T, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest",
              vmin=np.nanmin(z), vmax=np.nanmax(z))
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, pil_kwargs={"quality": 100})
    plt.close(fig)
    return "ok"


# ---------------------------------------------------------------- cohort
def cohort() -> pd.DataFrame:
    s1 = pd.read_csv(TABLE_S1, skiprows=1)
    demo = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv",
                       usecols=["healthCode", "professional-diagnosis"])
    lab = dict(zip(demo.healthCode.astype(str), demo["professional-diagnosis"]))
    df = pd.DataFrame({"recordId": s1.recordID.astype(str),
                       "subject_id": "mpower_" + s1.healthCode.astype(str)})
    df["label"] = s1.healthCode.astype(str).map(lab).map({True: 1, False: 0})
    df["wav"] = [MPOWER_DIR / "wav" / f"{r}__audio.wav" for r in df.recordId]
    df = df[df.wav.map(Path.exists) & df.label.notna()].copy()
    df["label"] = df.label.astype(int)
    return attach_sex(attach_age(df)).reset_index(drop=True)


# ---------------------------------------------------------------- InceptionV3 features
def inception_features(paths: list[Path]) -> np.ndarray:
    import keras
    base = keras.applications.InceptionV3(weights="imagenet", include_top=False,
                                          input_shape=(600, 600, 3), pooling="avg")
    base.trainable = False
    out = []
    for i in range(0, len(paths), 16):
        batch = np.stack([np.asarray(Image.open(p).convert("RGB").resize((600, 600)),
                                     dtype=np.float32) / 255.0 for p in paths[i:i + 16]])
        out.append(np.asarray(base.predict(batch, verbose=0)))
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- their head
def train_head(Xtr, ytr, seed: int, final_activation: str):
    import keras
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(Xtr.shape[1],))
    h = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(inp)
    h = keras.layers.Dense(1024, activation="relu")(h)
    h = keras.layers.Dense(1024, activation="relu")(h)
    h = keras.layers.Dropout(0.2)(h)
    out = keras.layers.Dense(2, activation=final_activation)(h)
    m = keras.Model(inp, out)
    m.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
    m.fit(Xtr, keras.utils.to_categorical(ytr, 2), batch_size=4, epochs=10,
          shuffle=True, verbose=0)
    return m


def hard_label_auc(y, prob):
    """Their V1 AUC: roc_curve on argmax labels."""
    fpr, tpr, _ = roc_curve(y, np.argmax(prob, axis=1))
    return sk_auc(fpr, tpr)


def keras_fixed_split(names: np.ndarray, y: np.ndarray, frac: float = 0.3):
    """Keras flow_from_directory(validation_split=0.3, subset=...): per class, files in
    sorted order; validation = the first int(0.3 n), training = the rest. Seed-independent."""
    te = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        idx = idx[np.argsort(names[idx])]
        te += idx[: int(frac * len(idx))].tolist()
    te = np.array(sorted(te))
    tr = np.setdiff1d(np.arange(len(y)), te)
    return tr, te


def main() -> None:
    warnings.simplefilter("ignore")
    df = cohort()
    print(f"Table S1 recordings with a local wav and a label: {len(df)}")

    # 1. spectrograms
    status = {}
    for r in df.itertuples():
        out = IMG_DIR / ("PD" if r.label else "HC") / f"{r.recordId}.jpg"
        status[r.recordId] = "ok" if out.exists() else mel_image(r.wav, out)
    df["img_status"] = df.recordId.map(status)
    img_counts = df.img_status.value_counts().to_dict()
    print("spectrogram status:", img_counts, "(their paper: every one of their 398 passed)")
    df = df[df.img_status == "ok"].reset_index(drop=True)
    df["img"] = [IMG_DIR / ("PD" if l else "HC") / f"{r}.jpg"
                 for r, l in zip(df.recordId, df.label)]

    # 2. frozen InceptionV3 features, computed once
    if FEATS.exists() and list(np.load(FEATS, allow_pickle=True)["recordId"]) == list(df.recordId):
        X = np.load(FEATS, allow_pickle=True)["X"]
    else:
        X = inception_features(list(df.img))
        np.savez(FEATS, X=X, recordId=df.recordId.to_numpy())
    y = df.label.to_numpy()
    print(f"features {X.shape};  PD {y.sum()} / HC {(y == 0).sum()}\n")

    results: dict[str, dict] = {"_cohort": dict(n=int(len(df)), n_pd=int(y.sum()),
                                                n_hc=int((y == 0).sum()),
                                                spectrogram_status=img_counts)}

    def report(name, aucs, note):
        a = np.asarray(aucs)
        results[name] = dict(mean=round(float(a.mean()), 4), sd=round(float(a.std()), 4),
                             n_runs=len(a), note=note)
        print(f"{name:<22} AUC {a.mean():.3f} +/- {a.std():.3f}   ({note})", flush=True)
        # saved after every protocol, so a run cut short keeps what it finished
        (LOGS / "rahmatallah_cnn.json").write_text(json.dumps(results, indent=2))

    # 3a. V1 as published: train on all, evaluate on the same images
    # The uploaded V1 notebook ends in Dense(2, activation='sigmoid')  #softmax  with
    # categorical cross-entropy. Under Keras 3 both sigmoid outputs saturate at 1.0 for every
    # input (checked: AUC 0.50, accuracy 0.53 on the training data itself), so that line
    # cannot have produced the published figure. The paper's Methods specify softmax, which
    # is used here; the two V1 defects that matter -- train = test and hard-label AUC -- are kept.
    hard, prob = [], []
    for run in range(N_RUNS):
        m = train_head(X, y, seed=run, final_activation="softmax")
        p = np.asarray(m.predict(X, verbose=0))
        hard.append(hard_label_auc(y, p))
        prob.append(roc_auc_score(y, p[:, 1]))
    report("v1_as_published", hard, "train = test, AUC from hard labels")
    report("v1_prob_auc", prob, "train = test, AUC from probabilities")

    # 3b. V2 as published: Keras' fixed split
    tr, te = keras_fixed_split(df.recordId.to_numpy(), y)
    v2 = []
    for run in range(N_RUNS):
        m = train_head(X[tr], y[tr], seed=run, final_activation="softmax")
        v2.append(roc_auc_score(y[te], np.asarray(m.predict(X[te], verbose=0))[:, 1]))
    report("v2_as_published", v2, f"their corrected code; fixed split {len(tr)}/{len(te)}")

    # 3c. what the paper describes: 100 random 70/30 splits
    sss = StratifiedShuffleSplit(N_RUNS, test_size=0.30, random_state=42)
    cnn, demo_lr = [], []
    D = np.c_[df.age.to_numpy(float), (df.sex == "Female").to_numpy(float)]
    has_demo = ~np.isnan(D).any(axis=1)
    for run, (tr, te) in enumerate(sss.split(X, y)):
        m = train_head(X[tr], y[tr], seed=run, final_activation="softmax")
        cnn.append(roc_auc_score(y[te], np.asarray(m.predict(X[te], verbose=0))[:, 1]))
        trd, ted = tr[has_demo[tr]], te[has_demo[te]]
        lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(D[trd], y[trd])
        demo_lr.append(roc_auc_score(y[ted], lr.predict_proba(D[ted])[:, 1]))
    report("random_70_30", cnn, "100 random stratified splits, participant-disjoint")
    report("age_sex_only_same_splits", demo_lr, "logistic regression on age + sex")

    # 3d. random splits on the age+sex-matched cohort (10 matching seeds x 10 splits)
    dm = df[has_demo].copy()
    matched, sizes = [], []
    for s in range(10):
        keep = demographic_match_ids(dm, on=("age", "sex"), seed=s)
        idx = np.flatnonzero(df.subject_id.isin(keep).to_numpy())
        sizes.append(len(idx))
        for k, (tr, te) in enumerate(StratifiedShuffleSplit(10, test_size=0.30, random_state=s)
                                     .split(idx, y[idx])):
            m = train_head(X[idx[tr]], y[idx[tr]], seed=100 * s + k, final_activation="softmax")
            matched.append(roc_auc_score(y[idx[te]],
                                         np.asarray(m.predict(X[idx[te]], verbose=0))[:, 1]))
    report("random_70_30_agesex_matched", matched,
           f"age+sex-matched, mean n = {np.mean(sizes):.0f} participants")

    print(f"\nwrote {LOGS / 'rahmatallah_cnn.json'}")


if __name__ == "__main__":
    main()
