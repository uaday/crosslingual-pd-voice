"""Equal-audio control for the task comparison.

In the main experiments the model hears 3 to 16 times more connected speech than vowel per
participant, so a speech advantage could reflect the amount of audio rather than the task.
Here every participant contributes exactly K segments of SEG_S seconds per task:

    vowel   the 2 s stable core of K different recordings (the core used everywhere else,
            which avoids the counting that opens Bridge2AI vowel recordings)
    ddk,    K non-overlapping 2 s segments of the voice-activity-trimmed recordings, spread
    speech  across recordings (first segment of each recording, then the second, ...)

Participants with fewer than K segments for a task are left out of that task. Each segment
is embedded exactly as elsewhere (XLS-R-300M layer 11, mean and standard-deviation pooling)
and treated as one recording, so a participant's score is the mean over K segments. Then the
within-dataset and transfer analyses of unified_experiments.py are repeated (same head,
locked calibration, five seeds, age matching, US-only Bridge2AI, common speech tasks), and
tasks are compared on the same participants as in task_comparison.py.

Outputs (outputs/logs/unified/): equal_audio_runs.csv, equal_audio_scores.csv.gz,
equal_audio_comparison.json; segment embeddings cached in data/embeddings/.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_experiments as U  # noqa: E402
from task_comparison import PAIRS, paired  # noqa: E402
from pdvoice.config import EMBEDDINGS, SAMPLE_RATE  # noqa: E402
from pdvoice.features.ssl import DEFAULT_MODELS, _embed_wave, load_encoder  # noqa: E402
from pdvoice.preprocess import preprocess_clip, voice_activity_trim  # noqa: E402
from pdvoice.utils.audio import load_audio, normalise_amplitude  # noqa: E402

SEG_S, K = 2.0, 3
TASKS = {"bridge2ai": ["vowel", "ddk", "speech"], "neurovoz": ["vowel", "ddk", "speech"],
         "ipvs": ["vowel", "speech"], "mdvr_kcl": ["speech"]}


def recording_segments(path: str, kind: str) -> list[np.ndarray]:
    if kind == "vowel":
        core = preprocess_clip(path, "vowel")
        return [core] if len(core) >= int(SEG_S * SAMPLE_RATE) * 0.95 else []
    clean = normalise_amplitude(voice_activity_trim(load_audio(path, sr=SAMPLE_RATE)))
    n = int(SEG_S * SAMPLE_RATE)
    return [clean[i:i + n] for i in range(0, len(clean) - n + 1, n)]


def build(ds: str, task: str, enc):
    """Segment-level table and embeddings for one dataset and task (cached)."""
    cache = EMBEDDINGS / f"equalaudio__{ds}__{task}__L{U.LAYER}__K{K}.npz"
    df, _ = U.load(ds, task)
    subj = df.drop_duplicates("subject_id").set_index("subject_id")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        seg = pd.DataFrame({"subject_id": d["subject_id"].astype(str), "path": d["seg"].astype(str)})
        X = d["X"]
    else:
        rng = np.random.default_rng(42)
        rows, segs = [], []
        for sid, g in df.groupby("subject_id", sort=True):
            paths = list(g.path)
            rng.shuffle(paths)
            per_rec = [recording_segments(p, task) for p in paths]
            picked = []
            for j in range(max((len(s) for s in per_rec), default=0)):   # round-robin
                for p, s in zip(paths, per_rec):
                    if j < len(s) and len(picked) < K:
                        picked.append((f"{p}#{j}", s[j]))
            if len(picked) < K:
                continue
            for name, w in picked:
                rows.append((sid, name)); segs.append(w)
        model, extractor, device = enc
        X = np.stack([_embed_wave(w, model, extractor, device, U.LAYER, "mean_std")
                      for w in segs]).astype(np.float32)
        seg = pd.DataFrame(rows, columns=["subject_id", "path"])
        np.savez(cache, X=X, subject_id=seg.subject_id.to_numpy(), seg=seg.path.to_numpy())
    for c in ("label", "age", "sex", "dataset"):
        seg[c] = seg.subject_id.map(subj[c]) if c in subj else np.nan
    seg["label"] = seg.label.astype(int)
    print(f"{ds:9s} {task:6s} {seg.subject_id.nunique():4d} participants x {K} segments "
          f"(of {df.subject_id.nunique()})", flush=True)
    return seg.reset_index(drop=True), np.asarray(X, np.float32)


def main() -> None:
    warnings.simplefilter("ignore")
    t0 = time.time()
    enc = load_encoder(DEFAULT_MODELS[U.MODEL])
    data = {(ds, t): build(ds, t, enc) for ds, ts in TASKS.items() for t in ts}
    rows, scores = [], []

    def record(exp, src, stask, cond, seed, tgt, ttask, coh, g):
        if g.y.nunique() < 2:
            return
        rows.append(dict(exp=exp, source=src, source_task=stask, condition=cond, seed=seed,
                         target=tgt, target_task=ttask, target_cohort=coh,
                         n_pd=int(g.y.sum()), n_hc=int((g.y == 0).sum()),
                         auroc=float(roc_auc_score(g.y, g.r))))
        scores.append(g.reset_index().assign(exp=exp, source=src, source_task=stask,
                                             condition=cond, seed=seed, target=tgt,
                                             target_task=ttask, target_cohort=coh))

    # within each dataset
    for (ds, t), (df, X) in data.items():
        for seed in U.SEEDS:
            for cond, m in U.train_conditions(ds, df, seed).items():
                if cond not in ("uncorrected", "agematched"):
                    continue
                sub = df[m].reset_index(drop=True)
                if min(sub[sub.label == 1].subject_id.nunique(),
                       sub[sub.label == 0].subject_id.nunique()) < 10:
                    continue
                record("within", ds, t, cond, seed, ds, t, cond, U.within_cv(sub, X[m], seed))
        print(f"[{(time.time()-t0)/60:5.1f} min] within {ds} {t}", flush=True)

    # transfer, the same task in every other dataset
    for task in ("vowel", "ddk", "speech"):
        keys = [k for k in data if k[1] == task]
        for (src, _) in keys:
            sdf, SX = data[(src, task)]
            for seed in U.SEEDS:
                for cond, m in U.train_conditions(src, sdf, seed).items():
                    if cond not in ("uncorrected", "agematched"):
                        continue
                    sub = sdf[m]
                    head, cal = U.fit_locked(SX[m], sub.label.to_numpy(), sub.subject_id.to_numpy(), seed)
                    for (tgt, _) in keys:
                        if tgt == src:
                            continue
                        tdf, TX = data[(tgt, task)]
                        r, p = U.predict(head, cal, TX)
                        g_all = U.subject_table(tdf, r, p)
                        for coh, cm in U.test_cohorts(tdf, seed).items():
                            if coh == "agesexmatched":
                                continue
                            record("transfer", src, task, cond, seed, tgt, task, coh,
                                   g_all[g_all.index.isin(set(tdf.subject_id[cm]))])
            print(f"[{(time.time()-t0)/60:5.1f} min] transfer {task} from {src}", flush=True)

    R, S = pd.DataFrame(rows), pd.concat(scores)
    R.to_csv(U.OUT / "equal_audio_runs.csv", index=False)
    S.to_csv(U.OUT / "equal_audio_scores.csv.gz", index=False)

    out = {"segments": {"seconds": SEG_S, "per_participant": K}, "within": {}, "transfer": {}}
    W = S[S.exp == "within"]
    for ds in ("bridge2ai", "neurovoz", "ipvs"):
        for cond in ("uncorrected", "agematched"):
            for ta, tb in PAIRS:
                a = W[(W.source == ds) & (W.source_task == ta) & (W.condition == cond)]
                b = W[(W.source == ds) & (W.source_task == tb) & (W.condition == cond)]
                if not a.empty and not b.empty and (r := paired(a, b)):
                    out["within"][f"{ds} | {cond} | {ta} - {tb}"] = r
    T = S[S.exp == "transfer"]
    for cond in ("uncorrected", "agematched"):
        for ta, tb in PAIRS:
            A, B = T[(T.source_task == ta) & (T.condition == cond)], T[(T.source_task == tb) & (T.condition == cond)]
            for (src, tgt), _ in A.groupby(["source", "target"]):
                for coh in ("released", "agematched"):
                    a = A[(A.source == src) & (A.target == tgt) & (A.target_cohort == coh)]
                    b = B[(B.source == src) & (B.target == tgt) & (B.target_cohort == coh)]
                    if not a.empty and not b.empty and (r := paired(a, b)):
                        out["transfer"][f"{src}->{tgt} | train {cond} | test {coh} | {ta} - {tb}"] = r
    (U.OUT / "equal_audio_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
