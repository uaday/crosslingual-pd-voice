"""Fusion baseline — SSL embeddings + classical features (RQ1/RQ2, "best of both").

Motivation: on the honest age-matched vowel signal, cheap classical features
(jitter/shimmer/HNR) are statistically competitive with 300M-param XLS-R
(classical_baseline.py). Does combining them beat either alone — SSL's reach plus the
interpretable classical signal?

Same mPower vowel cohort (cap-1) as the classical baseline, so the cached classical CSV
lines up row-for-row. SSL (XLS-R L15) is extracted for the same cohort and cached.
All three feature sets are run through the SAME MLP head + subject-disjoint 5-fold CV,
so the only thing that changes is the features. Reported for baseline + age-matched.

Run:  python scripts/fusion.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pdvoice.config import EMBEDDINGS, MPOWER_DIR, SEED
from pdvoice.data.mpower import load_mpower
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

MODEL, LAYER, CAP = "xls-r-300m", 15, 1
FEATS = ["jitter_local", "shimmer_local", "hnr", "f0_mean"] + [f"mfcc_{i}" for i in range(13)]


def build_cohort():
    """Identical to classical_baseline.build_cohort(cap=1) so rows align with the CSV."""
    df = load_mpower()
    lab = df[df["label"].isin([0, 1])].sample(frac=1.0, random_state=SEED)
    df = lab[lab.groupby("subject_id").cumcount().to_numpy() < CAP].reset_index(drop=True)
    df, _ = quality_control(df)
    df = df[df["path"].astype(str) != "nan"].reset_index(drop=True)
    d = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    age = dict(zip(d["healthCode"].astype(str), d["age"]))
    df["age"] = df["subject_id"].str.replace("mpower_", "", regex=False).map(age)
    return df


def embed_ssl(df):
    cache = EMBEDDINGS / f"mpower_pilot__{MODEL}__cap{CAP}__L{LAYER}__seed{SEED}.npz"
    if cache.exists():
        X = np.load(cache)["X"]
        if X.shape[0] == len(df):
            print(f"  SSL cache hit: {cache.name}  {X.shape}")
            return X.astype(np.float32)
    from tqdm import tqdm

    from pdvoice.features.ssl import DEFAULT_MODELS, _embed_wave, load_encoder
    from pdvoice.preprocess import preprocess_clip

    model, extractor, device = load_encoder(DEFAULT_MODELS[MODEL])
    print(f"  extracting SSL for {len(df)} recs on {device} ...")
    vecs = [_embed_wave(preprocess_clip(r.path, r.task_kind), model, extractor, device, LAYER,
                        "mean_std").astype(np.float16)
            for r in tqdm(df.itertuples(index=False), total=len(df), unit="rec")]
    X = np.vstack(vecs).astype(np.float16)
    np.savez_compressed(cache, X=X)
    return X.astype(np.float32)


def age_match_mask(df, seed=SEED):
    s = df.dropna(subset=["age"]).drop_duplicates("subject_id")[["subject_id", "label", "age"]]
    s = s.assign(bin=(s["age"] // 5 * 5))
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in s.groupby("bin"):
        k = min((g.label == 1).sum(), (g.label == 0).sum())
        if k:
            keep += g[g.label == 1]["subject_id"].sample(k, random_state=seed).tolist()
            keep += g[g.label == 0]["subject_id"].sample(k, random_state=seed).tolist()
    return df["subject_id"].isin(keep).to_numpy()


def within_cv(df, X, y):
    df = assign_folds(df.reset_index(drop=True), n_splits=5)
    oof = []
    for k in range(5):
        tr, va, te = train_val_test_indices(df, test_fold=k)
        m, _ = fit_mlp_head(X[tr], y[tr], X[va], y[va])
        with torch.no_grad():
            dev = next(m.parameters()).device
            lg = m(torch.tensor(X[te], dtype=torch.float32, device=dev)).squeeze(-1)
            p = torch.sigmoid(lg).cpu().numpy()
        oof.append(pd.DataFrame({"s": df.loc[te, "subject_id"].to_numpy(), "y": y[te], "p": p}))
    g = pd.concat(oof).groupby("s").agg(y=("y", "first"), p=("p", "mean")).reset_index()
    yy, pp = g["y"].to_numpy(), g["p"].to_numpy()
    a, lo, hi = bootstrap_ci(yy, pp, metric=auroc)
    return a, lo, hi, auprc(yy, pp), g


def main():
    print("=== Fusion: SSL + classical (mPower vowel, cap-1, XLS-R L15) ===")
    df = build_cohort()
    cls = pd.read_csv(EMBEDDINGS / f"mpower_classical__cap{CAP}.csv")
    assert len(cls) == len(df), f"classical {len(cls)} != cohort {len(df)} (rebuild mismatch)"

    # classical: median-impute + z-score (global; negligible for an AUROC comparison,
    # and applied identically to every feature set so the comparison stays fair)
    C = cls[FEATS].to_numpy(dtype=float)
    med = np.nanmedian(C, axis=0)
    C = np.where(np.isnan(C), med, C)
    C = (C - C.mean(0)) / (C.std(0) + 1e-8)

    S = embed_ssl(df)                      # SSL (2048)
    F = np.hstack([S, C]).astype(np.float32)   # fusion (2048 + 17)
    y = df["label"].astype(int).to_numpy()
    print(f"  cohort {len(df)} · SSL {S.shape[1]}d · classical {C.shape[1]}d · fusion {F.shape[1]}d\n")

    sets = {"SSL-only": S, "classical-only": C, "FUSION": F}
    print(f"  {'scenario':<13}{'features':<16}{'AUROC':<22}{'AUPRC'}")
    results = {"cohort": len(df), "dims": {"ssl": S.shape[1], "classical": C.shape[1]}, "scenarios": {}}
    for scen, mask in {"baseline": np.ones(len(df), bool), "age-matched": age_match_mask(df)}.items():
        results["scenarios"][scen] = {}
        for name, X in sets.items():
            a, lo, hi, ap, g = within_cv(df[mask].reset_index(drop=True), X[mask], y[mask])
            results["scenarios"][scen][name] = {"auroc": a, "auroc_ci": [lo, hi], "auprc": ap}
            from pdvoice.config import LOGS
            g.to_csv(LOGS / f"fusion_preds__{scen}__{name.replace('-','_')}.csv", index=False)
            star = "  <<<" if name == "FUSION" else ""
            print(f"  {scen:<13}{name:<16}{a:.3f} [{lo:.3f},{hi:.3f}]      {ap:.3f}{star}")
        print()
    import json

    from pdvoice.config import LOGS
    (LOGS / "fusion__xls-r-300m__L15.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
