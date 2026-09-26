"""Layer 2 (RQ2) — cross-dataset / cross-language transfer.

Lock the Layer-1 winner and apply it WITHOUT retraining to external datasets:

    train XLS-R -> head on mPower (subject-disjoint train/val) -> lock calibration
    + operating threshold on the mPower val fold -> for each external: embed with the
    SAME encoder, apply the locked head + calibrator, aggregate to subject level, and
    report AUROC / AUPRC / calibration / DCA + the transfer drop from the mPower ceiling.

Default = the clean cross-language VOWEL test (matches mPower's sustained /a/):
IPVS (Italian) + NeuroVoz (Spanish). MDVR-KCL (speech-only) is the separate cross-task
probe: run with --externals mdvr_kcl --task speech.

Run:  python scripts/layer2_transfer.py --model xls-r-300m --externals ipvs,neurovoz --task vowel
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from pdvoice.config import EMBEDDINGS, FIGURES, LOGS, SEED, VAL_FRACTION, ensure_dirs
from pdvoice.data import LOADERS
from pdvoice.cohort import load_mpower_cohort
from pdvoice.preprocess import WINDOW_MODES, quality_control

def _mpower_ceiling(model: str = "xls-r-300m", cap: int = 3, layer=None,
                    window: str = "current") -> float:
    """Layer-1 within-mPower AUROC, read from disk rather than pasted in.

    This was hardcoded at 0.839 and went stale when the cohort was corrected (the
    phantom-row fix took it to 0.827), overstating every transfer drop this script
    reports by 0.012. Read it so it cannot silently rot again.
    """
    import json as _json

    # The ceiling must come from the SAME layer being transferred, or the drop is an
    # apples-to-oranges subtraction. This used to read LNone unconditionally, which was
    # harmless while only the last layer was ever used and silently wrong the moment a
    # layer sweep started: a transfer at L11 was being scored against the last layer's
    # within-mPower number. Prefer the matching layer; fall back loudly, never quietly.
    # The WINDOW must match too, and this one never falls back across modes: a 2 s vowel
    # core and a full 10 s recording are different inputs, so subtracting one's ceiling
    # from the other's transfer would repeat exactly the mistake above in a second
    # dimension. Only the layer is allowed to degrade, and only with a warning.
    wtag = "" if window == "current" else f"__W{window}"
    f = LOGS / f"layer1_mpower__{model}__cap{cap}__L{layer}{wtag}.json"
    if not f.exists():
        alt = LOGS / f"layer1_mpower__{model}__cap{cap}__LNone{wtag}.json"
        if not alt.exists():
            raise SystemExit(f"missing {f}\nRun: scripts/layer1_mpower.py --model {model} "
                             f"--cap {cap} --layer {layer} --window {window}")
        print(f"      WARNING: no within-mPower ceiling for L{layer} (window={window}); "
              f"falling back to the last-layer ceiling.\n"
              f"      'transfer_drop_auroc' below is measured against a DIFFERENT layer "
              f"and is not comparable.\n"
              f"      Fix: scripts/layer1_mpower.py --model {model} --cap {cap} "
              f"--layer {layer} --window {window}")
        f = alt
    return float(_json.loads(f.read_text())["auroc"])


# ------------------------------------------------------------------ embedding (cached)
def embed(df: pd.DataFrame, model_key: str, layer, cache_path: Path,
          window: str = "current") -> np.ndarray:
    """Order-aligned pooled embeddings for df, cached to cache_path (keys 'X', 'path').

    A length check is NOT enough. The cohort's row ORDER and even its row SET can change
    while the count stays put -- swapping the cap and QC steps keeps 12,053 rows but
    exchanges 2,846 of them -- and X is consumed positionally, so a length-only guard
    hands back embeddings paired with the wrong recordings. Compare the paths.
    """
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        if "path" not in d:
            print("      cache predates the identity guard -> re-extracting")
        elif list(d["path"].astype(str)) != df.path.tolist():
            print("      cache does not match this cohort (order or membership)"
                  " -> re-extracting")
        else:
            print(f"      cache hit: {cache_path.name}  X={d['X'].shape}")
            return d["X"]

    if window != "current":
        raise SystemExit("\n".join([
            f"no cache for window={window!r} at {cache_path.name}.",
            "This script's own extraction path only implements the 'current' windowing,",
            "so it CANNOT build this cache -- it would silently write current-windowed",
            "embeddings under a non-current filename. Build it first:",
            f"  scripts/extract_windows.py --dataset <ds> --task <task> --window {window}",
            f"  scripts/slice_windows.py   --dataset <ds> --task <task> --window {window}"
            f" --layers {layer}",
        ]))

    from tqdm import tqdm

    from pdvoice.features.ssl import DEFAULT_MODELS, _embed_wave, load_encoder
    from pdvoice.preprocess import chunks, preprocess_clip

    model, extractor, device = load_encoder(DEFAULT_MODELS.get(model_key, model_key))
    print(f"      encoder {model_key} on {device}; embedding {len(df):,} recordings")
    vecs = []
    for row in tqdm(df.itertuples(index=False), total=len(df), unit="rec"):
        wav = preprocess_clip(row.path, row.task_kind)
        if row.task_kind == "vowel":
            v = _embed_wave(wav, model, extractor, device, layer, "mean_std")
        else:
            v = np.mean([_embed_wave(c, model, extractor, device, layer, "mean_std")
                         for c in chunks(wav)], axis=0)
        vecs.append(v.astype(np.float16))
    X = np.vstack(vecs).astype(np.float16)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, path=df.path.to_numpy().astype(str))
    return X


def subject_level(oof):
    return oof.groupby("subject_id").agg(y=("y", "first"), p=("p", "mean"),
                                         n_rec=("p", "size")).reset_index()


def evaluate(y, p, locked_threshold):
    from pdvoice.eval.discrimination import auprc, auroc, bootstrap_ci
    from pdvoice.eval.metrics import (calibration_error, sensitivity_at_specificity,
                                      threshold_metrics)
    from sklearn.metrics import brier_score_loss

    auc, alo, ahi = bootstrap_ci(y, p, metric=auroc)
    ap, plo, phi = bootstrap_ci(y, p, metric=auprc)
    ce = calibration_error(y, p)
    return {
        "n_subjects": int(len(y)), "n_pd": int((y == 1).sum()), "n_hc": int((y == 0).sum()),
        "auroc": auc, "auroc_ci": [alo, ahi], "auprc": ap, "auprc_ci": [plo, phi],
        "brier": float(brier_score_loss(y, p)), "ece": ce["ece"], "mce": ce["mce"],
        "at_locked_threshold": threshold_metrics(y, p, locked_threshold),
        "sensitivity_at_90_spec": sensitivity_at_specificity(y, p, 0.90),
    }


# --------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xls-r-300m")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--cap", type=int, default=3, help="mPower recordings/subject for training")
    ap.add_argument("--externals", default="ipvs,neurovoz")
    ap.add_argument("--task", default="vowel", help="external task_kind subset to test")
    # The 2 s vowel core and the 5 s speech chunk were inherited defaults, never measured.
    # "current" keeps them and reads the legacy caches unchanged, so every existing number
    # is reproducible; the other modes read caches built by scripts/extract_windows.py.
    ap.add_argument("--window", default="current", choices=list(WINDOW_MODES),
                    help="windowing mode; see pdvoice.preprocess.window_clip")
    ap.add_argument("--adapt", choices=["none", "hc-centroid", "hc-standardize"], default="none",
                    help="hc-centroid: shift target embeddings by (mu_source_HC - mu_target_HC) "
                         "(mean alignment). hc-standardize: align mean AND per-feature scale using "
                         "target-HC stats -> source-HC stats. Both report before + after.")
    args = ap.parse_args()
    externals = [e.strip() for e in args.externals.split(",") if e.strip()]

    ensure_dirs()
    t0 = time.time()
    print(f"=== Layer 2 transfer | encoder={args.model} L{args.layer} | task={args.task} ===")
    ceiling = _mpower_ceiling(args.model, args.cap, args.layer, args.window)
    import torch

    from pdvoice.calibration import select_best_by_brier
    from pdvoice.eval.metrics import youden_threshold
    from pdvoice.models.head import fit_mlp_head

    # ---- 1. mPower training embeddings (reuse Layer-1 cache) ----
    print("[1] mPower train set ...")
    # QC runs BEFORE the cap in here; this script used to do the reverse.
    mp = load_mpower_cohort(cap=args.cap, seed=SEED)
    wtag = "" if args.window == "current" else f"__W{args.window}"
    mp_cache = EMBEDDINGS / (f"mpower_pilot__{args.model}__cap{args.cap}"
                             f"__L{args.layer}__seed{SEED}{wtag}.npz")
    Xmp = embed(mp, args.model, args.layer, mp_cache, args.window)
    ymp = mp["label"].astype(int).to_numpy()

    # ---- 2. subject-disjoint train/val, fit head, LOCK calibration + threshold ----
    print("[2] Train head on mPower; lock calibration + threshold on val ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr, va = next(gss.split(Xmp, ymp, mp["subject_id"]))
    model, val_logits = fit_mlp_head(Xmp[tr], ymp[tr], Xmp[va], ymp[va])
    cal_name, cal = select_best_by_brier(val_logits, ymp[va])
    thr = youden_threshold(ymp[va], cal.predict_proba(val_logits))
    dev = next(model.parameters()).device
    print(f"      calibrator={cal_name}  locked threshold={thr:.3f}  "
          f"(train={len(tr)} val={len(va)} subjects)")

    def score(X):
        with torch.no_grad():
            lg = model(torch.tensor(np.asarray(X), dtype=torch.float32, device=dev)).squeeze(-1)
        return cal.predict_proba(lg.cpu().numpy())

    def eval_probs(df, p):
        subj = subject_level(pd.DataFrame(
            {"subject_id": df["subject_id"], "y": df["label"].astype(int), "p": p}))
        m = evaluate(subj["y"].to_numpy(), subj["p"].to_numpy(), thr)
        m["transfer_drop_auroc"] = round(ceiling - m["auroc"], 3)
        return subj, m

    # source healthy-voice statistics (mPower HC recordings), for domain adaptation
    _src_hc = Xmp.astype(np.float32)[ymp == 0]
    mu_src_hc = _src_hc.mean(axis=0)
    sig_src_hc = _src_hc.std(axis=0) + 1e-6

    # ---- 3. each external: embed, apply locked model, evaluate (before + optional after) ----
    results = {}
    for ds in externals:
        print(f"[3] External: {ds} ({args.task}) ...")
        df = LOADERS[ds]()
        df = df[df["task_kind"] == args.task].copy()
        if df.empty:
            print(f"      no '{args.task}' recordings in {ds} — skipping")
            continue
        df, _ = quality_control(df)
        df = df.reset_index(drop=True)
        cache = EMBEDDINGS / f"{ds}__{args.model}__{args.task}__L{args.layer}{wtag}.npz"
        X = embed(df, args.model, args.layer, cache, args.window)
        y_rec = df["label"].astype(int).to_numpy()
        stem = f"layer2__{args.model}__{ds}__{args.task}__L{args.layer}{wtag}"

        subj_b, m_b = eval_probs(df, score(X))
        m_b["dataset"] = ds
        m_b["language"] = str(df["language"].iloc[0])
        rec = {"dataset": ds, "language": m_b["language"], "before": m_b}
        (LOGS / f"{stem}.json").write_text(json.dumps(m_b, indent=2))
        subj_b.to_csv(LOGS / f"{stem}__subject_preds.csv", index=False)

        if args.adapt != "none":
            n_hc = int((y_rec == 0).sum())
            if n_hc == 0:
                print(f"      no HC recordings in {ds} — cannot estimate target centroid; skip adapt")
            else:
                # bring target embeddings into mPower's frame (where the head was trained)
                # by aligning the healthy-voice statistics target -> source.
                Xf = X.astype(np.float32)
                mu_tgt_hc = Xf[y_rec == 0].mean(axis=0)
                if args.adapt == "hc-centroid":                       # mean only
                    X_adj = Xf - mu_tgt_hc + mu_src_hc
                else:                                                 # hc-standardize: mean + scale
                    sig_tgt_hc = Xf[y_rec == 0].std(axis=0) + 1e-6
                    X_adj = (Xf - mu_tgt_hc) / sig_tgt_hc * sig_src_hc + mu_src_hc
                subj_a, m_a = eval_probs(df, score(X_adj))
                m_a["dataset"], m_a["language"] = ds, rec["language"]
                m_a["adapt"], m_a["target_hc_recordings_used"] = args.adapt, n_hc
                rec["after"] = m_a
                (LOGS / f"{stem}__adapt-{args.adapt}.json").write_text(json.dumps(m_a, indent=2))
                subj_a.to_csv(LOGS / f"{stem}__adapt-{args.adapt}__subject_preds.csv", index=False)
        results[ds] = rec

    # ---- 4. summary ----
    adapted = args.adapt != "none"
    print("\n===================== LAYER-2 TRANSFER =====================")
    print(f"  mPower internal ceiling (Layer-1 CV, from disk): AUROC {ceiling:.3f}")
    if adapted:
        print(f"  adaptation: {args.adapt}  (align target HC stats -> source HC stats)")
        print(f"  {'dataset':<11}{'lang':<5}{'subj':<6}{'AUROC before':<14}{'AUROC after':<14}{'delta':<8}{'ECE b->a'}")
        for ds, r in results.items():
            b = r["before"]; a = r.get("after")
            if a:
                d = a["auroc"] - b["auroc"]
                print(f"  {ds:<11}{r['language']:<5}{b['n_subjects']:<6}"
                      f"{b['auroc']:.3f}{'':9}{a['auroc']:.3f}{'':9}{d:+.3f}   "
                      f"{b['ece']:.2f}->{a['ece']:.2f}")
            else:
                print(f"  {ds:<11}{r['language']:<5}{b['n_subjects']:<6}{b['auroc']:.3f}   (no HC → no adapt)")
    else:
        print(f"  {'dataset':<12}{'lang':<6}{'subj':<7}{'AUROC':<20}{'AUPRC':<8}{'ECE':<7}{'drop'}")
        for ds, r in results.items():
            m = r["before"]; ci = f"[{m['auroc_ci'][0]:.3f},{m['auroc_ci'][1]:.3f}]"
            print(f"  {ds:<12}{m['language']:<6}{m['n_subjects']:<7}"
                  f"{m['auroc']:.3f} {ci:<14}{m['auprc']:.3f}   {m['ece']:.3f}  -{m['transfer_drop_auroc']:.3f}")
    print("============================================================")
    print(f"Total {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
