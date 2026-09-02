"""Is the within-IPVS speech AUROC = 1.000 explained by a PD-only recording batch?

IPVS contains an acquisition split that is confounded with the label: PD speaker
folders ``1-5`` and ``6-10`` (10 of 28 patients) are 44.1 kHz, while the other 18
patients and ALL 37 controls are 16 kHz. Native rate 44.1 kHz therefore implies PD
with perfect precision, so a model can partly shortcut the task.

``verify_ipvs.py`` could not detect this: label permutation destroys real and
artifact signal alike (it tests plumbing leaks, not class-correlated confounds), and
dropping the young controls is orthogonal to the batch split.

Tests:
  0. Rate alone -- AUROC of the single feature "native_sr == 44100" at subject level.
  1. Baseline -- all subjects (the reported 1.000).
  2. Drop the 10 x 44.1 kHz PD subjects -> 18 PD vs 37 HC, ALL natively 16 kHz.
  3. Also drop young HC -> 18 PD vs 22 elderly HC, all 16 kHz (both confounds removed).
  4. Within-PD rate probe -- predict native rate using PD subjects only, where the
     disease label is constant so any separation is pure artifact.

Reading: if [2] stays near 1.0 the speech signal is genuine and the verification is
STRENGTHENED. If [2] drops materially, the reported ceiling was partly the batch.

Usage:
    .venv\\Scripts\\python.exe -u scripts\\verify_ipvs_batch.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from pdvoice.calibration import select_best_by_brier
from pdvoice.config import EMBEDDINGS, LOGS, SEED
from pdvoice.data.ipvs import load_ipvs
from pdvoice.data.splits import assign_folds, train_val_test_indices
from pdvoice.eval.discrimination import auroc, bootstrap_ci
from pdvoice.models.head import fit_mlp_head
from pdvoice.preprocess import quality_control

MODEL, LAYER = "xls-r-300m", 15


def load_speech():
    df = load_ipvs()
    df = df[df["task_kind"] == "speech"].copy()
    df, _ = quality_control(df)
    df = df.reset_index(drop=True)
    X = np.load(EMBEDDINGS / f"ipvs__{MODEL}__speech__L{LAYER}.npz")["X"].astype(np.float32)
    assert X.shape[0] == len(df), f"cache/manifest misalignment: {X.shape[0]} != {len(df)}"
    df["grp"] = df["subject_id"].str.extract(r"ipvs_([a-z]+)_")
    return df, X


def within_cv(df, X, y, seed=SEED):
    """5-fold subject-disjoint CV -> subject-level AUROC + the per-subject frame."""
    df = df.reset_index(drop=True)
    dfa = assign_folds(df.assign(label=y), n_splits=5, seed=seed)
    oof = []
    for k in range(5):
        tr, va, te = train_val_test_indices(dfa, test_fold=k)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        m, vlog = fit_mlp_head(X[tr], y[tr], X[va], y[va])
        with torch.no_grad():
            dev = next(m.parameters()).device
            tl = m(torch.tensor(X[te], dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
        _, cal = select_best_by_brier(vlog, y[va])
        oof.append(pd.DataFrame({"s": dfa.loc[te, "subject_id"].to_numpy(), "y": y[te],
                                 "p": cal.predict_proba(tl)}))
    g = pd.concat(oof).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    return auroc(g["y"], g["p"]), g


def composition(df, mask) -> dict:
    s = df[mask].drop_duplicates("subject_id")
    out = {g: int((s["grp"] == g).sum()) for g in ("yhc", "ehc", "pd")}
    out["rates"] = {int(k): int(v) for k, v in s["native_sr"].value_counts().items()}
    return out


def main() -> None:
    df, X = load_speech()
    y = df["label"].astype(int).to_numpy()
    all_mask = np.ones(len(df), bool)
    res: dict = {"model": MODEL, "layer": LAYER, "task": "speech"}

    print("=== IPVS speech: is the 1.000 a PD-only recording batch? ===")
    print(f"  recordings {len(df)} | composition {composition(df, all_mask)}\n")

    # --- 0. what does the sample rate alone buy? -----------------------------
    sub = df.groupby("subject_id").agg(label=("label", "first"), sr=("native_sr", "max"))
    a_rate = auroc(sub["label"], (sub["sr"] == 44100).astype(float))
    res["rate_alone_auroc"] = round(a_rate, 4)
    print(f"[0 rate alone]  AUROC of 'native_sr == 44100' = {a_rate:.3f}   "
          "(a pure acquisition artifact)")

    # --- 1. baseline ---------------------------------------------------------
    a_all, g_all = within_cv(df, X, y)
    ci_all = bootstrap_ci(g_all["y"].to_numpy(), g_all["p"].to_numpy())
    res["baseline"] = {"auroc": round(a_all, 4), "ci": [round(ci_all[1], 4), round(ci_all[2], 4)],
                       "composition": composition(df, all_mask)}
    print(f"[1 baseline]    all subjects: AUROC = {a_all:.3f} "
          f"[{ci_all[1]:.3f}, {ci_all[2]:.3f}]")

    # --- 2. drop the 44.1 kHz PD batch --------------------------------------
    m2 = ~((df["native_sr"] == 44100) & (df["label"] == 1)).to_numpy()
    a2, g2 = within_cv(df[m2], X[m2], y[m2])
    ci2 = bootstrap_ci(g2["y"].to_numpy(), g2["p"].to_numpy())
    res["drop_44k_pd"] = {"auroc": round(a2, 4), "ci": [round(ci2[1], 4), round(ci2[2], 4)],
                          "composition": composition(df, m2)}
    print(f"[2 drop 44.1k PD batch]  all-16 kHz cohort: AUROC = {a2:.3f} "
          f"[{ci2[1]:.3f}, {ci2[2]:.3f}]   (subjects {composition(df, m2)})")

    # --- 3. strictest: also drop young HC -----------------------------------
    m3 = m2 & (df["grp"] != "yhc").to_numpy()
    a3, g3 = within_cv(df[m3], X[m3], y[m3])
    ci3 = bootstrap_ci(g3["y"].to_numpy(), g3["p"].to_numpy())
    res["drop_44k_pd_and_young_hc"] = {"auroc": round(a3, 4),
                                       "ci": [round(ci3[1], 4), round(ci3[2], 4)],
                                       "composition": composition(df, m3)}
    print(f"[3 + drop young HC]      elderly-HC vs 16 kHz PD: AUROC = {a3:.3f} "
          f"[{ci3[1]:.3f}, {ci3[2]:.3f}]   (subjects {composition(df, m3)})")

    # --- 4. within-PD probe: is the batch encoded at all? -------------------
    pdm = (df["label"] == 1).to_numpy()
    y_rate = (df.loc[pdm, "native_sr"] == 44100).astype(int).to_numpy()
    a4, _ = within_cv(df[pdm], X[pdm], y_rate)
    res["within_pd_rate_probe"] = round(a4, 4)
    print(f"[4 within-PD probe]      predict native rate, PD only: AUROC = {a4:.3f}   "
          "(0.5 = artifact invisible, 1.0 = fully encoded)")

    delta = a_all - a2
    print(f"\n  baseline - all-16 kHz = {delta:+.3f}")
    if abs(delta) < 0.03:
        print("  READING: the speech signal survives removing the batch. The 1.000 was NOT")
        print("           the artifact -- this STRENGTHENS the verification.")
    else:
        print("  READING: removing the batch moves the number materially. The reported")
        print("           ceiling was partly an acquisition shortcut, not disease signal.")

    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / "verify_ipvs_batch.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
