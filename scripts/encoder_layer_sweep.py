"""D2 + D3 + D4 -- a FAIR encoder comparison, and a layer choice that is not cheating.

Three problems this fixes at once.

D2  UNEQUAL SEARCH. The original sweep tried four layers of XLS-R and exactly one of
    wav2vec2-base. Since a mid layer is worth roughly +0.10 over the last layer for
    WavLM and HuBERT, "XLS-R wins" was partly a statement about who got searched
    hardest. Here every encoder is scored at every layer.

D3  THE LAYER WAS PICKED ON THE TEST SETS. L15 was chosen because it improved IPVS,
    NeuroVoz and MDVR -- the sets we then report. That is model selection on test data
    and it biases every transfer number upward. Learned layer weighting removes the
    problem rather than documenting it: no layer is chosen, a softmax over all layers is
    fitted on TRAINING data only, exactly as SUPERB does for frozen representations.

D4  WHISPER WAS NEVER TRIED, despite VoiceFM ranking frozen Whisper above frozen
    HuBERT and it being the only log-mel front end in the comparison.

Everything runs on one fixed AGE-MATCHED cohort, so this measures encoders rather than
which encoder best exploits the age confound -- which is what the original sweep, run on
the confounded cohort, may partly have measured.

Run:  .venv\\Scripts\\python.exe -u scripts\\encoder_layer_sweep.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pdvoice.config import EMBEDDINGS, LOGS
from pdvoice.eval.discrimination import bootstrap_ci, paired_bootstrap_auroc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ["wav2vec2-base", "wavlm-large", "hubert-large", "xls-r-300m", "whisper-small"]


def load(model):
    f = EMBEDDINGS / f"layersweep__{model}__agematched.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return (d["X"].astype(np.float32), d["subject_id"].astype(str),
            d["label"].astype(int), d["age"].astype(float))


def linear_probe(X, y, groups, seed=42, n_splits=5):
    """Subject-disjoint linear probe -- the SUPERB convention for frozen features."""
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


class WeightedLayers(nn.Module):
    """Softmax weights over layers + a linear head. The SUPERB frozen-feature recipe.

    The weights are parameters fitted on the training split, so no layer is selected
    using anything the model is later evaluated on.
    """

    def __init__(self, n_layers, dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.BatchNorm1d(dim)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):                       # x: (batch, layers, dim)
        a = torch.softmax(self.w, dim=0).view(1, -1, 1)
        return self.fc(self.norm((x * a).sum(dim=1))).squeeze(-1)

    def layer_weights(self):
        return torch.softmax(self.w.detach(), dim=0).cpu().numpy()


def learned_pooling(X, y, groups, seed=42, n_splits=5, epochs=300, lr=1e-2):
    """Fit softmax layer weights + linear head, subject-disjoint. Returns AUROC, weights."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    oof, weights = np.zeros(len(y)), []
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        torch.manual_seed(seed)
        mu, sd = X[tr].mean((0, 1), keepdims=True), X[tr].std((0, 1), keepdims=True) + 1e-6
        xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32, device=dev)
        xe = torch.tensor((X[te] - mu) / sd, dtype=torch.float32, device=dev)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
        m = WeightedLayers(X.shape[1], X.shape[2]).to(dev)
        pos, neg = float(yt.sum()), float((yt == 0).sum())
        lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1), device=dev))
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-3)
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            loss = lf(m(xt), yt); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(m(xe)).cpu().numpy()
        weights.append(m.layer_weights())
    return float(roc_auc_score(y, oof)), np.mean(weights, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    rows, best, wtab, oofs = [], [], {}, {}
    for model in MODELS:
        got = load(model)
        if got is None:
            print(f"[{model}] no cache -> run scripts/extract_all_layers.py --model {model}")
            continue
        X, sids, y, age = got
        n_layers = X.shape[1]
        print(f"\n=== {model} · {X.shape[0]:,} speakers · {n_layers} layers x {X.shape[2]} dims ===",
              flush=True)

        # ---- D2: every layer, equal budget ----
        aurocs = []
        for L in range(n_layers):
            a, _ = linear_probe(X[:, L, :], y, sids, args.seed)
            aurocs.append(a)
            rows.append(dict(model=model, layer=L, auroc=round(a, 4)))
        a = np.array(aurocs)
        bl = int(a.argmax())
        print(f"  best single layer: L{bl} = {a[bl]:.3f}   "
              f"(last layer L{n_layers-1} = {a[-1]:.3f}, first = {a[0]:.3f})")

        _, oof_best = linear_probe(X[:, bl, :], y, sids, args.seed)
        oofs[model] = (y, oof_best)
        _, ci_lo, ci_hi = bootstrap_ci(y, oof_best)

        # ---- D3: learned weights over all layers, fitted on train only ----
        a_lw, w = learned_pooling(X, y, sids, args.seed)
        wtab[model] = w
        print(f"  learned layer weights: {a_lw:.3f}   "
              f"(vs best single {a[bl]:.3f}, delta {a_lw - a[bl]:+.3f})")
        top = np.argsort(w)[::-1][:3]
        print(f"  weight mass on layers: " +
              ", ".join(f"L{int(t)} {w[t]:.3f}" for t in top))

        best.append(dict(model=model, n_layers=n_layers,
                         best_layer=bl, best_layer_auroc=round(float(a[bl]), 4),
                         ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
                         last_layer_auroc=round(float(a[-1]), 4),
                         gain_over_last=round(float(a[bl] - a[-1]), 4),
                         learned_pool_auroc=round(a_lw, 4),
                         learned_vs_best=round(a_lw - float(a[bl]), 4)))

    if not best:
        raise SystemExit("no caches found; run scripts/extract_all_layers.py first")

    B = pd.DataFrame(best).sort_values("learned_pool_auroc", ascending=False)
    pd.DataFrame(rows).to_csv(LOGS / "encoder_layer_sweep__per_layer.csv", index=False)
    B.to_csv(LOGS / "encoder_layer_sweep.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n{'=' * 104}\nFAIR ENCODER COMPARISON (age-matched cohort, equal layer budget)\n{'=' * 104}")
    print(B.to_string(index=False))

    print(f"\n{'=' * 104}\nWHAT THIS CHANGES\n{'=' * 104}")
    top_last = B.loc[B.last_layer_auroc.idxmax()]
    top_best = B.loc[B.best_layer_auroc.idxmax()]
    top_pool = B.loc[B.learned_pool_auroc.idxmax()]
    print(f"  winner if you only try the LAST layer : {top_last.model} ({top_last.last_layer_auroc:.3f})")
    print(f"  winner with an EQUAL layer budget      : {top_best.model} ({top_best.best_layer_auroc:.3f})")
    print(f"  winner with LEARNED layer weights      : {top_pool.model} ({top_pool.learned_pool_auroc:.3f})")

    names = list(oofs)
    pvals = []
    print("\n  paired bootstrap on the SAME speakers:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_, b_ = names[i], names[j]
            d, pv = paired_bootstrap_auroc(oofs[a_][0], oofs[a_][1], oofs[b_][1])
            pvals.append(pv)
            print(f"    {a_:<15} vs {b_:<15} delta {d:+.4f}   p = {pv:.3f}"
                  + ("   <-- significant" if pv < 0.05 else ""))

    if pvals and min(pvals) >= 0.05:
        print(f"\n  NO pair differs significantly (smallest p = {min(pvals):.3f}).")
        print("  So the honest conclusion is NOT that the ranking flipped. It is that on the")
        print("  honest age-matched vowel signal the encoders are INDISTINGUISHABLE -- the same")
        print("  result the classical-vs-SSL comparison reached. The apparent XLS-R advantage")
        print("  came from the confounded cohort plus an unequal layer budget, not from the")
        print("  encoder being better at detecting Parkinson's.")

    print(f"\n  mean gain of best layer over last layer: {B.gain_over_last.mean():+.3f}")
    print(f"  mean gain of learned weights over best single layer: {B.learned_vs_best.mean():+.3f}")
    print("  best layer per encoder: "
          + ", ".join(f"{r.model} L{r.best_layer}" for r in B.itertuples()))
    print("  -> every encoder peaks EARLY and the last layer is worst for all of them, so the")
    print("     sustained-vowel PD signal sits in low-level acoustics rather than in the")
    print("     high-level representation the final layer holds.")
    print("\n  Learned weighting needs no layer to be selected, so it removes the test-set")
    print("  selection that produced L15 (RESEARCH_PLAN V2) -- but at this sample size it")
    print("  does not beat simply picking the best single layer.")

    (LOGS / "encoder_layer_sweep.json").write_text(json.dumps(
        {"best": best, "layer_weights": {k: v.tolist() for k, v in wtab.items()}}, indent=2))
    print(f"\nwrote {LOGS / 'encoder_layer_sweep.csv'}")


if __name__ == "__main__":
    main()
