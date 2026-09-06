"""The shallow classifier head (PROJECT_CONTEXT.md §4.2).

One hidden layer of 256 units (ReLU + batch-norm + dropout) -> single logit.
Deliberately shallow to avoid overfitting mPower's noisy self-report labels, and
shared across all encoders so the encoder comparison is fair. Outputs LOGITS;
apply the sigmoid at eval/calibration time.
"""

from __future__ import annotations

import numpy as np


def build_head(in_dim: int, hidden: int = 256, dropout: float = 0.3):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def fit_mlp_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden: int = 256,
    dropout: float = 0.3,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 20,
    device: str | None = None,
    seed: int = 42,
):
    """Train the head on cached embeddings; early-stop on val loss.

    Returns (model, val_logits). Use val_logits to fit the calibration map and to
    pick the operating threshold — then lock both before touching any test fold.

    ``seed`` makes weight-init and dropout deterministic so results are reproducible
    run-to-run (an unseeded head swung AUROC by up to ~0.04 on the XLS-R embeddings).
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.tensor(np.asarray(x_train), dtype=torch.float32, device=device)
    yt = torch.tensor(np.asarray(y_train), dtype=torch.float32, device=device)
    xv = torch.tensor(np.asarray(x_val), dtype=torch.float32, device=device)
    yv = torch.tensor(np.asarray(y_val), dtype=torch.float32, device=device)

    model = build_head(xt.shape[1], hidden, dropout).to(device)
    # class imbalance -> positive weight
    pos = float(yt.sum().item()) or 1.0
    neg = float((yt == 0).sum().item()) or 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_state, best_val, wait = None, float("inf"), 0
    epochs_run, stopped_early = 0, False
    for _ in range(max_epochs):
        epochs_run += 1
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv).squeeze(-1), yv).item()
        if val_loss < best_val - 1e-5:
            best_val, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                stopped_early = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(xv).squeeze(-1).cpu().numpy()
    # Diagnostics for the caller (attributes, so the return signature is unchanged).
    # stopped_early False means training hit `max_epochs` with validation loss still
    # improving -- i.e. the head is UNDERFITTED and every downstream number is
    # pessimistic. Worth asserting on when the training set grows.
    model._epochs_run = epochs_run
    model._stopped_early = stopped_early
    model._best_val_loss = best_val
    return model, val_logits
