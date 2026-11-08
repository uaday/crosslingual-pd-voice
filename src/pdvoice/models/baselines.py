"""Layer-1 baselines.

- ``sklearn_baseline``: classical-feature or embedding classifiers (SVM / RF / GBT),
  standardised, with calibrated probability output.
- ``LogMelCNN``: a small end-to-end 2D-CNN on log-mel spectrograms. The class is
  provided; the log-mel dataloader/training loop is a TODO for the implementation phase.
"""

from __future__ import annotations


def sklearn_baseline(name: str = "svm", **kwargs):
    """Return a StandardScaler + classifier pipeline with predict_proba support."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    models = {
        "svm": SVC(probability=True, kernel="rbf", **kwargs),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, **kwargs),
        "gbt": GradientBoostingClassifier(**kwargs),
    }
    if name not in models:
        raise ValueError(f"unknown baseline '{name}' (choose from {list(models)})")
    return make_pipeline(StandardScaler(), models[name])


def build_logmel_cnn(n_mels: int = 64, n_classes: int = 1):
    """A compact 2D-CNN for log-mel spectrogram input (B, 1, n_mels, T)."""
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(1, 16),
        block(16, 32),
        block(32, 64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, n_classes),
    )
