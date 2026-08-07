"""Reproducibility: one call to seed every RNG we use."""

from __future__ import annotations

import os
import random

import numpy as np

from pdvoice.config import SEED


def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
