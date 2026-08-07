"""Decision Curve Analysis (PROJECT_CONTEXT.md §4.4).

Net benefit of the model vs treat-all / treat-none over the screening threshold band
(default 0.05-0.30). NB = TP/n - FP/n * (pt / (1 - pt)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdvoice.config import DCA_THRESHOLDS


def net_benefit(y, p, thresholds=DCA_THRESHOLDS) -> pd.DataFrame:
    """Return a DataFrame with net benefit for model / treat-all / treat-none."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    prevalence = float(y.mean()) if n else 0.0

    rows = []
    for pt in thresholds:
        w = pt / (1 - pt)
        pred = p >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        rows.append(
            {
                "threshold": float(pt),
                "model": tp / n - fp / n * w,
                "treat_all": prevalence - (1 - prevalence) * w,
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)
