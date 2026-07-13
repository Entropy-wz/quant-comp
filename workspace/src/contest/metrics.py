from __future__ import annotations

import numpy as np


def weighted_zero_mean_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denom = float(np.sum(weight * y_true * y_true))
    if denom <= 0.0:
        return 0.0
    numer = float(np.sum(weight * (y_true - y_pred) ** 2))
    return float(1.0 - numer / denom)
