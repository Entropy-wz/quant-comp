from __future__ import annotations

import numpy as np
import pandas as pd

from contest.io import responder_columns


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    w = w[mask]
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    xc = x - mx
    yc = y - my
    cov = float(np.sum(w * xc * yc))
    vx = float(np.sum(w * xc * xc))
    vy = float(np.sum(w * yc * yc))
    denom = np.sqrt(vx * vy)
    if denom <= 0:
        return 0.0
    return cov / denom


def rank_responders_by_target_corr(
    df: pd.DataFrame,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    cols = responder_columns(df)
    if not cols:
        return []
    y = pd.to_numeric(df["target"], errors="coerce").to_numpy(np.float64)
    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    scored: list[tuple[str, float]] = []
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
        scored.append((col, float(weighted_corr(x, y, w))))
    scored.sort(key=lambda t: abs(t[1]), reverse=True)
    return scored[: max(0, int(top_k))]


def choose_blend_weight(
    y_true: np.ndarray,
    pred_main: np.ndarray,
    pred_aux: np.ndarray,
    weight: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return (best_a, best_score) for a * main + (1-a) * aux."""
    from contest.metrics import weighted_zero_mean_r2

    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    best_a = 1.0
    best_score = -1e18
    for a in grid:
        blended = a * pred_main + (1.0 - a) * pred_aux
        score = weighted_zero_mean_r2(y_true, blended, weight)
        if score > best_score:
            best_score = score
            best_a = float(a)
    return best_a, float(best_score)