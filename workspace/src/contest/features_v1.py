from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from contest.features_v0 import FeatureV0State, fit_feature_v0, transform_feature_v0


@dataclass
class FeatureV1State:
    feature_cols: list[str]
    fill_values: dict[str, float]
    asset_levels: list[int]
    window: int  # max window (history length); kept for backward compat
    windows: list[int]
    roll_feature_cols: list[str]


def fit_feature_v1(
    train_df: pd.DataFrame,
    window: int = 5,
    windows: list[int] | None = None,
    roll_cols: list[str] | None = None,
    max_roll_cols: int = 32,
) -> FeatureV1State:
    if windows is None:
        windows = [int(window)]
    windows = sorted({int(w) for w in windows if int(w) > 0})
    if not windows:
        raise ValueError("windows must contain at least one positive int")
    max_w = max(windows)
    base = fit_feature_v0(train_df)
    if roll_cols is None:
        roll_cols = base.feature_cols[:max_roll_cols]
    else:
        roll_cols = [c for c in roll_cols if c in base.feature_cols][:max_roll_cols]
    return FeatureV1State(
        feature_cols=base.feature_cols,
        fill_values=base.fill_values,
        asset_levels=base.asset_levels,
        window=max_w,
        windows=windows,
        roll_feature_cols=list(roll_cols),
    )


def _causal_roll_mean_std(block: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n, k = block.shape
    if n == 0:
        z = np.zeros((0, k), dtype=np.float32)
        return z, z
    c1 = np.cumsum(block, axis=0, dtype=np.float64)
    c2 = np.cumsum(np.square(block, dtype=np.float64), axis=0)
    c1_pad = np.vstack([np.zeros((1, k), dtype=np.float64), c1])
    c2_pad = np.vstack([np.zeros((1, k), dtype=np.float64), c2])
    idx = np.arange(n)
    start = np.maximum(0, idx + 1 - window)
    sum1 = c1_pad[idx + 1] - c1_pad[start]
    sum2 = c2_pad[idx + 1] - c2_pad[start]
    count = (idx - start + 1).astype(np.float64)[:, None]
    mean64 = sum1 / count
    var = np.maximum(sum2 / count - mean64 ** 2, 0.0)
    mean = mean64.astype(np.float32)
    std = np.sqrt(var).astype(np.float32)
    std[count[:, 0] <= 1] = 0.0
    return mean, std


def _causal_diff1(block: np.ndarray) -> np.ndarray:
    n, k = block.shape
    diff = np.zeros((n, k), dtype=np.float32)
    if n > 1:
        diff[1:] = (block[1:] - block[:-1]).astype(np.float32)
    return diff


def transform_feature_v1_frame(df: pd.DataFrame, state: FeatureV1State) -> np.ndarray:
    """Offline causal multi-window rolling + (x-mean) + diff1."""
    base = transform_feature_v0(df, FeatureV0State(state.feature_cols, state.fill_values, state.asset_levels))
    if not state.roll_feature_cols:
        return base

    n = len(df)
    asset_ids = df["asset_id"].to_numpy()
    time_ids = df["time_id"].to_numpy()
    order = np.lexsort((time_ids, asset_ids))

    roll_cols = []
    for col in state.roll_feature_cols:
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32, copy=True)
        bad = ~np.isfinite(s)
        if bad.any():
            s[bad] = np.float32(state.fill_values.get(col, 0.0))
        roll_cols.append(s[order])
    roll_mat = np.column_stack(roll_cols)
    del roll_cols

    ordered_assets = asset_ids[order]
    k = roll_mat.shape[1]
    windows = list(state.windows) if getattr(state, "windows", None) else [int(state.window)]
    parts: list[np.ndarray] = []

    boundaries = np.flatnonzero(np.r_[True, ordered_assets[1:] != ordered_assets[:-1], True])
    for w in windows:
        mean_all = np.empty((n, k), dtype=np.float32)
        std_all = np.empty((n, k), dtype=np.float32)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            m, s = _causal_roll_mean_std(roll_mat[start:end], w)
            mean_all[start:end] = m
            std_all[start:end] = s
        delta = (roll_mat - mean_all).astype(np.float32, copy=False)
        parts.extend([mean_all, std_all, delta])

    diff1 = np.empty((n, k), dtype=np.float32)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        diff1[start:end] = _causal_diff1(roll_mat[start:end])
    parts.append(diff1)
    del roll_mat

    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    rolled = np.concatenate([p[inv] for p in parts], axis=1)
    del parts, inv, order
    return np.concatenate([base, rolled], axis=1)


class RollingState:
    """Online per-asset history matching transform_feature_v1_frame."""

    def __init__(self, state: FeatureV1State):
        self.state = state
        self.windows = list(state.windows) if getattr(state, "windows", None) else [int(state.window)]
        self.history: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=max(self.windows))
        )
        self.last_time_id: int | None = None
        self._roll_index = [state.feature_cols.index(c) for c in state.roll_feature_cols]

    def update_and_featurize(
        self,
        time_id: int,
        asset_ids: np.ndarray,
        base_matrix: np.ndarray,
    ) -> np.ndarray:
        if self.last_time_id is not None and time_id < self.last_time_id:
            raise ValueError("time_id must be non-decreasing in online mode")
        self.last_time_id = int(time_id)

        rows = []
        for i, asset_id in enumerate(np.asarray(asset_ids)):
            current = base_matrix[i]
            roll_vals = current[self._roll_index] if self._roll_index else np.zeros(0, dtype=np.float32)
            hist = self.history[int(asset_id)]
            prev = hist[-1] if hist else None
            hist.append(roll_vals.astype(np.float32, copy=True))
            stack = np.vstack(list(hist))
            feats = [current]
            for w in self.windows:
                window_stack = stack[-w:]
                r_mean = window_stack.mean(axis=0).astype(np.float32)
                r_std = (
                    window_stack.std(axis=0).astype(np.float32)
                    if len(window_stack) > 1
                    else np.zeros_like(r_mean)
                )
                delta = (roll_vals - r_mean).astype(np.float32)
                feats.extend([r_mean, r_std, delta])
            d1 = (roll_vals - prev).astype(np.float32) if prev is not None else np.zeros_like(roll_vals)
            feats.append(d1)
            rows.append(np.concatenate(feats))
        return np.vstack(rows).astype(np.float32, copy=False)


def state_to_jsonable(state: FeatureV1State) -> dict:
    return asdict(state)


def state_from_jsonable(payload: dict) -> FeatureV1State:
    windows = payload.get("windows")
    window = int(payload.get("window", 5))
    if not windows:
        windows = [window]
    windows = [int(w) for w in windows]
    return FeatureV1State(
        feature_cols=list(payload["feature_cols"]),
        fill_values={k: float(v) for k, v in payload["fill_values"].items()},
        asset_levels=[int(x) for x in payload["asset_levels"]],
        window=max(windows),
        windows=windows,
        roll_feature_cols=list(payload["roll_feature_cols"]),
    )
