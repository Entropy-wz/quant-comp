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
    window: int
    roll_feature_cols: list[str]


def fit_feature_v1(
    train_df: pd.DataFrame,
    window: int = 5,
    roll_cols: list[str] | None = None,
    max_roll_cols: int = 32,
) -> FeatureV1State:
    if window <= 0:
        raise ValueError("window must be positive")
    base = fit_feature_v0(train_df)
    if roll_cols is None:
        roll_cols = base.feature_cols[:max_roll_cols]
    else:
        roll_cols = [c for c in roll_cols if c in base.feature_cols][:max_roll_cols]
    return FeatureV1State(
        feature_cols=base.feature_cols,
        fill_values=base.fill_values,
        asset_levels=base.asset_levels,
        window=int(window),
        roll_feature_cols=list(roll_cols),
    )


def _filled_matrix(df: pd.DataFrame, cols: list[str], fill_values: dict[str, float]) -> np.ndarray:
    feats = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        fill = np.float32(fill_values.get(col, 0.0))
        s = s.fillna(fill).replace([np.inf, -np.inf], fill)
        feats.append(s.to_numpy(dtype=np.float32))
    return np.column_stack(feats) if feats else np.zeros((len(df), 0), dtype=np.float32)


def transform_feature_v1_frame(df: pd.DataFrame, state: FeatureV1State) -> np.ndarray:
    """Offline causal rolling: sort by asset,time; rolling uses current+past only."""
    base = transform_feature_v0(df, FeatureV0State(state.feature_cols, state.fill_values, state.asset_levels))
    if not state.roll_feature_cols:
        return base

    work = df.copy()
    work["_row"] = np.arange(len(work))
    work = work.sort_values(["asset_id", "time_id"], kind="mergesort").reset_index(drop=True)
    roll_mat = _filled_matrix(work, state.roll_feature_cols, state.fill_values)
    for j, col in enumerate(state.roll_feature_cols):
        work[f"__roll_{j}"] = roll_mat[:, j]

    g = work.groupby("asset_id", sort=False)
    mean_parts = []
    std_parts = []
    diff_parts = []
    for j, _col in enumerate(state.roll_feature_cols):
        s = work[f"__roll_{j}"]
        mean_parts.append(g[f"__roll_{j}"].transform(lambda x: x.rolling(state.window, min_periods=1).mean()).to_numpy(np.float32))
        std_parts.append(g[f"__roll_{j}"].transform(lambda x: x.rolling(state.window, min_periods=1).std(ddof=0)).fillna(0.0).to_numpy(np.float32))
        diff_parts.append(g[f"__roll_{j}"].transform(lambda x: x.diff().fillna(0.0)).to_numpy(np.float32))
        del s

    roll_mean = np.column_stack(mean_parts)
    roll_std = np.column_stack(std_parts)
    diff1 = np.column_stack(diff_parts)

    order = work["_row"].to_numpy()
    inv = np.empty(len(work), dtype=np.int64)
    inv[order] = np.arange(len(work))
    rolled = np.concatenate([roll_mean[inv], roll_std[inv], diff1[inv]], axis=1)
    return np.concatenate([base, rolled], axis=1).astype(np.float32, copy=False)


class RollingState:
    """Online per-asset deques matching transform_feature_v1_frame causality."""

    def __init__(self, state: FeatureV1State):
        self.state = state
        self.history: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=state.window)
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
            r_mean = stack.mean(axis=0).astype(np.float32)
            r_std = stack.std(axis=0).astype(np.float32) if len(stack) > 1 else np.zeros_like(r_mean)
            d1 = (roll_vals - prev).astype(np.float32) if prev is not None else np.zeros_like(roll_vals)
            rows.append(np.concatenate([current, r_mean, r_std, d1]))
        return np.vstack(rows).astype(np.float32, copy=False)


def state_to_jsonable(state: FeatureV1State) -> dict:
    return asdict(state)


def state_from_jsonable(payload: dict) -> FeatureV1State:
    return FeatureV1State(
        feature_cols=list(payload["feature_cols"]),
        fill_values={k: float(v) for k, v in payload["fill_values"].items()},
        asset_levels=[int(x) for x in payload["asset_levels"]],
        window=int(payload["window"]),
        roll_feature_cols=list(payload["roll_feature_cols"]),
    )
