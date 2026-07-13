from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from contest.io import feature_columns


@dataclass
class FeatureV0State:
    feature_cols: list[str]
    fill_values: dict[str, float]
    asset_levels: list[int]


def fit_feature_v0(train_df: pd.DataFrame) -> FeatureV0State:
    cols = feature_columns(train_df)
    fill_values = {}
    for col in cols:
        median = pd.to_numeric(train_df[col], errors="coerce").median()
        fill_values[col] = float(0.0 if pd.isna(median) else median)
    asset_levels = sorted(int(x) for x in pd.unique(train_df["asset_id"]))
    return FeatureV0State(feature_cols=cols, fill_values=fill_values, asset_levels=asset_levels)


def transform_feature_v0(df: pd.DataFrame, state: FeatureV0State) -> np.ndarray:
    feats = []
    for col in state.feature_cols:
        s = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        s = s.fillna(np.float32(state.fill_values[col]))
        s = s.replace([np.inf, -np.inf], np.float32(state.fill_values[col]))
        feats.append(s.to_numpy(dtype=np.float32))
    x = np.column_stack(feats) if feats else np.zeros((len(df), 0), dtype=np.float32)
    asset = df["asset_id"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return np.concatenate([x, asset], axis=1).astype(np.float32, copy=False)


def state_to_jsonable(state: FeatureV0State) -> dict:
    return asdict(state)


def state_from_jsonable(payload: dict) -> FeatureV0State:
    return FeatureV0State(
        feature_cols=list(payload["feature_cols"]),
        fill_values={k: float(v) for k, v in payload["fill_values"].items()},
        asset_levels=[int(x) for x in payload["asset_levels"]],
    )
