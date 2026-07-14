from __future__ import annotations

import gc
from typing import Any, Callable

import numpy as np
import pandas as pd

from contest.train_lgb import evaluate_booster, train_lightgbm
from contest.validation import expanding_folds


FeatureFitFn = Callable[[pd.DataFrame], Any]
FeatureTransformFn = Callable[[pd.DataFrame, Any], np.ndarray]


def build_xyw(
    df: pd.DataFrame,
    state: Any,
    transform_feature_fn: FeatureTransformFn,
    target_col: str = "target",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = transform_feature_fn(df, state)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return x, y, w


def run_expanding_cv(
    df: pd.DataFrame,
    fit_feature_fn: FeatureFitFn,
    transform_feature_fn: FeatureTransformFn,
    params: dict[str, Any],
    n_folds: int = 3,
    embargo: int = 5,
    num_threads: int = 4,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    target_col: str = "target",
) -> dict[str, Any]:
    folds_spec = expanding_folds(df["time_id"].to_numpy(), n_folds=n_folds, embargo=embargo)
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, (train_times, valid_times) in enumerate(folds_spec):
        print(f"  CV fold {fold_idx + 1}/{len(folds_spec)} starting...", flush=True)
        train_mask = df["time_id"].isin(set(train_times.tolist()))
        valid_mask = df["time_id"].isin(set(valid_times.tolist()))
        train_df = df.loc[train_mask]
        valid_df = df.loc[valid_mask]
        if len(train_df) == 0 or len(valid_df) == 0:
            continue

        state = fit_feature_fn(train_df)
        x_train, y_train, w_train = build_xyw(train_df, state, transform_feature_fn, target_col)
        x_valid, y_valid, w_valid = build_xyw(valid_df, state, transform_feature_fn, target_col)
        booster = train_lightgbm(
            x_train,
            y_train,
            w_train,
            x_valid,
            y_valid,
            w_valid,
            params,
            num_threads=num_threads,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        score = evaluate_booster(booster, x_valid, y_valid, w_valid)
        print(
            f"  CV fold {fold_idx + 1}/{len(folds_spec)} done "
            f"train={len(train_df):,} valid={len(valid_df):,} "
            f"best_iter={booster.best_iteration} wzm_r2={score:.6f}",
            flush=True,
        )
        fold_reports.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "best_iteration": int(booster.best_iteration or 0),
                "valid_wzm_r2": float(score),
            }
        )
        del x_train, y_train, w_train, x_valid, y_valid, w_valid, booster, state
        gc.collect()

    if not fold_reports:
        raise RuntimeError("no folds produced scores")

    scores = np.array([f["valid_wzm_r2"] for f in fold_reports], dtype=np.float64)
    return {
        "mean_valid_wzm_r2": float(scores.mean()),
        "std_valid_wzm_r2": float(scores.std(ddof=0)),
        "folds": fold_reports,
    }
