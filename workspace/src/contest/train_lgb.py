from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from contest.metrics import weighted_zero_mean_r2


def _early_stopping_callback(early_stopping_rounds: int) -> Any:
    try:
        return lgb.early_stopping(early_stopping_rounds, verbose=False)
    except TypeError:
        return lgb.early_stopping(early_stopping_rounds)


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    params: dict[str, Any],
    num_threads: int = 4,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train, weight=w_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, reference=train_set)
    model_params = dict(params)
    model_params["num_threads"] = int(num_threads)
    booster = lgb.train(
        model_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            _early_stopping_callback(early_stopping_rounds),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster


def evaluate_booster(
    booster: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> float:
    pred = booster.predict(X)
    return weighted_zero_mean_r2(y, pred, w)
