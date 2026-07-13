from __future__ import annotations

import numpy as np

from contest.train_lgb import train_lightgbm


def test_train_lightgbm_smoke():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 5)).astype(np.float32)
    y_train = X_train[:, 0] * 0.2 + rng.normal(scale=0.1, size=200)
    w_train = np.ones(200)
    X_valid = rng.normal(size=(50, 5)).astype(np.float32)
    y_valid = X_valid[:, 0] * 0.2
    w_valid = np.ones(50)
    params = {
        "objective": "regression",
        "learning_rate": 0.1,
        "num_leaves": 8,
        "min_data_in_leaf": 10,
        "verbosity": -1,
    }
    booster = train_lightgbm(
        X_train,
        y_train,
        w_train,
        X_valid,
        y_valid,
        w_valid,
        params,
        num_threads=2,
        num_boost_round=20,
    )
    pred = booster.predict(X_valid)
    assert len(pred) == 50
