from __future__ import annotations

import numpy as np

from contest.metrics import weighted_zero_mean_r2
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


def test_train_lightgbm_uses_weighted_zero_mean_r2_for_selection(monkeypatch):
    X_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_train = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    w_train = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    X_valid = np.array([[3.0], [4.0]], dtype=np.float32)
    y_valid = np.array([3.0, 4.0], dtype=np.float64)
    w_valid = np.array([1.0, 3.0], dtype=np.float64)
    predictions = np.array([2.5, 4.5], dtype=np.float64)
    captured = {}

    def fake_train(
        params,
        train_set,
        num_boost_round,
        valid_sets,
        valid_names,
        callbacks,
        feval,
    ):
        captured["metric"] = params.get("metric")
        captured["feval_result"] = feval(predictions, valid_sets[0])
        return object()

    monkeypatch.setattr("contest.train_lgb.lgb.train", fake_train)

    train_lightgbm(
        X_train,
        y_train,
        w_train,
        X_valid,
        y_valid,
        w_valid,
        {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
        },
        num_threads=2,
        num_boost_round=5,
    )

    name, value, is_higher_better = captured["feval_result"]
    assert captured["metric"] == "None"
    assert name == "wzm_r2"
    assert value == weighted_zero_mean_r2(y_valid, predictions, w_valid)
    assert is_higher_better is True
