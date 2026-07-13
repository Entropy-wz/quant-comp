import numpy as np
import pandas as pd

from contest.features_v0 import fit_feature_v0, transform_feature_v0
from contest.train_matrix import build_xyw, run_expanding_cv


def _toy_frame(n_times=30, n_assets=3):
    rows = []
    rid = 0
    rng = np.random.default_rng(0)
    for t in range(n_times):
        for a in range(n_assets):
            rows.append(
                {
                    "row_id": rid,
                    "time_id": t,
                    "asset_id": a,
                    "feature_000": float(t + a) + rng.normal(0, 0.01),
                    "feature_001": float(a),
                    "weight": 1.0,
                    "target": float(0.01 * t - 0.02 * a),
                }
            )
            rid += 1
    return pd.DataFrame(rows)


def test_build_xyw_shapes():
    df = _toy_frame(n_times=5)
    state = fit_feature_v0(df)
    x, y, w = build_xyw(df, state, transform_feature_v0)
    assert x.shape[0] == len(df)
    assert y.shape == (len(df),)
    assert w.shape == (len(df),)


def test_run_expanding_cv_returns_mean_score():
    df = _toy_frame()
    result = run_expanding_cv(
        df,
        fit_feature_fn=fit_feature_v0,
        transform_feature_fn=transform_feature_v0,
        params={
            "objective": "regression",
            "learning_rate": 0.1,
            "num_leaves": 8,
            "verbosity": -1,
            "min_data_in_leaf": 5,
        },
        n_folds=3,
        embargo=2,
        num_threads=2,
        num_boost_round=30,
        early_stopping_rounds=10,
    )
    assert "mean_valid_wzm_r2" in result
    assert "std_valid_wzm_r2" in result
    assert "folds" in result
    assert len(result["folds"]) >= 1
    assert np.isfinite(result["mean_valid_wzm_r2"])
