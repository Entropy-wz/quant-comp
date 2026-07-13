import numpy as np
import pandas as pd

from contest.features_v0 import FeatureV0State, transform_feature_v0
from contest.features_v1 import RollingState, fit_feature_v1, transform_feature_v1_frame


def _panel(n_times=8, n_assets=3):
    rows = []
    rid = 0
    rng = np.random.default_rng(1)
    for t in range(n_times):
        for a in range(n_assets):
            rows.append(
                {
                    "row_id": rid,
                    "time_id": t,
                    "asset_id": a,
                    "feature_000": float(t + 0.1 * a) + rng.normal(0, 0.01),
                    "feature_001": float(a - t),
                    "feature_002": rng.normal(),
                }
            )
            rid += 1
    return pd.DataFrame(rows)


def test_offline_matches_online_rolling():
    df = _panel()
    state = fit_feature_v1(df, window=3, roll_cols=["feature_000", "feature_001"])
    offline = transform_feature_v1_frame(df, state)

    online_rows = []
    roller = RollingState(state)
    for time_id, chunk in df.groupby("time_id", sort=True):
        base = transform_feature_v0(
            chunk,
            FeatureV0State(state.feature_cols, state.fill_values, state.asset_levels),
        )
        online_rows.append(
            roller.update_and_featurize(
                int(time_id),
                chunk["asset_id"].to_numpy(),
                base,
            )
        )
    online = np.vstack(online_rows)
    # offline is in original df order; online concatenated in time order which matches df order here
    assert offline.shape == online.shape
    assert np.allclose(offline, online, atol=1e-5, rtol=1e-5)
