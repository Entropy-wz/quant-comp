import numpy as np
import pandas as pd
from contest.features_v0 import fit_feature_v0, transform_feature_v0


def test_fit_transform_fills_nan_and_appends_asset():
    train = pd.DataFrame(
        {
            "asset_id": [0, 1],
            "feature_000": [1.0, np.nan],
            "feature_001": [np.nan, 4.0],
        }
    )
    state = fit_feature_v0(train)
    X = transform_feature_v0(train, state)
    assert X.shape == (2, 3)
    assert np.isfinite(X).all()
    assert X.dtype == np.float32
