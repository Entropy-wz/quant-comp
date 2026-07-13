import numpy as np
import pandas as pd

from contest.responders import rank_responders_by_target_corr, weighted_corr


def test_weighted_corr_perfect():
    y = np.linspace(-1, 1, 50)
    w = np.ones(50)
    assert abs(weighted_corr(y, y, w) - 1.0) < 1e-9


def test_rank_responders_orders_by_abs_corr():
    n = 200
    w = np.ones(n)
    target = np.linspace(-1, 1, n)
    df = pd.DataFrame(
        {
            "target": target,
            "weight": w,
            "responder_00": target + np.random.default_rng(0).normal(0, 0.01, n),
            "responder_01": np.random.default_rng(1).normal(0, 1, n),
        }
    )
    ranked = rank_responders_by_target_corr(df, top_k=2)
    assert ranked[0][0] == "responder_00"
    assert abs(ranked[0][1]) > abs(ranked[1][1])
