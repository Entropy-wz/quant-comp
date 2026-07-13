import numpy as np
from contest.metrics import weighted_zero_mean_r2


def test_all_zero_predictions_score_zero():
    y = np.array([1.0, -2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0])
    pred = np.zeros_like(y)
    assert abs(weighted_zero_mean_r2(y, pred, w) - 0.0) < 1e-12


def test_perfect_predictions_score_one():
    y = np.array([1.0, -2.0, 3.0])
    w = np.array([1.0, 2.0, 0.5])
    assert abs(weighted_zero_mean_r2(y, y, w) - 1.0) < 1e-12


def test_zero_denominator_returns_zero():
    y = np.zeros(3)
    w = np.ones(3)
    pred = np.array([0.1, -0.2, 0.3])
    assert weighted_zero_mean_r2(y, pred, w) == 0.0
