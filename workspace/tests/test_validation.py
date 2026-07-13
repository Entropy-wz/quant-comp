import numpy as np
from contest.validation import expanding_folds, holdout_time_split, unique_sorted_time_ids


def test_unique_sorted_time_ids():
    t = np.array([3, 1, 2, 2, 1])
    assert np.array_equal(unique_sorted_time_ids(t), np.array([1, 2, 3]))


def test_holdout_split_is_contiguous_tail():
    times = np.arange(10)
    train_t, valid_t = holdout_time_split(times, holdout_fraction=0.2)
    assert np.array_equal(valid_t, np.array([8, 9]))
    assert np.array_equal(train_t, np.arange(8))


def test_expanding_folds_respect_order_and_embargo():
    times = np.arange(20)
    folds = expanding_folds(times, n_folds=3, embargo=2)
    assert len(folds) == 3
    for train_t, valid_t in folds:
        assert train_t.max() < valid_t.min()
        # embargo gap: at least 2 time ids between train max and valid min when possible
        assert valid_t.min() - train_t.max() >= 2
        assert len(np.intersect1d(train_t, valid_t)) == 0
