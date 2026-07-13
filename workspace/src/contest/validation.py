from __future__ import annotations

import numpy as np


def unique_sorted_time_ids(time_ids: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(time_ids))


def holdout_time_split(
    time_ids: np.ndarray,
    holdout_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    times = unique_sorted_time_ids(time_ids)
    if len(times) < 2:
        raise ValueError("need at least 2 distinct time_id values")
    n_valid = max(1, int(round(len(times) * holdout_fraction)))
    n_valid = min(n_valid, len(times) - 1)
    valid = times[-n_valid:]
    train = times[:-n_valid]
    return train, valid


def expanding_folds(
    time_ids: np.ndarray,
    n_folds: int = 3,
    embargo: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    times = unique_sorted_time_ids(time_ids)
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if len(times) < n_folds + 2:
        raise ValueError("not enough time_ids for requested folds")

    # Reserve last 20% conceptually by splitting the earlier region into fold valids
    # Use equal-sized validation blocks walking forward over times excluding a final buffer.
    n = len(times)
    # Keep at least 1 time for initial train; split remaining into n_folds valid blocks
    min_train = max(1, n // (n_folds + 2))
    usable = times[min_train:]
    block = len(usable) // n_folds
    if block < 1:
        raise ValueError("validation block too small")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        start = i * block
        end = (i + 1) * block if i < n_folds - 1 else len(usable)
        valid = usable[start:end]
        # train = all times before valid start, minus embargo window
        valid_start_idx = int(np.searchsorted(times, valid[0], side="left"))
        train_end_idx = max(0, valid_start_idx - embargo)
        train = times[:train_end_idx]
        if len(train) == 0 or len(valid) == 0:
            continue
        if train.max() >= valid.min():
            raise RuntimeError("fold ordering violated")
        folds.append((train, valid))
    if not folds:
        raise RuntimeError("no valid folds constructed")
    return folds
