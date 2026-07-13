from __future__ import annotations

from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from contest.io import list_partition_files

TransformFn = Callable[[pd.DataFrame], np.ndarray]


def write_public_submission(row_ids: np.ndarray, preds: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = np.nan_to_num(np.asarray(preds, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    out = pd.DataFrame({"row_id": np.asarray(row_ids, dtype=np.int64), "target": preds})
    out.to_csv(path, index=False)


def iter_test_files(data_root: Path) -> list[Path]:
    return list_partition_files(data_root, "test")


def predict_test_partition(
    path: Path,
    columns: list[str],
    transform: TransformFn,
    booster: lgb.Booster,
    num_threads: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path, columns=columns)
    x = transform(df)
    pred = booster.predict(x, num_threads=num_threads)
    return df["row_id"].to_numpy(dtype=np.int64), np.asarray(pred, dtype=np.float64)


def predict_test_partition_chunked(
    path: Path,
    columns: list[str],
    transform: TransformFn,
    booster: lgb.Booster,
    num_threads: int = 4,
    column_chunk_width: int = 40,
    time_id_batch: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-safer path: filter by time_id batches using column-chunked Arrow reads."""
    pf = pq.ParquetFile(path)
    time_col = pf.read(columns=["time_id"]).column("time_id").to_numpy()
    unique_times = np.unique(time_col)
    row_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []

    meta_cols = [c for c in ("row_id", "time_id", "asset_id") if c in columns]
    feat_cols = [c for c in columns if c not in meta_cols]

    for start in range(0, len(unique_times), time_id_batch):
        batch_times = set(unique_times[start : start + time_id_batch].tolist())
        mask = np.isin(time_col, list(batch_times))
        if not mask.any():
            continue

        pieces: dict[str, np.ndarray] = {}
        for c in meta_cols:
            arr = pf.read(columns=[c]).column(0).to_numpy()
            pieces[c] = arr[mask]
            del arr

        for i in range(0, len(feat_cols), column_chunk_width):
            chunk = feat_cols[i : i + column_chunk_width]
            table = pf.read(columns=chunk)
            for name in chunk:
                col = table.column(name).to_numpy()
                pieces[name] = col[mask]
                del col
            del table

        df = pd.DataFrame(pieces)
        x = transform(df)
        pred = booster.predict(x, num_threads=num_threads)
        row_parts.append(df["row_id"].to_numpy(dtype=np.int64))
        pred_parts.append(np.asarray(pred, dtype=np.float64))
        del df, x, pred, pieces

    if not row_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    return np.concatenate(row_parts), np.concatenate(pred_parts)
