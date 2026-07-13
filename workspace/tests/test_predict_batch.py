import numpy as np
import pandas as pd
from pathlib import Path

from contest.predict_batch import write_public_submission


def test_write_public_submission_schema(tmp_path: Path):
    path = tmp_path / "sub.csv"
    write_public_submission(
        row_ids=np.array([10, 11], dtype=np.int64),
        preds=np.array([0.1, -0.2], dtype=np.float64),
        path=path,
    )
    df = pd.read_csv(path)
    assert list(df.columns) == ["row_id", "target"]
    assert len(df) == 2
    assert np.isfinite(df["target"].to_numpy()).all()


def test_write_public_submission_sanitizes_non_finite(tmp_path: Path):
    path = tmp_path / "sub.csv"
    write_public_submission(
        row_ids=np.array([1], dtype=np.int64),
        preds=np.array([np.nan], dtype=np.float64),
        path=path,
    )
    df = pd.read_csv(path)
    assert df["target"].iloc[0] == 0.0
