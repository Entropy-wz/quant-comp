from pathlib import Path

import pandas as pd
from contest.io import feature_columns, list_partition_files, load_partitions
from contest.paths import load_paths


def test_list_train_partitions_from_manifest():
    paths = load_paths()
    files = list_partition_files(paths["data_root"], "train")
    assert len(files) == 9
    assert all(p.exists() for p in files)


def test_load_single_partition_has_required_columns():
    paths = load_paths()
    files = list_partition_files(paths["data_root"], "train")
    # Read only one partition and a small column set for speed/memory
    cols = ["row_id", "time_id", "asset_id", "weight", "target", "feature_000"]
    df = load_partitions([files[0]], columns=cols)
    assert isinstance(df, pd.DataFrame)
    assert set(cols).issubset(df.columns)
    assert len(df) > 0


def test_feature_columns_sorted():
    df = pd.DataFrame(
        {
            "feature_001": [1.0],
            "feature_000": [2.0],
            "target": [0.0],
        }
    )
    assert feature_columns(df) == ["feature_000", "feature_001"]
