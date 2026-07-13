from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def list_partition_files(data_root: Path, split: str) -> list[Path]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rels = manifest.get("files", {}).get(split, [])
        if rels:
            return [data_root / rel for rel in rels]
    pattern = "train_partition_*.parquet" if split == "train" else "test_partition_*.parquet"
    return sorted((data_root / split).glob(pattern))


def load_partitions(files: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    if not files:
        raise ValueError("no parquet files provided")
    frames = [pd.read_parquet(path, columns=columns) for path in files]
    return pd.concat(frames, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith("feature_"))


def responder_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith("responder_"))
