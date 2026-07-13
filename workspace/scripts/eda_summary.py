from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.io import feature_columns, list_partition_files, load_partitions, responder_columns
from contest.paths import load_paths


def main() -> None:
    paths = load_paths()
    files = list_partition_files(paths["data_root"], "train")[:1]
    df = load_partitions(files)
    summary = {
        "partition": files[0].name,
        "rows": int(len(df)),
        "n_assets": int(df["asset_id"].nunique()),
        "n_time_ids": int(df["time_id"].nunique()),
        "n_features": len(feature_columns(df)),
        "n_responders": len(responder_columns(df)),
        "target_mean": float(pd.to_numeric(df["target"], errors="coerce").mean()),
        "target_std": float(pd.to_numeric(df["target"], errors="coerce").std()),
        "weight_mean": float(pd.to_numeric(df["weight"], errors="coerce").mean()),
        "feature_nan_frac_mean": float(
            df[feature_columns(df)].isna().mean().mean()
        ),
    }
    out = paths["experiments_dir"] / "eda_summary_partition0.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
