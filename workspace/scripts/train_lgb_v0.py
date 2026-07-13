from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.features_v0 import fit_feature_v0, state_to_jsonable, transform_feature_v0
from contest.io import list_partition_files, load_partitions
from contest.paths import load_paths
from contest.train_lgb import evaluate_booster, train_lightgbm
from contest.validation import holdout_time_split


def main() -> None:
    paths = load_paths()
    cfg = yaml.safe_load((ROOT / "configs" / "train_lgb_v0.yaml").read_text(encoding="utf-8"))
    files = list_partition_files(paths["data_root"], "train")
    max_parts = cfg.get("max_train_partitions")
    if max_parts is not None:
        files = files[-int(max_parts) :]

    df = load_partitions(files)
    train_times, valid_times = holdout_time_split(df["time_id"].to_numpy(), cfg["holdout_fraction"])
    train_df = df[df["time_id"].isin(set(train_times.tolist()))].copy()
    valid_df = df[df["time_id"].isin(set(valid_times.tolist()))].copy()

    state = fit_feature_v0(train_df)
    X_train = transform_feature_v0(train_df, state)
    X_valid = transform_feature_v0(valid_df, state)
    y_train = pd.to_numeric(train_df["target"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    y_valid = pd.to_numeric(valid_df["target"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w_train = pd.to_numeric(train_df["weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w_valid = pd.to_numeric(valid_df["weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)

    booster = train_lightgbm(
        X_train,
        y_train,
        w_train,
        X_valid,
        y_valid,
        w_valid,
        cfg["params"],
        num_threads=paths["num_threads"],
        num_boost_round=int(cfg["num_boost_round"]),
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
    )

    metrics = {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "partitions_used": [str(p.name) for p in files],
        "best_iteration": int(booster.best_iteration or 0),
        "train_wzm_r2": evaluate_booster(booster, X_train, y_train, w_train),
        "valid_wzm_r2": evaluate_booster(booster, X_valid, y_valid, w_valid),
    }

    out_dir = paths["artifacts_dir"] / "lgb_v0"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.txt"
    booster.save_model(str(model_path))
    (out_dir / "feature_state.json").write_text(
        json.dumps(state_to_jsonable(state), indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
