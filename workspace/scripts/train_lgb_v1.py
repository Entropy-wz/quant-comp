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
from contest.train_matrix import build_xyw, run_expanding_cv
from contest.validation import holdout_time_split


def _schema_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def _load_train(cfg: dict, data_root: Path) -> tuple[pd.DataFrame, list[str]]:
    files = list_partition_files(data_root, "train")
    max_parts = cfg.get("max_train_partitions")
    if max_parts is not None:
        files = files[-int(max_parts) :]
    names = _schema_columns(files[0])
    feat_cols = sorted(c for c in names if c.startswith("feature_"))
    cols = ["row_id", "time_id", "asset_id", "weight", "target", *feat_cols]
    if cfg.get("auxiliary_mode", "none") != "none":
        cols.extend(sorted(c for c in names if c.startswith("responder_")))
    ordered = [c for c in cols if c in names]
    return load_partitions(files, columns=ordered), [str(p.name) for p in files]


def main() -> None:
    paths = load_paths()
    cfg = yaml.safe_load((ROOT / "configs" / "train_lgb_v1.yaml").read_text(encoding="utf-8"))
    feature_version = cfg.get("feature_version", "v0")
    if feature_version != "v0":
        raise NotImplementedError(f"feature_version={feature_version} not wired yet; use Task 5")

    df, part_names = _load_train(cfg, paths["data_root"])
    train_times, holdout_times = holdout_time_split(
        df["time_id"].to_numpy(), float(cfg["holdout_fraction"])
    )
    pre = df[df["time_id"].isin(set(train_times.tolist()))].copy()
    holdout = df[df["time_id"].isin(set(holdout_times.tolist()))].copy()

    cv = run_expanding_cv(
        pre,
        fit_feature_fn=fit_feature_v0,
        transform_feature_fn=transform_feature_v0,
        params=cfg["params"],
        n_folds=int(cfg["n_folds"]),
        embargo=int(cfg["embargo"]),
        num_threads=paths["num_threads"],
        num_boost_round=int(cfg["num_boost_round"]),
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
    )

    best_iters = [f["best_iteration"] for f in cv["folds"] if f["best_iteration"] > 0]
    refit_rounds = int(max(best_iters)) if best_iters else int(cfg["num_boost_round"])
    # Use last fold's valid slice from holdout for early stopping during final fit:
    # split pre again with a small tail as internal valid for early stop, then evaluate holdout.
    inner_train_times, inner_valid_times = holdout_time_split(pre["time_id"].to_numpy(), 0.15)
    inner_train = pre[pre["time_id"].isin(set(inner_train_times.tolist()))]
    inner_valid = pre[pre["time_id"].isin(set(inner_valid_times.tolist()))]

    state = fit_feature_v0(inner_train)
    x_tr, y_tr, w_tr = build_xyw(inner_train, state, transform_feature_v0)
    x_va, y_va, w_va = build_xyw(inner_valid, state, transform_feature_v0)
    booster = train_lightgbm(
        x_tr,
        y_tr,
        w_tr,
        x_va,
        y_va,
        w_va,
        cfg["params"],
        num_threads=paths["num_threads"],
        num_boost_round=max(refit_rounds, 50),
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
    )

    # Refit fill state on all pre-holdout for export consistency
    state = fit_feature_v0(pre)
    x_hold, y_hold, w_hold = build_xyw(holdout, state, transform_feature_v0)
    # Note: booster was fit with inner_train fill; re-transform holdout with pre-fit state
    # For V0 medians, refitting state on `pre` then predicting is standard; retrain quickly:
    x_tr2, y_tr2, w_tr2 = build_xyw(inner_train, state, transform_feature_v0)
    x_va2, y_va2, w_va2 = build_xyw(inner_valid, state, transform_feature_v0)
    booster = train_lightgbm(
        x_tr2,
        y_tr2,
        w_tr2,
        x_va2,
        y_va2,
        w_va2,
        cfg["params"],
        num_threads=paths["num_threads"],
        num_boost_round=max(int(booster.best_iteration or refit_rounds), 50),
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
    )
    holdout_score = evaluate_booster(booster, x_hold, y_hold, w_hold)

    out_dir = paths["artifacts_dir"] / "lgb_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"))
    (out_dir / "feature_state.json").write_text(
        json.dumps(state_to_jsonable(state), indent=2), encoding="utf-8"
    )
    metrics = {
        "feature_version": feature_version,
        "partitions_used": part_names,
        "train_pre_rows": int(len(pre)),
        "holdout_rows": int(len(holdout)),
        "cv": cv,
        "best_iteration": int(booster.best_iteration or 0),
        "holdout_wzm_r2": float(holdout_score),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "cv_report.json").write_text(json.dumps(cv, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("partitions_used", "holdout_wzm_r2", "best_iteration")}, indent=2))
    print("cv_mean", cv["mean_valid_wzm_r2"], "cv_std", cv["std_valid_wzm_r2"])


if __name__ == "__main__":
    main()
