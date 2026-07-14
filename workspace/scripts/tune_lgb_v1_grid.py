from __future__ import annotations

import gc
import itertools
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.features_v1 import fit_feature_v1, state_to_jsonable, transform_feature_v1_frame
from contest.paths import load_paths
from contest.train_lgb import evaluate_booster, train_lightgbm
from contest.train_matrix import build_xyw
from contest.validation import expanding_folds, holdout_time_split

# Reuse loaders from train script
sys.path.insert(0, str(ROOT / "scripts"))
from train_lgb_v1 import _load_train  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _topk_indices(importance: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), len(importance))
    return np.argsort(importance)[::-1][:k].astype(np.int64)


def main() -> None:
    paths = load_paths()
    cfg = yaml.safe_load((ROOT / "configs" / "train_lgb_v1.yaml").read_text(encoding="utf-8"))
    # Force LB-like data regime
    cfg["max_train_partitions"] = 4
    cfg["auxiliary_mode"] = "none"
    windows = [int(w) for w in (cfg.get("windows") or [cfg.get("window", 5)])]
    max_roll = int(cfg.get("max_roll_cols", 32))

    grid_lr = [0.02, 0.03, 0.05]
    grid_leaves = [48, 64, 96]
    grid_topk = [256, 384, 548]  # 548 = keep all

    _log("[tune] loading data (4 partitions)...")
    df, part_names = _load_train(cfg, paths["data_root"], paths["experiments_dir"])
    train_times, holdout_times = holdout_time_split(df["time_id"].to_numpy(), float(cfg["holdout_fraction"]))
    pre = df[df["time_id"].isin(set(train_times.tolist()))].copy()
    holdout = df[df["time_id"].isin(set(holdout_times.tolist()))].copy()
    del df
    gc.collect()
    _log(f"[tune] pre={len(pre):,} holdout={len(holdout):,} parts={part_names}")

    model_path = paths["artifacts_dir"] / "lgb_v1" / "model.txt"
    if model_path.exists():
        booster0 = lgb.Booster(model_file=str(model_path))
        importance = booster0.feature_importance(importance_type="gain").astype(np.float64)
        _log(f"[tune] importance from existing model n={len(importance)}")
        del booster0
    else:
        importance = None

    folds = expanding_folds(pre["time_id"].to_numpy(), n_folds=int(cfg["n_folds"]), embargo=int(cfg["embargo"]))
    configs = list(itertools.product(grid_topk, grid_lr, grid_leaves))
    fold_scores: dict[tuple, list[float]] = {c: [] for c in configs}
    fold_best_iters: dict[tuple, list[int]] = {c: [] for c in configs}

    base_params = dict(cfg["params"])
    num_boost_round = int(cfg.get("num_boost_round", 600))
    early_stopping_rounds = int(cfg.get("early_stopping_rounds", 50))
    num_threads = paths["num_threads"]

    for fold_idx, (tr_times, va_times) in enumerate(folds):
        _log(f"[tune] building fold {fold_idx + 1}/{len(folds)} matrices...")
        train_df = pre[pre["time_id"].isin(set(tr_times.tolist()))]
        valid_df = pre[pre["time_id"].isin(set(va_times.tolist()))]
        state = fit_feature_v1(train_df, windows=windows, max_roll_cols=max_roll)
        x_tr, y_tr, w_tr = build_xyw(train_df, state, transform_feature_v1_frame)
        x_va, y_va, w_va = build_xyw(valid_df, state, transform_feature_v1_frame)
        n_feat = x_tr.shape[1]
        _log(f"[tune] fold {fold_idx + 1}: train={len(train_df):,} valid={len(valid_df):,} n_feat={n_feat}")

        if importance is None or len(importance) != n_feat:
            _log("[tune] computing fold importance with quick model...")
            quick = train_lightgbm(
                x_tr, y_tr, w_tr, x_va, y_va, w_va,
                {**base_params, "learning_rate": 0.05, "num_leaves": 64},
                num_threads=num_threads,
                num_boost_round=200,
                early_stopping_rounds=30,
            )
            importance = quick.feature_importance(importance_type="gain").astype(np.float64)
            del quick

        for top_k, lr, leaves in configs:
            idx = _topk_indices(importance, top_k if top_k < n_feat else n_feat)
            params = {**base_params, "learning_rate": float(lr), "num_leaves": int(leaves)}
            booster = train_lightgbm(
                x_tr[:, idx], y_tr, w_tr, x_va[:, idx], y_va, w_va,
                params,
                num_threads=num_threads,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            score = evaluate_booster(booster, x_va[:, idx], y_va, w_va)
            key = (top_k, lr, leaves)
            fold_scores[key].append(float(score))
            fold_best_iters[key].append(int(booster.best_iteration or 0))
            _log(
                f"  topk={top_k} lr={lr} leaves={leaves} "
                f"iter={booster.best_iteration} wzm_r2={score:.6f}"
            )
            del booster

        del x_tr, y_tr, w_tr, x_va, y_va, w_va, train_df, valid_df, state
        gc.collect()

    ranking = []
    for key, scores in fold_scores.items():
        arr = np.asarray(scores, dtype=np.float64)
        ranking.append(
            {
                "top_k": key[0],
                "learning_rate": key[1],
                "num_leaves": key[2],
                "mean_valid_wzm_r2": float(arr.mean()),
                "std_valid_wzm_r2": float(arr.std(ddof=0)),
                "fold_scores": [float(x) for x in arr],
                "mean_best_iteration": float(np.mean(fold_best_iters[key])),
            }
        )
    ranking.sort(key=lambda r: (r["mean_valid_wzm_r2"], -r["std_valid_wzm_r2"]), reverse=True)
    best = ranking[0]
    _log("[tune] best config: " + json.dumps(best, indent=2))

    # Final refit with best config
    _log("[tune] final refit on pre/holdout...")
    inner_tr_t, inner_va_t = holdout_time_split(pre["time_id"].to_numpy(), 0.15)
    inner_train = pre[pre["time_id"].isin(set(inner_tr_t.tolist()))].copy()
    inner_valid = pre[pre["time_id"].isin(set(inner_va_t.tolist()))].copy()
    state = fit_feature_v1(pre, windows=windows, max_roll_cols=max_roll)
    del pre
    gc.collect()

    # Refresh importance on final feature space if needed
    x_tr, y_tr, w_tr = build_xyw(inner_train, state, transform_feature_v1_frame)
    del inner_train
    gc.collect()
    x_va, y_va, w_va = build_xyw(inner_valid, state, transform_feature_v1_frame)
    del inner_valid
    gc.collect()

    if len(importance) != x_tr.shape[1]:
        quick = train_lightgbm(
            x_tr, y_tr, w_tr, x_va, y_va, w_va,
            {**base_params, "learning_rate": 0.05, "num_leaves": 64},
            num_threads=num_threads,
            num_boost_round=200,
            early_stopping_rounds=30,
        )
        importance = quick.feature_importance(importance_type="gain").astype(np.float64)
        del quick

    top_k = int(best["top_k"])
    idx = _topk_indices(importance, top_k if top_k < x_tr.shape[1] else x_tr.shape[1])
    params = {
        **base_params,
        "learning_rate": float(best["learning_rate"]),
        "num_leaves": int(best["num_leaves"]),
    }
    rounds = max(int(best["mean_best_iteration"]), 50)
    booster = train_lightgbm(
        x_tr[:, idx], y_tr, w_tr, x_va[:, idx], y_va, w_va,
        params,
        num_threads=num_threads,
        num_boost_round=max(rounds, num_boost_round),
        early_stopping_rounds=early_stopping_rounds,
    )
    del x_tr, y_tr, w_tr, x_va, y_va, w_va
    gc.collect()

    x_hold, y_hold, w_hold = build_xyw(holdout, state, transform_feature_v1_frame)
    del holdout
    gc.collect()
    holdout_score = evaluate_booster(booster, x_hold[:, idx], y_hold, w_hold)
    _log(f"[tune] holdout_wzm_r2={holdout_score:.6f} best_iteration={booster.best_iteration}")

    out_dir = paths["artifacts_dir"] / "lgb_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"))
    payload = state_to_jsonable(state)
    payload["selected_indices"] = idx.tolist()
    payload["selected_top_k"] = int(len(idx))
    (out_dir / "feature_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = {
        "partitions_used": part_names,
        "grid": {"learning_rate": grid_lr, "num_leaves": grid_leaves, "top_k": grid_topk},
        "ranking_top10": ranking[:10],
        "best": best,
        "holdout_wzm_r2": float(holdout_score),
        "best_iteration": int(booster.best_iteration or 0),
        "n_features_model": int(len(idx)),
    }
    (paths["experiments_dir"] / "tune_lgb_v1_grid.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "feature_version": "v1",
                "partitions_used": part_names,
                "holdout_wzm_r2": float(holdout_score),
                "best_iteration": int(booster.best_iteration or 0),
                "tune": best,
                "blend": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # clean blend leftovers
    for name in ("aux_model.txt", "blend_weight.json"):
        p = out_dir / name
        if p.exists():
            p.unlink()
    _log("[tune] DONE wrote experiments/tune_lgb_v1_grid.json and artifacts/lgb_v1/")


if __name__ == "__main__":
    main()
