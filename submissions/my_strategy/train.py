"""
LightGBM 训练脚本 v4 — 单模型 + 全量数据 + 内存安全分批训练.

逐分区构建特征(float32), 全局验证集, 增量训练(init_model)扫完全量数据.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM v4 — single model, full data.")
    p.add_argument("--release-root", default="../public_release_20260630/data")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--n-estimators", type=int, default=2000)
    p.add_argument("--early-stopping", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=127)
    p.add_argument("--trees-per-part", type=int, default=250,
                   help="每个分区新增的树数 (增量训练)")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def _manifest_files(release_root: Path, key: str) -> list[Path]:
    mpath = release_root / "manifest.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        files = m.get("files", {}).get(key, [])
        if files:
            return [release_root / str(f) for f in files]
    return sorted((release_root / key).glob("*.parquet"))


# ══════════════════════════════════════════════════════════════════════


def _causal_roll_mean_std(block: np.ndarray, window: int):
    n, k = block.shape
    if n == 0:
        z = np.zeros((0, k), dtype=np.float32)
        return z, z
    block = block.astype(np.float64)
    c1 = np.vstack([np.zeros((1, k)), np.cumsum(block, axis=0)])
    c2 = np.vstack([np.zeros((1, k)), np.cumsum(block ** 2, axis=0)])
    idx = np.arange(n)
    start = np.maximum(0, idx - window)
    count = np.maximum(idx - start, 1)[:, None]
    sum1 = c1[idx] - c1[start]
    sum2 = c2[idx] - c2[start]
    mean = sum1 / count
    var = np.maximum(sum2 / count - mean ** 2, 0)
    std = np.sqrt(var)
    std[count[:, 0] <= 1] = 0.0
    return mean.astype(np.float32), std.astype(np.float32)


def _causal_diff1(block: np.ndarray) -> np.ndarray:
    n, k = block.shape
    diff = np.zeros((n, k), dtype=np.float32)
    if n > 1:
        diff[1:] = block[1:] - block[:-1]
    return diff


def build_features(
    frame: pd.DataFrame,
    feat_cols: list[str],
    roll_cols: list[str],
    windows: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    单分区特征构建.
    返回 (X_float32, asset_ids, time_ids, col_names).
    """
    n_feat = len(feat_cols)
    n_roll = len(roll_cols)
    asset_ids = frame["asset_id"].to_numpy(copy=False)
    time_ids = frame["time_id"].to_numpy(copy=False)
    roll_idx = [feat_cols.index(c) for c in roll_cols]

    order = np.lexsort((time_ids, asset_ids))
    inv_order = np.argsort(order)

    base_feats = frame[feat_cols].to_numpy(dtype=np.float32)[order]
    base_feats = np.nan_to_num(base_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 截面
    cs_zscore = np.zeros_like(base_feats)
    cs_rank = np.zeros_like(base_feats)
    time_bounds = np.flatnonzero(np.r_[True, time_ids[order][1:] != time_ids[order][:-1], True])
    for start, end in zip(time_bounds[:-1], time_bounds[1:]):
        g = base_feats[start:end].astype(np.float64)
        m_ = g.mean(axis=0, keepdims=True)
        s_ = g.std(axis=0, ddof=0, keepdims=True)
        s_[s_ < 1e-8] = 1.0
        cs_zscore[start:end] = ((g - m_) / s_).astype(np.float32)
        r = np.argsort(np.argsort(g, axis=0), axis=0).astype(np.float64)
        cs_rank[start:end] = (r / max(len(g) - 1, 1)).astype(np.float32)

    # 滚动
    roll_mat = base_feats[:, roll_idx]
    ordered_assets = asset_ids[order]
    asset_bounds = np.flatnonzero(np.r_[True, ordered_assets[1:] != ordered_assets[:-1], True])

    parts: list[np.ndarray] = []

    # 1. base + asset_id
    asset_col = ordered_assets.reshape(-1, 1).astype(np.float32)
    parts.append(np.concatenate([base_feats, asset_col], axis=1)[inv_order])

    # 2. 滚动
    for w in windows:
        rm = np.empty((len(order), n_roll), dtype=np.float32)
        rs = np.empty((len(order), n_roll), dtype=np.float32)
        for sa, se in zip(asset_bounds[:-1], asset_bounds[1:]):
            m_, s_ = _causal_roll_mean_std(roll_mat[sa:se], w)
            rm[sa:se] = m_
            rs[sa:se] = s_
        delta = (roll_mat - rm).astype(np.float32)
        parts.extend([rm[inv_order], rs[inv_order], delta[inv_order]])

    # 3. diff1
    d1 = np.empty_like(roll_mat)
    for sa, se in zip(asset_bounds[:-1], asset_bounds[1:]):
        d1[sa:se] = _causal_diff1(roll_mat[sa:se])
    parts.append(d1[inv_order])

    # 4. 截面
    parts.append(cs_zscore[inv_order])
    parts.append(cs_rank[inv_order])

    X = np.concatenate(parts, axis=1).astype(np.float32)

    # 列名
    names = [f"base_{c}" for c in feat_cols] + ["asset_id"]
    for w in windows:
        for c in roll_cols:
            names.append(f"{c}_rm{w}")
        for c in roll_cols:
            names.append(f"{c}_rs{w}")
        for c in roll_cols:
            names.append(f"{c}_d{w}")
    for c in roll_cols:
        names.append(f"{c}_d1")
    for c in feat_cols:
        names.append(f"{c}_z")
    for c in feat_cols:
        names.append(f"{c}_rk")

    return X, asset_ids, time_ids, names


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    denom = np.sum(weight * y_true * y_true)
    if denom <= 0:
        return 0.0
    return float(1.0 - np.sum(weight * (y_true - y_pred) ** 2) / denom)


# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    release_root = Path(args.release_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    files = _manifest_files(release_root, "train")
    feat_cols = [f"feature_{i:03d}" for i in range(323)]
    windows = [5, 15, 50]
    max_roll_cols = 64
    embargo = 10
    all_assets = list(range(15))

    # ════════════════════════════════════════════════════════════
    # 阶段 1: 特征筛选
    # ════════════════════════════════════════════════════════════
    print("=" * 60)
    print("[阶段 1] 特征筛选")
    print("=" * 60)

    s_frames = []
    for path in files:
        df = pd.read_parquet(path)
        s_frames.append(df.sample(frac=0.05, random_state=args.seed))
    s_frame = pd.concat(s_frames, ignore_index=True)
    print(f"  数据: {len(s_frame):,} 行")

    t0 = time.perf_counter()
    s_X, _, _, s_names = build_features(s_frame, feat_cols, feat_cols[:max_roll_cols], [20])
    print(f"  特征: {len(s_names)} 维 ({time.perf_counter()-t0:.1f}s)")

    s_tids_u = sorted(s_frame["time_id"].unique())
    n_v = max(1, int(len(s_tids_u) * 0.15))
    v_tids = set(s_tids_u[-n_v:])
    sv = s_frame["time_id"].isin(v_tids).to_numpy()
    st = ~sv

    Xs = s_X[st].astype(np.float32)
    ys = s_frame.loc[st, "target"].to_numpy(dtype=np.float64)
    ws = s_frame.loc[st, "weight"].to_numpy(dtype=np.float64)
    Xsv = s_X[sv].astype(np.float32)
    ysv = s_frame.loc[sv, "target"].to_numpy(dtype=np.float64)
    wsv = s_frame.loc[sv, "weight"].to_numpy(dtype=np.float64)

    sm = Xs.mean(axis=0).astype(np.float64)
    ss = Xs.std(axis=0).astype(np.float64)
    ss[ss < 1e-8] = 1.0
    Xs = ((Xs - sm) / ss).astype(np.float32)
    Xsv = ((Xsv - sm) / ss).astype(np.float32)

    selector = lgb.LGBMRegressor(
        objective="regression", metric="l2",
        n_estimators=120, learning_rate=0.05, num_leaves=31,
        max_depth=6, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.0, reg_lambda=0.05, verbosity=0,
        random_state=args.seed, n_jobs=4,
    )
    selector.fit(Xs, ys, sample_weight=np.maximum(ws, 0.0),
                 eval_set=[(Xsv, ysv)], eval_sample_weight=[np.maximum(wsv, 0.0)])

    imp = selector.feature_importances_
    # 根据 base 特征重要性选 roll_cols
    base_imp = {}
    for i in range(len(feat_cols)):
        if i < len(imp):
            base_imp[feat_cols[i]] = imp[i]
    sorted_base = sorted(base_imp.items(), key=lambda x: -x[1])
    top_roll = [c for c, _ in sorted_base[:max_roll_cols]]
    del s_frame, s_X, Xs, ys, ws, Xsv, ysv, wsv
    print(f"  Roll cols (Top-{max_roll_cols}): {top_roll[:5]}...")

    # ════════════════════════════════════════════════════════════
    # 阶段 2: 全量训练 (增量 + 内存安全)
    # ════════════════════════════════════════════════════════════
    # 先扫描所有 time_id，确定 embargo 切分点
    all_tids = set()
    for path in files:
        df = pd.read_parquet(path, columns=["time_id"])
        all_tids.update(df["time_id"].unique())
    all_tids_sorted = sorted(all_tids)
    n_valid_times = 20000
    valid_start_idx = max(0, len(all_tids_sorted) - n_valid_times)
    train_end_idx = max(0, valid_start_idx - embargo)
    train_end_tid = all_tids_sorted[train_end_idx]
    valid_times_set = set(all_tids_sorted[valid_start_idx:])

    print(f"\n{'='*60}")
    print(f"[阶段 2] 全量训练 (增量, {len(files)} 分区)")
    print(f"  train ≤ {train_end_tid}, valid ≥ {all_tids_sorted[valid_start_idx]}")
    print(f"{'='*60}")

    # ── 第一遍: 收集全局标准化统计量 + 验证集 ──
    print("\n[第一遍] 收集标准化统计量 + 验证集...")
    feat_sum = np.zeros(1610, dtype=np.float64)
    feat_sum2 = np.zeros(1610, dtype=np.float64)
    n_train_global = 0
    val_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (X, y, w)

    for fi, path in enumerate(files):
        t0 = time.perf_counter()
        df = pd.read_parquet(path)
        if args.sample_frac < 1.0:
            df = df.sample(frac=args.sample_frac, random_state=args.seed)

        X_part, aid_part, tid_part, all_names = build_features(df, feat_cols, top_roll, windows)

        is_train = (tid_part <= train_end_tid) & ~np.isin(tid_part, list(valid_times_set))
        is_valid = np.isin(tid_part, list(valid_times_set))

        # 标准化统计量 (训练集)
        if is_train.sum() > 0:
            Xt = X_part[is_train].astype(np.float64)
            feat_sum += Xt.sum(axis=0)
            feat_sum2 += (Xt ** 2).sum(axis=0)
            n_train_global += is_train.sum()

        # 收集验证集
        if is_valid.sum() > 0:
            val_parts.append((
                X_part[is_valid].astype(np.float32),
                df.loc[is_valid, "target"].to_numpy(dtype=np.float64),
                df.loc[is_valid, "weight"].to_numpy(dtype=np.float64),
            ))

        del df, X_part, aid_part, tid_part
        print(f"  [{fi+1}/{len(files)}] {path.name} ({time.perf_counter()-t0:.1f}s)", flush=True)

    # 全局标准化参数
    feat_mean = feat_sum / n_train_global
    feat_var = feat_sum2 / n_train_global - feat_mean ** 2
    feat_std = np.sqrt(np.maximum(feat_var, 0))
    feat_std[feat_std < 1e-8] = 1.0
    n_features = len(feat_mean)
    print(f"  标准化完成, n_features={n_features}, train_rows={n_train_global:,}")

    # 合并验证集
    X_val_all = np.concatenate([v[0] for v in val_parts], axis=0)
    y_val_all = np.concatenate([v[1] for v in val_parts])
    w_val_all = np.concatenate([v[2] for v in val_parts])
    X_val_all = ((X_val_all - feat_mean) / feat_std).astype(np.float32)
    dvalid = lgb.Dataset(X_val_all, label=y_val_all, weight=np.maximum(w_val_all, 0.0),
                         free_raw_data=False)
    print(f"  验证集: {len(y_val_all):,} 行")

    # ── 第二遍: 增量训练 ──
    print(f"\n[第二遍] 增量训练 (每分区 +{args.trees_per_part} 棵树)...")
    params = {
        "objective": "regression", "metric": "l2", "boosting_type": "gbdt",
        "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
        "max_depth": 9, "min_child_samples": 20, "min_child_weight": 1e-3,
        "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.7,
        "reg_alpha": 0.0, "reg_lambda": 0.01,
        "verbosity": 0, "num_threads": 4, "seed": args.seed,
    }

    booster = None
    total_train = 0
    for fi, path in enumerate(files):
        t0 = time.perf_counter()
        df = pd.read_parquet(path)
        if args.sample_frac < 1.0:
            df = df.sample(frac=args.sample_frac, random_state=args.seed)

        X_part, aid_part, tid_part, _ = build_features(df, feat_cols, top_roll, windows)
        is_train = (tid_part <= train_end_tid) & ~np.isin(tid_part, list(valid_times_set))

        if is_train.sum() == 0:
            del df, X_part, aid_part, tid_part
            continue

        X_tr = X_part[is_train].astype(np.float32)
        y_tr = df.loc[is_train, "target"].to_numpy(dtype=np.float64)
        w_tr = df.loc[is_train, "weight"].to_numpy(dtype=np.float64)
        del df, X_part, aid_part, tid_part

        # 标准化
        X_tr = ((X_tr - feat_mean) / feat_std).astype(np.float32)

        # 过滤低权重
        w_min = float(np.percentile(w_tr[w_tr > 0], 5)) if (w_tr > 0).any() else 0.0
        keep = w_tr >= w_min
        X_tr, y_tr, w_tr = X_tr[keep], y_tr[keep], w_tr[keep]
        total_train += len(y_tr)

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=np.maximum(w_tr, 0.0))

        booster = lgb.train(
            params, dtrain,
            num_boost_round=args.trees_per_part,
            valid_sets=[dvalid], valid_names=["valid"],
            init_model=booster,
            callbacks=[lgb.log_evaluation(period=50)],
        )

        del X_tr, y_tr, w_tr, dtrain
        print(f"  [{fi+1}/{len(files)}] {path.name} → "
              f"总树数={booster.num_trees()} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ── 最终评估 ──
    pred_val = booster.predict(X_val_all)
    val_r2 = weighted_zero_mean_r2(y_val_all, pred_val, w_val_all)
    print(f"\n{'='*60}")
    print(f"[结果] 总树数={booster.num_trees()}, 训练行数={total_train:,}")
    print(f"  验证 R²: {val_r2:.6f}")

    # 特征重要性
    imp_f = booster.feature_importance()
    top_idx = np.argsort(imp_f)[-15:][::-1]
    print(f"\n[特征重要性 Top-15]")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {all_names[idx]:45s} = {imp_f[idx]:.0f}")

    # ── 保存 ──
    model_text = booster.model_to_string()
    (model_dir / "lgb_model.txt").write_text(model_text, encoding="utf-8")

    state = {
        "feature_cols": feat_cols,
        "roll_feature_cols": top_roll,
        "windows": windows,
        "window": max(windows),
        "asset_levels": all_assets,
        "fill_values": {c: 0.0 for c in feat_cols},
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "val_r2": float(val_r2),
        "n_trees": int(booster.num_trees()),
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
    }
    (model_dir / "config.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n模型已保存至 {model_dir}")


if __name__ == "__main__":
    main()
