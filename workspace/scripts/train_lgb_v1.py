from __future__ import annotations

import json
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.features_v0 import fit_feature_v0, state_to_jsonable as state_to_jsonable_v0, transform_feature_v0
from contest.features_v1 import fit_feature_v1, state_to_jsonable as state_to_jsonable_v1, transform_feature_v1_frame
from contest.io import list_partition_files, load_partitions
from contest.paths import load_paths
from contest.train_lgb import evaluate_booster, train_lightgbm
from contest.train_matrix import build_xyw, run_expanding_cv
from contest.validation import holdout_time_split


def _feature_fns(cfg: dict):
    version = cfg.get("feature_version", "v0")
    if version == "v0":
        return fit_feature_v0, transform_feature_v0, state_to_jsonable_v0
    if version == "v1":
        window = int(cfg.get("window", 5))
        windows = cfg.get("windows")
        if windows is None:
            windows = [window]
        else:
            windows = [int(w) for w in windows]
        max_roll = int(cfg.get("max_roll_cols", 32))

        def fit_fn(df):
            return fit_feature_v1(df, windows=windows, max_roll_cols=max_roll)

        return fit_fn, transform_feature_v1_frame, state_to_jsonable_v1
    raise NotImplementedError(f"feature_version={version}")


def _schema_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def _resolve_aux_responders(cfg: dict, names: list[str], experiments_dir: Path) -> list[str]:
    if cfg.get("auxiliary_mode", "none") == "none":
        return []
    configured = list(cfg.get("auxiliary_responders") or [])
    if configured:
        return [c for c in configured if c in names]
    corr_path = experiments_dir / "responder_corr.json"
    if corr_path.exists():
        ranked = json.loads(corr_path.read_text(encoding="utf-8")).get("ranked", [])
        for item in ranked:
            name = item.get("responder")
            if name in names:
                return [name]
    # fallback: first responder column
    for c in names:
        if c.startswith("responder_"):
            return [c]
    return []


def _load_train(cfg: dict, data_root: Path, experiments_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    files = list_partition_files(data_root, "train")
    max_parts = cfg.get("max_train_partitions")
    if max_parts is not None:
        files = files[-int(max_parts) :]
    names = _schema_columns(files[0])
    feat_cols = sorted(c for c in names if c.startswith("feature_"))
    cols = ["row_id", "time_id", "asset_id", "weight", "target", *feat_cols]
    aux_cols = _resolve_aux_responders(cfg, names, experiments_dir)
    if aux_cols:
        print(f"  loading aux responders only: {aux_cols}", flush=True)
        cols.extend(aux_cols)
    ordered = [c for c in cols if c in names]
    frames = []
    for i, path in enumerate(files, start=1):
        print(f"  reading {path.name} ({i}/{len(files)}) ...", flush=True)
        frames.append(pd.read_parquet(path, columns=ordered))
        print(f"  read {path.name}: {len(frames[-1]):,} rows", flush=True)
    return pd.concat(frames, ignore_index=True), [str(p.name) for p in files]


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    paths = load_paths()
    cfg = yaml.safe_load((ROOT / "configs" / "train_lgb_v1.yaml").read_text(encoding="utf-8"))
    fit_feature_fn, transform_feature_fn, state_to_jsonable = _feature_fns(cfg)
    feature_version = cfg.get("feature_version", "v0")
    _log(
        f"[1/6] config loaded: feature_version={feature_version} "
        f"max_train_partitions={cfg.get('max_train_partitions')} "
        f"auxiliary_mode={cfg.get('auxiliary_mode', 'none')}"
    )

    _log("[2/6] loading train parquet (this can take several minutes)...")
    df, part_names = _load_train(cfg, paths["data_root"], paths["experiments_dir"])
    _log(f"[2/6] loaded rows={len(df):,} partitions={part_names}")
    train_times, holdout_times = holdout_time_split(
        df["time_id"].to_numpy(), float(cfg["holdout_fraction"])
    )
    pre = df[df["time_id"].isin(set(train_times.tolist()))].copy()
    holdout = df[df["time_id"].isin(set(holdout_times.tolist()))].copy()
    del df
    gc.collect()
    _log(f"[2/6] split pre={len(pre):,} holdout={len(holdout):,}")

    _log(f"[3/6] expanding CV n_folds={cfg['n_folds']} (longest step; little LGB stdout because verbosity=-1)...")
    cv = run_expanding_cv(
        pre,
        fit_feature_fn=fit_feature_fn,
        transform_feature_fn=transform_feature_fn,
        params=cfg["params"],
        n_folds=int(cfg["n_folds"]),
        embargo=int(cfg["embargo"]),
        num_threads=paths["num_threads"],
        num_boost_round=int(cfg["num_boost_round"]),
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
    )
    _log(f"[3/6] cv_mean={cv['mean_valid_wzm_r2']:.6f} cv_std={cv['std_valid_wzm_r2']:.6f}")

    best_iters = [f["best_iteration"] for f in cv["folds"] if f["best_iteration"] > 0]
    refit_rounds = int(max(best_iters)) if best_iters else int(cfg["num_boost_round"])
    inner_train_times, inner_valid_times = holdout_time_split(pre["time_id"].to_numpy(), 0.15)
    inner_train = pre[pre["time_id"].isin(set(inner_train_times.tolist()))].copy()
    inner_valid = pre[pre["time_id"].isin(set(inner_valid_times.tolist()))].copy()
    pre_rows = int(len(pre))
    holdout_rows = int(len(holdout))

    aux_col = None
    y_tr_aux = y_va_aux = None
    if cfg.get("auxiliary_mode", "none") == "blend":
        aux_candidates = _resolve_aux_responders(cfg, list(pre.columns), paths["experiments_dir"])
        if not aux_candidates:
            raise RuntimeError("blend mode requires a responder column")
        aux_col = aux_candidates[0]
        y_tr_aux = pd.to_numeric(inner_train[aux_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
        y_va_aux = pd.to_numeric(inner_valid[aux_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
        _log(f"[4/6] aux responder fixed to {aux_col}")

    _log("[4/6] fitting feature state + final LightGBM...")
    state = fit_feature_fn(pre)
    del pre
    gc.collect()

    x_tr2, y_tr2, w_tr2 = build_xyw(inner_train, state, transform_feature_fn)
    del inner_train
    gc.collect()
    x_va2, y_va2, w_va2 = build_xyw(inner_valid, state, transform_feature_fn)
    del inner_valid
    gc.collect()
    booster = train_lightgbm(
        x_tr2,
        y_tr2,
        w_tr2,
        x_va2,
        y_va2,
        w_va2,
        cfg["params"],
        num_threads=paths["num_threads"],
        num_boost_round=max(refit_rounds, 50),
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
    )

    out_dir = paths["artifacts_dir"] / "lgb_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    blend_info: dict | None = None
    if cfg.get("auxiliary_mode", "none") == "blend":
        _log("[5/6] training auxiliary responder blend...")
        from contest.responders import choose_blend_weight

        aux_booster = train_lightgbm(
            x_tr2,
            y_tr_aux,
            w_tr2,
            x_va2,
            y_va_aux,
            w_va2,
            cfg["params"],
            num_threads=paths["num_threads"],
            num_boost_round=max(int(booster.best_iteration or refit_rounds), 50),
            early_stopping_rounds=int(cfg["early_stopping_rounds"]),
        )
        del x_tr2, y_tr2, w_tr2, x_va2, y_va2, w_va2, y_tr_aux, y_va_aux
        gc.collect()
        x_hold, y_hold, w_hold = build_xyw(holdout, state, transform_feature_fn)
        del holdout
        gc.collect()
        pred_main = booster.predict(x_hold)
        pred_aux = aux_booster.predict(x_hold)
        holdout_main = evaluate_booster(booster, x_hold, y_hold, w_hold)
        blend_a, blend_score = choose_blend_weight(y_hold, pred_main, pred_aux, w_hold)
        blend_info = {
            "aux_responder": aux_col,
            "blend_weight_main": blend_a,
            "holdout_wzm_r2_blend": blend_score,
            "holdout_wzm_r2_main_only": float(holdout_main),
        }
        holdout_score = blend_score
        aux_booster.save_model(str(out_dir / "aux_model.txt"))
        (out_dir / "blend_weight.json").write_text(json.dumps(blend_info, indent=2), encoding="utf-8")
        _log(
            f"[5/6] blend_a={blend_a:.3f} holdout_main={holdout_main:.6f} "
            f"holdout_blend={blend_score:.6f}"
        )
    else:
        _log("[5/6] blend skipped (auxiliary_mode != blend)")
        del x_tr2, y_tr2, w_tr2, x_va2, y_va2, w_va2
        gc.collect()
        x_hold, y_hold, w_hold = build_xyw(holdout, state, transform_feature_fn)
        del holdout
        gc.collect()
        holdout_score = evaluate_booster(booster, x_hold, y_hold, w_hold)
        _log(f"[4/6] holdout_wzm_r2={holdout_score:.6f} best_iteration={booster.best_iteration}")

    _log("[6/6] writing artifacts...")
    booster.save_model(str(out_dir / "model.txt"))
    (out_dir / "feature_state.json").write_text(
        json.dumps(state_to_jsonable(state), indent=2), encoding="utf-8"
    )
    metrics = {
        "feature_version": feature_version,
        "partitions_used": part_names,
        "train_pre_rows": pre_rows,
        "holdout_rows": holdout_rows,
        "cv": cv,
        "best_iteration": int(booster.best_iteration or 0),
        "holdout_wzm_r2": float(holdout_score),
        "blend": blend_info,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "cv_report.json").write_text(json.dumps(cv, indent=2), encoding="utf-8")
    _log(json.dumps({k: metrics[k] for k in ("partitions_used", "holdout_wzm_r2", "best_iteration")}, indent=2))
    if blend_info:
        _log(json.dumps(blend_info, indent=2))
    _log(f"DONE cv_mean={cv['mean_valid_wzm_r2']} cv_std={cv['std_valid_wzm_r2']}")


if __name__ == "__main__":
    main()
