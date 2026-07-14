from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.features_v0 import (
    FeatureV0State,
    state_from_jsonable as state_from_jsonable_v0,
    transform_feature_v0,
)
from contest.features_v1 import (
    RollingState,
    state_from_jsonable as state_from_jsonable_v1,
    transform_feature_v1_frame,
)
from contest.paths import load_paths
from contest.predict_batch import (
    iter_test_files,
    predict_test_partition_chunked,
    write_public_submission,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public-LB CSV without official runner.")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--time-id-batch", type=int, default=1500)
    args = parser.parse_args()

    paths = load_paths()
    art = Path(args.artifact_dir) if args.artifact_dir else (paths["artifacts_dir"] / "lgb_v1")
    out = (
        Path(args.output)
        if args.output
        else (paths["workspace_root"] / "submission" / "public" / "lgb_v1_submission.csv")
    )

    booster = lgb.Booster(model_file=str(art / "model.txt"))
    raw = json.loads((art / "feature_state.json").read_text(encoding="utf-8"))
    is_v1 = "window" in raw and "roll_feature_cols" in raw
    if is_v1:
        state = state_from_jsonable_v1(raw)

        def _transform_full(df):
            return transform_feature_v1_frame(df, state)

    else:
        state = state_from_jsonable_v0(raw)

        def _transform_full(df):
            return transform_feature_v0(df, state)

    selected = raw.get("selected_indices")
    if selected is not None:
        idx = np.asarray(selected, dtype=np.int64)

        def transform(df):
            return _transform_full(df)[:, idx]
    else:
        transform = _transform_full

    columns = ["row_id", "time_id", "asset_id", *state.feature_cols]
    blend_a = 1.0
    aux = None
    if (art / "blend_weight.json").exists() and (art / "aux_model.txt").exists():
        blend_a = float(json.loads((art / "blend_weight.json").read_text(encoding="utf-8"))["blend_weight_main"])
        aux = lgb.Booster(model_file=str(art / "aux_model.txt"))

    row_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for path in iter_test_files(paths["data_root"]):
        rows, preds = predict_test_partition_chunked(
            path,
            columns,
            transform,
            booster,
            num_threads=paths["num_threads"],
            time_id_batch=args.time_id_batch,
        )
        if aux is not None and blend_a < 1.0:
            # Recompute aux on same transform — reload partition chunk is expensive;
            # instead re-predict by transforming again inside a thin helper:
            # For simplicity, run second pass predict on same files with aux booster.
            _, aux_preds = predict_test_partition_chunked(
                path,
                columns,
                transform,
                aux,
                num_threads=paths["num_threads"],
                time_id_batch=args.time_id_batch,
            )
            preds = blend_a * preds + (1.0 - blend_a) * aux_preds
        row_parts.append(rows)
        pred_parts.append(preds)
        print(f"done {path.name}: {len(rows)} rows")

    write_public_submission(np.concatenate(row_parts), np.concatenate(pred_parts), out)
    print(f"wrote {out} rows={sum(len(r) for r in row_parts)}")


if __name__ == "__main__":
    main()
