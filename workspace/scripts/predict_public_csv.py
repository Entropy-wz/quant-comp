from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.features_v0 import state_from_jsonable, transform_feature_v0
from contest.paths import load_paths
from contest.predict_batch import (
    iter_test_files,
    predict_test_partition,
    predict_test_partition_chunked,
    write_public_submission,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public-LB CSV without official runner.")
    parser.add_argument("--artifact-dir", default=None, help="Dir with model.txt + feature_state.json")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--mode",
        choices=["simple", "chunked"],
        default="chunked",
        help="simple=pd.read_parquet whole partition; chunked=time_id batches (safer RAM)",
    )
    parser.add_argument("--time-id-batch", type=int, default=2000)
    args = parser.parse_args()

    paths = load_paths()
    art = Path(args.artifact_dir) if args.artifact_dir else (paths["artifacts_dir"] / "lgb_v0")
    out = (
        Path(args.output)
        if args.output
        else (paths["workspace_root"] / "submission" / "public" / "lgb_v0_submission.csv")
    )

    booster = lgb.Booster(model_file=str(art / "model.txt"))
    state = state_from_jsonable(json.loads((art / "feature_state.json").read_text(encoding="utf-8")))
    columns = ["row_id", "time_id", "asset_id", *state.feature_cols]

    def transform(df):
        return transform_feature_v0(df, state)

    row_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for path in iter_test_files(paths["data_root"]):
        if args.mode == "chunked":
            rows, preds = predict_test_partition_chunked(
                path,
                columns,
                transform,
                booster,
                num_threads=paths["num_threads"],
                time_id_batch=args.time_id_batch,
            )
        else:
            rows, preds = predict_test_partition(
                path, columns, transform, booster, paths["num_threads"]
            )
        row_parts.append(rows)
        pred_parts.append(preds)
        print(f"done {path.name}: {len(rows)} rows")

    write_public_submission(np.concatenate(row_parts), np.concatenate(pred_parts), out)
    print(f"wrote {out} rows={sum(len(r) for r in row_parts)}")


if __name__ == "__main__":
    main()
