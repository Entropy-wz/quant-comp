from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contest.paths import load_paths
from build_smoke_data_root import DEFAULT_N_TIME_IDS, build_smoke_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner gate for strategy_v1")
    parser.add_argument("--mode", choices=["smoke", "full"], default=os.environ.get("RUNNER_GATE_MODE", "smoke"))
    parser.add_argument("--n-time-ids", type=int, default=DEFAULT_N_TIME_IDS)
    parser.add_argument("--rebuild-smoke-root", action="store_true")
    parser.add_argument("--per-step-timeout-seconds", type=float, default=0.5)
    parser.add_argument("--timeout-policy", choices=["zero_step", "zero_remaining"], default="zero_step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths()
    runner = paths["release_root"] / "timeseries_api" / "run_timeseries_api.py"
    strategy = paths["workspace_root"] / "submission" / "strategy_v1"

    if args.mode == "smoke":
        data_root = paths["artifacts_dir"] / "smoke_data_root"
        build_smoke_data_root(
            data_root=paths["data_root"],
            out_root=data_root,
            n_time_ids=args.n_time_ids,
            force=args.rebuild_smoke_root,
        )
        out = paths["workspace_root"] / "submission" / "public" / "lgb_v1_submission_smoke.csv"
        gate_json = paths["experiments_dir"] / "runner_gate_v1_smoke.json"
    else:
        data_root = paths["data_root"]
        out = paths["workspace_root"] / "submission" / "public" / "lgb_v1_submission.csv"
        gate_json = paths["experiments_dir"] / "runner_gate_v1_full.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(runner),
        "--data-root",
        str(data_root),
        "--strategy-dir",
        str(strategy),
        "--output",
        str(out),
        "--per-step-timeout-seconds",
        str(args.per_step_timeout_seconds),
        "--timeout-policy",
        args.timeout_policy,
    ]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(proc.stdout)
    gate_json.write_text(proc.stdout, encoding="utf-8")
    print(f"wrote {gate_json}")


if __name__ == "__main__":
    main()
