from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contest.paths import load_paths

from build_smoke_data_root import DEFAULT_N_TIME_IDS, build_smoke_data_root

MODE_HELP = (
    "'smoke' (default) runs the official runner against a tiny sampled test "
    "data-root (see build_smoke_data_root.py) instead of the real data_root. "
    "The official runner has no --limit flag and loads each test partition "
    "whole via pd.read_parquet; on this host that raised a MemoryError against "
    "the real ~1M-row test_partition_000.parquet. Use --mode full only on a "
    "host with enough free RAM to hold a full test partition (all ~323 "
    "feature columns) in memory at once, e.g.:\n"
    "  python scripts/run_runner_gate_v0.py --mode full"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strategy_v0 through the official Time-Series API runner.",
        epilog=MODE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default=os.environ.get("RUNNER_GATE_MODE", "smoke"),
        help="Data-root to run against; see epilog. Overridable via RUNNER_GATE_MODE env var.",
    )
    parser.add_argument(
        "--n-time-ids",
        type=int,
        default=DEFAULT_N_TIME_IDS,
        help="Smoke mode only: number of leading time_ids to sample when (re)building the smoke data-root.",
    )
    parser.add_argument(
        "--rebuild-smoke-root",
        action="store_true",
        help="Smoke mode only: force rebuild of the smoke data-root even if it already exists.",
    )
    parser.add_argument("--per-step-timeout-seconds", type=float, default=0.5)
    parser.add_argument(
        "--timeout-policy",
        choices=["zero_step", "zero_remaining"],
        default="zero_step",
    )
    parser.add_argument("--output", default=None, help="Override submission CSV output path.")
    parser.add_argument(
        "--gate-json",
        default=None,
        help="Where to write the runner's stdout JSON (status/timing). Defaults under workspace/experiments/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths()
    runner = paths["release_root"] / "timeseries_api" / "run_timeseries_api.py"

    if args.mode == "smoke":
        data_root = paths["artifacts_dir"] / "smoke_data_root"
        build_smoke_data_root(
            data_root=paths["data_root"],
            out_root=data_root,
            n_time_ids=args.n_time_ids,
            force=args.rebuild_smoke_root,
        )
        default_out_name = "lgb_v0_submission_smoke.csv"
        default_gate_json_name = "runner_gate_v0_smoke.json"
    else:
        data_root = paths["data_root"]
        default_out_name = "lgb_v0_submission.csv"
        default_gate_json_name = "runner_gate_v0.json"

    out = Path(args.output) if args.output else paths["workspace_root"] / "submission" / "public" / default_out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    gate_json_path = (
        Path(args.gate_json) if args.gate_json else paths["experiments_dir"] / default_gate_json_name
    )

    cmd = [
        sys.executable,
        str(runner),
        "--data-root",
        str(data_root),
        "--strategy-dir",
        str(paths["submission_strategy_dir"]),
        "--output",
        str(out),
        "--per-step-timeout-seconds",
        str(args.per_step_timeout_seconds),
        "--timeout-policy",
        args.timeout_policy,
    ]
    print("mode:", args.mode)
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    print(proc.stdout)

    payload = json.loads(proc.stdout)
    payload["mode"] = args.mode
    payload["data_root"] = str(data_root)
    gate_json_path.parent.mkdir(parents=True, exist_ok=True)
    gate_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"gate JSON written to {gate_json_path}")


if __name__ == "__main__":
    main()
