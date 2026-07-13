from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_paths(config_path: Path | None = None) -> dict[str, Any]:
    cfg_path = config_path or (REPO_ROOT / "workspace" / "configs" / "paths.yaml")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (REPO_ROOT / path)

    return {
        "repo_root": REPO_ROOT,
        "data_root": resolve(raw["data_root"]),
        "release_root": resolve(raw["release_root"]),
        "workspace_root": resolve(raw["workspace_root"]),
        "artifacts_dir": resolve(raw["artifacts_dir"]),
        "experiments_dir": resolve(raw["experiments_dir"]),
        "submission_strategy_dir": resolve(raw["submission_strategy_dir"]),
        "num_threads": int(raw.get("num_threads", 4)),
    }
