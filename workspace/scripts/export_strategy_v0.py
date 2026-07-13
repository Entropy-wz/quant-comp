from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.paths import load_paths


def main() -> None:
    paths = load_paths()
    src = paths["artifacts_dir"] / "lgb_v0"
    dst = paths["submission_strategy_dir"]
    model_dir = dst / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "model.txt", model_dir / "model.txt")
    shutil.copy2(src / "feature_state.json", model_dir / "feature_state.json")
    print(f"exported artifacts to {model_dir}")
    if not (dst / "main.py").exists():
        raise FileNotFoundError("main.py missing in strategy_v0")


if __name__ == "__main__":
    main()
