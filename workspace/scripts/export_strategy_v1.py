from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.paths import load_paths


def main() -> None:
    paths = load_paths()
    src = paths["artifacts_dir"] / "lgb_v1"
    dst = paths["workspace_root"] / "submission" / "strategy_v1"
    model_dir = dst / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.txt", "feature_state.json"):
        p = src / name
        if not p.exists():
            raise FileNotFoundError(p)
        shutil.copy2(p, model_dir / name)
    for name in ("aux_model.txt", "blend_weight.json"):
        stale = model_dir / name
        if stale.exists():
            stale.unlink()
    if not (dst / "main.py").exists():
        raise FileNotFoundError("main.py missing in strategy_v1")
    print(f"exported artifacts to {model_dir}")


if __name__ == "__main__":
    main()
