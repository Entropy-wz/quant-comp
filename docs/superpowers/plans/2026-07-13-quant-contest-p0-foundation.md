# Quant Contest P0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local workspace that can load contest data, score with weighted zero-mean R², run time-series CV with embargo, train a LightGBM V0 baseline, and produce a Time-Series API strategy package that passes the official runner gate.

**Architecture:** Keep official `public_release_20260630` read-only. All work lives under `workspace/`. Shared library code in `workspace/src/` powers training scripts and is mirrored into `workspace/submission/strategy/` only as the minimal `main.py` + artifacts needed for inference. Validation and metrics are locked before any model tuning.

**Tech Stack:** Python 3.11, numpy, pandas, pyarrow, polars, scikit-learn, lightgbm, pytest. Official runner under `public_release_.../timeseries_api/`.

**Spec:** `docs/superpowers/specs/2026-07-13-quant-contest-gbdt-design.md`

## Global Constraints

- Official data root (read-only): `public_release_20260630/home/jovyan/shared/public_release_20260630/`
- Workspace root: `workspace/`
- Inference must stay CPU-friendly: local eval `num_threads <= 4`; never rely on GPU for private-LB `predict`
- No future leakage: fill/encode stats fit on train fold only
- Metric: weighted zero-mean R² only for model selection
- Do not modify files under `public_release_20260630/` except reading them
- This repo may not have git yet: Task 1 initializes git if missing; if user forbids commits, skip commit steps and continue
- P1–P3 (rolling features, responders, XGB, ensemble, label refill) are **out of scope** for this plan; write follow-up plans after P0 completes

---

## File Structure (locked for P0)

```text
workspace/
  README.md
  pyproject.toml                 # or requirements-dev.txt
  configs/
    paths.yaml                   # data_root, artifacts, threads
    train_lgb_v0.yaml
  src/
    __init__.py
    contest/
      __init__.py
      paths.py                   # resolve data/artifact paths
      metrics.py                 # weighted_zero_mean_r2
      io.py                      # load train/test partitions
      validation.py              # time-series split + embargo + folds
      features_v0.py             # raw features + fill + asset encoding
      train_lgb.py               # train/eval helpers
  tests/
    test_metrics.py
    test_io.py
    test_validation.py
    test_features_v0.py
  scripts/
    eda_summary.py               # lightweight EDA (no huge notebook required)
    train_lgb_v0.py
    export_strategy_v0.py
    run_public_predict_v0.py
  experiments/
    .gitkeep
  artifacts/
    .gitkeep
  submission/
    public/
      .gitkeep
    strategy_v0/
      main.py
      model/
        .gitkeep
  notebooks/
    .gitkeep
```

---

### Task 1: Workspace scaffold + dependencies + git

**Files:**
- Create: `workspace/README.md`
- Create: `workspace/requirements-dev.txt`
- Create: `workspace/configs/paths.yaml`
- Create: `workspace/src/__init__.py`
- Create: `workspace/src/contest/__init__.py`
- Create: `workspace/src/contest/paths.py`
- Create: `workspace/experiments/.gitkeep`
- Create: `workspace/artifacts/.gitkeep`
- Create: `workspace/submission/public/.gitkeep`
- Create: `workspace/submission/strategy_v0/model/.gitkeep`
- Create: `workspace/notebooks/.gitkeep`
- Create: `workspace/tests/__init__.py`

**Interfaces:**
- Consumes: none
- Produces: `contest.paths.load_paths()` → `dict` with keys `repo_root`, `data_root`, `workspace_root`, `artifacts_dir`, `num_threads`

- [ ] **Step 1: Create directories and path config**

Create `workspace/configs/paths.yaml`:

```yaml
# Paths are resolved relative to the quant repo root unless absolute.
data_root: public_release_20260630/home/jovyan/shared/public_release_20260630/data
release_root: public_release_20260630/home/jovyan/shared/public_release_20260630
workspace_root: workspace
artifacts_dir: workspace/artifacts
experiments_dir: workspace/experiments
submission_strategy_dir: workspace/submission/strategy_v0
num_threads: 4
```

Create `workspace/requirements-dev.txt`:

```text
numpy==1.24.3
pandas==2.0.3
pyarrow==11.0.0
scikit-learn==1.3.0
polars>=0.20.0
lightgbm>=4.0.0
pytest>=7.4.0
PyYAML>=6.0
```

Create `workspace/README.md`:

```markdown
# Contest Workspace (P0)

Local training and strategy export for the 2026 quant contest.

## Setup

```bash
pip install -r workspace/requirements-dev.txt
```

## Key commands

See `docs/superpowers/plans/2026-07-13-quant-contest-p0-foundation.md`.
```

- [ ] **Step 2: Implement `paths.py`**

```python
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
```

- [ ] **Step 3: Install deps and verify import**

Run:

```bash
pip install -r workspace/requirements-dev.txt
python -c "from workspace.src.contest.paths import load_paths; p=load_paths(); print(p['data_root']); print(p['data_root'].exists())"
```

Expected: path prints and `True` (data folder exists).

Note: package import may need `PYTHONPATH=workspace/src` or install as editable. Prefer running tests with:

```bash
cd workspace
$env:PYTHONPATH="src"
python -c "from contest.paths import load_paths; print(load_paths()['data_root'].exists())"
```

Expected: `True`

- [ ] **Step 4: Initialize git if missing, then commit scaffold**

```bash
# from repo root D:\exp_all\assign\quant
if (-not (Test-Path .git)) { git init }
git add workspace/docs/superpowers
git status
git commit -m "chore: scaffold contest workspace for P0 foundation"
```

If the user previously forbade commits, skip this step.

---

### Task 2: Weighted zero-mean R² metric

**Files:**
- Create: `workspace/src/contest/metrics.py`
- Create: `workspace/tests/test_metrics.py`

**Interfaces:**
- Consumes: none
- Produces: `weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from contest.metrics import weighted_zero_mean_r2


def test_all_zero_predictions_score_zero():
    y = np.array([1.0, -2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0])
    pred = np.zeros_like(y)
    assert abs(weighted_zero_mean_r2(y, pred, w) - 0.0) < 1e-12


def test_perfect_predictions_score_one():
    y = np.array([1.0, -2.0, 3.0])
    w = np.array([1.0, 2.0, 0.5])
    assert abs(weighted_zero_mean_r2(y, y, w) - 1.0) < 1e-12


def test_zero_denominator_returns_zero():
    y = np.zeros(3)
    w = np.ones(3)
    pred = np.array([0.1, -0.2, 0.3])
    assert weighted_zero_mean_r2(y, pred, w) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `contest.metrics`.

- [ ] **Step 3: Implement metric**

```python
from __future__ import annotations

import numpy as np


def weighted_zero_mean_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denom = float(np.sum(weight * y_true * y_true))
    if denom <= 0.0:
        return 0.0
    numer = float(np.sum(weight * (y_true - y_pred) ** 2))
    return float(1.0 - numer / denom)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_metrics.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add workspace/src/contest/metrics.py workspace/tests/test_metrics.py
git commit -m "feat: add weighted zero-mean R2 metric"
```

---

### Task 3: Data IO (manifest-aware parquet loading)

**Files:**
- Create: `workspace/src/contest/io.py`
- Create: `workspace/tests/test_io.py`

**Interfaces:**
- Consumes: `load_paths()`
- Produces:
  - `list_partition_files(data_root: Path, split: str) -> list[Path]`
  - `load_partitions(files: list[Path], columns: list[str] | None = None) -> pd.DataFrame`
  - `feature_columns(frame: pd.DataFrame) -> list[str]`
  - `responder_columns(frame: pd.DataFrame) -> list[str]`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import pandas as pd
from contest.io import feature_columns, list_partition_files, load_partitions
from contest.paths import load_paths


def test_list_train_partitions_from_manifest():
    paths = load_paths()
    files = list_partition_files(paths["data_root"], "train")
    assert len(files) == 9
    assert all(p.exists() for p in files)


def test_load_single_partition_has_required_columns():
    paths = load_paths()
    files = list_partition_files(paths["data_root"], "train")
    # Read only one partition and a small column set for speed/memory
    cols = ["row_id", "time_id", "asset_id", "weight", "target", "feature_000"]
    df = load_partitions([files[0]], columns=cols)
    assert isinstance(df, pd.DataFrame)
    assert set(cols).issubset(df.columns)
    assert len(df) > 0


def test_feature_columns_sorted():
    df = pd.DataFrame(
        {
            "feature_001": [1.0],
            "feature_000": [2.0],
            "target": [0.0],
        }
    )
    assert feature_columns(df) == ["feature_000", "feature_001"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_io.py -v
```

Expected: FAIL importing `contest.io`.

- [ ] **Step 3: Implement IO**

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def list_partition_files(data_root: Path, split: str) -> list[Path]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rels = manifest.get("files", {}).get(split, [])
        if rels:
            return [data_root / rel for rel in rels]
    pattern = "train_partition_*.parquet" if split == "train" else "test_partition_*.parquet"
    return sorted((data_root / split).glob(pattern))


def load_partitions(files: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    if not files:
        raise ValueError("no parquet files provided")
    frames = [pd.read_parquet(path, columns=columns) for path in files]
    return pd.concat(frames, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith("feature_"))


def responder_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith("responder_"))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_io.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add workspace/src/contest/io.py workspace/tests/test_io.py
git commit -m "feat: add manifest-aware parquet IO helpers"
```

---

### Task 4: Time-series validation with embargo

**Files:**
- Create: `workspace/src/contest/validation.py`
- Create: `workspace/tests/test_validation.py`

**Interfaces:**
- Consumes: none (operates on `time_id` arrays / frames)
- Produces:
  - `unique_sorted_time_ids(time_ids: np.ndarray) -> np.ndarray`
  - `holdout_time_split(time_ids: np.ndarray, holdout_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]`
  - `expanding_folds(time_ids: np.ndarray, n_folds: int = 3, embargo: int = 5) -> list[tuple[np.ndarray, np.ndarray]]`
    - each item is `(train_time_ids, valid_time_ids)`
    - embargo: drop this many `time_id` values immediately before each valid block from train

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from contest.validation import expanding_folds, holdout_time_split, unique_sorted_time_ids


def test_unique_sorted_time_ids():
    t = np.array([3, 1, 2, 2, 1])
    assert np.array_equal(unique_sorted_time_ids(t), np.array([1, 2, 3]))


def test_holdout_split_is_contiguous_tail():
    times = np.arange(10)
    train_t, valid_t = holdout_time_split(times, holdout_fraction=0.2)
    assert np.array_equal(valid_t, np.array([8, 9]))
    assert np.array_equal(train_t, np.arange(8))


def test_expanding_folds_respect_order_and_embargo():
    times = np.arange(20)
    folds = expanding_folds(times, n_folds=3, embargo=2)
    assert len(folds) == 3
    for train_t, valid_t in folds:
        assert train_t.max() < valid_t.min()
        # embargo gap: at least 2 time ids between train max and valid min when possible
        assert valid_t.min() - train_t.max() >= 2
        assert len(np.intersect1d(train_t, valid_t)) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_validation.py -v
```

Expected: FAIL importing `contest.validation`.

- [ ] **Step 3: Implement validation**

```python
from __future__ import annotations

import numpy as np


def unique_sorted_time_ids(time_ids: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(time_ids))


def holdout_time_split(
    time_ids: np.ndarray,
    holdout_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    times = unique_sorted_time_ids(time_ids)
    if len(times) < 2:
        raise ValueError("need at least 2 distinct time_id values")
    n_valid = max(1, int(round(len(times) * holdout_fraction)))
    n_valid = min(n_valid, len(times) - 1)
    valid = times[-n_valid:]
    train = times[:-n_valid]
    return train, valid


def expanding_folds(
    time_ids: np.ndarray,
    n_folds: int = 3,
    embargo: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    times = unique_sorted_time_ids(time_ids)
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if len(times) < n_folds + 2:
        raise ValueError("not enough time_ids for requested folds")

    # Reserve last 20% conceptually by splitting the earlier region into fold valids
    # Use equal-sized validation blocks walking forward over times excluding a final buffer.
    n = len(times)
    # Keep at least 1 time for initial train; split remaining into n_folds valid blocks
    min_train = max(1, n // (n_folds + 2))
    usable = times[min_train:]
    block = len(usable) // n_folds
    if block < 1:
        raise ValueError("validation block too small")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        start = i * block
        end = (i + 1) * block if i < n_folds - 1 else len(usable)
        valid = usable[start:end]
        # train = all times before valid start, minus embargo window
        valid_start_idx = int(np.searchsorted(times, valid[0], side="left"))
        train_end_idx = max(0, valid_start_idx - embargo)
        train = times[:train_end_idx]
        if len(train) == 0 or len(valid) == 0:
            continue
        if train.max() >= valid.min():
            raise RuntimeError("fold ordering violated")
        folds.append((train, valid))
    if not folds:
        raise RuntimeError("no valid folds constructed")
    return folds
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_validation.py -v
```

Expected: PASS. If `test_expanding_folds_respect_order_and_embargo` fails on the gap assertion for edge folds, adjust `embargo` application so `valid.min() - train.max() >= embargo` (not `embargo - 1`).

- [ ] **Step 5: Commit**

```bash
git add workspace/src/contest/validation.py workspace/tests/test_validation.py
git commit -m "feat: add time-series holdout and expanding folds with embargo"
```

---

### Task 5: Features V0 (fill + asset id)

**Files:**
- Create: `workspace/src/contest/features_v0.py`
- Create: `workspace/tests/test_features_v0.py`

**Interfaces:**
- Consumes: `feature_columns()`
- Produces:
  - `@dataclass FeatureV0State` with `feature_cols: list[str]`, `fill_values: dict[str, float]`, `asset_levels: list[int]`
  - `fit_feature_v0(train_df: pd.DataFrame) -> FeatureV0State`
  - `transform_feature_v0(df: pd.DataFrame, state: FeatureV0State) -> np.ndarray`  # shape (n, n_features+1), float32
  - `state_to_jsonable(state) -> dict` / `state_from_jsonable(d) -> FeatureV0State`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pandas as pd
from contest.features_v0 import fit_feature_v0, transform_feature_v0


def test_fit_transform_fills_nan_and_appends_asset():
    train = pd.DataFrame(
        {
            "asset_id": [0, 1],
            "feature_000": [1.0, np.nan],
            "feature_001": [np.nan, 4.0],
        }
    )
    state = fit_feature_v0(train)
    X = transform_feature_v0(train, state)
    assert X.shape == (2, 3)
    assert np.isfinite(X).all()
    assert X.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_features_v0.py -v
```

Expected: FAIL importing `contest.features_v0`.

- [ ] **Step 3: Implement features V0**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from contest.io import feature_columns


@dataclass
class FeatureV0State:
    feature_cols: list[str]
    fill_values: dict[str, float]
    asset_levels: list[int]


def fit_feature_v0(train_df: pd.DataFrame) -> FeatureV0State:
    cols = feature_columns(train_df)
    fill_values = {}
    for col in cols:
        median = pd.to_numeric(train_df[col], errors="coerce").median()
        fill_values[col] = float(0.0 if pd.isna(median) else median)
    asset_levels = sorted(int(x) for x in pd.unique(train_df["asset_id"]))
    return FeatureV0State(feature_cols=cols, fill_values=fill_values, asset_levels=asset_levels)


def transform_feature_v0(df: pd.DataFrame, state: FeatureV0State) -> np.ndarray:
    feats = []
    for col in state.feature_cols:
        s = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        s = s.fillna(np.float32(state.fill_values[col]))
        s = s.replace([np.inf, -np.inf], np.float32(state.fill_values[col]))
        feats.append(s.to_numpy(dtype=np.float32))
    x = np.column_stack(feats) if feats else np.zeros((len(df), 0), dtype=np.float32)
    asset = df["asset_id"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return np.concatenate([x, asset], axis=1).astype(np.float32, copy=False)


def state_to_jsonable(state: FeatureV0State) -> dict:
    return asdict(state)


def state_from_jsonable(payload: dict) -> FeatureV0State:
    return FeatureV0State(
        feature_cols=list(payload["feature_cols"]),
        fill_values={k: float(v) for k, v in payload["fill_values"].items()},
        asset_levels=[int(x) for x in payload["asset_levels"]],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_features_v0.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add workspace/src/contest/features_v0.py workspace/tests/test_features_v0.py
git commit -m "feat: add V0 features with median fill and asset_id"
```

---

### Task 6: LightGBM train helper + V0 train script

**Files:**
- Create: `workspace/src/contest/train_lgb.py`
- Create: `workspace/configs/train_lgb_v0.yaml`
- Create: `workspace/scripts/train_lgb_v0.py`
- Create: `workspace/tests/test_train_lgb_smoke.py`

**Interfaces:**
- Consumes: metrics, validation, features_v0, io, paths
- Produces:
  - `train_lightgbm(X_train, y_train, w_train, X_valid, y_valid, w_valid, params, num_threads) -> lgb.Booster`
  - script writes `workspace/artifacts/lgb_v0/model.txt`, `feature_state.json`, `metrics.json`

- [ ] **Step 1: Write smoke test on synthetic data**

```python
import numpy as np
from contest.train_lgb import train_lightgbm


def test_train_lightgbm_smoke():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 5)).astype(np.float32)
    y_train = X_train[:, 0] * 0.2 + rng.normal(scale=0.1, size=200)
    w_train = np.ones(200)
    X_valid = rng.normal(size=(50, 5)).astype(np.float32)
    y_valid = X_valid[:, 0] * 0.2
    w_valid = np.ones(50)
    params = {
        "objective": "regression",
        "learning_rate": 0.1,
        "num_leaves": 8,
        "min_data_in_leaf": 10,
        "verbosity": -1,
    }
    booster = train_lightgbm(
        X_train, y_train, w_train, X_valid, y_valid, w_valid, params, num_threads=2, num_boost_round=20
    )
    pred = booster.predict(X_valid)
    assert len(pred) == 50
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_train_lgb_smoke.py -v
```

Expected: FAIL importing `contest.train_lgb`.

- [ ] **Step 3: Implement `train_lgb.py`**

```python
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from contest.metrics import weighted_zero_mean_r2


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    params: dict[str, Any],
    num_threads: int = 4,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train, weight=w_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, reference=train_set)
    model_params = dict(params)
    model_params["num_threads"] = int(num_threads)
    booster = lgb.train(
        model_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster


def evaluate_booster(
    booster: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> float:
    pred = booster.predict(X)
    return weighted_zero_mean_r2(y, pred, w)
```

- [ ] **Step 4: Add config + train script (subset-friendly)**

`workspace/configs/train_lgb_v0.yaml`:

```yaml
holdout_fraction: 0.2
# For first successful end-to-end run on limited RAM, use only the last N train partitions.
# Set to null to use all partitions after smoke success.
max_train_partitions: 2
num_boost_round: 300
early_stopping_rounds: 40
params:
  objective: regression
  metric: l2
  learning_rate: 0.05
  num_leaves: 64
  min_data_in_leaf: 100
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 1
  verbosity: -1
```

`workspace/scripts/train_lgb_v0.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.features_v0 import fit_feature_v0, state_to_jsonable, transform_feature_v0
from contest.io import list_partition_files, load_partitions
from contest.paths import load_paths
from contest.train_lgb import evaluate_booster, train_lightgbm
from contest.validation import holdout_time_split


def main() -> None:
    paths = load_paths()
    cfg = yaml.safe_load((ROOT / "configs" / "train_lgb_v0.yaml").read_text(encoding="utf-8"))
    files = list_partition_files(paths["data_root"], "train")
    max_parts = cfg.get("max_train_partitions")
    if max_parts is not None:
        files = files[-int(max_parts) :]

    df = load_partitions(files)
    train_times, valid_times = holdout_time_split(df["time_id"].to_numpy(), cfg["holdout_fraction"])
    train_df = df[df["time_id"].isin(set(train_times.tolist()))].copy()
    valid_df = df[df["time_id"].isin(set(valid_times.tolist()))].copy()

    state = fit_feature_v0(train_df)
    X_train = transform_feature_v0(train_df, state)
    X_valid = transform_feature_v0(valid_df, state)
    y_train = pd.to_numeric(train_df["target"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    y_valid = pd.to_numeric(valid_df["target"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w_train = pd.to_numeric(train_df["weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w_valid = pd.to_numeric(valid_df["weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)

    booster = train_lightgbm(
        X_train,
        y_train,
        w_train,
        X_valid,
        y_valid,
        w_valid,
        cfg["params"],
        num_threads=paths["num_threads"],
        num_boost_round=int(cfg["num_boost_round"]),
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
    )

    metrics = {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "partitions_used": [str(p.name) for p in files],
        "best_iteration": int(booster.best_iteration or 0),
        "train_wzm_r2": evaluate_booster(booster, X_train, y_train, w_train),
        "valid_wzm_r2": evaluate_booster(booster, X_valid, y_valid, w_valid),
    }

    out_dir = paths["artifacts_dir"] / "lgb_v0"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.txt"
    booster.save_model(str(model_path))
    (out_dir / "feature_state.json").write_text(
        json.dumps(state_to_jsonable(state), indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run unit smoke + optional short train**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_train_lgb_smoke.py -v
python scripts/train_lgb_v0.py
```

Expected: pytest PASS; train script prints JSON with `valid_wzm_r2` key and writes artifacts under `workspace/artifacts/lgb_v0/`.  
Note: first full-partition train may take a long time / much RAM; keep `max_train_partitions: 2` until stable.

- [ ] **Step 6: Commit**

```bash
git add workspace/src/contest/train_lgb.py workspace/configs/train_lgb_v0.yaml workspace/scripts/train_lgb_v0.py workspace/tests/test_train_lgb_smoke.py
git commit -m "feat: add LightGBM V0 training pipeline"
```

---

### Task 7: Strategy package `main.py` + export script

**Files:**
- Create: `workspace/submission/strategy_v0/main.py`
- Create: `workspace/scripts/export_strategy_v0.py`

**Interfaces:**
- Consumes: artifacts from Task 6
- Produces: `Model` class with `__init__(self)` and `predict(self, test) -> np.ndarray`
- Official runner imports `main.py` and calls these methods only

- [ ] **Step 1: Write `main.py` (self-contained for runner)**

`main.py` must not import `contest.*` (runner only adds strategy dir). Embed minimal transform logic:

```python
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class Model:
    def __init__(self):
        root = Path(__file__).resolve().parent
        self.booster = lgb.Booster(model_file=str(root / "model" / "model.txt"))
        state = json.loads((root / "model" / "feature_state.json").read_text(encoding="utf-8"))
        self.feature_cols = list(state["feature_cols"])
        self.fill_values = {k: float(v) for k, v in state["fill_values"].items()}
        # Keep predict deterministic and CPU-bound
        self.num_threads = 4

    def _transform(self, test: pd.DataFrame) -> np.ndarray:
        feats = []
        for col in self.feature_cols:
            if col in test.columns:
                s = pd.to_numeric(test[col], errors="coerce").astype(np.float32)
            else:
                s = pd.Series(np.full(len(test), np.nan, dtype=np.float32))
            fill = np.float32(self.fill_values.get(col, 0.0))
            s = s.fillna(fill).replace([np.inf, -np.inf], fill)
            feats.append(s.to_numpy(dtype=np.float32))
        x = np.column_stack(feats)
        asset = test["asset_id"].to_numpy(dtype=np.float32).reshape(-1, 1)
        return np.concatenate([x, asset], axis=1).astype(np.float32, copy=False)

    def predict(self, test):
        x = self._transform(test)
        pred = self.booster.predict(x, num_threads=self.num_threads)
        pred = np.asarray(pred, dtype=np.float64)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred
```

- [ ] **Step 2: Write export script**

```python
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
```

- [ ] **Step 3: Export after a successful train**

```bash
cd workspace
python scripts/export_strategy_v0.py
```

Expected: copies `model.txt` and `feature_state.json` into `submission/strategy_v0/model/`.

- [ ] **Step 4: Commit**

```bash
git add workspace/submission/strategy_v0/main.py workspace/scripts/export_strategy_v0.py
git commit -m "feat: add strategy_v0 Model for Time-Series API"
```

Do **not** commit large model binaries if they bloat the repo; add `workspace/artifacts/**` and `workspace/submission/strategy_v0/model/*.txt` to `.gitignore` if needed, keeping only code + small JSON when possible.

---

### Task 8: Official runner gate + public CSV smoke

**Files:**
- Create: `workspace/scripts/run_runner_gate_v0.py`
- Create: `workspace/scripts/eda_summary.py`

**Interfaces:**
- Consumes: exported strategy_v0, official `run_timeseries_api.py`
- Produces: submission CSV under `workspace/submission/public/` + printed timing JSON

- [ ] **Step 1: Implement runner gate wrapper**

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.paths import load_paths


def main() -> None:
    paths = load_paths()
    runner = paths["release_root"] / "timeseries_api" / "run_timeseries_api.py"
    out = paths["workspace_root"] / "submission" / "public" / "lgb_v0_submission.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(runner),
        "--data-root",
        str(paths["data_root"]),
        "--strategy-dir",
        str(paths["submission_strategy_dir"]),
        "--output",
        str(out),
        "--per-step-timeout-seconds",
        "0.5",
        "--timeout-policy",
        "zero_step",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement lightweight EDA summary**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.io import feature_columns, list_partition_files, load_partitions, responder_columns
from contest.paths import load_paths


def main() -> None:
    paths = load_paths()
    files = list_partition_files(paths["data_root"], "train")[:1]
    df = load_partitions(files)
    summary = {
        "partition": files[0].name,
        "rows": int(len(df)),
        "n_assets": int(df["asset_id"].nunique()),
        "n_time_ids": int(df["time_id"].nunique()),
        "n_features": len(feature_columns(df)),
        "n_responders": len(responder_columns(df)),
        "target_mean": float(pd.to_numeric(df["target"], errors="coerce").mean()),
        "target_std": float(pd.to_numeric(df["target"], errors="coerce").std()),
        "weight_mean": float(pd.to_numeric(df["weight"], errors="coerce").mean()),
        "feature_nan_frac_mean": float(
            df[feature_columns(df)].isna().mean().mean()
        ),
    }
    out = paths["experiments_dir"] / "eda_summary_partition0.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run EDA + runner gate**

```bash
cd workspace
python scripts/eda_summary.py
python scripts/export_strategy_v0.py
python scripts/run_runner_gate_v0.py
```

Expected:
- EDA JSON written under `experiments/`
- Runner exits 0
- stdout JSON includes `status` success-like fields and `timing.mean_predict_seconds`
- CSV exists at `workspace/submission/public/lgb_v0_submission.csv` with columns `row_id,target`

Gate criteria to record in `experiments/runner_gate_v0.json` (manually or by copying stdout):
- `model_init_seconds < 180`
- `mean_predict_seconds` clearly below `0.05`
- no NaN in submission

- [ ] **Step 4: Commit scripts (not large CSV if huge)**

```bash
git add workspace/scripts/run_runner_gate_v0.py workspace/scripts/eda_summary.py
git commit -m "feat: add EDA summary and official runner gate script"
```

---

### Task 9: P0 acceptance checklist

**Files:**
- Create: `workspace/experiments/p0_acceptance.md`

- [ ] **Step 1: Write and fill acceptance doc**

```markdown
# P0 Acceptance

- [ ] `pytest` all green under `workspace/tests`
- [ ] `eda_summary.py` runs on one train partition
- [ ] `train_lgb_v0.py` produces `artifacts/lgb_v0/metrics.json` with finite `valid_wzm_r2`
- [ ] `export_strategy_v0.py` copies model artifacts
- [ ] Official runner succeeds for `strategy_v0`
- [ ] `mean_predict_seconds` recorded and < 0.05 with margin
- [ ] No future-leakage shortcuts in V0 code review

## Recorded numbers

- valid_wzm_r2:
- mean_predict_seconds:
- model_init_seconds:
```

- [ ] **Step 2: Run full unit suite**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests -v
```

Expected: all PASS

- [ ] **Step 3: Commit acceptance checklist**

```bash
git add workspace/experiments/p0_acceptance.md
git commit -m "docs: add P0 acceptance checklist"
```

---

## P0 Done → Next Plans

After P0 acceptance, create follow-up plans (do not expand this file ad hoc):

1. **P1** — `features_v1` rolling state in `Model`, responder auxiliary targets, full-partition training, expanding-fold CV selection  
2. **P2** — XGBoost/CatBoost, seed/fold ensembles, latency trim, public LB cadence  
3. **P3** — 8/23 label refill retrain, final strategy freeze, private submission pack

---

## Self-Review (plan author)

1. **Spec coverage (P0 slice):** workspace layout, metrics, IO, validation+embargo, V0 features, LGB baseline, strategy `main.py`, runner gate, EDA, experiment artifacts — covered by Tasks 1–9. Responder modeling, V1 features, XGB/ensemble, label refill intentionally deferred.  
2. **Placeholders:** none intended; configs use concrete defaults (`max_train_partitions: 2` for first RAM-safe run).  
3. **Type consistency:** `FeatureV0State`, `weighted_zero_mean_r2`, `holdout_time_split` / `expanding_folds`, `train_lightgbm`, `Model.predict` names align across tasks.
