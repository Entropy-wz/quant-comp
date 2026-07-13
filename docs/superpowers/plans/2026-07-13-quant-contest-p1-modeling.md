# Quant Contest P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the P0 LightGBM baseline into a P1 pipeline with memory-safe full-partition training, expanding-fold CV, rolling V1 features, responder-aware auxiliary training, a matching `strategy_v1` package, and a **public-LB batch CSV exporter** that does not depend on the OOM-prone full official runner.

**Architecture:** Keep `contest.*` library code as the source of truth for transforms. Training scripts use expanding folds + embargo. Inference has two consumers that must stay semantically identical: (1) `submission/strategy_v1/main.py` for Time-Series API / private LB, (2) `scripts/predict_public_csv.py` for chunked public CSV generation. Rolling state lives in a shared helper used by both train-matrix builders and `Model`.

**Tech Stack:** Python 3.11 preferred (contest parity); numpy/pandas/pyarrow/polars/lightgbm/pytest; official `timeseries_api` for smoke/full gates when RAM allows.

**Spec:** `docs/superpowers/specs/2026-07-13-quant-contest-gbdt-design.md` (P1 slice: V1 features, responders, full training, public CSV)  
**Depends on:** P0 complete on `main` (`workspace/` metrics, io, validation, features_v0, train_lgb, strategy_v0)

## Global Constraints

- Official data root (read-only): `public_release_20260630/home/jovyan/shared/public_release_20260630/`
- Workspace root: `workspace/`
- Local eval `num_threads <= 4`; never rely on GPU for private-LB `predict`
- No future leakage: fill stats and responder targets only from train folds; rolling features use current+past only
- Metric for selection: weighted zero-mean R² only
- Do not modify files under `public_release_20260630/` except reading them
- Public LB submitable CSV must cover **all** test `row_id`s (~3,217,458 rows), columns exactly `row_id,target`
- Private strategy package must remain self-contained (no `contest.*` imports inside `main.py`)
- Prefer Python 3.11 for contest-pinned deps when available; document if host is 3.13
- P2 (XGB/ensemble) and P3 (label refill freeze) are out of scope

---

## File Structure (P1 additions)

```text
workspace/
  src/contest/
    features_v1.py          # rolling mean/std/diff + FeatureV1State
    responders.py           # corr analysis + selected responder list helpers
    train_matrix.py         # build X/y/w for a fold (V0 or V1)
    predict_batch.py        # chunked test prediction shared by public CSV
  configs/
    train_lgb_v1.yaml
  scripts/
    analyze_responders.py
    train_lgb_v1.py
    export_strategy_v1.py
    predict_public_csv.py   # PUBLIC LB deliverable
    run_runner_gate_v1.py
  submission/
    strategy_v1/
      main.py
      model/
  tests/
    test_features_v1.py
    test_responders.py
    test_predict_batch.py
    test_train_matrix.py
```

---

### Task 1: Public-LB chunked CSV predictor (unblock submissions)

**Files:**
- Create: `workspace/src/contest/predict_batch.py`
- Create: `workspace/scripts/predict_public_csv.py`
- Create: `workspace/tests/test_predict_batch.py`

**Interfaces:**
- Consumes: `list_partition_files`, `load_partitions` (column subsets), V0 or V1 transform callable, LightGBM booster
- Produces: `iter_test_batches(...)` and `write_public_submission(row_ids, preds, path)`; script writes `workspace/submission/public/lgb_v*_submission.csv`

**Why first:** P0 has no full public CSV path; host OOMs on official full runner. Public LB only needs CSV.

- [ ] **Step 1: Write failing tests for batch writer + finite preds**

```python
import numpy as np
import pandas as pd
from pathlib import Path
from contest.predict_batch import write_public_submission


def test_write_public_submission_schema(tmp_path: Path):
    path = tmp_path / "sub.csv"
    write_public_submission(
        row_ids=np.array([10, 11], dtype=np.int64),
        preds=np.array([0.1, -0.2], dtype=np.float64),
        path=path,
    )
    df = pd.read_csv(path)
    assert list(df.columns) == ["row_id", "target"]
    assert len(df) == 2
    assert np.isfinite(df["target"].to_numpy()).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_predict_batch.py -v
```

Expected: FAIL importing `contest.predict_batch`.

- [ ] **Step 3: Implement `predict_batch.py` + script**

```python
# workspace/src/contest/predict_batch.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import lightgbm as lgb
import numpy as np
import pandas as pd

from contest.io import list_partition_files


TransformFn = Callable[[pd.DataFrame], np.ndarray]


def write_public_submission(row_ids: np.ndarray, preds: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = np.nan_to_num(np.asarray(preds, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    out = pd.DataFrame({"row_id": np.asarray(row_ids, dtype=np.int64), "target": preds})
    out.to_csv(path, index=False)


def iter_test_files(data_root: Path) -> list[Path]:
    return list_partition_files(data_root, "test")


def predict_test_partition(
    path: Path,
    columns: list[str],
    transform: TransformFn,
    booster: lgb.Booster,
    num_threads: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    # columns must include row_id, asset_id, and all feature cols needed by transform
    df = pd.read_parquet(path, columns=columns)
    x = transform(df)
    pred = booster.predict(x, num_threads=num_threads)
    return df["row_id"].to_numpy(dtype=np.int64), np.asarray(pred, dtype=np.float64)
```

`workspace/scripts/predict_public_csv.py` (V0-compatible first; V1 flag later in Task 5):

```python
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
from contest.predict_batch import iter_test_files, predict_test_partition, write_public_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=None, help="Directory with model.txt + feature_state.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    paths = load_paths()
    art = Path(args.artifact_dir) if args.artifact_dir else (paths["artifacts_dir"] / "lgb_v0")
    out = Path(args.output) if args.output else (paths["workspace_root"] / "submission" / "public" / "lgb_v0_submission.csv")

    booster = lgb.Booster(model_file=str(art / "model.txt"))
    state = state_from_jsonable(json.loads((art / "feature_state.json").read_text(encoding="utf-8")))
    columns = ["row_id", "time_id", "asset_id", *state.feature_cols]

    def transform(df):
        return transform_feature_v0(df, state)

    row_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for path in iter_test_files(paths["data_root"]):
        # Memory note: still one partition at a time. If a partition OOMs, add
        # row-group / time_id chunking in a follow-up commit inside predict_batch.
        rows, preds = predict_test_partition(path, columns, transform, booster, paths["num_threads"])
        row_parts.append(rows)
        pred_parts.append(preds)
        print(f"done {path.name}: {len(rows)} rows")

    write_public_submission(np.concatenate(row_parts), np.concatenate(pred_parts), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test; smoke-run script on one small fake file if full partitions OOM**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_predict_batch.py -v
# If RAM allows:
python scripts/predict_public_csv.py
# Else: run on online Jupyter after copying artifacts; or extend predict_batch with time_id chunking before full run
```

Expected: unit tests PASS. Full CSV row count must equal `manifest.rows.test` (3217458) before portal upload.

- [ ] **Step 5: Commit**

```bash
git add workspace/src/contest/predict_batch.py workspace/scripts/predict_public_csv.py workspace/tests/test_predict_batch.py
git commit -m "feat: add chunked public-LB CSV prediction path"
```

---

### Task 2: Wire expanding-fold CV into training helpers

**Files:**
- Create: `workspace/src/contest/train_matrix.py`
- Modify: `workspace/src/contest/train_lgb.py` (add multi-fold driver helper if needed)
- Create: `workspace/tests/test_train_matrix.py`

**Interfaces:**
- Consumes: `expanding_folds`, `fit_feature_v0`/`transform_feature_v0` (V1 later)
- Produces: `build_xyw(df, state, target_col="target") -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- Produces: `run_expanding_cv(df, feature_fit_fn, feature_transform_fn, lgb_params, ...) -> dict` with per-fold and mean `valid_wzm_r2`

- [ ] **Step 1: Write failing test that CV returns mean score key**

```python
import numpy as np
import pandas as pd
from contest.features_v0 import fit_feature_v0, transform_feature_v0
from contest.train_matrix import run_expanding_cv


def _toy_frame(n_times=30, n_assets=3):
    rows = []
    rid = 0
    for t in range(n_times):
        for a in range(n_assets):
            rows.append(
                {
                    "row_id": rid,
                    "time_id": t,
                    "asset_id": a,
                    "feature_000": float(t + a),
                    "feature_001": float(a),
                    "weight": 1.0,
                    "target": float(0.01 * t - 0.02 * a),
                }
            )
            rid += 1
    return pd.DataFrame(rows)


def test_run_expanding_cv_returns_mean_score():
    df = _toy_frame()
    result = run_expanding_cv(
        df,
        fit_feature_fn=fit_feature_v0,
        transform_feature_fn=transform_feature_v0,
        params={"objective": "regression", "learning_rate": 0.1, "num_leaves": 8, "verbosity": -1, "min_data_in_leaf": 5},
        n_folds=3,
        embargo=2,
        num_threads=2,
        num_boost_round=30,
        early_stopping_rounds=10,
    )
    assert "mean_valid_wzm_r2" in result
    assert "folds" in result
    assert len(result["folds"]) >= 1
```

- [ ] **Step 2: Run test — expect fail**

```bash
cd workspace
$env:PYTHONPATH="src"
pytest tests/test_train_matrix.py::test_run_expanding_cv_returns_mean_score -v
```

- [ ] **Step 3: Implement `train_matrix.py`**

Implement `build_xyw` and `run_expanding_cv`:
- For each fold from `expanding_folds(df["time_id"], n_folds, embargo)`:
  - split train/valid by time_id sets
  - `state = fit_feature_fn(train_df)` then transform both
  - train with existing `train_lightgbm`
  - evaluate with `evaluate_booster`
- Aggregate `mean_valid_wzm_r2` / `std_valid_wzm_r2`

- [ ] **Step 4: Tests pass + commit**

```bash
pytest tests/test_train_matrix.py -v
git add workspace/src/contest/train_matrix.py workspace/tests/test_train_matrix.py
git commit -m "feat: add expanding-fold CV training helper"
```

---

### Task 3: Memory-safe full / multi-partition training script (V0 features, all folds)

**Files:**
- Create: `workspace/configs/train_lgb_v1.yaml` (even for V0-full first; version field `feature_version: v0` initially)
- Create: `workspace/scripts/train_lgb_v1.py`
- Modify as needed: `workspace/src/contest/io.py` — add optional column projection already supported via `load_partitions(..., columns=)`

**Interfaces:**
- Config keys: `feature_version: v0|v1`, `max_train_partitions: null|int`, `n_folds`, `embargo`, `holdout_fraction` (final holdout after CV), LGB params
- Writes: `workspace/artifacts/lgb_v1/metrics.json`, `model.txt`, `feature_state.json`, `cv_report.json`

- [ ] **Step 1: Add config**

```yaml
feature_version: v0
max_train_partitions: null   # null = all 9 partitions
n_folds: 3
embargo: 5
holdout_fraction: 0.2
num_boost_round: 800
early_stopping_rounds: 50
params:
  objective: regression
  metric: None
  learning_rate: 0.03
  num_leaves: 64
  min_data_in_leaf: 200
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 1
  verbosity: -1
```

- [ ] **Step 2: Implement `train_lgb_v1.py`**

Flow:
1. Load train partitions (all or last N) with needed columns only: index + features + weight + target (+ responders later)
2. Split final holdout via `holdout_time_split`
3. On pre-holdout data, `run_expanding_cv` → log scores
4. Refit on all pre-holdout rows with best_iteration heuristic (mean of fold best iters or max)
5. Evaluate holdout `valid_wzm_r2`
6. Save artifacts under `artifacts/lgb_v1/`

If full 9 partitions OOM: support streaming fold construction by loading partition subsets; document `max_train_partitions` fallback; prefer polars scan when helpful but keep transform APIs on pandas for P1.

- [ ] **Step 3: Run on 2 partitions first, then null**

```bash
cd workspace
python scripts/train_lgb_v1.py
```

Expected: `cv_report.json` + holdout score printed; model saved.

- [ ] **Step 4: Commit**

```bash
git add workspace/configs/train_lgb_v1.yaml workspace/scripts/train_lgb_v1.py
git commit -m "feat: add full-partition LGB training with expanding CV"
```

---

### Task 4: Responder analysis + auxiliary target option

**Files:**
- Create: `workspace/src/contest/responders.py`
- Create: `workspace/scripts/analyze_responders.py`
- Create: `workspace/tests/test_responders.py`
- Modify: `workspace/configs/train_lgb_v1.yaml` — add `auxiliary_responders: []` and `auxiliary_mode: none|pretrain|multitask_weight`

**Interfaces:**
- `weighted_corr(x, y, w) -> float`
- `rank_responders_by_target_corr(df, top_k=10) -> list[tuple[str, float]]`
- Training option A (P1 default): **pretrain** on selected responders then continue on `target` (two-phase with same features)
- Do **not** feed true responders at inference

- [ ] **Step 1: Failing test for ranking**

```python
import numpy as np
import pandas as pd
from contest.responders import rank_responders_by_target_corr


def test_rank_responders_orders_by_abs_corr():
    n = 200
    w = np.ones(n)
    target = np.linspace(-1, 1, n)
    df = pd.DataFrame(
        {
            "target": target,
            "weight": w,
            "responder_00": target + np.random.default_rng(0).normal(0, 0.01, n),
            "responder_01": np.random.default_rng(1).normal(0, 1, n),
        }
    )
    ranked = rank_responders_by_target_corr(df, top_k=2)
    assert ranked[0][0] == "responder_00"
```

- [ ] **Step 2: Implement + analyze script writing `experiments/responder_corr.json`**

- [ ] **Step 3: Integrate optional two-phase train in `train_lgb_v1.py` when `auxiliary_mode: pretrain`**

Phase 1: train to predict top responder (or average of top-k one-by-one — start with single best responder).  
Phase 2: continue boosting / retrain on `target` initialized from phase-1 model **only if** LightGBM continue-train is clean; else simpler: use responder model predictions on train as **extra feature** fit inside train fold only, and at inference use the responder model's prediction as the extra feature (stacked), which requires shipping two boosters in strategy — heavier.

**P1 YAGNI choice (lock this):** use **sample_weight tilting** or **multi-label sequential**: train final model on `target` only, but add a second loss signal by training an auxiliary booster on best responder and blending predictions with validation-learned weight `a * pred_target + (1-a) * pred_resp` where `pred_resp` is produced by an auxiliary model that uses **features only** (no responder inputs). Ship both boosters in `strategy_v1` if blend improves CV; else keep single target model.

Minimal first implementation:
1. Analyze + save top responders
2. Train aux booster on best responder
3. Train main booster on target
4. Grid `a` on validation holdout for blend
5. Save `blend_weight.json`

- [ ] **Step 4: Tests pass + commit**

```bash
git commit -m "feat: add responder correlation analysis and optional aux blend"
```

---

### Task 5: Features V1 (rolling mean / std / diff) with API-reproducible state

**Files:**
- Create: `workspace/src/contest/features_v1.py`
- Create: `workspace/tests/test_features_v1.py`
- Modify: `train_lgb_v1.py`, `predict_public_csv.py`, configs (`feature_version: v1`)

**Interfaces:**
- `FeatureV1State`: V0 fields + `window: int` + `roll_feature_cols: list[str]` (subset of features to roll — start with top importance or first 32/64 to control latency)
- `fit_feature_v1(train_df, window, roll_cols) -> FeatureV1State`
- `transform_feature_v1_frame(df, state) -> np.ndarray` for offline training (group by asset, sort by time, causal rolling)
- `RollingState` class for online: `update_and_featurize(time_id, asset_ids, feature_matrix) -> np.ndarray` using `deque(maxlen=window)` per asset

**Critical:** offline `transform_feature_v1_frame` must match online `RollingState` outputs on a replayed time-ordered stream (write a replay equality test).

- [ ] **Step 1: Write failing replay consistency test**

```python
import numpy as np
import pandas as pd
from contest.features_v1 import RollingState, fit_feature_v1, transform_feature_v1_frame


def test_offline_matches_online_rolling():
    # small synthetic panel; assert max abs diff < 1e-5 between frame transform and per-time_id RollingState
    ...
```

- [ ] **Step 2: Implement features_v1 (window default 5)**

Roll only a configured subset (e.g. 32 cols) to protect 50ms budget. Append: roll_mean, roll_std, diff_1 for each selected col (+ V0 base features + asset_id).

- [ ] **Step 3: Hook into train when `feature_version: v1`**

- [ ] **Step 4: Tests pass + commit**

```bash
git commit -m "feat: add causal rolling V1 features with online/offline parity"
```

---

### Task 6: strategy_v1 package + export + smoke/full gate

**Files:**
- Create: `workspace/submission/strategy_v1/main.py`
- Create: `workspace/scripts/export_strategy_v1.py`
- Create: `workspace/scripts/run_runner_gate_v1.py` (clone v0 gate; point at strategy_v1; default smoke)

**Interfaces:**
- `Model.__init__` loads `model.txt`, optional `aux_model.txt`, `feature_state.json`, `blend_weight.json`
- `Model.predict` updates `RollingState` if V1, blends if configured, returns finite vector
- Must set `num_threads=4`

- [ ] **Step 1: Implement main.py self-contained (duplicate rolling logic inline or minimal vendored copy — no contest imports)**

- [ ] **Step 2: Export artifacts; run smoke gate**

```bash
python scripts/export_strategy_v1.py
python scripts/run_runner_gate_v1.py --mode smoke
```

Expected: `status=ok`; record `mean_predict_seconds`; must stay under 0.05 with margin. If over, reduce `roll_cols` / window / leaves.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: add strategy_v1 with rolling features and runner gate"
```

---

### Task 7: Public CSV for V1 + acceptance + one careful LB submit note

**Files:**
- Modify: `predict_public_csv.py` to accept `--feature-version v0|v1` and V1 state
- Create: `workspace/experiments/p1_acceptance.md`

**Checklist:**
- [ ] Expanding CV mean/holdout scores recorded
- [ ] Responder analysis artifact exists
- [ ] V1 offline/online parity tests green
- [ ] Smoke runner gate OK; latency noted
- [ ] Public CSV row count == 3217458 (generated on capable host/online if needed)
- [ ] Portal upload instructions confirmed by operator (human)

- [ ] **Step 1: Generate public CSV; verify row count**

```bash
python scripts/predict_public_csv.py --artifact-dir artifacts/lgb_v1 --output submission/public/lgb_v1_submission.csv
python -c "import pandas as pd; df=pd.read_csv('submission/public/lgb_v1_submission.csv'); print(len(df), df.columns.tolist())"
```

Expected: `3217458` and `['row_id','target']`.

- [ ] **Step 2: Write `p1_acceptance.md` + commit docs/scripts**

```bash
git commit -m "docs: add P1 acceptance checklist and public CSV V1 path"
```

---

## Public submission runbook (operator)

1. Train/export artifacts (`lgb_v1`).
2. Produce full CSV via `predict_public_csv.py` (online Jupyter if local OOM).
3. Login https://race.xhth.cn → upload CSV.
4. ≤5 successful scores/day; only submit when local CV improves.
5. Keep private `strategy_v1` in sync with the same feature/model semantics.

---

## Self-Review (plan author)

1. **Spec coverage (P1):** V1 rolling, responders, full training + expanding CV, public CSV path, strategy package — covered. XGB/ensemble deferred to P2. Label refill deferred to P3.
2. **Placeholders:** none intentional; OOM fallbacks are explicit (`max_train_partitions`, online Jupyter for CSV).
3. **Consistency:** `FeatureV1State` / `RollingState` / blend files named consistently across train, public predict, and `strategy_v1`.
