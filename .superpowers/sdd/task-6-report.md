# Task 6 Report: LightGBM Train Helper + V0 Train Script

## Summary

Implemented the LightGBM training helper, V0 train config, V0 train script, and smoke unit test. The end-to-end train was run with `max_train_partitions: 2` and wrote all expected artifacts under `workspace/artifacts/lgb_v0/`.

## TDD Evidence

### RED

Command:

```powershell
cd D:\exp_all\assign\quant\workspace
$env:PYTHONPATH='src'; pytest tests/test_train_lgb_smoke.py -v
```

Result:

```text
ERROR tests/test_train_lgb_smoke.py
E   ModuleNotFoundError: No module named 'contest.train_lgb'
```

This is the expected missing-module failure from the task brief before production code existed.

### GREEN

Command:

```powershell
cd D:\exp_all\assign\quant\workspace
$env:PYTHONPATH='src'; pytest tests/test_train_lgb_smoke.py -v
```

Result:

```text
tests/test_train_lgb_smoke.py::test_train_lightgbm_smoke PASSED
1 passed in 1.27s
```

## Verification

Command:

```powershell
cd D:\exp_all\assign\quant\workspace
$env:PYTHONPATH='src'; pytest tests -v
```

Result:

```text
11 passed in 1.26s
```

No linter diagnostics were reported for:

- `workspace/src/contest/train_lgb.py`
- `workspace/scripts/train_lgb_v0.py`
- `workspace/tests/test_train_lgb_smoke.py`

## Train Run

Command:

```powershell
cd D:\exp_all\assign\quant\workspace
$env:PYTHONPATH='src'; python scripts/train_lgb_v0.py
```

Result:

```json
{
  "train_rows": 2219816,
  "valid_rows": 565344,
  "partitions_used": [
    "train_partition_007.parquet",
    "train_partition_008.parquet"
  ],
  "best_iteration": 9,
  "train_wzm_r2": 0.004279229377187499,
  "valid_wzm_r2": 0.00032558472504018443
}
```

Confirmed artifacts:

- `workspace/artifacts/lgb_v0/model.txt`
- `workspace/artifacts/lgb_v0/feature_state.json`
- `workspace/artifacts/lgb_v0/metrics.json`

## Notes

- `num_threads` is loaded from `contest.paths.load_paths()` and remains configured as `4` in `workspace/configs/paths.yaml`.
- `workspace/configs/train_lgb_v0.yaml` keeps `max_train_partitions: 2` for the first end-to-end train.
- `contest.train_lgb` uses LightGBM callbacks for early stopping and includes a minimal fallback if `lgb.early_stopping(..., verbose=False)` is unsupported. The installed version accepted the callback during the train run.

## Important Finding Fix: WZM R2 Early Stopping

Updated `workspace/src/contest/train_lgb.py` so `lgb.train` receives a LightGBM-compatible `feval` named `wzm_r2`. The evaluator calls `contest.metrics.weighted_zero_mean_r2` with labels and sample weights from the validation dataset, falling back to unit weights only when LightGBM has no dataset weights. `train_lightgbm` now forces `metric: "None"` before training so LightGBM's default `l2` metric does not drive early stopping, and clamps `num_threads` to at most `4`.

Updated `workspace/configs/train_lgb_v0.yaml` from `metric: l2` to `metric: "None"`.

Regression RED:

```powershell
cd D:\exp_all\assign\quant\workspace
$env:PYTHONPATH='src'; pytest tests/test_train_lgb_smoke.py::test_train_lightgbm_uses_weighted_zero_mean_r2_for_selection -v
```

```text
FAILED tests/test_train_lgb_smoke.py::test_train_lightgbm_uses_weighted_zero_mean_r2_for_selection
E   TypeError: ... fake_train() missing 1 required positional argument: 'feval'
```

Verification GREEN:

```powershell
cd D:\exp_all\assign\quant\workspace
$env:PYTHONPATH='src'; pytest tests/test_train_lgb_smoke.py -v
```

```text
tests/test_train_lgb_smoke.py::test_train_lightgbm_smoke PASSED
tests/test_train_lgb_smoke.py::test_train_lightgbm_uses_weighted_zero_mean_r2_for_selection PASSED
2 passed in 1.16s
```

No linter diagnostics were reported for:

- `workspace/src/contest/train_lgb.py`
- `workspace/tests/test_train_lgb_smoke.py`

The full two-partition training job was not rerun for this targeted fix.

## Fix Evidence: WZM R2 Early Stopping

Addressed review finding that LightGBM model selection still used `l2` rather than the global weighted zero-mean R2 metric.

Changes:

- Added a LightGBM `feval` in `workspace/src/contest/train_lgb.py` returning `("wzm_r2", value, True)` using `contest.metrics.weighted_zero_mean_r2` and validation dataset weights when available.
- Passed `feval` to `lgb.train`, forced LightGBM `metric` to `"None"` so `l2` is not selected for early stopping, and clamped `num_threads` to `<= 4`.
- Updated `workspace/configs/train_lgb_v0.yaml` from `metric: l2` to `metric: "None"`.
- Added a focused unit test confirming `train_lightgbm` passes WZM R2 `feval` and disables the built-in metric.

RED command:

```powershell
cd D:\exp_all\assign\quant\workspace
$env:PYTHONPATH='src'; pytest tests/test_train_lgb_smoke.py::test_train_lightgbm_uses_weighted_zero_mean_r2_for_selection -v
```

RED result:

```text
FAILED tests/test_train_lgb_smoke.py::test_train_lightgbm_uses_weighted_zero_mean_r2_for_selection
E       TypeError: test_train_lightgbm_uses_weighted_zero_mean_r2_for_selection.<locals>.fake_train() missing 1 required positional argument: 'feval'
```

GREEN command:

```powershell
cd D:\exp_all\assign\quant\workspace
$env:PYTHONPATH='src'; pytest tests/test_train_lgb_smoke.py -v
```

GREEN result:

```text
tests/test_train_lgb_smoke.py::test_train_lightgbm_smoke PASSED
tests/test_train_lgb_smoke.py::test_train_lightgbm_uses_weighted_zero_mean_r2_for_selection PASSED
2 passed in 1.16s
```

No linter diagnostics were reported for:

- `workspace/src/contest/train_lgb.py`
- `workspace/tests/test_train_lgb_smoke.py`

The 2-partition training run was not rerun for this fix; the requested focused pytest smoke coverage was rerun successfully.
