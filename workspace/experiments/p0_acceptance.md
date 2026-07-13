# P0 Acceptance

Evidence collected on branch `p0-foundation` (HEAD `2fc67c3`). Unit suite re-run for this checklist.

- [x] `pytest` all green under `workspace/tests` — **12/12 passed** (`2026-07-13`, `pytest tests -v`, `PYTHONPATH=src`)
- [x] `eda_summary.py` runs on one train partition — `workspace/experiments/eda_summary_partition0.json` (partition `train_partition_000.parquet`, 1,499,352 rows)
- [x] `train_lgb_v0.py` produces `artifacts/lgb_v0/metrics.json` with finite `valid_wzm_r2` — present locally (not committed; gitignored runtime artifact)
- [x] `export_strategy_v0.py` copies model artifacts — `workspace/submission/strategy_v0/model/model.txt` + `feature_state.json` match `artifacts/lgb_v0/`
- [x] Official runner smoke gate for `strategy_v0` — `workspace/experiments/runner_gate_v0_smoke.json`: `status=ok`, 300 predict calls, 0 timeouts, `mean_predict_seconds=0.0414`
- [ ] Official runner full gate for `strategy_v0` — **FAIL on this host:** `run_runner_gate_v0.py --mode full` raises `MemoryError` loading `test_partition_000.parquet` (~1M rows × 323 features); excerpt in `workspace/experiments/runner_gate_stdout.txt`. Remediation: run full gate on competition dev environment or a RAM-capable host before submission; smoke gate suffices for P0 CI on this machine.
- [x] `mean_predict_seconds` recorded and < 0.05 with margin — **0.0414 s** (smoke gate; threshold 0.05; max single step 0.0697 s, under 0.5 s timeout)
- [x] No future-leakage shortcuts in V0 code review — `fit_feature_v0` / median fills computed on train split only; `holdout_time_split` uses contiguous tail; `expanding_folds` enforces embargo; strategy `Model.predict` uses only current-step columns (no target/weight/responder columns)

## Recorded numbers

- valid_wzm_r2: **0.000326** (2 partitions: `train_partition_007/008`, holdout tail 20%, best_iteration 9)
- mean_predict_seconds: **0.041362** (smoke runner, 300 steps)
- model_init_seconds: **0.857341** (smoke runner)

## Notes

- Training/export/runner artifacts under `workspace/artifacts/` and copied models are local runtime outputs; scripts and tests are in git.
- P1+ follow-ups per plan: `features_v1`, responder targets, full-partition training, ensembles, label refill.
