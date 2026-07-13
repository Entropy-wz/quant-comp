# P1 Acceptance

Branch: `p1-modeling`

## Checklist

- [x] Expanding CV helper + `train_lgb_v1.py` produce `artifacts/lgb_v1/cv_report.json` / `metrics.json`
- [x] Responder analysis: `experiments/responder_corr.json` (top corr ~0.39)
- [x] V1 offline/online parity test green (`tests/test_features_v1.py`)
- [x] `strategy_v1` smoke runner gate OK (`experiments/runner_gate_v1_smoke.json`)
- [ ] Full official runner gate on this host (still RAM-limited; use `--mode full` on capable host)
- [ ] Public CSV row count == 3217458 (generate via `python scripts/predict_public_csv.py`; may need online Jupyter if local OOM / long runtime)

## Recorded numbers (2 train partitions, feature_version=v1)

- cv_mean_valid_wzm_r2: **0.000509**
- holdout_wzm_r2: **-0.000910** (weak / unstable on 2-partition slice — need more data + tuning)
- smoke mean_predict_seconds: **0.0432** (under 0.05; thin margin)
- smoke model_init_seconds: **1.42**

## Public LB runbook

1. Ensure `artifacts/lgb_v1/{model.txt,feature_state.json}` exist
2. `python scripts/predict_public_csv.py --artifact-dir artifacts/lgb_v1 --output submission/public/lgb_v1_submission.csv`
3. Verify `len(df)==3217458` and columns `row_id,target`
4. Upload at https://race.xhth.cn (≤5 successful scores/day)

## Notes

- Prefer Python 3.11 for contest pin parity when available (host used 3.13)
- Next: more partitions / blend mode / latency trim before relying on LB score
