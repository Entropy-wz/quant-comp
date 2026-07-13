from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class Model:
    def __init__(self):
        root = Path(__file__).resolve().parent
        model_dir = root / "model"
        self.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        state = json.loads((model_dir / "feature_state.json").read_text(encoding="utf-8"))
        self.feature_cols = list(state["feature_cols"])
        self.fill_values = {k: float(v) for k, v in state["fill_values"].items()}
        self.window = int(state.get("window", 1))
        self.roll_feature_cols = list(state.get("roll_feature_cols", []))
        self._roll_index = [self.feature_cols.index(c) for c in self.roll_feature_cols]
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(self.window, 1)))
        self.last_time_id: int | None = None
        self.num_threads = 4

        self.blend_a = 1.0
        self.aux_booster = None
        blend_path = model_dir / "blend_weight.json"
        aux_path = model_dir / "aux_model.txt"
        if blend_path.exists() and aux_path.exists():
            blend = json.loads(blend_path.read_text(encoding="utf-8"))
            self.blend_a = float(blend.get("blend_weight_main", 1.0))
            self.aux_booster = lgb.Booster(model_file=str(aux_path))

    def _base_transform(self, test: pd.DataFrame) -> np.ndarray:
        feats = []
        for col in self.feature_cols:
            if col in test.columns:
                s = pd.to_numeric(test[col], errors="coerce").astype(np.float32)
            else:
                s = pd.Series(np.full(len(test), np.nan, dtype=np.float32))
            fill = np.float32(self.fill_values.get(col, 0.0))
            s = s.fillna(fill).replace([np.inf, -np.inf], fill)
            feats.append(s.to_numpy(dtype=np.float32))
        x = np.column_stack(feats) if feats else np.zeros((len(test), 0), dtype=np.float32)
        asset = test["asset_id"].to_numpy(dtype=np.float32).reshape(-1, 1)
        return np.concatenate([x, asset], axis=1).astype(np.float32, copy=False)

    def _transform(self, test: pd.DataFrame) -> np.ndarray:
        base = self._base_transform(test)
        if not self.roll_feature_cols:
            return base

        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id < self.last_time_id:
            raise ValueError("time_id must be non-decreasing")
        self.last_time_id = time_id

        asset_ids = test["asset_id"].to_numpy()
        rows = []
        for i, asset_id in enumerate(asset_ids):
            current = base[i]
            roll_vals = current[self._roll_index] if self._roll_index else np.zeros(0, dtype=np.float32)
            hist = self.history[int(asset_id)]
            prev = hist[-1] if hist else None
            hist.append(roll_vals.astype(np.float32, copy=True))
            stack = np.vstack(list(hist))
            r_mean = stack.mean(axis=0).astype(np.float32)
            r_std = stack.std(axis=0).astype(np.float32) if len(stack) > 1 else np.zeros_like(r_mean)
            d1 = (roll_vals - prev).astype(np.float32) if prev is not None else np.zeros_like(roll_vals)
            rows.append(np.concatenate([current, r_mean, r_std, d1]))
        return np.vstack(rows).astype(np.float32, copy=False)

    def predict(self, test):
        x = self._transform(test)
        pred = self.booster.predict(x, num_threads=self.num_threads)
        if self.aux_booster is not None and self.blend_a < 1.0:
            aux = self.aux_booster.predict(x, num_threads=self.num_threads)
            pred = self.blend_a * pred + (1.0 - self.blend_a) * aux
        pred = np.asarray(pred, dtype=np.float64)
        return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
