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
