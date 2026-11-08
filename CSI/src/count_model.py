"""People-counting ML: windowed features + trained RandomForest predictor.

processor.py gives a per-frame per-device "attention" score and an RSSI drop.
A short sliding window of those states is turned into a fixed-size feature
vector, and a RandomForest then predicts the number of people (0, 1, 2, ...).

Train it with train_count_model.py. When no model file exists, CountPredictor
reports "unavailable" and main.py falls back to the blob count / heuristic, so
the system still runs without ML.
"""
from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def feature_names(devices: List[Dict[str, Any]]) -> List[str]:
    """Canonical feature column order (matches CountFeatureExtractor.features)."""
    names: List[str] = []
    for dev in devices:
        did = dev["id"]
        names += [f"{did}_act_mean", f"{did}_act_std", f"{did}_act_max", f"{did}_drop_mean"]
    names += ["total_act_mean", "total_act_max", "presence_rate"]
    return names


class CountFeatureExtractor:
    """Builds a fixed-size feature vector from a window of per-frame states."""

    def __init__(
        self,
        devices: List[Dict[str, Any]],
        window_frames: int = 40,
    ) -> None:
        self.devices = devices
        self.window_frames = int(window_frames)
        self._hist: deque = deque(maxlen=self.window_frames)

    def push(self, state: Dict[str, Any]) -> None:
        self._hist.append(state)

    def reset(self) -> None:
        self._hist.clear()

    @property
    def ready(self) -> bool:
        return len(self._hist) >= max(6, self.window_frames // 2)

    def features(self) -> Tuple[List[float], List[str]]:
        vals: List[float] = []
        for dev in self.devices:
            did = dev["id"]
            acts = np.asarray(
                [s.get("activity", {}).get(did, 0.0) for s in self._hist], dtype=float
            )
            drops = np.asarray(
                [s.get("rssi_drop", {}).get(did, 0.0) for s in self._hist], dtype=float
            )
            vals += [
                float(acts.mean()),
                float(acts.std()),
                float(acts.max()),
                float(drops.mean()),
            ]

        totals = np.asarray(
            [sum(s.get("activity", {}).values()) for s in self._hist], dtype=float
        )
        presence_rate = float(
            np.mean([1.0 if s.get("presence") else 0.0 for s in self._hist])
        )
        vals += [float(totals.mean()), float(totals.max()), presence_rate]
        return vals, feature_names(self.devices)


class CountPredictor:
    """Loads a trained count model (or reports unavailable)."""

    def __init__(
        self,
        model_path: str = "count_model.pkl",
        scaler_path: str = "count_scaler.pkl",
    ) -> None:
        self.model = None
        self.scaler = None
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                import joblib

                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
            except Exception:
                self.model = None
                self.scaler = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, feature_vector: List[float]) -> Tuple[Optional[int], float]:
        if self.model is None:
            return None, 0.0
        try:
            X = self.scaler.transform([feature_vector])
            pred = int(self.model.predict(X)[0])
            proba = float(np.max(self.model.predict_proba(X)[0]))
            return pred, proba
        except Exception:
            return None, 0.0
