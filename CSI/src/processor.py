"""Signal processing: presence, motion, left/right position and rough count.

The raw CSI amplitude per subcarrier is turned into a small set of features per
device:

  * activity  - amplitude variance relative to a calibrated "empty room" baseline
                (a person moving near a device makes its channel fluctuate).
  * rssi_drop - how many dB the RSSI fell below baseline (a body blocks /
                absorbs the link).

These two are combined into a per-device "attention score" in [0, 1]. With two
devices placed left/right, the person's horizontal position is estimated by
barycentric interpolation of the device positions weighted by their scores.

Honesty notes (documented in README too):
  * The system detects *motion*; a perfectly still person is an "advanced" task.
  * X (left/right) is a solid estimate; Y (depth) is a coarse approximation.
  * Person *counting* is an experimental heuristic here and needs calibration /
    machine learning to be reliable.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


class CSIProcessor:
    def __init__(self, cfg: Dict[str, Any], devices: List[Dict[str, Any]]) -> None:
        p = cfg.get("processing", {})
        self.window_frames = int(max(3, p.get("window_seconds", 2.0) * 20))
        self.calibration_seconds = float(p.get("calibration_seconds", 4.0))
        self.activity_scale = float(p.get("activity_scale", 0.15))
        self.rssi_weight = float(p.get("rssi_weight", 0.5))
        self.motion_threshold = float(p.get("motion_threshold", 0.55))
        self.smoothing_alpha = float(p.get("smoothing_alpha", 0.30))
        self.count_thresholds = [float(t) for t in p.get("count_thresholds", [0.6, 1.7, 3.0])]
        self.devices = devices

        self._per: Dict[str, Dict[str, Any]] = {}
        for dev in devices:
            self._per[dev["id"]] = {
                "amp_hist": deque(maxlen=self.window_frames),
                "baseline_amp": None,   # mean amplitude vector (calibrated)
                "baseline_rssi": None,  # mean rssi (calibrated)
                "calib_start": time.monotonic(),
                "calib_amp_sum": np.zeros(1, dtype=np.float64),
                "calib_rssi_sum": 0.0,
                "calib_count": 0,
                "last_frame_ts": 0.0,
                "score": 0.0,
                "rssi": None,
            }

        self._x: Optional[float] = None
        self._y: Optional[float] = 0.5
        self._scores: Dict[str, float] = {dev["id"]: 0.0 for dev in devices}
        self._rssi_drop: Dict[str, float] = {dev["id"]: 0.0 for dev in devices}

    # ------------------------------------------------------------------
    def ingest(self, frames_by_device: Dict[str, List[Dict[str, Any]]]) -> None:
        """Feed new frames into per-device history and calibration."""
        for dev_id, frames in frames_by_device.items():
            st = self._per.get(dev_id)
            if st is None:
                continue
            for f in frames:
                amp = f.get("amp")
                if amp is None:
                    continue
                st["amp_hist"].append(np.asarray(amp, dtype=np.float32))
                st["last_frame_ts"] = f.get("ts", time.monotonic())
                rssi = f.get("rssi")
                if rssi is not None:
                    st["rssi"] = float(rssi)
                    if st["baseline_rssi"] is None:
                        st["calib_rssi_sum"] += float(rssi)
                if st["baseline_amp"] is None:
                    st["calib_amp_sum"] = st["calib_amp_sum"] + np.asarray(amp, dtype=np.float64)
                    st["calib_count"] += 1

    def _maybe_finalize(self, now: float) -> None:
        """Freeze the empty-room baseline only after the calibration window."""
        for st in self._per.values():
            if st["baseline_amp"] is None and st["calib_count"] > 0:
                if (now - st["calib_start"]) >= self.calibration_seconds:
                    st["baseline_amp"] = (st["calib_amp_sum"] / st["calib_count"]).astype(np.float32)
            if st["baseline_rssi"] is None and st["calib_count"] > 0:
                if (now - st["calib_start"]) >= self.calibration_seconds:
                    st["baseline_rssi"] = st["calib_rssi_sum"] / st["calib_count"]

    def recalibrate(self) -> None:
        """Restart the empty-room baseline calibration (bind to a key)."""
        for st in self._per.values():
            st["baseline_amp"] = None
            st["baseline_rssi"] = None
            st["calib_start"] = time.monotonic()
            st["calib_amp_sum"] = np.zeros(1, dtype=np.float64)
            st["calib_rssi_sum"] = 0.0
            st["calib_count"] = 0
            st["amp_hist"].clear()
        self._x = None

    def calibrating(self) -> bool:
        now = time.monotonic()
        self._maybe_finalize(now)
        for st in self._per.values():
            if st["baseline_amp"] is None and (now - st["calib_start"]) < self.calibration_seconds:
                return True
        return False

    # ------------------------------------------------------------------
    def process(self, frames_by_device: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Ingest frames and return the current sensing state."""
        self.ingest(frames_by_device)
        now = time.monotonic()
        self._maybe_finalize(now)

        scores: Dict[str, float] = {}
        rssi_drops: Dict[str, float] = {}
        online: List[str] = []
        rssi_map: Dict[str, float] = {}

        for dev_id, st in self._per.items():
            # Offline if no frame in the last 2.5 s.
            if (now - st["last_frame_ts"]) > 2.5:
                scores[dev_id] = 0.0
                rssi_drops[dev_id] = 0.0
                continue
            online.append(dev_id)
            rssi_map[dev_id] = st["rssi"] if st["rssi"] is not None else 0.0

            base_amp = st["baseline_amp"]
            base_rssi = st["baseline_rssi"]

            rel_var = 0.0
            if base_amp is not None and len(st["amp_hist"]) >= 3:
                hist = np.asarray(list(st["amp_hist"]), dtype=np.float32)
                # per-subcarrier variance across the window, then averaged.
                var_per_sc = np.var(hist, axis=0)
                base_mean = float(np.mean(base_amp)) + 1e-6
                rel_var = float(np.mean(var_per_sc)) / base_mean

            rssi_drop = 0.0
            if base_rssi is not None and st["rssi"] is not None:
                rssi_drop = max(0.0, base_rssi - st["rssi"])

            # Normalize to roughly [0,1] with soft saturating forms.
            rel_var_norm = rel_var / (rel_var + self.activity_scale)
            rssi_drop_norm = rssi_drop / (rssi_drop + 10.0)
            score = rel_var_norm + self.rssi_weight * rssi_drop_norm
            # Compress the sum back toward [0,1].
            score = score / (1.0 + self.rssi_weight)
            scores[dev_id] = float(np.clip(score, 0.0, 1.0))
            rssi_drops[dev_id] = rssi_drop

        # Presence: any online device above threshold.
        presence = bool(online) and any(scores[d] >= self.motion_threshold for d in online)

        # Horizontal position: barycentric over devices, weighted by attention.
        if presence and online:
            floor = self.motion_threshold * 0.5
            weights = {d: max(0.0, scores[d] - floor) for d in online}
            total = sum(weights.values())
            if total > 1e-6:
                x_num = 0.0
                for dev in self.devices:
                    if dev["id"] in weights:
                        x_num += weights[dev["id"]] * dev["x"]
                x = x_num / total
                self._x = x if self._x is None else (
                    self.smoothing_alpha * x + (1.0 - self.smoothing_alpha) * self._x
                )
        elif self._x is not None:
            self._x = self.smoothing_alpha * 0.5 + (1.0 - self.smoothing_alpha) * self._x

        x = self._x if self._x is not None else 0.5

        # Depth (coarse): strong overall signal -> person closer to the device
        # line; weak -> farther. Kept gentle and clearly approximate.
        total_drop = sum(rssi_drops.get(d, 0.0) for d in online)
        depth_norm = total_drop / (total_drop + 20.0) if total_drop > 0 else 0.0
        y_target = 0.72 - 0.35 * depth_norm
        if self._y is None:
            self._y = y_target
        else:
            self._y = self.smoothing_alpha * y_target + (1.0 - self.smoothing_alpha) * self._y

        # Experimental rough count from total attention.
        total_score = sum(scores.values())
        count = 0
        if presence:
            count = 1
            for thr in self.count_thresholds:
                if total_score >= thr:
                    count += 1
        count = min(count, 4)

        # Label helpers.
        position = "CENTER"
        if presence:
            if x < 0.36:
                position = "LEFT"
            elif x > 0.64:
                position = "RIGHT"

        self._scores = scores
        self._rssi_drop = rssi_drops

        return {
            "presence": presence,
            "state": "MOVING" if presence else "IDLE",
            "count": count,
            "x": float(x),
            "y": float(self._y),
            "position": position,
            "confidence": float(max(scores.values()) if scores else 0.0),
            "motion_threshold": self.motion_threshold,
            "activity": dict(scores),
            "rssi": dict(rssi_map),
            "rssi_drop": dict(rssi_drops),
            "online": online,
            "calibrating": self.calibrating(),
        }
