"""Position smoothing with a 2D constant-velocity Kalman filter.

The raw per-frame (x, y) estimate from processor.py jitters. A Kalman filter
tracks position AND velocity, so the on-screen dot follows a moving person
smoothly with very little lag ("live" feel) instead of shaking.

Pure numpy (no OpenCV) so it also works in --no-gui console mode.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np


class KalmanTracker:
    """2D constant-velocity Kalman filter over normalized (x, y) in [0, 1]."""

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-2,
        dt_default: float = 0.05,
    ) -> None:
        self.dt_default = dt_default
        # State: [x, y, vx, vy]
        self.x = np.array([[0.5], [0.5], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 0.5
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self._initialized = False
        self._last_t: Optional[float] = None

    def _F(self, dt: float) -> np.ndarray:
        return np.array([[1.0, 0.0, dt, 0.0],
                         [0.0, 1.0, 0.0, dt],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    def update(self, zx: float, zy: float, now: Optional[float] = None) -> Tuple[float, float]:
        now = time.monotonic() if now is None else now
        dt = self.dt_default
        if self._last_t is not None:
            dt = max(0.001, min(0.5, now - self._last_t))
        self._last_t = now
        F = self._F(dt)

        if not self._initialized:
            self.x[0, 0] = float(zx)
            self.x[1, 0] = float(zy)
            self._initialized = True

        # Predict.
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Update (measurement).
        z = np.array([[float(zx)], [float(zy)]], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return float(np.clip(self.x[0, 0], 0.0, 1.0)), float(np.clip(self.x[1, 0], 0.0, 1.0))

    def rest(self, now: Optional[float] = None) -> Tuple[float, float]:
        """No detection: hold position, zero the velocity (no runaway coasting)."""
        now = time.monotonic() if now is None else now
        dt = self.dt_default
        if self._last_t is not None:
            dt = max(0.001, min(0.5, now - self._last_t))
        self._last_t = now
        F = self._F(dt)
        self.x[2, 0] = 0.0
        self.x[3, 0] = 0.0
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return float(np.clip(self.x[0, 0], 0.0, 1.0)), float(np.clip(self.x[1, 0], 0.0, 1.0))

    @property
    def pos(self) -> Tuple[float, float]:
        return float(np.clip(self.x[0, 0], 0.0, 1.0)), float(np.clip(self.x[1, 0], 0.0, 1.0))
