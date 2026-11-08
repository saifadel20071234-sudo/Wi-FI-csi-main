"""Synthetic CSI source for the demo mode (no hardware required).

Produces realistic-looking CSI amplitude frames for the configured devices,
simulating an empty room for a short warmup, then a person walking left<->right.
Each device's frame carries the same shape as a real parsed frame (see
csi_reader.parse_csi_line), so the rest of the pipeline is identical in demo and
real modes.

Physics being modelled (simplified):
  * any body movement in the room perturbs BOTH links (a "global" motion term),
  * the device closer to the person is perturbed MORE and sees a bigger RSSI
    drop (a "differential" term) -> this is what lets us estimate left/right.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import numpy as np


class SimulatorSource:
    """Generates frames for N devices at a fixed rate."""

    N_SUBCARRIERS = 64  # classic ESP32 20 MHz -> 64 complex subcarriers

    def __init__(
        self,
        devices: List[Dict[str, Any]],
        rate_hz: float = 15.0,
        warmup_seconds: float = 5.0,
    ) -> None:
        self.devices = devices
        self.rate_hz = rate_hz
        self.warmup_seconds = warmup_seconds
        self._seq = 0
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(12345)
        self._last_emit = 0.0

        self._baselines: Dict[str, np.ndarray] = {}
        for dev in devices:
            self._baselines[dev["id"]] = self._make_baseline()

    def _make_baseline(self) -> np.ndarray:
        x = np.linspace(0.0, 1.0, self.N_SUBCARRIERS)
        base = 38.0 + 6.0 * np.sin(x * math.pi * 2.0 + 0.5)
        base += 4.0 * np.sin(x * math.pi * 5.0 + 1.2)
        # Two frequency-selective fading notches.
        base -= 12.0 * np.exp(-((x - 0.25) ** 2) / 0.002)
        base -= 10.0 * np.exp(-((x - 0.7) ** 2) / 0.003)
        return np.clip(base, 5.0, None).astype(np.float32)

    def tick(self) -> Dict[str, List[Dict[str, Any]]]:
        """Emit one frame per device. Returns {device_id: [frame]}."""
        now = time.monotonic()
        if now - self._last_emit < 1.0 / self.rate_hz:
            return {}
        self._last_emit = now

        t = now - self._t0
        k = np.arange(self.N_SUBCARRIERS, dtype=np.float32)
        walking = t >= self.warmup_seconds

        # Person trajectory (left<->right triangle wave).
        tt = max(0.0, t - self.warmup_seconds)
        period = 18.0
        phase = (tt % period) / period
        x_person = 0.12 + 0.76 * (1.0 - abs(2.0 * phase - 1.0))
        speed = 0.5 + 0.5 * abs(math.sin(phase * math.pi))

        out: Dict[str, List[Dict[str, Any]]] = {}
        for dev in self.devices:
            base = self._baselines[dev["id"]]
            amp = base.copy().astype(np.float32)

            if not walking:
                # Empty room: stable channel + tiny noise.
                amp += self._rng.normal(0.0, 0.4, self.N_SUBCARRIERS).astype(np.float32)
                rssi = int(round(-42.0 + self._rng.normal(0.0, 0.3)))
            else:
                dev_x = dev["x"]
                near = math.exp(-((x_person - dev_x) ** 2) / (2 * 0.15**2))

                # Global motion: any movement perturbs both links.
                global_amp = 6.0 * speed
                # Differential: the nearer device is perturbed more.
                diff_amp = 9.0 * near

                amp += self._rng.normal(0.0, 0.5, self.N_SUBCARRIERS).astype(np.float32)
                amp += global_amp * np.sin(k * 0.4 + tt * 2.5)
                amp += diff_amp * np.sin(k * 0.7 - tt * 3.5)
                amp = np.clip(amp, 1.0, None)

                rssi = int(round(-42.0 - 4.0 * speed - 7.0 * near + self._rng.normal(0.0, 0.5)))

            frame: Dict[str, Any] = {
                "seq": self._seq,
                "mac": f"ROUTER-{dev['id']}",
                "rssi": rssi,
                "channel": 6,
                "n_sub": self.N_SUBCARRIERS,
                "amp": amp,
                "device": dev["id"],
                "ts": now,
                # Expose ground truth only for debugging (not used by the logic).
                "demo_x": x_person,
            }
            out[dev["id"]] = [frame]
        self._seq += 1
        return out
