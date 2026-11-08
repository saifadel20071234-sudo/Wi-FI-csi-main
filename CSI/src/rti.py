"""Radio Tomographic Imaging (RTI) — per-link attention -> a 2D map.

With 3+ ESP32 receivers around the room (each one a radio link to the router),
the per-device "attention score" from processor.py (how much that link is being
blocked / perturbed by a body) is back-projected onto the room as a 2D image:

    for each link (device <-> router):
        weight(p) = gaussian of the point-to-segment distance from p to the link
    image(p)   = sum over links of (score_i * weight_i(p))

Where several perturbed links cross, the image gets hot -> that is a person.
OpenCV (OccupancyEngine + blob detection) then reads this image to count people
and find their positions — true 2D localization.

Honest note: 2 links only give a degenerate intersection (two wide bands), so
RTI is used for 3+ devices; with 2 devices the system keeps the simpler
barycentric point model (see main.py).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _point_segment_distance(x: np.ndarray, y: np.ndarray, a, b) -> np.ndarray:
    """Vectorized distance from points (x, y) to segment a->b (coords in 0..1)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby + 1e-9
    t = ((x - ax) * abx + (y - ay) * aby) / denom
    t = np.clip(t, 0.0, 1.0)
    px = ax + t * abx
    py = ay + t * aby
    return np.sqrt((x - px) ** 2 + (y - py) ** 2)


class RTIEngine:
    """Builds a 2D link-map image from per-device attention scores."""

    def __init__(
        self,
        devices: List[Dict[str, Any]],
        router: Dict[str, Any],
        grid_w: int = 64,
        grid_h: int = 80,
        link_width: float = 0.30,
    ) -> None:
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.link_width = float(link_width)
        self.router = (float(router["x"]), float(router["y"]))
        self.devices = devices

        ys, xs = np.mgrid[0:self.grid_h, 0:self.grid_w]
        self._px = (xs / max(1, self.grid_w - 1)).astype(np.float32)
        self._py = (ys / max(1, self.grid_h - 1)).astype(np.float32)

        self._w_maps: List[np.ndarray] = []
        for dev in devices:
            b = (float(dev["x"]), float(dev["y"]))
            d = _point_segment_distance(self._px, self._py, self.router, b)
            w = np.exp(-(d ** 2) / (2.0 * self.link_width ** 2))
            self._w_maps.append(w.astype(np.float32))

    def compute(self, scores: Dict[str, float]) -> np.ndarray:
        """Return the current 2D image given {device_id: attention_score}."""
        img = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for dev, w in zip(self.devices, self._w_maps):
            s = float(scores.get(dev["id"], 0.0))
            if s > 0:
                img += s * w
        return img
