"""OpenCV occupancy heatmap + blob detection (the "shape" of people).

Turns the smoothed (x, y, confidence) stream into a real 2D heat image and then
uses OpenCV to clean it up and pull out distinct "hot blobs":

    grid (float) -> normalize -> GaussianBlur -> threshold -> morphology
                 -> findContours -> blobs

Each blob is one presence region:
    * contour  = the heat "shape" (drawn on the room map)
    * centroid = refined (x, y) position
    * blob count = estimated number of people

Honest limits (also documented in README.md):
    * With two ESP32 links we resolve left<->right well, but people standing
      close together along the depth axis are not spatially separable, so the
      blob count is an *estimate* of distinct motion regions, not a hard count.
    * Adding a 3rd/4th ESP32 (more radio links) is what upgrades this from a
      rough 2D picture to true multi-person localization.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class OccupancyEngine:
    """Accumulates positions into a heat grid and detects blobs with OpenCV."""

    def __init__(
        self,
        grid_w: int = 96,
        grid_h: int = 80,
        decay: float = 0.965,
        sigma_cells: float = 2.0,
        kernel_amp: float = 8.0,
        blur_ksize: int = 5,
        threshold: int = 55,
        open_ksize: int = 3,
        close_ksize: int = 5,
        min_area_cells: float = 12.0,
    ) -> None:
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.decay = float(decay)
        self.kernel_amp = float(kernel_amp)
        self.blur_ksize = int(blur_ksize)
        self.threshold = int(threshold)
        self.open_ksize = int(open_ksize)
        self.close_ksize = int(close_ksize)
        self.min_area_cells = float(min_area_cells)

        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Precompute a gaussian stamp (2r+1 x 2r+1) to deposit heat per update.
        r = int(round(3.0 * sigma_cells))
        ax = np.arange(-r, r + 1, dtype=np.float32)
        g = np.exp(-(ax ** 2) / (2.0 * sigma_cells ** 2))
        self._stamp = np.outer(g, g).astype(np.float32)

    # -- heat accumulation -------------------------------------------------
    def update(self, x: float, y: float, weight: float = 1.0) -> None:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return
        cx = int(round(x * (self.grid_w - 1)))
        cy = int(round(y * (self.grid_h - 1)))
        stamp = self._stamp * (self.kernel_amp * max(0.0, float(weight)))
        r = stamp.shape[0] // 2
        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r + 1, cy + r + 1
        sx0, sy0 = 0, 0
        sx1, sy1 = stamp.shape[1], stamp.shape[0]
        if x0 < 0:
            sx0, x0 = -x0, 0
        if y0 < 0:
            sy0, y0 = -y0, 0
        if x1 > self.grid_w:
            sx1 -= (x1 - self.grid_w)
            x1 = self.grid_w
        if y1 > self.grid_h:
            sy1 -= (y1 - self.grid_h)
            y1 = self.grid_h
        if x0 >= x1 or y0 >= y1:
            return
        self.grid[y0:y1, x0:x1] += stamp[sy0:sy1, sx0:sx1]

    def accumulate_map(self, map_img, alpha: float = 2.0) -> None:
        """Add a full 2D map (e.g. an RTI frame) into the heat grid."""
        m = np.asarray(map_img, dtype=np.float32)
        if m.shape != self.grid.shape:
            m = cv2.resize(m, (self.grid_w, self.grid_h), interpolation=cv2.INTER_LINEAR)
        self.grid += float(alpha) * m

    def decay_grid(self, factor: Optional[float] = None) -> None:
        self.grid *= (self.decay if factor is None else float(factor))

    def clear(self) -> None:
        self.grid[:] = 0.0

    def heat_bgr(self) -> np.ndarray:
        """Return a BGR color-map image of the current heat (grid-sized)."""
        maxv = float(self.grid.max())
        if maxv <= 0:
            return np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)
        norm = (np.clip(self.grid / maxv, 0.0, 1.0) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # -- OpenCV blob detection --------------------------------------------
    def detect(self) -> List[Dict[str, Any]]:
        """Return a list of blobs (desc by area): {x, y, area_cells, bbox, contour}."""
        maxv = float(self.grid.max())
        if maxv <= 0:
            return []

        norm = (np.clip(self.grid / maxv, 0.0, 1.0) * 255).astype(np.uint8)

        if self.blur_ksize > 0:
            k = self.blur_ksize | 1  # keep it odd
            norm = cv2.GaussianBlur(norm, (k, k), 0)

        _, th = cv2.threshold(norm, self.threshold, 255, cv2.THRESH_BINARY)

        if self.open_ksize > 0:
            ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_ksize, self.open_ksize))
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, ok)
        if self.close_ksize > 0:
            ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_ksize, self.close_ksize))
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, ck)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs: List[Dict[str, Any]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area_cells:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"] / max(1, self.grid_w - 1)
            cy = m["m01"] / m["m00"] / max(1, self.grid_h - 1)
            bx, by, bw, bh = cv2.boundingRect(c)
            cnorm = [
                [float(p[0][0]) / max(1, self.grid_w - 1),
                 float(p[0][1]) / max(1, self.grid_h - 1)]
                for p in c
            ]
            blobs.append({
                "x": float(np.clip(cx, 0.0, 1.0)),
                "y": float(np.clip(cy, 0.0, 1.0)),
                "area_cells": area,
                "bbox": [int(bx), int(by), int(bw), int(bh)],
                "contour": c,          # int32 (N,1,2) grid-pixel coords
                "contour_norm": cnorm, # normalized [0,1] for drawing at any size
            })
        blobs.sort(key=lambda b: b["area_cells"], reverse=True)
        return blobs
