"""OpenCV visualizer: room heatmap + moving person dot + HUD.

Top-down view of the room. The person's estimated position is drawn as a moving
dot over an accumulating motion heatmap, with a live HUD (presence, position,
activity per device). Press 'r' to toggle a raw CSI-amplitude view.

NOTE: OpenCV's built-in font does not render Arabic, so the HUD text is kept in
English. The Arabic explanation lives in README.md.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


BG_COLOR = (34, 36, 46)
PANEL_COLOR = (24, 25, 33)
GRID_COLOR = (58, 62, 74)
TEXT_MAIN = (235, 235, 235)
TEXT_DIM = (150, 155, 168)
GREEN = (80, 220, 90)
RED = (70, 90, 240)
CYAN = (220, 200, 60)
ORANGE = (60, 140, 240)
YELLOW = (40, 200, 255)


class Visualizer:
    def __init__(self, cfg: Dict[str, Any], devices: List[Dict[str, Any]]) -> None:
        self.cfg = cfg
        self.devices = devices
        self.room = cfg["room"]
        self.router = cfg["router"]
        vis = cfg.get("visual", {})
        self.width = int(vis.get("width", 1200))
        self.height = int(vis.get("height", 820))
        self.win_name = vis.get("window_name", "WiFi CSI Sensing")
        self.show_raw = bool(vis.get("show_raw", False))

        self.trail: deque = deque(maxlen=60)
        self._fps_times: deque = deque(maxlen=30)
        self._last_frame_ts: Optional[float] = None

        # Layout: room on the left, HUD panel on the right.
        self.room_area_w = int(self.width * 0.72)
        self.hud_x = self.room_area_w
        self.hud_w = self.width - self.hud_x

        # Room pixel rect (preserves aspect ratio).
        aspect = self.room["width_m"] / max(self.room["height_m"], 0.001)
        margin = 46
        avail_w = self.room_area_w - 2 * margin
        avail_h = self.height - 2 * margin
        if avail_w / avail_h > aspect:
            self.room_px_h = avail_h
            self.room_px_w = int(avail_h * aspect)
        else:
            self.room_px_w = avail_w
            self.room_px_h = int(avail_w / aspect)
        self.room_x0 = margin + (avail_w - self.room_px_w) // 2
        self.room_y0 = margin + (avail_h - self.room_px_h) // 2

        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.width, self.height)

    # -- coordinate mapping -------------------------------------------------
    def to_px(self, x: float, y: float) -> Tuple[int, int]:
        px = int(self.room_x0 + x * self.room_px_w)
        py = int(self.room_y0 + y * self.room_px_h)
        return px, py

    def _blit_heat(self, canvas: np.ndarray, heat_bgr: np.ndarray) -> None:
        heat_bgr = cv2.resize(
            heat_bgr, (self.room_px_w, self.room_px_h), interpolation=cv2.INTER_LINEAR
        )
        roi = canvas[self.room_y0 : self.room_y0 + self.room_px_h,
                     self.room_x0 : self.room_x0 + self.room_px_w]
        cv2.addWeighted(heat_bgr, 0.55, roi, 0.45, 0, dst=roi)

    # -- drawing ------------------------------------------------------------
    def _draw_room(self, canvas: np.ndarray, state: Dict[str, Any]) -> None:
        x1 = self.room_x0 + self.room_px_w
        y1 = self.room_y0 + self.room_px_h
        cv2.rectangle(canvas, (self.room_x0, self.room_y0), (x1, y1), GRID_COLOR, 2)

        # Grid.
        for i in range(1, 5):
            gx = self.room_x0 + int(self.room_px_w * i / 5)
            gy = self.room_y0 + int(self.room_px_h * i / 5)
            cv2.line(canvas, (gx, self.room_y0), (gx, y1), GRID_COLOR, 1)
            cv2.line(canvas, (self.room_x0, gy), (x1, gy), GRID_COLOR, 1)

        # Router.
        rx, ry = self.to_px(self.router.get("x", 0.5), self.router.get("y", 0.06))
        cv2.circle(canvas, (rx, ry), 14, ORANGE, -1)
        cv2.putText(canvas, self.router.get("label", "Router"), (rx - 40, ry - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)

        # Devices (ESPs).
        for dev in self.devices:
            dx, dy = self.to_px(dev["x"], dev["y"])
            act = state["activity"].get(dev["id"], 0.0)
            color = GREEN if act >= state.get("motion_threshold", 0.55) else CYAN
            cv2.circle(canvas, (dx, dy), 11, color, 2)
            cv2.circle(canvas, (dx, dy), 4, color, -1)
            label = dev.get("label", dev["id"])
            cv2.putText(canvas, label, (dx - 30, dy + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # Activity ring.
            rad = int(np.clip(act * 30.0, 6, 30))
            cv2.circle(canvas, (dx, dy), rad, color, 1)

    def _draw_trail(self, canvas: np.ndarray) -> None:
        if len(self.trail) < 2:
            return
        pts = list(self.trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = (int(80 * alpha), int(220 * alpha), int(90 * alpha))
            cv2.line(canvas, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    def _draw_person(self, canvas: np.ndarray, state: Dict[str, Any]) -> None:
        if not state["presence"]:
            return
        px, py = self.to_px(state["x"], state["y"])
        self.trail.append((px, py))
        self._draw_trail(canvas)
        cv2.circle(canvas, (px, py), 13, GREEN, 2)
        cv2.circle(canvas, (px, py), 6, GREEN, -1)
        cv2.circle(canvas, (px, py), 3, (255, 255, 255), -1)

    def _draw_blobs(self, canvas: np.ndarray, state: Dict[str, Any]) -> None:
        """Draw OpenCV-detected presence blobs: contour (shape) + centroid label."""
        for i, blob in enumerate(state.get("blobs", []) or []):
            cnorm = blob.get("contour_norm")
            if cnorm:
                scaled = []
                for nx, ny in cnorm:
                    px = self.room_x0 + int(nx * self.room_px_w)
                    py = self.room_y0 + int(ny * self.room_px_h)
                    scaled.append([[px, py]])
                arr = np.array(scaled, dtype=np.int32)
                cv2.drawContours(canvas, [arr], -1, YELLOW, 2)

            bx, by = self.to_px(blob["x"], blob["y"])
            cv2.circle(canvas, (bx, by), 5, YELLOW, -1)
            cv2.putText(canvas, f"P{i + 1}", (bx + 8, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

    def _draw_raw(self, canvas: np.ndarray, frames: Dict[str, List[Dict[str, Any]]]) -> None:
        """Toggle view: last-frame amplitude curve per device."""
        region_h = self.height
        cv2.rectangle(canvas, (self.room_x0, self.room_y0),
                      (self.room_x0 + self.room_px_w, self.room_y0 + self.room_px_h),
                      BG_COLOR, -1)
        n_dev = max(1, len(self.devices))
        slot_h = self.room_px_h // n_dev
        for i, dev in enumerate(self.devices):
            fr = frames.get(dev["id"], [])
            if not fr:
                continue
            amp = fr[-1].get("amp")
            if amp is None:
                continue
            y0 = self.room_y0 + i * slot_h
            y1 = y0 + slot_h
            xs = np.linspace(self.room_x0, self.room_x0 + self.room_px_w, len(amp))
            mn, mx = float(np.min(amp)), float(np.max(amp))
            rng = max(mx - mn, 1e-6)
            ys = y0 + 20 + (1.0 - (amp - mn) / rng) * (slot_h - 40)
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, GREEN, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{dev['id']} amp[{int(mn)}..{int(mx)}]",
                        (self.room_x0 + 8, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv2.LINE_AA)

    def _draw_hud(self, canvas: np.ndarray, state: Dict[str, Any], fps: float) -> None:
        cv2.rectangle(canvas, (self.hud_x, 0), (self.width, self.height), PANEL_COLOR, -1)
        x = self.hud_x + 18
        y = 28
        lh = 24

        def line(text: str, color=TEXT_MAIN, scale: float = 0.52, bold: int = 1):
            nonlocal y
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)
            y += lh

        cv2.putText(canvas, "CSI HUMAN SENSING", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, YELLOW, 2, cv2.LINE_AA)
        y += 34

        if state.get("calibrating"):
            line("CALIBRATING... (empty room)", ORANGE)
            line("keep the room clear", TEXT_DIM, 0.45)
        else:
            pres = "YES" if state["presence"] else "NO"
            line(f"PRESENCE  : {pres}", GREEN if state["presence"] else TEXT_MAIN, 0.62, 2)
            line(f"STATE     : {state['state']}")
            line(f"POSITION  : {state['position']}")
            line(f"X (L-R)   : {state['x']:.2f}")
            line(f"Y (depth) : {state['y']:.2f}  (approx)")
            line(f"PEOPLE    : {state.get('people', state['count'])}  (OpenCV blobs)")
            line(f"CONFIDENCE: {state['confidence']:.2f}")

        y += 14
        cv2.line(canvas, (x, y), (self.width - 16, y), GRID_COLOR, 1)
        y += 14
        line("DEVICES", TEXT_DIM, 0.5)

        for dev in self.devices:
            dev_id = dev["id"]
            act = state["activity"].get(dev_id, 0.0)
            rssi = state["rssi"].get(dev_id)
            online = dev_id in state.get("online", [])
            status = "ON" if online else "OFF"
            col = GREEN if online else RED
            line(f"{dev.get('label', dev_id):12s} [{status}]", col, 0.5)
            # Activity bar.
            bx = x
            by = y
            bw = self.hud_w - 40
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + 10), GRID_COLOR, -1)
            fill = int(np.clip(act, 0.0, 1.0) * bw)
            cv2.rectangle(canvas, (bx, by), (bx + fill, by + 10), GREEN, -1)
            y += lh + 2
            rssi_txt = f"{int(rssi)} dBm" if rssi is not None else "--"
            line(f"     act {act:.2f}   rssi {rssi_txt}", TEXT_DIM, 0.45)
            y += 4

        y += 16
        line(f"FPS: {fps:4.1f}", TEXT_DIM, 0.45)
        y += 8
        line("keys: q quit | r raw | c recal", TEXT_DIM, 0.42)
        line("      h clear heat", TEXT_DIM, 0.42)

    # -- main frame ---------------------------------------------------------
    def render(
        self,
        state: Dict[str, Any],
        frames: Dict[str, List[Dict[str, Any]]],
        heat_bgr: Optional[np.ndarray] = None,
    ) -> int:
        """Draw one frame. Returns the pressed key code (or -1 if none)."""
        canvas = np.full((self.height, self.width, 3), BG_COLOR, dtype=np.uint8)

        # FPS.
        now = time.monotonic()
        self._fps_times.append(now)
        fps = 0.0
        if len(self._fps_times) >= 2:
            span = self._fps_times[-1] - self._fps_times[0]
            fps = (len(self._fps_times) - 1) / span if span > 0 else 0.0

        if self.show_raw:
            self._draw_raw(canvas, frames)
        else:
            if heat_bgr is not None:
                self._blit_heat(canvas, heat_bgr)
            self._draw_room(canvas, state)
            self._draw_person(canvas, state)
            self._draw_blobs(canvas, state)

        self._draw_hud(canvas, state, fps)
        cv2.imshow(self.win_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            self.show_raw = not self.show_raw
        return key

    def close(self) -> None:
        cv2.destroyAllWindows()
