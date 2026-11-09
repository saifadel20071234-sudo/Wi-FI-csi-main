"""Dual-Link 2D WiFi CSI Heatmap Tracker.

Combines TWO ESP32 CSI streams (perpendicular links) to estimate a person's
2D position (X, Y) in a room, shown as a moving dot on a live heatmap.

Architecture:
  Link 1 (ESP32 #1 ↔ AP1): Horizontal beam → signal drop encodes X position
  Link 2 (ESP32 #2 ↔ AP2): Vertical beam   → signal drop encodes Y position

  score1 ─┐
           ├─→ DualLinkFusion → (x, y) → KalmanTracker → smooth pos → heatmap
  score2 ─┘

Usage:
  python src/dual_link_tracker.py --demo          # Simulated 2D movement
  python src/dual_link_tracker.py                 # Real hardware (2× ESP32)
  python src/dual_link_tracker.py --port1 COM8 --port2 COM9
  python src/dual_link_tracker.py --no-gui        # Console mode

Controls (GUI window):
  [C]       Recalibrate baseline (keep room empty for 3s)
  [Z] / [R] Reset heatmap
  [+] / [-] Adjust detection threshold
  [Q] / Esc Exit

This file is INDEPENDENT of scope_monitor.py — it does NOT modify or import it.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import list_serial_ports, load_config
from csi_reader import CSIReader
from kalman import KalmanTracker


# =============================================================================
# Color Palette (BGR for OpenCV)
# =============================================================================
BG_COLOR        = (22, 24, 32)
PANEL_COLOR     = (28, 31, 42)
PANEL_BORDER    = (48, 53, 70)
GRID_COLOR      = (38, 42, 54)

TEXT_WHITE  = (245, 248, 252)
TEXT_MUTED  = (145, 153, 168)
TEXT_DARK   = (75, 80, 95)

COLOR_GREEN     = (70, 215, 95)
COLOR_GREEN_DK  = (20, 60, 28)
COLOR_RED       = (55, 70, 245)
COLOR_RED_DK    = (25, 30, 100)
COLOR_YELLOW    = (45, 205, 255)
COLOR_CYAN      = (230, 210, 50)
COLOR_ORANGE    = (40, 140, 255)
COLOR_PURPLE    = (220, 90, 200)
COLOR_BLUE      = (220, 140, 40)
COLOR_TEAL      = (160, 220, 80)


# =============================================================================
# Single-Link Signal Engine (re-implemented inline, no scope_monitor import)
# =============================================================================
class LinkEngine:
    """Processes one ESP32 CSI stream into an activity score [0–100%].

    Tracks:
      • current_score  — smoothed activity level
      • current_rssi   — latest RSSI
      • current_rssi_drop — drop from baseline
      • calibrating    — True during first 3 s (empty room)
    """

    def __init__(
        self,
        link_id: str,
        window_frames: int = 20,
        calibration_seconds: float = 3.0,
        threshold: float = 35.0,
        activity_scale: float = 0.15,
    ) -> None:
        self.link_id = link_id
        self.window_frames = window_frames
        self.calibration_seconds = calibration_seconds
        self.threshold = float(threshold)
        self.activity_scale = float(activity_scale)

        self._amp_hist: deque = deque(maxlen=window_frames)
        self._rssi_hist: deque = deque(maxlen=window_frames)

        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_rssi: Optional[float] = None
        self.baseline_var: Optional[float] = None
        self.calib_start: float = time.monotonic()
        self._calib_amps: List[np.ndarray] = []
        self._calib_rssis: List[float] = []

        self.current_score: float = 0.0
        self.raw_variance: float = 0.0
        self.current_rssi: float = -50.0
        self.current_rssi_drop: float = 0.0
        self.frame_count: int = 0
        self.last_ts: float = 0.0

        # Temporal EMA features
        self._var_ema_fast: float = 0.0
        self._var_ema_slow: float = 0.0
        self.var_ema_diff: float = 0.0

        # In-beam flag
        self.in_beam: bool = False

    @property
    def calibrating(self) -> bool:
        return self.baseline_mean is None

    @property
    def calib_progress(self) -> float:
        if not self.calibrating:
            return 1.0
        elapsed = time.monotonic() - self.calib_start
        return min(1.0, elapsed / max(0.1, self.calibration_seconds))

    @property
    def online(self) -> bool:
        """True if we received a frame in the last 2.5 s."""
        return (time.monotonic() - self.last_ts) < 2.5

    def recalibrate(self) -> None:
        self.baseline_mean = None
        self.baseline_rssi = None
        self.baseline_var = None
        self.calib_start = time.monotonic()
        self._calib_amps.clear()
        self._calib_rssis.clear()
        self._amp_hist.clear()
        self._var_ema_fast = 0.0
        self._var_ema_slow = 0.0
        self.var_ema_diff = 0.0
        self.current_score = 0.0
        self.in_beam = False

    def update(self, frame: Dict[str, Any]) -> float:
        amp = frame.get("amp")
        if amp is None or len(amp) == 0:
            return self.current_score

        amp = np.asarray(amp, dtype=np.float32)
        rssi = float(frame.get("rssi", -50.0))
        self.current_rssi = rssi
        now = time.monotonic()
        self.last_ts = now
        self.frame_count += 1

        self._amp_hist.append(amp)
        self._rssi_hist.append(rssi)

        # 1) Calibration phase
        if self.calibrating:
            self._calib_amps.append(amp)
            self._calib_rssis.append(rssi)
            elapsed = now - self.calib_start
            if elapsed >= self.calibration_seconds and len(self._calib_amps) >= 8:
                amps_arr = np.asarray(self._calib_amps, dtype=np.float32)
                self.baseline_mean = np.mean(amps_arr, axis=0)
                self.baseline_rssi = float(np.mean(self._calib_rssis))
                base_vars = np.var(amps_arr, axis=0)
                base_mean = float(np.mean(self.baseline_mean)) + 1e-6
                self.baseline_var = float(np.mean(base_vars)) / base_mean
            return 0.0

        # 2) Score computation
        if len(self._amp_hist) < 3:
            return self.current_score

        stack = np.asarray(list(self._amp_hist), dtype=np.float32)
        var_per_sc = np.var(stack, axis=0)
        base_mean = float(np.mean(self.baseline_mean)) + 1e-6
        rel_var = float(np.mean(var_per_sc)) / base_mean
        self.raw_variance = rel_var

        # Online EMA (fast α=0.5 ≈ span-3; slow α=0.18 ≈ span-10)
        if self._var_ema_fast == 0.0:
            self._var_ema_fast = rel_var
            self._var_ema_slow = rel_var
        else:
            self._var_ema_fast = 0.50 * rel_var + 0.50 * self._var_ema_fast
            self._var_ema_slow = 0.18 * rel_var + 0.82 * self._var_ema_slow
        self.var_ema_diff = max(0.0, self._var_ema_fast - self._var_ema_slow)

        # Subtract baseline noise floor so resting score drops to ~0%
        base_noise = self.baseline_var if self.baseline_var is not None else 0.05
        net_var = max(0.0, rel_var - (base_noise * 1.15))
        rel_var_norm = net_var / (net_var + self.activity_scale)

        raw_drop = max(0.0, (self.baseline_rssi or 0.0) - rssi)
        net_drop = max(0.0, raw_drop - 1.5)
        self.current_rssi_drop = raw_drop
        rssi_drop_norm = net_drop / (net_drop + 7.0) if net_drop > 0 else 0.0

        instant_pct = float(np.clip(
            ((rel_var_norm + 0.65 * rssi_drop_norm) / 1.65) * 100.0, 0.0, 100.0
        ))

        alpha = 0.50 if instant_pct > self.current_score else 0.18
        self.current_score = alpha * instant_pct + (1.0 - alpha) * self.current_score

        # In-beam with hysteresis
        thresh = self.threshold
        hysteresis_low = max(6.0, thresh - 14.0)
        if self.current_score >= thresh:
            self.in_beam = True
        elif self.current_score < hysteresis_low:
            self.in_beam = False

        return self.current_score


# =============================================================================
# 2D Position Fusion from Two Perpendicular Links
# =============================================================================
class DualLinkFusion:
    """Fuses Link 1 and Link 2 measurements into full 2D (x, y) coordinates.

    Geometric model:
      Link 1 (Horizontal Beam, left-to-right at y ≈ 0.35):
        • ESP1 is on the left (x=0), AP1 is on the right (x=1).
        • When disturbed, person is near Link 1's horizontal corridor.
        • Relative RSSI drop increases as person approaches ESP1 (x → 0).
      
      Link 2 (Vertical Beam, top-to-bottom at x ≈ 0.65):
        • AP2 is at the top (y=0), ESP2 is at the bottom (y=1).
        • When disturbed, person is near Link 2's vertical corridor.
        • Relative RSSI drop increases as person approaches ESP2 (y → 1).

    When BOTH links are online:
      • X coordinate is derived from Link 1 attenuation gradient + Link 2 beam proximity.
      • Y coordinate is derived from Link 2 attenuation gradient + Link 1 beam proximity.
      • True 2D coverage across the full room area [0.05 .. 0.95].

    When only ONE link is online (Single-Link Fallback):
      • X is derived from Link 1 RSSI gradient, Y from Link 1 beam distance.
      • System never hangs or blocks calibration.
    """

    BEAM1_Y = 0.35  # Horizontal beam position along Y
    BEAM2_X = 0.65  # Vertical beam position along X

    def __init__(self, smooth_alpha: float = 0.35) -> None:
        self.smooth_alpha = smooth_alpha
        self._x: float = 0.50
        self._y: float = 0.50
        self._presence_score: float = 0.0

    def fuse(
        self,
        score1: float,
        score2: float,
        rssi_drop1: float,
        rssi_drop2: float,
        thresh: float = 35.0,
        has_link2: bool = True,
    ) -> Tuple[float, float, float, bool]:
        """Return (x, y, presence_score, presence_bool)."""
        presence1 = score1 >= thresh
        presence2 = (score2 >= thresh) if has_link2 else False
        presence = presence1 or presence2

        presence_score = max(score1, score2 if has_link2 else 0.0)

        if not presence and score1 < 15.0 and (score2 < 15.0 or not has_link2):
            return self._x, self._y, 0.0, False

        # --- DUAL-LINK 2D FUSION (Router + 2 ESPs) ---
        if has_link2:
            # X Axis (Left <--> Right across room):
            # ESP1 is on the Left, ESP2 is on the Right.
            # Difference in activity directly spans the room width from 0.10 to 0.90:
            tot_score = max(1.0, score1 + score2)
            balance = (score2 - score1) / tot_score  # -1.0 (far left) to +1.0 (far right)
            
            target_x = 0.50 + balance * 0.44
            target_x = float(np.clip(target_x, 0.08, 0.92))

            # Y Axis (Front / Router <--> Back / ESPs):
            # Depth spans from 0.18 (front near router) to 0.86 (back near ESPs):
            avg_drop = (rssi_drop1 + rssi_drop2) / 2.0
            avg_score = (score1 + score2) / 2.0
            
            drop_norm = min(1.0, avg_drop / 8.0)
            score_norm = min(1.0, avg_score / 75.0)
            depth_factor = 0.5 * drop_norm + 0.5 * score_norm
            
            target_y = 0.18 + depth_factor * 0.68
            target_y = float(np.clip(target_y, 0.08, 0.92))

        # --- SINGLE-LINK FALLBACK (Link 1 only) ---
        else:
            d1_norm = min(1.0, rssi_drop1 / 10.0) if rssi_drop1 > 0 else 0.0
            target_x = 0.85 - 0.70 * d1_norm
            score_factor = (score1 / 100.0) - 0.5
            target_y = 0.50 + 0.35 * score_factor
            target_x = float(np.clip(target_x, 0.08, 0.92))
            target_y = float(np.clip(target_y, 0.08, 0.92))

        if presence:
            # When moving: smoothly track person's steps with broad span
            alpha = self.smooth_alpha
            self._x = alpha * target_x + (1.0 - alpha) * self._x
            self._y = alpha * target_y + (1.0 - alpha) * self._y
        else:
            # When stationary: Hold position rock-solid! Do NOT drift!
            pass

        return (
            float(np.clip(self._x, 0.05, 0.95)),
            float(np.clip(self._y, 0.05, 0.95)),
            float(presence_score),
            presence,
        )

    def reset(self) -> None:
        self._x = 0.5
        self._y = 0.5


# =============================================================================
# Heatmap Accumulator
# =============================================================================
class HeatmapEngine:
    """Accumulates a 2D Gaussian-blurred heatmap of person detections.

    Each time the person is detected at (x, y), a Gaussian blob is added.
    Over time, the heatmap shows where the person spends the most time.
    The heatmap slowly decays so old data fades out naturally.
    """

    def __init__(
        self,
        grid_w: int = 100,
        grid_h: int = 80,
        decay: float = 0.978,
        blob_sigma_px: int = 8,
        intensity: float = 18.0,
    ) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.decay = decay
        self.blob_sigma_px = blob_sigma_px
        self.intensity = intensity
        self._grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    def deposit(self, x: float, y: float, strength: float = 1.0) -> None:
        """Deposit a Gaussian blob at normalized (x, y)."""
        px = int(np.clip(x * self.grid_w, 0, self.grid_w - 1))
        py = int(np.clip(y * self.grid_h, 0, self.grid_h - 1))

        # Create a small Gaussian kernel
        ksize = self.blob_sigma_px * 4 + 1
        g = cv2.getGaussianKernel(ksize, self.blob_sigma_px)
        kernel = g @ g.T * self.intensity * strength

        # Paste into grid (clipped to bounds)
        half = ksize // 2
        x0 = max(0, px - half)
        x1 = min(self.grid_w, px + half + 1)
        y0 = max(0, py - half)
        y1 = min(self.grid_h, py + half + 1)
        kx0 = x0 - (px - half)
        ky0 = y0 - (py - half)
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)

        self._grid[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]

    def decay_step(self) -> None:
        self._grid *= self.decay
        np.clip(self._grid, 0.0, None, out=self._grid)

    def clear(self) -> None:
        self._grid[:] = 0.0

    def to_bgr(self, out_w: int, out_h: int) -> np.ndarray:
        """Return a colorized heatmap as a BGR uint8 image."""
        g = self._grid.copy()
        peak = float(np.max(g))
        if peak > 1e-3:
            g /= peak
        g_uint8 = (np.clip(g, 0.0, 1.0) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(g_uint8, cv2.COLORMAP_TURBO)
        # Make empty areas transparent (black → near-black)
        mask = g_uint8 < 15
        colored[mask] = (8, 9, 12)
        resized = cv2.resize(colored, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return resized


# =============================================================================
# 2D Tracker UI (OpenCV)
# =============================================================================
class TrackerUI:
    """Full-screen 2D heatmap + person dot visualizer for dual-link tracking."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        room_w_m: float = 5.0,
        room_h_m: float = 4.0,
        title: str = "WiFi CSI 2D Tracker",
    ) -> None:
        self.width = width
        self.height = height
        self.title = title

        # Room panel (left 68%)
        self.room_panel_w = int(width * 0.68)
        margin = 50
        aspect = room_w_m / max(room_h_m, 0.001)
        avail_w = self.room_panel_w - 2 * margin
        avail_h = height - 2 * margin - 60  # leave room for header
        if avail_w / avail_h > aspect:
            rh = avail_h
            rw = int(avail_h * aspect)
        else:
            rw = avail_w
            rh = int(avail_w / aspect)
        self.room_x0 = margin + (avail_w - rw) // 2
        self.room_y0 = margin + 55
        self.room_w = rw
        self.room_h = rh

        # Right HUD panel
        self.hud_x = self.room_panel_w
        self.hud_w = width - self.hud_x

        # Trail
        self.trail: deque = deque(maxlen=80)
        self._fps_times: deque = deque(maxlen=30)

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, width, height)

    def to_px(self, x: float, y: float) -> Tuple[int, int]:
        px = int(self.room_x0 + np.clip(x, 0, 1) * self.room_w)
        py = int(self.room_y0 + np.clip(y, 0, 1) * self.room_h)
        return px, py

    # -------------------------------------------------------------------------
    def render(
        self,
        eng1: LinkEngine,
        eng2: LinkEngine,
        fusion: DualLinkFusion,
        kalman: KalmanTracker,
        heatmap: HeatmapEngine,
        x: float,
        y: float,
        presence: bool,
        threshold: float,
        port1: str,
        port2: str,
        has_link2: bool = True,
    ) -> int:
        canvas = np.full((self.height, self.width, 3), BG_COLOR, dtype=np.uint8)

        # FPS
        now = time.monotonic()
        self._fps_times.append(now)
        fps = 0.0
        if len(self._fps_times) >= 2:
            span = self._fps_times[-1] - self._fps_times[0]
            fps = (len(self._fps_times) - 1) / span if span > 0 else 0.0

        # ----- HEADER -----
        cv2.putText(canvas, "WIFI CSI 2D HEATMAP TRACKER", (45, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, TEXT_WHITE, 1, cv2.LINE_AA)
        sub_title = "Dual-Link Perpendicular Sensing | 2D Position + Heatmap" if has_link2 else "Single-Link Sensing Mode | 2D Position + Heatmap"
        cv2.putText(canvas, sub_title, (45, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_MUTED, 1, cv2.LINE_AA)

        # Calibration / status banner
        if has_link2:
            both_calibrating = eng1.calibrating or eng2.calibrating
            avg_prog = int(((eng1.calib_progress + eng2.calib_progress) / 2) * 100)
        else:
            both_calibrating = eng1.calibrating
            avg_prog = int(eng1.calib_progress * 100)

        bw, bh = 360, 48
        bx = self.width - bw - 40
        by = 18

        if both_calibrating:
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), PANEL_COLOR, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), COLOR_YELLOW, 2)
            cv2.putText(canvas, f"CALIBRATING ({avg_prog}%)...", (bx + 18, by + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.68, COLOR_YELLOW, 1, cv2.LINE_AA)
        elif presence:
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), COLOR_RED_DK, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), COLOR_RED, 2)
            cv2.putText(canvas, "[!] PERSON DETECTED", (bx + 18, by + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (bx + bw - 22, by + 24), 9, COLOR_RED, -1)
        else:
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), COLOR_GREEN_DK, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), COLOR_GREEN, 1)
            cv2.putText(canvas, "[OK] ROOM CLEAR (IDLE)", (bx + 18, by + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, COLOR_GREEN, 1, cv2.LINE_AA)

        # ----- ROOM BOX -----
        rx1 = self.room_x0 + self.room_w
        ry1 = self.room_y0 + self.room_h
        cv2.rectangle(canvas, (self.room_x0, self.room_y0), (rx1, ry1), GRID_COLOR, 2)

        # Grid lines (5×5)
        for i in range(1, 5):
            gx = self.room_x0 + int(self.room_w * i / 5)
            gy = self.room_y0 + int(self.room_h * i / 5)
            cv2.line(canvas, (gx, self.room_y0), (gx, ry1), GRID_COLOR, 1)
            cv2.line(canvas, (self.room_x0, gy), (rx1, gy), GRID_COLOR, 1)

        # Heatmap overlay
        heat_bgr = heatmap.to_bgr(self.room_w, self.room_h)
        roi = canvas[self.room_y0:ry1, self.room_x0:rx1]
        cv2.addWeighted(heat_bgr, 0.60, roi, 0.40, 0, dst=roi)

        # Link beam visualizations
        # Link 1: horizontal beam (left ↔ right across room at y=0.35)
        y_b1_px = int(self.room_y0 + 0.35 * self.room_h)
        cv2.line(canvas, (self.room_x0, y_b1_px), (rx1, y_b1_px),
                 COLOR_BLUE if eng1.in_beam else (40, 55, 80), 1)
        cv2.circle(canvas, (self.room_x0, y_b1_px), 10, COLOR_BLUE, 2)
        cv2.putText(canvas, "ESP1", (self.room_x0 + 5, y_b1_px - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_BLUE, 1, cv2.LINE_AA)
        cv2.circle(canvas, (rx1, y_b1_px), 8, COLOR_ORANGE, 2)
        cv2.putText(canvas, "AP1", (rx1 - 32, y_b1_px - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ORANGE, 1, cv2.LINE_AA)

        # Link 2: vertical beam (top ↔ bottom across room at x=0.65)
        if has_link2:
            x_b2_px = int(self.room_x0 + 0.65 * self.room_w)
            cv2.line(canvas, (x_b2_px, self.room_y0), (x_b2_px, ry1),
                     COLOR_TEAL if eng2.in_beam else (30, 60, 55), 1)
            cv2.circle(canvas, (x_b2_px, self.room_y0), 8, COLOR_ORANGE, 2)
            cv2.putText(canvas, "AP2", (x_b2_px + 5, self.room_y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ORANGE, 1, cv2.LINE_AA)
            cv2.circle(canvas, (x_b2_px, ry1), 10, COLOR_TEAL, 2)
            cv2.putText(canvas, "ESP2", (x_b2_px + 5, ry1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEAL, 1, cv2.LINE_AA)

        # Beam activity labels
        s1_pct = int(eng1.current_score)
        s1_col = COLOR_RED if eng1.in_beam else TEXT_MUTED
        cv2.putText(canvas, f"Link1: {s1_pct}%", (self.room_x0 + 4, self.room_y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, s1_col, 1, cv2.LINE_AA)
        if has_link2:
            s2_pct = int(eng2.current_score)
            s2_col = COLOR_RED if eng2.in_beam else TEXT_MUTED
            cv2.putText(canvas, f"Link2: {s2_pct}%", (self.room_x0 + 4, self.room_y0 + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, s2_col, 1, cv2.LINE_AA)

        # Person trail
        if len(self.trail) >= 2:
            pts = list(self.trail)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                col = (int(60 * alpha), int(190 * alpha), int(80 * alpha))
                cv2.line(canvas, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)

        # Person dot (Single Person Follow)
        if presence:
            pdx, pdy = self.to_px(x, y)
            self.trail.append((pdx, pdy))

            # Outer glow
            for r, a in [(22, 0.15), (16, 0.25), (11, 0.5)]:
                overlay = canvas.copy()
                cv2.circle(overlay, (pdx, pdy), r, COLOR_GREEN, -1)
                cv2.addWeighted(overlay, a, canvas, 1 - a, 0, canvas)

            cv2.circle(canvas, (pdx, pdy), 10, COLOR_GREEN, 2, cv2.LINE_AA)
            cv2.circle(canvas, (pdx, pdy), 5, COLOR_GREEN, -1, cv2.LINE_AA)
            cv2.circle(canvas, (pdx, pdy), 2, (255, 255, 255), -1)

            # Crosshairs
            cv2.line(canvas, (self.room_x0, pdy), (rx1, pdy), (50, 100, 55), 1)
            cv2.line(canvas, (pdx, self.room_y0), (pdx, ry1), (50, 100, 55), 1)

            # Coordinate label
            cv2.putText(canvas, f"({x:.2f}, {y:.2f})", (pdx + 14, pdy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_WHITE, 1, cv2.LINE_AA)

        # Room label
        cv2.putText(canvas, "ROOM (top-down view)", (self.room_x0 + 2, self.room_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DARK, 1, cv2.LINE_AA)

        # ----- RIGHT HUD PANEL -----
        cv2.rectangle(canvas, (self.hud_x, 0), (self.width, self.height), PANEL_COLOR, -1)
        cv2.line(canvas, (self.hud_x, 0), (self.hud_x, self.height), PANEL_BORDER, 2)

        hx = self.hud_x + 18
        hy = 30
        lh = 24

        def hline(text: str, color=TEXT_WHITE, scale: float = 0.50, bold: int = 1) -> None:
            nonlocal hy
            cv2.putText(canvas, text, (hx, hy), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, bold, cv2.LINE_AA)
            hy += lh

        cv2.putText(canvas, "2D TRACKER HUD", (hx, hy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, COLOR_YELLOW, 2, cv2.LINE_AA)
        hy += 32

        hline(f"PRESENCE : {'YES' if presence else 'NO'}", COLOR_GREEN if presence else TEXT_MUTED, bold=2)
        hline(f"X pos    : {x:.3f}", COLOR_GREEN if presence else TEXT_MUTED)
        hline(f"Y pos    : {y:.3f}", COLOR_GREEN if presence else TEXT_MUTED)
        hy += 8

        # Divider
        cv2.line(canvas, (hx, hy), (self.width - 10, hy), PANEL_BORDER, 1)
        hy += 14

        # Link 1 status
        hline("LINK 1 (Horizontal / X)", COLOR_BLUE, 0.46)
        l1_col = COLOR_RED if eng1.in_beam else (TEXT_MUTED if eng1.online else (60, 60, 80))
        hline(f"  Score : {eng1.current_score:5.1f}%", l1_col)
        hline(f"  RSSI  : {eng1.current_rssi:5.0f} dBm", TEXT_MUTED, 0.44)
        hline(f"  Drop  : {eng1.current_rssi_drop:4.1f} dB", TEXT_MUTED, 0.44)
        hline(f"  Port  : {port1}", TEXT_DARK, 0.40)
        hline(f"  Status: {'ONLINE' if eng1.online else '-- OFFLINE --'}", COLOR_GREEN if eng1.online else COLOR_RED, 0.42)

        # Link 1 activity bar
        bw_ = self.hud_w - 36
        cv2.rectangle(canvas, (hx, hy), (hx + bw_, hy + 10), GRID_COLOR, -1)
        fill1 = int(np.clip(eng1.current_score / 100.0, 0, 1) * bw_)
        bar1_col = COLOR_RED if eng1.in_beam else COLOR_BLUE
        cv2.rectangle(canvas, (hx, hy), (hx + fill1, hy + 10), bar1_col, -1)
        tmark = int((threshold / 100.0) * bw_)
        cv2.line(canvas, (hx + tmark, hy - 3), (hx + tmark, hy + 13), COLOR_YELLOW, 2)
        hy += 20

        hy += 8
        cv2.line(canvas, (hx, hy), (self.width - 10, hy), PANEL_BORDER, 1)
        hy += 14

        # Link 2 status
        hline("LINK 2 (Vertical / Y)", COLOR_TEAL, 0.46)
        if has_link2:
            l2_col = COLOR_RED if eng2.in_beam else (TEXT_MUTED if eng2.online else (60, 60, 80))
            hline(f"  Score : {eng2.current_score:5.1f}%", l2_col)
            hline(f"  RSSI  : {eng2.current_rssi:5.0f} dBm", TEXT_MUTED, 0.44)
            hline(f"  Drop  : {eng2.current_rssi_drop:4.1f} dB", TEXT_MUTED, 0.44)
            hline(f"  Port  : {port2}", TEXT_DARK, 0.40)
            hline(f"  Status: {'ONLINE' if eng2.online else '-- OFFLINE --'}", COLOR_GREEN if eng2.online else COLOR_RED, 0.42)
            cv2.rectangle(canvas, (hx, hy), (hx + bw_, hy + 10), GRID_COLOR, -1)
            fill2 = int(np.clip(eng2.current_score / 100.0, 0, 1) * bw_)
            bar2_col = COLOR_RED if eng2.in_beam else COLOR_TEAL
            cv2.rectangle(canvas, (hx, hy), (hx + fill2, hy + 10), bar2_col, -1)
            cv2.line(canvas, (hx + tmark, hy - 3), (hx + tmark, hy + 13), COLOR_YELLOW, 2)
        else:
            hline("  Score : --", TEXT_DARK)
            hline("  RSSI  : --", TEXT_DARK, 0.44)
            hline("  Port  : Not Connected", TEXT_DARK, 0.40)
            hline("  Status: [STANDBY - SINGLE LINK]", COLOR_YELLOW, 0.42)
            cv2.rectangle(canvas, (hx, hy), (hx + bw_, hy + 10), GRID_COLOR, -1)
        hy += 20

        hy += 10
        cv2.line(canvas, (hx, hy), (self.width - 10, hy), PANEL_BORDER, 1)
        hy += 14

        # System telemetry
        hline(f"Threshold: {threshold:.1f}%  ([+]/[-])", COLOR_YELLOW, 0.44)
        hline(f"FPS: {fps:.1f}", TEXT_MUTED, 0.44)

        hy += 10
        hline("[C] Recalibrate  [Z] Clear heat", TEXT_DARK, 0.40)
        hline("[+]/[-] Thresh   [Q] Quit", TEXT_DARK, 0.40)

        # Footer
        footer_y = self.height - 20
        cv2.putText(canvas, "[C] Recalib  [Z] Clear  [+/-] Thresh  [Q/Esc] Quit",
                    (45, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_DARK, 1, cv2.LINE_AA)

        cv2.imshow(self.title, canvas)
        return cv2.waitKey(1) & 0xFF
        hy += 14

        # Link 1 status
        hline("LINK 1 (Horizontal / X)", COLOR_BLUE, 0.46)
        l1_col = COLOR_RED if eng1.in_beam else (TEXT_MUTED if eng1.online else (60, 60, 80))
        hline(f"  Score : {eng1.current_score:5.1f}%", l1_col)
        hline(f"  RSSI  : {eng1.current_rssi:5.0f} dBm", TEXT_MUTED, 0.44)
        hline(f"  Drop  : {eng1.current_rssi_drop:4.1f} dB", TEXT_MUTED, 0.44)
        hline(f"  Port  : {port1}", TEXT_DARK, 0.40)
        hline(f"  Status: {'ONLINE' if eng1.online else '-- OFFLINE --'}", COLOR_GREEN if eng1.online else COLOR_RED, 0.42)

        # Link 1 activity bar
        bw_ = self.hud_w - 36
        cv2.rectangle(canvas, (hx, hy), (hx + bw_, hy + 10), GRID_COLOR, -1)
        fill1 = int(np.clip(eng1.current_score / 100.0, 0, 1) * bw_)
        bar1_col = COLOR_RED if eng1.in_beam else COLOR_BLUE
        cv2.rectangle(canvas, (hx, hy), (hx + fill1, hy + 10), bar1_col, -1)
        # Threshold marker
        tmark = int((threshold / 100.0) * bw_)
        cv2.line(canvas, (hx + tmark, hy - 3), (hx + tmark, hy + 13), COLOR_YELLOW, 2)
        hy += 20

        hy += 8
        cv2.line(canvas, (hx, hy), (self.width - 10, hy), PANEL_BORDER, 1)
        hy += 14

        # Link 2 status
        hline("LINK 2 (Vertical / Y)", COLOR_TEAL, 0.46)
        if has_link2:
            l2_col = COLOR_RED if eng2.in_beam else (TEXT_MUTED if eng2.online else (60, 60, 80))
            hline(f"  Score : {eng2.current_score:5.1f}%", l2_col)
            hline(f"  RSSI  : {eng2.current_rssi:5.0f} dBm", TEXT_MUTED, 0.44)
            hline(f"  Drop  : {eng2.current_rssi_drop:4.1f} dB", TEXT_MUTED, 0.44)
            hline(f"  Port  : {port2}", TEXT_DARK, 0.40)
            hline(f"  Status: {'ONLINE' if eng2.online else '-- OFFLINE --'}", COLOR_GREEN if eng2.online else COLOR_RED, 0.42)
        else:
            hline("  Score : --", TEXT_DARK)
            hline("  RSSI  : --", TEXT_DARK, 0.44)
            hline("  Port  : Not Connected", TEXT_DARK, 0.40)
            hline("  Status: [STANDBY - SINGLE LINK]", COLOR_YELLOW, 0.42)

        # Link 2 activity bar
        cv2.rectangle(canvas, (hx, hy), (hx + bw_, hy + 10), GRID_COLOR, -1)
        if has_link2:
            fill2 = int(np.clip(eng2.current_score / 100.0, 0, 1) * bw_)
            bar2_col = COLOR_RED if eng2.in_beam else COLOR_TEAL
            cv2.rectangle(canvas, (hx, hy), (hx + fill2, hy + 10), bar2_col, -1)
            cv2.line(canvas, (hx + tmark, hy - 3), (hx + tmark, hy + 13), COLOR_YELLOW, 2)
        hy += 20

        hy += 10
        cv2.line(canvas, (hx, hy), (self.width - 10, hy), PANEL_BORDER, 1)
        hy += 14

        # System telemetry
        hline(f"Threshold: {threshold:.1f}%  ([+]/[-])", COLOR_YELLOW, 0.44)
        hline(f"FPS: {fps:.1f}", TEXT_MUTED, 0.44)
        calib_txt = "YES" if (eng1.calibrating or eng2.calibrating) else "NO"
        hline(f"Calibrating: {calib_txt}", TEXT_MUTED, 0.44)

        hy += 10
        hline("[C] Recalibrate  [Z] Clear heat", TEXT_DARK, 0.40)
        hline("[+]/[-] Thresh   [Q] Quit", TEXT_DARK, 0.40)

        # Footer
        footer_y = self.height - 20
        cv2.putText(canvas, "[C] Recalib  [Z] Clear  [+/-] Thresh  [Q/Esc] Quit",
                    (45, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_DARK, 1, cv2.LINE_AA)

        cv2.imshow(self.title, canvas)
        return cv2.waitKey(1) & 0xFF


# =============================================================================
# 2D Demo Simulator
# =============================================================================
class DualLinkDemoSource:
    """Simulates a person walking in 2D paths around the room.

    Generates two correlated CSI streams that change as the simulated person
    moves through the intersection of the two perpendicular beams.
    """

    def __init__(self, rate_hz: float = 30.0) -> None:
        self.rate_hz = rate_hz
        self.interval = 1.0 / rate_hz
        self.t0 = time.monotonic()
        self.last_emit = 0.0
        self.seq1 = self.seq2 = 0
        self.n_sub = 64
        x = np.linspace(0, 1, self.n_sub)
        self.base_amp = 36.0 + 8.0 * np.sin(x * 6.28) + 4.0 * np.cos(x * 12.56)

        # Movement path: (x_target, y_target, start_t, end_t)
        # Person walks in a figure-8 pattern over 40 seconds
        self._path = [
            (0.2, 0.5, 4,  9),    # Walk left-center
            (0.5, 0.2, 10, 15),   # Walk to top-center
            (0.8, 0.5, 16, 21),   # Walk right-center
            (0.5, 0.8, 22, 27),   # Walk to bottom-center
            (0.2, 0.5, 28, 33),   # Back to left-center
        ]
        self.cycle = 36.0

    def _person_pos(self, t: float) -> Tuple[float, float]:
        """Smooth interpolated person position at time t."""
        cycle_t = t % self.cycle
        # Find current segment
        for x_tgt, y_tgt, t0, t1 in self._path:
            if t0 <= cycle_t < t1:
                prog = (cycle_t - t0) / max(t1 - t0, 0.001)
                # Smooth step
                prog = prog * prog * (3 - 2 * prog)
                # Previous point (default: center)
                prev_x, prev_y = 0.5, 0.5
                for px, py, pt0, pt1 in reversed(self._path):
                    if pt1 <= t0:
                        prev_x, prev_y = px, py
                        break
                return (
                    prev_x + prog * (x_tgt - prev_x),
                    prev_y + prog * (y_tgt - prev_y),
                )
        return 0.5, 0.5  # idle

    def _in_segment(self, t: float) -> bool:
        cycle_t = t % self.cycle
        for _, _, t0, t1 in self._path:
            if t0 - 0.5 <= cycle_t < t1 + 0.5:
                return True
        return False

    def _make_frame(self, seq: int, rssi_base: float, intensity: float, rssi_drop: float) -> Dict[str, Any]:
        noise = np.random.normal(0.0, 0.5 + intensity * 7.0, size=self.n_sub)
        amp = np.clip(self.base_amp + noise - intensity * 8.0, 2.0, None).astype(np.float32)
        rssi = int(rssi_base - rssi_drop + np.random.normal(0, 0.8))
        return {"seq": seq, "mac": "aa:bb:cc:dd:ee:01", "rssi": rssi,
                "channel": 6, "n_sub": self.n_sub, "amp": amp}

    def tick(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Return (frame_link1, frame_link2). None if not yet time for a new frame."""
        now = time.monotonic()
        if now - self.last_emit < self.interval:
            return None, None
        self.last_emit = now
        self.seq1 += 1
        self.seq2 += 1
        elapsed = now - self.t0

        # Idle for first 3s (calibration)
        if elapsed < 3.5:
            f1 = self._make_frame(self.seq1, -45.0, 0.0, 0.0)
            f2 = self._make_frame(self.seq2, -48.0, 0.0, 0.0)
            return f1, f2

        px, py = self._person_pos(elapsed - 3.5)
        active = self._in_segment(elapsed - 3.5)

        # Link 1 (horizontal, beam at y=0.35): ESP1 is at x=0 (high drop), AP1 at x=1 (low drop)
        link1_dist = abs(py - 0.35) if active else 1.0
        link1_intensity = max(0.0, (1.0 - link1_dist * 2.2) * 0.95) if active else 0.0
        link1_drop = link1_intensity * (4.0 + 12.0 * (1.0 - px))

        # Link 2 (vertical, beam at x=0.65): AP2 is at y=0 (low drop), ESP2 at y=1 (high drop)
        link2_dist = abs(px - 0.65) if active else 1.0
        link2_intensity = max(0.0, (1.0 - link2_dist * 2.2) * 0.95) if active else 0.0
        link2_drop = link2_intensity * (4.0 + 12.0 * py)

        f1 = self._make_frame(self.seq1, -45.0, link1_intensity, link1_drop)
        f2 = self._make_frame(self.seq2, -48.0, link2_intensity, link2_drop)
        return f1, f2


# =============================================================================
# Main Application
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="WiFi CSI Dual-Link 2D Heatmap Tracker")
    parser.add_argument("--demo", action="store_true", help="Run in simulation mode (no hardware needed)")
    parser.add_argument("--port1", type=str, default="", help="Serial port of ESP32 #1 (Link 1 / horizontal)")
    parser.add_argument("--port2", type=str, default="", help="Serial port of ESP32 #2 (Link 2 / vertical)")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--threshold", type=float, default=35.0, help="Presence detection threshold (0..100)")
    parser.add_argument("--no-gui", action="store_true", help="Console-only mode")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--seconds", type=float, default=0.0, help="Auto-stop after N seconds (0=run forever)")
    args = parser.parse_args()

    threshold = args.threshold

    # --- Setup data sources ---
    demo_source: Optional[DualLinkDemoSource] = None
    reader1: Optional[CSIReader] = None
    reader2: Optional[CSIReader] = None
    port1_label = "DEMO"
    port2_label = "DEMO"

    if args.demo:
        print("== [DEMO MODE] Simulating 2D person movement ==")
        demo_source = DualLinkDemoSource(rate_hz=30.0)
    else:
        # Auto-detect or use provided ports
        port1 = args.port1.strip()
        port2 = args.port2.strip()

        # Try config.json if ports not provided
        if not port1 or not port2:
            try:
                cfg = load_config(args.config)
                devs = cfg.get("devices", [])
                if len(devs) >= 1 and not port1:
                    port1 = devs[0].get("port", "").strip()
                if len(devs) >= 2 and not port2:
                    port2 = devs[1].get("port", "").strip()
            except Exception:
                pass

        # Auto-detect from connected serial ports
        if not port1 or not port2:
            available = list_serial_ports()
            if len(available) >= 1 and not port1:
                port1 = available[0]["device"]
                print(f">> Auto-detected ESP32 #1 on {port1}")
            if len(available) >= 2 and not port2:
                port2 = available[1]["device"]
                print(f">> Auto-detected ESP32 #2 on {port2}")

        if not port1:
            print("=" * 60)
            print("[ERROR] Cannot find ESP32 #1 port!")
            print("  Options:")
            print("  1. Connect both ESP32s and try again")
            print("  2. Run: python src/dual_link_tracker.py --port1 COM8 --port2 COM9")
            print("  3. Test without hardware: python src/dual_link_tracker.py --demo")
            print("=" * 60)
            return 1

        port1_label = port1
        port2_label = port2 if port2 else "MISSING"

        print(f">> Connecting to ESP32 #1 (Link 1) on {port1} ...")
        reader1 = CSIReader(device_id="esp32_link1", port=port1, baud=args.baud, timeout=1.0)
        reader1.start()

        if port2:
            print(f">> Connecting to ESP32 #2 (Link 2) on {port2} ...")
            reader2 = CSIReader(device_id="esp32_link2", port=port2, baud=args.baud, timeout=1.0)
            reader2.start()
        else:
            print("[WARNING] Only 1 ESP32 found — running in single-link mode (1D position only)")

    # --- Initialize engines ---
    eng1 = LinkEngine("link1", threshold=threshold)
    eng2 = LinkEngine("link2", threshold=threshold)
    fusion = DualLinkFusion(smooth_alpha=0.35)
    kalman = KalmanTracker(process_noise=1e-3, measurement_noise=1.5e-2)
    heatmap = HeatmapEngine(grid_w=100, grid_h=80, decay=0.978, blob_sigma_px=8, intensity=18.0)

    # --- Setup UI ---
    ui: Optional[TrackerUI] = None
    if not args.no_gui:
        ui = TrackerUI(title=f"WiFi CSI 2D Tracker — [{port1_label} | {port2_label}]")
    else:
        print(f">> Console mode. Threshold: {threshold}%. Press Ctrl+C to quit.\n")

    has_link2 = (demo_source is not None) or (reader2 is not None)
    last_seq1 = last_seq2 = -1
    x_smooth = y_smooth = 0.5
    presence = False
    last_print = 0.0
    start_t = time.monotonic()

    try:
        while True:
            frame1: Optional[Dict[str, Any]] = None
            frame2: Optional[Dict[str, Any]] = None

            if demo_source is not None:
                frame1, frame2 = demo_source.tick()
            else:
                if reader1 is not None:
                    lf = reader1.get_last_frame()
                    if lf is not None and lf.get("seq") != last_seq1:
                        last_seq1 = lf.get("seq", -1)
                        frame1 = lf
                if reader2 is not None:
                    lf2 = reader2.get_last_frame()
                    if lf2 is not None and lf2.get("seq") != last_seq2:
                        last_seq2 = lf2.get("seq", -1)
                        frame2 = lf2

            if frame1 is not None:
                eng1.update(frame1)
            if frame2 is not None:
                eng2.update(frame2)

            # Fuse into 2D position
            x_raw, y_raw, pres_score, presence = fusion.fuse(
                eng1.current_score,
                eng2.current_score,
                eng1.current_rssi_drop,
                eng2.current_rssi_drop,
                thresh=threshold,
                has_link2=has_link2,
            )

            # Kalman smoothing
            if presence:
                x_smooth, y_smooth = kalman.update(x_raw, y_raw)
                heatmap.deposit(x_smooth, y_smooth, strength=min(1.5, pres_score / 60.0))
            else:
                x_smooth, y_smooth = kalman.rest()

            heatmap.decay_step()

            # Render
            if ui is not None:
                key = ui.render(
                    eng1, eng2, fusion, kalman, heatmap,
                    x_smooth, y_smooth, presence, threshold,
                    port1_label, port2_label,
                    has_link2=has_link2,
                )
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("c"), ord("C")):
                    print("\n>> Recalibrating — keep room empty for 3s!")
                    eng1.recalibrate()
                    eng2.recalibrate()
                    fusion.reset()
                    heatmap.clear()
                    if ui:
                        ui.trail.clear()
                elif key in (ord("z"), ord("Z"), ord("r"), ord("R")):
                    heatmap.clear()
                    if ui:
                        ui.trail.clear()
                    print("\n>> Heatmap cleared.")
                elif key in (ord("+"), ord("=")):
                    threshold = min(90.0, threshold + 2.5)
                    eng1.threshold = eng2.threshold = threshold
                elif key in (ord("-"), ord("_")):
                    threshold = max(10.0, threshold - 2.5)
                    eng1.threshold = eng2.threshold = threshold
            else:
                now = time.monotonic()
                if now - last_print >= 0.5:
                    last_print = now
                    pres_str = "YES" if presence else "NO "
                    print(
                        f"\r[{time.strftime('%H:%M:%S')}] "
                        f"Presence: {pres_str} | "
                        f"X={x_smooth:.3f} Y={y_smooth:.3f} | "
                        f"L1={eng1.current_score:4.0f}% L2={eng2.current_score:4.0f}%",
                        end="", flush=True,
                    )

            time.sleep(0.012)

            if args.seconds > 0 and (time.monotonic() - start_t) >= args.seconds:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if reader1 is not None:
            reader1.stop()
        if reader2 is not None:
            reader2.stop()
        if ui is not None:
            cv2.destroyAllWindows()
        print("\n[Exited cleanly]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
