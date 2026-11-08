"""Real-time WiFi CSI Link Oscilloscope & People Counter (Single ESP32 + AP).

Features:
  * Real-Time Oscilloscope of CSI RF Perturbation.
  * People Crossing Counter (Gate/Tripwire): Counts each person crossing (+1).
  * Multi-Person Discriminator: Detects when 2 people cross together (+2) vs 1 person (+1).
  * Real-Time Instant Count: Shows 0, 1, or 2 people in the beam right now.
  * Hysteresis & Debouncing to eliminate duplicate/false counts.
  * Visual Event Log showing recent crossing timestamps.
  * Interactive controls:
      [C] Recalibrate baseline
      [Z] Reset people counter
      [+] / [-] Adjust threshold sensitivity
      [Q] / [Esc] Exit

Usage:
  python src/scope_monitor.py --demo              # Simulate 1 person then 2 people crossing
  python src/scope_monitor.py                     # Use real ESP32 (auto-detects COM8)
  python src/scope_monitor.py --port COM8         # Specify serial port
  python src/scope_monitor.py --no-gui            # Console-only ASCII mode
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


# -----------------------------------------------------------------------------
# Color Palette (BGR for OpenCV)
# -----------------------------------------------------------------------------
BG_COLOR = (22, 24, 32)
PANEL_COLOR = (30, 33, 44)
PANEL_BORDER = (50, 55, 72)
GRID_LINE_COLOR = (38, 42, 54)

TEXT_WHITE = (245, 248, 252)
TEXT_MUTED = (145, 153, 168)
TEXT_DARK = (85, 90, 105)

COLOR_GREEN = (70, 215, 95)
COLOR_GREEN_DARK = (22, 65, 30)
COLOR_RED = (55, 70, 245)
COLOR_RED_DARK = (28, 32, 105)
COLOR_YELLOW = (45, 205, 255)
COLOR_CYAN = (235, 210, 50)
COLOR_ORANGE = (40, 140, 255)
COLOR_PURPLE = (220, 90, 200)


# -----------------------------------------------------------------------------
# Signal Engine: Baseline, Variance, Crossing State Machine & Counting
# -----------------------------------------------------------------------------
class SingleLinkEngine:
    """Processes CSI frames - Oscilloscope + Tripwire People Counter (1–6 people).

    Classifier uses THREE signals measured during each beam-crossing event:
      • peak_score  (0–100%) — how much the CSI variance spiked
      • rssi_drop   (dB)     — how deep the signal was blocked
      • duration    (s)      — how long the beam stayed disturbed

    More people = higher peak, deeper RSSI drop, and longer crossing duration.
    A 2.5 s cooldown after each event absorbs RF bounce-back noise so one
    real group crossing doesn't get counted multiple times.
    """

    # ── Multi-Person Classifier (1 or 2 People) ──────────────────────────────
    # 2 People requires severe blockage (>18 dB) + saturated peak (>88%) + duration >= 1.2s.
    # Otherwise, any crossing is reliably counted as 1 Person (+1).
    _CROWD_RULES: List[Tuple[float, float, float, int]] = [
        (1.2, 88.0, 18.0, 2),   # 2 people walking together
        (0.35,  0.0,  0.0, 1),   # 1 person (standard crossing)
    ]

    def __init__(
        self,
        window_frames: int = 20,
        calibration_seconds: float = 3.0,
        threshold: float = 38.0,
        activity_scale: float = 0.15,
    ) -> None:
        self.window_frames = window_frames
        self.calibration_seconds = calibration_seconds
        self.threshold = float(threshold)
        self.activity_scale = float(activity_scale)

        self._amp_hist: deque = deque(maxlen=self.window_frames)
        self._rssi_hist: deque = deque(maxlen=self.window_frames)

        # Baseline calibration
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_rssi: Optional[float] = None
        self.calib_start: float = time.monotonic()
        self._calib_amps: List[np.ndarray] = []
        self._calib_rssis: List[float] = []

        # Live signal metrics
        self.current_score: float = 0.0
        self.raw_variance: float = 0.0
        self.current_rssi: float = -50.0
        self.current_rssi_drop: float = 0.0
        self.last_ts: float = 0.0
        self.frame_count: int = 0

        # Temporal EMA & Quantile metrics (adapted from research pipeline)
        self._var_ema_fast: float = 0.0  # span ~ 3 frames
        self._var_ema_slow: float = 0.0  # span ~ 10 frames
        self.var_ema_diff: float = 0.0   # fast - slow: robust surge detector
        self._score_window: deque = deque(maxlen=20)  # rolling window for quantile filtering

        # People counting state machine
        self.total_crossings: int = 0
        self.instant_people: int = 0   # live estimate (0, 1, 2, 3 people)
        self._instant_hist: deque = deque(maxlen=7)  # temporal smoothing window
        self._in_crossing: bool = False
        self._cross_start_t: float = 0.0
        self._cross_peak_score: float = 0.0
        self._cross_peak_drop: float = 0.0
        self._cross_peak_var: float = 0.0
        self._cross_peak_ema_diff: float = 0.0
        self._cooldown_until: float = 0.0
        self.recent_events: deque = deque(maxlen=5)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def recalibrate(self) -> None:
        """Reset baseline — keep the room quiet for 3 s after pressing C."""
        self.baseline_mean = None
        self.baseline_rssi = None
        self.calib_start = time.monotonic()
        self._calib_amps.clear()
        self._calib_rssis.clear()
        self._amp_hist.clear()
        self._score_window.clear()
        self._var_ema_fast = 0.0
        self._var_ema_slow = 0.0
        self.var_ema_diff = 0.0
        self.current_score = 0.0
        self._in_crossing = False
        self._cooldown_until = 0.0
        self.instant_people = 0
        self._instant_hist.clear()

    def reset_counter(self) -> None:
        """Zero the passage counter; does NOT touch the RF baseline."""
        self.total_crossings = 0
        self.recent_events.clear()
        self._in_crossing = False
        self._cooldown_until = 0.0
        self.instant_people = 0
        self._instant_hist.clear()

    @property
    def calibrating(self) -> bool:
        return self.baseline_mean is None

    @property
    def calib_progress(self) -> float:
        if not self.calibrating:
            return 1.0
        elapsed = time.monotonic() - self.calib_start
        return min(1.0, max(0.0, elapsed / max(0.1, self.calibration_seconds)))

    @property
    def in_cooldown(self) -> bool:
        return time.monotonic() < self._cooldown_until

    # ── Multi-factor Crowd Metrics & Classifiers ──────────────────────────────
    def _crowd_metric(self, score: float, drop: float, rel_var: float) -> float:
        """Combined multi-subcarrier disturbance metric [0.0 .. ~1.5].
        
        Weights:
          • Activity Score (50%): general wave disturbance across all 64 subcarriers
          • RSSI Drop (30%): signal power absorption by human body volume
          • Subcarrier Variance (20%): frequency-selective multipath fading
        """
        s_norm = min(1.0, max(0.0, score) / 100.0)
        d_norm = min(1.0, max(0.0, drop) / 13.0)
        v_norm = min(1.0, max(0.0, rel_var) / 0.22)
        return 0.50 * s_norm + 0.30 * d_norm + 0.20 * v_norm

    def _estimate_instant(self, score: float, drop: float, rel_var: float) -> int:
        """Smooth real-time estimate of people in the signal area (0, 1, 2, or 3)."""
        # Hysteresis floor for empty room
        active_floor = max(14.0, self.threshold - 10.0)
        if score < active_floor:
            raw_est = 0
        else:
            metric = self._crowd_metric(score, drop, rel_var)
            if metric >= 0.78:
                raw_est = 3
            elif metric >= 0.52:
                raw_est = 2
            else:
                raw_est = 1

        self._instant_hist.append(raw_est)
        # Robust majority vote over recent frames to prevent 1-frame jitter
        counts: Dict[int, int] = {}
        for val in self._instant_hist:
            counts[val] = counts.get(val, 0) + 1
        return max(counts, key=counts.get)

    def _classify(self, duration: float, peak_score: float, peak_drop: float, peak_var: float) -> Tuple[int, str]:
        """Return (count, label) for a completed crossing event."""
        peak_metric = self._crowd_metric(peak_score, peak_drop, peak_var)
        if peak_metric >= 0.78:
            return 3, "3 People"
        elif peak_metric >= 0.52:
            return 2, "2 People"
        return 1, "1 Person"

    # ── Main update loop ──────────────────────────────────────────────────────
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

        # ── 1) Calibration ────────────────────────────────────────────────────
        if self.calibrating:
            self._calib_amps.append(amp)
            self._calib_rssis.append(rssi)
            elapsed = now - self.calib_start
            if elapsed >= self.calibration_seconds and len(self._calib_amps) >= 8:
                amps_arr = np.asarray(self._calib_amps, dtype=np.float32)
                self.baseline_mean = np.mean(amps_arr, axis=0)
                self.baseline_rssi = float(np.mean(self._calib_rssis))
            return 0.0

        # ── 2) Score (relative variance + RSSI drop) ──────────────────────────
        if len(self._amp_hist) < 3:
            return self.current_score

        stack = np.asarray(list(self._amp_hist), dtype=np.float32)
        var_per_sc = np.var(stack, axis=0)
        base_mean = float(np.mean(self.baseline_mean)) + 1e-6
        rel_var = float(np.mean(var_per_sc)) / base_mean
        self.raw_variance = rel_var

        # Online EMA calculation (fast alpha=0.5 ~span 3, slow alpha=0.18 ~span 10)
        if self._var_ema_fast == 0.0:
            self._var_ema_fast = rel_var
            self._var_ema_slow = rel_var
        else:
            self._var_ema_fast = 0.50 * rel_var + 0.50 * self._var_ema_fast
            self._var_ema_slow = 0.18 * rel_var + 0.82 * self._var_ema_slow
        self.var_ema_diff = max(0.0, self._var_ema_fast - self._var_ema_slow)

        rel_var_norm = rel_var / (rel_var + self.activity_scale)

        rssi_drop = max(0.0, (self.baseline_rssi or 0.0) - rssi)
        self.current_rssi_drop = rssi_drop
        rssi_drop_norm = rssi_drop / (rssi_drop + 8.0) if rssi_drop > 0 else 0.0

        instant_pct = float(np.clip(
            ((rel_var_norm + 0.6 * rssi_drop_norm) / 1.6) * 100.0, 0.0, 100.0
        ))

        alpha = 0.55 if instant_pct > self.current_score else 0.20
        self.current_score = alpha * instant_pct + (1.0 - alpha) * self.current_score
        self._score_window.append(self.current_score)

        # ── 3) Tripwire state machine ─────────────────────────────────────────
        thresh = self.threshold
        hysteresis_low = max(8.0, thresh - 12.0)

        # BEAM ACTIVE
        if self.current_score >= thresh:
            if not self._in_crossing:
                self._in_crossing = True
                self._cross_start_t = now
                self._cross_peak_score = self.current_score
                self._cross_peak_drop = rssi_drop
                self._cross_peak_var = self.raw_variance
                self._cross_peak_ema_diff = self.var_ema_diff
            else:
                self._cross_peak_score = max(self._cross_peak_score, self.current_score)
                self._cross_peak_drop = max(self._cross_peak_drop, rssi_drop)
                self._cross_peak_var = max(self._cross_peak_var, self.raw_variance)
                self._cross_peak_ema_diff = max(self._cross_peak_ema_diff, self.var_ema_diff)

        # BEAM CLEAR
        elif self.current_score < hysteresis_low:
            if self._in_crossing:
                duration = now - self._cross_start_t
                if duration >= 0.30 and not self.in_cooldown:
                    count_added, kind = self._classify(
                        duration, self._cross_peak_score, self._cross_peak_drop, self._cross_peak_var
                    )
                    self.total_crossings += count_added
                    self.recent_events.append((
                        time.strftime("%H:%M:%S"), count_added, kind,
                        self._cross_peak_score, duration
                    ))
                    self._cooldown_until = now + 2.5

                self._in_crossing = False
                self._cross_peak_score = 0.0
                self._cross_peak_drop = 0.0
                self._cross_peak_var = 0.0
                self._cross_peak_ema_diff = 0.0

        # Continuous smooth estimate of instant people in beam
        self.instant_people = self._estimate_instant(
            self.current_score, rssi_drop, self.raw_variance
        )

        return self.current_score


# -----------------------------------------------------------------------------
# Demo Simulator: Alternates between 1 Person and 2 People
# -----------------------------------------------------------------------------
class DemoSingleLinkSource:
    """Simulates realistic CSI: 1 person crossing (+1), then 2 people crossing (+2)."""

    def __init__(self, rate_hz: float = 30.0) -> None:
        self.rate_hz = rate_hz
        self.interval = 1.0 / rate_hz
        self.t0 = time.monotonic()
        self.last_emit = 0.0
        self.seq = 0
        self.n_sub = 64
        x = np.linspace(0, 1, self.n_sub)
        self.base_amp = 36.0 + 8.0 * np.sin(x * 6.28) + 4.0 * np.cos(x * 12.56)

    def tick(self) -> Optional[Dict[str, Any]]:
        now = time.monotonic()
        if now - self.last_emit < self.interval:
            return None
        self.last_emit = now
        self.seq += 1
        elapsed = now - self.t0

        # 22-second full cycle:
        # 0s - 3s: Idle
        # 3s - 6.5s: 1 Person walks across (duration 3.5s, intensity 0.70, drop 6 dB)
        # 6.5s - 11s: Idle
        # 11s - 16s: 2 People walk across together (duration 5s, intensity 1.0, drop 13 dB)
        # 16s - 22s: Idle
        cycle = elapsed % 22.0
        intensity = 0.0
        rssi_drop = 0.0

        if 3.0 <= cycle <= 6.5:
            # 1 Person
            progress = (cycle - 3.0) / 3.5
            intensity = 0.72 * math.sin(progress * math.pi)
            rssi_drop = 6.5 * intensity
        elif 11.0 <= cycle <= 16.0:
            # 2 People (Deeper & Wider)
            progress = (cycle - 11.0) / 5.0
            intensity = 1.0 * math.sin(progress * math.pi)
            rssi_drop = 13.5 * intensity

        noise = np.random.normal(0.0, 0.4 + intensity * 7.5, size=self.n_sub)
        amp = np.clip(self.base_amp + noise - (intensity * 8.0), 2.0, None).astype(np.float32)
        rssi = int(-45.0 - rssi_drop + np.random.normal(0, 0.7))

        return {
            "seq": self.seq,
            "mac": "aa:bb:cc:dd:ee:01",
            "rssi": rssi,
            "channel": 6,
            "n_sub": self.n_sub,
            "amp": amp,
        }


# -----------------------------------------------------------------------------
# OpenCV Oscilloscope & People Counter UI
# -----------------------------------------------------------------------------
class ScopeUI:
    """Renders the oscilloscope waveform, live people counter, and telemetry."""

    def __init__(self, width: int = 1120, height: int = 640, title: str = "WiFi CSI Real-Time Scope") -> None:
        self.width = width
        self.height = height
        self.title = title
        self.history: deque = deque(maxlen=160)
        self.last_render_ts = time.monotonic()
        self.fps = 30.0

        # Oscilloscope Layout (Left)
        self.scope_x = 45
        self.scope_y = 120
        self.scope_w = 720
        self.scope_h = 430

        # Level Bar Layout
        self.bar_x = self.scope_x + self.scope_w + 30
        self.bar_y = self.scope_y
        self.bar_w = 40
        self.bar_h = self.scope_h

        # Right Dashboard Panel
        self.panel_x = self.bar_x + self.bar_w + 25
        self.panel_w = self.width - self.panel_x - 35

        for _ in range(160):
            self.history.append(0.0)

    def draw(self, engine: SingleLinkEngine, frame_info: Dict[str, Any]) -> int:
        canvas = np.full((self.height, self.width, 3), BG_COLOR, dtype=np.uint8)

        now = time.monotonic()
        dt = now - self.last_render_ts
        self.last_render_ts = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        score = engine.current_score
        thresh = engine.threshold
        self.history.append(score)
        is_detected = score >= thresh and not engine.calibrating

        # ---------------------------------------------------------------------
        # 1) Top Header & Status Badge
        # ---------------------------------------------------------------------
        cv2.putText(canvas, "WIFI CSI LINK MONITOR & PEOPLE COUNTER", (45, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Single ESP32 + Hotspot / AP | Real-Time Crossing Detector", (45, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_MUTED, 1, cv2.LINE_AA)

        # Status Banner (Right Side of Header)
        banner_w = 400
        banner_h = 56
        banner_x = self.width - banner_w - 45
        banner_y = 26

        if engine.frame_count < 5:
            cv2.rectangle(canvas, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          PANEL_COLOR, -1)
            cv2.rectangle(canvas, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          COLOR_CYAN, 2)
            cv2.putText(canvas, "CONNECTING TO HOTSPOT...", (banner_x + 20, banner_y + 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.60, COLOR_CYAN, 1, cv2.LINE_AA)
        elif engine.calibrating:
            prog = int(engine.calib_progress * 100)
            cv2.rectangle(canvas, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          PANEL_COLOR, -1)
            cv2.rectangle(canvas, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          COLOR_YELLOW, 2)
            cv2.putText(canvas, f"CALIBRATING BASELINE ({prog}%)...", (banner_x + 20, banner_y + 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, COLOR_YELLOW, 1, cv2.LINE_AA)
        elif is_detected:
            # Person Detected - Glowing Red Badge
            badge_text = f"[!] {engine.instant_people} PERSON DETECTED" if engine.instant_people == 1 else f"[!] {engine.instant_people} PEOPLE DETECTED!"
            cv2.rectangle(canvas, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          COLOR_RED_DARK, -1)
            cv2.rectangle(canvas, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          COLOR_RED, 2)
            cv2.putText(canvas, badge_text, (banner_x + 18, banner_y + 37),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (banner_x + banner_w - 25, banner_y + 28), 10, COLOR_RED, -1)
        else:
            # Clear - Calm Green Badge
            cv2.rectangle(canvas, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          COLOR_GREEN_DARK, -1)
            cv2.rectangle(canvas, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          COLOR_GREEN, 1)
            cv2.putText(canvas, "[OK] ROOM CLEAR (IDLE)", (banner_x + 20, banner_y + 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, COLOR_GREEN, 1, cv2.LINE_AA)
            cv2.circle(canvas, (banner_x + banner_w - 25, banner_y + 28), 8, COLOR_GREEN, -1)

        # ---------------------------------------------------------------------
        # 2) Oscilloscope Box & Grid
        # ---------------------------------------------------------------------
        sx, sy, sw, sh = self.scope_x, self.scope_y, self.scope_w, self.scope_h
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), PANEL_COLOR, -1)
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), PANEL_BORDER, 1)

        for pct in (0, 25, 50, 75, 100):
            gy = int(sy + sh - (pct / 100.0) * sh)
            if 0 < pct < 100:
                cv2.line(canvas, (sx, gy), (sx + sw, gy), GRID_LINE_COLOR, 1)
            cv2.putText(canvas, f"{pct}%", (sx - 34, gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DARK, 1, cv2.LINE_AA)

        for i in range(1, 6):
            gx = sx + int(i * sw / 6.0)
            cv2.line(canvas, (gx, sy), (gx, sy + sh), GRID_LINE_COLOR, 1)

        # ---------------------------------------------------------------------
        # 3) Threshold Line (Adjustable)
        # ---------------------------------------------------------------------
        ty = int(sy + sh - (thresh / 100.0) * sh)
        ty = int(np.clip(ty, sy, sy + sh))
        dash_len = 10
        for dx in range(sx, sx + sw, dash_len * 2):
            cv2.line(canvas, (dx, ty), (min(dx + dash_len, sx + sw), ty), COLOR_YELLOW, 2)
        cv2.putText(canvas, f"THRESHOLD: {thresh:.1f}%", (sx + 15, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_YELLOW, 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 4) Real-Time Signal Waveform & Filled Glow
        # ---------------------------------------------------------------------
        pts: List[Tuple[int, int]] = []
        n_pts = len(self.history)
        step = sw / max(1, n_pts - 1)

        for i, val in enumerate(self.history):
            px = int(sx + i * step)
            py = int(sy + sh - (val / 100.0) * sh)
            py = int(np.clip(py, sy, sy + sh))
            pts.append((px, py))

        if len(pts) >= 2:
            poly_pts = [(sx, sy + sh)] + pts + [(sx + sw, sy + sh)]
            overlay = canvas.copy()
            fill_color = COLOR_RED_DARK if is_detected else COLOR_GREEN_DARK
            cv2.fillPoly(overlay, [np.array(poly_pts, dtype=np.int32)], fill_color)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

            line_color = COLOR_RED if is_detected else COLOR_GREEN
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], line_color, 2, cv2.LINE_AA)

            last_x, last_y = pts[-1]
            cv2.circle(canvas, (last_x, last_y), 7, line_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (last_x, last_y), 11, line_color, 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 5) Level Meter Bar
        # ---------------------------------------------------------------------
        bx, by, bw, bh = self.bar_x, self.bar_y, self.bar_w, self.bar_h
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), PANEL_COLOR, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), PANEL_BORDER, 1)

        fill_h = int((score / 100.0) * bh)
        if fill_h > 0:
            top_y = by + bh - fill_h
            bar_color = COLOR_RED if is_detected else COLOR_GREEN
            cv2.rectangle(canvas, (bx + 2, top_y), (bx + bw - 2, by + bh - 2), bar_color, -1)

        b_ty = int(by + bh - (thresh / 100.0) * bh)
        cv2.line(canvas, (bx - 2, b_ty), (bx + bw + 2, b_ty), COLOR_YELLOW, 2)
        cv2.putText(canvas, f"{score:.0f}%", (bx - 2, by + bh + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, TEXT_WHITE, 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 6) Right Dashboard: People Counter & Event Log
        # ---------------------------------------------------------------------
        px, py, pw = self.panel_x, sy, self.panel_w

        # --- A) TOTAL PASSAGE COUNTER CARD ---
        card1_h = 105
        cv2.rectangle(canvas, (px, py), (px + pw, py + card1_h), PANEL_COLOR, -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + card1_h), PANEL_BORDER, 1)

        cv2.putText(canvas, "TOTAL PEOPLE CROSSED", (px + 14, py + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, TEXT_MUTED, 1, cv2.LINE_AA)

        # Big digits
        count_str = f"{engine.total_crossings:02d}"
        cv2.putText(canvas, count_str, (px + 14, py + 84),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, COLOR_YELLOW, 2, cv2.LINE_AA)

        # Cooldown indicator so user can see when next count is ready
        if engine.in_cooldown:
            remaining = engine._cooldown_until - time.monotonic()
            cd_txt = f"[Cooldown {remaining:.1f}s]"
            cv2.putText(canvas, cd_txt, (px + 120, py + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_PURPLE, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "People Passed", (px + 130, py + 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)

        # --- B) INSTANT BEAM OCCUPANCY CARD ---
        card2_y = py + card1_h + 12
        card2_h = 75
        cv2.rectangle(canvas, (px, card2_y), (px + pw, card2_y + card2_h), PANEL_COLOR, -1)
        cv2.rectangle(canvas, (px, card2_y), (px + pw, card2_y + card2_h), PANEL_BORDER, 1)

        cv2.putText(canvas, "IN BEAM RIGHT NOW", (px + 14, card2_y + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, TEXT_MUTED, 1, cv2.LINE_AA)

        if engine.instant_people == 0:
            inst_text = "0  Clear"
            inst_col = COLOR_GREEN
        elif engine.instant_people == 1:
            inst_text = "~1  Person"
            inst_col = COLOR_ORANGE
        elif engine.instant_people == 2:
            inst_text = "~2  People"
            inst_col = COLOR_YELLOW
        else:
            inst_text = "~3  People!"
            inst_col = COLOR_RED

        cv2.putText(canvas, inst_text, (px + 14, card2_y + 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.70, inst_col, 2, cv2.LINE_AA)

        # --- C) RECENT CROSSINGS LOG ---
        card3_y = card2_y + card2_h + 12
        card3_h = 105
        cv2.rectangle(canvas, (px, card3_y), (px + pw, card3_y + card3_h), PANEL_COLOR, -1)
        cv2.rectangle(canvas, (px, card3_y), (px + pw, card3_y + card3_h), PANEL_BORDER, 1)

        cv2.putText(canvas, "RECENT PASSAGES LOG", (px + 14, card3_y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)

        if not engine.recent_events:
            cv2.putText(canvas, "Waiting for first crossing...", (px + 14, card3_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_DARK, 1, cv2.LINE_AA)
        else:
            ey = card3_y + 44
            for t_str, added, kind, pk, dur in list(engine.recent_events)[-3:]:
                if added == 1:
                    col = COLOR_GREEN
                elif added == 2:
                    col = COLOR_YELLOW
                else:
                    col = COLOR_RED
                txt = f"{t_str} | +{added} {kind} ({dur:.1f}s)"
                cv2.putText(canvas, txt, (px + 14, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
                ey += 20

        # --- D) TELEMETRY & HARDWARE INFO ---
        card4_y = card3_y + card3_h + 12
        card4_h = 105
        cv2.rectangle(canvas, (px, card4_y), (px + pw, card4_y + card4_h), PANEL_COLOR, -1)
        cv2.rectangle(canvas, (px, card4_y), (px + pw, card4_y + card4_h), PANEL_BORDER, 1)

        rssi = frame_info.get("rssi", engine.current_rssi)
        port_name = frame_info.get("port", "DEMO")
        drop = engine.current_rssi_drop

        telemetry = [
            f"Port: {port_name} (ESP32)",
            f"RSSI: {rssi:.0f} dBm  |  Drop: {drop:.1f} dB",
            f"Variance: {engine.raw_variance:.4f}",
            f"Stream Rate: {self.fps:.0f} Hz",
        ]
        ty_line = card4_y + 24
        for t_line in telemetry:
            cv2.putText(canvas, t_line, (px + 14, ty_line),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_MUTED, 1, cv2.LINE_AA)
            ty_line += 21

        # ---------------------------------------------------------------------
        # 7) Footer Shortcuts
        # ---------------------------------------------------------------------
        footer_y = self.height - 20
        controls_text = "[C] Recalibrate  |  [Z] Reset Counter  |  [+] / [-] Sensitivity  |  [Q / Esc] Quit"
        cv2.putText(canvas, controls_text, (45, footer_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_MUTED, 1, cv2.LINE_AA)

        cv2.imshow(self.title, canvas)
        return cv2.waitKey(1) & 0xFF


# -----------------------------------------------------------------------------
# Console Mode Fallback
# -----------------------------------------------------------------------------
def print_console_bar(engine: SingleLinkEngine) -> None:
    score = engine.current_score
    thresh = engine.threshold
    bar_width = 24
    filled = int((score / 100.0) * bar_width)
    bar = "=" * filled + " " * (bar_width - filled)
    if engine.instant_people > 0:
        status = f" [!] {engine.instant_people} PERSON  "
    else:
        status = " [OK] CLEAR     "
    cd = f" CD:{engine._cooldown_until - time.monotonic():.1f}s" if engine.in_cooldown else ""
    line = (
        f"\r[{time.strftime('%H:%M:%S')}] {status} |{bar}| {score:4.0f}% "
        f"(Count: {engine.total_crossings:02d}{cd} | RSSI: {engine.current_rssi:3.0f} dBm)"
    )
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="WiFi CSI Single-Link People Counter")
    parser.add_argument("--demo", action="store_true", help="Run in simulation demo mode")
    parser.add_argument("--port", type=str, default="", help="Serial port of the ESP32 (e.g. COM8)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default 921600)")
    parser.add_argument("--threshold", type=float, default=38.0, help="Presence threshold percentage (0..100)")
    parser.add_argument("--no-gui", action="store_true", help="Run in console-only mode")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--seconds", type=float, default=0.0, help="Auto-stop after N seconds (0=run forever)")
    args = parser.parse_args()

    engine = SingleLinkEngine(threshold=args.threshold)

    # 1) Setup Source
    source_demo: Optional[DemoSingleLinkSource] = None
    reader: Optional[CSIReader] = None
    port_label = "DEMO"

    if args.demo:
        print("== [DEMO MODE] Simulating 1 person then 2 people crossing ==")
        source_demo = DemoSingleLinkSource(rate_hz=30.0)
    else:
        port = args.port.strip()
        if not port:
            try:
                cfg = load_config(args.config)
                for dev in cfg.get("devices", []):
                    if dev.get("port"):
                        port = dev["port"].strip()
                        break
            except Exception:
                pass

        if not port:
            ports = list_serial_ports()
            if ports:
                port = ports[0]["device"]
                desc = ports[0]["description"].encode("ascii", errors="ignore").decode("ascii")
                print(f">> Auto-detected ESP32 port: {port} ({desc})")
            else:
                print("=================================================================")
                print("[ERROR] No ESP32 serial port found!")
                print("  1. Make sure your ESP32 is plugged in on COM8.")
                print("  2. If testing without hardware, run: run_monitor_demo.bat")
                print("=================================================================")
                return 1

        port_label = port
        print(f">> Connecting to ESP32 on {port} (baud {args.baud}) ...")
        reader = CSIReader(device_id="esp32_link", port=port, baud=args.baud, timeout=1.0)
        reader.start()

    # 2) Setup UI
    ui = None if args.no_gui else ScopeUI(title=f"WiFi CSI Scope & People Counter - [{port_label}]")
    if args.no_gui:
        print(f">> Console mode active. Threshold: {engine.threshold}%. Press Ctrl+C to quit.\n")

    last_seq = -1
    last_print = 0.0
    start_t = time.monotonic()

    try:
        while True:
            frame: Optional[Dict[str, Any]] = None

            if source_demo is not None:
                frame = source_demo.tick()
            elif reader is not None:
                last_f = reader.get_last_frame()
                if last_f is not None and last_f.get("seq") != last_seq:
                    last_seq = last_f.get("seq", -1)
                    frame = last_f

            if frame is not None:
                engine.update(frame)

            # Render UI
            if ui is not None:
                info = {"port": port_label, "rssi": engine.current_rssi}
                key = ui.draw(engine, info)

                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("c"), ord("C")):
                    print("\n>> Recalibrating baseline... Keep room still!")
                    engine.recalibrate()
                elif key in (ord("z"), ord("Z"), ord("r"), ord("R")):
                    engine.reset_counter()
                    print("\n>> Counter reset to 0.")
                elif key in (ord("+"), ord("=")):
                    engine.threshold = min(95.0, engine.threshold + 2.5)
                elif key in (ord("-"), ord("_")):
                    engine.threshold = max(10.0, engine.threshold - 2.5)
            else:
                now = time.monotonic()
                if now - last_print >= 0.08:
                    last_print = now
                    print_console_bar(engine)

            time.sleep(0.012)

            if args.seconds > 0 and (time.monotonic() - start_t) >= args.seconds:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.stop()
        if ui is not None:
            cv2.destroyAllWindows()
        print("\n[Exited cleanly]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
