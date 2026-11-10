"""Configuration loading, validation and serial-port helpers."""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List


# Fallback defaults merged under any value missing from config.json.
DEFAULT_CONFIG: Dict[str, Any] = {
    "room": {"width_m": 4.0, "height_m": 5.0, "label": "Room"},
    "router": {"label": "Router", "x": 0.5, "y": 0.06},
    "devices": [],
    "serial": {"baud": 921600, "timeout": 1.0},
    "processing": {
        "window_seconds": 2.0,
        "calibration_seconds": 4.0,
        "activity_scale": 0.15,
        "rssi_weight": 0.5,
        "motion_threshold": 0.55,
        "smoothing_alpha": 0.30,
        "heat_decay": 0.965,
        "heat_intensity": 14.0,
        "count_thresholds": [1.5, 2.5, 3.5],
        "kalman_process_noise": 0.001,
        "kalman_measurement_noise": 0.01,
        "blob_threshold": 55,
        "blob_min_area": 12.0,
        "blob_open_ksize": 3,
        "blob_close_ksize": 5,
        "blob_blur_ksize": 5,
        "count_window_frames": 40,
        "count_model": "count_model.pkl",
        "count_scaler": "count_scaler.pkl",
        "count_ml_override": False,
        "count_ml_confidence_min": 0.60,
        "rti_link_width": 0.30,
    },
    "tracker": {
        "threshold": 50.0,
        "stationary_hold_sec": 10.0,
        "static_shadow_db": 1.5,
        "calibration_seconds": 3.0,
        "smoothing_alpha": 0.35,
        "max_step": 0.065,
        "deadband": 0.016,
        "heat_decay": 0.978,
        "heat_intensity": 16.0,
        "kalman_process_noise": 0.10,
        "kalman_measurement_noise": 0.001,
    },
    "visual": {
        "window_name": "WiFi CSI Sensing",
        "width": 1200,
        "height": 820,
        "show_raw": False,
        "fps_target": 30,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load config from a JSON file, merging over defaults."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Create it (see README.md) or pass --config <path>."
        )
    with open(path, "r", encoding="utf-8-sig") as fh:
        user_cfg = json.load(fh)
    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)

    # Normalize room dimensions and device fields.
    room = cfg["room"]
    room["width_m"] = float(room["width_m"])
    room["height_m"] = float(room["height_m"])

    for dev in cfg["devices"]:
        dev.setdefault("id", dev.get("label", "dev"))
        dev.setdefault("label", dev["id"])
        dev.setdefault("port", "")
        dev["x"] = float(dev.get("x", 0.5))
        dev["y"] = float(dev.get("y", 0.5))

    return cfg


def list_serial_ports() -> List[Dict[str, str]]:
    """Return available serial ports (idempotent if pyserial missing)."""
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    result: List[Dict[str, str]] = []
    for port in list_ports.comports():
        result.append(
            {
                "device": port.device,
                "description": port.description or "",
                "hwid": port.hwid or "",
            }
        )
    return result


def assign_ports(cfg: Dict[str, Any], port_override: List[str] | None) -> None:
    """Assign serial ports to devices, from override list or auto-detection."""
    devices = cfg["devices"]
    if port_override:
        for i, dev in enumerate(devices):
            if i < len(port_override):
                dev["port"] = port_override[i].strip()
        return

    # Auto-detection: fill any empty port with detected ports in order.
    detected = [p["device"] for p in list_serial_ports()]
    idx = 0
    for dev in devices:
        if not dev.get("port"):
            if idx < len(detected):
                dev["port"] = detected[idx]
                idx += 1
            else:
                dev["port"] = ""
