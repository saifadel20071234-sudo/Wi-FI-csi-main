"""Entry point: WiFi CSI human sensing (real serial or demo simulator).

Examples:
    python src/main.py --demo                 # simulator, no hardware
    python src/main.py                        # real ESP32s (ports from config)
    python src/main.py --ports COM3 COM5      # override serial ports
    python src/main.py --list-ports           # show available serial ports
    python src/main.py --no-gui               # console-only (no OpenCV)
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional

from config import assign_ports, list_serial_ports, load_config
from processor import CSIProcessor
from kalman import KalmanTracker
from count_model import CountFeatureExtractor, CountPredictor
from rti import RTIEngine

try:
    from occupancy import OccupancyEngine
except Exception:  # OpenCV missing -> no blob detection (e.g. --no-gui)
    OccupancyEngine = None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WiFi CSI human sensing")
    p.add_argument("--config", default="config.json", help="path to config.json")
    p.add_argument("--demo", action="store_true", help="run with a simulated person")
    p.add_argument("--ports", nargs="+", help="serial ports, e.g. --ports COM3 COM5")
    p.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    p.add_argument("--no-gui", action="store_true", help="console mode (no OpenCV window)")
    p.add_argument("--record", action="store_true", help="save parsed CSI to CSV files")
    p.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds (0=run forever)")
    return p


def print_ports() -> None:
    ports = list_serial_ports()
    if not ports:
        print("No serial ports detected (is the ESP32 plugged in?).")
        return
    print("Available serial ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p['device']}  -  {p['description']}")


def collect_from_readers(readers: Dict[str, Any], last_seen: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
    """Collect the newest unprocessed frame per device."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for dev_id, reader in readers.items():
        frame = reader.get_last_frame()
        if frame is None:
            continue
        if last_seen.get(dev_id) == frame.get("seq"):
            continue
        last_seen[dev_id] = frame.get("seq")
        out[dev_id] = [frame]
    return out


def print_state(state: Dict[str, Any]) -> None:
    pres = "YES" if state["presence"] else "NO "
    line = (
        f"[{time.strftime('%H:%M:%S')}] presence={pres} state={state['state']:7s} "
        f"pos={state['position']:6s} x={state['x']:.2f} y={state['y']:.2f} "
        f"people={state.get('people', state['count'])} conf={state['confidence']:.2f} "
        + "  ".join(
            f"{d}=act{state['activity'].get(d, 0):.2f}/rssi{state['rssi'].get(d, '--')}"
            for d in state["online"]
        )
    )
    print(line)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.list_ports:
        print_ports()
        return 0

    cfg = load_config(args.config)
    devices: List[Dict[str, Any]] = cfg["devices"]
    if not devices:
        print("ERROR: no devices in config.json. Add at least one ESP32 entry.")
        return 1

    proc_cfg = cfg.get("processing", {})
    ml_override = bool(proc_cfg.get("count_ml_override", False))
    ml_conf_min = float(proc_cfg.get("count_ml_confidence_min", 0.6))

    if args.demo:
        from simulator import SimulatorSource

        warmup = proc_cfg.get("calibration_seconds", 4.0) + 1.0
        source = SimulatorSource(devices, rate_hz=15.0, warmup_seconds=warmup)
        readers: Dict[str, Any] = {}
        print("== DEMO MODE (simulated person) ==")
    else:
        from csi_reader import CSIReader

        assign_ports(cfg, args.ports)
        readers = {}
        for dev in devices:
            if not dev.get("port"):
                print(f"WARNING: no serial port for '{dev['id']}'. "
                      "Set it in config.json or pass --ports. "
                      "(use --list-ports to see available ports)")
                continue
            record_csv = f"csi_{dev['id']}.csv" if args.record else None
            reader = CSIReader(
                device_id=dev["id"],
                port=dev["port"],
                baud=cfg["serial"].get("baud", 921600),
                timeout=cfg["serial"].get("timeout", 1.0),
                record_csv=record_csv,
            )
            reader.start()
            readers[dev["id"]] = reader
            print(f"  opened {dev['id']} on {dev['port']} (baud {cfg['serial'].get('baud', 921600)})")
        source = None

    # Brief pause so calibration / first frames can arrive.
    time.sleep(0.5)

    processor = CSIProcessor(cfg, devices)

    kalman = KalmanTracker(
        process_noise=proc_cfg.get("kalman_process_noise", 1e-3),
        measurement_noise=proc_cfg.get("kalman_measurement_noise", 1e-2),
    )

    room = cfg["room"]
    grid_h = 80
    grid_w = max(1, int(round(grid_h * room["width_m"] / max(room["height_m"], 0.001))))

    occupancy = None
    if OccupancyEngine is not None:
        occupancy = OccupancyEngine(
            grid_w=grid_w,
            grid_h=grid_h,
            decay=proc_cfg.get("heat_decay", 0.965),
            kernel_amp=proc_cfg.get("heat_intensity", 14.0) * 0.6,
            threshold=proc_cfg.get("blob_threshold", 55),
            min_area_cells=proc_cfg.get("blob_min_area", 12.0),
            open_ksize=proc_cfg.get("blob_open_ksize", 3),
            close_ksize=proc_cfg.get("blob_close_ksize", 5),
            blur_ksize=proc_cfg.get("blob_blur_ksize", 5),
        )

    count_fe = CountFeatureExtractor(
        devices, window_frames=proc_cfg.get("count_window_frames", 40)
    )
    count_pred = CountPredictor(
        model_path=proc_cfg.get("count_model", "count_model.pkl"),
        scaler_path=proc_cfg.get("count_scaler", "count_scaler.pkl"),
    )

    # 3+ devices -> Radio Tomographic Imaging for true 2D localization.
    rti = None
    if len(devices) >= 3:
        rti = RTIEngine(
            devices, cfg["router"],
            grid_w=grid_w, grid_h=grid_h,
            link_width=proc_cfg.get("rti_link_width", 0.30),
        )

    if args.no_gui:
        visualizer = None
        print("== CONSOLE MODE (Ctrl+C to exit) ==")
    else:
        try:
            from visualizer import Visualizer
        except Exception as exc:
            print(f"ERROR: cannot import OpenCV visualizer: {exc}")
            print("       Run `python -m pip install opencv-python` or use --no-gui.")
            for r in readers.values():
                r.stop()
            return 1
        visualizer = Visualizer(cfg, devices)

    last_seen: Dict[str, int] = {}
    start = time.monotonic()
    last_print = 0.0

    try:
        while True:
            if args.demo and source is not None:
                frames = source.tick()
            else:
                frames = collect_from_readers(readers, last_seen)

            state = processor.process(frames)

            # Smooth the position with a Kalman filter for a stable "live" follow.
            if state["presence"]:
                kx, ky = kalman.update(state["x"], state["y"])
            else:
                kx, ky = kalman.rest()
            state["x"], state["y"] = kx, ky

            # Accumulate heat + detect people blobs with OpenCV.
            heat_bgr = None
            if occupancy is not None:
                if rti is not None:
                    occupancy.accumulate_map(rti.compute(state["activity"]))
                else:
                    occupancy.update(kx, ky, 0.5 + state["confidence"])
                occupancy.decay_grid()
                blobs = occupancy.detect()
                state["blobs"] = blobs
                state["blob_count"] = len(blobs) if state["presence"] else 0
                heat_bgr = occupancy.heat_bgr()
            else:
                state["blobs"] = []
                state["blob_count"] = state.get("count", 0) if state["presence"] else 0

            # ML count is an auxiliary estimate. The geometric blob count stays
            # authoritative unless count_ml_override is enabled AND the model is
            # confident enough (the shipped model was trained on one-person data).
            count_fe.push(state)
            state["ml_count"] = None
            state["ml_confidence"] = 0.0
            if count_pred.available and count_fe.ready:
                vals, _ = count_fe.features()
                ml_count, ml_conf = count_pred.predict(vals)
                state["ml_count"] = ml_count
                state["ml_confidence"] = ml_conf

            # Final people count shown to the user.
            blob_count = state.get("blob_count", 0)
            ml_count = state.get("ml_count")
            ml_conf = state.get("ml_confidence", 0.0)
            if (
                ml_override
                and ml_count is not None
                and ml_conf >= ml_conf_min
            ):
                state["people"] = ml_count
            else:
                state["people"] = blob_count

            if visualizer is not None:
                key = visualizer.render(state, frames, heat_bgr)
                if key == ord("q") or key == 27:
                    break
                if key == ord("c"):
                    processor.recalibrate()
                    if occupancy is not None:
                        occupancy.clear()
                    print(">> recalibrating baseline (keep the room empty) ...")
                if key == ord("h") and occupancy is not None:
                    occupancy.clear()
            else:
                now = time.monotonic()
                if now - last_print >= 1.0:
                    print_state(state)
                    last_print = now

            if args.seconds and (time.monotonic() - start) >= args.seconds:
                break

            time.sleep(0.016)
    except KeyboardInterrupt:
        pass
    finally:
        for reader in readers.values():
            reader.stop()
        if visualizer is not None:
            visualizer.close()
        print("\nbye.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
