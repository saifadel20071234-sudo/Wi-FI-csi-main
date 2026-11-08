"""Collect labeled windowed features for training the count model.

Run once per scenario with the number of people actually in the room:

    python collect_count_data.py --count 1 --seconds 60
    python collect_count_data.py --count 2 --seconds 60
    python collect_count_data.py --count 0 --seconds 60   # empty room

    # without hardware:
    python collect_count_data.py --demo --count 1 --seconds 30

Appends rows to count_data.csv (header written once). Then run:
    python train_count_model.py --csv count_data.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, "src")

from config import assign_ports, load_config
from processor import CSIProcessor
from count_model import CountFeatureExtractor


def main(argv=None):
    p = argparse.ArgumentParser(description="Collect labeled count features")
    p.add_argument("--config", default="config.json")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--ports", nargs="+", default=None)
    p.add_argument("--count", type=int, required=True, help="number of people in the room")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--out", default="count_data.csv")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    devices = cfg["devices"]
    proc_cfg = cfg.get("processing", {})
    extractor = CountFeatureExtractor(devices, window_frames=proc_cfg.get("count_window_frames", 40))

    if args.demo:
        from simulator import SimulatorSource
        source = SimulatorSource(devices, rate_hz=15.0, warmup_seconds=proc_cfg.get("calibration_seconds", 4.0) + 1.0)
        readers = {}
    else:
        from csi_reader import CSIReader
        assign_ports(cfg, args.ports)
        readers = {}
        for dev in devices:
            if not dev.get("port"):
                print(f"WARNING: no port for '{dev['id']}'. Use --ports or config.json.")
                continue
            r = CSIReader(device_id=dev["id"], port=dev["port"],
                          baud=cfg["serial"].get("baud", 921600),
                          timeout=cfg["serial"].get("timeout", 1.0))
            r.start()
            readers[dev["id"]] = r
        source = None

    time.sleep(0.5)
    processor = CSIProcessor(cfg, devices)

    from count_model import feature_names
    header = ["count"] + feature_names(devices)

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        start = time.monotonic()
        rows = 0
        last_seen = {}
        print(f"Collecting count={args.count} for {args.seconds}s ...")
        while time.monotonic() - start < args.seconds:
            if args.demo and source is not None:
                frames = source.tick()
            else:
                out = {}
                for did, reader in readers.items():
                    fr = reader.get_last_frame()
                    if fr is not None and last_seen.get(did) != fr.get("seq"):
                        last_seen[did] = fr.get("seq")
                        out[did] = [fr]
                frames = out

            state = processor.process(frames)
            # Skip the empty-room calibration phase so the window is clean.
            if state.get("calibrating"):
                extractor.reset()
                time.sleep(0.016)
                continue
            extractor.push(state)
            if extractor.ready:
                vals, _ = extractor.features()
                w.writerow([args.count] + vals)
                rows += 1
            time.sleep(0.016)

    for r in readers.values():
        r.stop()
    print(f"Done. {rows} feature rows appended to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
