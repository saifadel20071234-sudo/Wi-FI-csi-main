"""Serial CSI reader: opens one ESP32 serial port, parses CSI_DATA lines.

The firmware (Espressif ESP-CSI `csi_recv_router`) prints one line per received
packet, e.g.:

    CSI_DATA,0,94:d9:b3:80:8c:81,-30,11,1,6,1,0,1,0,1,0,0,-93,0,13,2,2751923,0,67,0,128,1,"[67,48,4,0,...]"

The last field is a JSON array holding [imag0, real0, imag1, real1, ...] for
every subcarrier (imaginary part first). We only need the amplitude, which is
sqrt(real^2 + imag^2) per subcarrier.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np


class CSIReader:
    """Background serial reader for a single ESP32 device."""

    def __init__(
        self,
        device_id: str,
        port: str,
        baud: int = 921600,
        timeout: float = 1.0,
        record_csv: Optional[str] = None,
        maxlen: int = 512,
    ) -> None:
        self.device_id = device_id
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.record_csv = record_csv
        self._frames: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self._error: Optional[str] = None
        self._bytes_total = 0
        self._csv_file = None
        self._csv_writer = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()
        self._running = True
        if self.record_csv:
            self._csv_file = open(self.record_csv, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["device", "seq", "mac", "rssi", "channel", "n_sub", "amp_mean"]
            )
        self._thread = threading.Thread(target=self._run, name=f"csi-{self.device_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._running = False
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def bytes_total(self) -> int:
        return self._bytes_total

    # -- data access -----------------------------------------------------
    def get_frames(self, count: Optional[int] = None) -> list:
        """Return a snapshot of recent parsed frames (oldest first)."""
        with self._lock:
            items = list(self._frames)
        if count is None:
            return items
        return items[-count:]

    def get_last_frame(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    # -- internals -------------------------------------------------------
    def _run(self) -> None:
        try:
            import serial  # local import keeps the rest importable without it
        except Exception as exc:  # pragma: no cover
            self._error = f"pyserial not installed: {exc}"
            self._running = False
            return

        ser = None
        retries = 0
        while not self._stop.is_set() and ser is None and retries < 8:
            try:
                ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baud,
                    bytesize=8,
                    parity="N",
                    stopbits=1,
                    timeout=self.timeout,
                )
            except Exception as exc:
                retries += 1
                self._error = f"waiting for {self.port} ({retries}/8)..."
                time.sleep(0.8)

        if ser is None:
            self._error = f"cannot open {self.port} (Access Denied or Not Found)"
            self._running = False
            return

        while not self._stop.is_set():
            try:
                raw = ser.readline()
            except Exception as exc:
                self._error = f"read error on {self.port}: {exc}"
                break
            if not raw:
                continue
            self._bytes_total += len(raw)
            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            frame = parse_csi_line(line)
            if frame is None:
                continue
            frame["device"] = self.device_id
            frame["ts"] = time.monotonic()

            with self._lock:
                self._frames.append(frame)

            if self._csv_writer is not None and frame.get("amp") is not None:
                self._csv_writer.writerow(
                    [
                        self.device_id,
                        frame.get("seq"),
                        frame.get("mac"),
                        frame.get("rssi"),
                        frame.get("channel"),
                        frame.get("n_sub"),
                        round(float(np.mean(frame["amp"])), 3),
                    ]
                )
                self._csv_file.flush()

        try:
            ser.close()
        except Exception:
            pass
        self._running = False


def parse_csi_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single CSI_DATA line into a frame dict, or None if invalid.

    Returns:
        dict with keys: seq, mac, rssi, channel, n_sub, amp (float32 ndarray).
    """
    if not line:
        return None
    if "CSI_DATA" not in line:
        return None

    # The data payload is a JSON array wrapped in quotes; everything before the
    # first '[' is comma-separated metadata.
    bracket = line.find("[")
    rbracket = line.rfind("]")
    if bracket == -1 or rbracket == -1 or rbracket < bracket:
        return None
    header = line[:bracket].rstrip('", ')
    try:
        payload = json.loads(line[bracket : rbracket + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    fields = header.split(",")
    # Header ends with: ...,<len>,<first_word>
    if len(fields) < 5:
        return None

    try:
        seq = int(fields[1])
        mac = fields[2]
        rssi = int(fields[3])
        # channel sits at a fixed position in the classic ESP32 header
        # (type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,...,channel=idx16,...).
        channel = int(fields[16]) if len(fields) > 16 else 0
        declared_len = int(fields[-2]) if len(fields) >= 2 else 0
    except (ValueError, IndexError):
        return None

    if not isinstance(payload, list) or len(payload) == 0:
        return None
    if declared_len and declared_len != len(payload):
        # Mismatch usually means a torn line; still usable if length is even.
        pass

    vals = np.asarray(payload, dtype=np.float32)
    # Only use full real/imag pairs.
    usable = (len(vals) // 2) * 2
    if usable == 0:
        return None
    pairs = vals[:usable].reshape(-1, 2)
    real = pairs[:, 1]
    imag = pairs[:, 0]
    amp = np.sqrt(real * real + imag * imag).astype(np.float32)

    return {
        "seq": seq,
        "mac": mac,
        "rssi": rssi,
        "channel": channel,
        "n_sub": amp.shape[0],
        "amp": amp,
    }
