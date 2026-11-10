# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **Presence no longer locks forever.** `DualLinkFusion` now keeps presence for a
  bounded grace window after the last motion (`stationary_hold_sec`, default
  10 s; doubled while a static body shadow is detected) and then clears to
  `NO`, so an empty room always returns to empty.
- **Static shadow is real now.** `static_shadow_db` (default 1.5 dB) drives the
  hold extension instead of a fake `presence_score = threshold + 5`.
- **ML no longer overrides the people count by default.** The geometric blob
  count is authoritative; ML only overrides when `count_ml_override` is `true`
  and confidence ≥ `count_ml_confidence_min`.
- **Sklearn feature-name warnings** removed by feeding named DataFrames when
  pandas is available (with a narrow warning filter fallback).
- **Training script can no longer destroy a working model.** It refuses to save
  when every row has the same label, maps legacy feature columns by position,
  and supports `--out-dir`.
- **ESP32 #2 CSI format** now emits comma-separated values like ESP32 #1.
- **Serial reader** accepts both comma- and space-separated CSI payloads and
  reconnects automatically if a board is unplugged.
- **`config.py`** tolerates a UTF-8 BOM in `config.json`.

### Added

- `tracker` section in `config.json` (threshold, hold, shadow, calibration,
  smoothing, step/deadband, heatmap and Kalman parameters).
- CLI flags `--hold` and `--shadow`; `--threshold` defaults from config.
- `secrets.h` support for both Arduino sketches, with committed
  `secrets.h.example` files and a `.gitignore` rule.
- `CSI/firmware_arduino/README.md` with credential and CSI-format instructions.
- `README.ar.md`, `LICENSE`, `CHANGELOG.md`, `.gitattributes`, `.editorconfig`.
- Real UI preview image at `CSI/assets/tracker_preview.png`.

### Changed

- Demo cycle extended from 36 s to 45 s (33 s movement + 12 s empty room) so
  presence clearing is observable.
- `PRESENTATION_DOCUMENTATION.md` numbers synced with the code (window 12,
  EMA 0.55/0.20, drop 7.0, instant 0.65/1.65, alpha 0.85/0.38, Kalman
  Q=0.10 / R=1e-3, sigma 5, threshold 50 %, cooldown 0.6 s). Removed the
  "Zero Blind Spots" claim and clarified the cross-recording caveat.
- `requirements.txt` now lists pandas, scikit-learn, joblib and matplotlib.
- Root and `CSI/` READMEs rewritten and reorganised.

### Security

- Wi-Fi credentials moved out of source files into gitignored `secrets.h`.
  Credentials that appeared in earlier commits should be rotated.

## [0.1.0] - 2025

- Initial dual-link heatmap tracker, scope monitor, simulator, RTI support and
  count-model pipeline.
