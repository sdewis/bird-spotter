# 🐦 British Bird Spotter

Autonomous British garden bird spotting, PTZ auto-framing photography, and analytics system powered by **Raspberry Pi 5 + Hailo-8 NPU + OBSBOT Tiny SE**.

---

## Key Features

1. **Two-Stage Detection & British Bird Species Identification**
   - **Stage 1 (30 FPS Localization)**: Runs [`yolov8s_h8.hef`](file:///usr/share/hailo-models/yolov8s_h8.hef) on the Hailo-8 NPU (~15ms/frame) with IoU + centroid multi-frame tracking.
   - **Stage 2 (707-Species Collins Bird Guide Classifier)**: Uses `birder-project/regnet_y_1_6g_eu-common` exported to [`models/uk_birds_regnet_y_1_6g.onnx`](file:///home/sean/Projects/bird-spotter/models/uk_birds_regnet_y_1_6g.onnx) (and optional Hailo-8 [`models/uk_birds_regnet_y_1_6g_h8.hef`](file:///home/sean/Projects/bird-spotter/models/uk_birds_regnet_y_1_6g_h8.hef)) with a curated ~85-species British garden & countryside mask ([`config/uk_birds.json`](file:///home/sean/Projects/bird-spotter/config/uk_birds.json)).
2. **PTZ Auto-Framing & Burst Sharpness Photography (`AutoFrameController`)**
   - Automatically pans, tilts (with zoom-scaled proportional gain), and dynamically zooms (`1.0x–4.0x`) so detected birds fill **25–40%** of the frame.
   - Waits for gimbal settling, captures a **5-frame burst**, picks the sharpest frame via **Laplacian variance**, saves a clean full-resolution JPEG + square thumbnail crop, and returns to the saved **Home / Feeder Preset (`1.0x`)**.
   - **Smart Photo Policy**: Immediately photographs any new species; for already-spotted species, saves a new photo if its composite quality score beats the existing best photo or after a configurable per-species cooldown (`300s`).
3. **Responsive Desktop & Mobile Web Dashboard**
   - Live MJPEG stream (`/stream.mjpg`) with toggleable HUD vs Clean view.
   - Manual PTZ D-Pad, Zoom slider (`1.0x–4.0x`), Home Preset controls (`Go Home`, `Set Home`), and `📸 Manual Snapshot` button.
   - Touching any manual PTZ control automatically pauses auto-framing for **60 seconds** (`MANUAL_OVERRIDE`) with a live countdown banner and instant resume button.
   - **🦉 Bird Dex (Life List)**: Best photograph, scientific name, peak confidence, and total sighting count per British species.
   - **🖼️ Captures & Review Modal**: Inspect full-resolution clean JPEGs, view top-3 classifier predictions, **Re-label** species, or **Delete** false positives.
   - **📈 Time Analytics**: 24-hour sightings-by-hour distribution and 14-day activity trends backed by SQLite (`data/bird_spotter.db`).

---

## Quick Start

### Remote Development from PC (`kbuntuPC` -> Raspberry Pi 5 `192.168.100.122`)

```bash
# 1. Run fast unit tests locally on the PC (.venv)
./scripts/pi5.sh test-local

# 2. Full deploy: run tests, rsync code to Pi 5, restart bird-spotter.service & verify health
./scripts/pi5.sh deploy

# 3. Other remote Pi 5 commands
./scripts/pi5.sh sync        # Rsync code to Pi 5 without restarting service
./scripts/pi5.sh check       # Run --check-hardware on Pi 5
./scripts/pi5.sh status      # Check systemd service + live /api/state telemetry
./scripts/pi5.sh logs        # View recent Pi 5 service logs
./scripts/pi5.sh pull-data   # Pull SQLite DB (data/) & snapshots/ from Pi 5 to PC
```

### Direct Execution on Raspberry Pi 5

```bash
# 1. Verify camera, Hailo-8 NPU, classifier models, and OBSBOT CLI
PYTHONPATH=src python3 -m bird_spotter.main --check-hardware

# 2. Run the live Bird Spotter server and web dashboard (port 8080)
PYTHONPATH=src python3 -m bird_spotter.main

# 3. Compile Hailo-8 .hef for the UK Bird Classifier on x86_64 host (kbuntuPC)
python3 scripts/compile_hailo_classifier.py --compile-hef --host 192.168.100.101 --user sean
```
