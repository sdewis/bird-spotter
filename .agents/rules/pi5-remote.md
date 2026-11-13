---
description: Remote Raspberry Pi 5 development, testing, and deployment workflow for British Bird Spotter
---

# PC-to-Raspberry Pi 5 Workflow (`bird-spotter`)

This repository is developed and edited locally on the PC (`/home/sean/Projects/bird-spotter`), while the live application, Hailo-8 NPU inference, OBSBOT Tiny SE PTZ camera, and systemd service (`bird-spotter.service`) run on the remote **Raspberry Pi 5** (`sean@192.168.100.122:/home/sean/Projects/bird-spotter`).

## Key Rules for Agents

1. **Always Edit Locally**:
   - Edit source files in `/home/sean/Projects/bird-spotter` on the local PC.
   - Do not edit over SFTP/KIO-FUSE mounts (`/run/user/1000/kio-fuse-...`).

2. **Fast Local Unit Testing**:
   - Use the local virtual environment (`.venv/bin/python`) for rapid unit testing on the PC:
     ```bash
     ./scripts/pi5.sh test-local
     ```

3. **Syncing, Remote Testing, Hardware Checks & Deployment on the Pi 5**:
   - Hardware devices (`/dev/video0`, Hailo-8 NPU, `obsbot-cli`) and the live web dashboard (`http://192.168.100.122:8080`) exist **only on the Pi 5**.
   - Use `./scripts/pi5.sh` to sync files and run commands on the Pi 5 over SSH:
     - `./scripts/pi5.sh sync` — Rsync code/config/tests to the Pi 5 (automatically preserves `data/`, `snapshots/`, and heavy `models/*.{onnx,pt,hef}` on the Pi 5).
     - `./scripts/pi5.sh test` — Run unit tests locally on the PC and remotely on the Pi 5.
     - `./scripts/pi5.sh check` — Sync and run `python3 -m bird_spotter.main --check-hardware` on the Pi 5.
     - `./scripts/pi5.sh deploy` — Run tests, sync code to the Pi 5, restart `bird-spotter.service`, and verify live dashboard health.
     - `./scripts/pi5.sh status` — Check `bird-spotter.service` status and `/api/state` telemetry on the Pi 5.
     - `./scripts/pi5.sh logs` — View recent `journalctl -u bird-spotter.service` output from the Pi 5.
     - `./scripts/pi5.sh pull-data` — Pull the latest `data/bird_spotter.db` and `snapshots/` from the Pi 5 to the PC when inspecting captures or database records locally.
     - `./scripts/pi5.sh exec "<command>"` — Run any custom shell command in `/home/sean/Projects/bird-spotter` on the Pi 5.

4. **Protect Live Pi 5 State**:
   - Never overwrite `data/bird_spotter.db`, `data/home_preset.json`, or `snapshots/` on the Pi 5 when syncing code from the PC.
