# Deployment & Operations

This guide covers systemd service management, remote development workflow from a PC, data backups, and routine maintenance.

---

## 1. Systemd Service Management

Bird Spotter runs as a continuous system service on the Raspberry Pi 5.

### Service Unit File (`/etc/systemd/system/bird-spotter.service`)

```ini
[Unit]
Description=British Bird Spotter — Hailo-8 & OBSBOT PTZ Web Dashboard
After=network.target multi-user.target

[Service]
Type=simple
User=sean
WorkingDirectory=/home/sean/Projects/bird-spotter
Environment=PYTHONPATH=/home/sean/Projects/bird-spotter/src
ExecStart=/usr/bin/python3 -m bird_spotter.main
Restart=always
RestartSec=5
KillMode=mixed
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
```

### Essential Systemctl Commands

```bash
# Check status and recent logs
sudo systemctl status bird-spotter.service

# Restart service (e.g. after code change)
sudo systemctl restart bird-spotter.service

# Stop service cleanly
sudo systemctl stop bird-spotter.service

# View live streaming logs
journalctl -u bird-spotter.service -f -n 50
```

---

## 2. Remote Development Workflow (`scripts/pi5.sh`)

The project includes [`scripts/pi5.sh`](file:///home/sean/Projects/bird-spotter/scripts/pi5.sh) to streamline development between a host PC (`kbuntuPC`) and the Raspberry Pi 5 (`192.168.100.122`).

| Command | Action |
| :--- | :--- |
| `./scripts/pi5.sh deploy` | Runs local tests, runs Pi 5 tests, rsyncs code, restarts `bird-spotter.service`, and checks health. |
| `./scripts/pi5.sh test-local` | Executes unit tests locally in `.venv` (completes in $<1.0\text{s}$). |
| `./scripts/pi5.sh test` | Runs local tests, syncs to Pi 5, and runs unit tests on the Pi 5. |
| `./scripts/pi5.sh sync` | Rsyncs source code, models, and config to Pi 5 without restarting the service. |
| `./scripts/pi5.sh check` | Syncs and runs `--check-hardware` on the Pi 5. |
| `./scripts/pi5.sh status` | Inspects `systemctl status` and curls `http://<pi-ip>:8080/api/state`. |
| `./scripts/pi5.sh logs -f` | Streams remote journalctl logs directly in your PC terminal. |
| `./scripts/pi5.sh pull-data` | Rsyncs SQLite database (`data/`) and snapshots (`snapshots/`) from Pi 5 to PC. |

---

## 3. Database & Storage Maintenance

### Backup Procedure

To take an instant compressed backup of the SQLite database and captured photographs:

```bash
mkdir -p backups
tar -czf backups/backup_$(date +%Y%m%d_%H%M%S).tar.gz data/bird_spotter.db snapshots/
```

### Resetting for a Clean Slate

To clear the sighting history while preserving calibrated Home/Feeder presets (`home_preset.json`):

```bash
# 1. Stop service
sudo systemctl stop bird-spotter.service

# 2. Clear photos and thumbnails
rm -rf snapshots/*
mkdir -p snapshots/thumbs

# 3. Reinitialize SQLite database
rm -f data/bird_spotter.db*
python3 -c "from bird_spotter.db import BirdDatabase; BirdDatabase('data/bird_spotter.db')"

# 4. Restart service
sudo systemctl start bird-spotter.service
```

---

## 4. Solar Twilight Scheduler

Bird Spotter computes civil twilight, dawn, and dusk dynamically based on latitude/longitude (defaulting to central England $52.35^\circ\text{N}, 1.17^\circ\text{W}$).

* **Dawn Wake**: 30 minutes before civil dawn (`wake_before_dawn_min = 30`), the scheduler calls `ptz.wakeup_camera()`, moving the gimbal from resting position to the feeder preset.
* **Dusk Sleep**: 30 minutes after sunset (`sleep_after_sunset_min = 30`), the scheduler calls `ptz.standby_camera()`, pointing the camera downwards into sleep mode to conserve power and protect the sensor from nighttime moisture.
* **Web UI Display**: While in night mode, the live stream displays a dark sleep banner and camera controls are greyed out to prevent accidental motor wakeups.
