# Halloween — Interactive Bipolar Animatronic Skull 💀🎃

An interactive animatronic skull powered by a **Raspberry Pi 5 + Hailo-8 AI HAT**, an **OBSBOT Tiny SE** USB 3 PTZ camera, and a networked **Raspberry Pi 2 + Picon Zero** servo controller.

## Hardware Inventory & Verified Status

| Component | Device / Node | Status |
| :--- | :--- | :--- |
| **Brain & CV Accelerator** | Raspberry Pi 5 + **Hailo-8** (`0001:01:00.0`, FW 4.23.0) | Verified Online |
| **Vision & Primary PTZ** | **OBSBOT Tiny SE** (`/dev/video0`, 1080p/720p MJPG + UVC Pan/Tilt/Zoom) | Verified Online |
| **Microphone Input** | OBSBOT Tiny SE USB Audio (`hw:2,0`) + optional external USB Mic | Verified Online |
| **Audio Output** | **Samsung Level Box** speaker (via 3.5mm Phono on RPi2 or Bluetooth on RPi5) | Ready |
| **Skull Motion Controller** | **Raspberry Pi 2 + Picon Zero HAT** (`192.168.100.121`, I2C `0x22` FW `[11, 2]`) | Verified Online |

## Project Structure

```text
halloween/
├── config/
│   └── settings.toml          # Hardware, Hailo-8, PTZ, Servo, and Bipolar thresholds
├── scripts/
│   └── discover_pi2.py        # Subnet scanner to locate the Raspberry Pi 2 over SSH
├── src/halloween/
│   ├── main.py                # CLI entry point & hardware self-check
│   ├── config.py              # TOML configuration loader
│   ├── vision/
│   │   ├── detector.py        # Hailo-8 YOLOv8 object detection & target tracking
│   │   └── ptz.py             # OBSBOT Tiny SE UVC Pan/Tilt/Zoom controller
│   ├── personality/
│   │   └── engine.py          # Bipolar state machine (SWEET / MENACING / CHAOTIC)
│   ├── servos/
│   │   ├── client.py          # Low-latency socket client on RPi5
│   │   └── pi2_daemon.py      # Standalone Picon Zero servo daemon for RPi2
│   ├── audio/
│   │   └── speaker.py         # Hybrid pre-baked WAV + TTS playback & jaw sync
│   └── dashboard/
│       └── server.py          # Browser live MJPEG stream & manual override UI
└── pyproject.toml
```

## Quick Start

```bash
# Run hardware self-check using system Python (which includes Hailo + OpenCV + GStreamer)
PYTHONPATH=src python3 -m halloween.main --check-hardware

# Center the OBSBOT Tiny SE PTZ gimbal
PYTHONPATH=src python3 -m halloween.main --center-ptz

# Scan local network for the Raspberry Pi 2
python3 scripts/discover_pi2.py
```
