# Hardware Setup and Bill of Materials

This page details the hardware components, physical mounting recommendations, electrical considerations, and driver setup required to run **British Bird Spotter**.

---

## 1. Bill of Materials (BOM)

| Component | Specification | Purpose |
| :--- | :--- | :--- |
| **SBC** | **Raspberry Pi 5** (8GB or 4GB) | Main host running computer vision, FSM tracker, AI classifier, web dashboard, and SQLite storage. |
| **NPU Accelerator** | **Hailo-8 M.2 Module** (26 TOPS) | Runs Stage 1 YOLOv8s bird localization at 30 FPS with ~13ms latency and <2W power draw. |
| **HAT Adapter** | **Raspberry Pi M.2 HAT+** (or Dual M.2 / AI HAT+) | Connects the Hailo-8 M.2 module to the Raspberry Pi 5 PCIe 2.0/3.0 ribbon interface. |
| **Camera** | **OBSBOT Tiny SE** (1080p 30/60 FPS) | 2-axis motorized brushless gimbal (pan $\pm 130^\circ$, tilt $\pm 90^\circ$), autofocus, and UVC control. |
| **Power Supply** | **Official Raspberry Pi 27W USB-C PSU** (5.1V / 5.0A) | Crucial for powering the Pi 5, active cooler, Hailo-8 NPU, and OBSBOT Tiny SE gimbal motors simultaneously without brownouts. |
| **Cooling** | **Raspberry Pi 5 Active Cooler** | Fan + anodized heatsink. Keeps the Broadcom BCM2712 CPU below $60^\circ\text{C}$ during 24/7 vision workloads. |
| **Storage** | **SanDisk Extreme Pro A2 64GB+** or NVMe SSD | High write endurance for SQLite WAL sightings and full-resolution JPEG captures. |
| **Enclosure** | Weatherproof camera housing / sheltered mounting | Protects the OBSBOT Tiny SE (rated for indoor/sheltered use) from rain and direct wind. |

---

## 2. Physical Placement & Optical Optimization

The OBSBOT Tiny SE features a fixed-focal-length lens with a selectable Field of View ($65^\circ$ Narrow, $78^\circ$ Medium, $86^\circ$ Wide).

> [!IMPORTANT]
> Because the camera uses **digital sensor zoom** rather than moving optical glass, physical camera placement directly dictates how many optical pixels cover the bird.

* **Recommended Distance**: Place the camera **0.8 meters to 1.5 meters** from the main bird feeder, perch, or bath.
  * At **1.0 meter**, a typical 10 cm Blue Tit fills $\sim 150 \times 150$ optical sensor pixels ($\sim 22,500$ pixels).
  * At **3.0 meters**, that same bird fills only $\sim 50 \times 50$ optical sensor pixels ($\sim 2,500$ pixels)—forcing the digital zoom to interpolate from minimal data.
* **Camera Field of View**: The system locks the camera to **$65^\circ$ Narrow FOV** on startup (`fov = 65`), maximizing optical magnification before any software digital zoom.
* **Sun & Lighting**: Position the camera facing North or East to avoid direct backlighting from the low afternoon winter sun. Ensure the feeder is well-lit for fast shutter speeds ($<1/500\text{s}$) to freeze bird motion.

---

## 3. Raspberry Pi 5 & Hailo-8 Setup

### Enable PCIe Gen 3

Add the following to `/boot/firmware/config.txt` to unlock maximum PCIe throughput between the BCM2712 SoC and the Hailo-8 NPU:

```ini
# Enable PCIe external connector
dtparam=pciex1

# Optional: Enable PCIe Gen 3.0 speeds (8 GT/s)
dtparam=pciex1_gen=3
```

Reboot the Raspberry Pi:
```bash
sudo reboot
```

### Verify Hailo-8 Device Enumeration

Verify that the PCIe device and HailoRT drivers are loaded:

```bash
# Check PCIe bus
lspci -nn | grep -i hailo
# Expected output: 0001:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor [1e60:2864]

# Run HailoRT CLI scan
hailortcli scan
# Expected output:
# Hailo Devices:
# [-] Device: 0001:01:00.0
```

Verify Hailo model files exist in `/usr/share/hailo-models/`:
```bash
ls -la /usr/share/hailo-models/yolov8s_h8.hef
```

---

## 4. OBSBOT Tiny SE Configuration & UVC Driver

The OBSBOT Tiny SE enumerates over standard USB Video Class (UVC) and USB Audio Class (UAC).

### Device Node Discovery

The camera typically appears at `/dev/video0`. If multiple cameras or capture devices are attached, the driver dynamically resolves `/dev/v4l/by-id/*OBSBOT*index0`.

Verify supported formats with `v4l2-ctl`:
```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```
The camera supports:
* `1920x1080` @ 30 / 60 FPS MJPEG (Used for Bird Spotter)
* `1280x720` @ 30 / 60 FPS MJPEG
* `640x360` / `640x480` YUYV raw

### UVC Hardware Controls

Bird Spotter controls pan, tilt, zoom, focus, and ISP sharpness directly via standard V4L2 UVC controls:

| Control | UVC Parameter | Values | Bird Spotter Setting |
| :--- | :--- | :--- | :--- |
| **Pan** | `pan_absolute` | -468,000 to +468,000 ($3600\text{ units} = 1^\circ$) | Dynamic ($0$ default) |
| **Tilt** | `tilt_absolute` | -324,000 to +324,000 ($3600\text{ units} = 1^\circ$) | Dynamic ($0$ default) |
| **Zoom** | `zoom_absolute` | 0 to 12 ($0.25\times$ per hardware step) | **Capped at $\le 1.8\times$** (steps 0..3) |
| **Focus Mode** | `focus_automatic_continuous` | 0 (Manual) or 1 (Auto) | `1` (Continuous Autofocus) |
| **Focus Range** | `focus_absolute` | 0 to 100 | Calibrated per preset |
| **ISP Sharpness** | `sharpness` | 0 to 100 (factory default 74) | **`45`** (tuned to remove white ringing) |

### Built-In Human AI Tracking Disablement

By default, the OBSBOT Tiny SE firmware attempts to detect human faces and lock exposure/tracking to people. Bird Spotter disables this via the OBSBOT SDK (`set_ai_mode(AI_MODE_NONE)` and `set_face_ae(False)`) and the companion CLI tool `/usr/local/bin/obsbot-cli` so the gimbal exclusively obeys wildlife tracking.
