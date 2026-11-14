# Troubleshooting & FAQ

Frequently asked questions and solutions to common hardware and software issues.

---

## 1. Troubleshooting Common Issues

### Issue 1: Camera Device `/dev/video0` Missing or Fails to Open
* **Symptoms**: Service logs report `ERROR: Could not open camera /dev/video0`.
* **Causes**:
  * The camera was connected after boot, or the USB port had a momentary power droop.
  * In rare cases, the OBSBOT Tiny SE enumerates as `/dev/video1` or `/dev/video2`.
* **Fix**:
  1. Check device listing: `ls -la /dev/video* /dev/v4l/by-id/`
  2. Verify USB connection: `lsusb | grep -i obsbot`
  3. Bird Spotter includes an auto-fallback routine that inspects `/dev/v4l/by-id/*OBSBOT*index0`. Ensure the camera is plugged into a **blue USB 3.0 port** on the Pi 5.

---

### Issue 2: Hailo-8 NPU Not Detected (`hailortcli scan` reports 0 devices)
* **Symptoms**: `check_hardware` fails on Hailo-8 detector.
* **Causes**: PCIe ribbon cable is unseated, or PCIe is not enabled in firmware.
* **Fix**:
  1. Verify `/boot/firmware/config.txt` contains:
     ```ini
     dtparam=pciex1
     dtparam=pciex1_gen=3
     ```
  2. Check PCIe enumeration: `lspci | grep -i hailo`. If missing, power off and reseat both ends of the PCIe FPC ribbon cable carefully.
  3. Verify kernel driver: `dmesg | grep -i hailo`.

---

### Issue 3: Captures Appear Pixelated or Blurry
* **Causes**:
  * The camera is placed too far from the feeder ($>2.5\text{m}$), forcing digital zoom.
  * Hardware sharpness is set too high in camera firmware.
  * The lens front glass is dirty or has rain spots.
* **Fix**:
  1. Verify that `zoom_max = 1.8` in `config/settings.toml` (do not set zoom above 2.0x on a fixed-lens webcam).
  2. Check hardware sharpness: `v4l2-ctl -d /dev/video0 --get-ctrl sharpness`. It should be `45`.
  3. Move the camera physically closer to the feeder ($1.0\text{m} - 1.5\text{m}$). This gives $4\times$ to $9\times$ more genuine optical sensor pixels on the bird.
  4. Ensure `super_resolution = true` and `burst_stacking = true` are active in `config/settings.toml`.

---

### Issue 4: PTZ Gimbal Wanders or Focuses on Human Faces
* **Symptoms**: The camera tilts away from birds towards people in the background.
* **Causes**: OBSBOT's built-in human AI tracking was triggered.
* **Fix**:
  Bird Spotter automatically disables human tracking on startup. If it was re-enabled manually via the OBSBOT app, reset it via the CLI:
  ```bash
  /usr/local/bin/obsbot-cli stop
  /usr/local/bin/obsbot-cli sleep
  /usr/local/bin/obsbot-cli wake
  ```

---

### Issue 5: Raspberry Pi 5 Throttles or Overheats
* **Symptoms**: FPS drops below 10 FPS, or `vcgencmd get_throttled` returns non-zero.
* **Causes**: Ambient temperature or inadequate cooling.
* **Fix**:
  * Install the official **Raspberry Pi 5 Active Cooler**.
  * Check temperature: `vcgencmd measure_temp`. Temperatures should remain below $65^\circ\text{C}$ under full vision workload.

---

## 2. Frequently Asked Questions (FAQ)

### Does Bird Spotter require an active internet connection?
**No.** All neural networks (YOLOv8s on Hailo-8, RegNet-Y-1.6G classifier on CPU, and FSRCNN super-resolution) run **100% locally** on the Raspberry Pi 5. The web dashboard is served over your local home network (LAN). An internet connection is only needed if you use optional solar scheduler auto-geolocation.

### Can I use another camera model?
Yes, any standard V4L2 USB camera can be used for fixed detection. However, motorized pan, tilt, FOV switching, and focus controls require a camera supporting standard UVC Pan/Tilt controls or the OBSBOT SDK (such as OBSBOT Tiny 4K, Tiny 2, or Tiny SE).

### What happens when the disk fills up?
Full-resolution snapshots are $\sim 150\text{ KB} - 300\text{ KB}$ each, and 320x320 thumbnails are $\sim 15\text{ KB}$. A 64GB SD card can hold over **150,000 sightings**. You can delete unneeded sightings or batch-delete species directly from the web dashboard.

### How do I correct a misidentified bird?
Open the **🖼️ Captures Gallery** in the web dashboard, click on the capture card, and click **🏷️ Re-label**. You can choose any of the curated British bird species from the dropdown. The update immediately recalculates the species Life List (Bird Dex) and statistics.
