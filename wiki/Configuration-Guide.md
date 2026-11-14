# Configuration Guide

All core operational parameters are defined in [`config/settings.toml`](file:///home/sean/Projects/bird-spotter/config/settings.toml) and runtime presets in `data/home_preset.json`.

---

## 1. Complete `settings.toml` Reference

```toml
[camera]
device = "/dev/video0"    # V4L2 device path or /dev/v4l/by-id/
width = 1920             # Sensor capture width
height = 1080            # Sensor capture height
fps = 30                 # Target capture frame rate
format = "MJPG"          # Video pixel format: MJPG (1080p 30/60fps)

[camera.ptz]
enabled = true           # Enable PTZ gimbal tracking
pan_min = -468000        # UVC minimum pan limit (-130 deg * 3600)
pan_max = 468000         # UVC maximum pan limit (+130 deg * 3600)
pan_step = 3600          # 1 degree per UVC step
tilt_min = -324000       # UVC minimum tilt limit (-90 deg * 3600)
tilt_max = 324000        # UVC maximum tilt limit (+90 deg * 3600)
tilt_step = 3600
deadzone_norm = 0.08     # Center deadband (8% of screen) to prevent hunting
kp_pan = 45000.0         # Proportional gain for horizontal pan tracking
kp_tilt = 35000.0        # Proportional gain for vertical tilt tracking
invert_pan = true        # Gimbal mirror inversion
invert_tilt = true
default_zoom = 1.0       # Initial zoom on cold boot
zoom_min = 1.0           # Minimum hardware zoom (1.0x native sensor)
zoom_max = 1.8           # Maximum zoom cap (prevents digital pixelation)
zoom_step = 0.25         # Hardware step increment
home_pan = 0             # Fallback home pan position
home_tilt = 0            # Fallback home tilt position
home_zoom = 1.0          # Fallback home zoom
home_fov = 65            # Hardware FOV: 65 (Narrow), 78 (Medium), 86 (Wide)
autofocus = true         # Continuous lens autofocus
focus = 50               # Manual focus fallback value (0..100)
sharpness = 45           # Hardware ISP sharpness (45 removes halo ringing)

[hailo]
hef_path = "/usr/share/hailo-models/yolov8s_h8.hef" # NPU model path
score_threshold = 0.35   # Minimum YOLO detection confidence
iou_threshold = 0.45     # Non-Maximum Suppression (NMS) IoU
target_classes = ["bird"]# COCO classes to detect

[classifier]
hef_path = "models/uk_birds_regnet_y_1_6g_h8.hef"  # Hailo-8 classifier HEF (if compiled)
onnx_path = "models/uk_birds_regnet_y_1_6g.onnx"   # OpenCV DNN classifier ONNX
labels_path = "models/eu_common_labels.json"       # 707 Collins Bird Guide labels
uk_birds_path = "config/uk_birds.json"             # 85-species UK curated mask
uk_filter_enabled = true # Filter non-UK species
min_confidence = 0.25    # Below 25% marked as 'Unidentified Bird'

[spotter]
auto_framing = true      # Automatic PTZ framing state machine
auto_snapshot = true     # Automatic photography on valid sightings
show_preview_bbox = true # Draw bounding box on dashboard live preview
show_capture_bbox = true # Draw identification banner on saved snapshot
target_min_area = 0.03   # Target min bird area (3% of frame)
target_max_area = 0.12   # Target max bird area (12% of frame)
settle_delay_sec = 0.30  # Gimbal settle delay before burst capture
burst_count = 5          # Number of burst frames to capture
max_framing_sec = 4.0    # Maximum framing timeout before capturing burst
lost_timeout_sec = 1.2   # Target lost timeout before returning Home
return_home_settle_sec = 2.0 # Settling time at Home preset
manual_pause_sec = 60.0  # Seconds to pause auto-framing on manual D-Pad touch
per_species_cooldown_sec = 300.0 # 5 min cooldown before re-photographing same species
snapshots_dir = "snapshots"      # Directory for saved photos & thumbs
db_path = "data/bird_spotter.db" # SQLite database path
confidence_threshold = 0.50
super_resolution = true  # Enable AI super-resolution
sr_model = "fsrcnn"      # Model: 'fsrcnn' (sharp) or 'espcn' (fast)
sr_scale = 4             # Upscaling factor (4x)
burst_stacking = true    # Multi-frame sub-pixel burst fusion

[dashboard]
host = "0.0.0.0"         # Listen address
port = 8080              # Web server port

[solar]
enabled = true           # Solar schedule: dawn wake / dusk sleep
auto_location = true     # Auto-detect latitude/longitude from IP
latitude = 52.3555       # Fallback coordinates (Central England)
longitude = -1.1743
wake_before_dawn_min = 30 # Wake camera 30 mins before civil dawn
sleep_after_sunset_min = 30 # Put camera to sleep 30 mins after sunset
```

---

## 2. Calibrated Preset File (`data/home_preset.json`)

When you click **"Set Home"** or configure a **Zone of Interest (ROI)** in the web dashboard, the pose is persisted to `data/home_preset.json`:

```json
{
  "home_pan": 18000,
  "home_tilt": 39600,
  "home_zoom": 1.5,
  "home_fov": 65,
  "home_autofocus": true,
  "home_focus": 94,
  "roi_enabled": true,
  "roi_x": 0.1002,
  "roi_y": 0.0,
  "roi_w": 0.7302,
  "roi_h": 0.7882,
  "roi_ref_pan": 18000,
  "roi_ref_tilt": 39600,
  "roi_ref_zoom": 1.5
}
```

* **World-Locked ROI**: When the camera pans or zooms during manual inspection, the dashboard projects the screen bounding box back to world space using `roi_ref_pan`, `roi_ref_tilt`, and `roi_ref_zoom`.

---

## 3. Practical Tuning Scenarios

### Tuning for Windy Days & Blowing Foliage
* **Problem**: Trees and waving branches generate occasional false detections.
* **Fix**:
  1. Open the dashboard and click **🎯 Select Zone** to draw a box strictly around the feeder tray.
  2. Increase `score_threshold = 0.45` in `[hailo]`.

### Tuning for Skittish Birds (Fast Visitors)
* **Problem**: Fast birds like Coal Tits grab a seed and leave within 1.5 seconds.
* **Fix**:
  1. Lower `settle_delay_sec = 0.20`.
  2. Set `burst_count = 3`.
  3. Ensure `zoom_max = 1.5` so framing settles instantly.
