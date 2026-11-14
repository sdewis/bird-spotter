# Web Dashboard & REST API

The British Bird Spotter dashboard is a lightweight, responsive web interface served on port `8080` (e.g. `http://<pi-ip>:8080`). It requires no external internet connection or cloud service.

---

## 1. Web Dashboard Features

```text
+--------------------------------------------------------------------------+
| 🐦 British Bird Spotter          [STANDBY / ACTIVE] [HUD: ON] [Day/Night] |
+--------------------------------------------------------------------------+
| [ LIVE MJPEG FEED: 1080p 30 FPS ]       |  PTZ Controls:                 |
|                                         |  [^] Pan/Tilt D-Pad            |
|   Target Bounding Box Overlay           |  [<]  [o]  [>]                 |
|   Zone of Interest (ROI) Box            |  [v]                           |
|   HUD Telemetry: FPS, Infer ms, FSM     |  Zoom: [1.0x ====== 1.8x]      |
|                                         |  FOV:  [65°] [78°] [86°]       |
|                                         |  [ Go Home ]  [ Set Home ]     |
|                                         |  [ 📸 Manual Snapshot ]        |
+-----------------------------------------+--------------------------------+
| 🦉 Bird Dex (Life List):                                                 |
| [Blue Tit: 98%]  [Robin: 94%]  [Goldfinch: 89%]  [Great Spotted Woodpecker]|
+--------------------------------------------------------------------------+
| 🖼️ Recent Captures Gallery:                                              |
| [Card 1] [Card 2] [Card 3] [Batch Relabel Selected] [Delete]             |
+--------------------------------------------------------------------------+
| 📈 Analytics: [24h Sightings by Hour]     [14-Day Activity Trends]       |
+--------------------------------------------------------------------------+
| 📋 Real-Time Execution Log Console (Collapsible)                         |
+--------------------------------------------------------------------------+
```

### Key UI Capabilities
* **Interactive Zone of Interest (ROI) Selector**: Click **🎯 Select Zone** on the video feed to draw an area bounding box. Detections outside this box are ignored.
* **Deep-Zoom Capture Modal**: Click any card in the gallery or Bird Dex to open the full-resolution snapshot. Supports mouse scroll-wheel zoom, click-drag pan, magnifying glass, and mobile pinch-to-zoom.
* **Manual Override Safe Window**: Touching any PTZ control immediately suspends auto-framing for **60 seconds**, displaying a live countdown timer and instant "Resume Tracking" button.
* **Batch Re-labeling**: Select multiple capture cards and reclassify them to another British species simultaneously.

---

## 2. REST API Reference

### `GET /api/state`
Returns complete real-time telemetry, PTZ pose, FSM state, analytics, and Dex summary.

**Response Example:**
```json
{
  "fps": 16.7,
  "infer_ms": 13.1,
  "resolution": "1920x1080",
  "zoom": 1.0,
  "fov": 65,
  "autofocus": true,
  "focus": 50,
  "roi": {
    "enabled": true,
    "x": 0.1,
    "y": 0.0,
    "w": 0.73,
    "h": 0.78
  },
  "fsm_state": "WATCHING_HOME",
  "fsm_label": "WATCHING (HOME 1.0x)",
  "manual_override_remaining": 0.0,
  "species_dex": [
    {
      "species": "Blue Tit",
      "scientific_name": "Cyanistes caeruleus",
      "best_confidence": 0.94,
      "total_sightings": 14,
      "best_thumb_filename": "thumbs/thumb_20261114_085507_blue_tit.jpg"
    }
  ]
}
```

---

### `POST /api/control`
Executes gimbal motion, presets, camera power, and overlay toggles.

**Payload Actions:**
```json
// Gimbal Pan/Tilt Nudge
{"action": "nudge", "pan": 7200, "tilt": -3600}

// Absolute Zoom
{"action": "set_zoom", "zoom": 1.5}

// Field of View
{"action": "set_fov", "fov": 65}

// Set/Go Home Preset
{"action": "go_home"}
{"action": "save_home"}

// Zone of Interest (ROI)
{"action": "set_roi", "x": 0.15, "y": 0.1, "w": 0.6, "h": 0.7}
{"action": "clear_roi"}

// Resume Auto-Framing from Manual Override
{"action": "resume_tracking"}

// Toggle Overlays
{"action": "toggle_preview_bbox"}
{"action": "toggle_capture_bbox"}

// Power & Day/Night Mode
{"action": "toggle_system_power"}
{"action": "activate_day_mode"}
{"action": "activate_night_mode"}
```

---

### `POST /api/snapshot`
Forces an immediate clean full-resolution snapshot, passes the bird through AI super-resolution, runs classification, and logs the sighting to SQLite.

**Response:**
```json
{
  "success": true,
  "filename": "bird_20261114_091500_123_blue_tit.jpg",
  "species": "Blue Tit",
  "sighting": {"id": 1, "species": "Blue Tit", "confidence": 0.92}
}
```

---

### Species Management Endpoints

| Endpoint | Method | Payload | Description |
| :--- | :--- | :--- | :--- |
| `/api/catalog` | `GET` | *None* | Returns the ~85 curated British bird species list. |
| `/api/species_sightings` | `GET` | `?species=Robin` | Returns all sighting history for a species. |
| `/api/relabel_sighting` | `POST` | `{"sighting_id": 42, "new_species": "Coal Tit"}` | Updates classification of a single record. |
| `/api/batch_relabel` | `POST` | `{"sighting_ids": [42, 43], "new_species": "Coal Tit"}` | Batch updates multiple records. |
| `/api/delete_sighting` | `POST` | `{"sighting_id": 42}` | Deletes record and disk image files. |
| `/api/delete_species` | `POST` | `{"species": "Woodpigeon"}` | Deletes all records and photos for a species. |
| `/snapshots/<path>` | `GET` | *None* | Serves static full JPEG snapshots and thumbnails. |
