"""Main entry point for the British Bird Spotter application."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import threading
import time
from typing import Any

import cv2

from bird_spotter.config import PROJECT_ROOT, load_config
from bird_spotter.dashboard.server import (
    DashboardServer,
    SNAPSHOTS_DIR,
    draw_capture_bbox,
    draw_overlay,
)
from bird_spotter.db import BirdDatabase, compute_quality_score
from bird_spotter.logger import spotter_log
from bird_spotter.solar import SolarScheduler
from bird_spotter.vision.classifier import BirdSpeciesClassifier
from bird_spotter.vision.detector import (
    HailoDetector,
    compute_crop_sharpness,
    extract_bird_crop,
)
from bird_spotter.vision.framing import (
    AutoFrameController,
    BurstCandidate,
    save_clean_snapshot_and_thumb,
)
from bird_spotter.vision.ptz import OBSBOT_CLI, ObsbotPTZ


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _find_camera_device(preferred: str = "/dev/video0") -> str:
    """Find the OBSBOT Tiny SE or active UVC video device, falling back from preferred device."""
    pref_path = Path(preferred)
    if pref_path.exists():
        return preferred
    # Search /dev/v4l/by-id first for OBSBOT
    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        for p in sorted(by_id.glob("*OBSBOT*index0*")):
            if p.exists():
                return str(p.resolve())
        for p in sorted(by_id.glob("*index0*")):
            if p.exists():
                return str(p.resolve())
    # Next try /dev/video1, /dev/video0, etc.
    for dev in ("/dev/video0", "/dev/video1", "/dev/video2"):
        if Path(dev).exists():
            return dev
    return preferred


def check_hardware(cfg: dict) -> int:
    """Verify camera device, Hailo-8 HEF models, classifier metadata, and OBSBOT PTZ CLI."""
    cam_dev = cfg["camera"]["device"]
    hef_path = cfg["hailo"]["hef_path"]
    cls_cfg = cfg.get("classifier", {})
    dash_cfg = cfg["dashboard"]

    cam_ok = Path(cam_dev).exists()
    detector = HailoDetector(hef_path, cfg["hailo"]["score_threshold"])
    hef_ok = detector.is_available()
    cli_ok = Path(OBSBOT_CLI).is_file()

    cls_hef = _resolve_path(cls_cfg.get("hef_path", "models/uk_birds_regnet_y_1_6g_h8.hef"))
    cls_onnx = _resolve_path(cls_cfg.get("onnx_path", "models/uk_birds_regnet_y_1_6g.onnx"))
    cls_labels = _resolve_path(cls_cfg.get("labels_path", "models/eu_common_labels.json"))

    print("=== Bird Spotter Hardware & Model Self-Check ===")
    print(f"  Camera Device ({cam_dev}): {'OK' if cam_ok else 'MISSING (check USB connection)'}")
    print(f"  Hailo-8 YOLOv8 Detector ({hef_path}): {'OK' if hef_ok else 'MISSING'}")
    print(
        f"  UK Bird Classifier HEF ({cls_hef.name}): "
        f"{'OK' if cls_hef.is_file() else 'PENDING (.onnx fallback if present)'}"
    )
    print(
        f"  UK Bird Classifier ONNX ({cls_onnx.name}): "
        f"{'OK' if cls_onnx.is_file() else 'MISSING'}"
    )
    print(
        f"  Collins Bird Guide Labels ({cls_labels.name}): "
        f"{'OK' if cls_labels.is_file() else 'MISSING'}"
    )
    print(f"  OBSBOT CLI Tool ({OBSBOT_CLI}): {'OK' if cli_ok else 'MISSING'}")
    print(f"  Web Dashboard Target: http://{dash_cfg['host']}:{dash_cfg['port']}")

    return 0 if (cam_ok and hef_ok) else 1


def run_spotter(cfg: dict, duration_sec: float | None = None) -> int:
    """Run live bird detection, OBSBOT PTZ auto-framing, species classification, and web dashboard."""
    cam_cfg = cfg["camera"]
    ptz_cfg = cam_cfg.get("ptz", {})
    hailo_cfg = cfg["hailo"]
    cls_cfg = cfg.get("classifier", {})
    spot_cfg = cfg.get("spotter", {})
    dash_cfg = cfg["dashboard"]
    solar_cfg = cfg.get("solar", {})

    solar = SolarScheduler(
        enabled=solar_cfg.get("enabled", True),
        auto_location=solar_cfg.get("auto_location", True),
        latitude=float(solar_cfg.get("latitude", 52.3555)),
        longitude=float(solar_cfg.get("longitude", -1.1743)),
        wake_before_dawn_min=int(solar_cfg.get("wake_before_dawn_min", 30)),
        sleep_after_sunset_min=int(solar_cfg.get("sleep_after_sunset_min", 30)),
    )
    solar.refresh_async()

    snapshots_dir = _resolve_path(spot_cfg.get("snapshots_dir", SNAPSHOTS_DIR))
    db_path = _resolve_path(spot_cfg.get("db_path", "data/bird_spotter.db"))
    db = BirdDatabase(db_path)

    cam_dev = _find_camera_device(cam_cfg.get("device", "/dev/video0"))
    cam_cfg["device"] = cam_dev

    ptz = ObsbotPTZ(
        device=cam_dev,
        pan_min=ptz_cfg.get("pan_min", -468000),
        pan_max=ptz_cfg.get("pan_max", 468000),
        pan_step=ptz_cfg.get("pan_step", 3600),
        tilt_min=ptz_cfg.get("tilt_min", -324000),
        tilt_max=ptz_cfg.get("tilt_max", 324000),
        tilt_step=ptz_cfg.get("tilt_step", 3600),
        deadzone_norm=ptz_cfg.get("deadzone_norm", 0.08),
        kp_pan=ptz_cfg.get("kp_pan", 45000.0),
        kp_tilt=ptz_cfg.get("kp_tilt", 35000.0),
        invert_pan=ptz_cfg.get("invert_pan", True),
        invert_tilt=ptz_cfg.get("invert_tilt", True),
        tracking_enabled=ptz_cfg.get("enabled", True),
        zoom=ptz_cfg.get("default_zoom", 1.0),
        zoom_min=ptz_cfg.get("zoom_min", 1.0),
        zoom_max=ptz_cfg.get("zoom_max", 4.0),
        zoom_step=ptz_cfg.get("zoom_step", 0.25),
        home_pan=ptz_cfg.get("home_pan", 0),
        home_tilt=ptz_cfg.get("home_tilt", 0),
        home_zoom=ptz_cfg.get("home_zoom", 1.0),
        home_fov=ptz_cfg.get("home_fov", 65),
        home_autofocus=ptz_cfg.get("autofocus", True),
        home_focus=ptz_cfg.get("focus", 50),
    )
    ptz.return_to_home()

    classifier = BirdSpeciesClassifier(
        hef_path=_resolve_path(cls_cfg.get("hef_path", "models/uk_birds_regnet_y_1_6g_h8.hef")),
        onnx_path=_resolve_path(cls_cfg.get("onnx_path", "models/uk_birds_regnet_y_1_6g.onnx")),
        labels_path=_resolve_path(cls_cfg.get("labels_path", "models/eu_common_labels.json")),
        uk_birds_path=_resolve_path(cls_cfg.get("uk_birds_path", "config/uk_birds.json")),
        uk_filter_enabled=cls_cfg.get("uk_filter_enabled", True),
        min_confidence=cls_cfg.get("min_confidence", 0.25),
    )

    detector = HailoDetector(
        hef_path=hailo_cfg["hef_path"],
        score_threshold=hailo_cfg["score_threshold"],
        target_classes=hailo_cfg.get("target_classes", ["bird"]),
        classifier=classifier,
    )

    framer = AutoFrameController(
        ptz=ptz,
        target_min_area=spot_cfg.get("target_min_area", 0.25),
        target_max_area=spot_cfg.get("target_max_area", 0.40),
        settle_delay_sec=spot_cfg.get("settle_delay_sec", 0.35),
        burst_count=spot_cfg.get("burst_count", 5),
        max_framing_sec=spot_cfg.get("max_framing_sec", 7.5),
        lost_timeout_sec=spot_cfg.get("lost_timeout_sec", 1.8),
        return_home_settle_sec=spot_cfg.get("return_home_settle_sec", 2.2),
        manual_pause_sec=spot_cfg.get("manual_pause_sec", 60.0),
    )

    per_species_cooldown = float(spot_cfg.get("per_species_cooldown_sec", 300.0))
    auto_snapshot_enabled = bool(spot_cfg.get("auto_snapshot", True))
    show_preview_bbox = bool(spot_cfg.get("show_preview_bbox", True))
    show_capture_bbox = bool(spot_cfg.get("show_capture_bbox", True))

    dashboard: DashboardServer | None = None
    latest_primary_lock = threading.Lock()
    latest_primary_det: Any = None

    def on_burst_capture_ready(best: BurstCandidate, pred: Any) -> None:
        """Handle completed PTZ framing burst: evaluate species, save photo if new/better/cooldown, log to DB."""
        det = best.detection
        min_clf_conf = float(cls_cfg.get("min_confidence", 0.25))
        if pred is not None and pred.confidence >= min_clf_conf:
            species_name = pred.species
            sci_name = pred.scientific_name
            sp_conf = pred.confidence
            top_preds = pred.top_predictions
            if det.track_id is not None:
                detector.tracker.assign_species(
                    det.track_id, species_name, sci_name, sp_conf, top_preds
                )
        elif pred is not None:
            # Low confidence classifier prediction (< 25%)
            species_name = "Unidentified Bird"
            sci_name = ""
            sp_conf = pred.confidence
            top_preds = pred.top_predictions
            spotter_log.info(
                "CLASSIFIER",
                f"Prediction below min_confidence ({pred.confidence:.1%} < {min_clf_conf:.1%}): "
                f"top was {pred.species} ({pred.confidence:.1%}) -> recorded as 'Unidentified Bird'",
            )
        else:
            if det.species and (det.species_confidence or 0.0) >= min_clf_conf:
                species_name = det.species
                sci_name = det.scientific_name or ""
                sp_conf = det.species_confidence or det.confidence
            else:
                species_name = "Unidentified Bird"
                sci_name = ""
                sp_conf = det.species_confidence or det.confidence
            top_preds = det.top_predictions or []

        q_score = compute_quality_score(sp_conf, best.sharpness)
        should_save, reason = db.should_save_auto_photo(
            species=species_name,
            quality_score=q_score,
            per_species_cooldown_sec=per_species_cooldown,
        )

        snap_file = None
        thumb_file = None
        if auto_snapshot_enabled and should_save:
            capture_frame = (
                draw_capture_bbox(best.bgr_frame, det, species_name, sp_conf)
                if show_capture_bbox
                else best.bgr_frame
            )
            snap_file, thumb_file = save_clean_snapshot_and_thumb(
                bgr_frame=capture_frame,
                bgr_crop=best.bgr_crop,
                species_name=species_name,
                snapshots_dir=snapshots_dir,
            )

        rec = db.record_sighting(
            species=species_name,
            scientific_name=sci_name,
            yolo_confidence=det.confidence,
            species_confidence=sp_conf,
            sharpness=best.sharpness,
            quality_score=q_score,
            snapshot_filename=snap_file,
            thumb_filename=thumb_file,
            zoom_level=best.zoom,
            is_manual=False,
            top_predictions=top_preds,
        )

        photo_desc = f"photo saved: {snap_file} (reason: {reason})" if snap_file else f"no photo (reason: {reason})"
        spotter_log.info(
            "SPOTTED",
            f"Sighting #{rec['id']}: {species_name} ({sp_conf:.1%}) | zoom={best.zoom:.1f}x, "
            f"sharpness={best.sharpness:.1f}, Q={q_score:.3f} | {photo_desc}",
        )

    def handle_manual_snapshot() -> dict[str, Any]:
        """Capture clean full-res snapshot on manual request, classify if bird visible, and record in DB."""
        if dashboard is None:
            return {"success": False}
        raw_frame = dashboard.get_raw_frame()
        if raw_frame is None:
            return {"success": False}

        with latest_primary_lock:
            p_det = latest_primary_det

        crop = (
            extract_bird_crop(raw_frame, p_det, padding_ratio=0.18)
            if p_det is not None
            else raw_frame
        )
        sharpness = compute_crop_sharpness(crop if crop is not None else raw_frame)
        pred = detector.classify_crop(crop) if crop is not None else None

        if pred is not None:
            species_name = pred.species
            sci_name = pred.scientific_name
            sp_conf = pred.confidence
            top_preds = pred.top_predictions
        elif p_det is not None and p_det.species:
            species_name = p_det.species
            sci_name = p_det.scientific_name
            sp_conf = p_det.species_confidence or p_det.confidence
            top_preds = p_det.top_predictions
        else:
            species_name = "Manual Snapshot"
            sci_name = ""
            sp_conf = p_det.confidence if p_det is not None else 1.0
            top_preds = []

        capture_frame = (
            draw_capture_bbox(raw_frame, p_det, species_name, sp_conf)
            if (show_capture_bbox and p_det is not None)
            else raw_frame
        )
        snap_file, thumb_file = save_clean_snapshot_and_thumb(
            bgr_frame=capture_frame,
            bgr_crop=crop if p_det is not None else None,
            species_name=species_name,
            snapshots_dir=snapshots_dir,
        )
        if not snap_file:
            return {"success": False}

        rec = db.record_sighting(
            species=species_name,
            scientific_name=sci_name,
            yolo_confidence=p_det.confidence if p_det is not None else 0.0,
            species_confidence=sp_conf,
            sharpness=sharpness,
            snapshot_filename=snap_file,
            thumb_filename=thumb_file,
            zoom_level=ptz.zoom,
            is_manual=True,
            top_predictions=top_preds,
        )
        return {"success": True, "filename": snap_file, "species": species_name, "sighting": rec}

    manual_night_mode: bool | None = None

    def handle_control(cmd: dict[str, Any]) -> None:
        nonlocal show_preview_bbox, show_capture_bbox, manual_night_mode
        action = cmd.get("action")
        if action == "toggle_system_power":
            if ptz.system_power_on:
                manual_night_mode = True
                threading.Thread(target=ptz.standby_camera, daemon=True).start()
            else:
                manual_night_mode = False
                threading.Thread(target=ptz.wakeup_camera, daemon=True).start()
        elif action == "activate_day_mode":
            manual_night_mode = False
            threading.Thread(target=ptz.wakeup_camera, daemon=True).start()
            framer.resume_auto_tracking()
        elif action == "activate_night_mode":
            manual_night_mode = True
            threading.Thread(target=ptz.standby_camera, daemon=True).start()
        elif action == "center":
            framer.trigger_manual_override()
            ptz.center()
        elif action == "nudge":
            framer.trigger_manual_override()
            ptz.nudge(int(cmd.get("d_pan", 0)), int(cmd.get("d_tilt", 0)))
        elif action == "set_zoom":
            framer.trigger_manual_override()
            ptz.set_zoom(float(cmd.get("zoom", 1.0)), force=True)
        elif action == "zoom_in":
            framer.trigger_manual_override()
            ptz.zoom_in(float(cmd.get("step", 0.2)))
        elif action == "zoom_out":
            framer.trigger_manual_override()
            ptz.zoom_out(float(cmd.get("step", 0.25)))
        elif action == "set_fov":
            ptz.set_fov(int(cmd.get("fov", 65)))
        elif action == "toggle_autofocus":
            ptz.set_focus(not ptz.autofocus, ptz.focus)
        elif action == "set_focus":
            ptz.set_focus(False, int(cmd.get("focus", ptz.focus)))
        elif action == "save_home_preset":
            ptz.save_current_as_home()
        elif action == "set_roi":
            ptz.set_roi(
                float(cmd.get("x", 0.25)),
                float(cmd.get("y", 0.25)),
                float(cmd.get("w", 0.5)),
                float(cmd.get("h", 0.5)),
                enabled=True,
                lock_aspect=bool(cmd.get("lock_aspect", False)),
            )
            if dashboard is not None:
                with dashboard._cond:
                    dashboard._state["roi"] = ptz.get_roi_dict()
        elif action == "clear_roi":
            ptz.clear_roi()
            if dashboard is not None:
                with dashboard._cond:
                    dashboard._state["roi"] = ptz.get_roi_dict()
        elif action == "go_home":
            framer.trigger_manual_override()
            ptz.return_to_home()
        elif action == "resume_auto_tracking":
            framer.resume_auto_tracking()
            ptz.return_to_home()
        elif action == "toggle_tracking":
            ptz.tracking_enabled = not ptz.tracking_enabled
            if ptz.tracking_enabled:
                framer.resume_auto_tracking()
        elif action == "toggle_uk_filter":
            classifier.uk_filter_enabled = not classifier.uk_filter_enabled
        elif action == "toggle_preview_bbox":
            show_preview_bbox = not show_preview_bbox
        elif action == "toggle_capture_bbox":
            show_capture_bbox = not show_capture_bbox
        elif action == "toggle_invert_pan":
            ptz.invert_pan = not ptz.invert_pan
        elif action == "toggle_invert_tilt":
            ptz.invert_tilt = not ptz.invert_tilt
        elif action == "toggle_solar_schedule":
            solar.enabled = not solar.enabled
            solar.force_recheck()
            if dashboard is not None:
                with dashboard._cond:
                    dashboard._state["solar"] = solar.get_status_dict()
        elif action == "clear_stats":
            db.clear_all()
            spotter_log.warn("DATABASE", "Database cleared (sightings and species_dex wiped)")
        elif action == "set_log_level":
            new_lvl = spotter_log.set_level(str(cmd.get("level", "INFO")))
            if dashboard is not None:
                with dashboard._cond:
                    dashboard._state["log_level"] = new_lvl
        elif action == "clear_logs":
            spotter_log.clear()
        elif action == "shutdown_pi":
            import subprocess
            threading.Thread(
                target=lambda: (time.sleep(1.0), subprocess.run(["sudo", "poweroff"], check=False)),
                daemon=True,
            ).start()

    def handle_relabel(sighting_id: int, new_species: str) -> bool:
        sci_name = classifier.uk_name_to_sci.get(new_species, "")
        return db.relabel_sighting(sighting_id, new_species, scientific_name=sci_name)

    def handle_batch_relabel(sighting_ids: list[int], new_species: str) -> int:
        sci_name = classifier.uk_name_to_sci.get(new_species, "")
        count = db.batch_relabel_sightings(sighting_ids, new_species, scientific_name=sci_name)
        spotter_log.info("DATABASE", f"Batch re-labeled {count} sightings to '{new_species}'")
        return count

    def handle_delete(sighting_id: int) -> bool:
        return db.delete_sighting(sighting_id, snapshots_dir=snapshots_dir)

    def handle_delete_species(species: str) -> int:
        count = db.delete_species(species, snapshots_dir=snapshots_dir)
        spotter_log.info("DATABASE", f"Deleted all {count} records and photos for species '{species}'")
        return count

    dashboard = DashboardServer(
        host=dash_cfg["host"],
        port=dash_cfg["port"],
        snapshots_dir=snapshots_dir,
        on_control=handle_control,
        on_snapshot_requested=handle_manual_snapshot,
        on_relabel_sighting=handle_relabel,
        on_batch_relabel=handle_batch_relabel,
        on_delete_sighting=handle_delete,
        on_delete_species=handle_delete_species,
        on_get_species_sightings=lambda sp: db.get_sightings_for_species(sp),
        get_catalog_cb=classifier.get_uk_species_catalog,
    )
    dashboard.start_background()

    # Ensure OBSBOT Tiny SE is awake and wait up to 15s for V4L2 device enumeration on cold boot
    ptz.wakeup_camera()
    cap = None
    for _attempt in range(15):
        cap = cv2.VideoCapture(cam_cfg["device"], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam_cfg["format"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
        if cap.isOpened():
            break
        cap.release()
        time.sleep(1.0)

    if cap is None or not cap.isOpened():
        print(f"ERROR: Could not open camera {cam_cfg['device']}")
        dashboard.stop()
        return 1

    # Apply saved Home/Feeder PTZ & Zoom preset after V4L2 stream is open
    ptz.return_to_home()

    stop_requested = False

    def _sig_handler(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print(
        f"🐦 British Bird Spotter Live! Open dashboard: http://localhost:{dash_cfg['port']} "
        f"or http://<pi-ip>:{dash_cfg['port']}"
    )

    fps_ema = 0.0
    last_db_poll_ts = 0.0
    last_solar_check_ts = 0.0
    cached_dex: list[dict[str, Any]] = db.get_species_dex()
    cached_stats: dict[str, dict[str, Any]] = db.get_species_stats_dict()
    cached_snaps: list[dict[str, Any]] = db.get_recent_sightings(limit=24)
    cached_hourly: list[int] = db.get_hourly_distribution()
    cached_daily: list[dict[str, Any]] = db.get_daily_trends()

    try:
        with detector:
            start_ts = time.monotonic()
            prev_ts = start_ts
            while not stop_requested:
                if duration_sec is not None and (time.monotonic() - start_ts) >= duration_sec:
                    break

                now = time.monotonic()

                # Refresh SQLite analytics every 0.8s so web dashboard stays updated at all times
                if (now - last_db_poll_ts) >= 0.8:
                    last_db_poll_ts = now
                    cached_dex = db.get_species_dex()
                    cached_stats = db.get_species_stats_dict()
                    cached_snaps = db.get_recent_sightings(limit=24)
                    cached_hourly = db.get_hourly_distribution()
                    cached_daily = db.get_daily_trends()

                # Evaluate solar sunset sleep / dawn wake schedule every ~5s
                if (now - last_solar_check_ts) >= 5.0:
                    last_solar_check_ts = now
                    trans = solar.check_transition()
                    if trans == "sleep" and ptz.system_power_on:
                        print(
                            "🌙 [Solar Scheduler] Entering night sleep window. Putting camera into standby."
                        )
                        manual_night_mode = True
                        threading.Thread(target=ptz.standby_camera, daemon=True).start()
                    elif trans == "wake" and not ptz.system_power_on:
                        print(
                            "🌅 [Solar Scheduler] Entering dawn wake window. Waking up camera."
                        )
                        manual_night_mode = False
                        threading.Thread(target=ptz.wakeup_camera, daemon=True).start()

                if manual_night_mode is True:
                    is_night = True
                elif manual_night_mode is False:
                    is_night = False
                else:
                    is_night = bool(solar.is_night_sleep_window() or not ptz.system_power_on)

                if not ptz.system_power_on or is_night:
                    # Camera is offline / in standby / night mode.
                    # Keep web dashboard responsive and state fresh without hammering V4L2.
                    offline_state = {
                        "fps": 0.0,
                        "infer_ms": 0.0,
                        "resolution": f"{cam_cfg.get('width', 1920)}x{cam_cfg.get('height', 1080)}",
                        "zoom": round(ptz.zoom, 2),
                        "fov": ptz.fov,
                        "autofocus": ptz.autofocus,
                        "focus": ptz.focus,
                        "roi": ptz.get_roi_dict(),
                        "detections_count": 0,
                        "primary_target": None,
                        "primary_area": 0.0,
                        "pan": ptz.pan,
                        "tilt": ptz.tilt,
                        "home_preset": {
                            "pan": ptz.home_pan,
                            "tilt": ptz.home_tilt,
                            "zoom": ptz.home_zoom,
                            "fov": ptz.home_fov,
                            "autofocus": ptz.home_autofocus,
                            "focus": ptz.home_focus,
                        },
                        "system_power_on": ptz.system_power_on,
                        "tracking_enabled": ptz.tracking_enabled,
                        "fsm_state": "NIGHT_SLEEP" if is_night else "STANDBY",
                        "fsm_label": "NIGHT MODE (CAMERA OFFLINE)" if is_night else "CAMERA STANDBY",
                        "manual_override_remaining": 0.0,
                        "invert_pan": ptz.invert_pan,
                        "invert_tilt": ptz.invert_tilt,
                        "show_preview_bbox": show_preview_bbox,
                        "show_capture_bbox": show_capture_bbox,
                        "uk_filter_enabled": classifier.uk_filter_enabled,
                        "classifier_backend": classifier.active_backend,
                        "species_dex": cached_dex,
                        "species_stats": cached_stats,
                        "recent_snapshots": cached_snaps,
                        "hourly_distribution": cached_hourly,
                        "daily_trends": cached_daily,
                        "solar": solar.get_status_dict(),
                        "log_level": spotter_log.level,
                        "is_night_mode": is_night,
                    }
                    dashboard.update_state_only(offline_state)
                    time.sleep(0.25)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue

                dt = max(1e-4, now - prev_ts)
                prev_ts = now
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0.0 else (0.88 * fps_ema + 0.12 * inst_fps)

                detections = (
                    detector.detect(frame, roi=ptz.get_roi_tuple())
                    if ptz.system_power_on
                    else []
                )
                primary = max(detections, key=lambda d: d.area_ratio) if detections else None

                with latest_primary_lock:
                    latest_primary_det = primary

                # Advance PTZ Auto-Framing & Burst Photography State Machine
                framer.step(
                    bgr_frame=frame,
                    primary_det=primary,
                    classify_cb=detector.classify_crop,
                    on_capture_ready=on_burst_capture_ready,
                    now=now,
                )

                annotated = draw_overlay(
                    frame=frame,
                    detections=detections,
                    primary=primary,
                    fps=fps_ema,
                    infer_ms=detector.last_infer_ms,
                    zoom=ptz.zoom,
                    deadzone_norm=ptz.deadzone_norm,
                    fsm_label=framer.status_label,
                    show_bbox=show_preview_bbox,
                    roi=ptz.get_roi_tuple(),
                )

                state = {
                    "fps": round(fps_ema, 1),
                    "infer_ms": round(detector.last_infer_ms, 2),
                    "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                    "zoom": round(ptz.zoom, 2),
                    "fov": ptz.fov,
                    "autofocus": ptz.autofocus,
                    "focus": ptz.focus,
                    "roi": ptz.get_roi_dict(),
                    "detections_count": len(detections),
                    "primary_target": (
                        f"#{primary.track_id} {primary.display_name}"
                        if primary is not None
                        else None
                    ),
                    "primary_area": round(primary.area_ratio, 4) if primary is not None else 0.0,
                    "pan": ptz.pan,
                    "tilt": ptz.tilt,
                    "home_preset": {
                        "pan": ptz.home_pan,
                        "tilt": ptz.home_tilt,
                        "zoom": ptz.home_zoom,
                        "fov": ptz.home_fov,
                        "autofocus": ptz.home_autofocus,
                        "focus": ptz.home_focus,
                    },
                    "system_power_on": ptz.system_power_on,
                    "tracking_enabled": ptz.tracking_enabled,
                    "fsm_state": framer.state.value,
                    "fsm_label": framer.status_label,
                    "manual_override_remaining": framer.manual_override_remaining(now=now),
                    "invert_pan": ptz.invert_pan,
                    "invert_tilt": ptz.invert_tilt,
                    "show_preview_bbox": show_preview_bbox,
                    "show_capture_bbox": show_capture_bbox,
                    "uk_filter_enabled": classifier.uk_filter_enabled,
                    "classifier_backend": classifier.active_backend,
                    "species_dex": cached_dex,
                    "species_stats": cached_stats,
                    "recent_snapshots": cached_snaps,
                    "hourly_distribution": cached_hourly,
                    "daily_trends": cached_daily,
                    "solar": solar.get_status_dict(),
                    "log_level": spotter_log.level,
                    "is_night_mode": False,
                }
                dashboard.update_frame_and_state(frame, annotated, state)
    finally:
        cap.release()
        dashboard.stop()
        print("Stopped Bird Spotter cleanly.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bird Spotter — British Bird Watch & PTZ Tracker")
    parser.add_argument(
        "--check-hardware",
        action="store_true",
        help="Verify camera device and Hailo-8 model availability",
    )
    parser.add_argument(
        "--center-ptz",
        action="store_true",
        help="Center the OBSBOT Tiny SE gimbal (pan=0, tilt=0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds to run before exiting (useful for tests)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to custom settings.toml configuration",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.center_ptz:
        ptz = ObsbotPTZ(device=cfg["camera"]["device"])
        ptz.center()
        print("Centered OBSBOT Tiny SE gimbal.")
        return 0

    if args.check_hardware:
        return check_hardware(cfg)

    return run_spotter(cfg, duration_sec=args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
