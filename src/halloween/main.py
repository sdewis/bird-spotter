"""Main entry point for the Halloween Animatronic Skull."""

import argparse
from pathlib import Path
import signal
import socket
import threading
import time
from typing import Any

import cv2

from halloween.audio.speaker import BipolarSpeaker
from halloween.config import load_config
from halloween.dashboard.server import DashboardServer, draw_overlay
from halloween.personality.engine import BipolarEngine, Mood
from halloween.servos.client import PiconZeroClient
from halloween.servos.deployer import deploy_to_pi
from halloween.vision.detector import DEFAULT_PANEL_CLASSES, HailoDetector
from halloween.vision.ptz import ObsbotPTZ


def check_hardware(cfg: dict) -> int:
    """Verify camera device, UVC PTZ controls, Hailo-8 HEF model, and RPi2 reachability."""
    cam_dev = cfg["camera"]["device"]
    hef_path = cfg["hailo"]["hef_path"]
    pi2_host = cfg["pi2_servos"]["host"]

    cam_ok = Path(cam_dev).exists()
    detector = HailoDetector(hef_path, cfg["hailo"]["score_threshold"])
    hef_ok = detector.is_available()

    try:
        with socket.create_connection((pi2_host, 22), timeout=1.0):
            pi2_ok = True
    except OSError:
        pi2_ok = False

    print("=== Halloween Hardware Self-Check ===")
    print(f"  Camera ({cam_dev}): {'OK' if cam_ok else 'MISSING'}")
    print(f"  Hailo-8 Model ({hef_path}): {'OK' if hef_ok else 'MISSING'}")
    print(f"  RPi2 Picon Zero Node ({pi2_host}:22): {'ONLINE' if pi2_ok else 'OFFLINE'}")
    print(f"  Dashboard Target: http://{cfg['dashboard']['host']}:{cfg['dashboard']['port']}")
    return 0 if (cam_ok and hef_ok) else 1


def run_tracker(cfg: dict, duration_sec: float | None = None) -> int:
    """Run the live Hailo-8 detection, OBSBOT PTZ tracker, Bipolar Voice, and MJPEG web dashboard."""
    cam_cfg = cfg["camera"]
    ptz_cfg = cam_cfg["ptz"]
    hailo_cfg = cfg["hailo"]
    pers_cfg = cfg["personality"]
    audio_cfg = cfg.get("audio", {})
    dash_cfg = cfg["dashboard"]
    servo_cfg = cfg["pi2_servos"]

    ptz = ObsbotPTZ(
        device=cam_cfg["device"],
        pan_min=ptz_cfg["pan_min"],
        pan_max=ptz_cfg["pan_max"],
        pan_step=ptz_cfg["pan_step"],
        tilt_min=ptz_cfg["tilt_min"],
        tilt_max=ptz_cfg["tilt_max"],
        tilt_step=ptz_cfg["tilt_step"],
        deadzone_norm=ptz_cfg["deadzone_norm"],
        kp_pan=ptz_cfg["kp_pan"],
        kp_tilt=ptz_cfg["kp_tilt"],
        invert_pan=ptz_cfg.get("invert_pan", True),
        invert_tilt=ptz_cfg.get("invert_tilt", True),
        tracking_enabled=ptz_cfg["enabled"],
    )
    ptz.center()

    bipolar = BipolarEngine(
        looming_bbox_area_ratio=pers_cfg["looming_bbox_area_ratio"],
        crowd_threshold=pers_cfg["crowd_threshold"],
        mood_swing_interval_sec=pers_cfg["mood_swing_interval_sec"],
        sweet_classes=list(pers_cfg["sweet_classes"]),
    )

    servo_client = PiconZeroClient(
        host=servo_cfg["host"],
        port=servo_cfg["port"],
        enabled=servo_cfg["enabled"],
        init_pan_center=int(ptz.calib.get("servo_pan_center", 90)),
        init_tilt_center=int(ptz.calib.get("servo_tilt_center", 90)),
    )
    servo_client.start_monitor()

    speaker = BipolarSpeaker(
        servo_client=servo_client,
        alsa_device=audio_cfg.get("pi2_alsa_device", "plughw:1,0"),
        enabled=audio_cfg.get("enabled", True),
        auto_speak=audio_cfg.get("auto_speak", True),
        cooldown_sec=audio_cfg.get("cooldown_sec", 12.0),
        jaw_closed_deg=audio_cfg.get("jaw_closed_deg", 85),
        jaw_open_deg=audio_cfg.get("jaw_open_deg", 135),
    )

    detector = HailoDetector(
        hef_path=hailo_cfg["hef_path"],
        score_threshold=hailo_cfg["score_threshold"],
        target_classes=hailo_cfg["target_classes"],
    )

    def handle_control(cmd: dict[str, Any]) -> None:
        action = cmd.get("action")
        if action == "toggle_system_power":
            if ptz.system_power_on:
                def _power_off() -> None:
                    servo_client.send_standby()
                    ptz.standby_camera()
                threading.Thread(target=_power_off, daemon=True).start()
            else:
                def _power_on() -> None:
                    s_pan, s_tilt = ptz.wakeup_camera()
                    servo_client.send_startup_init(pan_center=s_pan, tilt_center=s_tilt)
                threading.Thread(target=_power_on, daemon=True).start()
            return

        if action == "center":
            ptz.tracking_enabled = False
            ptz.center()
        elif action == "center_all":
            ptz.tracking_enabled = False
            ptz.center_all()
            servo_client.send_angles(pan=ptz.servo_pan, tilt=ptz.servo_tilt, force=True)
        elif action == "nudge":
            ptz.nudge_camera_manual(int(cmd.get("d_pan", 0)), int(cmd.get("d_tilt", 0)))
        elif action == "nudge_servo":
            ptz.nudge_servo_manual(int(cmd.get("d_pan_deg", 0)), int(cmd.get("d_tilt_deg", 0)))
            servo_client.send_angles(pan=ptz.servo_pan, tilt=ptz.servo_tilt, force=True)
        elif action == "center_servo":
            ptz.tracking_enabled = False
            ptz.center_servos()
            servo_client.send_angles(pan=ptz.servo_pan, tilt=ptz.servo_tilt, force=True)
        elif action == "calibrate":
            direction = str(cmd.get("direction", "")).lower()
            if direction in (
                "left",
                "right",
                "up",
                "down",
                "center",
                "center_cam",
                "center_servo",
            ):
                ptz.calibrate_direction(direction)
                servo_client.init_pan_center = int(ptz.calib.get("servo_pan_center", 90))
                servo_client.init_tilt_center = int(ptz.calib.get("servo_tilt_center", 90))
        elif action == "calib_reset":
            ptz.reset_calibration()
            servo_client.init_pan_center = 90
            servo_client.init_tilt_center = 90
            servo_client.send_angles(pan=ptz.servo_pan, tilt=ptz.servo_tilt, force=True)
        elif action == "toggle_tracking":
            ptz.tracking_enabled = not ptz.tracking_enabled
        elif action == "toggle_invert_pan":
            ptz.invert_pan = not ptz.invert_pan
        elif action == "toggle_invert_tilt":
            ptz.invert_tilt = not ptz.invert_tilt
        elif action == "toggle_filter":
            detector.filter_target_classes = not detector.filter_target_classes
        elif action == "toggle_classes":
            cls_list = [str(c) for c in cmd.get("classes", [])]
            if cls_list:
                # If all in group are currently enabled, disable them; otherwise enable all in group
                if all(c in detector.enabled_classes for c in cls_list):
                    for c in cls_list:
                        detector.enabled_classes.discard(c)
                else:
                    for c in cls_list:
                        detector.enabled_classes.add(c)
        elif action == "class_preset":
            preset = str(cmd.get("preset", "toggle_all_panel"))
            if preset in ("all_panel", "toggle_all_panel"):
                if all(c in detector.enabled_classes for c in DEFAULT_PANEL_CLASSES):
                    detector.set_classes(())
                else:
                    detector.set_classes(DEFAULT_PANEL_CLASSES)
            elif preset == "people_pets":
                detector.set_classes(
                    ("person", "upper body", "frontal face", "side face", "cat", "dog", "bird", "horse")
                )
            elif preset == "faces_only":
                detector.set_classes(("frontal face", "side face"))
        elif action == "toggle_auto_speak":
            speaker.auto_speak = not speaker.auto_speak
        elif action == "preset_audio":
            speaker.trigger_preset(str(cmd.get("preset", "menacing")))
        elif action == "speak_custom":
            text = str(cmd.get("text", "")).strip()
            mood_str = str(cmd.get("mood", "MENACING")).upper()
            mood_val = Mood(mood_str) if mood_str in Mood.__members__ else Mood.MENACING
            speaker.speak_async(text, mood_val)
        elif action == "pi2_init":
            threading.Thread(target=servo_client.send_startup_init, daemon=True).start()
        elif action == "pi2_standby":
            threading.Thread(target=servo_client.send_standby, daemon=True).start()
        elif action == "deploy_node":
            new_host = str(cmd.get("host", "")).strip()
            new_user = str(cmd.get("user", "sean")).strip() or "sean"
            new_pw = str(cmd.get("password", " "))
            if new_host:
                def _do_deploy() -> None:
                    res = deploy_to_pi(host=new_host, user=new_user, password=new_pw)
                    if res.success:
                        servo_client.host = new_host
                        servo_client.pi2_online = False
                threading.Thread(target=_do_deploy, daemon=True).start()
        elif action == "test_servo":
            ch_name = str(cmd.get("channel", "center"))

            def _sweep() -> None:
                c_pan = int(ptz.calib.get("servo_pan_center", 90))
                c_tilt = int(ptz.calib.get("servo_tilt_center", 90))
                if ch_name == "center":
                    ptz.center_all()
                    servo_client.send_angles(
                        pan=c_pan, tilt=c_tilt, jaw=speaker.jaw_closed_deg, force=True
                    )
                elif ch_name == "pan":
                    for p in (max(0, c_pan - 25), min(180, c_pan + 25), c_pan):
                        servo_client.send_angles(pan=p, tilt=c_tilt, force=True)
                        time.sleep(0.45)
                elif ch_name == "tilt":
                    for t in (max(0, c_tilt - 25), min(180, c_tilt + 25), c_tilt):
                        servo_client.send_angles(pan=c_pan, tilt=t, force=True)
                        time.sleep(0.45)
                elif ch_name == "jaw":
                    for j in (
                        speaker.jaw_open_deg,
                        speaker.jaw_closed_deg,
                        speaker.jaw_open_deg,
                        speaker.jaw_closed_deg,
                    ):
                        servo_client.send_angles(pan=c_pan, tilt=c_tilt, jaw=j, force=True)
                        time.sleep(0.28)

            threading.Thread(target=_sweep, daemon=True).start()

    dashboard = DashboardServer(
        host=dash_cfg["host"],
        port=dash_cfg["port"],
        on_control=handle_control,
    )
    dashboard.start_background()

    cap = cv2.VideoCapture(cam_cfg["device"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam_cfg["format"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {cam_cfg['device']}")
        return 1

    stop_requested = False

    def _sig_handler(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print(
        f"🎃 Halloween Tracker Live! Open http://192.168.100.122:{dash_cfg['port']} "
        f"(or http://localhost:{dash_cfg['port']}) in your browser."
    )

    fps_ema = 0.0
    prev_ts = time.monotonic()

    try:
        with detector:
            start_ts = time.monotonic()
            prev_ts = start_ts
            while not stop_requested:
                if duration_sec is not None and (time.monotonic() - start_ts) >= duration_sec:
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                now = time.monotonic()
                dt = max(1e-4, now - prev_ts)
                prev_ts = now
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0.0 else (0.88 * fps_ema + 0.12 * inst_fps)

                detections = detector.detect(frame) if ptz.system_power_on else []
                mood, primary = bipolar.evaluate(detections)

                if ptz.system_power_on:
                    if primary is not None:
                        # Aim directly at face/eyes (aim_x, aim_y) rather than mid-torso (center_y)
                        ptz.track_target(primary.aim_x, primary.aim_y)
                        speaker.evaluate_auto_reaction(
                            mood=mood,
                            target_label=primary.label,
                            area_ratio=primary.area_ratio,
                            looming_threshold=bipolar.looming_bbox_area_ratio,
                        )
                    else:
                        speaker.evaluate_auto_reaction(
                            mood=Mood.IDLE, target_label=None, area_ratio=0.0
                        )

                    s_pan, s_tilt = ptz.get_active_servo_angles()
                    servo_client.send_angles(pan=s_pan, tilt=s_tilt)
                else:
                    s_pan, s_tilt = ptz.servo_pan, ptz.servo_tilt

                annotated = draw_overlay(
                    frame=frame,
                    detections=detections,
                    primary=primary,
                    mood=mood,
                    fps=fps_ema,
                    infer_ms=detector.last_infer_ms,
                    deadzone_norm=ptz.deadzone_norm,
                )

                state = {
                    "fps": round(fps_ema, 1),
                    "infer_ms": round(detector.last_infer_ms, 2),
                    "mood": mood.value,
                    "detections_count": len(detections),
                    "primary_target": (
                        f"#{primary.track_id} {primary.label} ({primary.confidence:.2f})"
                        if primary is not None
                        else None
                    ),
                    "primary_area": round(primary.area_ratio, 4) if primary is not None else 0.0,
                    "pan": ptz.pan,
                    "tilt": ptz.tilt,
                    "servo_pan": s_pan,
                    "servo_tilt": s_tilt,
                    "calib": ptz.calib,
                    "system_power_on": ptz.system_power_on,
                    "tracking_enabled": ptz.tracking_enabled,
                    "invert_pan": ptz.invert_pan,
                    "invert_tilt": ptz.invert_tilt,
                    "filter_target_classes": detector.filter_target_classes,
                    "enabled_classes": sorted(detector.enabled_classes),
                    "auto_speak": speaker.auto_speak,
                    "last_spoken_text": speaker.last_spoken_text,
                    "last_spoken_mood": speaker.last_spoken_mood,
                    "pi2_online": servo_client.pi2_online,
                    "pi2_state": servo_client.pi2_state,
                    "pi2_ping_ms": servo_client.pi2_ping_ms,
                    "pi2_uptime_sec": round(servo_client.pi2_uptime_sec, 1),
                    "pi2_throttled": servo_client.pi2_throttled,
                    "pi2_host": servo_client.host,
                }
                dashboard.update_frame_and_state(annotated, state)
    finally:
        servo_client.stop_monitor()
        cap.release()
        dashboard.stop()
        print("Stopped Halloween Tracker cleanly.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Halloween Bipolar Animatronic Skull")
    parser.add_argument(
        "--check-hardware",
        action="store_true",
        help="Verify camera, Hailo-8 model, and RPi2 configuration",
    )
    parser.add_argument(
        "--center-ptz",
        action="store_true",
        help="Center the OBSBOT Tiny SE UVC Pan/Tilt mechanism",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds to run before exiting (useful for smoke tests)",
    )
    parser.add_argument(
        "--deploy-node",
        metavar="HOST",
        help="Deploy the packaged Picon Zero servo node bundle to a Raspberry Pi host/IP",
    )
    parser.add_argument("--node-user", default="sean", help="SSH user for --deploy-node")
    parser.add_argument("--node-password", default=" ", help="SSH password for --deploy-node")
    args = parser.parse_args()

    if args.deploy_node:
        res = deploy_to_pi(
            host=args.deploy_node,
            user=args.node_user,
            password=args.node_password,
        )
        print(res.log)
        return 0 if res.success else 1

    cfg = load_config()

    if args.center_ptz:
        ptz = ObsbotPTZ(device=cfg["camera"]["device"])
        ptz.center()
        print("Centered OBSBOT Tiny SE pan/tilt.")
        return 0

    if args.check_hardware:
        return check_hardware(cfg)

    return run_tracker(cfg, duration_sec=args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
