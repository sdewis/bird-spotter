"""OBSBOT Tiny SE Hardware SDK & UVC Pan/Tilt/Zoom/FOV/Focus controller."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from bird_spotter.config import PROJECT_ROOT
from bird_spotter.logger import spotter_log

try:
    import obsbot_sdk
except ImportError:
    obsbot_sdk = None  # type: ignore[assignment]

OBSBOT_CLI = "/usr/local/bin/obsbot-cli"
DEFAULT_PRESET_PATH = PROJECT_ROOT / "data" / "home_preset.json"


@dataclass
class ObsbotPTZ:
    """Controls OBSBOT Tiny SE hardware pan, tilt, zoom, FOV (86/78/65°), and lens focus."""

    device: str = "/dev/video0"
    pan_min: int = -468000
    pan_max: int = 468000
    pan_step: int = 3600
    tilt_min: int = -324000
    tilt_max: int = 324000
    tilt_step: int = 3600

    deadzone_norm: float = 0.08
    kp_pan: float = 45000.0
    kp_tilt: float = 35000.0
    invert_pan: bool = True
    invert_tilt: bool = True
    tracking_enabled: bool = True
    system_power_on: bool = True
    min_update_interval_sec: float = 0.08
    min_zoom_interval_sec: float = 0.25

    # Zoom settings (OBSBOT Tiny SE supports 1.0x to 4.0x in 0.25x hardware steps)
    zoom: float = 1.0
    zoom_min: float = 1.0
    zoom_max: float = 4.0
    zoom_step: float = 0.25

    # Hardware Field of View (86 = Wide, 78 = Medium, 65 = Narrow)
    fov: int = 65

    # Hardware Motorized Lens Focus (autofocus or manual 0..100)
    autofocus: bool = True
    focus: int = 50

    # Home / Feeder preset (usual detection mode position)
    home_pan: int = 0
    home_tilt: int = 0
    home_zoom: float = 1.0
    home_fov: int = 65
    home_autofocus: bool = True
    home_focus: int = 50
    preset_path: Path = field(default_factory=lambda: DEFAULT_PRESET_PATH)

    # Zone of Interest (ROI)
    roi_enabled: bool = False
    roi_x: float = 0.25
    roi_y: float = 0.25
    roi_w: float = 0.50
    roi_h: float = 0.50
    roi_ref_pan: int = 0
    roi_ref_tilt: int = 0
    roi_ref_zoom: float = 1.0

    pan: int = 0
    tilt: int = 0

    _sdk_dev: Any = field(default=None, init=False, repr=False)
    _last_cmd_ts: float = field(default=0.0, init=False, repr=False)
    _last_zoom_ts: float = field(default=0.0, init=False, repr=False)
    _last_motion_ts: float = field(default=0.0, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.load_home_preset()
        self.pan = self.home_pan
        self.tilt = self.home_tilt
        self.zoom = self.home_zoom
        self.fov = self.home_fov
        self.autofocus = self.home_autofocus
        self.focus = self.home_focus

    def connect_sdk(self, wait_ms: int = 3500) -> bool:
        """Connect to the native OBSBOT C++ SDK (obsbot_sdk) and configure camera for bird spotting."""
        if obsbot_sdk is None:
            return False
        try:
            devs = obsbot_sdk.get_devices(wait_ms)
            if devs:
                self._sdk_dev = devs[0]
                # Disable human face autofocus and built-in human AI tracking so gimbal & focus obey Bird Spotter
                try:
                    self._sdk_dev.set_ai_mode(obsbot_sdk.AI_MODE_NONE, 0)
                    self._sdk_dev.set_ai_enabled(False)
                    self._sdk_dev.set_face_focus(False)
                    self._sdk_dev.set_face_ae(False)
                except Exception:
                    pass
                return True
        except Exception:
            self._sdk_dev = None
        return False

    @property
    def last_motion_ts(self) -> float:
        """Monotonic timestamp of the most recent PTZ or zoom command."""
        return self._last_motion_ts

    def load_home_preset(self) -> None:
        """Load saved Home/Feeder PTZ, Zoom, FOV, and Focus preset from disk if present."""
        if self.preset_path and self.preset_path.is_file():
            try:
                with self.preset_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.home_pan = int(data.get("home_pan", self.home_pan))
                self.home_tilt = int(data.get("home_tilt", self.home_tilt))
                self.home_zoom = float(data.get("home_zoom", self.home_zoom))
                self.home_fov = int(data.get("home_fov", self.home_fov))
                self.home_autofocus = bool(data.get("home_autofocus", self.home_autofocus))
                self.home_focus = int(data.get("home_focus", self.home_focus))
                self.roi_enabled = bool(data.get("roi_enabled", self.roi_enabled))
                self.roi_x = float(data.get("roi_x", self.roi_x))
                self.roi_y = float(data.get("roi_y", self.roi_y))
                self.roi_w = float(data.get("roi_w", self.roi_w))
                self.roi_h = float(data.get("roi_h", self.roi_h))
                self.roi_ref_pan = int(data.get("roi_ref_pan", self.roi_ref_pan))
                self.roi_ref_tilt = int(data.get("roi_ref_tilt", self.roi_ref_tilt))
                self.roi_ref_zoom = float(data.get("roi_ref_zoom", self.roi_ref_zoom))
            except Exception:
                pass

    def save_preset_file(self) -> None:
        """Persist current home preset and Zone of Interest (ROI) settings to disk."""
        if not self.preset_path:
            return
        payload: dict[str, Any] = {}
        if self.preset_path.is_file():
            try:
                with self.preset_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
        with self._lock:
            payload.update(
                {
                    "home_pan": self.home_pan,
                    "home_tilt": self.home_tilt,
                    "home_zoom": self.home_zoom,
                    "home_fov": self.home_fov,
                    "home_autofocus": self.home_autofocus,
                    "home_focus": self.home_focus,
                    "roi_enabled": self.roi_enabled,
                    "roi_x": self.roi_x,
                    "roi_y": self.roi_y,
                    "roi_w": self.roi_w,
                    "roi_h": self.roi_h,
                    "roi_ref_pan": self.roi_ref_pan,
                    "roi_ref_tilt": self.roi_ref_tilt,
                    "roi_ref_zoom": self.roi_ref_zoom,
                }
            )
        try:
            self.preset_path.parent.mkdir(parents=True, exist_ok=True)
            with self.preset_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def save_current_as_home(self, home_zoom: float | None = None) -> dict[str, Any]:
        """Save current pan, tilt, zoom, FOV, and focus as the Home/Feeder preset."""
        with self._lock:
            self.home_pan = self.pan
            self.home_tilt = self.tilt
            target_zoom = self.zoom if home_zoom is None else home_zoom
            self.home_zoom = round(max(self.zoom_min, min(self.zoom_max, target_zoom)), 2)
            self.home_fov = self.fov
            self.home_autofocus = self.autofocus
            self.home_focus = self.focus
        self.save_preset_file()
        with self._lock:
            return {
                "home_pan": self.home_pan,
                "home_tilt": self.home_tilt,
                "home_zoom": self.home_zoom,
                "home_fov": self.home_fov,
                "home_autofocus": self.home_autofocus,
                "home_focus": self.home_focus,
                "roi_enabled": self.roi_enabled,
                "roi_x": self.roi_x,
                "roi_y": self.roi_y,
                "roi_w": self.roi_w,
                "roi_h": self.roi_h,
                "roi_ref_pan": self.roi_ref_pan,
                "roi_ref_tilt": self.roi_ref_tilt,
                "roi_ref_zoom": self.roi_ref_zoom,
            }

    def set_roi(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        enabled: bool = True,
        lock_aspect: bool = False,
    ) -> dict[str, Any]:
        """Set Zone of Interest (ROI) with optional aspect ratio lock and record reference camera pose."""
        if lock_aspect:
            scale = round(max(0.10, min(1.0, (float(w) + float(h)) * 0.5)), 4)
            clamped_w = scale
            clamped_h = scale
        else:
            clamped_w = round(max(0.10, min(1.0, float(w))), 4)
            clamped_h = round(max(0.10, min(1.0, float(h))), 4)

        clamped_x = round(max(0.0, min(1.0 - clamped_w, float(x))), 4)
        clamped_y = round(max(0.0, min(1.0 - clamped_h, float(y))), 4)

        with self._lock:
            self.roi_x = clamped_x
            self.roi_y = clamped_y
            self.roi_w = clamped_w
            self.roi_h = clamped_h
            self.roi_enabled = bool(enabled)
            self.roi_ref_pan = self.pan
            self.roi_ref_tilt = self.tilt
            self.roi_ref_zoom = max(1.0, self.zoom)

        self.save_preset_file()
        return self.get_roi_dict()

    def clear_roi(self) -> dict[str, Any]:
        """Disable Zone of Interest (ROI) and persist state to disk."""
        with self._lock:
            self.roi_enabled = False
        self.save_preset_file()
        return self.get_roi_dict()

    def get_roi_dict(self) -> dict[str, Any]:
        """Get Zone of Interest dict including screen rect projection."""
        with self._lock:
            screen_rect = self.get_roi_tuple()
            return {
                "enabled": self.roi_enabled,
                "x": self.roi_x,
                "y": self.roi_y,
                "w": self.roi_w,
                "h": self.roi_h,
                "ref_pan": self.roi_ref_pan,
                "ref_tilt": self.roi_ref_tilt,
                "ref_zoom": self.roi_ref_zoom,
                "screen_rect": list(screen_rect) if screen_rect is not None else None,
            }

    def get_roi_tuple(self) -> tuple[float, float, float, float] | None:
        """Compute the World-Locked ROI in current screen coordinates."""
        with self._lock:
            if not self.roi_enabled:
                return None

            ref_zoom = max(1.0, self.roi_ref_zoom)
            cur_zoom = max(1.0, self.zoom)
            cx_ref = self.roi_x + 0.5 * self.roi_w
            cy_ref = self.roi_y + 0.5 * self.roi_h

            # Optical axis offset normalized to 1.0x zoom
            rel_x = (cx_ref - 0.5) / ref_zoom
            rel_y = (cy_ref - 0.5) / ref_zoom

            # Angular shift from pan/tilt movement (3600 UVC units = 1 degree)
            fov_h_units = max(100000.0, float(self.fov) * 3600.0 * 0.87)
            fov_v_units = max(60000.0, float(self.fov) * 3600.0 * 0.49)
            sign_x = -1.0 if self.invert_pan else 1.0
            sign_y = -1.0 if self.invert_tilt else 1.0

            cur_rel_x = rel_x - sign_x * float(self.pan - self.roi_ref_pan) / fov_h_units
            cur_rel_y = rel_y - sign_y * float(self.tilt - self.roi_ref_tilt) / fov_v_units

            cur_cx = 0.5 + cur_rel_x * cur_zoom
            cur_cy = 0.5 + cur_rel_y * cur_zoom
            zoom_ratio = cur_zoom / ref_zoom
            cur_w = min(1.0, self.roi_w * zoom_ratio)
            cur_h = min(1.0, self.roi_h * zoom_ratio)

            x1 = max(0.0, min(0.92, cur_cx - 0.5 * cur_w))
            y1 = max(0.0, min(0.92, cur_cy - 0.5 * cur_h))
            x2 = max(x1 + 0.08, min(1.0, cur_cx + 0.5 * cur_w))
            y2 = max(y1 + 0.08, min(1.0, cur_cy + 0.5 * cur_h))
            return (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))

    def return_to_home(self) -> None:
        """Return camera to the configured Home/Feeder preset position, zoom, FOV, and focus."""
        self.set_fov(self.home_fov)
        self.set_zoom(self.home_zoom, force=True)
        self.set_focus(self.home_autofocus, self.home_focus)
        self.set_pan_tilt(self.home_pan, self.home_tilt, force=True)

    def _clamp_step(self, val: int, low: int, high: int, step: int) -> int:
        clamped = max(low, min(high, val))
        return int(round(clamped / step) * step)

    def set_pan_tilt(self, pan: int, tilt: int, force: bool = False) -> bool:
        """Set absolute UVC pan and tilt values on the camera."""
        now = time.monotonic()
        with self._lock:
            if not self.system_power_on:
                return False
            if not force and (now - self._last_cmd_ts) < self.min_update_interval_sec:
                return False

            new_pan = self._clamp_step(pan, self.pan_min, self.pan_max, self.pan_step)
            new_tilt = self._clamp_step(tilt, self.tilt_min, self.tilt_max, self.tilt_step)
            if not force and new_pan == self.pan and new_tilt == self.tilt:
                return False

            self.pan = new_pan
            self.tilt = new_tilt
            self._last_cmd_ts = now
            self._last_motion_ts = now

        subprocess.Popen(
            [
                "v4l2-ctl",
                "-d",
                self.device,
                "--set-ctrl",
                f"pan_absolute={self.pan},tilt_absolute={self.tilt}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True

    def set_zoom(self, zoom_level: float, force: bool = False) -> bool:
        """Set camera hardware zoom between zoom_min (1.0x) and zoom_max (4.0x) via OBSBOT SDK & UVC."""
        now = time.monotonic()
        clamped = max(self.zoom_min, min(self.zoom_max, float(zoom_level)))
        # OBSBOT Tiny SE maps 1.0x..4.0x across 12 hardware steps (0..12, 0.25x per step)
        zoom_abs = max(0, min(12, int(round((clamped - 1.0) / 3.0 * 12.0))))
        effective_zoom = round(1.0 + (zoom_abs / 12.0) * 3.0, 2)

        with self._lock:
            if not self.system_power_on:
                return False
            if not force and abs(effective_zoom - self.zoom) < 0.05:
                return False
            if not force and (now - self._last_zoom_ts) < self.min_zoom_interval_sec:
                return False
            self.zoom = effective_zoom
            self._last_zoom_ts = now
            self._last_motion_ts = now
            sdk_dev = self._sdk_dev

        # 1. Drive via native OBSBOT C++ SDK if connected (libdev maps 1.00..1.12 -> hardware step 0..12)
        sdk_ok = False
        if sdk_dev is not None:
            try:
                sdk_ok = sdk_dev.set_zoom(1.0 + zoom_abs * 0.01) == 0
            except Exception:
                sdk_ok = False

        # 2. Also synchronize V4L2 UVC zoom_absolute control
        if not sdk_ok:
            subprocess.Popen(
                [
                    "v4l2-ctl",
                    "-d",
                    self.device,
                    "--set-ctrl",
                    f"zoom_absolute={zoom_abs}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True

    def set_fov(self, fov_deg: int) -> bool:
        """Set OBSBOT Tiny SE hardware Field of View (86 = Wide, 78 = Medium, 65 = Narrow)."""
        if fov_deg not in (86, 78, 65):
            fov_deg = 65
        with self._lock:
            self.fov = fov_deg
            sdk_dev = self._sdk_dev

        if sdk_dev is not None:
            try:
                return sdk_dev.set_fov(fov_deg) == 0
            except Exception:
                pass
        return False

    def set_focus(self, autofocus: bool, focus_val: int | None = None) -> bool:
        """Configure OBSBOT Tiny SE motorized lens autofocus or manual focus (0..100)."""
        with self._lock:
            self.autofocus = bool(autofocus)
            if focus_val is not None:
                self.focus = max(0, min(100, int(focus_val)))
            af_int = 1 if self.autofocus else 0
            f_val = self.focus
            sdk_dev = self._sdk_dev

        if sdk_dev is not None:
            try:
                sdk_dev.set_face_focus(False)
            except Exception:
                pass

        if af_int == 1:
            ctrl_str = "focus_automatic_continuous=1"
        else:
            ctrl_str = f"focus_automatic_continuous=0,focus_absolute={f_val}"

        try:
            subprocess.Popen(
                [
                    "v4l2-ctl",
                    "-d",
                    self.device,
                    "--set-ctrl",
                    ctrl_str,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return False
        return True

    def zoom_in(self, step: float | None = None) -> float:
        """Zoom in by step."""
        st = step or self.zoom_step
        self.set_zoom(self.zoom + st, force=True)
        return self.zoom

    def zoom_out(self, step: float | None = None) -> float:
        """Zoom out by step."""
        st = step or self.zoom_step
        self.set_zoom(self.zoom - st, force=True)
        return self.zoom

    def center(self) -> None:
        """Return camera to absolute center pan=0, tilt=0, zoom=1.0."""
        self.set_zoom(1.0, force=True)
        self.set_pan_tilt(0, 0, force=True)

    def standby_camera(self) -> None:
        """Put OBSBOT Tiny SE into sleep/standby mode."""
        with self._lock:
            self.system_power_on = False
            self.tracking_enabled = False
            sdk_dev = self._sdk_dev
        if sdk_dev is not None:
            try:
                sdk_dev.gimbal_stop()
                sdk_dev.sleep()
                return
            except Exception:
                pass
        if Path(OBSBOT_CLI).is_file():
            subprocess.run(
                [OBSBOT_CLI, "stop"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            subprocess.run(
                [OBSBOT_CLI, "sleep"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    def wakeup_camera(self) -> None:
        """Wake OBSBOT Tiny SE from standby, connect SDK, and return to Home preset."""
        with self._lock:
            sdk_dev = self._sdk_dev
        if sdk_dev is None:
            self.connect_sdk(wait_ms=3500)
            sdk_dev = self._sdk_dev

        if sdk_dev is not None:
            try:
                sdk_dev.wakeup()
                time.sleep(0.8)
            except Exception:
                pass
        elif Path(OBSBOT_CLI).is_file():
            subprocess.run(
                [OBSBOT_CLI, "wake"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            time.sleep(1.2)
        with self._lock:
            self.system_power_on = True
            self.tracking_enabled = True
        self.return_to_home()

    def nudge(self, d_pan: int, d_tilt: int) -> None:
        """Manual jog nudge of camera PTZ."""
        self.set_pan_tilt(self.pan + d_pan, self.tilt + d_tilt, force=True)

    def is_centered_on(self, target_x: float, target_y: float, tolerance: float | None = None) -> bool:
        """Check if normalized target coordinates are within the center deadzone."""
        tol = tolerance if tolerance is not None else (self.deadzone_norm * 1.15)
        return abs(target_x - 0.5) <= tol and abs(target_y - 0.5) <= tol

    def track_target(self, target_x: float, target_y: float) -> bool:
        """Update pan/tilt proportionally (scaled inversely by current zoom) to center target."""
        if not self.tracking_enabled or not self.system_power_on:
            return False

        err_x = target_x - 0.5
        err_y = target_y - 0.5

        zoom_scale = 1.0 / max(1.0, self.zoom)
        d_pan = 0
        d_tilt = 0

        if abs(err_x) > self.deadzone_norm:
            sign_x = -1.0 if self.invert_pan else 1.0
            d_pan = int(sign_x * err_x * self.kp_pan * zoom_scale)

        if abs(err_y) > self.deadzone_norm:
            sign_y = -1.0 if self.invert_tilt else 1.0
            d_tilt = int(sign_y * err_y * self.kp_tilt * zoom_scale)

        if d_pan == 0 and d_tilt == 0:
            return False

        prev_pan, prev_tilt = self.pan, self.tilt
        moved = self.set_pan_tilt(self.pan + d_pan, self.tilt + d_tilt)
        if moved:
            spotter_log.debug_throttled(
                "ptz_track",
                "PTZ",
                f"Gimbal step: aim=({target_x:.2f},{target_y:.2f}) err=({err_x:+.2f},{err_y:+.2f}) -> "
                f"pan {prev_pan}->{self.pan} ({self.pan - prev_pan:+d}), tilt {prev_tilt}->{self.tilt} ({self.tilt - prev_tilt:+d})",
                interval_sec=0.35,
            )
        return moved

    def adjust_zoom_for_area(
        self,
        current_area_ratio: float,
        target_min_area: float = 0.10,
        target_max_area: float = 0.35,
    ) -> bool:
        """Dynamically adjust zoom so the bird's bounding box occupies target_min_area..target_max_area of the frame."""
        if not self.tracking_enabled or not self.system_power_on:
            return False
        if current_area_ratio <= 0.001:
            return False

        if target_min_area <= current_area_ratio <= target_max_area:
            return False

        target_mid = (target_min_area + target_max_area) * 0.5
        scale_factor = (target_mid / current_area_ratio) ** 0.5
        desired_zoom = max(self.home_zoom, self.zoom * scale_factor)

        # Smooth zoom step so we don't jump more than 0.5x in a single update
        delta = max(-0.5, min(0.5, desired_zoom - self.zoom))
        if abs(delta) < 0.20:
            return False

        prev_zoom = self.zoom
        moved = self.set_zoom(self.zoom + delta)
        if moved:
            spotter_log.debug_throttled(
                "ptz_zoom",
                "PTZ",
                f"Zoom step: area={current_area_ratio * 100:.1f}% (target {target_min_area * 100:.0f}-{target_max_area * 100:.0f}%) -> "
                f"zoom {prev_zoom:.2f}x -> {self.zoom:.2f}x",
                interval_sec=0.35,
            )
        return moved

    def auto_frame_detection(
        self,
        det,
        target_min_area: float = 0.10,
        target_max_area: float = 0.35,
    ) -> bool:
        """Center bird in frame and adjust zoom dynamically once roughly centered."""
        if det is None:
            return False
        moved_pt = self.track_target(det.aim_x, det.aim_y)
        moved_zoom = False
        if abs(det.aim_x - 0.5) < 0.22 and abs(det.aim_y - 0.5) < 0.22:
            moved_zoom = self.adjust_zoom_for_area(
                det.area_ratio,
                target_min_area=target_min_area,
                target_max_area=target_max_area,
            )
        return moved_pt or moved_zoom
