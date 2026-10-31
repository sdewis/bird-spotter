"""UVC Pan/Tilt/Zoom controller, Skew-Aware Servo Calibration Manager, and target tracker."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_PATH = PROJECT_ROOT / "config" / "calibration.json"

DEFAULT_CALIB = {
    "calibrated": False,
    "cam_pan_center": 0,
    "cam_tilt_center": 0,
    "cam_pan_left": 260000,
    "cam_pan_right": -260000,
    "cam_tilt_up": 180000,
    "cam_tilt_down": -180000,
    "servo_pan_center": 90,
    "servo_tilt_center": 90,
    "servo_pan_left": 175,
    "servo_pan_right": 5,
    "servo_tilt_up": 165,
    "servo_tilt_down": 15,
}


def _interp_skewed_axis(
    val: float,
    x_neg: float,
    x_center: float,
    x_pos: float,
    y_neg: float,
    y_center: float,
    y_pos: float,
) -> int:
    """Two-segment piecewise linear interpolation anchored at a potentially skewed (x_center, y_center).

    Segment 1: (x_center, y_center) -> (x_pos, y_pos)
    Segment 2: (x_center, y_center) -> (x_neg, y_neg)
    Handles arbitrary sign conventions or inverted servo linkages on either side of center.
    """
    if abs(val - x_center) < 1e-6:
        return int(round(y_center))

    # Determine which side of x_center `val` lies on by checking sign of (val - x_center)
    delta = val - x_center
    delta_pos = x_pos - x_center
    delta_neg = x_neg - x_center

    # If `val` is on the `x_pos` side of `x_center`
    if (delta * delta_pos) > 0 and abs(delta_pos) > 1e-6:
        t = max(0.0, min(1.0, delta / delta_pos))
        return int(round(y_center + t * (y_pos - y_center)))

    # If `val` is on the `x_neg` side of `x_center`
    if (delta * delta_neg) > 0 and abs(delta_neg) > 1e-6:
        t = max(0.0, min(1.0, delta / delta_neg))
        return int(round(y_center + t * (y_neg - y_center)))

    # Fallback if x_pos and x_neg were calibrated on the same side of x_center
    if abs(val - x_pos) < abs(val - x_neg) and abs(delta_pos) > 1e-6:
        t = max(0.0, min(1.0, delta / delta_pos))
        return int(round(y_center + t * (y_pos - y_center)))
    if abs(delta_neg) > 1e-6:
        t = max(0.0, min(1.0, delta / delta_neg))
        return int(round(y_center + t * (y_neg - y_center)))

    return int(round(y_center))


@dataclass
class ObsbotPTZ:
    """Controls OBSBOT Tiny SE pan/tilt and manages skew-aware Camera <-> Skull Servo calibration."""

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
    invert_pan: bool = True   # Camera looking out from skull POV
    invert_tilt: bool = True  # In image coords y=0 is top, y=1 is bottom
    tracking_enabled: bool = True
    system_power_on: bool = True
    min_update_interval_sec: float = 0.08

    pan: int = 0
    tilt: int = 0
    servo_pan: int = 90
    servo_tilt: int = 90

    calib: dict = field(default_factory=lambda: dict(DEFAULT_CALIB))

    _last_cmd_ts: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.load_calibration()
        # Start at calibrated center positions
        self.pan = int(self.calib.get("cam_pan_center", 0))
        self.tilt = int(self.calib.get("cam_tilt_center", 0))
        self.servo_pan = int(self.calib.get("servo_pan_center", 90))
        self.servo_tilt = int(self.calib.get("servo_tilt_center", 90))

    def load_calibration(self) -> None:
        """Load persisted calibration from config/calibration.json if present."""
        if CALIBRATION_PATH.is_file():
            try:
                loaded = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
                merged = dict(DEFAULT_CALIB)
                merged.update(loaded)
                self.calib = merged
            except Exception as exc:
                print(f"[ObsbotPTZ] Failed to load calibration: {exc}")

    def save_calibration(self) -> None:
        """Persist current calibration dictionary to config/calibration.json."""
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_PATH.write_text(json.dumps(self.calib, indent=2), encoding="utf-8")

    def calibrate_direction(self, direction: str) -> dict:
        """Record current camera PTZ and/or skull servo position for `direction`

        Supported directions:
        - 'left', 'right', 'up', 'down': sets max for that orientation for both camera and skull
        - 'center': sets center for both camera and skull
        - 'center_cam': sets center for camera only
        - 'center_servo': sets center for skull servos only
        """
        with self._lock:
            if direction == "left":
                self.calib["cam_pan_left"] = self.pan
                self.calib["servo_pan_left"] = self.servo_pan
            elif direction == "right":
                self.calib["cam_pan_right"] = self.pan
                self.calib["servo_pan_right"] = self.servo_pan
            elif direction == "up":
                self.calib["cam_tilt_up"] = self.tilt
                self.calib["servo_tilt_up"] = self.servo_tilt
            elif direction == "down":
                self.calib["cam_tilt_down"] = self.tilt
                self.calib["servo_tilt_down"] = self.servo_tilt
            elif direction == "center":
                self.calib["cam_pan_center"] = self.pan
                self.calib["cam_tilt_center"] = self.tilt
                self.calib["servo_pan_center"] = self.servo_pan
                self.calib["servo_tilt_center"] = self.servo_tilt
            elif direction == "center_cam":
                self.calib["cam_pan_center"] = self.pan
                self.calib["cam_tilt_center"] = self.tilt
            elif direction == "center_servo":
                self.calib["servo_pan_center"] = self.servo_pan
                self.calib["servo_tilt_center"] = self.servo_tilt
            self.calib["calibrated"] = True
            self.save_calibration()
            return dict(self.calib)

    def reset_calibration(self) -> dict:
        """Forget calibration data, restore defaults, persist, and center both camera and skull."""
        with self._lock:
            self.calib = dict(DEFAULT_CALIB)
            self.save_calibration()
        self.center_all()
        return dict(self.calib)

    def _active_cam_pan_bounds(self) -> tuple[int, int]:
        if not self.calib.get("calibrated", False):
            return self.pan_min, self.pan_max
        pts = (
            int(self.calib["cam_pan_left"]),
            int(self.calib["cam_pan_center"]),
            int(self.calib["cam_pan_right"]),
        )
        return max(self.pan_min, min(pts)), min(self.pan_max, max(pts))

    def _active_cam_tilt_bounds(self) -> tuple[int, int]:
        if not self.calib.get("calibrated", False):
            return self.tilt_min, self.tilt_max
        pts = (
            int(self.calib["cam_tilt_down"]),
            int(self.calib["cam_tilt_center"]),
            int(self.calib["cam_tilt_up"]),
        )
        return max(self.tilt_min, min(pts)), min(self.tilt_max, max(pts))

    def _clamp_step(self, val: int, low: int, high: int, step: int) -> int:
        clamped = max(low, min(high, val))
        return int(round(clamped / step) * step)

    def set_pan_tilt(
        self,
        pan: int,
        tilt: int,
        force: bool = False,
        apply_calib_bounds: bool = True,
        sync_servos: bool = True,
    ) -> bool:
        """Set absolute UVC pan and tilt values, optionally syncing skull servo targets."""
        now = time.monotonic()
        with self._lock:
            if not self.system_power_on:
                return False
            if not force and (now - self._last_cmd_ts) < self.min_update_interval_sec:
                return False

            p_min, p_max = (
                self._active_cam_pan_bounds()
                if apply_calib_bounds
                else (self.pan_min, self.pan_max)
            )
            t_min, t_max = (
                self._active_cam_tilt_bounds()
                if apply_calib_bounds
                else (self.tilt_min, self.tilt_max)
            )

            new_pan = self._clamp_step(pan, p_min, p_max, self.pan_step)
            new_tilt = self._clamp_step(tilt, t_min, t_max, self.tilt_step)
            if not force and new_pan == self.pan and new_tilt == self.tilt:
                return False

            self.pan = new_pan
            self.tilt = new_tilt
            self._last_cmd_ts = now
            if sync_servos:
                self.servo_pan, self.servo_tilt = self._compute_mapped_servo_angles_unlocked()

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

    def center(self) -> None:
        """Return camera to its calibrated center position immediately."""
        c_pan = int(self.calib.get("cam_pan_center", 0))
        c_tilt = int(self.calib.get("cam_tilt_center", 0))
        self.set_pan_tilt(
            c_pan,
            c_tilt,
            force=True,
            apply_calib_bounds=False,
            sync_servos=False,
        )

    def center_servos(self) -> None:
        """Return skull servos to their calibrated center degrees."""
        with self._lock:
            self.servo_pan = int(self.calib.get("servo_pan_center", 90))
            self.servo_tilt = int(self.calib.get("servo_tilt_center", 90))

    def center_all(self) -> None:
        """Return both camera and skull servos to their calibrated center positions."""
        self.center()
        self.center_servos()

    def standby_camera(self) -> None:
        """Stop all camera positioning and put OBSBOT Tiny SE into SDK sleep/standby mode."""
        with self._lock:
            self.system_power_on = False
            self.tracking_enabled = False
        subprocess.run(
            ["/usr/local/bin/obsbot-cli", "stop"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["/usr/local/bin/obsbot-cli", "sleep"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def wakeup_camera(self) -> tuple[int, int]:
        """Wake OBSBOT Tiny SE from SDK standby and move camera + skull to calibrated centre (or default home)."""
        subprocess.run(
            ["/usr/local/bin/obsbot-cli", "wake"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(1.2)
        with self._lock:
            self.system_power_on = True
            self.tracking_enabled = True
            if self.calib.get("calibrated", False):
                c_pan = int(self.calib.get("cam_pan_center", 0))
                c_tilt = int(self.calib.get("cam_tilt_center", 0))
                s_pan = int(self.calib.get("servo_pan_center", 90))
                s_tilt = int(self.calib.get("servo_tilt_center", 90))
            else:
                c_pan, c_tilt = 0, 0
                s_pan, s_tilt = 90, 90
            self.servo_pan = s_pan
            self.servo_tilt = s_tilt

        self.set_pan_tilt(
            c_pan,
            c_tilt,
            force=True,
            apply_calib_bounds=False,
            sync_servos=False,
        )
        return s_pan, s_tilt

    def nudge_camera_manual(self, d_pan: int, d_tilt: int) -> None:
        """Disengage auto-track and nudge only the camera PTZ (allowing independent calibration)."""
        self.tracking_enabled = False
        self.set_pan_tilt(
            self.pan + d_pan,
            self.tilt + d_tilt,
            force=True,
            apply_calib_bounds=False,
            sync_servos=False,
        )

    def nudge_servo_manual(self, d_pan_deg: int, d_tilt_deg: int) -> None:
        """Disengage auto-track and nudge the skull Pan/Tilt servos directly."""
        self.tracking_enabled = False
        with self._lock:
            self.servo_pan = max(0, min(180, self.servo_pan + d_pan_deg))
            self.servo_tilt = max(0, min(180, self.servo_tilt + d_tilt_deg))

    def track_target(self, target_x: float, target_y: float) -> bool:
        """Update pan/tilt proportionally to center a target, scaling for asymmetric/skewed calibration."""
        if not self.tracking_enabled or not self.system_power_on:
            return False

        err_x = target_x - 0.5
        err_y = target_y - 0.5

        d_pan = 0
        d_tilt = 0

        if abs(err_x) > self.deadzone_norm:
            sign_x = -1.0 if self.invert_pan else 1.0
            raw_step_x = sign_x * err_x
            # Scale gain proportionally if calibrated Left/Right spans around cam_pan_center are skewed
            c_pan = float(self.calib.get("cam_pan_center", 0))
            span_l = abs(float(self.calib.get("cam_pan_left", 260000)) - c_pan)
            span_r = abs(float(self.calib.get("cam_pan_right", -260000)) - c_pan)
            avg_span_x = max(1.0, 0.5 * (span_l + span_r))
            dir_span_x = (
                span_l
                if (raw_step_x * (float(self.calib.get("cam_pan_left", 260000)) - c_pan)) > 0
                else span_r
            )
            skew_scale_x = max(0.35, min(2.0, dir_span_x / avg_span_x))
            d_pan = int(raw_step_x * self.kp_pan * skew_scale_x)

        if abs(err_y) > self.deadzone_norm:
            sign_y = -1.0 if self.invert_tilt else 1.0
            raw_step_y = sign_y * err_y
            c_tilt = float(self.calib.get("cam_tilt_center", 0))
            span_u = abs(float(self.calib.get("cam_tilt_up", 180000)) - c_tilt)
            span_d = abs(float(self.calib.get("cam_tilt_down", -180000)) - c_tilt)
            avg_span_y = max(1.0, 0.5 * (span_u + span_d))
            dir_span_y = (
                span_u
                if (raw_step_y * (float(self.calib.get("cam_tilt_up", 180000)) - c_tilt)) > 0
                else span_d
            )
            skew_scale_y = max(0.35, min(2.0, dir_span_y / avg_span_y))
            d_tilt = int(raw_step_y * self.kp_tilt * skew_scale_y)

        if d_pan == 0 and d_tilt == 0:
            return False

        return self.set_pan_tilt(
            self.pan + d_pan,
            self.tilt + d_tilt,
            apply_calib_bounds=True,
            sync_servos=True,
        )

    def _compute_mapped_servo_angles_unlocked(self) -> tuple[int, int]:
        """Map current camera (pan, tilt) to skull servo degrees via skew-aware two-segment interpolation."""
        s_pan = _interp_skewed_axis(
            val=float(self.pan),
            x_neg=float(self.calib["cam_pan_right"]),
            x_center=float(self.calib.get("cam_pan_center", 0)),
            x_pos=float(self.calib["cam_pan_left"]),
            y_neg=float(self.calib["servo_pan_right"]),
            y_center=float(self.calib.get("servo_pan_center", 90)),
            y_pos=float(self.calib["servo_pan_left"]),
        )
        s_tilt = _interp_skewed_axis(
            val=float(self.tilt),
            x_neg=float(self.calib["cam_tilt_down"]),
            x_center=float(self.calib.get("cam_tilt_center", 0)),
            x_pos=float(self.calib["cam_tilt_up"]),
            y_neg=float(self.calib["servo_tilt_down"]),
            y_center=float(self.calib.get("servo_tilt_center", 90)),
            y_pos=float(self.calib["servo_tilt_up"]),
        )
        return max(0, min(180, s_pan)), max(0, min(180, s_tilt))

    def get_active_servo_angles(self) -> tuple[int, int]:
        """Return current skull servo (pan, tilt) angles in 0..180 degrees."""
        with self._lock:
            if self.tracking_enabled:
                self.servo_pan, self.servo_tilt = self._compute_mapped_servo_angles_unlocked()
            return self.servo_pan, self.servo_tilt
