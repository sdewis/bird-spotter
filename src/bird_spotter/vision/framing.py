"""Deterministic PTZ Auto-Framing and Burst Photography State Machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import re
import time
from typing import Any, Callable

import cv2
import numpy as np

from bird_spotter.db import compute_quality_score
from bird_spotter.vision.detector import (
    Detection,
    compute_crop_sharpness,
    extract_bird_crop,
)
from bird_spotter.vision.ptz import ObsbotPTZ


class FramingState(str, Enum):
    """States of the automatic bird framing and photography controller."""

    WATCHING_HOME = "WATCHING_HOME"
    FRAMING_BIRD = "FRAMING_BIRD"
    SETTLING_BURST = "SETTLING_BURST"
    RETURNING_HOME = "RETURNING_HOME"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


@dataclass
class BurstCandidate:
    """Single frame collected during SETTLING_BURST."""

    bgr_frame: np.ndarray
    bgr_crop: np.ndarray
    detection: Detection
    sharpness: float
    zoom: float


def save_clean_snapshot_and_thumb(
    bgr_frame: np.ndarray,
    bgr_crop: np.ndarray | None,
    species_name: str,
    snapshots_dir: Path,
    now_dt: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Save clean full-resolution JPEG and square thumbnail crop to snapshots_dir."""
    if bgr_frame is None or bgr_frame.size == 0:
        return None, None

    dt = now_dt or datetime.now()
    ts_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    slug = re.sub(r"[^a-z0-9]+", "_", species_name.lower()).strip("_") or "bird"

    snapshots_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = snapshots_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    snap_filename = f"bird_{ts_str}_{slug}.jpg"
    snap_path = snapshots_dir / snap_filename
    ok = cv2.imwrite(str(snap_path), bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return None, None

    thumb_rel: str | None = snap_filename
    if bgr_crop is not None and bgr_crop.size > 0:
        thumb_img = cv2.resize(bgr_crop, (320, 320), interpolation=cv2.INTER_AREA)
        thumb_name = f"thumb_{ts_str}_{slug}.jpg"
        thumb_path = thumbs_dir / thumb_name
        if cv2.imwrite(str(thumb_path), thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            thumb_rel = f"thumbs/{thumb_name}"

    return snap_filename, thumb_rel


@dataclass
class AutoFrameController:
    """Manages PTZ auto-framing, dynamic zoom, burst sharpness selection, and return-to-home."""

    ptz: ObsbotPTZ
    target_min_area: float = 0.25
    target_max_area: float = 0.40
    settle_delay_sec: float = 0.35
    burst_count: int = 5
    max_framing_sec: float = 7.5
    lost_timeout_sec: float = 1.8
    return_home_settle_sec: float = 2.2
    manual_pause_sec: float = 60.0

    state: FramingState = FramingState.WATCHING_HOME
    state_entered_ts: float = field(default_factory=time.monotonic)
    last_bird_seen_ts: float = 0.0
    manual_override_until_ts: float = 0.0
    locked_track_id: int | None = None

    _burst_buffer: list[BurstCandidate] = field(default_factory=list, init=False, repr=False)

    def _transition(self, new_state: FramingState, now: float | None = None) -> None:
        t = now if now is not None else time.monotonic()
        self.state = new_state
        self.state_entered_ts = t
        if new_state != FramingState.SETTLING_BURST:
            self._burst_buffer.clear()

    def trigger_manual_override(
        self, duration_sec: float | None = None, now: float | None = None
    ) -> None:
        """Pause automatic framing for duration_sec (default 60s) when user touches PTZ controls."""
        t = now if now is not None else time.monotonic()
        dur = duration_sec if duration_sec is not None else self.manual_pause_sec
        self.manual_override_until_ts = t + dur
        self._transition(FramingState.MANUAL_OVERRIDE, now=t)

    def resume_auto_tracking(self, now: float | None = None) -> None:
        """Clear manual override timer and return to WATCHING_HOME."""
        t = now if now is not None else time.monotonic()
        self.manual_override_until_ts = 0.0
        self.ptz.tracking_enabled = True
        self._transition(FramingState.WATCHING_HOME, now=t)

    def manual_override_remaining(self, now: float | None = None) -> float:
        """Seconds remaining on the 60s manual override pause."""
        t = now if now is not None else time.monotonic()
        if self.state != FramingState.MANUAL_OVERRIDE:
            return 0.0
        if not self.ptz.tracking_enabled:
            return -1.0  # Indefinitely locked in manual mode
        return max(0.0, round(self.manual_override_until_ts - t, 1))

    @property
    def status_label(self) -> str:
        """Human-readable status badge text for the dashboard."""
        if not self.ptz.system_power_on:
            return "STANDBY"
        if not self.ptz.tracking_enabled:
            return "MANUAL MODE (LOCKED)"
        if self.state == FramingState.MANUAL_OVERRIDE:
            rem = self.manual_override_remaining()
            return f"MANUAL OVERRIDE ({int(rem)}s)" if rem > 0 else "MANUAL OVERRIDE"
        if self.state == FramingState.FRAMING_BIRD:
            return f"FRAMING BIRD ({self.ptz.zoom:.1f}x)"
        if self.state == FramingState.SETTLING_BURST:
            return f"CAPTURING BURST ({len(self._burst_buffer)}/{self.burst_count})"
        if self.state == FramingState.RETURNING_HOME:
            return f"RETURNING TO HOME ({self.ptz.home_zoom:.1f}x)"
        return f"WATCHING (HOME {self.ptz.home_zoom:.1f}x)"

    def step(
        self,
        bgr_frame: np.ndarray,
        primary_det: Detection | None,
        classify_cb: Callable[[np.ndarray], Any] | None = None,
        on_capture_ready: Callable[[BurstCandidate, Any], None] | None = None,
        now: float | None = None,
    ) -> FramingState:
        """Advance the PTZ framing & photography state machine by one frame."""
        t = now if now is not None else time.monotonic()

        if not self.ptz.system_power_on:
            return self.state

        if not self.ptz.tracking_enabled:
            if self.state != FramingState.MANUAL_OVERRIDE:
                self._transition(FramingState.MANUAL_OVERRIDE, now=t)
            return self.state

        # 1. Handle MANUAL_OVERRIDE state
        if self.state == FramingState.MANUAL_OVERRIDE:
            if t >= self.manual_override_until_ts:
                self.ptz.return_to_home()
                self._transition(FramingState.RETURNING_HOME, now=t)
            return self.state

        # 2. Handle RETURNING_HOME state
        if self.state == FramingState.RETURNING_HOME:
            if (t - self.state_entered_ts) >= self.return_home_settle_sec:
                self.locked_track_id = None
                self._transition(FramingState.WATCHING_HOME, now=t)
            return self.state

        # 3. Handle WATCHING_HOME state
        if self.state == FramingState.WATCHING_HOME:
            if primary_det is not None:
                self.locked_track_id = primary_det.track_id
                self.last_bird_seen_ts = t
                self._transition(FramingState.FRAMING_BIRD, now=t)
                self.ptz.auto_frame_detection(primary_det)
            return self.state

        # 4. Handle FRAMING_BIRD state
        if self.state == FramingState.FRAMING_BIRD:
            if primary_det is not None:
                self.last_bird_seen_ts = t
                self.ptz.auto_frame_detection(primary_det)

                centered = self.ptz.is_centered_on(primary_det.aim_x, primary_det.aim_y)
                area_ok = (
                    primary_det.area_ratio >= self.target_min_area
                    or self.ptz.zoom >= (self.ptz.zoom_max - 0.15)
                )
                framing_elapsed = t - self.state_entered_ts
                motion_settled = (t - self.ptz.last_motion_ts) >= self.settle_delay_sec

                # Transition to SETTLING_BURST once centered & framed (or after partial timeout if centered)
                if (centered and area_ok and motion_settled) or (
                    centered and framing_elapsed >= (self.max_framing_sec * 0.6) and motion_settled
                ) or (framing_elapsed >= self.max_framing_sec):
                    self._transition(FramingState.SETTLING_BURST, now=t)
            else:
                if (t - self.last_bird_seen_ts) >= self.lost_timeout_sec:
                    self.ptz.return_to_home()
                    self._transition(FramingState.RETURNING_HOME, now=t)
            return self.state

        # 5. Handle SETTLING_BURST state
        if self.state == FramingState.SETTLING_BURST:
            if primary_det is not None:
                crop = extract_bird_crop(bgr_frame, primary_det, padding_ratio=0.18)
                if crop is not None:
                    sharpness = compute_crop_sharpness(crop)
                    self._burst_buffer.append(
                        BurstCandidate(
                            bgr_frame=bgr_frame.copy(),
                            bgr_crop=crop,
                            detection=primary_det,
                            sharpness=sharpness,
                            zoom=self.ptz.zoom,
                        )
                    )

            burst_timed_out = (t - self.state_entered_ts) >= 1.5
            if len(self._burst_buffer) >= self.burst_count or (
                burst_timed_out and len(self._burst_buffer) > 0
            ):
                # Pick sharpest candidate from burst via Laplacian variance
                best = max(self._burst_buffer, key=lambda c: c.sharpness)
                prediction = classify_cb(best.bgr_crop) if classify_cb is not None else None
                if on_capture_ready is not None:
                    on_capture_ready(best, prediction)

                self._burst_buffer.clear()
                self.ptz.return_to_home()
                self._transition(FramingState.RETURNING_HOME, now=t)
            elif burst_timed_out and len(self._burst_buffer) == 0:
                self.ptz.return_to_home()
                self._transition(FramingState.RETURNING_HOME, now=t)

        return self.state
