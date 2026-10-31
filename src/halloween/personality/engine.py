"""Bipolar personality state machine for the animatronic skull."""

from dataclasses import dataclass, field
from enum import Enum
import time
from halloween.vision.detector import Detection


class Mood(str, Enum):
    IDLE = "IDLE"
    SWEET = "SWEET"       # Charming, overly polite host, loves pets/animals
    MENACING = "MENACING" # Spooky, roasting, territorial when someone looms close
    CHAOTIC = "CHAOTIC"   # Unhinged/paranoid when outnumbered or mid-mood-swing


HUMAN_LABELS = {"person", "upper body", "frontal face", "side face"}


@dataclass
class BipolarEngine:
    """Evaluates detections and time triggers to shift the skull's mood."""

    looming_bbox_area_ratio: float = 0.22
    crowd_threshold: int = 2
    mood_swing_interval_sec: float = 20.0
    sweet_classes: list[str] = field(default_factory=lambda: ["cat", "dog", "bird", "horse"])

    current_mood: Mood = Mood.IDLE
    last_swing_ts: float = field(default_factory=time.monotonic)
    inverted_phase: bool = False
    _locked_track_id: int | None = field(default=None, init=False, repr=False)

    def _effective_priority(self, det: Detection) -> float:
        """Score detections so human faces/bodies and pets take gaze priority over background furniture."""
        if det.label in ("frontal face", "side face"):
            # Boost face boxes so they can win primary lock when person box is disabled or equally close
            return det.area_ratio * 6.5
        if det.label == "upper body":
            return det.area_ratio * 2.2
        if det.label == "person" or det.label in self.sweet_classes:
            return det.area_ratio * 1.5
        return det.area_ratio

    def evaluate(self, detections: list[Detection]) -> tuple[Mood, Detection | None]:
        """Select primary target with sticky hysteresis and compute current bipolar mood."""
        now = time.monotonic()
        if now - self.last_swing_ts >= self.mood_swing_interval_sec:
            self.inverted_phase = not self.inverted_phase
            self.last_swing_ts = now

        if not detections:
            self.current_mood = Mood.IDLE
            self._locked_track_id = None
            return self.current_mood, None

        best_candidate = max(detections, key=self._effective_priority)
        primary = best_candidate
        if self._locked_track_id is not None:
            locked_matches = [d for d in detections if d.track_id == self._locked_track_id]
            if locked_matches:
                locked_det = locked_matches[0]
                # Keep tracking the currently locked target unless challenger is >50% stronger
                if self._effective_priority(best_candidate) <= self._effective_priority(locked_det) * 1.50:
                    primary = locked_det

        self._locked_track_id = primary.track_id

        # Count distinct people without double-counting person + upper body + face of the same person
        if any(d.label == "person" for d in detections):
            person_count = sum(1 for d in detections if d.label == "person")
        elif any(d.label == "upper body" for d in detections):
            person_count = sum(1 for d in detections if d.label == "upper body")
        else:
            person_count = sum(1 for d in detections if d.label in ("frontal face", "side face"))

        if person_count >= self.crowd_threshold:
            self.current_mood = Mood.CHAOTIC
            return self.current_mood, primary

        # Looming check (scaled for faces if primary is a face box)
        effective_area = (
            primary.area_ratio * 5.0
            if primary.label in ("frontal face", "side face")
            else primary.area_ratio
        )
        if effective_area >= self.looming_bbox_area_ratio:
            self.current_mood = Mood.SWEET if self.inverted_phase else Mood.MENACING
            return self.current_mood, primary

        if primary.label in self.sweet_classes:
            self.current_mood = Mood.SWEET
        else:
            self.current_mood = Mood.MENACING if self.inverted_phase else Mood.SWEET

        return self.current_mood, primary
