"""Hailo-8 YOLOv8 object detection, Face/Upper-Body sub-detection, and multi-frame target tracker."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import cv2
import numpy as np
from hailo_platform import (
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    HEF,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Default curated list of selectable objects requested for the control panel
DEFAULT_PANEL_CLASSES: tuple[str, ...] = (
    "person",
    "upper body",
    "frontal face",
    "side face",
    "cat",
    "dog",
    "bird",
    "horse",
    "backpack",
    "umbrella",
    "cell phone",
    "book",
    "bottle",
    "wine glass",
    "cup",
    "knife",
    "fork",
    "spoon",
    "chair",
    "couch",
    "potted plant",
    "tv",
    "car",
    "motorcycle",
    "truck",
)

HAAR_DIR = Path("/usr/share/opencv4/haarcascades")


@dataclass
class Detection:
    """Single detected object in normalized frame coordinates [0.0, 1.0]."""

    label: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    track_id: int | None = None
    custom_aim_x: float | None = None
    custom_aim_y: float | None = None

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) * 0.5

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) * 0.5

    @property
    def aim_x(self) -> float:
        """Horizontal gaze point for PTZ/skull tracking."""
        if self.custom_aim_x is not None:
            return self.custom_aim_x
        return self.center_x

    @property
    def aim_y(self) -> float:
        """Vertical gaze point for PTZ/skull tracking (targets face/eyes on people instead of mid-torso)."""
        if self.custom_aim_y is not None:
            return self.custom_aim_y
        h = self.ymax - self.ymin
        if self.label == "person":
            # Look at head/eye level (~18% down from top of person bbox) instead of 50% waist/torso
            return self.ymin + 0.18 * h
        if self.label == "upper body":
            # Look at head/face region (~28% down from top of upper-body bbox)
            return self.ymin + 0.28 * h
        return self.center_y

    @property
    def area_ratio(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)


def _iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.xmin, b.xmin)
    iy1 = max(a.ymin, b.ymin)
    ix2 = min(a.xmax, b.xmax)
    iy2 = min(a.ymax, b.ymax)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = a.area_ratio + b.area_ratio - inter
    return inter / union if union > 0.0 else 0.0


class CentroidTracker:
    """Lightweight IoU + centroid tracker to preserve target IDs across frames."""

    def __init__(self, max_age_sec: float = 1.2) -> None:
        self.max_age_sec = max_age_sec
        self._next_id = 1
        self._tracks: dict[int, tuple[Detection, float]] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        now = time.monotonic()
        self._tracks = {
            tid: (det, ts)
            for tid, (det, ts) in self._tracks.items()
            if now - ts <= self.max_age_sec
        }

        used_tracks: set[int] = set()
        for det in detections:
            best_tid: int | None = None
            best_score = 0.0
            for tid, (prev_det, _) in self._tracks.items():
                if tid in used_tracks or prev_det.label != det.label:
                    continue
                iou_val = _iou(det, prev_det)
                dist = float(
                    np.hypot(det.center_x - prev_det.center_x, det.center_y - prev_det.center_y)
                )
                score = iou_val + max(0.0, 0.35 - dist)
                if score > best_score and (iou_val > 0.1 or dist < 0.25):
                    best_score = score
                    best_tid = tid

            if best_tid is None:
                best_tid = self._next_id
                self._next_id += 1

            det.track_id = best_tid
            used_tracks.add(best_tid)
            self._tracks[best_tid] = (det, now)

        return detections


class FaceUpperBodyAnalyzer:
    """Detects frontal face, side/profile face, and upper body inside person ROIs."""

    def __init__(self) -> None:
        frontal_path = HAAR_DIR / "haarcascade_frontalface_alt2.xml"
        profile_path = HAAR_DIR / "haarcascade_profileface.xml"
        upper_path = HAAR_DIR / "haarcascade_upperbody.xml"

        self.frontal_clf = (
            cv2.CascadeClassifier(str(frontal_path)) if frontal_path.is_file() else None
        )
        self.profile_clf = (
            cv2.CascadeClassifier(str(profile_path)) if profile_path.is_file() else None
        )
        self.upper_clf = (
            cv2.CascadeClassifier(str(upper_path)) if upper_path.is_file() else None
        )

    def analyze_person(
        self,
        gray_frame: np.ndarray,
        person_det: Detection,
        want_frontal: bool,
        want_side: bool,
        want_upper: bool,
    ) -> list[Detection]:
        """Extract frontal face, side face, and upper body detections within a person bbox."""
        sub_dets: list[Detection] = []
        fh, fw = gray_frame.shape[:2]

        px1 = int(person_det.xmin * fw)
        py1 = int(person_det.ymin * fh)
        px2 = int(person_det.xmax * fw)
        py2 = int(person_det.ymax * fh)
        pw = px2 - px1
        ph = py2 - py1
        if pw < 36 or ph < 48:
            return sub_dets

        # Only scan upper 60% of person box for faces to keep CPU time < 1.5ms
        head_y2 = py1 + int(ph * 0.62)
        roi = gray_frame[py1:head_y2, px1:px2]
        rh, rw = roi.shape[:2]
        if rw < 24 or rh < 24:
            return sub_dets

        # Downscale wide ROIs to max 220px width for fast Haar cascade evaluation
        scale = min(1.0, 220.0 / float(rw))
        small = (
            cv2.resize(roi, (int(rw * scale), int(rh * scale))) if scale < 1.0 else roi
        )
        sh, sw = small.shape[:2]

        face_found: Detection | None = None

        if (want_frontal or True) and self.frontal_clf is not None:
            faces = self.frontal_clf.detectMultiScale(
                small, scaleFactor=1.18, minNeighbors=4, minSize=(22, 22)
            )
            if len(faces) > 0:
                # Pick largest frontal face
                fx, fy, f_w, f_h = max(faces, key=lambda b: b[2] * b[3])
                xmin = (px1 + fx / scale) / fw
                ymin = (py1 + fy / scale) / fh
                xmax = (px1 + (fx + f_w) / scale) / fw
                ymax = (py1 + (fy + f_h) / scale) / fh
                face_found = Detection(
                    label="frontal face",
                    confidence=min(0.98, person_det.confidence + 0.12),
                    xmin=max(0.0, min(1.0, xmin)),
                    ymin=max(0.0, min(1.0, ymin)),
                    xmax=max(0.0, min(1.0, xmax)),
                    ymax=max(0.0, min(1.0, ymax)),
                )
                if want_frontal:
                    sub_dets.append(face_found)

        if face_found is None and (want_side or True) and self.profile_clf is not None:
            # Check left-facing profile
            profiles = self.profile_clf.detectMultiScale(
                small, scaleFactor=1.18, minNeighbors=4, minSize=(22, 22)
            )
            if len(profiles) > 0:
                fx, fy, f_w, f_h = max(profiles, key=lambda b: b[2] * b[3])
                xmin = (px1 + fx / scale) / fw
                ymin = (py1 + fy / scale) / fh
                xmax = (px1 + (fx + f_w) / scale) / fw
                ymax = (py1 + (fy + f_h) / scale) / fh
                face_found = Detection(
                    label="side face",
                    confidence=min(0.95, person_det.confidence + 0.08),
                    xmin=max(0.0, min(1.0, xmin)),
                    ymin=max(0.0, min(1.0, ymin)),
                    xmax=max(0.0, min(1.0, xmax)),
                    ymax=max(0.0, min(1.0, ymax)),
                )
            else:
                # Check right-facing profile on horizontally flipped ROI
                flipped = cv2.flip(small, 1)
                profiles_r = self.profile_clf.detectMultiScale(
                    flipped, scaleFactor=1.18, minNeighbors=4, minSize=(22, 22)
                )
                if len(profiles_r) > 0:
                    fx, fy, f_w, f_h = max(profiles_r, key=lambda b: b[2] * b[3])
                    orig_fx = sw - (fx + f_w)
                    xmin = (px1 + orig_fx / scale) / fw
                    ymin = (py1 + fy / scale) / fh
                    xmax = (px1 + (orig_fx + f_w) / scale) / fw
                    ymax = (py1 + (fy + f_h) / scale) / fh
                    face_found = Detection(
                        label="side face",
                        confidence=min(0.95, person_det.confidence + 0.08),
                        xmin=max(0.0, min(1.0, xmin)),
                        ymin=max(0.0, min(1.0, ymin)),
                        xmax=max(0.0, min(1.0, xmax)),
                        ymax=max(0.0, min(1.0, ymax)),
                    )
            if face_found is not None and want_side:
                sub_dets.append(face_found)

        # Anchor the parent person detection's aim point directly onto the detected face!
        if face_found is not None:
            person_det.custom_aim_x = face_found.center_x
            person_det.custom_aim_y = face_found.center_y

        if want_upper:
            # Upper body box: top 52% of full-body person bbox (or full box if already a close-up bust)
            aspect = ph / max(1.0, float(pw))
            ub_ratio = 0.52 if aspect > 1.25 else 0.80
            ub_ymax = person_det.ymin + (person_det.ymax - person_det.ymin) * ub_ratio
            ub_det = Detection(
                label="upper body",
                confidence=person_det.confidence,
                xmin=person_det.xmin,
                ymin=person_det.ymin,
                xmax=person_det.xmax,
                ymax=min(1.0, ub_ymax),
                custom_aim_x=person_det.aim_x,
                custom_aim_y=person_det.aim_y,
            )
            sub_dets.append(ub_det)

        return sub_dets


class HailoDetector:
    """Persistent Hailo-8 YOLOv8 object detector session with Face & Upper-Body analysis."""

    def __init__(
        self,
        hef_path: str | Path,
        score_threshold: float = 0.45,
        target_classes: Iterable[str] | None = DEFAULT_PANEL_CLASSES,
    ) -> None:
        self.hef_path = Path(hef_path)
        self.score_threshold = score_threshold
        self.enabled_classes: set[str] = (
            set(target_classes) if target_classes is not None else set(DEFAULT_PANEL_CLASSES)
        )
        self.filter_target_classes: bool = True
        self.tracker = CentroidTracker()
        self.face_analyzer = FaceUpperBodyAnalyzer()

        self._vdevice: VDevice | None = None
        self._infer_ctx: InferVStreams | None = None
        self._act_ctx = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._model_w: int = 640
        self._model_h: int = 640
        self.last_infer_ms: float = 0.0

    def is_available(self) -> bool:
        return self.hef_path.is_file()

    def toggle_class(self, cls_name: str) -> None:
        if cls_name in self.enabled_classes:
            self.enabled_classes.remove(cls_name)
        else:
            self.enabled_classes.add(cls_name)

    def set_classes(self, classes: Iterable[str]) -> None:
        self.enabled_classes = set(classes)

    def open(self) -> "HailoDetector":
        hef = HEF(str(self.hef_path))
        in_info = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        self._input_name = in_info.name
        self._output_name = out_info.name
        self._model_h, self._model_w = in_info.shape[0], in_info.shape[1]

        self._vdevice = VDevice()
        cfg_params = ConfigureParams.create_from_hef(
            hef=hef, interface=HailoStreamInterface.PCIe
        )
        net_groups = self._vdevice.configure(hef, cfg_params)
        net_group = net_groups[0]
        net_group_params = net_group.create_params()

        in_params = InputVStreamParams.make(net_group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(net_group, format_type=FormatType.FLOAT32)

        self._infer_ctx = InferVStreams(net_group, in_params, out_params)
        self._infer_ctx.__enter__()
        self._act_ctx = net_group.activate(net_group_params)
        self._act_ctx.__enter__()
        return self

    def close(self) -> None:
        if self._act_ctx is not None:
            self._act_ctx.__exit__(None, None, None)
            self._act_ctx = None
        if self._infer_ctx is not None:
            self._infer_ctx.__exit__(None, None, None)
            self._infer_ctx = None
        if self._vdevice is not None:
            self._vdevice.release()
            self._vdevice = None

    def __enter__(self) -> "HailoDetector":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def detect(self, bgr_frame: np.ndarray) -> list[Detection]:
        """Run YOLOv8 inference + Face/Upper-Body sub-detection on a BGR frame."""
        if self._infer_ctx is None:
            raise RuntimeError("HailoDetector is not open.")

        t0 = time.perf_counter()
        resized = cv2.resize(bgr_frame, (self._model_w, self._model_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb, axis=0)

        raw = self._infer_ctx.infer({self._input_name: input_tensor})
        nms_classes = raw[self._output_name][0]

        want_person = (not self.filter_target_classes) or ("person" in self.enabled_classes)
        want_upper = (not self.filter_target_classes) or ("upper body" in self.enabled_classes)
        want_frontal = (not self.filter_target_classes) or ("frontal face" in self.enabled_classes)
        want_side = (not self.filter_target_classes) or ("side face" in self.enabled_classes)
        need_person_scan = want_person or want_upper or want_frontal or want_side

        detections: list[Detection] = []
        person_candidates: list[Detection] = []

        for cls_id, boxes in enumerate(nms_classes):
            if len(boxes) == 0:
                continue
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"

            if label == "person":
                if not need_person_scan:
                    continue
            elif self.filter_target_classes and label not in self.enabled_classes:
                continue

            for box in boxes:
                ymin, xmin, ymax, xmax, score = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(box[4]),
                )
                if score < self.score_threshold:
                    continue
                det = Detection(
                    label=label,
                    confidence=score,
                    xmin=max(0.0, min(1.0, xmin)),
                    ymin=max(0.0, min(1.0, ymin)),
                    xmax=max(0.0, min(1.0, xmax)),
                    ymax=max(0.0, min(1.0, ymax)),
                )
                if label == "person":
                    person_candidates.append(det)
                else:
                    detections.append(det)

        if person_candidates:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            for p_det in person_candidates:
                sub_dets = self.face_analyzer.analyze_person(
                    gray_frame=gray,
                    person_det=p_det,
                    want_frontal=want_frontal,
                    want_side=want_side,
                    want_upper=want_upper,
                )
                if want_person:
                    detections.append(p_det)
                detections.extend(sub_dets)

        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return self.tracker.update(detections)
