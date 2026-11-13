"""Hailo-8 YOLOv8 bird and wildlife detection and multi-frame target tracker."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
try:
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

    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    ConfigureParams = FormatType = HailoStreamInterface = HEF = None  # type: ignore[assignment,misc]
    InferVStreams = InputVStreamParams = OutputVStreamParams = VDevice = None  # type: ignore[assignment,misc]

from bird_spotter.vision.classifier import BirdSpeciesClassifier, SpeciesPrediction

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

COMMON_BRITISH_BIRDS: tuple[str, ...] = (
    "Robin",
    "Blue Tit",
    "Great Tit",
    "Blackbird",
    "House Sparrow",
    "Goldfinch",
    "Magpie",
    "Woodpigeon",
    "Chaffinch",
    "Starling",
    "Wren",
    "Collared Dove",
    "Dunnock",
    "Jay",
    "Coal Tit",
    "Long-tailed Tit",
    "Song Thrush",
    "Greenfinch",
    "Nuthatch",
    "Great Spotted Woodpecker",
)

DEFAULT_SPOTTER_CLASSES: tuple[str, ...] = (
    "bird",
    "cat",
    "dog",
    "person",
)


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
    species: str | None = None
    scientific_name: str = ""
    species_confidence: float | None = None
    top_predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) * 0.5

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) * 0.5

    @property
    def aim_x(self) -> float:
        """Horizontal gaze point for PTZ tracking."""
        return self.center_x

    @property
    def aim_y(self) -> float:
        """Vertical gaze point for PTZ tracking."""
        return self.center_y

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area_ratio(self) -> float:
        return self.width * self.height

    @property
    def display_name(self) -> str:
        """User-friendly name showing species if identified, or generic label."""
        if self.species:
            conf = self.species_confidence or self.confidence
            return f"{self.species} ({conf:.0%})"
        return f"{self.label.title()} ({self.confidence:.0%})"


def extract_bird_crop(
    bgr_frame: np.ndarray, det: Detection, padding_ratio: float = 0.15
) -> np.ndarray | None:
    """Extract a padded BGR crop around a Detection in pixel coordinates."""
    if bgr_frame is None or bgr_frame.size == 0:
        return None
    h, w = bgr_frame.shape[:2]
    bw = det.width * w
    bh = det.height * h
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    x1 = max(0, int(round(det.xmin * w - pad_x)))
    y1 = max(0, int(round(det.ymin * h - pad_y)))
    x2 = min(w, int(round(det.xmax * w + pad_x)))
    y2 = min(h, int(round(det.ymax * h + pad_y)))

    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None
    return bgr_frame[y1:y2, x1:x2].copy()


def compute_crop_sharpness(bgr_crop: np.ndarray) -> float:
    """Compute Laplacian variance sharpness metric on a BGR crop."""
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def filter_detections_by_roi(
    detections: list[Detection],
    roi: tuple[float, float, float, float] | None,
) -> list[Detection]:
    """Filter detections to only those whose center (center_x, center_y) falls inside the ROI bounds."""
    if roi is None:
        return detections
    rx1, ry1, rx2, ry2 = roi
    return [
        det
        for det in detections
        if rx1 <= det.center_x <= rx2 and ry1 <= det.center_y <= ry2
    ]


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
    """IoU + centroid tracker with multi-frame confirmation, coordinate smoothing, and static false-positive suppression."""

    def __init__(
        self,
        max_age_sec: float = 0.85,
        min_confirm_frames: int = 3,
        ema_alpha: float = 0.45,
    ) -> None:
        self.max_age_sec = max_age_sec
        self.min_confirm_frames = min_confirm_frames
        self.ema_alpha = ema_alpha
        self._next_id = 1
        # tid -> {"det": Detection, "ts": float, "hits": int, "anchor_x": float, "anchor_y": float, "first_seen_ts": float, "max_displacement": float}
        self._tracks: dict[int, dict[str, Any]] = {}
        # list of (anchor_x, anchor_y, added_ts) for static false positives
        self._static_mask: list[tuple[float, float, float]] = []

    def assign_species(
        self,
        track_id: int,
        species: str,
        scientific_name: str,
        confidence: float,
        top_predictions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist species classification onto an active track."""
        if track_id in self._tracks:
            trk = self._tracks[track_id]
            det = trk["det"]
            det.species = species
            det.scientific_name = scientific_name
            det.species_confidence = confidence
            if top_predictions is not None:
                det.top_predictions = top_predictions

    def update(
        self, detections: list[Detection], now: float | None = None
    ) -> list[Detection]:
        cur_time = time.monotonic() if now is None else float(now)

        # 1. Expire static mask entries older than 120s
        self._static_mask = [
            (sx, sy, sts)
            for sx, sy, sts in self._static_mask
            if (cur_time - sts) <= 120.0
        ]

        # 2. Filter incoming detections against active static mask (ignore motionless twigs/leaves unless moved)
        valid_detections: list[Detection] = []
        for det in detections:
            suppressed = False
            for sx, sy, _ in self._static_mask:
                dist = float(np.hypot(det.center_x - sx, det.center_y - sy))
                if dist < 0.020 and det.confidence < 0.65:
                    suppressed = True
                    break
            if not suppressed:
                valid_detections.append(det)

        # 3. Clean up stale tracks
        cleaned: dict[int, dict[str, Any]] = {}
        for tid, trk in self._tracks.items():
            age = cur_time - trk["ts"]
            if age <= self.max_age_sec:
                decayed_hits = max(1, trk["hits"] - 1) if age > 0.15 else trk["hits"]
                trk["hits"] = decayed_hits
                cleaned[tid] = trk
        self._tracks = cleaned

        used_tracks: set[int] = set()
        confirmed_detections: list[Detection] = []

        for det in valid_detections:
            best_tid: int | None = None
            best_score = 0.0
            for tid, trk in self._tracks.items():
                if tid in used_tracks:
                    continue
                prev_det = trk["det"]
                iou_val = _iou(det, prev_det)
                dist = float(
                    np.hypot(det.center_x - prev_det.center_x, det.center_y - prev_det.center_y)
                )
                score = iou_val + max(0.0, 0.35 - dist)
                if score > best_score and (iou_val > 0.12 or dist < 0.25):
                    best_score = score
                    best_tid = tid

            if best_tid is None:
                best_tid = self._next_id
                self._next_id += 1
                hits = 1
                anchor_x = det.center_x
                anchor_y = det.center_y
                first_seen_ts = cur_time
                max_displacement = 0.0
            else:
                trk = self._tracks[best_tid]
                prev_det = trk["det"]
                hits = min(30, trk["hits"] + 1)
                anchor_x = trk["anchor_x"]
                anchor_y = trk["anchor_y"]
                first_seen_ts = trk["first_seen_ts"]
                disp = float(np.hypot(det.center_x - anchor_x, det.center_y - anchor_y))
                max_displacement = max(trk["max_displacement"], disp)

                a = self.ema_alpha
                det.xmin = a * det.xmin + (1.0 - a) * prev_det.xmin
                det.ymin = a * det.ymin + (1.0 - a) * prev_det.ymin
                det.xmax = a * det.xmax + (1.0 - a) * prev_det.xmax
                det.ymax = a * det.ymax + (1.0 - a) * prev_det.ymax
                # Preserve species assignment from previous track if not re-evaluated
                if det.species is None and prev_det.species is not None:
                    det.species = prev_det.species
                    det.scientific_name = prev_det.scientific_name
                    det.species_confidence = prev_det.species_confidence
                    det.top_predictions = prev_det.top_predictions

            det.track_id = best_tid
            used_tracks.add(best_tid)
            self._tracks[best_tid] = {
                "det": det,
                "ts": cur_time,
                "hits": hits,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "first_seen_ts": first_seen_ts,
                "max_displacement": max_displacement,
            }

            # If an object sits completely motionless for >15s, mask it as static false-positive
            is_static_false_positive = (
                (cur_time - first_seen_ts) > 15.0 and max_displacement < 0.020
            )
            if is_static_false_positive:
                if not any(
                    np.hypot(anchor_x - sx, anchor_y - sy) < 0.010
                    for sx, sy, _ in self._static_mask
                ):
                    self._static_mask.append((anchor_x, anchor_y, cur_time))
            elif hits >= self.min_confirm_frames:
                confirmed_detections.append(det)

        return confirmed_detections


class HailoDetector:
    """Persistent Hailo-8 YOLOv8 object detector session paired with British bird classifier."""

    def __init__(
        self,
        hef_path: str | Path,
        score_threshold: float = 0.50,
        target_classes: Iterable[str] | None = None,
        classifier: BirdSpeciesClassifier | None = None,
    ) -> None:
        self.hef_path = Path(hef_path)
        self.score_threshold = score_threshold
        if target_classes is not None:
            self.enabled_classes: set[str] = set(target_classes)
        else:
            self.enabled_classes = {"bird"}
        self.filter_target_classes: bool = True
        self.tracker = CentroidTracker()
        self.classifier = classifier or BirdSpeciesClassifier()

        self._vdevice: VDevice | None = None
        self._net_group: Any = None
        self._net_group_params: Any = None
        self._in_params: Any = None
        self._out_params: Any = None
        self._infer_ctx: InferVStreams | None = None
        self._act_ctx: Any = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._model_w: int = 640
        self._model_h: int = 640
        self.last_infer_ms: float = 0.0
        self._bg_gray: np.ndarray | None = None

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
        self.classifier.open_onnx_fallback()
        try:
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
            self._net_group = net_groups[0]
            self._net_group_params = self._net_group.create_params()

            self._in_params = InputVStreamParams.make(
                self._net_group, format_type=FormatType.UINT8
            )
            self._out_params = OutputVStreamParams.make(
                self._net_group, format_type=FormatType.FLOAT32
            )

            self.classifier.configure_hailo(self._vdevice)

            self._infer_ctx = InferVStreams(
                self._net_group, self._in_params, self._out_params
            )
            self._infer_ctx.__enter__()
            self._act_ctx = self._net_group.activate(self._net_group_params)
            self._act_ctx.__enter__()
        except Exception as exc:
            print(f"[HailoDetector] Warning: Hailo-8 NPU not ready yet ({exc}). Streaming camera without NPU until ready.")
            self._infer_ctx = None
        return self

    @contextmanager
    def yield_activation(self) -> Iterator[None]:
        """Temporarily suspend YOLOv8 network group activation so the Hailo classifier can run."""
        if self._act_ctx is not None:
            self._act_ctx.__exit__(None, None, None)
            self._act_ctx = None
        try:
            yield
        finally:
            if self._net_group is not None and self._act_ctx is None:
                self._act_ctx = self._net_group.activate(self._net_group_params)
                self._act_ctx.__enter__()

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

    def classify_crop(self, crop: np.ndarray) -> SpeciesPrediction | None:
        """Run fine-grained British bird species classification on a BGR crop."""
        if not self.classifier.is_ready():
            return None
        return self.classifier.classify(
            crop, hailo_context_Pause_cb=self.yield_activation
        )

    def _run_yolo_on_patch(
        self,
        bgr_frame: np.ndarray,
        px1: int,
        py1: int,
        px2: int,
        py2: int,
        threshold: float,
    ) -> list[Detection]:
        """Run YOLOv8 inference on a specific pixel patch of the frame and project back to [0..1]."""
        if self._infer_ctx is None or bgr_frame is None or bgr_frame.size == 0:
            return []
        h, w = bgr_frame.shape[:2]
        px1 = max(0, min(w, px1))
        py1 = max(0, min(h, py1))
        px2 = max(0, min(w, px2))
        py2 = max(0, min(h, py2))
        if (px2 - px1) < 16 or (py2 - py1) < 16:
            return []

        patch = bgr_frame[py1:py2, px1:px2]
        resized = cv2.resize(patch, (self._model_w, self._model_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb, axis=0)

        raw = self._infer_ctx.infer({self._input_name: input_tensor})
        nms_classes = raw[self._output_name][0]

        patch_w = float(px2 - px1)
        patch_h = float(py2 - py1)
        fw = patch_w / float(w)
        fh = patch_h / float(h)
        fx1 = float(px1) / float(w)
        fy1 = float(py1) / float(h)

        detections: list[Detection] = []
        for cls_id, boxes in enumerate(nms_classes):
            if len(boxes) == 0:
                continue
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
            if self.filter_target_classes and label not in self.enabled_classes:
                continue
            for box in boxes:
                ymin, xmin, ymax, xmax, score = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(box[4]),
                )
                if score < threshold:
                    continue
                full_xmin = fx1 + xmin * fw
                full_xmax = fx1 + xmax * fw
                full_ymin = fy1 + ymin * fh
                full_ymax = fy1 + ymax * fh
                det = Detection(
                    label=label,
                    confidence=score,
                    xmin=max(0.0, min(1.0, full_xmin)),
                    ymin=max(0.0, min(1.0, full_ymin)),
                    xmax=max(0.0, min(1.0, full_xmax)),
                    ymax=max(0.0, min(1.0, full_ymax)),
                )
                if det.area_ratio >= 0.0004:
                    detections.append(det)
        return detections

    def detect(
        self,
        bgr_frame: np.ndarray,
        roi: tuple[float, float, float, float] | None = None,
    ) -> list[Detection]:
        """Run YOLOv8 inference on a BGR frame (or cropped ROI) with motion-boosted refinement."""
        if self._infer_ctx is None or bgr_frame is None or bgr_frame.size == 0:
            return []

        t0 = time.perf_counter()
        h, w = bgr_frame.shape[:2]

        # 1. Primary pass (on ROI crop if set, else on full frame)
        rx1, ry1, rx2, ry2 = 0.0, 0.0, 1.0, 1.0
        if roi is not None:
            rx1 = max(0.0, min(1.0, float(roi[0])))
            ry1 = max(0.0, min(1.0, float(roi[1])))
            rx2 = max(0.0, min(1.0, float(roi[2])))
            ry2 = max(0.0, min(1.0, float(roi[3])))
            if (rx2 - rx1) >= 0.05 and (ry2 - ry1) >= 0.05:
                px1 = int(round(rx1 * w))
                py1 = int(round(ry1 * h))
                px2 = int(round(rx2 * w))
                py2 = int(round(ry2 * h))
            else:
                px1, py1, px2, py2 = 0, 0, w, h
        else:
            px1, py1, px2, py2 = 0, 0, w, h

        detections = self._run_yolo_on_patch(
            bgr_frame, px1, py1, px2, py2, threshold=self.score_threshold
        )

        # 2. Motion detector update & Motion-Boosted second pass
        small = cv2.resize(bgr_frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._bg_gray is None or self._bg_gray.shape != gray_blur.shape:
            self._bg_gray = gray_blur.astype(np.float32)

        cv2.accumulateWeighted(gray_blur, self._bg_gray, 0.08)
        bg_abs = cv2.convertScaleAbs(self._bg_gray)
        diff = cv2.absdiff(gray_blur, bg_abs)

        if roi is not None:
            mask = np.zeros_like(diff)
            mx1 = max(0, min(320, int(round(rx1 * 320))))
            my1 = max(0, min(180, int(round(ry1 * 180))))
            mx2 = max(0, min(320, int(round(rx2 * 320))))
            my2 = max(0, min(180, int(round(ry2 * 180))))
            mask[my1:my2, mx1:mx2] = 255
            diff = cv2.bitwise_and(diff, mask)

        has_high_conf_bird = any(
            (d.label == "bird" and d.confidence >= 0.45) for d in detections
        )
        if not has_high_conf_bird:
            _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [
                c
                for c in contours
                if 0.0005 <= (cv2.contourArea(c) / (320.0 * 180.0)) <= 0.15
            ]
            if valid_contours:
                best_contour = max(valid_contours, key=cv2.contourArea)
                cx, cy, cw, ch = cv2.boundingRect(best_contour)
                scale_x = w / 320.0
                scale_y = h / 180.0
                mcx1 = int(cx * scale_x)
                mcy1 = int(cy * scale_y)
                mcx2 = int((cx + cw) * scale_x)
                mcy2 = int((cy + ch) * scale_y)

                # Expand into high-res patch of at least 360x360
                mid_x = (mcx1 + mcx2) // 2
                mid_y = (mcy1 + mcy2) // 2
                half_size = max(180, max(mcx2 - mcx1, mcy2 - mcy1))
                b_px1 = max(px1, mid_x - half_size)
                b_py1 = max(py1, mid_y - half_size)
                b_px2 = min(px2, mid_x + half_size)
                b_py2 = min(py2, mid_y + half_size)

                boost_thresh = max(0.20, self.score_threshold * 0.65)
                boosted_dets = self._run_yolo_on_patch(
                    bgr_frame, b_px1, b_py1, b_px2, b_py2, threshold=boost_thresh
                )
                for b_det in boosted_dets:
                    if not any(_iou(b_det, p_det) > 0.45 for p_det in detections):
                        detections.append(b_det)

        if roi is not None:
            detections = filter_detections_by_roi(detections, roi)

        tracked = self.tracker.update(detections)
        if roi is not None:
            tracked = filter_detections_by_roi(tracked, roi)

        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return tracked
