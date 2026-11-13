"""Fine-grained British bird species classifier (Hailo-8 .hef + ONNX Runtime/OpenCV DNN)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any

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

from bird_spotter.config import PROJECT_ROOT

DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "eu_common_labels.json"
DEFAULT_UK_BIRDS_PATH = PROJECT_ROOT / "config" / "uk_birds.json"
DEFAULT_HEF_PATH = PROJECT_ROOT / "models" / "uk_birds_regnet_y_1_6g_h8.hef"
DEFAULT_ONNX_PATH = PROJECT_ROOT / "models" / "uk_birds_regnet_y_1_6g.onnx"


@dataclass
class SpeciesPrediction:
    """Result of fine-grained bird species classification on a cropped bird image."""

    species: str
    scientific_name: str
    confidence: float
    top_predictions: list[dict[str, Any]]
    infer_ms: float = 0.0
    backend: str = "none"


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    return exps / denom if denom > 0 else np.zeros_like(x)


class BirdSpeciesClassifier:
    """Classifies cropped bird detections into British bird species using Hailo-8 (.hef) or ONNX."""

    def __init__(
        self,
        hef_path: str | Path | None = None,
        onnx_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        uk_birds_path: str | Path | None = None,
        uk_filter_enabled: bool = True,
        min_confidence: float = 0.25,
    ) -> None:
        self.hef_path = Path(hef_path) if hef_path else DEFAULT_HEF_PATH
        self.onnx_path = Path(onnx_path) if onnx_path else DEFAULT_ONNX_PATH
        self.labels_path = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
        self.uk_birds_path = Path(uk_birds_path) if uk_birds_path else DEFAULT_UK_BIRDS_PATH
        self.uk_filter_enabled = uk_filter_enabled
        self.min_confidence = min_confidence

        self._lock = threading.RLock()
        self.input_w = 256
        self.input_h = 256
        self.mean = np.array([0.5248, 0.5372, 0.5086], dtype=np.float32)
        self.std = np.array([0.2135, 0.2103, 0.2622], dtype=np.float32)

        self.all_labels: list[str] = []
        self.uk_index_map: dict[int, dict[str, str]] = {}
        self.uk_name_to_sci: dict[str, str] = {}
        self._uk_mask_indices: np.ndarray = np.array([], dtype=np.int64)

        self._load_metadata()

        # Hailo-8 state (configured on shared VDevice or standalone)
        self._hef: Any = None
        self._net_group: Any = None
        self._net_group_params: Any = None
        self._in_params: Any = None
        self._out_params: Any = None
        self._input_name: str = ""
        self._output_name: str = ""

        # ONNX / OpenCV DNN state
        self._cv_net: Any = None
        self._ort_session: Any = None
        self._ort_input_name: str = ""
        self.active_backend: str = "uninitialized"
        self.last_infer_ms: float = 0.0

    def _load_metadata(self) -> None:
        if self.labels_path.is_file():
            with self.labels_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            self.all_labels = list(meta.get("labels", []))
            size = meta.get("input_size", [256, 256])
            self.input_h, self.input_w = int(size[0]), int(size[1])
            stats = meta.get("rgb_stats", {})
            if "mean" in stats:
                self.mean = np.array(stats["mean"], dtype=np.float32)
            if "std" in stats:
                self.std = np.array(stats["std"], dtype=np.float32)

        if self.uk_birds_path.is_file():
            with self.uk_birds_path.open("r", encoding="utf-8") as f:
                uk_data = json.load(f)
            indices: list[int] = []
            for entry in uk_data.get("species", []):
                idx = int(entry["index"])
                uk_name = entry["uk_name"]
                sci_name = entry.get("scientific_name", "")
                self.uk_index_map[idx] = {
                    "uk_name": uk_name,
                    "scientific_name": sci_name,
                    "model_label": entry.get("model_label", uk_name),
                    "category": entry.get("category", "garden"),
                }
                self.uk_name_to_sci[uk_name] = sci_name
                indices.append(idx)
            self._uk_mask_indices = np.array(sorted(set(indices)), dtype=np.int64)

    def get_uk_species_catalog(self) -> list[dict[str, str]]:
        """Return sorted list of UK bird species for dashboard dropdowns and Life List."""
        items = [
            {
                "species": info["uk_name"],
                "common_name": info["uk_name"],
                "name": info["uk_name"],
                "scientific_name": info["scientific_name"],
                "category": info["category"],
            }
            for info in self.uk_index_map.values()
        ]
        return sorted(items, key=lambda x: x["species"])

    def configure_hailo(self, vdevice: Any) -> bool:
        """Configure the Hailo-8 .hef classifier on a shared Hailo VDevice."""
        if not HAILO_AVAILABLE or vdevice is None or not self.hef_path.is_file():
            return False
        try:
            self._hef = HEF(str(self.hef_path))
            in_info = self._hef.get_input_vstream_infos()[0]
            out_info = self._hef.get_output_vstream_infos()[0]
            self._input_name = in_info.name
            self._output_name = out_info.name
            self.input_h, self.input_w = int(in_info.shape[0]), int(in_info.shape[1])

            cfg_params = ConfigureParams.create_from_hef(
                hef=self._hef, interface=HailoStreamInterface.PCIe
            )
            net_groups = vdevice.configure(self._hef, cfg_params)
            self._net_group = net_groups[0]
            self._net_group_params = self._net_group.create_params()
            self._in_params = InputVStreamParams.make(
                self._net_group, format_type=FormatType.UINT8
            )
            self._out_params = OutputVStreamParams.make(
                self._net_group, format_type=FormatType.FLOAT32
            )
            self.active_backend = "hailo8"
            return True
        except Exception as exc:
            print(f"[Classifier] Warning: Could not configure Hailo HEF ({exc}); falling back to ONNX.")
            self._net_group = None
            return False

    def open_onnx_fallback(self) -> bool:
        """Load ONNX classifier model via ONNX Runtime or OpenCV DNN if available."""
        if not self.onnx_path.is_file():
            return False
        try:
            import onnxruntime as ort  # type: ignore

            self._ort_session = ort.InferenceSession(
                str(self.onnx_path), providers=["CPUExecutionProvider"]
            )
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            if self.active_backend == "uninitialized":
                self.active_backend = "onnxruntime"
            return True
        except ImportError:
            pass
        except Exception as exc:
            print(f"[Classifier] ONNX Runtime load warning: {exc}")

        try:
            self._cv_net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
            if self.active_backend == "uninitialized":
                self.active_backend = "opencv_dnn"
            return True
        except Exception as exc:
            print(f"[Classifier] OpenCV DNN load warning: {exc}")
            return False

    def is_ready(self) -> bool:
        return (
            self._net_group is not None
            or self._ort_session is not None
            or self._cv_net is not None
        )

    def _preprocess_uint8(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Resize BGR crop to (1, H, W, 3) RGB UINT8 for Hailo-8 HEF with baked normalization."""
        resized = cv2.resize(
            bgr_crop, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb, axis=0).astype(np.uint8)

    def _preprocess_float32_nchw(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Resize and normalize BGR crop to (1, 3, H, W) FLOAT32 for ONNX."""
        resized = cv2.resize(
            bgr_crop, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self.mean) / self.std
        chw = np.transpose(norm, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def _infer_hailo_logits(self, bgr_crop: np.ndarray) -> np.ndarray | None:
        if self._net_group is None:
            return None
        input_tensor = self._preprocess_uint8(bgr_crop)
        with InferVStreams(self._net_group, self._in_params, self._out_params) as infer_ctx:
            with self._net_group.activate(self._net_group_params):
                raw = infer_ctx.infer({self._input_name: input_tensor})
                logits = np.asarray(raw[self._output_name]).reshape(-1)
                return logits

    def _infer_onnx_logits(self, bgr_crop: np.ndarray) -> np.ndarray | None:
        nchw = self._preprocess_float32_nchw(bgr_crop)
        if self._ort_session is not None:
            res = self._ort_session.run(None, {self._ort_input_name: nchw})
            return np.asarray(res[0]).reshape(-1)
        if self._cv_net is not None:
            self._cv_net.setInput(nchw)
            out = self._cv_net.forward()
            return np.asarray(out).reshape(-1)
        return None

    def _decode_logits(
        self, raw_scores: np.ndarray, backend_used: str, infer_ms: float
    ) -> SpeciesPrediction | None:
        if raw_scores.size == 0:
            return None

        # Determine if raw_scores are already probabilities (sum ~ 1.0, all >= 0) or logits
        is_prob = bool(
            np.all(raw_scores >= -1e-4)
            and np.all(raw_scores <= 1.0001)
            and 0.90 <= float(np.sum(raw_scores)) <= 1.10
        )

        if self.uk_filter_enabled and self._uk_mask_indices.size > 0:
            valid_indices = self._uk_mask_indices[self._uk_mask_indices < raw_scores.size]
            if is_prob:
                masked_probs = np.zeros_like(raw_scores, dtype=np.float64)
                masked_probs[valid_indices] = raw_scores[valid_indices]
                total_p = float(np.sum(masked_probs))
                if total_p > 1e-9:
                    probs = masked_probs / total_p
                else:
                    probs = _softmax(raw_scores)
            else:
                masked_logits = np.full_like(raw_scores, -1e9, dtype=np.float64)
                masked_logits[valid_indices] = raw_scores[valid_indices]
                probs = _softmax(masked_logits)
        else:
            probs = raw_scores.astype(np.float64) if is_prob else _softmax(raw_scores.astype(np.float64))

        top_k = min(3, probs.size)
        top_indices = np.argsort(probs)[::-1][:top_k]

        top_preds: list[dict[str, Any]] = []
        for idx in top_indices:
            idx_int = int(idx)
            conf = float(probs[idx_int])
            if idx_int in self.uk_index_map:
                info = self.uk_index_map[idx_int]
                sp_name = info["uk_name"]
                sci_name = info["scientific_name"]
            elif idx_int < len(self.all_labels):
                sp_name = self.all_labels[idx_int]
                sci_name = ""
            else:
                sp_name = f"Species #{idx_int}"
                sci_name = ""

            top_preds.append(
                {
                    "species": sp_name,
                    "scientific_name": sci_name,
                    "confidence": round(conf, 4),
                    "class_index": idx_int,
                }
            )

        if not top_preds:
            return None

        best = top_preds[0]
        return SpeciesPrediction(
            species=best["species"],
            scientific_name=best["scientific_name"],
            confidence=best["confidence"],
            top_predictions=top_preds,
            infer_ms=round(infer_ms, 2),
            backend=backend_used,
        )

    def classify(
        self,
        bgr_crop: np.ndarray,
        hailo_context_Pause_cb: Any | None = None,
    ) -> SpeciesPrediction | None:
        """Classify a BGR bird crop into a British bird species.

        If hailo_context_Pause_cb is provided (a context manager that temporarily releases
        the YOLOv8 Hailo network group activation for ~5ms), uses the Hailo-8 .hef model;
        otherwise falls back to ONNX / OpenCV DNN if available.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return None

        with self._lock:
            t0 = time.perf_counter()
            scores: np.ndarray | None = None
            backend_used = "none"

            if self._net_group is not None:
                try:
                    if hailo_context_Pause_cb is not None:
                        with hailo_context_Pause_cb():
                            scores = self._infer_hailo_logits(bgr_crop)
                    else:
                        scores = self._infer_hailo_logits(bgr_crop)
                    backend_used = "hailo8"
                except Exception as exc:
                    print(f"[Classifier] Hailo inference fallback ({exc})")
                    scores = None

            if scores is None and (self._ort_session is not None or self._cv_net is not None):
                try:
                    scores = self._infer_onnx_logits(bgr_crop)
                    backend_used = (
                        "onnxruntime" if self._ort_session is not None else "opencv_dnn"
                    )
                except Exception as exc:
                    print(f"[Classifier] ONNX inference error: {exc}")
                    scores = None

            infer_ms = (time.perf_counter() - t0) * 1000.0
            self.last_infer_ms = infer_ms
            if scores is None:
                return None

            return self._decode_logits(scores, backend_used, infer_ms)
