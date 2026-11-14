"""AI Super-Resolution, Burst Fusion Stacking, and Photographic Detail Enhancer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from bird_spotter.config import PROJECT_ROOT
from bird_spotter.logger import spotter_log

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def unsharp_mask(
    bgr_img: np.ndarray,
    sigma: float = 1.0,
    strength: float = 0.4,
    threshold: int = 2,
) -> np.ndarray:
    """Apply subtle, noise-gated unsharp masking to enhance plumage texture and contours."""
    if bgr_img is None or bgr_img.size == 0:
        return bgr_img
    blurred = cv2.GaussianBlur(bgr_img, (0, 0), sigma)
    diff = cv2.absdiff(bgr_img, blurred)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) >= threshold
    sharpened = cv2.addWeighted(bgr_img, 1.0 + strength, blurred, -strength, 0)
    out = bgr_img.copy()
    out[mask] = sharpened[mask]
    return out


@dataclass
class SuperResolutionEnhancer:
    """Applies deep-learning super-resolution (FSRCNN / ESPCN) and burst frame fusion.

    When cv2.dnn_superres is available (e.g. on Raspberry Pi 5 with OpenCV 4.10+), uses
    trained sub-pixel CNN weights to reconstruct clean, non-pixelated edges and textures.
    Gracefully falls back to high-order Lanczos-4 interpolation + unsharp masking
    when neural weights or the native C++ superres module are unavailable.
    """

    model_name: str = "fsrcnn"
    scale: int = 4
    models_dir: Path = field(default_factory=lambda: DEFAULT_MODELS_DIR)
    enable_ai: bool = True

    _sr: Any = field(default=None, init=False, repr=False)
    _active_model: str = field(default="none", init=False)
    _active_scale: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.enable_ai:
            self._init_dnn_superres()

    def _init_dnn_superres(self) -> bool:
        """Initialize OpenCV dnn_superres module with FSRCNN or ESPCN model weights."""
        if not hasattr(cv2, "dnn_superres"):
            spotter_log.info(
                "ENHANCER",
                "cv2.dnn_superres not compiled in this OpenCV build; using Lanczos-4 high-grade upsampler",
            )
            return False

        # Match model file pattern: e.g. fsrcnn_x4.pb or espcn_x4.pb
        m_name = self.model_name.lower().strip()
        sc = self.scale if self.scale in (2, 3, 4) else 4
        pb_candidates = [
            self.models_dir / f"{m_name}_x{sc}.pb",
            self.models_dir / f"fsrcnn_x{sc}.pb",
            self.models_dir / f"espcn_x{sc}.pb",
            self.models_dir / f"fsrcnn_x2.pb",
            self.models_dir / f"espcn_x2.pb",
        ]

        found_path: Path | None = None
        for p in pb_candidates:
            if p.is_file():
                found_path = p
                break

        if not found_path:
            spotter_log.debug(
                "ENHANCER",
                f"No super-resolution weights (.pb) found in {self.models_dir}; using Lanczos-4 fallback",
            )
            return False

        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(found_path))
            # Model name in OpenCV dnn_superres must be one of: 'edsr', 'espcn', 'fsrcnn', 'lapsrn'
            arch = "fsrcnn" if "fsrcnn" in found_path.name.lower() else "espcn"
            scale_val = sc if f"_x{sc}" in found_path.name.lower() else (2 if "_x2" in found_path.name.lower() else 4)
            sr.setModel(arch, scale_val)
            self._sr = sr
            self._active_model = arch
            self._active_scale = scale_val
            spotter_log.info(
                "ENHANCER",
                f"AI Super-Resolution active: {arch.upper()} x{scale_val} ({found_path.name})",
            )
            return True
        except Exception as exc:
            spotter_log.warn("ENHANCER", f"Failed to initialize dnn_superres with {found_path.name}: {exc}")
            self._sr = None
            return False

    @property
    def is_ai_active(self) -> bool:
        """True if deep learning super-resolution is loaded and operational."""
        return self._sr is not None

    def upsample(self, bgr_img: np.ndarray) -> np.ndarray:
        """Upscale image using AI super-resolution (or Lanczos-4 fallback)."""
        if bgr_img is None or bgr_img.size == 0:
            return bgr_img

        if self._sr is not None:
            try:
                return self._sr.upsample(bgr_img)
            except Exception as exc:
                spotter_log.debug("ENHANCER", f"dnn_superres upsample failed: {exc}; using Lanczos-4")

        # High-order Lanczos-4 fallback
        h, w = bgr_img.shape[:2]
        target_w = w * self.scale
        target_h = h * self.scale
        return cv2.resize(bgr_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    def enhance_crop(
        self,
        bgr_crop: np.ndarray,
        target_size: tuple[int, int] = (320, 320),
    ) -> np.ndarray:
        """Produce an anti-aliased, un-pixelated, super-resolved square portrait crop.

        Takes a bird bounding box crop (which may only be 60-150px) and reconstructs
        a clean 320x320 portrait using subpixel neural upsampling and subtle edge sharpening.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        h, w = bgr_crop.shape[:2]
        req_w, req_h = target_size

        # If crop is already equal or larger than target, downsample cleanly with INTER_AREA
        if w >= req_w and h >= req_h:
            resized = cv2.resize(bgr_crop, target_size, interpolation=cv2.INTER_AREA)
            return unsharp_mask(resized, sigma=1.0, strength=0.25)

        # 1. Apply AI Super-Resolution (or Lanczos-4 fallback)
        upscaled = self.upsample(bgr_crop)

        # 2. Resize cleanly to exact target_size if upscaled dimensions differ
        uh, uw = upscaled.shape[:2]
        if (uw, uh) != target_size:
            interp = cv2.INTER_AREA if (uw >= req_w and uh >= req_h) else cv2.INTER_LANCZOS4
            upscaled = cv2.resize(upscaled, target_size, interpolation=interp)

        # 3. Apply subtle noise-gated sharpening for crisp feather definition
        return unsharp_mask(upscaled, sigma=1.0, strength=0.35)

    def align_and_fuse_burst(
        self,
        candidates: list[Any],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Align burst candidate crops via sub-pixel phase correlation and fuse to suppress sensor noise.

        Micro-vibrations of the gimbal and small bird movements provide sub-pixel optical diversity.
        Stacking aligned burst crops boosts SNR by up to ~2.2x (sqrt(5)) and eliminates random sensor
        noise and compression artifacts before final AI enhancement.

        Returns (best_bgr_frame, fused_bgr_crop).
        """
        if not candidates:
            return None

        # Sort candidates so the sharpest candidate is the reference
        valid = [c for c in candidates if c.bgr_crop is not None and c.bgr_crop.size > 0]
        if not valid:
            c0 = candidates[0]
            return c0.bgr_frame, c0.bgr_crop

        valid.sort(key=lambda c: getattr(c, "sharpness", 0.0), reverse=True)
        ref = valid[0]
        ref_crop = ref.bgr_crop
        ref_frame = ref.bgr_frame
        ref_h, ref_w = ref_crop.shape[:2]

        if len(valid) == 1:
            return ref_frame, ref_crop

        ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
        ref_f32 = np.float32(ref_gray)

        aligned_crops: list[np.ndarray] = [ref_crop.astype(np.float32)]
        weights: list[float] = [1.0]

        for cand in valid[1:]:
            c_crop = cand.bgr_crop
            ch, cw = c_crop.shape[:2]
            # Skip if aspect or scale is wildly different (>25%)
            if abs(cw - ref_w) > (ref_w * 0.25) or abs(ch - ref_h) > (ref_h * 0.25):
                continue

            # Standardize size to reference for correlation
            if (cw, ch) != (ref_w, ref_h):
                c_crop = cv2.resize(c_crop, (ref_w, ref_h), interpolation=cv2.INTER_LANCZOS4)

            c_gray = cv2.cvtColor(c_crop, cv2.COLOR_BGR2GRAY)
            try:
                shift, response = cv2.phaseCorrelate(ref_f32, np.float32(c_gray))
                dx, dy = float(shift[0]), float(shift[1])
                # Only fuse if correlation is high and shift is small (< 18% of crop dimensions)
                if response >= 0.45 and abs(dx) < (ref_w * 0.18) and abs(dy) < (ref_h * 0.18):
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    warped = cv2.warpAffine(
                        c_crop, M, (ref_w, ref_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
                    )
                    weight = float(response) * max(0.2, min(1.0, cand.sharpness / (ref.sharpness + 1e-6)))
                    aligned_crops.append(warped.astype(np.float32))
                    weights.append(weight)
            except Exception:
                continue

        if len(aligned_crops) <= 1:
            return ref_frame, ref_crop

        # Weighted blend across aligned candidates
        total_w = sum(weights)
        fused = np.zeros_like(aligned_crops[0])
        for crop, w_val in zip(aligned_crops, weights):
            fused += crop * (w_val / total_w)

        fused_bgr = np.clip(fused, 0, 255).astype(np.uint8)
        spotter_log.debug(
            "ENHANCER",
            f"Burst fusion: stacked {len(aligned_crops)}/{len(valid)} frames "
            f"(SNR gain ~{len(aligned_crops)**0.5:.2f}x) around Track #{ref.detection.track_id if ref.detection else '?'}",
        )
        return ref_frame, fused_bgr
