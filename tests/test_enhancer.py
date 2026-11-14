"""Unit tests for AI Super-Resolution and Burst Stacking Enhancer."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from bird_spotter.vision.detector import Detection
from bird_spotter.vision.enhancer import SuperResolutionEnhancer, unsharp_mask
from bird_spotter.vision.framing import BurstCandidate, save_clean_snapshot_and_thumb


class TestEnhancer(unittest.TestCase):
    def setUp(self) -> None:
        self.enhancer = SuperResolutionEnhancer(model_name="fsrcnn", scale=4)

    def test_unsharp_mask_dimensions_and_type(self) -> None:
        img = np.full((120, 120, 3), 128, dtype=np.uint8)
        cv2.circle(img, (60, 60), 30, (200, 50, 50), -1)
        sharp = unsharp_mask(img, sigma=1.0, strength=0.5)
        self.assertEqual(sharp.shape, img.shape)
        self.assertEqual(sharp.dtype, np.uint8)

    def test_enhance_crop_small_input(self) -> None:
        small = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(small, (15, 15), (48, 48), (255, 200, 100), -1)
        enhanced = self.enhancer.enhance_crop(small, target_size=(320, 320))
        self.assertEqual(enhanced.shape, (320, 320, 3))
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_enhance_crop_large_input(self) -> None:
        large = np.zeros((480, 480, 3), dtype=np.uint8)
        cv2.circle(large, (240, 240), 100, (100, 255, 100), -1)
        enhanced = self.enhancer.enhance_crop(large, target_size=(320, 320))
        self.assertEqual(enhanced.shape, (320, 320, 3))
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_enhance_crop_empty_input(self) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        res = self.enhancer.enhance_crop(empty, target_size=(320, 320))
        self.assertEqual(res.shape, (320, 320, 3))

    def test_align_and_fuse_burst_single_candidate(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = np.zeros((40, 40, 3), dtype=np.uint8)
        det = Detection(label="bird", confidence=0.9, xmin=0.3, ymin=0.3, xmax=0.7, ymax=0.7)
        c0 = BurstCandidate(bgr_frame=frame, bgr_crop=crop, detection=det, sharpness=100.0, zoom=1.0)
        res = self.enhancer.align_and_fuse_burst([c0])
        self.assertIsNotNone(res)
        f_frame, f_crop = res
        self.assertEqual(f_crop.shape, crop.shape)

    def test_align_and_fuse_burst_multi_candidate_noise_reduction(self) -> None:
        ref_crop = np.full((80, 80, 3), 100, dtype=np.uint8)
        cv2.circle(ref_crop, (40, 40), 18, (220, 180, 120), -1)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        det = Detection(label="bird", confidence=0.88, xmin=0.2, ymin=0.2, xmax=0.6, ymax=0.6)

        candidates = [
            BurstCandidate(bgr_frame=frame, bgr_crop=ref_crop.copy(), detection=det, sharpness=250.0, zoom=1.0)
        ]

        # Add shifted noisy copies
        for shift_x, shift_y in [(1.0, -1.0), (-1.0, 1.0), (0.5, 0.5)]:
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted = cv2.warpAffine(ref_crop, M, (80, 80), borderMode=cv2.BORDER_REFLECT)
            noise = np.random.normal(0, 15, shifted.shape).astype(np.int16)
            noisy = np.clip(shifted.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            candidates.append(
                BurstCandidate(bgr_frame=frame, bgr_crop=noisy, detection=det, sharpness=200.0, zoom=1.0)
            )

        res = self.enhancer.align_and_fuse_burst(candidates)
        self.assertIsNotNone(res)
        f_frame, f_crop = res
        self.assertEqual(f_crop.shape, (80, 80, 3))
        self.assertEqual(f_crop.dtype, np.uint8)

    def test_save_clean_snapshot_and_thumb_with_enhancer(self) -> None:
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        crop = np.full((70, 70, 3), 160, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp_dir:
            snap_path, thumb_path = save_clean_snapshot_and_thumb(
                bgr_frame=frame,
                bgr_crop=crop,
                species_name="Blue Tit",
                snapshots_dir=Path(tmp_dir),
                enhancer=self.enhancer,
            )
            self.assertIsNotNone(snap_path)
            self.assertIsNotNone(thumb_path)
            full_snap = Path(tmp_dir) / snap_path
            full_thumb = Path(tmp_dir) / thumb_path
            self.assertTrue(full_snap.is_file())
            self.assertTrue(full_thumb.is_file())
            thumb_img = cv2.imread(str(full_thumb))
            self.assertEqual(thumb_img.shape, (320, 320, 3))


if __name__ == "__main__":
    unittest.main()
