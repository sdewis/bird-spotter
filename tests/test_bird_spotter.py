"""Unit and integration tests for British Bird Spotter."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bird_spotter.dashboard.server import draw_overlay
from bird_spotter.db import BirdDatabase, compute_quality_score
from bird_spotter.logger import spotter_log
from bird_spotter.solar import SolarScheduler, calculate_solar_times_noaa
from bird_spotter.vision.classifier import BirdSpeciesClassifier
from bird_spotter.vision.detector import (
    CentroidTracker,
    Detection,
    compute_crop_sharpness,
    extract_bird_crop,
    filter_detections_by_roi,
)
from bird_spotter.vision.framing import AutoFrameController, FramingState
from bird_spotter.vision.ptz import ObsbotPTZ


class DummyPTZ(ObsbotPTZ):
    """In-memory ObsbotPTZ subclass that avoids hardware subprocess calls during unit tests."""

    def set_pan_tilt(self, pan: int, tilt: int, force: bool = False) -> bool:
        if not self.system_power_on:
            return False
        self.pan = self._clamp_step(pan, self.pan_min, self.pan_max, self.pan_step)
        self.tilt = self._clamp_step(tilt, self.tilt_min, self.tilt_max, self.tilt_step)
        return True

    def set_zoom(self, zoom_level: float, force: bool = False) -> bool:
        if not self.system_power_on:
            return False
        self.zoom = round(max(self.zoom_min, min(self.zoom_max, float(zoom_level))), 2)
        return True

    def set_focus(self, autofocus: bool, focus_val: int | None = None) -> bool:
        self.autofocus = bool(autofocus)
        if focus_val is not None:
            self.focus = max(0, min(100, int(focus_val)))
        return True


class TestBirdSpotter(unittest.TestCase):
    def test_database_sightings_dex_relabel_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = BirdDatabase(tmp_path / "test.db")

            # 1. New species should always trigger auto photo save
            should_save, reason = db.should_save_auto_photo("Robin", quality_score=1.5, now_epoch=1000.0)
            self.assertTrue(should_save)
            self.assertEqual(reason, "new_species")

            snap1 = tmp_path / "bird_1.jpg"
            snap1.write_bytes(b"fake_jpg_1")
            rec1 = db.record_sighting(
                species="Robin",
                scientific_name="Erithacus rubecula",
                yolo_confidence=0.82,
                species_confidence=0.91,
                sharpness=120.0,
                quality_score=1.5,
                snapshot_filename="bird_1.jpg",
                thumb_filename="bird_1.jpg",
                zoom_level=2.2,
            )
            self.assertEqual(rec1["species"], "Robin")

            # 2. Immediate second visit with lower quality should be blocked by cooldown
            should_save2, reason2 = db.should_save_auto_photo(
                "Robin",
                quality_score=1.2,
                now_epoch=rec1["id"] + 10.0,
                per_species_cooldown_sec=9999999999.0,
            )
            self.assertFalse(should_save2)
            self.assertEqual(reason2, "cooldown_active")

            # 3. Higher quality frame beats existing best photo even during cooldown
            should_save3, reason3 = db.should_save_auto_photo(
                "Robin",
                quality_score=2.2,
                now_epoch=rec1["id"] + 10.0,
                per_species_cooldown_sec=9999999999.0,
            )
            self.assertTrue(should_save3)
            self.assertEqual(reason3, "better_quality")

            # 4. Re-label sighting to Dunnock and verify species_dex rebuilds
            ok_relabel = db.relabel_sighting(rec1["id"], "Dunnock", "Prunella modularis")
            self.assertTrue(ok_relabel)
            dex = db.get_species_dex()
            self.assertEqual(len(dex), 1)
            self.assertEqual(dex[0]["species"], "Dunnock")
            self.assertEqual(dex[0]["scientific_name"], "Prunella modularis")

            # 5. Delete sighting and verify snapshot file + species_dex are cleaned up
            ok_del = db.delete_sighting(rec1["id"], snapshots_dir=tmp_path)
            self.assertTrue(ok_del)
            self.assertFalse(snap1.exists())
            self.assertEqual(len(db.get_species_dex()), 0)

            # 6. Unidentified Bird sightings are recorded but excluded from species_dex
            rec_unid = db.record_sighting(
                species="Unidentified Bird",
                scientific_name="",
                yolo_confidence=0.75,
                species_confidence=0.18,
                sharpness=85.0,
                snapshot_filename="unid_1.jpg",
                thumb_filename="unid_1.jpg",
            )
            self.assertEqual(rec_unid["species"], "Unidentified Bird")
            self.assertEqual(len(db.get_species_dex()), 0)
            self.assertEqual(db.get_total_sightings_count(), 1)
            self.assertEqual(len(db.get_recent_sightings()), 1)

            # 7. Batch relabel multiple sightings
            snap_wt1 = tmp_path / "wt1.jpg"
            snap_wt2 = tmp_path / "wt2.jpg"
            snap_wt1.write_bytes(b"jpeg_data_wt1")
            snap_wt2.write_bytes(b"jpeg_data_wt2")
            s1 = db.record_sighting(
                species="Willow Tit",
                scientific_name="Poecile montanus",
                yolo_confidence=0.8,
                species_confidence=0.7,
                sharpness=100.0,
                snapshot_filename="wt1.jpg",
            )
            s2 = db.record_sighting(
                species="Willow Tit",
                scientific_name="Poecile montanus",
                yolo_confidence=0.85,
                species_confidence=0.75,
                sharpness=110.0,
                snapshot_filename="wt2.jpg",
            )
            self.assertEqual(len(db.get_species_dex()), 1)
            self.assertEqual(db.get_species_dex()[0]["total_sightings"], 2)

            batch_count = db.batch_relabel_sightings(
                [s1["id"], s2["id"]], "Blue Tit", "Cyanistes caeruleus"
            )
            self.assertEqual(batch_count, 2)
            dex = db.get_species_dex()
            self.assertEqual(len(dex), 1)
            self.assertEqual(dex[0]["species"], "Blue Tit")
            self.assertEqual(dex[0]["total_sightings"], 2)

            # 8. Delete species and verify all sightings + photo files removed
            del_sp_count = db.delete_species("Blue Tit", snapshots_dir=tmp_path)
            self.assertEqual(del_sp_count, 2)
            self.assertFalse(snap_wt1.exists())
            self.assertFalse(snap_wt2.exists())
            self.assertEqual(len(db.get_species_dex()), 0)

    def test_spotter_logger_levels_and_throttling(self) -> None:
        spotter_log.clear()
        spotter_log.set_level("INFO")
        self.assertEqual(spotter_log.level, "INFO")

        spotter_log.debug("TEST", "debug message 1")
        spotter_log.info("TEST", "info message 1")
        spotter_log.warn("TEST", "warning message 1")

        info_entries = spotter_log.get_entries(min_level="INFO")
        self.assertTrue(any(e["msg"] == "info message 1" for e in info_entries))
        self.assertTrue(any(e["msg"] == "warning message 1" for e in info_entries))
        self.assertFalse(any(e["msg"] == "debug message 1" for e in info_entries))

        all_entries = spotter_log.get_entries(min_level="DEBUG")
        self.assertTrue(any(e["msg"] == "debug message 1" for e in all_entries))

        # Throttling test
        e1 = spotter_log.debug_throttled("k1", "TEST", "throttled 1", interval_sec=1.0)
        e2 = spotter_log.debug_throttled("k1", "TEST", "throttled 2", interval_sec=1.0)
        self.assertIsNotNone(e1)
        self.assertIsNone(e2)

    def test_uk_species_mask_filters_non_british_birds(self) -> None:
        clf = BirdSpeciesClassifier(uk_filter_enabled=True)
        self.assertGreater(len(clf.all_labels), 700)
        self.assertGreater(len(clf.uk_index_map), 70)

        # Create synthetic logits where index 0 ('African blue tit') has highest raw logit (10.0)
        # and index 221 ('Eurasian blue tit' -> 'Blue Tit') has second highest raw logit (8.5).
        logits = np.zeros(707, dtype=np.float32)
        logits[0] = 10.0    # African blue tit (non-UK)
        logits[221] = 8.5   # Eurasian blue tit (UK -> Blue Tit)

        # With UK filter enabled, 'Blue Tit' must win!
        pred_uk = clf._decode_logits(logits, backend_used="test", infer_ms=1.0)
        self.assertIsNotNone(pred_uk)
        assert pred_uk is not None
        self.assertEqual(pred_uk.species, "Blue Tit")
        self.assertEqual(pred_uk.scientific_name, "Cyanistes caeruleus")
        self.assertGreater(pred_uk.confidence, 0.95)

        # With UK filter disabled, 'African blue tit' wins
        clf.uk_filter_enabled = False
        pred_all = clf._decode_logits(logits, backend_used="test", infer_ms=1.0)
        assert pred_all is not None
        self.assertEqual(pred_all.species, "African blue tit")

    def test_auto_frame_controller_fsm_and_burst_sharpness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ptz = DummyPTZ(preset_path=Path(tmp) / "home.json")
            framer = AutoFrameController(
                ptz=ptz,
                target_min_area=0.25,
                target_max_area=0.40,
                settle_delay_sec=0.1,
                burst_count=3,
                return_home_settle_sec=0.5,
                manual_pause_sec=60.0,
            )

            frame_blurry = np.full((720, 1280, 3), 128, dtype=np.uint8)
            frame_sharp = np.full((720, 1280, 3), 128, dtype=np.uint8)
            # Draw high-contrast checkerboard in center of frame_sharp so Laplacian variance is high
            for y in range(200, 520, 8):
                for x in range(440, 840, 8):
                    if ((x + y) // 8) % 2 == 0:
                        frame_sharp[y : y + 8, x : x + 8] = 255
                    else:
                        frame_sharp[y : y + 8, x : x + 8] = 0

            # 1. Start in WATCHING_HOME; seeing an off-center small bird transitions to FRAMING_BIRD
            det_far = Detection(
                label="bird", confidence=0.85, xmin=0.65, ymin=0.60, xmax=0.75, ymax=0.70, track_id=1
            )
            st = framer.step(frame_blurry, det_far, now=10.0)
            self.assertEqual(st, FramingState.FRAMING_BIRD)

            # 2. Bird is now centered and fills 30% of frame; after settle delay -> SETTLING_BURST
            det_framed = Detection(
                label="bird", confidence=0.90, xmin=0.25, ymin=0.20, xmax=0.75, ymax=0.80, track_id=1
            )
            st = framer.step(frame_blurry, det_framed, now=11.0)
            self.assertEqual(st, FramingState.SETTLING_BURST)

            # 3. Feed 3 burst frames (blurry, sharp, blurry) and verify sharpest frame is chosen
            captured = []
            framer.step(frame_blurry, det_framed, now=11.1)
            framer.step(frame_sharp, det_framed, now=11.2)
            st = framer.step(
                frame_blurry,
                det_framed,
                classify_cb=lambda crop: ("Robin", 0.95),
                on_capture_ready=lambda best, pred: captured.append((best, pred)),
                now=11.3,
            )
            self.assertEqual(st, FramingState.RETURNING_HOME)
            self.assertEqual(len(captured), 1)
            best_cand, _ = captured[0]
            self.assertGreater(best_cand.sharpness, 500.0)

            # 4. After return_home_settle_sec, returns to WATCHING_HOME at 1.0x zoom
            st = framer.step(frame_blurry, None, now=12.0)
            self.assertEqual(st, FramingState.WATCHING_HOME)
            self.assertEqual(ptz.zoom, 1.0)

            # 5. Manual override pauses auto-framing for 60s
            framer.trigger_manual_override(now=20.0)
            self.assertEqual(framer.state, FramingState.MANUAL_OVERRIDE)
            self.assertAlmostEqual(framer.manual_override_remaining(now=35.0), 45.0, places=1)
            framer.step(frame_sharp, det_framed, now=81.0)
            self.assertEqual(framer.state, FramingState.RETURNING_HOME)

    def test_roi_aspect_ratio_constraint_and_detection_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preset_file = Path(tmp) / "home_preset.json"
            ptz = DummyPTZ(preset_path=preset_file)

            # 1. Verify set_roi() with lock_aspect=True enforces w == h (matching camera aspect ratio)
            res = ptz.set_roi(x=0.10, y=0.15, w=0.30, h=0.70, enabled=True, lock_aspect=True)
            # scale = round(max(0.10, min(1.0, (0.30 + 0.70) * 0.5)), 4) = 0.5
            self.assertEqual(ptz.roi_w, 0.5)
            self.assertEqual(ptz.roi_h, 0.5)
            self.assertEqual(ptz.roi_x, 0.10)
            self.assertEqual(ptz.roi_y, 0.15)
            self.assertTrue(ptz.roi_enabled)
            self.assertEqual(res["w"], 0.5)
            self.assertEqual(res["h"], 0.5)

            # Verify free aspect ratio when lock_aspect=False (allows tightening closely around feeder)
            res_free = ptz.set_roi(x=0.10, y=0.15, w=0.40, h=0.25, enabled=True, lock_aspect=False)
            self.assertEqual(ptz.roi_w, 0.40)
            self.assertEqual(ptz.roi_h, 0.25)
            self.assertEqual(res_free["w"], 0.40)
            self.assertEqual(res_free["h"], 0.25)

            # Verify get_roi_tuple returns normalized (rx1, ry1, rx2, ry2) at reference pose
            roi_tuple = ptz.get_roi_tuple()
            self.assertIsNotNone(roi_tuple)
            assert roi_tuple is not None
            self.assertAlmostEqual(roi_tuple[0], 0.10)
            self.assertAlmostEqual(roi_tuple[1], 0.15)
            self.assertAlmostEqual(roi_tuple[2], 0.50)
            self.assertAlmostEqual(roi_tuple[3], 0.40)

            # Verify persistence across restart
            ptz2 = DummyPTZ(preset_path=preset_file)
            self.assertTrue(ptz2.roi_enabled)
            self.assertEqual(ptz2.roi_x, 0.10)
            self.assertEqual(ptz2.roi_y, 0.15)
            self.assertEqual(ptz2.roi_w, 0.40)
            self.assertEqual(ptz2.roi_h, 0.25)

            # Verify clear_roi() disables ROI and persists
            ptz.clear_roi()
            self.assertFalse(ptz.roi_enabled)
            self.assertIsNone(ptz.get_roi_tuple())
            ptz3 = DummyPTZ(preset_path=preset_file)
            self.assertFalse(ptz3.roi_enabled)

            # 2. Verify filter_detections_by_roi() only keeps detections inside the Zone of Interest
            d_inside = Detection(
                label="bird",
                confidence=0.92,
                xmin=0.25,
                ymin=0.25,
                xmax=0.35,
                ymax=0.35,
                track_id=10,
            )  # center = (0.30, 0.30)
            d_outside = Detection(
                label="bird",
                confidence=0.88,
                xmin=0.80,
                ymin=0.80,
                xmax=0.90,
                ymax=0.90,
                track_id=11,
            )  # center = (0.85, 0.85)

            test_roi = (0.20, 0.20, 0.50, 0.50)
            filtered = filter_detections_by_roi([d_inside, d_outside], test_roi)
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0].track_id, 10)

            # When roi is None, returns detections unchanged
            all_dets = filter_detections_by_roi([d_inside, d_outside], None)
            self.assertEqual(len(all_dets), 2)

            # 3. Verify draw_overlay() dims pixels outside ROI by 70% while preserving brightness inside ROI
            raw_frame = np.full((360, 640, 3), 200, dtype=np.uint8)
            roi_bounds = (0.20, 0.20, 0.80, 0.80)
            annotated = draw_overlay(
                frame=raw_frame,
                detections=[],
                primary=None,
                fps=30.0,
                infer_ms=8.0,
                roi=roi_bounds,
            )
            # Point inside ROI (300, 200) away from ROI borders
            inside_bgr = annotated[200, 300]
            self.assertEqual(int(inside_bgr[0]), 200)
            self.assertEqual(int(inside_bgr[1]), 200)
            self.assertEqual(int(inside_bgr[2]), 200)

            # Bottom-right corner far outside ROI and away from HUD (620, 340) should be dimmed by 70% (* 0.30 = 60)
            outside_bgr = annotated[340, 620]
            self.assertAlmostEqual(int(outside_bgr[0]), 60, delta=2)

    def test_world_locked_roi_shifting_and_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ptz = DummyPTZ(preset_path=Path(tmp) / "home_preset.json")
            ptz.set_roi(x=0.25, y=0.25, w=0.50, h=0.40, enabled=True, lock_aspect=False)
            initial_roi = ptz.get_roi_tuple()
            self.assertIsNotNone(initial_roi)
            assert initial_roi is not None
            self.assertEqual(initial_roi, (0.25, 0.25, 0.75, 0.65))

            # When camera pans to the right, world-locked ROI shifts left on screen
            ptz.pan = ptz.roi_ref_pan + 36000  # 10 degrees pan
            shifted_roi = ptz.get_roi_tuple()
            self.assertIsNotNone(shifted_roi)
            assert shifted_roi is not None
            self.assertNotEqual(shifted_roi[0], initial_roi[0])

            # When camera zooms in (e.g. 2.0x), world-locked ROI expands on screen
            ptz.pan = ptz.roi_ref_pan
            ptz.tilt = ptz.roi_ref_tilt
            ptz.zoom = 2.0
            zoomed_roi = ptz.get_roi_tuple()
            self.assertIsNotNone(zoomed_roi)
            assert zoomed_roi is not None
            zoomed_w = zoomed_roi[2] - zoomed_roi[0]
            initial_w = initial_roi[2] - initial_roi[0]
            self.assertGreater(zoomed_w, initial_w)

    def test_static_false_positive_suppression(self) -> None:
        tracker = CentroidTracker()

        # Synthetic stationary detection (e.g. twig or hanging leaf) at (0.40, 0.40)
        d_twig = Detection(label="bird", confidence=0.55, xmin=0.35, ymin=0.35, xmax=0.45, ymax=0.45)

        # 1. Initial frames at 0.1s intervals -> track gets confirmed
        tracker.update([d_twig], now=0.0)
        tracker.update([d_twig], now=0.1)
        confirmed = tracker.update([d_twig], now=0.2)
        self.assertEqual(len(confirmed), 1)

        # 2. Advance time continuously past 15.0 seconds at 0.5s intervals (keeping track alive)
        t = 0.2
        while t < 15.5:
            t += 0.5
            confirmed_later = tracker.update([d_twig], now=t)

        # Motionless for >15s -> suppressed from confirmed detections and added to static mask!
        self.assertEqual(len(confirmed_later), 0)
        self.assertEqual(len(tracker._static_mask), 1)

        # 3. New detection at same static position with low confidence (<0.65) is suppressed immediately by mask
        d_new_twig = Detection(label="bird", confidence=0.58, xmin=0.35, ymin=0.35, xmax=0.45, ymax=0.45)
        confirmed_suppressed = tracker.update([d_new_twig], now=t + 1.0)
        self.assertEqual(len(confirmed_suppressed), 0)

        # 4. A real bird arrives and moves away (>0.025 distance from static mask) -> tracked!
        d_real_bird = Detection(label="bird", confidence=0.85, xmin=0.70, ymin=0.70, xmax=0.80, ymax=0.80)
        tracker.update([d_real_bird], now=t + 1.1)
        tracker.update([d_real_bird], now=t + 1.2)
        confirmed_real = tracker.update([d_real_bird], now=t + 1.3)
        self.assertEqual(len(confirmed_real), 1)
        self.assertEqual(confirmed_real[0].label, "bird")

    def test_solar_scheduler_sleep_and_wake_windows(self) -> None:
        target_d = date(2026, 11, 12)
        dawn, sr, ss = calculate_solar_times_noaa(
            target_d, lat=52.3555, lon=-1.1743, tz_offset_hours=0.0
        )
        self.assertLess(dawn, sr)
        self.assertLess(sr, ss)

        scheduler = SolarScheduler(
            enabled=True,
            auto_location=False,
            latitude=52.3555,
            longitude=-1.1743,
            location_name="UK Test",
            wake_before_dawn_min=30,
            sleep_after_sunset_min=30,
        )
        scheduler._calculate_fallback(target_d)

        wake_dt = scheduler._wake_dt
        sleep_dt = scheduler._sleep_dt
        self.assertIsNotNone(wake_dt)
        self.assertIsNotNone(sleep_dt)
        assert wake_dt is not None and sleep_dt is not None

        # Verify 30m offset: wake is 30m before dawn, sleep is 30m after sunset
        self.assertEqual(wake_dt, scheduler._dawn_dt - timedelta(minutes=30))
        self.assertEqual(sleep_dt, scheduler._sunset_dt + timedelta(minutes=30))

        # Deep night (e.g. 02:00) should be night sleep
        t_night = datetime(
            target_d.year, target_d.month, target_d.day, 2, 0, tzinfo=wake_dt.tzinfo
        )
        self.assertTrue(scheduler.is_night_sleep_window(t_night))

        # 10 mins before wake_dt -> night sleep
        t_pre_wake = wake_dt - timedelta(minutes=10)
        self.assertTrue(scheduler.is_night_sleep_window(t_pre_wake))

        # 10 mins after wake_dt (20 mins before dawn) -> awake!
        t_post_wake = wake_dt + timedelta(minutes=10)
        self.assertFalse(scheduler.is_night_sleep_window(t_post_wake))

        # Midday (12:00) -> awake
        t_noon = datetime(
            target_d.year, target_d.month, target_d.day, 12, 0, tzinfo=wake_dt.tzinfo
        )
        self.assertFalse(scheduler.is_night_sleep_window(t_noon))

        # 10 mins before sleep_dt -> still awake
        t_pre_sleep = sleep_dt - timedelta(minutes=10)
        self.assertFalse(scheduler.is_night_sleep_window(t_pre_sleep))

        # 10 mins after sleep_dt -> night sleep
        t_post_sleep = sleep_dt + timedelta(minutes=10)
        self.assertTrue(scheduler.is_night_sleep_window(t_post_sleep))

        # Test transitions
        scheduler.force_recheck()
        trans_night = scheduler.check_transition(t_night)
        self.assertEqual(trans_night, "sleep")
        self.assertIsNone(scheduler.check_transition(t_night))

        trans_day = scheduler.check_transition(t_noon)
        self.assertEqual(trans_day, "wake")
        self.assertIsNone(scheduler.check_transition(t_noon))

        trans_evening = scheduler.check_transition(t_post_sleep)
        self.assertEqual(trans_evening, "sleep")

        # When disabled, check_transition returns None
        scheduler.enabled = False
        self.assertIsNone(scheduler.check_transition(t_noon))

        # Status dictionary check
        status = scheduler.get_status_dict(t_noon)
        self.assertFalse(status["enabled"])
        self.assertIn("wake_time", status)
        self.assertIn("sleep_time", status)
        self.assertIn("dawn", status)
        self.assertIn("sunset", status)
        self.assertEqual(status["location"], "UK Test")

    def test_dashboard_server_endpoints_and_night_mode(self) -> None:
        """Test DashboardServer popout, state, batch relabel, dex delete, and night mode handling."""
        import json
        import urllib.request
        from bird_spotter.dashboard.server import DashboardServer

        batch_called = []
        delete_species_called = []

        def mock_batch_relabel(ids: list[int], sp: str) -> int:
            batch_called.append((ids, sp))
            return len(ids)

        def mock_delete_species(sp: str) -> int:
            delete_species_called.append(sp)
            return 3

        server = DashboardServer(
            host="127.0.0.1",
            port=18088,
            on_batch_relabel=mock_batch_relabel,
            on_delete_species=mock_delete_species,
            on_get_species_sightings=lambda sp: [{"id": 99, "species": sp}],
        )
        server.start_background()
        import time
        time.sleep(0.1)

        try:
            # 1. Test /popout page
            with urllib.request.urlopen("http://127.0.0.1:18088/popout", timeout=2.0) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
                self.assertIn("Bird Spotter — Live Pop-out Feed", html)
                self.assertIn("Night mode", html)

            # 2. Test /api/state initial and update_state_only
            server.update_state_only({"is_night_mode": True, "fps": 0.0, "system_power_on": False})
            with urllib.request.urlopen("http://127.0.0.1:18088/api/state", timeout=2.0) as resp:
                self.assertEqual(resp.status, 200)
                st = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(st["is_night_mode"])
                self.assertFalse(st["system_power_on"])

            # 3. Test POST /api/sightings/batch_relabel
            req_batch = urllib.request.Request(
                "http://127.0.0.1:18088/api/sightings/batch_relabel",
                data=json.dumps({"ids": [101, 102, 103], "species": "Blue Tit"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_batch, timeout=2.0) as resp:
                self.assertEqual(resp.status, 200)
                res = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(res["success"])
                self.assertEqual(res["count"], 3)
            self.assertEqual(len(batch_called), 1)
            self.assertEqual(batch_called[0], ([101, 102, 103], "Blue Tit"))

            # 4. Test POST /api/dex/delete
            req_del = urllib.request.Request(
                "http://127.0.0.1:18088/api/dex/delete",
                data=json.dumps({"species": "Magpie"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_del, timeout=2.0) as resp:
                self.assertEqual(resp.status, 200)
                res = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(res["success"])
                self.assertEqual(res["count"], 3)
            self.assertEqual(len(delete_species_called), 1)
            self.assertEqual(delete_species_called[0], "Magpie")

            # 5. Test GET /api/dex/sightings
            with urllib.request.urlopen("http://127.0.0.1:18088/api/dex/sightings?species=Robin", timeout=2.0) as resp:
                self.assertEqual(resp.status, 200)
                res = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(res["species"], "Robin")
                self.assertEqual(len(res["sightings"]), 1)
                self.assertEqual(res["sightings"][0]["id"], 99)

        finally:
            server.stop()




if __name__ == "__main__":
    unittest.main()
