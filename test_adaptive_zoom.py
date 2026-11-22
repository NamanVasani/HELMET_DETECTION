import unittest
from unittest.mock import patch

import numpy as np

from app import (adaptive_zoom, belongs_to_focus, event_track_keys, focus_target, should_save_detection,
                face_targets, read_detections, should_save_event, target_key,
                visible_detections, zoom_region)


class AdaptiveZoomTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)

    def test_stays_wide_without_a_zoom_target(self):
        self.assertIs(adaptive_zoom(self.frame, []), self.frame)

    def test_can_zoom_on_a_rider_partly_at_frame_edge(self):
        partial = {"class_name": "rider", "box": [0, 100, 40, 160]}
        preview = adaptive_zoom(self.frame, [partial])
        self.assertFalse(np.array_equal(preview, self.frame))

    def test_zooms_for_distant_rider_without_changing_frame_size(self):
        distant = {"class_name": "rider", "box": [280, 120, 320, 220]}
        preview = adaptive_zoom(self.frame, [distant])
        self.assertEqual(preview.shape, self.frame.shape)
        self.assertFalse(np.array_equal(preview, self.frame))

    def test_does_not_zoom_on_a_near_rider(self):
        near = {"class_name": "rider", "box": [200, 20, 440, 450]}
        self.assertIs(adaptive_zoom(self.frame, [near]), self.frame)

    def test_zoom_region_returns_crop_mapping_for_close_prediction(self):
        distant = {"class_name": "rider", "box": [280, 120, 320, 220]}
        crop, mapping = zoom_region(self.frame, [distant])
        self.assertEqual(crop.shape, self.frame.shape)
        self.assertIsNotNone(mapping)
        left, top, zoom = mapping
        self.assertGreater(zoom, 1)
        self.assertLessEqual(left, 280)
        self.assertLessEqual(top, 120)

    def test_focus_stays_on_leftmost_rider_until_a_picture_is_saved(self):
        first = {"class_name": "rider", "track_id": 10, "box": [100, 80, 160, 220]}
        second = {"class_name": "rider", "track_id": 11, "box": [380, 80, 440, 220]}
        self.assertEqual(focus_target([first, second])["track_id"], 10)
        self.assertEqual(focus_target([first, second])["track_id"], 10)
        self.assertEqual(focus_target([first, second], [first])["track_id"], 11)

    def test_focus_scan_limits_candidates_to_three(self):
        riders = [{"class_name": "rider", "track_id": i, "box": [i * 50, 50, i * 50 + 30, 100]}
                  for i in range(5)]
        from app import zoom_targets
        self.assertEqual(len(zoom_targets(riders)), 3)

    def test_focus_reaches_people_after_the_first_three_are_captured(self):
        riders = [{"class_name": "rider", "track_id": i, "box": [i * 60, 50, i * 60 + 30, 100]}
                  for i in range(5)]
        self.assertIs(focus_target(riders, riders[:3]), riders[3])

    def test_overlapping_different_rider_tracks_are_not_the_same_person(self):
        from app import same_focus
        first = {"class_name": "rider", "track_id": 1, "box": [100, 40, 200, 260]}
        other = dict(first, track_id=2)
        self.assertFalse(same_focus(first, other))

    def test_face_target_keeps_focus_even_without_a_bike_or_rider_detection(self):
        face = {"class_name": "face", "track_id": None, "box": [200, 100, 280, 190]}
        rider = {"class_name": "rider", "track_id": 2, "box": [100, 80, 300, 440]}
        self.assertIs(focus_target([face]), face)
        self.assertIs(focus_target([face, rider]), face)

    def test_captured_face_is_skipped_and_next_person_gets_focus(self):
        first = {"class_name": "face", "box": [100, 80, 180, 170]}
        jittered = {"class_name": "face", "box": [110, 85, 190, 175]}
        second = {"class_name": "face", "box": [360, 80, 440, 170]}
        from app import same_focus
        self.assertTrue(same_focus(first, jittered))
        self.assertIs(focus_target([jittered, second], [first]), second)
        self.assertIsNone(focus_target([jittered], [first]))

    def test_face_detector_returns_whole_face_boxes(self):
        class Detector:
            def setInputSize(self, size):
                self.size = size

            def detect(self, frame):
                return 1, np.array([[12, 24, 80, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.93]])

        from app import detect_faces
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        face = detect_faces(frame, Detector())[0]
        self.assertEqual(face["box"], [12, 24, 92, 114])
        self.assertEqual(face["confidence"], 0.93)

    def test_face_focus_survives_brief_detector_misses(self):
        face = {"class_name": "face", "box": [20, 30, 60, 70]}
        current, previous, missed = face_targets([face], [], 0)
        self.assertEqual(current, [face])
        current, previous, missed = face_targets([], previous, missed)
        self.assertEqual(current, [face])
        for _ in range(3):
            current, previous, missed = face_targets([], previous, missed)
        self.assertEqual(face_targets([], previous, missed)[0], [])

    def test_event_dedup_links_head_box_to_rider_track(self):
        no_helmet = {"class_name": "no-helmet", "track_id": 12, "box": [118, 90, 140, 114]}
        rider = {"class_name": "rider", "track_id": 7, "box": [100, 80, 160, 220]}
        keys = event_track_keys(no_helmet, {"rider": rider})
        self.assertIn(target_key(no_helmet), keys)
        self.assertIn(target_key(rider), keys)

    def test_zoom_confidence_is_lower_than_full_frame(self):
        self.assertTrue(should_save_detection({"confidence": 0.50, "zoomed": True}))
        self.assertFalse(should_save_detection({"confidence": 0.79, "zoomed": False}))
        self.assertTrue(should_save_detection({"confidence": 0.80, "zoomed": False}))

    def test_reads_each_box_tracking_id_independently(self):
        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Coordinates:
            def cpu(self):
                return self

            def tolist(self):
                return [10, 20, 30, 40]

        class Box:
            def __init__(self, track_id):
                self.conf = [Scalar(0.9)]
                self.cls = [Scalar(0)]
                self.xyxy = [Coordinates()]
                self.id = [Scalar(track_id)]

        result = type("Result", (), {"boxes": [Box(11), Box(12)]})()
        model = type("Model", (), {"names": {0: "rider"}})()
        detections = read_detections(result, model)
        self.assertEqual([item["track_id"] for item in detections], [11, 12])

    def test_device_choice_and_gpu_failure_falls_back_to_cpu(self):
        import app
        self.assertEqual(app.pick_device("cpu"), {})
        self.assertEqual(app.pick_device("cuda:0"), {"device": "cuda:0"})
        calls = []

        def flaky(frame, **kwargs):
            calls.append(kwargs)
            if kwargs.get("device") == "mps":
                raise RuntimeError("MPS backend out of memory")
            return "ok"

        with patch.object(app, "DEVICE", {"device": "mps"}):
            self.assertEqual(app.run_model(flaky, "frame", conf=0.5), "ok")
            self.assertEqual(app.DEVICE, {})
        self.assertEqual(calls, [{"conf": 0.5, "device": "mps"}, {"conf": 0.5}])

    def test_keeps_less_certain_plates_and_bikes_but_not_heads(self):
        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Coordinates:
            def cpu(self):
                return self

            def tolist(self):
                return [10, 20, 90, 40]

        class Box:
            def __init__(self, cls, conf):
                self.conf, self.cls, self.xyxy, self.id = [Scalar(conf)], [Scalar(cls)], [Coordinates()], None

        model = type("Model", (), {"names": {0: "helmet", 2: "number-plate", 4: "bike"}})()
        result = type("Result", (), {"boxes": [Box(2, 0.4), Box(4, 0.4), Box(0, 0.4), Box(2, 0.2)]})()
        names = [item["class_name"] for item in read_detections(result, model)]
        self.assertEqual(names, ["number-plate", "bike"])

    def test_saves_confident_home_test_prediction_without_rider_match_once(self):
        detection = {"class_name": "helmet", "confidence": 0.56, "zoomed": True,
                     "track_id": 7, "box": [110, 90, 150, 130]}
        self.assertTrue(should_save_event(detection, {"rider": None}, set()))
        rider = {"class_name": "rider", "track_id": 4, "box": [80, 40, 200, 260]}
        association = {"rider": rider}
        self.assertTrue(should_save_event(detection, association, set()))
        self.assertFalse(should_save_event(detection, association,
                                           event_track_keys(detection, association)))

    def test_only_the_currently_focused_rider_can_be_saved(self):
        rider = {"class_name": "rider", "track_id": 4, "box": [80, 40, 200, 260]}
        other = {"class_name": "rider", "track_id": 5, "box": [300, 40, 420, 260]}
        self.assertTrue(belongs_to_focus({"rider": rider}, rider))
        self.assertFalse(belongs_to_focus({"rider": other}, rider))
        face = {"class_name": "face", "box": [110, 55, 150, 95]}
        self.assertTrue(belongs_to_focus({"rider": rider}, face))

    def test_home_test_saves_prediction_near_focused_face_without_rider(self):
        face = {"class_name": "face", "box": [110, 80, 150, 120]}
        next_face = {"class_name": "face", "box": [310, 80, 350, 120]}
        for name in ("helmet", "no-helmet"):
            prediction = {"class_name": name, "confidence": 0.55, "zoomed": True,
                          "box": [112, 66, 148, 100]}
            self.assertTrue(belongs_to_focus({"rider": None}, face, prediction))
            self.assertTrue(should_save_event(prediction, {"rider": None}, set(), face))
            done = event_track_keys(prediction, {"rider": None}, face)
            self.assertFalse(should_save_event(prediction, {"rider": None}, done, face))
            self.assertIs(focus_target([face, next_face], [face]), next_face)

    def test_unmatched_head_must_be_near_the_focused_face(self):
        face = {"class_name": "face", "box": [110, 80, 150, 120]}
        distant = {"class_name": "helmet", "confidence": 0.95, "box": [310, 66, 350, 100]}
        self.assertFalse(belongs_to_focus({"rider": None}, face, distant))

    def test_live_overlay_shows_head_predictions_without_a_rider_match(self):
        helmet = {"class_name": "helmet", "confidence": 0.9, "box": [20, 20, 50, 50]}
        self.assertEqual(visible_detections([helmet]), [helmet])
        rider = {"class_name": "rider", "confidence": 0.9, "box": [10, 10, 100, 150]}
        self.assertEqual(visible_detections([helmet, rider]), [helmet, rider])

    def test_live_overlay_uses_80_percent_full_frame_and_50_percent_zoom(self):
        rider = {"class_name": "rider", "confidence": 0.9, "box": [10, 10, 100, 150]}
        full_frame = {"class_name": "helmet", "confidence": 0.79, "box": [20, 20, 50, 50]}
        self.assertEqual(visible_detections([full_frame, rider]), [rider])
        full_frame["confidence"] = 0.80
        self.assertEqual(visible_detections([full_frame]), [full_frame])
        zoomed = dict(full_frame, confidence=0.50, zoomed=True)
        self.assertEqual(visible_detections([zoomed]), [zoomed])


    def test_zoom_uses_calibrated_target_size_per_class(self):
        from app import zoom_region
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        head = {"class_name": "helmet", "confidence": 0.9, "track_id": 1, "box": [600, 300, 700, 400]}
        rider = {"class_name": "rider", "confidence": 0.9, "track_id": 2, "box": [500, 200, 700, 400]}
        self.assertAlmostEqual(zoom_region(frame, [head], head)[1][2], 0.20 * 720 / 100)
        self.assertAlmostEqual(zoom_region(frame, [rider], rider)[1][2], 0.45 * 720 / 200)


if __name__ == "__main__":
    unittest.main()
