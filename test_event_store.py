import json
from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from event_store import (associate_objects, clean_expired_events, delete_event, list_events,
                         read_plate_text, save_event, update_review)


class EventStoreTests(unittest.TestCase):
    def test_reopening_review_resets_plate_association(self):
        with tempfile.TemporaryDirectory() as temp:
            head = {"class_name": "helmet", "confidence": 0.9, "track_id": 1, "box": [10, 10, 30, 30]}
            plate = dict(head, class_name="number-plate", box=[20, 50, 40, 60])
            event = save_event(np.zeros((80, 100, 3), dtype=np.uint8), head, [head, plate],
                               {"rider": None, "bike": None, "plate": plate}, root=temp, ocr=lambda _: "")
            for status, expected in (("confirmed", "confirmed"), ("needs_review", "suggested"),
                                     ("false_positive", "rejected"), ("needs_review", "suggested")):
                with self.subTest(status=status):
                    reviewed = update_review(event["date"], event["id"], status, root=temp)
                    self.assertEqual(reviewed["association_status"], expected)

    def test_video_source_is_preserved_in_database_and_json(self):
        with tempfile.TemporaryDirectory() as temp:
            head = {"class_name": "helmet", "confidence": 0.9, "track_id": 1, "box": [10, 10, 30, 30]}
            source = {"type": "video", "filename": "junction.mp4", "id": "clip1",
                      "frame": 24, "position_seconds": 2.4}
            event = save_event(np.zeros((80, 100, 3), dtype=np.uint8), head, [head],
                               associate_objects(head, [head]), root=temp, source=source)
            self.assertEqual(list_events(event["date"], temp)[0]["source"], source)
            path = Path(temp) / event["date"] / "helmet" / event["id"] / "event.json"
            self.assertEqual(json.loads(path.read_text())["source"], source)

    def test_failed_ocr_does_not_accept_stdout_as_plate_text(self):
        from types import SimpleNamespace
        with patch("event_store.TESSERACT", "tesseract"), patch("event_store.subprocess.run", return_value=
                SimpleNamespace(returncode=1, stdout=b"ERROR123")):
            self.assertEqual(read_plate_text(np.zeros((20, 60, 3), dtype=np.uint8)), "")

    def test_ambiguous_riders_and_plates_are_left_unmatched(self):
        head = {"class_name": "no-helmet", "box": [70, 50, 90, 70]}
        rider = {"class_name": "rider", "box": [30, 30, 150, 260]}
        bike = {"class_name": "bike", "box": [35, 150, 210, 290]}
        plate = {"class_name": "number-plate", "box": [180, 240, 200, 255]}
        other_rider = dict(rider, box=[50, 25, 180, 270])
        other_plate = dict(plate, box=[80, 245, 100, 260])
        self.assertIsNone(associate_objects(head, [head, rider, other_rider, bike, plate])["rider"])
        self.assertIsNone(associate_objects(head, [head, rider, bike, plate, other_plate])["plate"])
        outside = dict(plate, box=[215, 240, 240, 255])
        self.assertIsNone(associate_objects(head, [head, rider, bike, outside])["plate"])

    def test_retention_ignores_invalid_calendar_folder_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            invalid = root / "2025-99-99"
            invalid.mkdir()
            self.assertEqual(clean_expired_events(root, today=date(2025, 1, 1)), 0)
            self.assertTrue(invalid.is_dir())

    def test_save_failure_does_not_leave_a_partial_event(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            detection = {"class_name": "helmet", "confidence": 0.9, "track_id": 1,
                         "box": [10, 10, 40, 40]}
            with patch("event_store.cv2.imwrite", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_event(np.zeros((80, 100, 3), dtype=np.uint8), detection, [detection],
                               associate_objects(detection, [detection]), root=root)
            self.assertEqual(list(root.glob("*/*/event_*")), [])

    def test_invalid_detection_boxes_are_rejected_without_saving_pictures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for box in ([150, 150, 180, 180], [-100, -100, -50, -50], [40, 40, 10, 10]):
                detection = {"class_name": "helmet", "confidence": 0.9, "track_id": 1, "box": box}
                with self.subTest(box=box), self.assertRaises(ValueError):
                    save_event(np.zeros((80, 100, 3), dtype=np.uint8), detection, [detection],
                               associate_objects(detection, [detection]), root=root)
            self.assertEqual(list(root.glob("*/*/event_*")), [])

    def test_far_away_bike_does_not_supply_an_unrelated_plate(self):
        head = {"class_name": "no-helmet", "box": [20, 20, 40, 40]}
        rider = {"class_name": "rider", "box": [10, 10, 80, 180]}
        bike = {"class_name": "bike", "box": [800, 400, 1000, 600]}
        plate = {"class_name": "number-plate", "box": [950, 540, 990, 565]}
        association = associate_objects(head, [head, rider, bike, plate])
        self.assertIsNone(association["bike"])
        self.assertIsNone(association["plate"])

    def test_imports_legacy_json_events_into_sqlite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "2025-01-01" / "event_120000_123456"
            folder.mkdir(parents=True)
            event = {"date": "2025-01-01", "id": folder.name, "timestamp": "2025-01-01T12:00:00", "status": "needs_review"}
            (folder / "event.json").write_text(json.dumps(event), encoding="utf-8")
            self.assertEqual(list_events("2025-01-01", root), [event])
            self.assertTrue((root / "events.db").is_file())
            clean_expired_events(root, today=date(2025, 1, 1))
            categorized = root / "2025-01-01" / "no_helmet" / folder.name
            self.assertTrue((categorized / "event.json").is_file())
            self.assertEqual(list_events("2025-01-01", root)[0]["category"], "no_helmet")

    def test_suggests_rider_bike_and_nearby_plate(self):
        detections = [
            {"class_name": "no-helmet", "confidence": 0.9, "track_id": 1, "box": [70, 50, 90, 70]},
            {"class_name": "rider", "confidence": 0.8, "track_id": 2, "box": [30, 30, 150, 260]},
            {"class_name": "bike", "confidence": 0.8, "track_id": 3, "box": [35, 150, 210, 290]},
            {"class_name": "number-plate", "confidence": 0.9, "track_id": 4, "box": [180, 240, 230, 255]},
        ]
        match = associate_objects(detections[0], detections)
        self.assertEqual(match["rider"]["track_id"], 2)
        self.assertEqual(match["bike"]["track_id"], 3)
        self.assertEqual(match["plate"]["track_id"], 4)

    def test_save_review_and_retention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            detections = [
                {"class_name": "no-helmet", "confidence": 0.9, "track_id": 1, "box": [70, 50, 90, 70]},
                {"class_name": "rider", "confidence": 0.8, "track_id": 2, "box": [30, 30, 150, 260]},
                {"class_name": "bike", "confidence": 0.8, "track_id": 3, "box": [35, 150, 210, 290]},
                {"class_name": "number-plate", "confidence": 0.9, "track_id": 4, "box": [180, 240, 230, 255]},
            ]
            event = save_event(
                np.zeros((320, 400, 3), dtype=np.uint8), detections[0], detections,
                associate_objects(detections[0], detections), ocr=lambda _crop: "GJ01AB1234", root=root,
            )
            folder = root / event["date"] / "no_helmet" / event["id"]
            self.assertTrue((folder / "person.jpg").is_file())
            self.assertTrue((folder / "plate.jpg").is_file())
            self.assertEqual(json.loads((folder / "event.json").read_text())["plate_text"], "GJ01AB1234")
            self.assertEqual(event["category"], "no_helmet")
            reviewed = update_review(event["date"], event["id"], "confirmed", "gj 01 ab 1234", root)
            self.assertEqual(reviewed["plate_text"], "GJ01AB1234")
            self.assertEqual(reviewed["association_status"], "confirmed")
            self.assertTrue((root / "events.db").is_file())
            self.assertEqual(list_events(event["date"], root)[0]["status"], "confirmed")
            self.assertTrue(delete_event(event["date"], event["id"], root))
            self.assertFalse(folder.exists())
            self.assertEqual(list_events(event["date"], root), [])

            old = root / "2024-12-31"
            recent = root / "2025-01-15"
            old.mkdir()
            recent.mkdir()
            (root / "notes.txt").write_text("keep")
            self.assertEqual(clean_expired_events(root, today=date(2025, 1, 31), retention_days=30), 1)
            self.assertFalse(old.exists())
            self.assertTrue(recent.exists())
            self.assertTrue((root / "notes.txt").exists())

    def test_saves_helmet_images_in_helmet_category(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            detections = [
                {"class_name": "helmet", "confidence": 0.87, "track_id": 1, "box": [70, 50, 90, 70]},
                {"class_name": "rider", "confidence": 0.78, "track_id": 2, "box": [30, 30, 150, 260]},
            ]
            event = save_event(
                np.zeros((320, 400, 3), dtype=np.uint8), detections[0], detections,
                associate_objects(detections[0], detections), root=root,
            )
            folder = root / event["date"] / "helmet" / event["id"]
            self.assertTrue((folder / "person.jpg").is_file())
            self.assertEqual(event["category"], "helmet")
            self.assertEqual(list_events(event["date"], root), [event])
            self.assertTrue(delete_event(event["date"], event["id"], root))
            self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
