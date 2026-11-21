import json
from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from event_store import (associate_objects, bulk_apply, clean_expired_events, delete_event, event_stats,
                         export_csv, export_zip, get_setting, import_storage, list_audit, list_events,
                         normalize_plate, query_events, read_plate_text, record_audit, save_event,
                         set_setting, update_review)


class EventStoreTests(unittest.TestCase):
    def test_reopening_review_resets_plate_association(self):
        with tempfile.TemporaryDirectory() as temp:
            head = {"class_name": "no-helmet", "confidence": 0.9, "track_id": 1, "box": [10, 10, 30, 30]}
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

    def test_plate_links_to_rider_when_bike_box_is_missed(self):
        head = {"class_name": "no-helmet", "box": [40, 20, 60, 40]}
        rider = {"class_name": "rider", "box": [10, 10, 100, 200]}
        long_plate = {"class_name": "number-plate", "box": [30, 190, 90, 205]}
        far_plate = {"class_name": "number-plate", "box": [400, 190, 460, 205]}
        self.assertIs(associate_objects(head, [head, rider, long_plate])["plate"], long_plate)
        self.assertIs(associate_objects(head, [head, rider, long_plate, far_plate])["plate"], long_plate)

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


    def _event(self, root, name="no-helmet", **kwargs):
        detection = {"class_name": name, "confidence": 0.9, "track_id": 1, "box": [10, 10, 40, 40]}
        return save_event(np.zeros((80, 100, 3), dtype=np.uint8), detection, [detection],
                          associate_objects(detection, [detection]), root=root, **kwargs)

    def _age(self, root, event, day):
        """Move an event to an older date folder and record (simulates aged evidence)."""
        import sqlite3
        import shutil
        old_folder = Path(root) / event["date"] / event["category"] / event["id"]
        new_folder = Path(root) / day / event["category"] / event["id"]
        new_folder.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_folder), str(new_folder))
        event = dict(event, date=day, timestamp=day + event["timestamp"][10:])
        connection = sqlite3.connect(Path(root) / "events.db")
        connection.execute("UPDATE events SET date=?, timestamp=?, data=? WHERE id=?",
                           (day, event["timestamp"], json.dumps(event), event["id"]))
        connection.commit()
        connection.close()
        return event

    def test_retention_keeps_confirmed_evidence_and_removes_the_rest(self):
        with tempfile.TemporaryDirectory() as temp:
            keep = self._age(temp, self._event(temp), "2025-01-01")
            drop = self._age(temp, self._event(temp), "2025-01-01")
            update_review(keep["date"], keep["id"], "confirmed", root=temp)
            self.assertEqual(clean_expired_events(temp, today=date(2025, 3, 1), retention_days=30,
                                                  keep_confirmed=True), 0)
            remaining = list_events("2025-01-01", temp)
            self.assertEqual([event["id"] for event in remaining], [keep["id"]])
            self.assertTrue((Path(temp) / "2025-01-01" / "no_helmet" / keep["id"] / "person.jpg").is_file())
            self.assertFalse((Path(temp) / "2025-01-01" / "no_helmet" / drop["id"]).exists())
            self.assertEqual(clean_expired_events(temp, today=date(2025, 3, 1), retention_days=30,
                                                  keep_confirmed=False), 1)
            self.assertEqual(list_events("2025-01-01", temp), [])

    def test_retention_zero_keeps_everything(self):
        with tempfile.TemporaryDirectory() as temp:
            event = self._age(temp, self._event(temp), "2020-01-01")
            self.assertEqual(clean_expired_events(temp, today=date(2025, 3, 1), retention_days=0), 0)
            self.assertEqual(len(list_events(event["date"], temp)), 1)

    def test_privacy_mode_blurs_faces_and_drops_plates_for_helmet_riders(self):
        with tempfile.TemporaryDirectory() as temp:
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[20:60, 20:60] = np.indices((40, 40)).sum(axis=0)[..., None] % 2 * 255
            head = {"class_name": "helmet", "confidence": 0.9, "track_id": 1, "box": [15, 5, 65, 65]}
            face = {"class_name": "face", "confidence": 0.9, "track_id": None, "box": [20, 20, 60, 60]}
            plate = {"class_name": "number-plate", "confidence": 0.9, "track_id": 2, "box": [100, 90, 150, 110]}
            event = save_event(frame, head, [head, face, plate], {"rider": None, "bike": None, "plate": plate},
                               root=temp, ocr=lambda _: "GJ01AB1234", privacy=True)
            self.assertIsNone(event["plate"])
            self.assertEqual(event["plate_text"], "")
            self.assertEqual(event["privacy"], {"faces_blurred": 1, "plate_dropped": True})
            import cv2
            saved = cv2.imread(str(Path(temp) / event["date"] / "helmet" / event["id"] / "person.jpg"))
            self.assertLess(saved.std(), frame[20:60, 20:60].std() / 2)
            violation = save_event(frame, dict(head, class_name="no-helmet"), [head, face, plate],
                                   {"rider": None, "bike": None, "plate": plate}, root=temp,
                                   ocr=lambda _: "GJ01AB1234", privacy=True)
            self.assertEqual(violation["plate_text"], "GJ01AB1234")
            self.assertEqual(violation["privacy"], {"faces_blurred": 0, "plate_dropped": False})

    def test_auto_zoom_catches_are_marked(self):
        with tempfile.TemporaryDirectory() as temp:
            head = {"class_name": "no-helmet", "confidence": 0.7, "track_id": 1, "box": [10, 10, 40, 40], "zoomed": True}
            zoomed = save_event(np.zeros((80, 100, 3), dtype=np.uint8), head, [head],
                                associate_objects(head, [head]), root=temp)
            self.assertTrue(zoomed["event_detection"]["zoomed"])
            normal = self._event(temp)
            self.assertNotIn("zoomed", normal["event_detection"])

    def test_plate_normalisation_fixes_common_ocr_confusions(self):
        self.assertEqual(normalize_plate("GJO1AB1234"), ("GJ01AB1234", True))
        self.assertEqual(normalize_plate("6J01A8I234"), ("GJ01AB1234", True))
        self.assertEqual(normalize_plate("22BH1234AA"), ("22BH1234AA", True))
        self.assertEqual(normalize_plate("dl3cab1234"), ("DL3CAB1234", True))
        self.assertEqual(normalize_plate("HELLO"), ("HELLO", False))

    def test_query_filters_stats_and_bulk_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            violation = self._event(temp, ocr=lambda _: "", source={"type": "rtsp", "camera": "Gate 1"})
            helmet = self._event(temp, "helmet")
            update_review(violation["date"], violation["id"], "confirmed", "GJ01AB1234", temp)
            events, total = query_events(temp, category="no_helmet")
            self.assertEqual((total, events[0]["id"]), (1, violation["id"]))
            self.assertEqual(query_events(temp, q="ab12")[1], 1)
            self.assertEqual(query_events(temp, status="needs_review")[0][0]["id"], helmet["id"])
            self.assertEqual(query_events(temp, camera="Gate 1")[1], 1)
            self.assertEqual(query_events(temp, limit=1, offset=1)[1], 2)
            self.assertEqual(len(query_events(temp, limit=1, offset=1)[0]), 1)
            with self.assertRaises(ValueError):
                query_events(temp, status="bogus")
            stats = event_stats(temp)
            self.assertEqual(stats["totals"]["events"], 2)
            self.assertEqual(stats["totals"]["confirmed"], 1)
            self.assertEqual(stats["totals"]["false_positive_rate"], 0.0)
            self.assertIn("Gate 1", stats["cameras"])
            results = bulk_apply([{"date": helmet["date"], "id": helmet["id"]}, {"date": "x", "id": "y"}],
                                 "review", temp, status="false_positive")
            self.assertEqual([item["ok"] for item in results], [True, False])
            self.assertEqual(event_stats(temp)["totals"]["false_positive"], 1)
            results = bulk_apply([{"date": helmet["date"], "id": helmet["id"]}], "delete", temp)
            self.assertTrue(results[0]["ok"])
            self.assertEqual(query_events(temp)[1], 1)

    def test_csv_and_zip_export_include_evidence_and_block_formulas(self):
        import io
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            event = self._event(temp, source={"type": "video", "filename": "=cmd().mp4", "id": "c",
                                               "frame": 1, "position_seconds": 0.1})
            text = export_csv([event])
            self.assertIn("'=cmd().mp4", text)
            self.assertIn(event["id"], text)
            archive = zipfile.ZipFile(export_zip([event], temp, io.BytesIO()))
            names = archive.namelist()
            self.assertIn("events.csv", names)
            self.assertIn(f"{event['date']}/no_helmet/{event['id']}/person.jpg", names)

    def test_import_storage_merges_backup_without_overwriting(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            event = self._event(source)
            self.assertEqual(import_storage(source, target), 1)
            self.assertEqual(import_storage(source, target), 0)
            self.assertTrue((Path(target) / event["date"] / "no_helmet" / event["id"] / "event.jpg").is_file())
            self.assertEqual(list_events(event["date"], target)[0]["id"], event["id"])

    def test_audit_log_and_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            record_audit(temp, "admin", "review", "2025-01-01/event_1", {"status": "confirmed"})
            self.assertEqual(list_audit(temp)[0]["details"], {"status": "confirmed"})
            self.assertIsNone(get_setting(temp, "alerts"))
            set_setting(temp, "alerts", {"webhook_url": "https://example.test"})
            self.assertEqual(get_setting(temp, "alerts")["webhook_url"], "https://example.test")


    def test_repeat_offenders_group_plates_and_skip_false_positives(self):
        from event_store import repeat_offenders
        with tempfile.TemporaryDirectory() as temp:
            events = [self._event(temp, source={"type": "rtsp", "camera": cam}) for cam in ("Gate 1", "Gate 2", "Gate 1")]
            for event in events:
                update_review(event["date"], event["id"], "needs_review", "GJ01AB1234", temp)
            update_review(events[2]["date"], events[2]["id"], "confirmed", "GJ01AB1234", temp)
            once = self._event(temp)
            update_review(once["date"], once["id"], "needs_review", "MH12XY0001", temp)
            wrong = self._event(temp)
            update_review(wrong["date"], wrong["id"], "false_positive", "MH12XY0001", temp)
            helmet = self._event(temp, "helmet")
            update_review(helmet["date"], helmet["id"], "needs_review", "MH12XY0001", temp)
            result = repeat_offenders(temp)
            self.assertEqual([row["plate_text"] for row in result], ["GJ01AB1234"])
            self.assertEqual((result[0]["events"], result[0]["confirmed"]), (3, 1))
            self.assertEqual(result[0]["cameras"], ["Gate 1", "Gate 2"])
            self.assertEqual(result[0]["latest"]["id"], events[2]["id"])
            self.assertEqual(len(repeat_offenders(temp, min_count=1)), 2)


if __name__ == "__main__":
    unittest.main()
