import json
from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import cv2
import numpy as np

from dashboard import DashboardServer
from event_store import associate_objects, save_event

STATIC = Path(__file__).resolve().parent / "static"
PAGE = (STATIC / "index.html").read_text() + (STATIC / "app.js").read_text()


class DashboardTests(unittest.TestCase):
    def test_retention_runs_while_server_stays_open_and_retries_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            server = DashboardServer(port=0, storage=Path(temp), require_auth=False)
            try:
                deadline = server.cleanup_after
                with patch("dashboard.time.monotonic", return_value=deadline - 1), \
                     patch("dashboard.clean_expired_events") as cleanup:
                    server.cleanup_expired()
                    cleanup.assert_not_called()
                with patch("dashboard.time.monotonic", return_value=deadline), \
                     patch("dashboard.clean_expired_events", side_effect=OSError("disk unavailable")) as cleanup, \
                     self.assertLogs(level="ERROR"):
                    server.cleanup_expired()
                    cleanup.assert_called_once_with(Path(temp))
                with patch("dashboard.time.monotonic", return_value=deadline + 3600), \
                     patch("dashboard.clean_expired_events") as cleanup:
                    server.cleanup_expired()
                    cleanup.assert_called_once_with(Path(temp))
            finally:
                server.close()

    def test_full_frame_people_confirm_together_with_video_provenance(self):
        import hi
        from event_store import list_events
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "clip.mp4"
            video.write_bytes(b"test")
            server = DashboardServer(port=0, storage=Path(temp) / "events", require_auth=False)
            server.next_video = Mock(side_effect=[{"path": str(video), "filename": "junction.mp4", "id": "clip1"}, None, None])
            server.next_frame = Mock(side_effect=KeyboardInterrupt)
            capture = Mock()
            capture.isOpened.return_value = True
            frame = np.zeros((120, 200, 3), dtype=np.uint8)
            capture.read.side_effect = [(True, frame)] * 3 + [(False, None)]
            capture.get.side_effect = lambda prop: {cv2.CAP_PROP_POS_FRAMES: 3, cv2.CAP_PROP_FPS: 10,
                                                   cv2.CAP_PROP_FRAME_COUNT: 3}.get(prop, 0)
            model = Mock(names=dict(enumerate(hi.EXPECTED_CLASSES)), predictor=None)
            model.track.return_value = [None]
            heads = [{"class_name": "helmet", "confidence": 0.95, "track_id": i,
                      "box": [x, 20, x + 20, 40], "zoomed": False} for i, x in enumerate((20, 100))]
            with patch.object(hi, "YOLO", return_value=model), \
                 patch.object(hi, "load_thresholds", return_value={"helmet": 0.8, "no-helmet": 0.8}), \
                 patch.object(hi.cv2.FaceDetectorYN, "create"), \
                 patch.object(hi.cv2, "VideoCapture", return_value=capture), \
                 patch.object(hi, "detect_faces", return_value=[]), \
                 patch.object(hi, "read_detections", side_effect=lambda *args, **kwargs: [dict(head) for head in heads]), \
                 self.assertRaises(KeyboardInterrupt):
                hi.run_detector(server)
            model.predict.assert_not_called()
            events = list_events(date.today().isoformat(), server.storage)
            self.assertEqual(len(events), 2)
            for event in events:
                self.assertEqual(event["confirmation_frames"], 3)
                self.assertEqual(event["source"], {"type": "video", "filename": "junction.mp4", "id": "clip1",
                                                   "frame": 2, "position_seconds": 0.2})
            self.assertFalse(video.exists())
            server.close()

    def test_opened_video_with_no_readable_frames_reports_failure_and_is_deleted(self):
        import hi
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "empty.mp4"
            video.write_bytes(b"empty")
            server = DashboardServer(port=0, storage=Path(temp) / "events", require_auth=False)
            server.next_video = Mock(side_effect=[{"path": str(video), "filename": "empty.mp4", "id": "empty"}, None, None])
            server.next_frame = Mock(side_effect=KeyboardInterrupt)
            capture = Mock()
            capture.isOpened.return_value = True
            capture.read.return_value = (False, None)
            capture.get.return_value = 0
            model = SimpleNamespace(names=dict(enumerate(hi.EXPECTED_CLASSES)), predictor=None)
            with patch.object(hi, "YOLO", return_value=model), \
                 patch.object(hi, "load_thresholds", return_value={"helmet": 0.8, "no-helmet": 0.8}), \
                 patch.object(hi.cv2.FaceDetectorYN, "create"), \
                 patch.object(hi.cv2, "VideoCapture", return_value=capture), \
                 self.assertRaises(KeyboardInterrupt):
                hi.run_detector(server)
            self.assertEqual(server.video["state"], "failed")
            self.assertIn("no decodable frames", server.video["error"])
            self.assertFalse(video.exists())
            server.close()

    def test_detector_failure_keeps_event_review_available(self):
        import hi
        with tempfile.TemporaryDirectory() as temp:
            server = DashboardServer(port=0, storage=Path(temp), require_auth=False)

            def check_review(_seconds):
                base = f"http://127.0.0.1:{server.port}"
                with urlopen(base + "/api/status") as response:
                    status = json.load(response)
                self.assertFalse(status["camera_enabled"])
                self.assertFalse(status["video_enabled"])
                self.assertIn("model failed", status["status"])
                with urlopen(base) as response:
                    self.assertEqual(response.status, 200)
                raise KeyboardInterrupt

            with patch.object(hi, "DashboardServer", return_value=server), \
                 patch.object(hi, "clean_expired_events"), \
                 patch.object(hi, "YOLO", side_effect=RuntimeError("model failed")), \
                 patch.object(hi.traceback, "print_exc"), \
                 patch.object(server.restart_detector, "wait", side_effect=check_review):
                hi.main()

    def test_failed_detector_is_retried_without_rebinding_dashboard(self):
        import hi
        with tempfile.TemporaryDirectory() as temp:
            server = DashboardServer(port=0, storage=Path(temp), require_auth=False)
            with patch.object(hi, "DashboardServer", return_value=server), \
                 patch.object(hi, "run_detector", side_effect=[RuntimeError("transient"), KeyboardInterrupt]) as detector, \
                 patch.object(server.restart_detector, "wait", return_value=False), \
                 patch.object(hi.traceback, "print_exc"):
                hi.main()
            self.assertEqual(detector.call_count, 2)
            self.assertTrue(server.detector_managed)

    def test_restart_review_and_video_progress_endpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            detection = {"class_name": "helmet", "confidence": 0.9, "track_id": 1, "box": [10, 10, 40, 40]}
            event = save_event(np.zeros((80, 100, 3), dtype=np.uint8), detection, [detection],
                               associate_objects(detection, [detection]), root=root)
            server = DashboardServer(port=0, storage=root, require_auth=False).start()
            base = f"http://127.0.0.1:{server.port}"
            try:
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + "/api/detector/restart", data=b""))
                self.assertEqual(error.exception.code, 503)
                server.detector_managed = True
                with urlopen(Request(base + "/api/detector/restart", data=b"")) as response:
                    self.assertEqual(response.status, 202)
                self.assertTrue(server.restart_detector.is_set())
                request = Request(base + "/api/review", data=json.dumps({"date": event["date"], "id": event["id"],
                                  "status": "confirmed", "plate_text": "gj 01 ab 1234"}).encode())
                with urlopen(request) as response:
                    saved = json.load(response)
                self.assertEqual(saved["plate_text"], "GJ01AB1234")
                self.assertEqual(saved["status"], "confirmed")
                server.update_video("processing", frames=12, total=30)
                with urlopen(base + "/api/status") as response:
                    self.assertEqual(json.load(response)["video"]["frames"], 12)
            finally:
                server.close()

    def test_local_dashboard_rejects_foreign_hosts_and_camera_origins(self):
        with tempfile.TemporaryDirectory() as temp:
            server = DashboardServer(port=0, storage=Path(temp), accept_frames=True, require_auth=False).start()
            try:
                base = f"http://127.0.0.1:{server.port}"
                requests = [
                    Request(base + "/api/status", headers={"Host": "attacker.example"}),
                    Request(base + "/api/frame", data=b"\xff\xd8test",
                            headers={"Content-Type": "image/jpeg", "Origin": "https://attacker.example"}),
                ]
                for request in requests:
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request)
                    self.assertEqual(error.exception.code, 403)
                self.assertIsNone(server.next_frame(timeout=0))
                with urlopen(Request(base + "/api/status", headers={"Origin": base})) as response:
                    self.assertEqual(response.status, 200)
            finally:
                server.close()

    def test_review_mode_reports_camera_disabled_and_rejects_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            server = DashboardServer(port=0, storage=Path(temp), require_auth=False).start()
            try:
                base = f"http://127.0.0.1:{server.port}"
                with urlopen(base + "/api/status") as response:
                    status = json.load(response)
                self.assertFalse(status["camera_enabled"])
                self.assertFalse(status["video_enabled"])
                request = Request(base + "/api/frame", data=b"\xff\xd8test",
                                  headers={"Content-Type": "image/jpeg"}, method="POST")
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 503)
            finally:
                server.close()

    def test_status_and_both_event_categories_are_served(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            for name in ("helmet", "no-helmet"):
                detection = {"class_name": name, "confidence": 0.9, "track_id": 1,
                             "box": [10, 10, 40, 40]}
                save_event(frame, detection, [detection], associate_objects(detection, [detection]), root=root)

            server = DashboardServer(port=0, storage=root, accept_video=True, accept_frames=True, require_auth=False).start()
            queued = None
            try:
                port = server.port
                server.update_status("Camera permission denied")
                server.update_frame(b"test-frame")
                with urlopen(f"http://127.0.0.1:{port}/api/status") as response:
                    status = json.load(response)
                self.assertEqual(status["status"], "Camera permission denied")
                self.assertTrue(status["camera_enabled"])
                self.assertTrue(status["video_enabled"])
                with urlopen(f"http://127.0.0.1:{port}/api/events?date={date.today().isoformat()}") as response:
                    events = json.load(response)["events"]
                self.assertEqual({event["category"] for event in events}, {"helmet", "no_helmet"})
                for event in events:
                    url = (f"http://127.0.0.1:{port}/media/{event['date']}/"
                           f"{event['category']}/{event['id']}/person.jpg")
                    with urlopen(url) as response:
                        self.assertEqual(response.status, 200)
                        self.assertTrue(response.read().startswith(b"\xff\xd8"))
                with urlopen(f"http://127.0.0.1:{port}/stream", timeout=3) as response:
                    self.assertIn(b"test-frame", response.read(200))
                request = Request(f"http://127.0.0.1:{port}/api/video", data=b"test-video",
                                  headers={"X-File-Name": "sample.mp4"}, method="POST")
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                job = server.next_video()
                queued = Path(job["path"])
                self.assertEqual(queued.read_bytes(), b"test-video")
                self.assertEqual(job["filename"], "sample.mp4")
                self.assertTrue(job["id"].startswith("helmet_video_"))
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 409)
                self.assertIn('data-category="helmet"', PAGE)
                self.assertIn('data-category="no_helmet"', PAGE)
                self.assertNotIn('id="category-filter"', PAGE)
                self.assertIn("navigator.mediaDevices.getUserMedia", PAGE)
                self.assertIn("/api/frame", PAGE)
                ok, encoded = cv2.imencode(".jpg", frame)
                self.assertTrue(ok)
                request = Request(f"http://127.0.0.1:{port}/api/frame", data=encoded.tobytes(),
                                  headers={"Content-Type": "image/jpeg"}, method="POST")
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                self.assertEqual(server.next_frame(timeout=0.1), encoded.tobytes())
            finally:
                if queued:
                    queued.unlink(missing_ok=True)
                queued_path = server.next_video()
                if queued_path:
                    Path(queued_path["path"]).unlink(missing_ok=True)
                server.close()


if __name__ == "__main__":
    unittest.main()
