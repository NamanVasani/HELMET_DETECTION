from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

import alerts
import cameras


class RecordingHandler(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        RecordingHandler.received.append((self.headers.get("Authorization"), json.loads(body)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args):
        pass


def sample_event(category="no_helmet"):
    today = date.today().isoformat()
    return {"id": "event_1", "date": today, "timestamp": today + "T10:00:00+05:30", "category": category,
            "plate_text": "GJ01AB1234", "event_detection": {"confidence": 0.91},
            "source": {"type": "rtsp", "camera": "Gate 1"}}


class AlertTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        RecordingHandler.received = []
        self.http = HTTPServer(("127.0.0.1", 0), RecordingHandler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}/hook"

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.temp.cleanup()

    def test_webhook_payload_rate_limit_and_category_filter(self):
        alerts.save_settings(self.root, {"enabled": True, "webhook_url": self.url, "webhook_token": "secret",
                                         "min_interval_seconds": 60})
        clock = Mock(return_value=1000.0)
        dispatcher = alerts.AlertDispatcher(self.root, clock=clock)
        self.assertEqual(dispatcher.deliver(sample_event()), "sent:webhook")
        token, payload = RecordingHandler.received[0]
        self.assertEqual(token, "Bearer secret")
        self.assertIn("No helmet detected on Gate 1", payload["text"])
        self.assertIn("event=event_1", payload["event"]["link"])
        self.assertEqual(dispatcher.deliver(sample_event()), "rate_limited")
        clock.return_value = 1061.0
        self.assertEqual(dispatcher.deliver(sample_event("helmet")), "disabled")
        self.assertEqual(dispatcher.deliver(sample_event()), "sent:webhook")

    def test_background_failures_are_recorded_not_raised(self):
        alerts.save_settings(self.root, {"enabled": True, "webhook_url": "http://127.0.0.1:9/unreachable",
                                         "min_interval_seconds": 0})
        dispatcher = alerts.AlertDispatcher(self.root)
        dispatcher.notify(sample_event())
        deadline = time.monotonic() + 15
        while not dispatcher.last_error and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(dispatcher.last_error)
        from event_store import list_audit
        self.assertTrue(any(entry["action"] == "alert_failed" for entry in list_audit(self.root)))

    def test_email_requires_complete_settings_and_uses_starttls(self):
        with self.assertRaises(ValueError):
            alerts.save_settings(self.root, {"email_enabled": True, "smtp_host": "smtp.example.test"})
        alerts.save_settings(self.root, {"enabled": True, "email_enabled": True, "smtp_host": "smtp.example.test",
                                         "smtp_from": "helmet@example.test", "email_to": "a@example.test, b@example.test",
                                         "smtp_user": "u", "smtp_password": "p"})
        with patch("alerts.smtplib.SMTP") as smtp:
            result = alerts.AlertDispatcher(self.root).deliver(sample_event())
        self.assertEqual(result, "sent:email")
        server = smtp.return_value  # smtplib.SMTP.__enter__ returns the same object
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("u", "p")
        message = server.send_message.call_args[0][0]
        self.assertIn("b@example.test", message["To"])


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        pass


class CameraTests(unittest.TestCase):
    def test_redaction_and_validation(self):
        self.assertEqual(cameras.redact_url("rtsp://admin:pw@10.0.0.2:554/s?token=x"), "rtsp://***:***@10.0.0.2:554/s")
        self.assertEqual(cameras.redact_url("0"), "Local device 0")
        with self.assertRaises(ValueError):
            cameras.validate_camera({"name": "Gate", "url": "file:///etc/passwd"})
        with self.assertRaises(ValueError):
            cameras.validate_camera({"name": "browser", "url": "rtsp://x/y"})
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            cameras.save_cameras(temp, [{"name": "A", "url": "rtsp://x/1"}, {"name": "a", "url": "rtsp://x/2"}])

    def test_reader_keeps_latest_frame_and_reconnects(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        opened = []

        def factory(_source):
            opened.append(1)
            return FakeCapture([frame, frame + 1])

        reader = cameras.CameraReader({"id": "c1", "name": "Gate", "url": "rtsp://cam/1"}, capture_factory=factory)
        with patch("cameras.time.sleep"):
            reader.start()
            deadline = time.monotonic() + 5
            while reader.status()["state"] != "reconnecting" and time.monotonic() < deadline:
                time.sleep(0.01)
        latest = reader.latest()
        self.assertIsNotNone(latest)
        self.assertEqual(int(latest[0, 0, 0]), 1, "only the newest frame is kept")
        self.assertIsNone(reader.latest(), "a frame is consumed once")
        self.assertIn("stopped sending", reader.status()["error"])
        reader.stop()

    def test_manager_starts_and_stops_readers_to_match_settings(self):
        started = []

        class Reader:
            def __init__(self, camera):
                self.camera = camera
                self.stopped = False

            def start(self):
                started.append(self.camera["id"])
                return self

            def stop(self):
                self.stopped = True

            def status(self):
                return {"state": "live"}

        with tempfile.TemporaryDirectory() as temp:
            manager = cameras.CameraManager(temp, reader_factory=Reader)
            manager.sync([{"id": "a", "name": "A", "url": "rtsp://a", "enabled": True},
                          {"id": "b", "name": "B", "url": "rtsp://b", "enabled": False}])
            self.assertEqual(list(manager.readers), ["a"])
            first = manager.readers["a"]
            manager.sync([{"id": "a", "name": "A", "url": "rtsp://a2", "enabled": True}])
            self.assertTrue(first.stopped)
            self.assertEqual(started, ["a", "a"])
            manager.stop_all()
            self.assertEqual(manager.readers, {})


class MultiSourceDetectorTests(unittest.TestCase):
    def test_cctv_frames_get_their_own_pipeline_and_camera_source(self):
        import hi
        from dashboard import DashboardServer
        from event_store import list_events
        with tempfile.TemporaryDirectory() as temp:
            server = DashboardServer(port=0, storage=Path(temp), require_auth=False)
            frame = np.zeros((120, 200, 3), dtype=np.uint8)
            reader = Mock()
            reader.camera = {"name": "Gate 1", "site": "Plant A"}
            reader.latest.side_effect = [frame, frame, frame, KeyboardInterrupt]
            manager = Mock(readers={"cam1": reader})
            manager.statuses.return_value = {"cam1": {"state": "live"}}
            server.next_frame = Mock(return_value=None)
            models = []

            def make_model(*_args):
                model = Mock(names=dict(enumerate(hi.EXPECTED_CLASSES)), predictor=None)
                model.track.return_value = [None]
                models.append(model)
                return model

            head = {"class_name": "no-helmet", "confidence": 0.95, "track_id": 4, "box": [20, 20, 40, 40], "zoomed": False}
            alerts_mock = Mock()
            with patch.object(hi, "YOLO", side_effect=make_model), \
                 patch.object(hi, "load_thresholds", return_value={"helmet": 0.8, "no-helmet": 0.8}), \
                 patch.object(hi.cv2.FaceDetectorYN, "create"), \
                 patch.object(hi, "detect_faces", return_value=[]), \
                 patch.object(hi, "read_detections", side_effect=lambda *a, **k: [dict(head)]), \
                 self.assertRaises(KeyboardInterrupt):
                hi.run_detector(server, manager, alerts_mock)
            events = list_events(date.today().isoformat(), Path(temp))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["source"], {"type": "rtsp", "camera": "Gate 1", "camera_id": "cam1", "site": "Plant A"})
            self.assertEqual(len(models), 2, "browser and CCTV sources must not share tracker state")
            self.assertIn("camera:cam1", server.metrics)
            self.assertIn("camera:cam1", server.frames)
            alerts_mock.notify.assert_called_once()
            self.assertEqual(server.camera_status, {"cam1": {"state": "live"}})
            server.close()


if __name__ == "__main__":
    unittest.main()
