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
        if self.headers.get("Content-Type", "").startswith("application/x-www-form-urlencoded"):
            from urllib.parse import parse_qs
            parsed = {key: values[0] for key, values in parse_qs(body.decode()).items()}
            RecordingHandler.received.append((self.headers.get("Authorization"), dict(parsed, path=self.path)))
        else:
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

    def test_whatsapp_via_twilio_sends_to_each_number(self):
        import base64
        sid = "AC" + "0" * 32
        with self.assertRaises(ValueError):
            alerts.save_settings(self.root, {"whatsapp_enabled": True, "twilio_account_sid": "bad",
                                             "twilio_auth_token": "t", "whatsapp_from": "+14155238886",
                                             "whatsapp_to": "+919876543210"})
        with self.assertRaises(ValueError):
            alerts.save_settings(self.root, {"whatsapp_enabled": True, "twilio_account_sid": sid,
                                             "twilio_auth_token": "t", "whatsapp_from": "+14155238886",
                                             "whatsapp_to": "98765 43210"})
        saved = alerts.save_settings(self.root, {
            "enabled": True, "whatsapp_enabled": True, "twilio_account_sid": sid, "twilio_auth_token": "tok",
            "whatsapp_from": "whatsapp:+14155238886", "whatsapp_to": "+91 98765 43210, +919999999999"})
        self.assertEqual(saved["whatsapp_from"], "+14155238886")
        self.assertEqual(saved["whatsapp_to"], "+919876543210,+919999999999")
        self.assertEqual(alerts.public_settings(saved)["twilio_auth_token"], "")
        with patch("alerts.TWILIO_API", self.url.replace("/hook", "/{sid}/Messages.json")):
            result = alerts.AlertDispatcher(self.root).deliver(sample_event())
        self.assertEqual(result, "sent:whatsapp")
        self.assertEqual(len(RecordingHandler.received), 2)
        auth, form = RecordingHandler.received[0]
        self.assertEqual(auth, "Basic " + base64.b64encode(f"{sid}:tok".encode()).decode())
        self.assertEqual(form["path"], f"/{sid}/Messages.json")
        self.assertEqual(form["From"], "whatsapp:+14155238886")
        self.assertEqual(form["To"], "whatsapp:+919876543210")
        self.assertIn("No helmet detected on Gate 1", form["Body"])
        self.assertEqual(RecordingHandler.received[1][1]["To"], "whatsapp:+919999999999")

    def test_health_alerts_are_sent_rate_limited_and_can_be_switched_off(self):
        alerts.save_settings(self.root, {"enabled": True, "webhook_url": self.url, "min_interval_seconds": 0})
        clock = Mock(return_value=100.0)
        dispatcher = alerts.AlertDispatcher(self.root, clock=clock)
        self.assertEqual(dispatcher.deliver_health("camera:c1", "Camera Gate 1 is offline for 61 s (timeout)."),
                         "sent:webhook")
        self.assertIn("Helmet system alert: Camera Gate 1 is offline", RecordingHandler.received[0][1]["text"])
        self.assertEqual(dispatcher.deliver_health("camera:c1", "Camera Gate 1 is offline again."), "rate_limited")
        self.assertEqual(dispatcher.deliver_health("camera:c1", "Camera Gate 1 is back online."), "sent:webhook")
        clock.return_value = 100.0 + alerts.HEALTH_INTERVAL
        self.assertEqual(dispatcher.deliver_health("camera:c1", "Camera Gate 1 is offline again."), "sent:webhook")
        alerts.save_settings(self.root, {"health_alerts": False})
        self.assertEqual(dispatcher.deliver_health("detector", "Detector stopped."), "disabled")

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


class ZoneTests(unittest.TestCase):
    def test_zone_validation_and_point_in_polygon(self):
        lane = [[0.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]]  # bottom-left quarter
        self.assertTrue(cameras.in_zone(lane, [10, 60, 30, 80], 100, 100))
        self.assertFalse(cameras.in_zone(lane, [60, 60, 80, 80], 100, 100))
        self.assertFalse(cameras.in_zone(lane, [10, 10, 30, 30], 100, 100))
        self.assertTrue(cameras.in_zone(None, [60, 60, 80, 80], 100, 100), "no zone = whole picture")
        self.assertIsNone(cameras.validate_zone([]))
        for bad in ([[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 0]], [[0, 0], [1], [1, 0]]):
            with self.assertRaises(ValueError):
                cameras.validate_zone(bad)

    def test_pipeline_ignores_riders_outside_the_zone(self):
        import app
        from event_store import list_events
        head = {"class_name": "no-helmet", "confidence": 0.95, "track_id": 1, "box": [60, 10, 80, 30], "zoomed": False}
        with tempfile.TemporaryDirectory() as temp:
            model = Mock(names=dict(enumerate(app.EXPECTED_CLASSES)), predictor=None)
            model.track.return_value = [None]
            dashboard = Mock(storage=Path(temp), url="http://test")
            pipeline = app.SourcePipeline("camera:c1", model, Mock(), {"helmet": 0.8, "no-helmet": 0.8})
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            left_half = [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]
            with patch.object(app, "detect_faces", return_value=[]), \
                 patch.object(app, "read_detections", side_effect=lambda *a, **k: [dict(head)]):
                for _ in range(4):
                    pipeline.process(frame, {"type": "rtsp", "camera": "Gate"}, dashboard, zone=left_half)
                self.assertEqual(list_events(date.today().isoformat(), Path(temp)), [])
                pipeline.reset()
                for _ in range(4):
                    pipeline.process(frame, {"type": "rtsp", "camera": "Gate"}, dashboard,
                                     zone=[[0.5, 0], [1, 0], [1, 1], [0.5, 1]])
            self.assertEqual(len(list_events(date.today().isoformat(), Path(temp))), 1)


class CameraHealthTests(unittest.TestCase):
    def test_offline_reported_once_after_grace_and_recovery_reported(self):
        monitor = cameras.CameraHealthMonitor(grace_seconds=60)
        names = {"c1": "Gate 1"}
        down = {"c1": {"state": "reconnecting", "error": "Could not open stream"}}
        self.assertEqual(monitor.update(down, names, 0), [])
        self.assertEqual(monitor.update(down, names, 30), [], "short drops are ignored")
        offline = monitor.update(down, names, 61)
        self.assertEqual(len(offline), 1)
        self.assertIn("Gate 1 is offline for 61 s (Could not open stream)", offline[0][1])
        self.assertEqual(monitor.update(down, names, 120), [], "reported only once")
        back = monitor.update({"c1": {"state": "live"}}, names, 125)
        self.assertEqual(back, [("camera:c1", "Camera Gate 1 is back online.")])
        self.assertEqual(monitor.update({"c1": {"state": "live"}}, names, 130), [])
        self.assertEqual(monitor.update({"c1": {"state": "reconnecting"}}, names, 140), [])
        self.assertEqual(monitor.update({"c1": {"state": "live"}}, names, 150), [], "unreported blip has no recovery")


class MultiSourceDetectorTests(unittest.TestCase):
    def test_face_locator_runs_every_second_frame_and_reuses_faces(self):
        import app
        face = {"class_name": "face", "confidence": 0.9, "track_id": None, "box": [10, 10, 30, 30], "zoomed": False}
        model = Mock(names=dict(enumerate(app.EXPECTED_CLASSES)), predictor=None)
        model.track.return_value = [None]
        dashboard = Mock(storage=Path("."), url="http://test")
        pipeline = app.SourcePipeline("browser", model, Mock(), {"helmet": 0.8, "no-helmet": 0.8})
        seen = []
        with patch.object(app, "FACE_EVERY", 2), \
             patch.object(app, "detect_faces", return_value=[face]) as faces, \
             patch.object(app, "read_detections", side_effect=lambda *a, **k: []), \
             patch.object(app, "zoom_region", side_effect=lambda frame, detections, target: (frame, None)), \
             patch.object(app.CaptureState, "select", side_effect=lambda detections, *a: seen.append(
                 [d for d in detections if d["class_name"] == "face"])):
            for _ in range(4):
                pipeline.process(np.zeros((60, 80, 3), dtype=np.uint8), {"type": "camera"}, dashboard)
        self.assertEqual(faces.call_count, 2)
        self.assertTrue(all(found == [face] for found in seen), "faces are reused on skipped frames")

    def test_cctv_frames_get_their_own_pipeline_and_camera_source(self):
        import app
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
                model = Mock(names=dict(enumerate(app.EXPECTED_CLASSES)), predictor=None)
                model.track.return_value = [None]
                models.append(model)
                return model

            head = {"class_name": "no-helmet", "confidence": 0.95, "track_id": 4, "box": [20, 20, 40, 40], "zoomed": False}
            alerts_mock = Mock()
            with patch.object(app, "YOLO", side_effect=make_model), \
                 patch.object(app, "load_thresholds", return_value={"helmet": 0.8, "no-helmet": 0.8}), \
                 patch.object(app.cv2.FaceDetectorYN, "create"), \
                 patch.object(app, "detect_faces", return_value=[]), \
                 patch.object(app, "read_detections", side_effect=lambda *a, **k: [dict(head)]), \
                 self.assertRaises(KeyboardInterrupt):
                app.run_detector(server, manager, alerts_mock)
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
