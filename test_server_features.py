"""HTTP tests for login, roles, exports, bulk review, cameras, alerts and security headers."""
from datetime import date
from http.cookiejar import CookieJar
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener
import zipfile

import numpy as np

import auth
from dashboard import DashboardServer
from event_store import associate_objects, save_event


def make_event(root, name="no-helmet", **kwargs):
    detection = {"class_name": name, "confidence": 0.9, "track_id": 1, "box": [10, 10, 40, 40]}
    return save_event(np.zeros((80, 100, 3), dtype=np.uint8), detection, [detection],
                      associate_objects(detection, [detection]), root=root, **kwargs)


class Client:
    def __init__(self, base):
        self.base = base
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def call(self, path, payload=None, method=None, raw=False):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.base + path, data=data, method=method,
                          headers={"Content-Type": "application/json"} if data else {})
        with self.opener.open(request, timeout=10) as response:
            body = response.read()
            return (response, body) if raw else json.loads(body)

    def status_of(self, path, payload=None, method=None):
        try:
            self.call(path, payload, method, raw=True)
            return 200
        except HTTPError as error:
            return error.code


class ServerFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.server = DashboardServer(port=0, storage=self.root, require_auth=True).start()
        self.base = f"http://127.0.0.1:{self.server.port}"

    def tearDown(self):
        self.server.close()
        self.temp.cleanup()

    def admin(self):
        client = Client(self.base)
        client.call("/api/setup", {"username": "admin", "password": "correct horse battery"})
        return client

    def test_first_run_setup_then_login_and_roles(self):
        anonymous = Client(self.base)
        self.assertTrue(anonymous.call("/api/session")["setup_required"])
        self.assertEqual(anonymous.status_of("/api/events"), 401)
        self.assertEqual(anonymous.status_of("/api/status"), 401)
        self.assertEqual(anonymous.status_of("/api/setup", {"username": "admin", "password": "short"}), 400)
        admin = self.admin()
        self.assertEqual(admin.call("/api/session")["user"]["role"], "admin")
        self.assertEqual(anonymous.status_of("/api/setup", {"username": "x2", "password": "correct horse battery"}), 409)
        admin.call("/api/users", {"username": "viewer1", "password": "viewer password", "role": "viewer"})
        viewer = Client(self.base)
        self.assertEqual(viewer.status_of("/api/login", {"username": "viewer1", "password": "wrong password"}), 401)
        viewer.call("/api/login", {"username": "viewer1", "password": "viewer password"})
        event = make_event(self.root)
        self.assertEqual(viewer.call(f"/api/events?date={event['date']}")["total"], 1)
        self.assertEqual(viewer.status_of("/api/review", {"date": event["date"], "id": event["id"],
                                                          "status": "confirmed"}), 403)
        self.assertEqual(viewer.status_of("/api/users"), 403)
        self.assertEqual(viewer.status_of("/api/export.csv"), 403)
        viewer.call("/api/logout", {})
        self.assertEqual(viewer.status_of("/api/events"), 401)
        actions = [entry["action"] for entry in admin.call("/api/audit")["entries"]]
        self.assertIn("setup_admin", actions)
        self.assertIn("login_failed", actions)

    def test_login_is_rate_limited_after_repeated_failures(self):
        self.admin()
        client = Client(self.base)
        codes = [client.status_of("/api/login", {"username": "admin", "password": "nope nope nope"}) for _ in range(6)]
        self.assertEqual(codes[:5], [401] * 5)
        self.assertEqual(codes[5], 429)

    def test_last_admin_cannot_be_removed_or_demoted(self):
        admin = self.admin()
        self.assertEqual(admin.status_of("/api/users/admin", {"role": "viewer"}, method="PATCH"), 400)
        self.assertEqual(admin.status_of("/api/users/admin", method="DELETE"), 400)

    def test_security_headers_and_static_assets(self):
        client = Client(self.base)
        response, body = client.call("/", raw=True)
        self.assertIn(b'src="/static/app.js"', body)
        self.assertNotIn(b"<script>", body)
        csp = response.headers["Content-Security-Policy"]
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("unsafe-inline", csp)
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        response, body = client.call("/static/app.js", raw=True)
        self.assertIn("javascript", response.headers["Content-Type"])
        self.assertEqual(client.status_of("/static/../dashboard.py"), 404)
        self.assertEqual(client.status_of("/static/secret.txt"), 404)

    def test_filters_exports_report_and_bulk_review(self):
        admin = self.admin()
        violation = make_event(self.root, ocr=lambda _: "GJ01AB1234")
        helmet = make_event(self.root, "helmet")
        today = date.today().isoformat()
        data = admin.call(f"/api/events?start={today}&end={today}&category=no_helmet")
        self.assertEqual([item["id"] for item in data["events"]], [violation["id"]])
        self.assertEqual(admin.status_of("/api/events?status=bogus"), 400)
        self.assertEqual(admin.call("/api/stats")["totals"]["events"], 2)
        admin.call("/api/review", {"date": violation["date"], "id": violation["id"], "status": "needs_review",
                                   "plate_text": "GJ01AB1234"})
        offenders = admin.call("/api/offenders?min=1")["offenders"]
        self.assertEqual([row["plate_text"] for row in offenders], ["GJ01AB1234"])
        self.assertEqual(admin.call("/api/offenders")["offenders"], [])
        response, body = admin.call(f"/api/export.csv?date={today}", raw=True)
        self.assertIn("attachment", response.headers["Content-Disposition"])
        self.assertIn(violation["id"], body.decode())
        response, body = admin.call(f"/api/export.zip?date={today}", raw=True)
        names = zipfile.ZipFile(io.BytesIO(body)).namelist()
        self.assertIn(f"{today}/helmet/{helmet['id']}/person.jpg", names)
        response, body = admin.call(f"/api/report.pdf?start={today}&end={today}", raw=True)
        self.assertTrue(body.startswith(b"%PDF"))
        result = admin.call("/api/bulk", {"action": "review", "status": "confirmed", "items": [
            {"date": today, "id": violation["id"]}, {"date": today, "id": helmet["id"]}]})
        self.assertTrue(all(item["ok"] for item in result["results"]))
        self.assertEqual(admin.call("/api/stats")["totals"]["confirmed"], 2)
        result = admin.call("/api/bulk", {"action": "delete", "items": [{"date": today, "id": helmet["id"]}]})
        self.assertTrue(result["results"][0]["ok"])
        self.assertEqual(admin.call(f"/api/events?date={today}&status=")["total"], 1)

    def test_media_requires_login_and_is_cacheable(self):
        admin = self.admin()
        event = make_event(self.root)
        path = f"/media/{event['date']}/no_helmet/{event['id']}/person.jpg"
        self.assertEqual(Client(self.base).status_of(path), 401)
        response, body = admin.call(path, raw=True)
        self.assertTrue(body.startswith(b"\xff\xd8"))
        self.assertIn("max-age", response.headers["Cache-Control"])
        self.assertEqual(admin.status_of(f"/media/{event['date']}/no_helmet/{event['id']}/event.json"), 404)

    def test_camera_urls_are_redacted_and_preserved(self):
        admin = self.admin()
        saved = admin.call("/api/cameras", {"cameras": [
            {"name": "Gate 1", "site": "Plant A", "url": "rtsp://user:secret@10.0.0.5:554/stream1", "enabled": True}]},
            method="PUT")["cameras"]
        self.assertNotIn("secret", json.dumps(saved))
        listed = admin.call("/api/cameras")["cameras"]
        self.assertEqual(listed[0]["url"], "rtsp://***:***@10.0.0.5:554/stream1")
        admin.call("/api/cameras", {"cameras": [dict(listed[0], name="Gate 1 north")]}, method="PUT")
        from cameras import load_cameras
        stored = load_cameras(self.root)[0]
        self.assertEqual(stored["url"], "rtsp://user:secret@10.0.0.5:554/stream1")
        self.assertEqual(stored["name"], "Gate 1 north")
        self.assertEqual(admin.status_of("/api/cameras", {"cameras": [{"name": "x", "url": "file:///etc/passwd"}]},
                                         method="PUT"), 400)
        zone = [[0.1, 0.5], [0.9, 0.5], [0.9, 1.0], [0.1, 1.0]]
        admin.call("/api/cameras", {"cameras": [dict(listed[0], name="Gate 1 north", zone=zone)]}, method="PUT")
        self.assertEqual(load_cameras(self.root)[0]["zone"], zone)
        self.assertEqual(admin.call("/api/cameras")["cameras"][0]["zone"], zone)
        self.assertEqual(admin.status_of("/api/cameras", {"cameras": [dict(listed[0], zone=[[0, 0], [2, 2], [0, 1]])]},
                                         method="PUT"), 400)
        camera_id = stored["id"]
        self.assertEqual(admin.status_of(f"/api/cameras/{camera_id}/snapshot"), 404)
        self.server.update_frame(b"\xff\xd8snapshot", source=f"camera:{camera_id}")
        response, body = admin.call(f"/api/cameras/{camera_id}/snapshot", raw=True)
        self.assertEqual(body, b"\xff\xd8snapshot")

    def test_alert_secrets_are_never_returned(self):
        admin = self.admin()
        shown = admin.call("/api/settings/alerts", {"enabled": True, "webhook_url": "https://hooks.example.test/x",
                                                    "webhook_token": "tok-123"}, method="PUT")
        self.assertEqual(shown["webhook_token"], "")
        self.assertTrue(shown["webhook_token_set"])
        shown = admin.call("/api/settings/alerts", {"webhook_token": "", "min_interval_seconds": 5}, method="PUT")
        self.assertTrue(shown["webhook_token_set"])
        self.assertEqual(admin.status_of("/api/settings/alerts", {"webhook_url": "ftp://x"}, method="PUT"), 400)
        with patch("alerts.send_webhook") as webhook:
            self.assertEqual(admin.call("/api/alerts/test", {})["result"], "sent:webhook")
            self.assertEqual(webhook.call_args[0][0]["webhook_token"], "tok-123")

    def test_video_cancel_requires_active_job(self):
        admin = self.admin()
        self.assertEqual(admin.status_of("/api/video/cancel", {}), 409)
        self.server.accept_video = True
        request = Request(self.base + "/api/video", data=b"video-bytes", headers={"X-File-Name": "a.mp4"}, method="POST")
        with admin.opener.open(request) as response:
            self.assertEqual(response.status, 202)
        admin.call("/api/video/cancel", {})
        self.assertEqual(self.server.video["state"], "failed")
        self.assertIsNone(self.server.next_video())

    def test_password_change_signs_out_other_sessions(self):
        admin = self.admin()
        other = Client(self.base)
        other.call("/api/login", {"username": "admin", "password": "correct horse battery"})
        self.assertEqual(admin.status_of("/api/account/password", {"current": "bad", "password": "new password 123"}), 403)
        admin.call("/api/account/password", {"current": "correct horse battery", "password": "new password 123"})
        self.assertEqual(other.status_of("/api/events"), 401)
        self.assertEqual(admin.call("/api/session")["user"]["username"], "admin")


class ServerConfigTests(unittest.TestCase):
    def test_refuses_network_listen_without_login(self):
        with self.assertRaises(ValueError):
            DashboardServer(host="0.0.0.0", port=0, require_auth=False)

    def test_environment_bootstraps_first_admin(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {
                "HELMET_ADMIN_USER": "ops", "HELMET_ADMIN_PASSWORD": "bootstrap password"}):
            server = DashboardServer(port=0, storage=Path(temp), require_auth=True)
            try:
                self.assertEqual(auth.authenticate(Path(temp), "ops", "bootstrap password")["role"], "admin")
            finally:
                server.close()

    def test_extra_allowed_host(self):
        with tempfile.TemporaryDirectory() as temp:
            server = DashboardServer(port=0, storage=Path(temp), require_auth=False,
                                     allowed_hosts="helmet.local").start()
            try:
                request = Request(f"http://127.0.0.1:{server.port}/healthz",
                                  headers={"Host": f"helmet.local:{server.port}"})
                with build_opener().open(request) as response:
                    self.assertTrue(json.load(response)["ok"])
            finally:
                server.close()


    def test_https_serves_dashboard_and_secure_session_cookie(self):
        import ssl
        import subprocess
        from install_service import ensure_certificate
        with tempfile.TemporaryDirectory() as temp:
            try:
                cert, key = ensure_certificate(Path(temp), ["192.168.1.8:8003"])
            except (OSError, subprocess.CalledProcessError):
                self.skipTest("openssl unavailable")
            self.assertIn(b"IP Address:192.168.1.8", subprocess.run(
                ["openssl", "x509", "-in", str(cert), "-noout", "-text"], capture_output=True).stdout)
            server = DashboardServer(port=0, storage=Path(temp) / "data", require_auth=True,
                                     tls_cert=cert, tls_key=key).start()
            try:
                self.assertTrue(server.url.startswith("https://"))
                context = ssl.create_default_context(cafile=str(cert))
                context.check_hostname = False
                opener = build_opener(HTTPCookieProcessor(CookieJar()),
                                      __import__("urllib.request").request.HTTPSHandler(context=context))
                base = f"https://127.0.0.1:{server.port}"
                with opener.open(base + "/", timeout=10) as response:
                    self.assertIn(b"/static/app.js", response.read())
                request = Request(base + "/api/setup", data=json.dumps(
                    {"username": "admin", "password": "correct horse battery"}).encode(),
                    headers={"Content-Type": "application/json", "Origin": base})
                with opener.open(request, timeout=10) as response:
                    cookie = response.headers["Set-Cookie"]
                self.assertIn("Secure", cookie)
                with opener.open(base + "/helmet-certificate.crt", timeout=10) as response:
                    body = response.read()
                self.assertTrue(body.startswith(b"-----BEGIN CERTIFICATE-----"))
                self.assertNotIn(b"PRIVATE KEY", body)
                self.assertIn("HttpOnly", cookie)
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
