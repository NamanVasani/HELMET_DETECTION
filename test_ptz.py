import base64
import threading
import unittest
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ptz import (DIGEST_TYPE, SOAP, TPTZ, TT, WSSE, WSU, OnvifPTZ, PTZError, PTZFollower, from_config,
                 password_digest)

PASSWORD = "s3cret"


def envelope(body):
    return (f'<?xml version="1.0"?><s:Envelope xmlns:s="{SOAP}" xmlns:tt="{TT}" '
            f'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
            f'xmlns:tptz="{TPTZ}"><s:Body>{body}</s:Body></s:Envelope>').encode()


class FakeCamera(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"])).decode()
        server = self.server
        server.requests.append((self.path, body))
        if server.fault:
            self._reply(500, envelope("<s:Fault><s:Code><s:Value>s:Sender</s:Value></s:Code>"
                                      '<s:Reason><s:Text xml:lang="en">Not authorized</s:Text></s:Reason></s:Fault>'))
        elif "GetCapabilities" in body:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            self._reply(200, envelope(f"<tds:GetCapabilitiesResponse><tds:Capabilities>"
                                      f"<tt:Media><tt:XAddr>{base}/cam/media</tt:XAddr></tt:Media>"
                                      f"<tt:PTZ><tt:XAddr>{base}/cam/ptz</tt:XAddr></tt:PTZ>"
                                      f"</tds:Capabilities></tds:GetCapabilitiesResponse>"))
        elif "GetProfiles" in body:
            self._reply(200, envelope('<trt:GetProfilesResponse><trt:Profiles token="main_profile"/>'
                                      '<trt:Profiles token="sub_profile"/></trt:GetProfilesResponse>'))
        else:
            action = next(name for name in ("RelativeMove", "ContinuousMove", "Stop", "GotoHomePosition")
                          if name in body)
            self._reply(200, envelope(f"<tptz:{action}Response/>"))

    def _reply(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/soap+xml")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class FakeCameraTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCamera)
        self.server.requests, self.server.fault = [], False
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = OnvifPTZ("127.0.0.1", self.server.server_address[1], "admin", PASSWORD)

    def bodies(self, path):
        return [body for request_path, body in self.server.requests if request_path == path]

    def test_digest_header_is_valid(self):
        self.client.profiles()
        root = ET.fromstring(self.server.requests[0][1])
        token = root.find(f".//{{{WSSE}}}UsernameToken")
        self.assertEqual(token.find(f"{{{WSSE}}}Username").text, "admin")
        password = token.find(f"{{{WSSE}}}Password")
        self.assertEqual(password.get("Type"), DIGEST_TYPE)
        nonce = base64.b64decode(token.find(f"{{{WSSE}}}Nonce").text)
        created = token.find(f"{{{WSU}}}Created").text
        self.assertEqual(len(nonce), 16)
        self.assertEqual(password.text, password_digest(nonce, created, PASSWORD))
        self.assertNotIn(PASSWORD, self.server.requests[0][1])

    def test_discovers_services_and_first_profile(self):
        self.assertEqual(self.client.profiles(), ["main_profile", "sub_profile"])
        self.assertEqual(self.client.profile_token, "main_profile")
        self.assertEqual([path for path, _ in self.server.requests], ["/onvif/device_service", "/cam/media"])

    def test_relative_move_stop_and_home(self):
        self.client.relative_move(0.2, -0.1, 0.3)
        self.client.stop()
        self.client.goto_home()
        moves = self.bodies("/cam/ptz")
        self.assertEqual(len(moves), 3)
        self.assertIn('<tt:PanTilt x="0.2000" y="-0.1000"/>', moves[0])
        self.assertIn('<tt:Zoom x="0.3000"/>', moves[0])
        self.assertIn("<tptz:ProfileToken>main_profile</tptz:ProfileToken>", moves[0])
        self.assertIn("<tptz:Stop>", moves[1])
        self.assertIn("<tptz:GotoHomePosition>", moves[2])

    def test_soap_fault_raises(self):
        self.server.fault = True
        with self.assertRaisesRegex(PTZError, "Not authorized"):
            OnvifPTZ("127.0.0.1", self.server.server_address[1], "admin", "wrong", profile_token="p").stop()

    def test_unreachable_camera_raises(self):
        port = self.server.server_address[1]
        self.server.shutdown()
        self.server.server_close()
        with self.assertRaises(PTZError):
            OnvifPTZ("127.0.0.1", port, profile_token="p", timeout=1).stop()

    def test_follower_drives_real_client(self):
        follower = PTZFollower(self.client)
        move = follower.update([1500, 100, 1600, 200], 1920, 1080, now=10.0)
        self.assertEqual(move, (0.3, 0.3, 0.3))  # far right, high up, tiny: all clamped to max_step
        self.assertIn('<tt:PanTilt x="0.3000" y="0.3000"/>', self.bodies("/cam/ptz")[0])


class RecordingClient:
    def __init__(self, fail=False):
        self.moves, self.homes, self.fail = [], 0, fail

    def relative_move(self, pan, tilt, zoom):
        if self.fail:
            raise PTZError("offline")
        self.moves.append((pan, tilt, zoom))

    def goto_home(self):
        if self.fail:
            raise PTZError("offline")
        self.homes += 1


class FollowerTest(unittest.TestCase):
    def test_signs_follow_the_target(self):
        follower = PTZFollower(RecordingClient())
        pan, tilt, zoom = follower.command([100, 800, 300, 1000], 1920, 1080)  # left, low, small
        self.assertLess(pan, 0)
        self.assertLess(tilt, 0)
        self.assertGreater(zoom, 0)
        pan, tilt, zoom = follower.command([760, 0, 1160, 1080], 1920, 1080)  # centred but too large
        self.assertEqual((pan, tilt), (0.0, 0.0))
        self.assertLess(zoom, 0)

    def test_dead_zone_sends_nothing(self):
        client = RecordingClient()
        follower = PTZFollower(client)
        box = [860, 297, 1060, 783]  # centred, 45% of frame height
        self.assertIsNone(follower.update(box, 1920, 1080, now=1.0))
        self.assertEqual(client.moves, [])

    def test_rate_limit(self):
        client = RecordingClient()
        follower = PTZFollower(client, min_interval=1.0)
        box = [0, 0, 50, 50]
        self.assertIsNotNone(follower.update(box, 1920, 1080, now=5.0))
        self.assertIsNone(follower.update(box, 1920, 1080, now=5.5))
        self.assertIsNotNone(follower.update(box, 1920, 1080, now=6.1))
        self.assertEqual(len(client.moves), 2)

    def test_errors_are_swallowed_and_counted(self):
        follower = PTZFollower(RecordingClient(fail=True))
        self.assertIsNone(follower.update([0, 0, 50, 50], 1920, 1080, now=1.0))
        self.assertFalse(follower.idle(now=30.0))
        self.assertEqual(follower.errors, 2)
        self.assertEqual(follower.last_error, "offline")

    def test_idle_goes_home_once(self):
        client = RecordingClient()
        follower = PTZFollower(client)
        self.assertFalse(follower.idle(now=100.0))  # never had a target: stay put
        follower.update([0, 0, 50, 50], 1920, 1080, now=1.0)
        self.assertFalse(follower.idle(now=10.0, after=20))
        self.assertTrue(follower.idle(now=21.5, after=20))
        self.assertFalse(follower.idle(now=40.0, after=20))
        self.assertEqual(client.homes, 1)

    def test_from_config(self):
        self.assertIsNone(from_config(None))
        self.assertIsNone(from_config({"port": 80}))
        follower = from_config({"host": "10.0.0.5", "port": 8000, "username": "u", "password": "p",
                                "profile": "prof1"}, min_interval=2.0)
        self.assertEqual(follower.client.base, "http://10.0.0.5:8000")
        self.assertEqual(follower.client.profile_token, "prof1")
        self.assertEqual(follower.min_interval, 2.0)


if __name__ == "__main__":
    unittest.main()
