"""Optional ONVIF pan/tilt/zoom control: turn and zoom a PTZ camera toward a distant rider.

Standard library only. A camera entry may carry {"ptz": {"host", "port", "username", "password", "profile"}};
PTZFollower never raises into the detection loop, so a missing or offline PTZ head only costs a counter.
"""
import base64
import hashlib
import os
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

SOAP = "http://www.w3.org/2003/05/soap-envelope"
WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
DIGEST_TYPE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
NONCE_TYPE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary"
TDS = "http://www.onvif.org/ver10/device/wsdl"
TRT = "http://www.onvif.org/ver10/media/wsdl"
TPTZ = "http://www.onvif.org/ver20/ptz/wsdl"
TT = "http://www.onvif.org/ver10/schema"


class PTZError(Exception):
    """The camera refused or could not be reached."""


def password_digest(nonce, created, password):
    """WS-UsernameToken digest: Base64(SHA1(nonce + created + password))."""
    return base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()


def _clamp(value, limit):
    return max(-limit, min(limit, value))


def _num(value):
    return f"{float(value):.4f}"


class OnvifPTZ:
    def __init__(self, host, port=80, username="", password="", profile_token=None, timeout=3):
        self.host, self.port = host, int(port)
        self.username, self.password = username, password
        self.profile_token = profile_token
        self.timeout = timeout
        self.base = f"http://{host}:{self.port}"
        self._services = None

    # ------------------------------------------------------------------ transport
    def _security(self):
        if not self.username:
            return ""
        nonce = os.urandom(16)
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return (
            f'<s:Header><wsse:Security xmlns:wsse="{WSSE}" xmlns:wsu="{WSU}" s:mustUnderstand="1">'
            f"<wsse:UsernameToken><wsse:Username>{_escape(self.username)}</wsse:Username>"
            f'<wsse:Password Type="{DIGEST_TYPE}">{password_digest(nonce, created, self.password)}</wsse:Password>'
            f'<wsse:Nonce EncodingType="{NONCE_TYPE}">{base64.b64encode(nonce).decode()}</wsse:Nonce>'
            f"<wsu:Created>{created}</wsu:Created></wsse:UsernameToken></wsse:Security></s:Header>"
        )

    def _call(self, url, body):
        envelope = (f'<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="{SOAP}" '
                    f'xmlns:tds="{TDS}" xmlns:trt="{TRT}" xmlns:tptz="{TPTZ}" xmlns:tt="{TT}">'
                    f"{self._security()}<s:Body>{body}</s:Body></s:Envelope>").encode()
        request = urllib.request.Request(url, data=envelope, method="POST",
                                         headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            raise PTZError(f"HTTP {error.code}: {_fault_reason(error.read()) or error.reason}") from None
        except (urllib.error.URLError, OSError) as error:
            raise PTZError(f"camera unreachable: {error}") from None
        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            raise PTZError("camera sent an invalid reply") from None
        reason = _fault_reason(payload, root)
        if reason is not None:
            raise PTZError(f"SOAP fault: {reason}")
        return root

    # ------------------------------------------------------------------ discovery
    def services(self):
        if self._services is None:
            services = {"media": f"{self.base}/onvif/media_service", "ptz": f"{self.base}/onvif/ptz_service"}
            try:
                root = self._call(f"{self.base}/onvif/device_service",
                                  '<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>')
                for key, tag in (("media", "Media"), ("ptz", "PTZ")):
                    found = root.find(f".//{{{TT}}}{tag}/{{{TT}}}XAddr")
                    if found is not None and (found.text or "").strip():
                        services[key] = found.text.strip()
            except PTZError:
                pass  # older cameras: fall back to the conventional paths
            self._services = services
        return self._services

    def profiles(self):
        root = self._call(self.services()["media"], "<trt:GetProfiles/>")
        tokens = [item.get("token") for item in root.iter(f"{{{TRT}}}Profiles") if item.get("token")]
        if not tokens:
            raise PTZError("camera reported no media profiles")
        if self.profile_token is None:
            self.profile_token = tokens[0]
        return tokens

    def _token(self):
        if self.profile_token is None:
            self.profiles()
        return f"<tptz:ProfileToken>{_escape(self.profile_token)}</tptz:ProfileToken>"

    # ------------------------------------------------------------------ moves
    def relative_move(self, pan=0.0, tilt=0.0, zoom=0.0):
        self._call(self.services()["ptz"],
                   f"<tptz:RelativeMove>{self._token()}<tptz:Translation>"
                   f'<tt:PanTilt x="{_num(pan)}" y="{_num(tilt)}"/><tt:Zoom x="{_num(zoom)}"/>'
                   f"</tptz:Translation></tptz:RelativeMove>")

    def continuous_move(self, pan=0.0, tilt=0.0, zoom=0.0):
        self._call(self.services()["ptz"],
                   f"<tptz:ContinuousMove>{self._token()}<tptz:Velocity>"
                   f'<tt:PanTilt x="{_num(pan)}" y="{_num(tilt)}"/><tt:Zoom x="{_num(zoom)}"/>'
                   f"</tptz:Velocity></tptz:ContinuousMove>")

    def stop(self):
        self._call(self.services()["ptz"],
                   f"<tptz:Stop>{self._token()}<tptz:PanTilt>true</tptz:PanTilt>"
                   f"<tptz:Zoom>true</tptz:Zoom></tptz:Stop>")

    def goto_home(self):
        self._call(self.services()["ptz"], f"<tptz:GotoHomePosition>{self._token()}</tptz:GotoHomePosition>")


def _escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fault_reason(payload, root=None):
    try:
        root = ET.fromstring(payload) if root is None else root
    except ET.ParseError:
        return None
    fault = root.find(f".//{{{SOAP}}}Fault")
    if fault is None:
        return None
    text = fault.find(f".//{{{SOAP}}}Reason/{{{SOAP}}}Text")
    return (text.text or "").strip() if text is not None else "unknown fault"


class PTZFollower:
    """Turn a PTZ camera toward the current zoom target, gently and never more than once per min_interval."""

    def __init__(self, client, min_interval=1.0, dead_zone=0.15, target_fraction=0.45, max_step=0.3):
        self.client = client
        self.min_interval = min_interval
        self.dead_zone = dead_zone
        self.target_fraction = target_fraction
        self.max_step = max_step
        self.last_sent = None
        self.last_target = None
        self.homed = True
        self.errors = 0
        self.last_error = None

    def command(self, box, frame_width, frame_height):
        x1, y1, x2, y2 = box
        dx = (x1 + x2) / 2 / frame_width - 0.5
        dy = (y1 + y2) / 2 / frame_height - 0.5
        pan = _clamp(dx, self.max_step) if abs(dx) > self.dead_zone else 0.0
        tilt = _clamp(-dy, self.max_step) if abs(dy) > self.dead_zone else 0.0  # image y grows downward
        error = self.target_fraction - (y2 - y1) / frame_height
        zoom = _clamp(error, self.max_step) if abs(error) > self.dead_zone * self.target_fraction else 0.0
        return round(pan, 4), round(tilt, 4), round(zoom, 4)

    def update(self, target_box, frame_width, frame_height, now):
        if target_box is None or frame_width <= 0 or frame_height <= 0:
            return None
        self.last_target = now
        self.homed = False
        if self.last_sent is not None and now - self.last_sent < self.min_interval:
            return None
        move = self.command(target_box, frame_width, frame_height)
        if move == (0.0, 0.0, 0.0):
            return None
        self.last_sent = now
        try:
            self.client.relative_move(*move)
        except Exception as error:  # noqa: BLE001 - a PTZ failure must never stop detection
            self.errors += 1
            self.last_error = str(error)
            return None
        return move

    def idle(self, now, after=20):
        if self.homed or self.last_target is None or now - self.last_target < after:
            return False
        self.homed = True
        try:
            self.client.goto_home()
        except Exception as error:  # noqa: BLE001
            self.errors += 1
            self.last_error = str(error)
            return False
        return True


def from_config(config, **follower_options):
    """Build a PTZFollower from a camera's optional "ptz" dict; None when absent or incomplete."""
    if not isinstance(config, dict) or not config.get("host"):
        return None
    client = OnvifPTZ(config["host"], port=config.get("port", 80), username=config.get("username", ""),
                      password=config.get("password", ""), profile_token=config.get("profile") or None)
    return PTZFollower(client, **follower_options)
