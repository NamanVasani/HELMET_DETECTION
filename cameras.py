"""CCTV / IP camera sources (RTSP, HTTP MJPEG, or a local device index) read on background threads."""

import re
import secrets
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import cv2

from event_store import get_setting, set_setting

MAX_CAMERAS = 16
NAME_RE = re.compile(r"[\w .,()#/-]{1,60}\Z")


def redact_url(url):
    """Hide credentials embedded in a camera URL before showing or logging it."""
    url = str(url)
    if url.isdigit():
        return f"Local device {url}"
    parts = urlsplit(url)
    if parts.password or parts.username:
        host = parts.hostname or ""
        if parts.port:
            host += f":{parts.port}"
        return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, "", ""))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def validate_camera(camera):
    name = str(camera.get("name", "")).strip()
    url = str(camera.get("url", "")).strip()
    site = str(camera.get("site", "") or "").strip()[:60]
    if not NAME_RE.fullmatch(name):
        raise ValueError("Camera name must be 1-60 letters, numbers, spaces or . , ( ) # / -")
    if name.lower() in {"browser", "video", "camera", "unknown"}:
        raise ValueError("That camera name is reserved")
    if not (url.isdigit() or urlsplit(url).scheme in {"rtsp", "rtsps", "http", "https"}):
        raise ValueError("Camera URL must be rtsp://, http(s)://, or a local device number")
    return {"id": str(camera.get("id") or secrets.token_hex(4)), "name": name, "url": url, "site": site,
            "enabled": bool(camera.get("enabled", True)), "zone": validate_zone(camera.get("zone")),
            "ptz": validate_ptz(camera.get("ptz"))}


def validate_ptz(ptz):
    """Optional ONVIF pan/tilt/zoom login: {host, port, username, password, profile}. Empty means no PTZ."""
    if not isinstance(ptz, dict) or not str(ptz.get("host") or "").strip():
        return None
    host = str(ptz["host"]).strip()
    if not re.fullmatch(r"[A-Za-z0-9.\-:\[\]]{1,100}", host):
        raise ValueError("PTZ host must be an IP address or host name")
    try:
        port = int(ptz.get("port") or 80)
    except (TypeError, ValueError):
        raise ValueError("PTZ port must be a number") from None
    if not 1 <= port <= 65535:
        raise ValueError("PTZ port must be 1-65535")
    return {"host": host, "port": port, "username": str(ptz.get("username") or "")[:100],
            "password": str(ptz.get("password") or "")[:200], "profile": str(ptz.get("profile") or "")[:100]}


def validate_zone(zone):
    """A watch area: 3-20 points as [x, y] fractions of the frame (0-1). Empty/None means the whole frame."""
    if not zone:
        return None
    if not isinstance(zone, list) or not 3 <= len(zone) <= 20:
        raise ValueError("A zone needs 3 to 20 points")
    points = []
    for point in zone:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each zone point must be [x, y]")
        x, y = (float(value) for value in point)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Zone points must lie inside the picture")
        points.append([round(x, 4), round(y, 4)])
    return points


def in_zone(zone, box, width, height):
    """True if the centre of a detection box lies inside the zone (or there is no zone)."""
    if not zone:
        return True
    x, y = (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height
    inside = False
    for (x1, y1), (x2, y2) in zip(zone, zone[1:] + zone[:1], strict=True):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def load_cameras(root):
    return get_setting(root, "cameras", []) or []


def save_cameras(root, cameras):
    cleaned = [validate_camera(camera) for camera in cameras]
    if len(cleaned) > MAX_CAMERAS:
        raise ValueError(f"At most {MAX_CAMERAS} cameras are supported")
    names = [camera["name"].lower() for camera in cleaned]
    if len(set(names)) != len(names):
        raise ValueError("Camera names must be unique")
    set_setting(root, "cameras", cleaned)
    return cleaned


def public_camera(camera, status=None):
    return {"id": camera["id"], "name": camera["name"], "site": camera.get("site", ""),
            "enabled": camera["enabled"], "url": redact_url(camera["url"]), "zone": camera.get("zone"),
            "ptz": ({"host": camera["ptz"]["host"], "port": camera["ptz"]["port"]} if camera.get("ptz") else None),
            **(status or {})}


class CameraReader:
    """Keep only the newest frame from a stream; reconnect with backoff when it drops."""

    def __init__(self, camera, capture_factory=cv2.VideoCapture):
        self.camera = camera
        self.capture_factory = capture_factory
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.consumed = 0
        self.state = "connecting"
        self.error = ""
        self.last_frame_at = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{camera['id']}")

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stopped.set()

    def latest(self):
        """Return a frame not yet consumed, or None."""
        with self.lock:
            if self.frame is None or self.sequence == self.consumed:
                return None
            self.consumed = self.sequence
            return self.frame

    def status(self):
        with self.lock:
            age = None if self.last_frame_at is None else round(time.monotonic() - self.last_frame_at, 1)
            return {"state": self.state, "error": self.error, "last_frame_age": age}

    def _set(self, state, error=""):
        with self.lock:
            self.state, self.error = state, error

    def _run(self):
        backoff = 1
        source = int(self.camera["url"]) if self.camera["url"].isdigit() else self.camera["url"]
        while not self.stopped.is_set():
            self._set("connecting")
            capture = self.capture_factory(source)
            try:
                if not capture.isOpened():
                    raise OSError("Could not open stream")
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._set("live")
                backoff = 1
                failures = 0
                while not self.stopped.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        failures += 1
                        if failures >= 25:
                            raise OSError("Stream stopped sending frames")
                        time.sleep(0.04)
                        continue
                    failures = 0
                    with self.lock:
                        self.frame = frame
                        self.sequence += 1
                        self.last_frame_at = time.monotonic()
            except Exception as exc:  # any decoder/network error: reconnect
                self._set("reconnecting", f"{exc}"[:200])
            finally:
                capture.release()
            self.stopped.wait(backoff)
            backoff = min(backoff * 2, 30)
        self._set("stopped")


class CameraManager:
    """Start/stop readers to match the saved camera list."""

    def __init__(self, root, reader_factory=CameraReader):
        self.root = root
        self.reader_factory = reader_factory
        self.readers = {}
        self.signature = None

    def sync(self, cameras=None):
        cameras = load_cameras(self.root) if cameras is None else cameras
        wanted = {camera["id"]: camera for camera in cameras if camera.get("enabled")}
        signature = sorted((key, value["url"], value["name"], str(value.get("zone")), str(value.get("ptz")))
                           for key, value in wanted.items())
        if signature == self.signature:
            return
        self.signature = signature
        for key, reader in self.readers.items():
            if key in wanted and wanted[key]["url"] == reader.camera["url"]:
                # zone and PTZ edits need no reconnect
                reader.camera = {**reader.camera, "zone": wanted[key].get("zone"), "ptz": wanted[key].get("ptz")}
        for key in list(self.readers):
            reader = self.readers[key]
            if key not in wanted or wanted[key]["url"] != reader.camera["url"] or wanted[key]["name"] != reader.camera["name"]:
                reader.stop()
                del self.readers[key]
        for key, camera in wanted.items():
            if key not in self.readers:
                self.readers[key] = self.reader_factory(camera).start()

    def statuses(self):
        return {key: reader.status() for key, reader in self.readers.items()}

    def stop_all(self):
        for reader in self.readers.values():
            reader.stop()
        self.readers.clear()
        self.signature = None

    def names(self):
        return {key: reader.camera["name"] for key, reader in self.readers.items()}


class CameraHealthMonitor:
    """Turn camera states into offline / back-online messages, ignoring short drops."""

    def __init__(self, grace_seconds=60):
        self.grace = grace_seconds
        self.down_since = {}
        self.reported = set()

    def update(self, statuses, names, now):
        messages = []
        for key, status in statuses.items():
            name = names.get(key, key)
            if status.get("state") == "live":
                self.down_since.pop(key, None)
                if key in self.reported:
                    self.reported.discard(key)
                    messages.append((f"camera:{key}", f"Camera {name} is back online."))
                continue
            since = self.down_since.setdefault(key, now)
            if key not in self.reported and now - since >= self.grace:
                self.reported.add(key)
                reason = status.get("error") or status.get("state") or "no frames"
                messages.append((f"camera:{key}", f"Camera {name} is offline for {int(now - since)} s ({reason})."))
        for key in set(self.down_since) - set(statuses):  # camera removed from settings
            self.down_since.pop(key, None)
            self.reported.discard(key)
        return messages
