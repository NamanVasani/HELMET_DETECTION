"""Helmet review dashboard: FastAPI app with login, review, export, cameras, alerts and live preview."""

import argparse
import asyncio
from datetime import date, timedelta
import hashlib
import ipaddress
import json
import logging
import os
from pathlib import Path
import queue
import re
import socket
import tempfile
import threading
import time
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
import uvicorn

import auth
from event_store import (STORAGE, RETENTION_DAYS, KEEP_CONFIRMED, PRIVACY_MODE, bulk_apply, clean_expired_events,
                         delete_event, event_folder, event_stats, export_csv, export_zip, list_audit,
                         query_events, record_audit, update_review)

PROJECT = Path(__file__).resolve().parent
STATIC = PROJECT / "static"
DEFAULT_PORT = 8003
SESSION_COOKIE = "helmet_session"
MAX_VIDEO_BYTES = 250 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CODE_FILES = ("hi.py", "dashboard.py", "event_store.py", "auth.py", "alerts.py", "cameras.py", "reports_pdf.py",
              "static/index.html", "static/app.js", "static/app.css")
CSP = ("default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; "
       "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
STATIC_FILES = {"app.js": "text/javascript; charset=utf-8", "app.css": "text/css; charset=utf-8",
                "icon.svg": "image/svg+xml"}


def code_version(project=PROJECT):
    """Short hash of the runtime source files, used to detect a stale installed service."""
    digest = hashlib.sha256()
    for name in CODE_FILES:
        path = Path(project) / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


class HostOriginMiddleware:
    """Reject foreign Host headers (DNS rebinding) and cross-site Origins; add security headers."""

    def __init__(self, app, owner):
        self.app = app
        self.owner = owner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope["headers"]}
        host = headers.get("host", "").lower()
        origin = headers.get("origin")
        allowed = self.owner.allowed_hosts()
        if host not in allowed or (origin is not None and origin not in {f"http://{host}", f"https://{host}"}):
            response = JSONResponse({"error": "Use this dashboard from its configured address"}, status_code=403)
            return await response(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                extra = [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                         (b"referrer-policy", b"no-referrer"), (b"content-security-policy", CSP.encode()),
                         (b"permissions-policy", b"camera=(self), microphone=(), geolocation=()")]
                message["headers"] = list(message.get("headers", [])) + extra
            await send(message)

        await self.app(scope, receive, send_with_headers)


class DashboardServer:
    def __init__(self, host=None, port=DEFAULT_PORT, storage=STORAGE, accept_video=False, accept_frames=False,
                 require_auth=None, allowed_hosts=None):
        host = host or os.environ.get("HELMET_HOST", "127.0.0.1")
        self.host = host
        self.storage = Path(storage)
        self.require_auth = (os.environ.get("HELMET_AUTH", "1") != "0") if require_auth is None else require_auth
        if not _is_loopback(host) and not self.require_auth:
            raise ValueError("Refusing to listen beyond this computer without login (set HELMET_AUTH=1)")
        extra_hosts = allowed_hosts if allowed_hosts is not None else os.environ.get("HELMET_ALLOWED_HOSTS", "")
        self.extra_hosts = {item.strip().lower() for item in (extra_hosts.split(",") if isinstance(extra_hosts, str)
                                                             else extra_hosts) if item.strip()}
        self.frame_lock = threading.Lock()
        self.frames = {}
        self.metrics = {}
        self.camera_status = {}
        self.status = ("Waiting for browser camera permission." if accept_frames
                       else "Saved-event review only. Camera detector is not running.")
        self.accept_video = accept_video
        self.video_queue = queue.Queue(maxsize=1)
        self.frame_queue = queue.Queue(maxsize=1)
        self.accept_frames = accept_frames
        self.restart_detector = threading.Event()
        self.cancel_video = threading.Event()
        self.detector_managed = False
        self.alerts = None
        self.video = {"state": "idle", "frames": 0, "total": 0, "error": ""}
        self.cleanup_after = time.monotonic() + 3600
        self.version = code_version()
        self.login_failures = {}
        self.setup_lock = threading.Lock()
        self.stopping = threading.Event()

        self.socket = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.bind((host, port))
        except OSError:
            self.socket.close()
            raise
        self.socket.listen(128)
        self.port = self.socket.getsockname()[1]
        self.app = self._build_app()
        config = uvicorn.Config(HostOriginMiddleware(self.app, self), log_level="warning", access_log=False,
                                lifespan="off", timeout_graceful_shutdown=1, proxy_headers=False, ws="none")
        self.uvicorn = uvicorn.Server(config)
        self.thread = threading.Thread(target=self._serve, daemon=True, name="helmet-http")
        self.maintenance = threading.Thread(target=self._maintain, daemon=True, name="helmet-retention")
        if self.require_auth:
            self._bootstrap_admin()

    # ------------------------------------------------------------------ lifecycle

    @property
    def url(self):
        shown = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{shown}:{self.port}"

    def allowed_hosts(self):
        hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}"}
        for item in self.extra_hosts:
            hosts.add(item if ":" in item.rsplit("]", 1)[-1] else f"{item}:{self.port}")
        return hosts

    def _serve(self):
        try:
            self.uvicorn.run(sockets=[self.socket])
        except Exception:
            logging.exception("Dashboard HTTP server stopped")

    def _maintain(self):
        while not self.stopping.wait(60):
            self.cleanup_expired()

    def start(self):
        self.thread.start()
        self.maintenance.start()
        deadline = time.monotonic() + 10
        while not self.uvicorn.started and self.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        return self

    def close(self):
        self.stopping.set()
        if self.thread.is_alive():
            self.uvicorn.should_exit = True
            self.thread.join(3)
            if self.thread.is_alive():
                self.uvicorn.force_exit = True
                self.thread.join(2)
        self.socket.close()

    def _bootstrap_admin(self):
        username = os.environ.get("HELMET_ADMIN_USER")
        password = os.environ.get("HELMET_ADMIN_PASSWORD")
        if username and password and auth.user_count(self.storage) == 0:
            auth.create_user(self.storage, username, password, "admin")
            record_audit(self.storage, "system", "bootstrap_admin", username)

    def cleanup_expired(self):
        if time.monotonic() < self.cleanup_after:
            return
        self.cleanup_after = time.monotonic() + 3600
        try:
            clean_expired_events(self.storage)
        except Exception:
            logging.exception("Event cleanup failed; it will retry in one hour")

    # ------------------------------------------------------------------ detector interface

    def update_frame(self, frame, source="browser"):
        with self.frame_lock:
            self.frames[source] = frame
            if source in {"browser", "video"}:
                self.frames["live"] = frame

    @property
    def frame(self):
        with self.frame_lock:
            return self.frames.get("live")

    def update_input_frame(self, frame):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def next_frame(self, timeout=1):
        try:
            if not timeout:
                return self.frame_queue.get_nowait()
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def update_status(self, status):
        with self.frame_lock:
            self.status = status

    def update_metrics(self, source, values):
        with self.frame_lock:
            self.metrics[source] = values

    def update_cameras(self, statuses):
        with self.frame_lock:
            self.camera_status = dict(statuses)

    def update_video(self, state, **values):
        with self.frame_lock:
            self.video.update(state=state, **values)

    def next_video(self):
        try:
            return self.video_queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------ HTTP app

    def _build_app(self):
        owner = self
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        def current_user(request):
            if not owner.require_auth:
                return {"username": "local", "role": "admin"}
            return auth.session_user(owner.storage, request.cookies.get(SESSION_COOKIE))

        def require(request, permission):
            user = current_user(request)
            if not user:
                raise HTTPException(401, "Sign in to continue")
            if not auth.allowed(user, permission):
                raise HTTPException(403, "Your role does not allow this")
            return user

        def audit(user, action, target="", details=None):
            record_audit(owner.storage, user["username"], action, target, details)

        async def read_json(request, limit=4096):
            length = int(request.headers.get("content-length") or 0)
            if not 0 < length <= limit:
                raise HTTPException(400, "invalid request size")
            body = await request.body()
            if len(body) > limit:
                raise HTTPException(400, "invalid request size")
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise HTTPException(400, "Request body must be JSON") from exc
            if not isinstance(data, dict):
                raise HTTPException(400, "Request body must be a JSON object")
            return data

        def filters(request):
            params = request.query_params
            values = {key: params.get(key) or None for key in ("start", "end", "category", "status", "q", "camera")}
            if params.get("date"):
                values["start"] = values["end"] = params["date"]
            return values

        @app.exception_handler(HTTPException)
        async def http_error(_request, exc):
            return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

        @app.exception_handler(ValueError)
        async def value_error(_request, exc):
            return JSONResponse({"error": str(exc)}, status_code=400)

        @app.exception_handler(FileNotFoundError)
        async def not_found(_request, _exc):
            return JSONResponse({"error": "Not found"}, status_code=404)

        # -------------------------------------------------------------- pages and static files

        @app.get("/")
        def index():
            return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8",
                                headers={"Cache-Control": "no-cache"})

        @app.get("/static/{name}")
        def static(name: str):
            if name not in STATIC_FILES:
                raise HTTPException(404, "Not found")
            return FileResponse(STATIC / name, media_type=STATIC_FILES[name], headers={"Cache-Control": "no-cache"})

        @app.get("/healthz")
        def health():
            return {"ok": True, "version": owner.version}

        # -------------------------------------------------------------- session

        @app.get("/api/session")
        def session(request: Request):
            user = current_user(request)
            setup = owner.require_auth and auth.user_count(owner.storage) == 0
            return {"auth_enabled": owner.require_auth, "user": user, "setup_required": setup,
                    "permissions": sorted(auth.PERMISSIONS.get(user["role"], ())) if user else []}

        def login_response(user, request):
            token = auth.create_session(owner.storage, user["username"])
            response = JSONResponse({"user": user})
            response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="strict", path="/",
                                secure=request.url.scheme == "https", max_age=auth.SESSION_HOURS * 3600)
            return response

        @app.post("/api/setup")
        async def setup(request: Request):
            if not owner.require_auth or auth.user_count(owner.storage):
                raise HTTPException(409, "Setup is already complete")
            if not _is_loopback(request.client.host if request.client else ""):
                raise HTTPException(403, "Create the first admin from the computer running Helmet")
            data = await read_json(request)
            with owner.setup_lock:
                if auth.user_count(owner.storage):
                    raise HTTPException(409, "Setup is already complete")
                user = auth.create_user(owner.storage, data.get("username"), data.get("password"), "admin")
            audit(user, "setup_admin", user["username"])
            return login_response(user, request)

        @app.post("/api/login")
        async def login(request: Request):
            data = await read_json(request)
            client = request.client.host if request.client else "?"
            failures, until = owner.login_failures.get(client, (0, 0))
            if until > time.monotonic():
                raise HTTPException(429, "Too many failed sign-ins. Wait a minute and retry.")
            user = auth.authenticate(owner.storage, data.get("username", ""), data.get("password", ""))
            if not user:
                failures += 1
                owner.login_failures[client] = (failures, time.monotonic() + 60 if failures >= 5 else 0)
                record_audit(owner.storage, str(data.get("username", ""))[:64], "login_failed", client)
                raise HTTPException(401, "Wrong username or password")
            owner.login_failures.pop(client, None)
            audit(user, "login", client)
            return login_response(user, request)

        @app.post("/api/logout")
        def logout(request: Request):
            auth.end_session(owner.storage, request.cookies.get(SESSION_COOKIE))
            response = JSONResponse({"ok": True})
            response.delete_cookie(SESSION_COOKIE, path="/")
            return response

        # -------------------------------------------------------------- status and events

        @app.get("/api/status")
        def status(request: Request):
            if owner.require_auth:
                require(request, "view")
            with owner.frame_lock:
                data = {"status": owner.status, "video": dict(owner.video), "metrics": dict(owner.metrics),
                        "cameras": dict(owner.camera_status), "sources": sorted(owner.frames)}
            alerts = owner.alerts
            data.update({"camera_enabled": owner.accept_frames, "video_enabled": owner.accept_video,
                         "detector_managed": owner.detector_managed, "version": owner.version,
                         "retention_days": RETENTION_DAYS, "keep_confirmed": KEEP_CONFIRMED,
                         "privacy_mode": PRIVACY_MODE,
                         "alerts": {"sent": alerts.sent, "last_error": alerts.last_error} if alerts else None})
            return data

        @app.get("/api/events")
        def events(request: Request):
            require(request, "view")
            values = filters(request)
            try:
                limit = int(request.query_params.get("limit", 200))
                offset = int(request.query_params.get("offset", 0))
            except ValueError as exc:
                raise HTTPException(400, "invalid paging") from exc
            if not any(values.values()):
                values["start"] = values["end"] = date.today().isoformat()
            try:
                found, total = query_events(owner.storage, limit=limit, offset=offset, **values)
            except ValueError as exc:
                raise HTTPException(400, "invalid date" if "date" in str(exc).lower() else str(exc)) from exc
            return {"events": found, "total": total, "limit": limit, "offset": offset}

        @app.get("/api/stats")
        def stats(request: Request):
            require(request, "view")
            params = request.query_params
            end = params.get("end") or date.today().isoformat()
            start = params.get("start") or (date.fromisoformat(end) - timedelta(days=29)).isoformat()
            return event_stats(owner.storage, start, end, params.get("camera") or None)

        @app.post("/api/review")
        async def review(request: Request):
            user = require(request, "review")
            data = await read_json(request)
            try:
                result = update_review(data["date"], data["id"], data["status"], data.get("plate_text", ""),
                                       owner.storage)
            except (KeyError, TypeError) as exc:
                raise HTTPException(400, f"Missing or invalid field: {exc}") from exc
            audit(user, "review", f"{data['date']}/{data['id']}",
                  {"status": data["status"], "plate_text": result["plate_text"]})
            return result

        @app.post("/api/delete")
        async def delete(request: Request):
            user = require(request, "delete")
            data = await read_json(request)
            try:
                removed = delete_event(data["date"], data["id"], owner.storage)
            except (KeyError, TypeError) as exc:
                raise HTTPException(400, f"Missing or invalid field: {exc}") from exc
            audit(user, "delete", f"{data['date']}/{data['id']}")
            return {"removed": removed}

        @app.post("/api/bulk")
        async def bulk(request: Request):
            data = await read_json(request, limit=200_000)
            action = data.get("action")
            user = require(request, "delete" if action == "delete" else "review")
            status_value = data.get("status")
            if action == "review" and status_value not in ("needs_review", "confirmed", "false_positive"):
                raise HTTPException(400, "Invalid review status")
            results = bulk_apply(data.get("items"), action, owner.storage, status=status_value)
            audit(user, f"bulk_{action}", f"{sum(item['ok'] for item in results)} events",
                  {"status": status_value, "failed": sum(not item["ok"] for item in results)})
            return {"results": results}

        def export_events(request):
            values = filters(request)
            if not any(values.values()):
                values["start"] = values["end"] = date.today().isoformat()
            found, _total = query_events(owner.storage, limit=5000, **values)
            return found, values

        def export_name(values, suffix):
            label = "-".join(str(values[key]) for key in ("start", "end") if values.get(key)) or "all"
            return re.sub(r"[^A-Za-z0-9_.-]", "_", f"helmet-events-{label}.{suffix}")

        @app.get("/api/export.csv")
        def export_csv_route(request: Request):
            user = require(request, "export")
            found, values = export_events(request)
            audit(user, "export_csv", f"{len(found)} events", values)
            return Response(export_csv(found), media_type="text/csv; charset=utf-8", headers={
                "Content-Disposition": f'attachment; filename="{export_name(values, "csv")}"'})

        @app.get("/api/export.zip")
        def export_zip_route(request: Request):
            user = require(request, "export")
            found, values = export_events(request)
            audit(user, "export_zip", f"{len(found)} events", values)
            spooled = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024)
            export_zip(found, owner.storage, spooled)
            spooled.seek(0)

            def chunks():
                try:
                    while chunk := spooled.read(1024 * 1024):
                        yield chunk
                finally:
                    spooled.close()

            return StreamingResponse(chunks(), media_type="application/zip", headers={
                "Content-Disposition": f'attachment; filename="{export_name(values, "zip")}"'})

        @app.get("/api/report.pdf")
        def report(request: Request):
            user = require(request, "export")
            from reports_pdf import build_report
            params = request.query_params
            end = params.get("end") or date.today().isoformat()
            start = params.get("start") or end
            camera = params.get("camera") or None
            pdf = build_report(owner.storage, start, end, camera)
            audit(user, "report_pdf", f"{start}..{end}", {"camera": camera})
            return Response(pdf, media_type="application/pdf", headers={
                "Content-Disposition": f'attachment; filename="helmet-report-{start}-{end}.pdf"'})

        @app.get("/media/{day}/{category}/{event_id}/{name}")
        def media(request: Request, day: str, category: str, event_id: str, name: str):
            require(request, "view")
            if not (re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) and category in {"helmet", "no_helmet"}
                    and re.fullmatch(r"event_[\w-]{1,80}", event_id) and name in {"person.jpg", "plate.jpg", "event.jpg"}):
                raise HTTPException(404, "Not found")
            try:
                folder = event_folder(owner.storage, day, event_id, category)
            except ValueError as exc:
                raise HTTPException(404, "Not found") from exc
            path = folder / name
            if path.is_symlink() or not path.is_file():
                raise HTTPException(404, "Not found")
            return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

        @app.get("/stream")
        async def stream(request: Request):
            require(request, "view")
            source = request.query_params.get("source", "live")
            with owner.frame_lock:
                available = source in owner.frames
            if not available and not owner.accept_frames:
                raise HTTPException(503, "No live detector is running")

            async def frames():
                last, sent_at = None, 0.0
                while not owner.stopping.is_set():
                    with owner.frame_lock:
                        frame = owner.frames.get(source)
                    # Send new frames immediately and repeat the latest one twice a second as a keep-alive.
                    if frame is not None and (frame is not last or time.monotonic() - sent_at >= 0.5):
                        last, sent_at = frame, time.monotonic()
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode()
                               + b"\r\n\r\n" + frame + b"\r\n")
                    await asyncio.sleep(0.08)

            return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                     headers={"Cache-Control": "no-store"})

        # -------------------------------------------------------------- live input

        @app.post("/api/detector/restart")
        def restart(request: Request):
            user = require(request, "camera")
            if not owner.detector_managed:
                raise HTTPException(503, "Run hi.py to manage the detector")
            owner.restart_detector.set()
            audit(user, "detector_restart")
            return JSONResponse({"restarting": True}, status_code=202)

        @app.post("/api/frame")
        async def frame(request: Request):
            require(request, "camera")
            if not owner.accept_frames:
                raise HTTPException(503, "Camera detector is not running")
            length = int(request.headers.get("content-length") or 0)
            content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type != "image/jpeg" or not 0 < length <= 5 * 1024 * 1024:
                return JSONResponse({"error": "Invalid camera frame"}, status_code=400)
            body = await request.body()
            if len(body) != length or not body.startswith(b"\xff\xd8"):
                return JSONResponse({"error": "Invalid JPEG frame"}, status_code=400)
            owner.update_input_frame(body)
            return JSONResponse({"accepted": True}, status_code=202)

        @app.post("/api/video")
        async def video(request: Request):
            user = require(request, "video")
            if not owner.accept_video:
                raise HTTPException(503, "Video analysis needs the detector (run hi.py)")
            with owner.frame_lock:
                busy = owner.video["state"] in {"uploading", "queued", "processing"}
                if not busy:
                    owner.video = {"state": "uploading", "frames": 0, "total": 0, "error": ""}
            if busy:
                return JSONResponse({"error": "Wait for the current video to finish"}, status_code=409)
            temporary = None
            try:
                length = int(request.headers.get("content-length") or 0)
                filename = unquote(request.headers.get("x-file-name", ""))
                suffix = Path(filename).suffix.lower()
                if not 0 < length <= MAX_VIDEO_BYTES:
                    raise ValueError("Video must be no larger than 250 MB")
                if suffix not in VIDEO_SUFFIXES:
                    raise ValueError("Unsupported video format")
                temporary = tempfile.NamedTemporaryFile(prefix="helmet_video_", suffix=suffix, delete=False)
                received = 0
                try:
                    async for chunk in request.stream():
                        received += len(chunk)
                        if received > length:
                            raise ValueError("Upload is larger than declared")
                        temporary.write(chunk)
                finally:
                    temporary.close()
                if received != length:
                    raise ValueError("Incomplete video upload")
                with owner.frame_lock:
                    owner.video_queue.put_nowait({"path": temporary.name,
                                                  "filename": Path(filename.replace("\\", "/")).name[:255],
                                                  "id": Path(temporary.name).stem})
                    owner.video = {"state": "queued", "frames": 0, "total": 0, "error": ""}
            except queue.Full:
                Path(temporary.name).unlink(missing_ok=True)
                owner.update_video("failed", error="A video is already queued")
                return JSONResponse({"error": "A video is already queued"}, status_code=409)
            except (ValueError, OSError) as exc:
                if temporary is not None:
                    Path(temporary.name).unlink(missing_ok=True)
                owner.update_video("failed", error=str(exc))
                return JSONResponse({"error": str(exc)}, status_code=400)
            audit(user, "video_upload", Path(filename.replace("\\", "/")).name[:255], {"bytes": length})
            return JSONResponse({"queued": True}, status_code=202)

        @app.post("/api/video/cancel")
        def video_cancel(request: Request):
            user = require(request, "video")
            with owner.frame_lock:
                state = owner.video["state"]
            if state not in {"queued", "processing"}:
                raise HTTPException(409, "No video analysis is running")
            if state == "queued":
                job = owner.next_video()
                if job:
                    Path(job["path"]).unlink(missing_ok=True)
                owner.update_video("failed", error="Video analysis canceled.")
            else:
                owner.cancel_video.set()
            audit(user, "video_cancel")
            return JSONResponse({"canceling": True}, status_code=202)

        # -------------------------------------------------------------- cameras, alerts, users, audit

        @app.get("/api/cameras")
        def cameras_list(request: Request):
            require(request, "view")
            from cameras import load_cameras, public_camera
            with owner.frame_lock:
                statuses = dict(owner.camera_status)
                metrics = dict(owner.metrics)
            return {"cameras": [public_camera(camera, {**statuses.get(camera["id"], {}),
                                                       **metrics.get(f"camera:{camera['id']}", {})})
                                for camera in load_cameras(owner.storage)]}

        @app.put("/api/cameras")
        async def cameras_save(request: Request):
            user = require(request, "admin")
            from cameras import load_cameras, public_camera, save_cameras
            data = await read_json(request, limit=20_000)
            existing = {camera["id"]: camera for camera in load_cameras(owner.storage)}
            incoming = []
            for camera in data.get("cameras", []):
                if not isinstance(camera, dict):
                    raise HTTPException(400, "Invalid camera")
                previous = existing.get(camera.get("id"))
                # The browser only sees redacted URLs; keep the stored URL when it was not edited.
                if previous and (not camera.get("url") or camera.get("url") == public_camera(previous)["url"]):
                    camera = {**camera, "url": previous["url"]}
                incoming.append(camera)
            saved = save_cameras(owner.storage, incoming)
            audit(user, "cameras_saved", f"{len(saved)} cameras", {"names": [camera["name"] for camera in saved]})
            return {"cameras": [public_camera(camera) for camera in saved]}

        @app.get("/api/settings/alerts")
        def alerts_get(request: Request):
            require(request, "admin")
            from alerts import load_settings, public_settings
            return public_settings(load_settings(owner.storage))

        @app.put("/api/settings/alerts")
        async def alerts_put(request: Request):
            user = require(request, "admin")
            from alerts import public_settings, save_settings
            data = await read_json(request, limit=20_000)
            saved = save_settings(owner.storage, data)
            audit(user, "alerts_saved", "", {"enabled": saved["enabled"], "webhook": bool(saved["webhook_url"]),
                                             "email": saved["email_enabled"]})
            return public_settings(saved)

        @app.post("/api/alerts/test")
        def alerts_test(request: Request):
            user = require(request, "admin")
            from alerts import AlertDispatcher
            dispatcher = owner.alerts or AlertDispatcher(owner.storage)
            now = date.today().isoformat()
            sample = {"id": "event_test", "date": now, "timestamp": now + "T00:00:00", "category": "no_helmet",
                      "plate_text": "TEST", "event_detection": {"confidence": 0.99},
                      "source": {"type": "test", "camera": "Test alert"}}
            try:
                result = dispatcher.deliver(sample, force=True)
            except Exception as exc:
                audit(user, "alert_test_failed", "", {"error": str(exc)[:300]})
                raise HTTPException(502, f"Alert failed: {exc}") from exc
            audit(user, "alert_test", "", {"result": result})
            return {"result": result}

        @app.get("/api/users")
        def users_list(request: Request):
            require(request, "admin")
            return {"users": auth.list_users(owner.storage), "roles": list(auth.ROLES)}

        @app.post("/api/users")
        async def users_create(request: Request):
            user = require(request, "admin")
            data = await read_json(request)
            created = auth.create_user(owner.storage, data.get("username"), data.get("password"),
                                       data.get("role", "reviewer"))
            audit(user, "user_created", created["username"], {"role": created["role"]})
            return created

        @app.patch("/api/users/{username}")
        async def users_update(request: Request, username: str):
            user = require(request, "admin")
            data = await read_json(request)
            updated = auth.update_user(owner.storage, username, data.get("role") or None, data.get("password") or None)
            audit(user, "user_updated", username, {"role": data.get("role"), "password_changed": bool(data.get("password"))})
            return updated

        @app.delete("/api/users/{username}")
        def users_delete(request: Request, username: str):
            user = require(request, "admin")
            if username == user["username"]:
                raise HTTPException(400, "You cannot delete your own account")
            auth.delete_user(owner.storage, username)
            audit(user, "user_deleted", username)
            return {"removed": True}

        @app.post("/api/account/password")
        async def change_password(request: Request):
            user = require(request, "view")
            data = await read_json(request)
            if not auth.authenticate(owner.storage, user["username"], data.get("current", "")):
                raise HTTPException(403, "Current password is wrong")
            auth.update_user(owner.storage, user["username"], password=data.get("password"))
            audit(user, "password_changed", user["username"])
            return login_response(user, request)

        @app.get("/api/audit")
        def audit_list(request: Request):
            require(request, "admin")
            return {"entries": list_audit(owner.storage, int(request.query_params.get("limit", 200)))}

        return app


def main():
    parser = argparse.ArgumentParser(description="Run the helmet event dashboard without the camera detector")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    clean_expired_events()
    server = DashboardServer(port=args.port).start()
    print(f"Event review dashboard: {server.url}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.close()


if __name__ == "__main__":
    main()
