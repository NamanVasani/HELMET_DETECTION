from collections import deque
import threading
import time
import traceback
import os
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from dashboard import DashboardServer
from cameras import in_zone
from event_store import TESSERACT, associate_objects, clean_expired_events, save_event

PROJECT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("HELMET_MODEL", next(
    (path for path in (PROJECT / "best.pt", PROJECT / "runs/detect/helmet_v3/weights/best.pt") if path.is_file()),
    PROJECT / "best.pt",
)))
FACE_MODEL_PATH = PROJECT / "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MAX_DISPLAY_ZOOM = 4.0
# Share of the frame height a zoom target fills. Calibrated 2026-11-19 (reports/zoom-calibration-2026-11-19.json):
# heads at 65% were far larger than anything in training (0.30 correct); 20% gives 0.82, riders do best at 45%.
HEAD_ZOOM_FRACTION = 0.20
RIDER_ZOOM_FRACTION = 0.45
MAX_FOCUS_TARGETS = 3
FACE_MISS_GRACE = 4
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
INFERENCE_SIZE = 800
DETECTION_CONFIDENCE = 0.80
ZOOM_CONFIDENCE = 0.50
CONFIRMATION_FRAMES = 3
FOCUS_FRAME_LIMIT = 8
CAPTURE_COOLDOWN = 30
# Run the face locator on every Nth processed frame (1 = every frame).
FACE_EVERY = max(1, int(os.environ.get("HELMET_FACE_EVERY", "2")))
# Heads taller than this share of the frame are too close to judge: the model was trained on riders at road
# distance and mistakes close-up hair for a helmet. 0 disables the guard.
MAX_HEAD_FRACTION = float(os.environ.get("HELMET_MAX_HEAD_FRACTION", "0.45"))


def too_close(detection, frame_height):
    """True if a head box fills more of the frame than the model can judge reliably."""
    if MAX_HEAD_FRACTION <= 0 or detection.get("zoomed"):
        return False
    return (detection["box"][3] - detection["box"][1]) > MAX_HEAD_FRACTION * frame_height


def pick_device(choice=None):
    """Apple GPU when present (2026-11-22: full pipeline 7.1 vs 6.3 fps at 1080p, 3/3 runs, no hangs).

    HELMET_DEVICE=cpu turns it off; any other value (e.g. cuda:0) is passed to ultralytics as is.
    """
    choice = os.environ.get("HELMET_DEVICE", "auto") if choice is None else choice
    if choice in ("", "cpu"):
        return {}
    if choice == "auto":
        try:
            import torch
            return {"device": "mps"} if torch.backends.mps.is_available() else {}
        except Exception:  # noqa: BLE001 - no torch GPU support means CPU
            return {}
    return {"device": choice}


DEVICE = pick_device()


def run_model(call, *args, **kwargs):
    """Run model.track/predict on DEVICE; if the GPU errors, switch to the CPU for good instead of stopping."""
    global DEVICE
    try:
        return call(*args, **kwargs, **DEVICE)
    except RuntimeError as error:
        if not DEVICE:
            raise
        print(f"GPU inference failed ({error}); continuing on the CPU.")
        DEVICE = {}
        return call(*args, **kwargs)
EXPECTED_CLASSES = ["helmet", "no-helmet", "number-plate", "rider", "bike"]
FULL_FRAME_THRESHOLDS = {"helmet": DETECTION_CONFIDENCE, "no-helmet": DETECTION_CONFIDENCE}
# Plates and bikes are context, not violations, so they use a lower bar. Measured 2026-11-22 on the 462 test
# images: at 0.80 no long one-line plate was found (0/33); at 0.35 23/33 long and 67/96 square plates, 80% precise.
OBJECT_CONFIDENCE = {"number-plate": float(os.environ.get("HELMET_PLATE_CONFIDENCE", "0.35")), "bike": 0.35}


def load_thresholds(model_path, path):
    if not path.is_file():
        return {"helmet": DETECTION_CONFIDENCE, "no-helmet": DETECTION_CONFIDENCE}
    calibration = json.loads(path.read_text())
    if calibration["model_sha256"] != model_digest(model_path):
        raise ValueError("Threshold calibration does not match this model; rerun evaluate_model.py")
    thresholds = calibration["full_frame"]
    if set(thresholds) != {"helmet", "no-helmet"} or any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1
        for value in thresholds.values()
    ):
        raise ValueError("Invalid calibrated thresholds")
    return thresholds


class CaptureState:
    """Give zoom targets a bounded turn without controlling other captures."""
    def __init__(self):
        self.current = None
        self.started = 0
        self.recent = []
        self.deferred = []

    def select(self, detections, frame_number, now):
        self.recent = [(item, when) for item, when in self.recent if now - when < CAPTURE_COOLDOWN]
        self.deferred = [(item, when) for item, when in self.deferred if now - when < 5]
        # shortcut: spatial cooldown limits track-ID churn; crossing people still need human review.
        captured = [dict(item, track_id=None) for item, _ in self.recent]
        candidates = zoom_targets(detections, captured + [item for item, _ in self.deferred])
        matched = next((item for item in candidates if self.current and same_focus(item, self.current)), None)
        if matched and frame_number - self.started < FOCUS_FRAME_LIMIT:
            self.current = matched
            return matched
        if matched:
            self.deferred.append((matched, now))
            candidates = zoom_targets(detections, captured + [item for item, _ in self.deferred])
        self.current = candidates[0] if candidates else None
        self.started = frame_number
        return self.current

    def saved(self, now):
        self.recent.append((dict(self.current), now))
        self.current = None


class EventConfirmation:
    """Confirm every visible track independently; zoom selection must not block capture."""
    def __init__(self):
        self.pending = []
        self.recent = []

    def update(self, candidates, now):
        self.recent = [(target, keys, when) for target, keys, when in self.recent if now - when < CAPTURE_COOLDOWN]
        pending, ready = [], []
        for detection, association in sorted(candidates, key=lambda item: item[0]["confidence"], reverse=True):
            target = association.get("rider") or detection
            keys = event_track_keys(detection, association)
            if any(keys & item["keys"] or same_focus(target, item["target"]) for item in pending):
                continue
            previous = next((item for item in self.pending
                             if keys & item["keys"] or same_focus(target, item["target"])), None)
            agreements = previous["agreements"] + 1 if previous and previous["label"] == detection["class_name"] else 1
            pending.append({"target": target, "keys": keys, "label": detection["class_name"], "agreements": agreements})
            if agreements >= CONFIRMATION_FRAMES and not any(
                keys & saved_keys or same_focus(target, item) for item, saved_keys, _ in self.recent
            ):
                ready.append((detection, association))
        self.pending = pending
        return ready

    def saved(self, detection, association, now):
        # shortcut: untracked heads still use overlap; persistent identity needs independent video validation.
        self.recent.append((dict(association.get("rider") or detection), event_track_keys(detection, association), now))


def zoom_targets(detections, captured=()):
    """Keep a face centered when present; otherwise use rider/head detections."""
    targets = []
    for classes in ({"face"}, {"rider"}, {"helmet", "no-helmet"}):
        targets = [item for item in detections if item["class_name"] in classes
                   and not any(same_focus(item, previous) for previous in captured)]
        if targets:
            break
    return sorted(targets, key=lambda item: (item.get("track_id") is None,
                                              item.get("track_id") or 0,
                                              item["box"][0]))[:MAX_FOCUS_TARGETS]


def same_focus(first, second):
    # shortcut: overlap approximates face identity; upgrade to persistent face tracking if movement causes recaptures.
    if first.get("track_id") is not None and second.get("track_id") is not None:
        return first["track_id"] == second["track_id"]
    ax1, ay1, ax2, ay2 = first["box"]
    bx1, by1, bx2, by2 = second["box"]
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    smaller_area = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return smaller_area > 0 and overlap / smaller_area >= 0.5


def focus_target(detections, captured=()):
    targets = zoom_targets(detections, captured)
    return targets[0] if targets else None


def target_key(item):
    if item.get("track_id") is not None:
        return ("track", item["track_id"])
    return ("box", tuple(round(value / 32) for value in item["box"]))


def event_track_keys(event_detection, association, target=None):
    keys = {target_key(event_detection)}
    if association["rider"]:
        keys.add(target_key(association["rider"]))
    if target:
        keys.add(target_key(target))
    return keys


def meets_detection_threshold(detection):
    threshold = ZOOM_CONFIDENCE if detection.get("zoomed") else \
        FULL_FRAME_THRESHOLDS.get(detection.get("class_name"), DETECTION_CONFIDENCE)
    return detection["confidence"] >= threshold


def should_save_detection(detection):
    return meets_detection_threshold(detection)


def should_save_event(detection, association, saved_tracks, target=None):
    if not should_save_detection(detection):
        return False
    return not saved_tracks.intersection(event_track_keys(detection, association, target))


def belongs_to_focus(association, target, detection=None):
    rider = association.get("rider")
    if not target:
        return False
    if rider and target["class_name"] == "rider" and target.get("track_id") is not None:
        return rider.get("track_id") == target["track_id"]
    if rider:
        x1, y1, x2, y2 = rider["box"]
    elif detection:
        x1, y1, x2, y2 = target["box"]
        if target["class_name"] == "face":
            width, height = x2 - x1, y2 - y1
            # Helmet boxes sit above faces; allow a small head-sized region around the focused face.
            x1, x2 = x1 - width * 0.75, x2 + width * 0.75
            y1, y2 = y1 - height * 1.5, y2 + height * 0.4
            dx1, dy1, dx2, dy2 = detection["box"]
            center_x, center_y = (dx1 + dx2) / 2, (dy1 + dy2) / 2
            return x1 <= center_x <= x2 and y1 <= center_y <= y2
        elif target["class_name"] in {"helmet", "no-helmet"}:
            return target_key(detection) == target_key(target)
    else:
        return target["class_name"] == "rider"
    tx1, ty1, tx2, ty2 = target["box"]
    if not rider and target["class_name"] in {"helmet", "no-helmet"}:
        return target_key(detection or target) == target_key(target)
    center_x, center_y = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def visible_detections(detections):
    return [item for item in detections
            if item["class_name"] not in {"helmet", "no-helmet"}
            or meets_detection_threshold(item)]


def zoom_region(frame, detections, target=None):
    """Return an enlarged crop and its full-frame mapping for a selected person."""
    height, width = frame.shape[:2]
    if target is None:
        targets = zoom_targets(detections)
        target = min(targets, key=lambda item:
                      (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1])) if targets else None
    if not target:
        return frame, None

    x1, y1, x2, y2 = target["box"]
    if target["class_name"] == "face":
        y1 -= int((y2 - y1) * 1.2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    box_height = y2 - y1
    if box_height <= 0:
        return frame, None
    fraction = RIDER_ZOOM_FRACTION if target["class_name"] == "rider" else HEAD_ZOOM_FRACTION
    zoom = min(MAX_DISPLAY_ZOOM, max(1.0, fraction * height / max(box_height, 1)))
    if zoom <= 1.05:
        return frame, None

    crop_width, crop_height = int(width / zoom), int(height / zoom)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    left = min(max(center_x - crop_width // 2, 0), width - crop_width)
    top = min(max(center_y - crop_height // 2, 0), height - crop_height)
    crop = frame[top:top + crop_height, left:left + crop_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR), (left, top, zoom)


def adaptive_zoom(frame, detections, target=None):
    """Zoom the preview toward the currently selected detection."""
    return zoom_region(frame, detections, target)[0]


def read_detections(result, model, min_confidence=0.80, offset=(0, 0), scale=1.0,
                    track_id_override=None, locator=False):
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        if confidence < min(min_confidence, OBJECT_CONFIDENCE.get(class_name, 1.0)) and class_name != "rider" and not (
            locator and class_name in {"helmet", "no-helmet"}
        ):
            continue
        coords = [int(value) for value in box.xyxy[0].cpu().tolist()]
        x_offset, y_offset = offset
        coords = [int(coords[0] / scale + x_offset), int(coords[1] / scale + y_offset),
                  int(coords[2] / scale + x_offset), int(coords[3] / scale + y_offset)]
        track_id = (track_id_override if track_id_override is not None else
                    int(box.id[0].item()) if box.id is not None and scale == 1.0 else None)
        detections.append({"class_name": class_name, "confidence": confidence,
                           "track_id": track_id, "box": coords, "zoomed": scale != 1.0})
    return detections


def draw_detections(frame, detections):
    colors = {"helmet": (60, 210, 120), "no-helmet": (60, 60, 240),
              "number-plate": (240, 180, 60), "rider": (255, 210, 40), "bike": (200, 100, 240),
              "face": (255, 120, 40)}
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        color = colors.get(item["class_name"], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = item["class_name"] if item.get("confidence") is None else f"{item['class_name']} {item['confidence']:.2f}"
        cv2.putText(frame, label,
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def detect_faces(frame, detector):
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return [{"class_name": "face", "confidence": float(rest[-1]) if rest else None, "track_id": None,
             "box": [int(x), int(y), int(x + w), int(y + h)], "zoomed": False}
            for x, y, w, h, *rest in faces]


def face_targets(current, previous, missed_frames):
    if current:
        return current, current, 0
    missed_frames += 1
    return (previous if missed_frames <= FACE_MISS_GRACE else []), previous, missed_frames


def reset_tracks(model):
    for tracker in getattr(model.predictor, "trackers", None) or []:
        tracker.reset()


def model_digest(model_path):
    """SHA-256 of a checkpoint file, or of every file inside an exported model directory."""
    model_path = Path(model_path)
    digest = hashlib.sha256()
    if model_path.is_dir():
        for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
            digest.update(path.relative_to(model_path).as_posix().encode())
            digest.update(path.read_bytes())
    else:
        digest.update(model_path.read_bytes())
    return digest.hexdigest()


def load_zoom_threshold(model_path, path):
    """Calibrated enlarged-crop threshold bound to the same checkpoint, else the 0.50 default."""
    if not path.is_file():
        return ZOOM_CONFIDENCE
    calibration = json.loads(path.read_text())
    value = calibration.get("zoom_crop", ZOOM_CONFIDENCE)
    if calibration.get("model_sha256") != model_digest(model_path) or type(value) not in (int, float) \
            or not math.isfinite(value) or not 0 < value < 1:
        return ZOOM_CONFIDENCE
    return float(value)


class SourcePipeline:
    """Detection, confirmation and capture state for one input source (browser, video, or a CCTV camera)."""

    def __init__(self, key, model, face_detector, thresholds, zoom_threshold=ZOOM_CONFIDENCE):
        self.key = key
        self.model = model
        self.face_detector = face_detector
        self.thresholds = thresholds
        self.zoom_threshold = zoom_threshold
        self.frames = 0
        self.recent = deque(maxlen=30)
        self.fps = 0.0
        self.events = 0
        self.ptz, self.ptz_config = None, None
        self.ptz_lock = threading.Lock()
        self.reset()

    def configure_ptz(self, config):
        """Follow distant riders with an ONVIF PTZ camera; None turns it off. Rebuilt only when the login changes."""
        if config == self.ptz_config:
            return
        from ptz import from_config
        self.ptz_config, self.ptz = config, from_config(config)

    def _steer(self, target, width, height):
        """Send a PTZ move in the background: a slow or offline camera must never stall detection."""
        if self.ptz is None or not self.ptz_lock.acquire(blocking=False):
            return
        follower, box, now = self.ptz, list(target["box"]) if target else None, time.monotonic()

        def step():
            try:
                if box is None:
                    follower.idle(now)
                else:
                    follower.update(box, width, height, now)
            finally:
                self.ptz_lock.release()
        threading.Thread(target=step, daemon=True, name=f"ptz-{self.key}").start()

    def reset(self):
        self.captures = CaptureState()
        self.confirmations = EventConfirmation()
        self.last_faces, self.face_misses = [], 0
        self.cached_faces = []
        reset_tracks(self.model)

    def _qualifies(self, detection):
        threshold = self.zoom_threshold if detection.get("zoomed") else \
            self.thresholds.get(detection.get("class_name"), DETECTION_CONFIDENCE)
        return detection["confidence"] >= threshold

    def process(self, frame, source, dashboard, alerts=None, zone=None):
        result = run_model(
            self.model.track,
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.15,
            iou=0.4,
            imgsz=INFERENCE_SIZE,
            classes=[0, 1, 2, 3, 4],
            verbose=False,
        )[0]
        detections = read_detections(result, self.model, locator=True)
        # Faces only steer zoom focus, so refreshing them every FACE_EVERY frames is enough.
        if self.frames % FACE_EVERY == 0:
            self.cached_faces = detect_faces(frame, self.face_detector)
        faces, self.last_faces, self.face_misses = face_targets(
            self.cached_faces, self.last_faces, self.face_misses
        )
        detections.extend(faces)
        target = self.captures.select(detections, self.frames, time.monotonic())
        zoomed, region = zoom_region(frame, detections, target)
        if self.ptz is not None:
            self._steer(target if target and target.get("class_name") != "face" else None,
                        frame.shape[1], frame.shape[0])
        focused_head = any(self._qualifies(item) and belongs_to_focus(
            associate_objects(item, detections), target, item
        ) for item in detections if item["class_name"] in {"helmet", "no-helmet"})
        if target and region and not focused_head:
            left, top, zoom = region
            close_result = run_model(self.model.predict, zoomed, conf=min(self.zoom_threshold, *OBJECT_CONFIDENCE.values()),
                                              imgsz=INFERENCE_SIZE,
                                              classes=[0, 1, 2, 4], verbose=False)[0]
            detections.extend(read_detections(close_result, self.model, min_confidence=self.zoom_threshold,
                                              offset=(left, top), scale=zoom,
                                              track_id_override=target.get("track_id") if target else None))

        eligible = []
        close_heads = 0
        for event_detection in (item for item in detections if item["class_name"] in {"helmet", "no-helmet"}):
            if too_close(event_detection, frame.shape[0]):
                close_heads += 1
                continue
            association = associate_objects(event_detection, detections)
            if self._qualifies(event_detection) and (
                not event_detection.get("zoomed") or belongs_to_focus(association, target, event_detection)
            ) and in_zone(zone, event_detection["box"], frame.shape[1], frame.shape[0]):
                eligible.append((event_detection, association))
        now = time.monotonic()
        saved = []
        for event_detection, association in self.confirmations.update(eligible, now):
            event = save_event(frame, event_detection, detections, association, root=dashboard.storage,
                               confirmation_frames=CONFIRMATION_FRAMES, source=source)
            self.confirmations.saved(event_detection, association, now)
            if self.captures.current and belongs_to_focus(association, target, event_detection):
                self.captures.saved(now)
            self.events += 1
            saved.append(event)
            print(f"Saved {event['category']} {event['id']} from {self.key} — review at {dashboard.url}")
            if alerts is not None:
                folder = Path(dashboard.storage) / event["date"] / event["category"] / event["id"]
                alerts.notify(event, folder / "person.jpg")

        visible = [item for item in detections
                   if item["class_name"] not in {"helmet", "no-helmet"}
                   or (self._qualifies(item) and not too_close(item, frame.shape[0]))]
        # Lightweight result for clients that show their own camera locally (phones): boxes instead of a picture.
        dashboard.update_detections(self.key, {
            "width": int(frame.shape[1]), "height": int(frame.shape[0]),
            "boxes": [{"label": item["class_name"], "box": [int(v) for v in item["box"]],
                       "confidence": None if item.get("confidence") is None else round(float(item["confidence"]), 2)}
                      for item in visible if item["class_name"] != "face"],
            "too_close": bool(close_heads), "zoom": round(float(region[2]), 1) if region else None,
            "saved": len(saved), "at": time.time()})
        annotated = draw_detections(frame.copy(), visible)
        if close_heads:
            cv2.rectangle(annotated, (0, 0), (frame.shape[1], 34), (20, 20, 20), -1)
            cv2.putText(annotated, "Too close to judge - move back (about 1.5-2 m)", (10, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        if zone:
            outline = np.array([[int(x * frame.shape[1]), int(y * frame.shape[0])] for x, y in zone], dtype=np.int32)
            cv2.polylines(annotated, [outline], True, (0, 220, 255), 2)
        preview = adaptive_zoom(annotated, detections, target)
        if region:  # tell the viewer that the distant rider is being enlarged
            label = f"Auto-zoom {region[2]:.1f}x on distant rider"
            size = max(0.6, preview.shape[1] / 1500)  # readable at 720p and 1080p
            thickness = max(2, round(size * 2))
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)
            pad = int(12 * size)
            cv2.rectangle(preview, (0, 0), (text_w + 2 * pad, text_h + 2 * pad), (20, 20, 20), -1)
            cv2.putText(preview, label, (pad, text_h + pad), cv2.FONT_HERSHEY_SIMPLEX, size, (191, 212, 45), thickness)
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            dashboard.update_frame(encoded.tobytes(), source=self.key)
        self.frames += 1
        self.recent.append(time.monotonic())
        if len(self.recent) >= 2 and self.recent[-1] > self.recent[0]:
            self.fps = (len(self.recent) - 1) / (self.recent[-1] - self.recent[0])
        dashboard.update_metrics(self.key, {"fps": round(self.fps, 2), "frames": self.frames, "events": self.events})
        return saved


def run_detector(dashboard, cameras=None, alerts=None):
    cv2.setNumThreads(min(2, os.cpu_count() or 1))
    clean_expired_events(dashboard.storage)
    print(f"Events save locally under: {dashboard.storage}")
    dashboard.update_status("Loading helmet detector.")
    video_cap = None
    video_path = None
    video_source = None
    if not TESSERACT:
        print("OCR is not active; install Tesseract OCR to read plate text.")

    try:
        print("Loading helmet detector; open the website to grant camera access.")
        model = YOLO(MODEL_PATH)
        names = [model.names[index] for index in range(len(model.names))]
        if names != EXPECTED_CLASSES:
            raise ValueError(f"Helmet model classes must be {EXPECTED_CLASSES}; got {names}")
        thresholds = {**FULL_FRAME_THRESHOLDS, **load_thresholds(MODEL_PATH, PROJECT / "thresholds.json")}
        try:
            zoom_threshold = load_zoom_threshold(MODEL_PATH, PROJECT / "thresholds.json")
        except (OSError, ValueError):
            zoom_threshold = ZOOM_CONFIDENCE
        pipelines = {}

        def pipeline(key):
            if key not in pipelines:
                face_detector = cv2.FaceDetectorYN.create(
                    str(FACE_MODEL_PATH), "", (CAMERA_WIDTH, CAMERA_HEIGHT), 0.6, 0.3, 5000,
                )
                # Each source needs its own tracker state, so later sources get their own model instance.
                pipelines[key] = SourcePipeline(key, model if not pipelines else YOLO(MODEL_PATH), face_detector,
                                                thresholds, zoom_threshold)
            return pipelines[key]

        pipeline("browser")
        dashboard.accept_frames = True
        dashboard.accept_video = True
        dashboard.update_status("Detector ready. Waiting for browser camera permission.")
        print("Detector ready. Allow camera access in the website to start live detection.")
        next_sync = 0
        from cameras import CameraHealthMonitor
        health = CameraHealthMonitor()
        while not dashboard.restart_detector.is_set():
            if cameras is not None and time.monotonic() >= next_sync:
                cameras.sync()
                statuses = cameras.statuses()
                dashboard.update_cameras(statuses)
                if alerts is not None and isinstance(statuses, dict):
                    for key, text in health.update(statuses, cameras.names(), time.monotonic()):
                        print(text)
                        alerts.notify_health(key, text)
                next_sync = time.monotonic() + 2
            if video_path is None:
                requested_video = dashboard.next_video()
                if requested_video:
                    dashboard.cancel_video.clear()
                    video_cap = cv2.VideoCapture(requested_video["path"])
                    if video_cap.isOpened():
                        dashboard.update_status("Analyzing uploaded video; live camera is paused.")
                        video_path = requested_video["path"]
                        video_source = {key: requested_video[key] for key in ("filename", "id")}
                        pipeline("video").reset()
                        dashboard.update_video("processing", frames=0, total=int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                        print("Analyzing uploaded video...")
                    else:
                        video_cap.release()
                        video_cap = None
                        Path(requested_video["path"]).unlink(missing_ok=True)
                        dashboard.update_video("failed", error="The uploaded video could not be decoded.")
                        print("Could not open uploaded video.")

            if video_path:
                canceled = dashboard.cancel_video.is_set()
                ret, frame = (False, None) if canceled else video_cap.read()
                if not ret:
                    completed_frames = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    video_cap.release()
                    video_cap = None
                    Path(video_path).unlink(missing_ok=True)
                    video_path = None
                    video_source = None
                    pipeline("video").reset()
                    if canceled:
                        dashboard.cancel_video.clear()
                        dashboard.update_video("failed", error="Video analysis canceled. Events saved so far remain.")
                    else:
                        dashboard.update_video("complete" if completed_frames else "failed",
                                               error="" if completed_frames else "The video contains no decodable frames.")
                    dashboard.update_status("Video analysis complete. Waiting for browser camera frames.")
                    print("Video analysis complete. Live browser camera resumed.")
                    continue
                frame_index = max(0, int(video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                position = video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not math.isfinite(position) or position < 0 or (position == 0 and frame_index > 0):
                    fps = video_cap.get(cv2.CAP_PROP_FPS)
                    position = frame_index / fps if math.isfinite(fps) and fps > 0 else None
                source = dict(video_source, type="video", frame=frame_index,
                              position_seconds=round(position, 3) if position is not None else None)
                pipeline("video").process(frame, source, dashboard, alerts)
                dashboard.update_video("processing", frames=int(video_cap.get(cv2.CAP_PROP_POS_FRAMES)))
                continue

            worked = False
            for key, reader in list(cameras.readers.items()) if cameras is not None else []:
                camera_frame = reader.latest()
                if camera_frame is None:
                    continue
                worked = True
                source = {"type": "rtsp", "camera": reader.camera["name"], "camera_id": key,
                          "site": reader.camera.get("site", "")}
                camera_pipeline = pipeline(f"camera:{key}")
                camera_pipeline.configure_ptz(reader.camera.get("ptz"))
                camera_pipeline.process(camera_frame, source, dashboard, alerts, zone=reader.camera.get("zone"))
            frame_data = dashboard.next_frame(timeout=0 if worked else 0.05 if cameras and cameras.readers else 1)
            if frame_data is None:
                continue
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            dashboard.update_status("Live browser camera and helmet detection are running.")
            pipeline("browser").process(frame, {"type": "camera", "camera": "browser"}, dashboard, alerts)

    finally:
        if video_path:
            dashboard.update_video("failed", error="Analysis was interrupted; saved events remain available.")
        if video_cap is not None:
            video_cap.release()
        if video_path:
            Path(video_path).unlink(missing_ok=True)
        while True:
            queued_video = dashboard.next_video()
            if not queued_video:
                break
            Path(queued_video["path"]).unlink(missing_ok=True)
            dashboard.update_video("failed", error="Queued video canceled during detector restart. Upload it again.")


def main():
    from alerts import AlertDispatcher, DailyReporter
    from cameras import CameraManager

    dashboard = DashboardServer().start()
    dashboard.detector_managed = True
    alerts = AlertDispatcher(dashboard.storage)
    DailyReporter(dashboard.storage).start()
    import backup
    if backup.BACKUP_ENABLED:
        # Back up the storage this server actually uses, into a folder beside it (never a global default).
        destination = Path(os.environ.get("HELMET_BACKUP_DIR") or Path(dashboard.storage).resolve().parent / "backups")
        backup.BackupScheduler(dashboard.storage, destination).start()
    cameras = CameraManager(dashboard.storage)
    dashboard.alerts = alerts
    print(f"Dashboard: {dashboard.url}")
    try:
        while True:
            dashboard.restart_detector.clear()
            dashboard.accept_frames = False
            dashboard.accept_video = False
            try:
                run_detector(dashboard, cameras, alerts)
            except Exception as exc:
                dashboard.accept_frames = False
                dashboard.accept_video = False
                dashboard.update_status(f"Camera detector stopped: {exc}. Retrying in 5 seconds; "
                                        "saved-event review remains available.")
                traceback.print_exc()
                alerts.notify_health("detector", f"Detector stopped: {exc}. It restarts automatically in 5 seconds.")
                dashboard.restart_detector.wait(5)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        cameras.stop_all()
        dashboard.close()


if __name__ == "__main__":
    main()
