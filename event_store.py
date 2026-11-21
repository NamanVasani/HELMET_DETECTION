"""Local event storage: SQLite index, evidence files, review, retention, export, audit, settings."""

from contextlib import contextmanager
import csv
from datetime import date, datetime, timedelta
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import threading
import zipfile

import cv2

ROOT = Path(__file__).resolve().parent
STORAGE = Path(os.environ.get("HELMET_STORAGE", ROOT / "storage")).resolve()
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
EVENT_RE = re.compile(r"event_[\w-]{1,80}\Z")
EVENT_CATEGORIES = {"helmet", "no_helmet"}
REVIEW_STATUSES = ("needs_review", "confirmed", "false_positive")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# 0 disables automatic deletion. Confirmed events are kept unless HELMET_KEEP_CONFIRMED=0.
RETENTION_DAYS = _env_int("HELMET_RETENTION_DAYS", 30)
KEEP_CONFIRMED = os.environ.get("HELMET_KEEP_CONFIRMED", "1") != "0"
# Privacy mode: blur faces and drop plate evidence for riders who are wearing a helmet.
PRIVACY_MODE = os.environ.get("HELMET_PRIVACY", "1") != "0"
TESSERACT = shutil.which("tesseract") or next(
    (path for path in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract") if Path(path).is_file()),
    None,
)
SCHEMA_VERSION = "2"
_initialized = set()
_init_lock = threading.Lock()


def _center(box):
    x1, y1, x2, y2 = box["box"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def _area(box):
    x1, y1, x2, y2 = box["box"]
    return max(1, x2 - x1) * max(1, y2 - y1)


def _inside(point, box):
    x1, y1, x2, y2 = box["box"]
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def _distance(a, b):
    return math.dist(_center(a), _center(b))


# --------------------------------------------------------------------------- database

def _row_values(data):
    detection = data.get("event_detection") or data.get("helmet") or data.get("no_helmet") or {}
    source = data.get("source") or {}
    return {
        "date": data["date"], "id": data["id"], "timestamp": data["timestamp"],
        "status": data.get("status", "needs_review"),
        "data": json.dumps(data),
        "category": data.get("category") or ("helmet" if detection.get("class_name") == "helmet" else "no_helmet"),
        "camera": str(source.get("camera") or source.get("type") or "unknown"),
        "plate_text": data.get("plate_text") or "",
        "confidence": detection.get("confidence"),
    }


def _write_row(connection, data, replace=True):
    values = _row_values(data)
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    connection.execute(
        f"{verb} INTO events (date, id, timestamp, status, data, category, camera, plate_text, confidence) "
        "VALUES (:date, :id, :timestamp, :status, :data, :category, :camera, :plate_text, :confidence)", values)


def _initialize(connection, root):
    connection.execute("CREATE TABLE IF NOT EXISTS events (date TEXT, id TEXT, timestamp TEXT, status TEXT, "
                       "data TEXT, PRIMARY KEY (date, id))")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
    for name, kind in (("category", "TEXT"), ("camera", "TEXT"), ("plate_text", "TEXT"), ("confidence", "REAL")):
        if name not in columns:
            connection.execute(f"ALTER TABLE events ADD COLUMN {name} {kind}")
    connection.executescript("""
        CREATE INDEX IF NOT EXISTS events_by_time ON events (timestamp);
        CREATE INDEX IF NOT EXISTS events_by_status ON events (status, date);
        CREATE INDEX IF NOT EXISTS events_by_category ON events (category, date);
        CREATE TABLE IF NOT EXISTS audit (at TEXT, user TEXT, action TEXT, target TEXT, details TEXT);
        CREATE INDEX IF NOT EXISTS audit_by_time ON audit (at);
        CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT, created TEXT);
        CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, username TEXT, expires TEXT);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
    """)
    imported = connection.execute("SELECT value FROM metadata WHERE key='json_imported'").fetchone()
    if not imported:
        paths = set(root.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/event_*/event.json"))
        for category in EVENT_CATEGORIES:
            paths.update(root.glob(f"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/{category}/event_*/event.json"))
        for path in paths:
            try:
                _write_row(connection, json.loads(path.read_text(encoding="utf-8")), replace=False)
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
        connection.execute("INSERT INTO metadata VALUES ('json_imported', '1')")
    version = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if not version or version[0] != SCHEMA_VERSION:
        for raw, in connection.execute("SELECT data FROM events WHERE category IS NULL").fetchall():
            try:
                _write_row(connection, json.loads(raw))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
    connection.commit()


@contextmanager
def _database(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "events.db"
    key = str(path.resolve())
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        with _init_lock:
            if key not in _initialized or not path.is_file():
                connection.execute("PRAGMA journal_mode=WAL")
                _initialize(connection, root)
                _initialized.add(key)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _validate_day(day):
    if not isinstance(day, str) or not DATE_RE.fullmatch(day):
        raise ValueError("Invalid date")
    date.fromisoformat(day)
    return day


def _validate_ref(day, event_id):
    if not isinstance(day, str) or not isinstance(event_id, str) or not DATE_RE.fullmatch(day) \
            or not EVENT_RE.fullmatch(event_id):
        raise ValueError("Invalid event path")
    date.fromisoformat(day)


# --------------------------------------------------------------------------- association

def associate_objects(no_helmet, detections):
    """Suggest only unambiguous overlapping rider/vehicle/plate boxes."""
    # Geometry cannot establish plate ownership; crossings still need human review.
    riders = [item for item in detections if item["class_name"] == "rider"]
    bikes = [item for item in detections if item["class_name"] == "bike"]
    plates = [item for item in detections if item["class_name"] == "number-plate"]
    point = _center(no_helmet)

    containing = [item for item in riders if _inside(point, item)]
    rider = containing[0] if len(containing) == 1 else None

    if rider is None:
        return {"rider": None, "bike": None, "plate": None}

    nearby = [item for item in bikes
              if _distance(rider, item) / math.sqrt(_area(rider)) <= 1.25
              and max(rider["box"][0], item["box"][0]) < min(rider["box"][2], item["box"][2])
              and max(rider["box"][1], item["box"][1]) < min(rider["box"][3], item["box"][3])]
    bike = nearby[0] if len(nearby) == 1 else None
    if bike:
        contained_plates = [item for item in plates if _inside(_center(item), bike)]
    else:
        # The bike box is often missed; a plate just below/inside the rider box still belongs to that rider.
        x1, y1, x2, y2 = rider["box"]
        below = {"box": [x1, y1, x2, y2 + 0.25 * (y2 - y1)]}
        contained_plates = [item for item in plates if _inside(_center(item), below)]
    plate = contained_plates[0] if len(contained_plates) == 1 else None
    return {"rider": rider, "bike": bike, "plate": plate}


def _crop(frame, box, padding=0.08):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = min(width, max(0, int(x1 - pad_x))), min(height, max(0, int(y1 - pad_y)))
    x2, y2 = min(width, max(0, int(x2 + pad_x))), min(height, max(0, int(y2 + pad_y)))
    return frame[y1:y2, x1:x2]


def blur_regions(frame, boxes):
    """Return a copy of frame with each box strongly blurred (privacy redaction)."""
    output = frame.copy()
    height, width = output.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        region = output[y1:y2, x1:x2]
        kernel = max(15, (max(x2 - x1, y2 - y1) // 3) | 1)
        output[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
    return output


# --------------------------------------------------------------------------- plate OCR

# Indian registration formats: standard (GJ01AB1234, DL3CAB1234) and Bharat series (22BH1234AB).
_STANDARD_PLATE = re.compile(r"[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}\Z")
_BH_PLATE = re.compile(r"\d{2}BH\d{4}[A-Z]{1,2}\Z")
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A", "7": "T"}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "T": "7", "A": "4"}


def _shape(text, pattern):
    """Coerce OCR confusions to a L/D shape string; return None if impossible."""
    if len(text) != len(pattern):
        return None
    out = []
    for char, kind in zip(text, pattern, strict=True):
        if kind == "L":
            char = char if char.isalpha() else _TO_LETTER.get(char)
        else:
            char = char if char.isdigit() else _TO_DIGIT.get(char)
        if char is None:
            return None
        out.append(char)
    return "".join(out)


def normalize_plate(text):
    """Return (text, valid) after fixing common OCR confusions against Indian plate formats."""
    text = re.sub(r"[^A-Z0-9]", "", str(text or "").upper())[:16]
    if _STANDARD_PLATE.fullmatch(text) or _BH_PLATE.fullmatch(text):
        return text, True
    shapes = ["DDLLDDDDL", "DDLLDDDDLL"]
    for district in (2, 1):
        for series in (2, 1, 3, 0):
            shapes.append("LL" + "D" * district + "L" * series + "DDDD")
    for shape in shapes:
        fixed = _shape(text, shape)
        if fixed and (_STANDARD_PLATE.fullmatch(fixed) or _BH_PLATE.fullmatch(fixed)):
            return fixed, True
    return text, False


def _tesseract(image, psm):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    try:
        result = subprocess.run(
            [TESSERACT, "stdin", "stdout", "--psm", str(psm), "-c",
             "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"],
            input=encoded.tobytes(), capture_output=True, timeout=8, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode:
        return ""
    return re.sub(r"[^A-Z0-9]", "", result.stdout.decode("utf-8", "ignore").upper())[:16]


def read_plate_text(crop):
    """Run local Tesseract OCR when installed; return an empty string otherwise.

    Tries several binarisations/page modes and prefers a reading that fits an Indian plate format.
    """
    if not TESSERACT or crop is None or crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    candidates = []
    for image, psm in ((otsu, 7), (cv2.bitwise_not(otsu), 7), (otsu, 8)):
        text = _tesseract(image, psm)
        if not text:
            continue
        fixed, valid = normalize_plate(text)
        if valid:
            return fixed
        candidates.append(text)
    return max(candidates, key=len) if candidates else ""


# --------------------------------------------------------------------------- events

def save_event(frame, event_detection, detections, association, ocr=read_plate_text, root=STORAGE,
               confirmation_frames=1, source=None, privacy=None):
    privacy = PRIVACY_MODE if privacy is None else privacy
    now = datetime.now().astimezone()
    day = now.date().isoformat()
    category = "helmet" if event_detection["class_name"] == "helmet" else "no_helmet"
    event_id = "event_" + now.strftime("%H%M%S_%f")
    folder = Path(root) / day / category / event_id

    rider = association["rider"]
    plate = association["plate"]
    bike = association["bike"]
    redacted = {"faces_blurred": 0, "plate_dropped": False}
    if privacy and category == "helmet":
        # Compliant riders are not violations: keep the helmet evidence, drop identifying detail.
        faces = [item["box"] for item in detections if item.get("class_name") == "face"]
        if faces:
            frame = blur_regions(frame, faces)
            redacted["faces_blurred"] = len(faces)
        if plate:
            redacted["plate_dropped"] = True
        plate = None
    person_box = rider["box"] if rider else event_detection["box"]
    person_crop = _crop(frame, person_box, 0.12)
    plate_crop = _crop(frame, plate["box"], 0.15) if plate else None
    context_boxes = [event_detection["box"], person_box]
    if bike:
        context_boxes.append(bike["box"])
    if plate:
        context_boxes.append(plate["box"])
    x1 = min(box[0] for box in context_boxes)
    y1 = min(box[1] for box in context_boxes)
    x2 = max(box[2] for box in context_boxes)
    y2 = max(box[3] for box in context_boxes)
    context = _crop(frame, [x1, y1, x2, y2], 0.12)
    if not person_crop.size or not context.size:
        raise ValueError("Detection is outside the image")
    plate_text = ocr(plate_crop) if plate_crop is not None and plate_crop.size else ""
    plate_text, plate_valid = normalize_plate(plate_text) if plate_text else ("", False)

    def summary(item):
        if not item:
            return None
        summary = {key: item.get(key) for key in ("class_name", "confidence", "track_id", "box")}
        if item.get("zoomed"):
            summary["zoomed"] = True  # caught by auto-zoom on a distant rider
        return summary

    metadata = {
        "id": event_id,
        "date": day,
        "timestamp": now.isoformat(timespec="seconds"),
        "source": source or {"type": "camera"},
        "status": "needs_review",
        "category": category,
        "confirmation_frames": confirmation_frames,
        "association_status": "suggested" if plate else "unmatched",
        "detections": [summary(item) for item in detections],
        "event_detection": summary(event_detection),
        "helmet": summary(event_detection) if category == "helmet" else None,
        "no_helmet": summary(event_detection) if category == "no_helmet" else None,
        "rider": summary(rider),
        "bike": summary(bike),
        "plate": summary(plate),
        "plate_text": plate_text,
        "plate_format_valid": plate_valid,
        "ocr_engine": "tesseract" if plate_text else "not available or no text read",
        "privacy": redacted,
        "files": {
            "person": "person.jpg",
            "event": "event.jpg",
            "plate": "plate.jpg" if plate_crop is not None and plate_crop.size else None,
        },
    }
    folder.mkdir(parents=True, exist_ok=False)
    try:
        if not cv2.imwrite(str(folder / "person.jpg"), person_crop) or not cv2.imwrite(str(folder / "event.jpg"), context):
            raise OSError("Could not save event images")
        if plate_crop is not None and plate_crop.size and not cv2.imwrite(str(folder / "plate.jpg"), plate_crop):
            raise OSError("Could not save plate crop")
        (folder / "event.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with _database(root) as connection:
            _write_row(connection, metadata)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return metadata


def list_events(day, root=STORAGE):
    _validate_day(day)
    with _database(root) as connection:
        rows = connection.execute(
            "SELECT data FROM events WHERE date=? ORDER BY timestamp DESC", (day,)
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def _filters(start=None, end=None, category=None, status=None, q=None, camera=None):
    clauses, params = [], []
    if start:
        clauses.append("date >= ?")
        params.append(_validate_day(start))
    if end:
        clauses.append("date <= ?")
        params.append(_validate_day(end))
    if category:
        if category not in EVENT_CATEGORIES:
            raise ValueError("Invalid category")
        clauses.append("category = ?")
        params.append(category)
    if status:
        if status not in REVIEW_STATUSES:
            raise ValueError("Invalid review status")
        clauses.append("status = ?")
        params.append(status)
    if camera:
        clauses.append("camera = ?")
        params.append(str(camera)[:120])
    if q:
        needle = re.sub(r"[^A-Za-z0-9_]", "", str(q))[:32].upper()
        if needle:
            clauses.append("(plate_text LIKE ? OR UPPER(id) LIKE ?)")
            params += [f"%{needle}%", f"%{needle}%"]
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def query_events(root=STORAGE, start=None, end=None, category=None, status=None, q=None, camera=None,
                 limit=100, offset=0):
    """Filtered, paginated event list (newest first) and the total match count."""
    where, params = _filters(start, end, category, status, q, camera)
    limit = max(1, min(int(limit), 5000))
    offset = max(0, int(offset))
    with _database(root) as connection:
        total = connection.execute(f"SELECT COUNT(*) FROM events{where}", params).fetchone()[0]
        rows = connection.execute(f"SELECT data FROM events{where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                                  params + [limit, offset]).fetchall()
    return [json.loads(row[0]) for row in rows], total


def event_stats(root=STORAGE, start=None, end=None, camera=None):
    """Per-day counts by category and review status, for charts and reports."""
    where, params = _filters(start, end, camera=camera)
    with _database(root) as connection:
        rows = connection.execute(
            f"SELECT date, category, status, COUNT(*) FROM events{where} GROUP BY date, category, status ORDER BY date",
            params).fetchall()
        cameras = [row[0] for row in connection.execute("SELECT DISTINCT camera FROM events ORDER BY camera")]
    days, totals = {}, {"events": 0, "helmet": 0, "no_helmet": 0,
                        "needs_review": 0, "confirmed": 0, "false_positive": 0}
    for day, category, status, count in rows:
        bucket = days.setdefault(day, {"date": day, "events": 0, "helmet": 0, "no_helmet": 0,
                                       "needs_review": 0, "confirmed": 0, "false_positive": 0})
        for target in (bucket, totals):
            target["events"] += count
            target[category or "no_helmet"] = target.get(category or "no_helmet", 0) + count
            target[status or "needs_review"] = target.get(status or "needs_review", 0) + count
    reviewed = totals["confirmed"] + totals["false_positive"]
    totals["false_positive_rate"] = round(totals["false_positive"] / reviewed, 4) if reviewed else None
    return {"days": list(days.values()), "totals": totals, "cameras": cameras}


def repeat_offenders(root=STORAGE, start=None, end=None, camera=None, min_count=2, limit=50):
    """No-helmet events grouped by plate text (false positives excluded), most frequent first."""
    where, params = _filters(start, end, "no_helmet", None, None, camera)
    where += (" AND " if where else " WHERE ") + "status != 'false_positive' AND plate_text != ''"
    with _database(root) as connection:
        rows = connection.execute(
            f"SELECT plate_text, COUNT(*), SUM(status = 'confirmed'), MIN(timestamp), MAX(timestamp), "
            f"GROUP_CONCAT(DISTINCT camera) FROM events{where} GROUP BY plate_text HAVING COUNT(*) >= ? "
            f"ORDER BY COUNT(*) DESC, MAX(timestamp) DESC LIMIT ?",
            params + [max(1, int(min_count)), max(1, min(int(limit), 500))]).fetchall()
        result = []
        for plate, count, confirmed, first, last, cameras in rows:
            latest = connection.execute(
                f"SELECT date, id FROM events{where} AND plate_text = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
                params + [plate]).fetchone()
            result.append({"plate_text": plate, "events": count, "confirmed": confirmed or 0,
                           "first_seen": first, "last_seen": last,
                           "cameras": sorted(filter(None, (cameras or "").split(","))),
                           "latest": {"date": latest[0], "id": latest[1]} if latest else None})
    return result


def get_event(day, event_id, root=STORAGE):
    _validate_ref(day, event_id)
    with _database(root) as connection:
        row = connection.execute("SELECT data FROM events WHERE date=? AND id=?", (day, event_id)).fetchone()
    if not row:
        raise FileNotFoundError(event_id)
    return json.loads(row[0])


def update_review(day, event_id, status, plate_text="", root=STORAGE):
    if not all(isinstance(value, str) for value in (day, event_id, status, plate_text)):
        raise ValueError("Review fields must be text")
    _validate_ref(day, event_id)
    if status not in REVIEW_STATUSES:
        raise ValueError("Invalid review status")
    root = Path(root)
    data = get_event(day, event_id, root)
    folder = event_folder(root, day, event_id, data.get("category"))
    path = folder / "event.json"
    data["status"] = status
    data["association_status"] = (
        "rejected" if status == "false_positive" else
        "confirmed" if status == "confirmed" and data.get("plate") else
        "suggested" if data.get("plate") else "unmatched"
    )
    data["plate_text"] = re.sub(r"[^A-Z0-9]", "", plate_text.upper())[:16]
    data["plate_format_valid"] = normalize_plate(data["plate_text"])[1] if data["plate_text"] else False
    data["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if folder.is_dir():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)
    with _database(root) as connection:
        _write_row(connection, data)
    return data


def delete_event(day, event_id, root=STORAGE):
    _validate_ref(day, event_id)
    root = Path(root)
    data = get_event(day, event_id, root)
    folder = event_folder(root, day, event_id, data.get("category"))
    if folder.is_symlink():
        raise ValueError("Invalid event path")
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=False)
    with _database(root) as connection:
        connection.execute("DELETE FROM events WHERE date=? AND id=?", (day, event_id))
    return True


def bulk_apply(items, action, root=STORAGE, status=None):
    """Apply review/delete to many {date,id} items; returns per-item results without stopping on errors."""
    if action not in {"review", "delete"}:
        raise ValueError("Invalid bulk action")
    if not isinstance(items, list) or not 0 < len(items) <= 500:
        raise ValueError("Choose between 1 and 500 events")
    results = []
    for item in items:
        try:
            day, event_id = item["date"], item["id"]
            if action == "delete":
                delete_event(day, event_id, root)
            else:
                current = get_event(day, event_id, root)
                update_review(day, event_id, status, current.get("plate_text") or "", root)
            results.append({"date": day, "id": event_id, "ok": True})
        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as exc:
            results.append({"date": item.get("date") if isinstance(item, dict) else None,
                            "id": item.get("id") if isinstance(item, dict) else None,
                            "ok": False, "error": str(exc) or type(exc).__name__})
    return results


def event_folder(root, day, event_id, category=None):
    """Resolve current categorized folders while retaining old saved event paths."""
    root = Path(root)
    date_folder = root / day
    if root.is_symlink() or date_folder.is_symlink():
        raise ValueError("Invalid event path")
    legacy = date_folder / event_id
    if legacy.is_symlink():
        raise ValueError("Invalid event path")
    if legacy.is_dir():
        return legacy
    if category in EVENT_CATEGORIES:
        category_folder = date_folder / category
        candidate = category_folder / event_id
        if category_folder.is_symlink() or candidate.is_symlink():
            raise ValueError("Invalid event path")
        return candidate
    for candidate in (root / day / name / event_id for name in EVENT_CATEGORIES):
        if candidate.parent.is_symlink() or candidate.is_symlink():
            raise ValueError("Invalid event path")
        if candidate.is_dir():
            return candidate
    return root / day / "no_helmet" / event_id


def migrate_event_categories(root=STORAGE):
    """Move legacy captures into their date/category folders and label their records."""
    root = Path(root)
    moved = 0
    with _database(root) as connection:
        rows = connection.execute("SELECT date, id, data FROM events").fetchall()
        for day, event_id, raw_data in rows:
            data = json.loads(raw_data)
            detection = data.get("event_detection") or data.get("helmet") or data.get("no_helmet") or {}
            category = data.get("category")
            if category not in EVENT_CATEGORIES:
                category = "helmet" if detection.get("class_name") == "helmet" else "no_helmet"
            legacy = root / day / event_id
            category_folder = root / day / category
            target = category_folder / event_id
            if (root / day).is_symlink() or category_folder.is_symlink() or legacy.is_symlink() or target.is_symlink():
                continue
            if legacy.is_dir() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy), str(target))
                moved += 1
            if not target.is_dir():
                continue
            if data.get("category") == category and (target / "event.json").is_file():
                continue
            data["category"] = category
            data["event_detection"] = detection or None
            temporary = (target / "event.json").with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.replace(target / "event.json")
            _write_row(connection, data)
    return moved


def _event_dirs(date_folder):
    for path in date_folder.iterdir():
        if path.is_symlink():
            continue
        if path.is_dir() and EVENT_RE.fullmatch(path.name):
            yield path
        elif path.is_dir() and path.name in EVENT_CATEGORIES:
            yield from (child for child in path.iterdir()
                        if child.is_dir() and not child.is_symlink() and EVENT_RE.fullmatch(child.name))


def clean_expired_events(root=STORAGE, today=None, retention_days=None, keep_confirmed=None):
    """Remove dated evidence older than retention; keep confirmed events and unknown files.

    Returns the number of date folders removed completely.
    """
    root = Path(root)
    today = today or date.today()
    retention_days = RETENTION_DAYS if retention_days is None else retention_days
    keep_confirmed = KEEP_CONFIRMED if keep_confirmed is None else keep_confirmed
    if not root.is_dir() or root.is_symlink():
        return 0
    if retention_days <= 0:
        migrate_event_categories(root)
        return 0
    cutoff = (today - timedelta(days=retention_days)).isoformat()
    kept = set()
    database = root / "events.db"
    if keep_confirmed and database.is_file() and not database.is_symlink():
        with _database(root) as connection:
            kept = {(day, event_id) for day, event_id in connection.execute(
                "SELECT date, id FROM events WHERE date < ? AND status = 'confirmed'", (cutoff,))}
    removed = 0
    for folder in root.iterdir():
        if folder.is_symlink() or not folder.is_dir() or not DATE_RE.fullmatch(folder.name):
            continue
        try:
            age = (today - date.fromisoformat(folder.name)).days
        except ValueError:
            continue
        if age <= retention_days:
            continue
        keep_here = {event_id for day, event_id in kept if day == folder.name}
        if not keep_here:
            shutil.rmtree(folder)
            removed += 1
            continue
        for event_dir in list(_event_dirs(folder)):
            if event_dir.name not in keep_here:
                shutil.rmtree(event_dir)
    if database.is_file() and not database.is_symlink():
        with _database(root) as connection:
            if keep_confirmed:
                connection.execute("DELETE FROM events WHERE date < ? AND status != 'confirmed'", (cutoff,))
            else:
                connection.execute("DELETE FROM events WHERE date < ?", (cutoff,))
    migrate_event_categories(root)
    return removed


def import_storage(source, root=STORAGE):
    """Merge events from another storage folder (e.g. a backup). Existing events are never overwritten."""
    source, root = Path(source).resolve(), Path(root).resolve()
    if source == root:
        raise ValueError("Source and destination are the same storage")
    if not (source / "events.db").is_file():
        raise ValueError("Source has no events.db")
    imported = 0
    connection = sqlite3.connect(f"file:{source / 'events.db'}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT data FROM events").fetchall()
    finally:
        connection.close()
    for raw, in rows:
        data = json.loads(raw)
        day, event_id = data["date"], data["id"]
        _validate_ref(day, event_id)
        origin = event_folder(source, day, event_id, data.get("category"))
        category = data.get("category") if data.get("category") in EVENT_CATEGORIES else origin.parent.name
        if category not in EVENT_CATEGORIES:
            category = "no_helmet"
        data["category"] = category
        target = root / day / category / event_id
        with _database(root) as db:
            if db.execute("SELECT 1 FROM events WHERE date=? AND id=?", (day, event_id)).fetchone():
                continue
        if origin.is_dir() and not target.exists():
            shutil.copytree(origin, target)
        with _database(root) as db:
            _write_row(db, data, replace=False)
        imported += 1
    return imported


# --------------------------------------------------------------------------- export

EXPORT_FIELDS = ["date", "time", "id", "category", "status", "camera", "source_type", "video_filename",
                 "video_position_seconds", "confidence", "plate_text", "plate_format_valid",
                 "association_status", "rider_track", "bike_track", "reviewed_at"]


def _export_row(event):
    detection = event.get("event_detection") or event.get("helmet") or event.get("no_helmet") or {}
    source = event.get("source") or {}
    timestamp = event.get("timestamp") or ""
    value = {
        "date": event.get("date"), "time": timestamp[11:19], "id": event.get("id"),
        "category": event.get("category"), "status": event.get("status"),
        "camera": source.get("camera") or source.get("type") or "", "source_type": source.get("type", ""),
        "video_filename": source.get("filename", ""), "video_position_seconds": source.get("position_seconds", ""),
        "confidence": round(detection["confidence"], 4) if isinstance(detection.get("confidence"), (int, float)) else "",
        "plate_text": event.get("plate_text", ""), "plate_format_valid": event.get("plate_format_valid", ""),
        "association_status": event.get("association_status", ""),
        "rider_track": (event.get("rider") or {}).get("track_id", ""),
        "bike_track": (event.get("bike") or {}).get("track_id", ""),
        "reviewed_at": event.get("reviewed_at", ""),
    }
    # Neutralise spreadsheet formula injection from free-text fields.
    return {key: ("'" + val if isinstance(val, str) and val[:1] in "=+-@" else val) for key, val in value.items()}


def export_csv(events):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_FIELDS)
    writer.writeheader()
    for event in events:
        writer.writerow(_export_row(event))
    return buffer.getvalue()


def export_zip(events, root, fileobj):
    """Write events.csv plus each event's evidence folder into a ZIP archive."""
    root = Path(root)
    with zipfile.ZipFile(fileobj, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("events.csv", export_csv(events))
        for event in events:
            try:
                folder = event_folder(root, event["date"], event["id"], event.get("category"))
            except ValueError:
                continue
            if not folder.is_dir():
                continue
            for name in ("person.jpg", "event.jpg", "plate.jpg", "event.json"):
                path = folder / name
                if path.is_file() and not path.is_symlink():
                    archive.write(path, f"{event['date']}/{event.get('category')}/{event['id']}/{name}")
    return fileobj


# --------------------------------------------------------------------------- audit and settings

def record_audit(root, user, action, target="", details=None):
    try:
        with _database(root) as connection:
            connection.execute("INSERT INTO audit VALUES (?, ?, ?, ?, ?)", (
                datetime.now().astimezone().isoformat(timespec="seconds"), str(user or "system")[:64],
                str(action)[:64], str(target)[:200], json.dumps(details or {})[:2000]))
    except sqlite3.Error:
        logging.exception("Could not write audit entry")


def list_audit(root=STORAGE, limit=200):
    with _database(root) as connection:
        rows = connection.execute("SELECT at, user, action, target, details FROM audit ORDER BY at DESC, rowid DESC "
                                  "LIMIT ?", (max(1, min(int(limit), 2000)),)).fetchall()
    return [{"at": at, "user": user, "action": action, "target": target, "details": json.loads(details or "{}")}
            for at, user, action, target, details in rows]


def get_setting(root, key, default=None):
    with _database(root) as connection:
        row = connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_setting(root, key, value):
    with _database(root) as connection:
        connection.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value)))
    return value
