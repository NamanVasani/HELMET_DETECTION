"""Local event storage, review, expiry, and lightweight plate association."""

from datetime import date, datetime, timedelta
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import sqlite3
import shutil
import subprocess
import os

import cv2

ROOT = Path(__file__).resolve().parent
STORAGE = Path(os.environ.get("HELMET_STORAGE", ROOT / "storage")).resolve()
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
EVENT_RE = re.compile(r"event_[\w-]{1,80}\Z")
EVENT_CATEGORIES = {"helmet", "no_helmet"}
RETENTION_DAYS = 30
TESSERACT = shutil.which("tesseract") or next(
    (path for path in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract") if Path(path).is_file()),
    None,
)


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


@contextmanager
def _database(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "events.db")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS events (date TEXT, id TEXT, timestamp TEXT, status TEXT, data TEXT, PRIMARY KEY (date, id))"
    )
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    imported = connection.execute("SELECT value FROM metadata WHERE key='json_imported'").fetchone()
    if not imported:
        paths = set(root.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/event_*/event.json"))
        for category in EVENT_CATEGORIES:
            paths.update(root.glob(f"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/{category}/event_*/event.json"))
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?)",
                    (data["date"], data["id"], data["timestamp"], data["status"], json.dumps(data)),
                )
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
        connection.execute("INSERT INTO metadata VALUES ('json_imported', '1')")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def associate_objects(no_helmet, detections):
    """Suggest only unambiguous overlapping rider/vehicle/plate boxes."""
    # shortcut: geometry cannot establish plate ownership; crossings still need human review.
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
    contained_plates = [item for item in plates if bike and _inside(_center(item), bike)]
    plate = contained_plates[0] if len(contained_plates) == 1 else None
    return {"rider": rider, "bike": bike, "plate": plate}


def _crop(frame, box, padding=0.08):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = min(width, max(0, int(x1 - pad_x))), min(height, max(0, int(y1 - pad_y)))
    x2, y2 = min(width, max(0, int(x2 + pad_x))), min(height, max(0, int(y2 + pad_y)))
    return frame[y1:y2, x1:x2]


def read_plate_text(crop):
    """Run local Tesseract OCR when installed; return an empty string otherwise."""
    executable = TESSERACT
    if not executable or crop is None or crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        return ""
    try:
        result = subprocess.run(
            [executable, "stdin", "stdout", "--psm", "7", "-c",
             "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"],
            input=encoded.tobytes(), capture_output=True, timeout=8, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode:
        return ""
    text = re.sub(r"[^A-Z0-9]", "", result.stdout.decode("utf-8", "ignore").upper())
    return text[:16]


def save_event(frame, event_detection, detections, association, ocr=read_plate_text, root=STORAGE,
               confirmation_frames=1, source=None):
    now = datetime.now().astimezone()
    day = now.date().isoformat()
    category = "helmet" if event_detection["class_name"] == "helmet" else "no_helmet"
    event_id = "event_" + now.strftime("%H%M%S_%f")
    folder = Path(root) / day / category / event_id

    rider = association["rider"]
    plate = association["plate"]
    bike = association["bike"]
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

    def summary(item):
        if not item:
            return None
        return {key: item[key] for key in ("class_name", "confidence", "track_id", "box")}

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
        "ocr_engine": "tesseract" if plate_text else "not available or no text read",
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
            connection.execute(
                "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?)",
                (day, event_id, metadata["timestamp"], metadata["status"], json.dumps(metadata)),
            )
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return metadata


def list_events(day, root=STORAGE):
    if not DATE_RE.fullmatch(day):
        raise ValueError("Invalid date")
    date.fromisoformat(day)
    with _database(root) as connection:
        rows = connection.execute(
            "SELECT data FROM events WHERE date=? ORDER BY timestamp DESC", (day,)
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def update_review(day, event_id, status, plate_text="", root=STORAGE):
    if not all(isinstance(value, str) for value in (day, event_id, status, plate_text)):
        raise ValueError("Review fields must be text")
    if not DATE_RE.fullmatch(day) or not EVENT_RE.fullmatch(event_id):
        raise ValueError("Invalid event path")
    date.fromisoformat(day)
    if status not in {"confirmed", "false_positive", "needs_review"}:
        raise ValueError("Invalid review status")
    root = Path(root)
    with _database(root) as connection:
        row = connection.execute("SELECT data FROM events WHERE date=? AND id=?", (day, event_id)).fetchone()
    if not row:
        raise FileNotFoundError(event_id)
    data = json.loads(row[0])
    folder = event_folder(root, day, event_id, data.get("category"))
    path = folder / "event.json"
    data["status"] = status
    data["association_status"] = (
        "rejected" if status == "false_positive" else
        "confirmed" if status == "confirmed" and data.get("plate") else
        "suggested" if data.get("plate") else "unmatched"
    )
    data["plate_text"] = re.sub(r"[^A-Z0-9]", "", plate_text.upper())[:16]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)
    with _database(root) as connection:
        connection.execute(
            "UPDATE events SET timestamp=?, status=?, data=? WHERE date=? AND id=?",
            (data["timestamp"], data["status"], json.dumps(data), day, event_id),
        )
    return data


def delete_event(day, event_id, root=STORAGE):
    if not DATE_RE.fullmatch(day) or not EVENT_RE.fullmatch(event_id):
        raise ValueError("Invalid event path")
    date.fromisoformat(day)
    root = Path(root)
    with _database(root) as connection:
        row = connection.execute("SELECT data FROM events WHERE date=? AND id=?", (day, event_id)).fetchone()
    if not row:
        raise FileNotFoundError(event_id)
    data = json.loads(row[0])
    folder = event_folder(root, day, event_id, data.get("category"))
    if folder.is_symlink():
        raise ValueError("Invalid event path")
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=False)
    with _database(root) as connection:
        connection.execute("DELETE FROM events WHERE date=? AND id=?", (day, event_id))
    return True


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
            connection.execute("UPDATE events SET data=? WHERE date=? AND id=?",
                               (json.dumps(data), day, event_id))
    return moved


def clean_expired_events(root=STORAGE, today=None, retention_days=RETENTION_DAYS):
    """Remove only dated storage folders older than retention; leave unknown files alone."""
    root = Path(root)
    today = today or date.today()
    if not root.is_dir() or root.is_symlink():
        return 0
    removed = 0
    for folder in root.iterdir():
        if folder.is_symlink() or not folder.is_dir() or not DATE_RE.fullmatch(folder.name):
            continue
        try:
            age = (today - date.fromisoformat(folder.name)).days
        except ValueError:
            continue
        if age > retention_days:
            shutil.rmtree(folder)
            removed += 1
    database = root / "events.db"
    if database.is_file() and not database.is_symlink():
        connection = sqlite3.connect(database)
        try:
            connection.execute("DELETE FROM events WHERE date < ?", ((today - timedelta(days=retention_days)).isoformat(),))
            connection.commit()
        finally:
            connection.close()
    migrate_event_categories(root)
    return removed
