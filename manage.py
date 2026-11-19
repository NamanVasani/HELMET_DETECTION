"""Helmet admin command line.

  python manage.py create-user NAME --role admin          (prompts for the password)
  python manage.py reset-password NAME
  python manage.py list-users
  python manage.py import-storage storage.before-login-service
  python manage.py export --start 2026-11-01 --end 2026-11-30 --out events.zip
  python manage.py prepare-model [--source runs/detect/helmet_v3/weights/best.pt]
  python manage.py training-export --out reviewed_dataset [--start ...] [--end ...]
  python manage.py cleanup
"""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2

import auth
from event_store import (STORAGE, clean_expired_events, event_folder, export_csv, export_zip,
                         import_storage, query_events, record_audit)

PROJECT = Path(__file__).resolve().parent
CLASSES = ["helmet", "no-helmet", "number-plate", "rider", "bike"]


def ask_password(confirm=True):
    password = getpass.getpass("Password (10+ characters): ")
    if confirm and password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match")
    return password


def prepare_model(source, target, manifest_path):
    """Copy a trusted checkpoint to best.pt and verify it against model-manifest.json."""
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise SystemExit(f"{source} does not match model-manifest.json (sha256 {digest[:12]}…). "
                         "Use the trusted checkpoint or re-run evaluate_model.py for a new one.")
    if target.resolve() != source.resolve():
        shutil.copy2(source, target)
    return digest


def yolo_box(box, width, height):
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def training_export(root, out, start=None, end=None):
    """Turn human-reviewed events into a YOLO dataset for the next fine-tuning round.

    Confirmed events keep the detector's head label (a human agreed). False positives become hard negatives: the
    image is kept with the wrong head label removed. Context crops are stored with box coordinates translated
    from the full frame. Boxes for other classes are kept only when they lie inside the crop.
    """
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    counts = {"confirmed": 0, "false_positive": 0, "skipped": 0}
    for status in ("confirmed", "false_positive"):
        events, _ = query_events(root, start=start, end=end, status=status, limit=5000)
        for event in events:
            folder = event_folder(root, event["date"], event["id"], event.get("category"))
            image = cv2.imread(str(folder / "event.jpg"))
            if image is None or event.get("privacy", {}).get("faces_blurred"):
                counts["skipped"] += 1  # blurred evidence would teach the model the wrong thing
                continue
            boxes = [item for item in event.get("detections", []) if item and item.get("class_name") in CLASSES]
            head = event.get("event_detection") or {}
            # Context crop origin: the union of the boxes used when saving, padded 12%.
            used = [head.get("box")] + [(event.get(key) or {}).get("box") for key in ("rider", "bike", "plate")]
            used = [box for box in used if box]
            x1, y1 = min(b[0] for b in used), min(b[1] for b in used)
            x2, y2 = max(b[2] for b in used), max(b[3] for b in used)
            pad_x, pad_y = (x2 - x1) * 0.12, (y2 - y1) * 0.12
            origin = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)))
            height, width = image.shape[:2]
            lines = []
            for item in boxes:
                if status == "false_positive" and item.get("box") == head.get("box") and \
                        item.get("class_name") == head.get("class_name"):
                    continue
                shifted = [item["box"][0] - origin[0], item["box"][1] - origin[1],
                           item["box"][2] - origin[0], item["box"][3] - origin[1]]
                converted = yolo_box(shifted, width, height)
                if converted:
                    lines.append(f"{CLASSES.index(item['class_name'])} " + " ".join(f"{v:.6f}" for v in converted))
            name = f"{event['date']}_{event['id']}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), image)
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts[status] += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\nnames:\n" +
        "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES)))
    (out / "README.txt").write_text(
        "Human-reviewed Helmet events exported for fine-tuning.\n"
        "Confirmed events keep the reviewed label; false positives are hard negatives.\n"
        "Merge into a training split only; never into validation/test, to keep evaluation independent.\n")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Helmet administration", formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--storage", type=Path, default=STORAGE)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-user")
    create.add_argument("username")
    create.add_argument("--role", choices=auth.ROLES, default="reviewer")
    reset = commands.add_parser("reset-password")
    reset.add_argument("username")
    commands.add_parser("list-users")
    importer = commands.add_parser("import-storage", help="merge events from a backup storage folder")
    importer.add_argument("source", type=Path)
    export = commands.add_parser("export")
    export.add_argument("--start")
    export.add_argument("--end")
    export.add_argument("--out", type=Path, required=True, help=".csv or .zip")
    model = commands.add_parser("prepare-model")
    model.add_argument("--source", type=Path, default=PROJECT / "runs/detect/helmet_v3/weights/best.pt")
    training = commands.add_parser("training-export", help="export reviewed events as a YOLO dataset")
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--start")
    training.add_argument("--end")
    commands.add_parser("cleanup", help="apply the retention policy now")
    args = parser.parse_args(argv)
    root = args.storage

    if args.command == "create-user":
        user = auth.create_user(root, args.username, ask_password(), args.role)
        record_audit(root, "cli", "user_created", user["username"], {"role": user["role"]})
        print(f"Created {user['username']} ({user['role']})")
    elif args.command == "reset-password":
        auth.update_user(root, args.username, password=ask_password())
        record_audit(root, "cli", "password_reset", args.username)
        print(f"Password reset for {args.username}; their sessions were signed out")
    elif args.command == "list-users":
        for user in auth.list_users(root):
            print(f"{user['username']:24} {user['role']:9} {user['created']}")
    elif args.command == "import-storage":
        count = import_storage(args.source, root)
        record_audit(root, "cli", "import_storage", str(args.source), {"imported": count})
        print(f"Imported {count} event(s) from {args.source}. Existing events were left unchanged.")
    elif args.command == "export":
        events, _ = query_events(root, start=args.start, end=args.end, limit=5000)
        if args.out.suffix == ".zip":
            with args.out.open("wb") as handle:
                export_zip(events, root, handle)
        else:
            args.out.write_text(export_csv(events), encoding="utf-8")
        record_audit(root, "cli", "export", str(args.out), {"events": len(events)})
        print(f"Exported {len(events)} event(s) to {args.out}")
    elif args.command == "prepare-model":
        digest = prepare_model(args.source, PROJECT / "best.pt", PROJECT / "model-manifest.json")
        print(f"best.pt ready (sha256 {digest[:12]}…)")
    elif args.command == "training-export":
        counts = training_export(root, args.out, args.start, args.end)
        print(f"Exported {counts['confirmed']} confirmed and {counts['false_positive']} false-positive events "
              f"({counts['skipped']} skipped) to {args.out}")
    elif args.command == "cleanup":
        print(f"Removed {clean_expired_events(root)} expired date folder(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
