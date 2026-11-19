"""Calibrate the zoom-crop helmet/no-helmet score threshold on labelled validation crops.

Each labelled head box in the overlap-filtered validation split becomes a zoom target. The crop
geometry copies app.zoom_region (target box, zoom = min(4, 0.65 * H / box_height), crop resized
back to the frame size). The model's best on-target helmet/no-helmet prediction is the decision.
The threshold is chosen by macro F1 on validation, then frozen and reported on test.

Finding (2026-11-19): app.zoom_region sizes every target to 65% of the frame height. For a head/face
target that makes the head far larger than anything in training and on-target accuracy collapses
(0.30 correct at 65% vs 0.82 at 20%). Calibration therefore uses a target fraction per target type
(--head-fraction, --rider-fraction) and records it as "zoom_geometry" in the candidate calibration; app.py
must use the same geometry for the threshold to be valid.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MAX_DISPLAY_ZOOM = 4.0  # mirrors app.MAX_DISPLAY_ZOOM
INFERENCE_SIZE = 800    # mirrors app.INFERENCE_SIZE
HEADS = {0: "helmet", 1: "no-helmet"}
THRESHOLDS = [round(t, 2) for t in np.arange(0.02, 0.96, 0.01)]


def zoom_crop(frame, box, fraction=0.65):
    """fraction = share of the frame height the target box should fill after zooming."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
    box_height = y2 - y1
    if box_height <= 0:
        return None
    zoom = min(MAX_DISPLAY_ZOOM, max(1.0, fraction * height / max(box_height, 1)))
    if zoom <= 1.05:
        return None
    crop_width, crop_height = int(width / zoom), int(height / zoom)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    left = min(max(cx - crop_width // 2, 0), width - crop_width)
    top = min(max(cy - crop_height // 2, 0), height - crop_height)
    crop = frame[top:top + crop_height, left:left + crop_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR), (left, top, zoom)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def labels_for(image):
    label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
    rows = []
    if label.is_file():
        for line in label.read_text().splitlines():
            parts = line.split()
            if parts:
                rows.append((int(parts[0]), *map(float, parts[1:5])))
    return rows


def collect(model, images, device, head_fraction=0.65, rider_fraction=0.65):
    """Return one (gt_class, predicted_class_or_None, score) row per zoomed head target."""
    rows = []
    for image in images:
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        labels = labels_for(image)
        boxes = [(c, [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height])
                 for c, x, y, w, h in labels]
        riders = [b for c, b in boxes if c == 3]
        for cls, head in boxes:
            if cls not in HEADS:
                continue
            cx, cy = (head[0] + head[2]) / 2, (head[1] + head[3]) / 2
            containing = [r for r in riders if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]]
            target = containing[0] if len(containing) == 1 else head  # app.py prefers the rider target
            fraction = head_fraction if target is head else rider_fraction
            zoomed = zoom_crop(frame, [int(v) for v in target], fraction)
            if zoomed is None:
                continue
            crop, (left, top, zoom) = zoomed
            result = model.predict(crop, conf=0.01, imgsz=INFERENCE_SIZE, classes=[0, 1],
                                   device=device, verbose=False)[0]
            best = None
            for pcls, score, pbox in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                         result.boxes.xyxy.tolist(), strict=False):
                mapped = [pbox[0] / zoom + left, pbox[1] / zoom + top, pbox[2] / zoom + left, pbox[3] / zoom + top]
                mx, my = (mapped[0] + mapped[2]) / 2, (mapped[1] + mapped[3]) / 2
                on_target = iou(mapped, head) >= 0.3 or (head[0] <= mx <= head[2] and head[1] <= my <= head[3])
                if on_target and (best is None or score > best[1]):
                    best = (HEADS[int(pcls)], float(score))
            rows.append({"gt": HEADS[cls], "pred": best[0] if best else None,
                         "score": best[1] if best else 0.0, "zoom": round(zoom, 2),
                         "target": "rider" if target is not head else "head"})
    return rows


def scores_at(rows, threshold):
    out = {}
    for name in HEADS.values():
        tp = sum(r["gt"] == name and r["pred"] == name and r["score"] >= threshold for r in rows)
        fp = sum(r["gt"] != name and r["pred"] == name and r["score"] >= threshold for r in rows)
        gt = sum(r["gt"] == name for r in rows)
        precision, recall = tp / max(tp + fp, 1), tp / max(gt, 1)
        out[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
                     "targets": gt}
    out["macro_f1"] = round(np.mean([out[n]["f1"] for n in HEADS.values()]), 4)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "runs/detect/helmet_v3/weights/best.pt")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--head-fraction", type=float, default=0.20)
    parser.add_argument("--rider-fraction", type=float, default=0.45)
    parser.add_argument("--min-precision", type=float, default=0.80,
                        help="minimum validation precision per class (full-frame level)")
    parser.add_argument("--write-candidate", action="store_true",
                        help="write a candidate calibration to reports/ (never edits thresholds.json)")
    args = parser.parse_args()
    model = YOLO(args.model)
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    split_rows, current_rows = {}, {}
    for split, listing in (("val", "val-images.txt"), ("test", "test-images.txt")):
        images = [Path(p) for p in (args.reports / listing).read_text().split()]
        split_rows[split] = collect(model, images, args.device, args.head_fraction, args.rider_fraction)
        current_rows[split] = collect(model, images, args.device, 0.65, 0.65)
    current_sweep = {t: scores_at(current_rows["val"], t) for t in THRESHOLDS}
    # Selection rule: zoomed-crop precision must stay at or above full-frame precision (>= MIN_PRECISION) for
    # BOTH classes on validation; among those thresholds take the best macro F1 (i.e. the most recall).
    # Unconstrained F1 keeps rising toward the lowest score, which would accept almost any zoomed prediction.
    sweep = {t: scores_at(split_rows["val"], t) for t in THRESHOLDS}
    def precise(table, t):
        return all(table[t][name]["precision"] >= args.min_precision for name in HEADS.values())
    allowed = [t for t in THRESHOLDS if precise(sweep, t)] or [max(THRESHOLDS)]
    chosen = max(allowed, key=lambda t: (sweep[t]["macro_f1"], t))
    current_allowed = [t for t in THRESHOLDS if precise(current_sweep, t)] or [max(THRESHOLDS)]
    current_best = max(current_allowed, key=lambda t: (current_sweep[t]["macro_f1"], t))
    per_class = {name: max(allowed, key=lambda t: sweep[t][name]["f1"]) for name in HEADS.values()}
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "model_sha256": digest,
        "imgsz": INFERENCE_SIZE, "method": __doc__.strip(),
        "targets": {s: len(r) for s, r in split_rows.items()},
        "selection": f"max validation macro F1 among thresholds with validation precision >= {args.min_precision} "
                     "for both helmet and no-helmet",
        "chosen_zoom_crop_threshold": chosen,
        "val_sweep": {str(t): sweep[t]["macro_f1"] for t in THRESHOLDS},
        "unconstrained_val_best": max(THRESHOLDS, key=lambda t: sweep[t]["macro_f1"]),
        "per_class_best_threshold_on_val": per_class,
        "val_at_chosen": sweep[chosen], "test_at_chosen": scores_at(split_rows["test"], chosen),
        "zoom_geometry": {"head_fraction": args.head_fraction, "rider_fraction": args.rider_fraction},
        "val_at_previous_0.50": sweep[0.5], "test_at_previous_0.50": scores_at(split_rows["test"], 0.5),
        "current_hi_geometry_0.65": {"best_val_threshold": current_best,
                                     "test_at_best": scores_at(current_rows["test"], current_best),
                                     "test_at_0.50": scores_at(current_rows["test"], 0.5)},
        "limitations": ["Crops are upscaled source-dataset images, not live camera crops.",
                        "Face-target zoom (YuNet) is approximated by the head or containing rider box.",
                        "Only the best on-target prediction is scored; off-target zoom detections are ignored, "
                        "as app.py discards zoom detections that do not belong to the focus."],
    }
    (args.reports / "zoom-calibration-2026-11-19.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("targets", "chosen_zoom_crop_threshold", "per_class_best_threshold_on_val",
                                             "val_at_chosen", "test_at_chosen", "test_at_previous_0.50",
                                             "current_hi_geometry_0.65")}, indent=1))
    if args.write_candidate:
        calibration = json.loads((ROOT / "thresholds.json").read_text())
        if calibration["model_sha256"] != digest:
            raise SystemExit("thresholds.json belongs to a different model; candidate not written")
        calibration.update(zoom_crop=chosen, zoom_geometry=report["zoom_geometry"],
                           zoom_limit=("Validation precision >= {:.2f} per class on upscaled dataset head crops with "
                                       "zoom_geometry; valid only if app.py zooms with the same geometry. Not validated "
                                       "on live camera crops.").format(args.min_precision))
        out = args.reports / "thresholds-zoom-calibrated-2026-11-19.json"
        out.write_text(json.dumps(calibration, indent=2) + "\n")
        print(f"Candidate written to {out}")


if __name__ == "__main__":
    main()
