"""Fill classes that a source never annotates (partial annotation) with teacher pseudo-labels.

Only the TRAIN split is changed; validation/test keep their original ground truth.
- rider/bike for ds2 and ds3 come from a COCO teacher (person on motorcycle, motorcycle).
- number-plate for ds3 comes from the current helmet model at a high score.
Rules are tuned against ds1, the only source that annotates rider and bike.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "merged_dataset"
CACHE = ROOT / "runs/pseudo"
MISSING = {"ds2": {"rider", "bike"}, "ds3": {"rider", "bike", "number-plate"}}
IDS = {"helmet": 0, "no-helmet": 1, "number-plate": 2, "rider": 3, "bike": 4}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect(model_path, images, classes, cache, device, chunk=16):
    """Cache normalized xyxy detections at a low score so rules can be tuned offline.

    Progress is checkpointed every chunk so an interrupted run (e.g. a stalled GPU) resumes.
    """
    out = json.loads(cache.read_text()) if cache.is_file() else {}
    todo = [p for p in images if p.name not in out]
    if not todo:
        return out
    model = YOLO(model_path)
    for start in range(0, len(todo), chunk):
        batch = todo[start:start + chunk]
        for path in batch:
            result = model.predict(str(path), conf=0.25, classes=classes, imgsz=640, device=device,
                                   verbose=False)[0]
            out[path.name] = [[int(c), float(s), *[float(v) for v in b]] for c, s, b in
                              zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                  result.boxes.xyxyn.tolist(), strict=False)]
        cache.write_text(json.dumps(out))
        print(f"{cache.name}: {len(out)}/{len(images)}", flush=True)
    return out


def teacher_boxes(dets, person_conf, bike_conf, overlap, link_conf=0.25):
    """COCO person(0) overlapping a motorcycle(3) -> rider; motorcycle -> bike.

    A person is linked to any motorcycle at link_conf, which is lower than the score needed to emit the
    motorcycle itself as a bike label; this recovers riders whose motorcycle is only weakly detected.
    """
    bikes = [d[2:] for d in dets if d[0] == 3 and d[1] >= link_conf]
    riders = []
    for d in dets:
        if d[0] != 0 or d[1] < person_conf:
            continue
        p = d[2:]
        if any(iou(p, b) >= overlap or (b[0] <= (p[0] + p[2]) / 2 <= b[2] and p[3] > b[1] and p[1] < b[1])
               for b in bikes):
            riders.append(p)
    return riders, [d[2:] for d in dets if d[0] == 3 and d[1] >= bike_conf]


def gt_boxes(label, cls):
    rows = []
    if label.is_file():
        for line in label.read_text().splitlines():
            parts = line.split()
            if parts and int(parts[0]) == cls:
                x, y, w, h = map(float, parts[1:5])
                rows.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    return rows


def match(pred, gt, thr=0.5):
    used, tp = set(), 0
    for p in pred:
        best = max(((iou(p, g), i) for i, g in enumerate(gt) if i not in used), default=(0, None))
        if best[0] >= thr:
            used.add(best[1])
            tp += 1
    return tp


def tune(teacher, labels):
    """Grid-search teacher rules on ds1 train, maximizing F1 per class."""
    best = {}
    for name, cls in (("rider", 3), ("bike", 4)):
        results = []
        for conf in (0.3, 0.4, 0.5, 0.6, 0.7):
            for overlap in (0.05, 0.1, 0.2):  # link_conf fixed at 0.25
                tp = npred = ngt = 0
                for image, dets in teacher.items():
                    riders, bikes = teacher_boxes(dets, conf, conf, overlap)
                    pred = riders if name == "rider" else bikes
                    gt = gt_boxes(labels / (Path(image).stem + ".txt"), cls)
                    tp += match(pred, gt)
                    npred += len(pred)
                    ngt += len(gt)
                p, r = tp / max(npred, 1), tp / max(ngt, 1)
                results.append({"conf": conf, "overlap": overlap, "precision": round(p, 4), "recall": round(r, 4),
                                "f1": round(2 * p * r / max(p + r, 1e-9), 4)})
        best[name] = max(results, key=lambda item: (item["precision"] >= 0.8, item["f1"]))
        best[name + "_grid"] = results
    return best


def plate_quality(plates, labels, conf):
    tp = npred = ngt = 0
    for image, dets in plates.items():
        pred = [d[2:] for d in dets if d[0] == 2 and d[1] >= conf]
        gt = gt_boxes(labels / (Path(image).stem + ".txt"), 2)
        tp += match(pred, gt)
        npred += len(pred)
        ngt += len(gt)
    p, r = tp / max(npred, 1), tp / max(ngt, 1)
    return {"conf": conf, "precision": round(p, 4), "recall": round(r, 4)}


def yolo_line(cls, box):
    x1, y1, x2, y2 = [min(1.0, max(0.0, v)) for v in box]
    return f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/datasets/merged_v4")
    parser.add_argument("--teacher", default="yolov8x.pt")
    parser.add_argument("--plate-model", default=str(ROOT / "runs/detect/helmet_v3/weights/best.pt"))
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    train_images = sorted((SOURCE / "train/images").iterdir())
    labels = SOURCE / "train/labels"
    teacher = detect(args.teacher, train_images, [0, 3], CACHE / "teacher_train.json", args.device)
    plates = detect(args.plate_model, train_images, [2], CACHE / "plates_train.json", args.device)

    ds1_teacher = {k: v for k, v in teacher.items() if k.startswith("ds1_")}
    rules = tune(ds1_teacher, labels)
    ds_with_plates = {k: v for k, v in plates.items() if k.startswith(("ds1_", "ds2_"))}
    plate_grid = [plate_quality(ds_with_plates, labels, c) for c in (0.5, 0.6, 0.7, 0.8)]
    plate_rule = max(plate_grid, key=lambda item: (item["precision"] >= 0.85, item["recall"]))

    if args.output.exists():
        shutil.rmtree(args.output)
    added = {"ds2": {"rider": 0, "bike": 0}, "ds3": {"rider": 0, "bike": 0, "number-plate": 0}}
    for split in ("train", "valid", "test"):
        (args.output / split / "images").mkdir(parents=True)
        (args.output / split / "labels").mkdir(parents=True)
        for image in sorted((SOURCE / split / "images").iterdir()):
            os.symlink(image, args.output / split / "images" / image.name)
            label = SOURCE / split / "labels" / (image.stem + ".txt")
            lines = label.read_text().splitlines() if label.is_file() else []
            prefix = image.name.split("_", 1)[0]
            if split == "train" and prefix in MISSING:
                riders, bikes = teacher_boxes(teacher.get(image.name, []), rules["rider"]["conf"],
                                              rules["bike"]["conf"], rules["rider"]["overlap"])
                lines += [yolo_line(IDS["rider"], b) for b in riders] + [yolo_line(IDS["bike"], b) for b in bikes]
                added[prefix]["rider"] += len(riders)
                added[prefix]["bike"] += len(bikes)
                if "number-plate" in MISSING[prefix]:
                    found = [d[2:] for d in plates.get(image.name, []) if d[0] == 2 and d[1] >= plate_rule["conf"]]
                    lines += [yolo_line(IDS["number-plate"], b) for b in found]
                    added[prefix]["number-plate"] += len(found)
            (args.output / split / "labels" / (image.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    data = yaml.safe_load((SOURCE / "data.yaml").read_text())
    data.update(path=str(args.output), train="train/images", val="valid/images", test="test/images")
    (args.output / "data.yaml").write_text(yaml.safe_dump(data))
    summary = {"rules": {k: v for k, v in rules.items() if not k.endswith("_grid")},
               "rule_grids": {k: v for k, v in rules.items() if k.endswith("_grid")},
               "plate_rule": plate_rule, "plate_grid": plate_grid, "added_train_instances": added,
               "dataset": str(args.output),
               "note": "Teacher rules tuned on ds1 train ground truth; validation/test labels are unchanged."}
    (CACHE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("rules", "plate_rule", "added_train_instances")}, indent=1))


if __name__ == "__main__":
    main()
