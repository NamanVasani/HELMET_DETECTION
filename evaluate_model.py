"""Evaluate recorded YOLO splits; tune on validation and freeze thresholds for test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from hi import EXPECTED_CLASSES, MODEL_PATH, INFERENCE_SIZE


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def thresholds_from_validation(metrics):
    thresholds = {}
    for row, class_id in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(class_id)]
        if name in {"helmet", "no-helmet"}:
            index = int(np.argmax(metrics.box.f1_curve[row]))
            thresholds[name] = round(float(metrics.box.px[index]), 4)
    if set(thresholds) != {"helmet", "no-helmet"}:
        raise ValueError("Validation must include both helmet classes")
    return thresholds


def split_audit(dataset):
    root = Path(dataset.get("path", "."))
    seen, names, overlaps, counts, eligible = {}, {}, [], {}, {}
    for split in ("train", "val", "test"):
        images = sorted((root / dataset[split]).glob("*"))
        images = [path for path in images if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        counts[split] = len(images)
        eligible[split] = []
        for path in images:
            digest = sha256(path)
            source_name = re.sub(r"^ds\d+_", "", path.name).split(".rf.")[0]
            previous = seen.get(digest) or names.get(source_name)
            if previous and split != "train":
                overlaps.append({"first_split": previous[0], "second_split": split,
                                 "first_image": previous[1], "second_image": path.name,
                                 "reason": "identical bytes" if digest in seen else "shared source filename"})
            else:
                eligible[split].append(str(path.resolve()))
            seen.setdefault(digest, (split, path.name))
            names.setdefault(source_name, (split, path.name))
    return {"images": counts, "eligible_images": {split: len(paths) for split, paths in eligible.items()},
            "excluded_overlaps": overlaps,
            "limit": ("Byte hashes and source names cannot exclude all augmented duplicates "
                      "or prove scene independence.")}, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "merged_dataset/data.yaml")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reports")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    dataset = yaml.safe_load(args.data.read_text())
    if dataset["names"] != EXPECTED_CLASSES:
        raise ValueError("Dataset classes do not match the detector")
    if "path" not in dataset or not Path(dataset["path"]).is_absolute():
        dataset["path"] = str((args.data.parent / dataset.get("path", ".")).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    audit, eligible = split_audit(dataset)
    for split in ("val", "test"):
        if not eligible[split]:
            raise ValueError(f"No nonoverlapping {split} images remain")
        listing = (args.output / f"{split}-images.txt").resolve()
        listing.write_text("\n".join(eligible[split]) + "\n")
        dataset[split] = str(listing)
    resolved = args.output / "evaluation-data.yaml"
    resolved.write_text(yaml.safe_dump(dataset))
    report = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "model_sha256": sha256(args.model),
              "imgsz": INFERENCE_SIZE, "device": args.device, "iou_nms": 0.4,
              "dataset_audit": audit, "splits": {},
              "limitations": ["These source dataset splits are not new independent road footage.",
                              "Mixed annotations and augmented scene leakage can affect results.",
                              "Full-frame metrics do not validate zoom-crop thresholds or temporal event accuracy."]}
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
    model = YOLO(args.model)
    for split in ("val", "test"):
        metrics = model.val(data=str(resolved), split=split, imgsz=INFERENCE_SIZE, batch=4,
                            device=args.device, workers=0, plots=False, conf=0.001, iou=0.4,
                            project=str(args.output), name=split, exist_ok=True, verbose=False)
        if split == "val":
            report["full_frame_thresholds"] = thresholds_from_validation(metrics)
        fixed = {}
        for row, class_id in enumerate(metrics.box.ap_class_index):
            name = metrics.names[int(class_id)]
            if name in report["full_frame_thresholds"]:
                threshold = report["full_frame_thresholds"][name]
                fixed[name] = {"threshold": threshold,
                               "precision": float(np.interp(threshold, metrics.box.px, metrics.box.p_curve[row])),
                               "recall": float(np.interp(threshold, metrics.box.px, metrics.box.r_curve[row]))}
        report["splits"][split] = {"per_class": metrics.summary(), "metrics": metrics.results_dict,
                                    "speed_ms_per_image": metrics.speed,
                                    "interpolated_at_validation_thresholds": fixed}
        text = json.dumps(report, indent=2, default=lambda value: value.item())
        (args.output / "evaluation.json").write_text(text)
        print(f"Saved {split} evaluation to {args.output / 'evaluation.json'}", flush=True)
    calibration = {"model_sha256": report["model_sha256"], "source": "validation maximum class F1 after excluding known overlaps",
                   "imgsz": INFERENCE_SIZE, "full_frame": report["full_frame_thresholds"],
                   "zoom_crop": 0.50, "zoom_limit": "Uncalibrated; needs labeled crop footage."}
    (args.output / "thresholds.json").write_text(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
