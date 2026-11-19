"""Benchmark helmet-model runtimes on this machine and check exported models agree with the .pt.

Frames are 1280x720 resizes of eligible test images (the app's camera resolution); inference uses
imgsz 800 like app.py. Reports median latency/FPS for predict and for track (ByteTrack), plus
detection agreement against the .pt CPU reference (same class, IoU >= 0.7, score >= 0.25).
"""
import argparse
from datetime import datetime, timezone
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def boxes(result, conf=0.25):
    return [(int(c), float(s), b) for c, s, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                                    result.boxes.xyxy.tolist(), strict=False) if s >= conf]


def agreement(reference, candidate):
    matched = total = 0
    score_diffs = []
    for ref, cand in zip(reference, candidate, strict=False):
        used = set()
        for cls, score, box in ref:
            total += 1
            best = max(((iou(box, b), i, s) for i, (c, s, b) in enumerate(cand) if c == cls and i not in used),
                       default=(0, None, 0))
            if best[0] >= 0.7:
                matched += 1
                used.add(best[1])
                score_diffs.append(abs(score - best[2]))
    extra = sum(len(c) for c in candidate) - matched
    return {"reference_boxes": total, "matched": matched, "match_rate": round(matched / max(total, 1), 4),
            "unmatched_candidate_boxes": extra,
            "median_abs_score_diff": round(statistics.median(score_diffs), 4) if score_diffs else None}


def run(model_path, device, frames, warmup=5, track=True):
    model = YOLO(model_path, task="detect")
    kwargs = {"imgsz": 800, "conf": 0.15, "iou": 0.4, "verbose": False}
    if device:
        kwargs["device"] = device
    for frame in frames[:warmup]:
        model.predict(frame, **kwargs)
    times, outputs = [], []
    for frame in frames:
        start = time.perf_counter()
        result = model.predict(frame, **kwargs)[0]
        times.append(time.perf_counter() - start)
        outputs.append(boxes(result))
    report = {"predict_median_ms": round(statistics.median(times) * 1000, 2),
              "predict_fps": round(1 / statistics.median(times), 2)}
    if track:
        track_times = []
        for frame in frames:
            start = time.perf_counter()
            model.track(frame, persist=True, tracker="bytetrack.yaml", **kwargs)
            track_times.append(time.perf_counter() - start)
        report.update(track_median_ms=round(statistics.median(track_times) * 1000, 2),
                      track_fps=round(1 / statistics.median(track_times), 2), track_works=True)
    report["names"] = [model.names[i] for i in range(len(model.names))]
    return report, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--export-dir", type=Path, default=ROOT / "runs/export/helmet_v3")
    parser.add_argument("--pt", type=Path, default=ROOT / "runs/detect/helmet_v3/weights/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/speed-2026-11-19.json")
    args = parser.parse_args()
    images = [Path(p) for p in (ROOT / "reports/test-images.txt").read_text().split()][:args.frames]
    frames = [cv2.resize(cv2.imread(str(p)), (1280, 720)) for p in images]
    candidates = [("pt-cpu", args.pt, "cpu"), ("pt-mps", args.pt, "mps")]
    for pattern, name in (("*.mlpackage", "coreml"), ("*.onnx", "onnx-cpu")):
        found = sorted(args.export_dir.glob(pattern))
        if found:
            candidates.append((name, found[0], None))
    results = {}
    reference = None
    for name, path, device in candidates:
        try:
            report, outputs = run(path, device, frames)
        except Exception as exc:  # report runtime failures instead of hiding them
            results[name] = {"model": str(path), "error": f"{type(exc).__name__}: {exc}"}
            continue
        if reference is None:
            reference = outputs
        report["agreement_vs_pt_cpu"] = agreement(reference, outputs)
        report["model"] = str(Path(path).relative_to(ROOT))
        results[name] = report
        print(name, json.dumps({k: v for k, v in report.items() if k != "names"}), flush=True)
    out = {"measured_at": datetime.now(timezone.utc).isoformat(), "machine": platform.platform(),
           "processor": platform.processor(), "frames": len(frames), "frame_size": [1280, 720], "imgsz": 800,
           "results": results,
           "note": "Model-only latency (predict/track). The live app also runs YuNet, optional zoom inference, "
                   "drawing and JPEG encoding, so end-to-end FPS is lower."}
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
