"""Local timing check: no camera access, HTTP server, or saved detection events."""
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hi import FACE_MODEL_PATH, MODEL_PATH, detect_faces

sample = sorted(Path("storage/2026-10-13").glob("*/event_*/event.jpg"))[-1]
frame = cv2.imread(str(sample))
assert frame is not None and frame.size
frame = cv2.resize(frame, (1280, 720))
rows = []
for device in ("cpu", "mps"):
    if device == "mps" and not torch.backends.mps.is_available():
        continue
    model = YOLO(MODEL_PATH)
    for size in (800, 640):
        model.predict(frame, imgsz=size, device=device, verbose=False)
        timings = []
        for _ in range(6):
            start = time.perf_counter()
            result = model.predict(frame, imgsz=size, device=device, verbose=False)[0]
            timings.append((time.perf_counter() - start) * 1000)
        assert all(value > 0 for value in timings)
        row = {"device": device, "size": size, "median_ms": round(statistics.median(timings), 1),
               "head_predictions": [model.names[int(box.cls[0])] for box in result.boxes
                                    if int(box.cls[0]) in (0, 1)]}
        rows.append(row)
        print(json.dumps(row), flush=True)

detector = cv2.FaceDetectorYN.create(str(FACE_MODEL_PATH), "", (1280, 720), 0.6, 0.3, 5000)
for threads in (cv2.getNumThreads(), 1):
    cv2.setNumThreads(threads)
    detect_faces(frame, detector)
    timings = []
    for _ in range(6):
        start = time.perf_counter()
        faces = detect_faces(frame, detector)
        timings.append((time.perf_counter() - start) * 1000)
    print(json.dumps({"face_threads": threads, "median_ms": round(statistics.median(timings), 1),
                      "face_count": len(faces)}), flush=True)
