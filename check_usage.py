"""Real-model HTTP/video replay in disposable storage; no physical camera or user events."""
import argparse
from datetime import date
import json
from pathlib import Path
import tempfile
import threading
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np

from dashboard import DashboardServer
from event_store import list_events
from hi import run_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path(__file__).parent / "reports/test-images.txt")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reports/usage.json")
    args = parser.parse_args()
    candidates = []
    for raw in args.images.read_text().splitlines():
        path = Path(raw)
        labels = Path(str(path.parent).replace("/images", "/labels")) / (path.stem + ".txt")
        heads = sum(line.split()[0] in {"0", "1"} for line in labels.read_text().splitlines() if line.strip())
        candidates.append((heads, path))
    heads, sample = max(candidates)
    assert heads >= 2, "Replay needs a labeled scene containing several heads"
    image = cv2.imread(str(sample))
    assert image is not None
    image = cv2.resize(image, (1280, 720))
    dark = (image.astype(np.float32) * 0.35).astype(np.uint8)
    distant = np.zeros_like(image)
    distant[180:540, 320:960] = cv2.resize(image, (640, 360))
    rows, failures = [], []
    with tempfile.TemporaryDirectory(prefix="helmet_usage_") as temp:
        root = Path(temp)
        server = DashboardServer(port=0, storage=root / "events").start()
        base = f"http://127.0.0.1:{server.server.server_address[1]}"

        def detect():
            try:
                run_detector(server)
            except Exception as exc:
                failures.append(str(exc))

        worker = threading.Thread(target=detect, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 30
            while not server.accept_video and not failures and time.monotonic() < deadline:
                time.sleep(0.1)
            assert server.accept_video and not failures, failures
            for name, frame in (("multiple_people", image), ("synthetic_low_light", dark), ("synthetic_distance", distant)):
                video = root / f"{name}.mp4"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
                assert writer.isOpened()
                for _ in range(24):
                    writer.write(frame)
                writer.release()
                previous_ids = {event["id"] for event in list_events(date.today().isoformat(), server.storage)}
                start = time.monotonic()
                with urlopen(Request(base + "/api/video", data=video.read_bytes(),
                                    headers={"X-File-Name": video.name}), timeout=10) as response:
                    assert response.status == 202
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    with urlopen(base + "/api/status", timeout=3) as response:
                        status = json.load(response)
                    if status["video"]["state"] in {"complete", "failed"}:
                        break
                    assert not failures, failures
                    time.sleep(0.1)
                assert status["video"]["state"] == "complete", status
                elapsed = time.monotonic() - start
                events = list_events(date.today().isoformat(), server.storage)
                assert all(event["confirmation_frames"] == 3 for event in events)
                added = [event for event in events if event["id"] not in previous_ids]
                assert all(event["source"]["filename"] == video.name
                           and event["source"]["type"] == "video"
                           and 0 <= event["source"]["frame"] < 24
                           and event["source"]["position_seconds"] is not None for event in added)
                rows.append({"scenario": name, "frames": 24, "elapsed_seconds": round(elapsed, 3),
                             "processed_fps": round(24 / elapsed, 2), "saved_events": len(added)})
                print(json.dumps(rows[-1]), flush=True)
            assert rows[0]["saved_events"] > 0, "The baseline scene produced no saved event"
            ok, jpeg = cv2.imencode(".jpg", image)
            assert ok
            for _ in range(12):
                with urlopen(Request(base + "/api/frame", data=jpeg.tobytes(),
                                    headers={"Content-Type": "image/jpeg"}), timeout=3) as response:
                    assert response.status == 202
                time.sleep(0.2)
            with urlopen(base + "/stream", timeout=3) as response:
                assert b"Content-Type: image/jpeg" in response.read(80)
            assert not failures, failures
        finally:
            server.restart_detector.set()
            worker.join(timeout=20)
            server.close()
    report = {"source_image": sample.name, "labeled_heads": heads, "scenarios": rows,
              "browser_jpeg_endpoint": "passed", "mjpeg_preview": "passed",
              "limit": "Repeated dataset images and synthetic perturbations; not independent footage or a physical camera test."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
