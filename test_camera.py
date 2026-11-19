"""Bounded physical-camera check. Frames remain in memory and are never saved."""
import argparse
import json
import platform
from pathlib import Path
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'reports/camera.json')
    args = parser.parse_args()
    if not 0 < args.seconds <= 30 or args.index < 0:
        parser.error('Use a positive camera index and duration up to 30 seconds')
    backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.index, backend)
    start, frames, size = time.monotonic(), 0, None
    try:
        if cap.isOpened():
            while time.monotonic() - start < args.seconds:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                size = [frame.shape[1], frame.shape[0]]
    finally:
        cap.release()
    elapsed = time.monotonic() - start
    report = {'physical_camera': 'passed' if frames else 'blocked', 'frames': frames, 'size': size,
              'read_fps': round(frames / max(elapsed, 0.001), 2), 'images_saved': False,
              'limit': 'Camera access only; this does not validate road-scene detection accuracy.'}
    if not frames:
        report['reason'] = 'No readable frames. Check macOS camera permission, camera availability, and other apps.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if frames else 1


if __name__ == '__main__':
    raise SystemExit(main())
