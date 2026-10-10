
import platform
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"
WEBCAM_INDEX = 0
WINDOW_NAME = "Helmet Detection"


def open_webcam():
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else 0
    cap = cv2.VideoCapture(WEBCAM_INDEX, backend)
    if not cap.isOpened():
        raise RuntimeError(
            "Webcam not found or camera access denied.\n"
            "Fix: System Settings > Privacy & Security > Camera\n"
            "Enable Terminal (or Cursor), then restart the terminal and try again."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def show_camera_test(cap, seconds=3):
    """Show raw webcam first so you know the window works before AI loads."""
    print(f"Showing live camera for {seconds}s (no AI yet)...")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()

    end = time.time() + seconds
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera opened but cannot read frames.")

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            sys.exit(0)

    print("Camera window OK.")


def main():
    print("Step 1/3: Opening webcam...")
    cap = open_webcam()

    print("Step 2/3: Testing camera window...")
    print("If you do NOT see a video window in 3 seconds, read the note below.")
    show_camera_test(cap, seconds=3)

    print("Step 3/3: Loading AI model...")
    model = YOLO(MODEL_PATH)
    print("Detection running. Press Q or Esc in the video window to quit.")

    frame_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost webcam connection.")
            break

        results = model(
            frame,
            conf=0.80,
            iou=0.4,
            imgsz=640,
            classes=[0,1,2],
            verbose=False,
        )

        cv2.imshow(WINDOW_NAME, results[0].plot())

        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / max(time.time() - start, 0.001)
            print(f"Running... ~{fps:.1f} FPS (press Q in video window to stop)")

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
