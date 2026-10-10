"""Run: python test_camera.py
Only tests webcam + window. No AI. Press Q to quit.
"""
import platform
import cv2

backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else 0
cap = cv2.VideoCapture(0, backend)

if not cap.isOpened():
    print("FAIL: Camera not opened. Check System Settings > Camera > Terminal.")
    raise SystemExit(1)

print("OK: Camera opened. You should see a window in 1 second. Press Q to quit.")

cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        print("FAIL: Cannot read frames.")
        break
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
print("DONE: Camera test finished.")
