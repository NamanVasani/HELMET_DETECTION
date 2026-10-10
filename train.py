from pathlib import Path
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
   data=str(Path(__file__).resolve().parent / "merged_dataset/data.yaml"),
    epochs=20,
    imgsz=640,
    batch=8,
    device='mps',
    name='helmet_v2'
)

print("Training done!")