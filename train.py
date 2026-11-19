"""Deliberate training run using the dataset in this project folder."""
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO


def main():
    project = Path(__file__).resolve().parent
    source = project / 'merged_dataset/data.yaml'
    data = yaml.safe_load(source.read_text())
    data['path'] = str(source.parent)
    config = project / 'runs/training-data.yaml'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(data))
    model = YOLO(str(project / 'yolov8n.pt'))
    model.train(data=str(config), epochs=20, imgsz=640, batch=8,
                device='mps' if torch.backends.mps.is_available() else 'cpu', name='helmet_v2')
    print('Training done!')


if __name__ == '__main__':
    main()
