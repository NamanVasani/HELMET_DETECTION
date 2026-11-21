"""Deliberate training run using a dataset in this project folder.

Default (no flags) reproduces the original nano run on merged_dataset. The helmet_v4 candidate was
trained with the pseudo-labelled dataset built by pseudo_label.py:

    .venv/bin/python pseudo_label.py
    .venv/bin/python train.py --data runs/datasets/merged_v4/data.yaml --model yolov8s.pt \
        --name helmet_v4 --hours 2.5 --imgsz 640 --batch 16
"""
import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=project / "merged_dataset/data.yaml")
    parser.add_argument("--model", default=str(project / "yolov8n.pt"))
    parser.add_argument("--name", default="helmet_v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hours", type=float, default=None, help="time budget; overrides --epochs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr0", type=float, default=None,
                        help="initial learning rate; use ~0.001 when fine-tuning an already trained model")
    args = parser.parse_args()

    source = (project / args.data) if not args.data.is_absolute() else args.data
    data = yaml.safe_load(source.read_text())
    if not Path(data.get("path", "")).is_absolute():
        data["path"] = str(source.parent)
    config = project / f"runs/training-data-{args.name}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(data))
    options = dict(data=str(config), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                   patience=args.patience, project=str(project / "runs/detect"), name=args.name,
                   device="mps" if torch.backends.mps.is_available() else "cpu", workers=4, exist_ok=False)
    if args.hours:
        options["time"] = args.hours
    if args.lr0:
        options.update(lr0=args.lr0, warmup_epochs=0.5)  # gentle fine-tune: keep what the model already knows
    YOLO(args.model).train(**options)
    print("Training done!")


if __name__ == "__main__":
    main()
