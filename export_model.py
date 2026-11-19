"""Export a trusted helmet checkpoint to CoreML (Apple Neural Engine/GPU) and/or ONNX.

The source .pt is copied into runs/export/<name>/ first so the trained run folder is never modified.
Each export is checked: identical class names and the exported file's SHA-256 are written to a manifest.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
EXPECTED_CLASSES = ["helmet", "no-helmet", "number-plate", "rider", "bike"]


def digest(path):
    path = Path(path)
    if path.is_dir():
        sha = hashlib.sha256()
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            sha.update(item.relative_to(path).as_posix().encode())
            sha.update(item.read_bytes())
        return sha.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "runs/detect/helmet_v3/weights/best.pt")
    parser.add_argument("--name", default="helmet_v3")
    parser.add_argument("--formats", nargs="+", default=["coreml", "onnx"], choices=["coreml", "onnx"])
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--half", action="store_true", help="fp16 weights (CoreML)")
    args = parser.parse_args()
    folder = ROOT / "runs/export" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "model.pt"
    shutil.copy2(args.model, source)
    manifest = {"source_checkpoint": str(args.model), "source_sha256": digest(args.model),
                "imgsz": args.imgsz, "exports": {}}
    for fmt in args.formats:
        options = {"format": fmt, "imgsz": args.imgsz, "nms": False}
        if fmt == "coreml":
            options["half"] = args.half
        if fmt == "onnx":
            options.update(simplify=True, dynamic=False)
        exported = Path(YOLO(source).export(**options))
        names = [YOLO(exported, task="detect").names[i] for i in range(len(EXPECTED_CLASSES))]
        if names != EXPECTED_CLASSES:
            raise SystemExit(f"{fmt}: class names changed to {names}")
        manifest["exports"][fmt] = {"path": str(exported.relative_to(ROOT)), "sha256": digest(exported),
                                    "classes": names, "half": bool(options.get("half"))}
        print(f"{fmt}: {exported}")
    (folder / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
