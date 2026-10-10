import os
import shutil
import glob
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# ─────────────────────────────────────────
# STEP 1 — DOWNLOAD ALL 3 DATASETS
# ─────────────────────────────────────────

print("\n" + "="*50)
print("STEP 1: Downloading all datasets...")
print("="*50)

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Set the ROBOFLOW_API_KEY environment variable first.")
rf = Roboflow(api_key=api_key)

# Dataset 1 — Traffic Helmet Detection (793 images)
print("\n[1/3] Downloading Traffic Helmet Detection...")
project1 = rf.workspace("patelkrishuday2139").project("traffic-helmet-detection")
dataset1 = project1.version(3).download("yolov8")
print(f"Dataset 1 saved at: {dataset1.location}")

# Dataset 2 — Helmet and no helmet rider detection (1533 images)
print("\n[2/3] Downloading Helmet and no helmet rider detection...")
project2 = rf.workspace("gw-khadatkar-and-sv-wasule").project("helmet-and-no-helmet-rider-detection")
dataset2 = project2.version(6).download("yolov8")
print(f"Dataset 2 saved at: {dataset2.location}")

# Dataset 3 — Helmet Detection YOLOv8 (1737 images)
print("\n[3/3] Downloading Helmet Detection YOLOv8...")
project3 = rf.workspace("learning-evidence").project("helmet-detection_yolov8")
dataset3 = project3.version(3).download("yolov8")
print(f"Dataset 3 saved at: {dataset3.location}")

print("\n✅ All datasets downloaded!")

# ─────────────────────────────────────────
# STEP 2 — MERGE ALL DATASETS
# ─────────────────────────────────────────

print("\n" + "="*50)
print("STEP 2: Merging all datasets...")
print("="*50)

# Final merged dataset location
MERGED_DIR = os.path.join(os.path.dirname(__file__), "merged_dataset")

# Create folder structure
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(MERGED_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(MERGED_DIR, split, "labels"), exist_ok=True)

print(f"Merged dataset folder: {MERGED_DIR}")

# ─────────────────────────────────────────
# CLASS MAPPING
# ─────────────────────────────────────────
# Final unified classes:
# 0 = helmet
# 1 = no-helmet
# 2 = number-plate
# 3 = rider
# 4 = bike

# Dataset 1 class mapping (Helmet, bike, no-helmet, number-plate, rider)
# 0=Helmet → 0, 1=bike → 4, 2=no-helmet → 1, 3=number-plate → 2, 4=rider → 3
DS1_MAP = {0: 0, 1: 4, 2: 1, 3: 2, 4: 3}

# Dataset 2 class mapping (With Helmet, Without Helmet, Licence Plate)
# 0=With Helmet → 0, 1=Without Helmet → 1, 2=Licence Plate → 2
DS2_MAP = {0: 0, 1: 1, 2: 2}

# Dataset 3 class mapping (With helmet, Without helmet)
# 0=With helmet → 0, 1=Without helmet → 1
DS3_MAP = {0: 0, 1: 1}


def remap_and_copy(dataset_path, class_map, dataset_name):
    """Copy images and remap label class IDs to unified classes."""
    copied = 0
    for split in ["train", "valid", "test"]:
        img_src = os.path.join(dataset_path, split, "images")
        lbl_src = os.path.join(dataset_path, split, "labels")

        # Some datasets use 'val' instead of 'valid'
        if not os.path.exists(img_src):
            img_src = os.path.join(dataset_path, split.replace("valid", "val"), "images")
            lbl_src = os.path.join(dataset_path, split.replace("valid", "val"), "labels")

        if not os.path.exists(img_src):
            continue

        img_dst = os.path.join(MERGED_DIR, split, "images")
        lbl_dst = os.path.join(MERGED_DIR, split, "labels")

        # Copy images
        for img_file in glob.glob(os.path.join(img_src, "*")):
            fname = f"{dataset_name}_{os.path.basename(img_file)}"
            shutil.copy2(img_file, os.path.join(img_dst, fname))

        # Copy + remap labels
        for lbl_file in glob.glob(os.path.join(lbl_src, "*.txt")):
            fname = f"{dataset_name}_{os.path.basename(lbl_file)}"
            out_path = os.path.join(lbl_dst, fname)

            with open(lbl_file, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                new_cls = class_map.get(old_cls, None)
                if new_cls is not None:
                    new_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")

            with open(out_path, "w") as f:
                f.writelines(new_lines)

            copied += 1

    print(f"  {dataset_name}: {copied} label files copied and remapped ✅")


print("\nRemapping and copying datasets...")
remap_and_copy(dataset1.location, DS1_MAP, "ds1")
remap_and_copy(dataset2.location, DS2_MAP, "ds2")
remap_and_copy(dataset3.location, DS3_MAP, "ds3")

# ─────────────────────────────────────────
# STEP 3 — CREATE UNIFIED data.yaml
# ─────────────────────────────────────────

print("\nCreating unified data.yaml...")

data_yaml = {
    "train": os.path.join(MERGED_DIR, "train", "images"),
    "val":   os.path.join(MERGED_DIR, "valid", "images"),
    "test":  os.path.join(MERGED_DIR, "test",  "images"),
    "nc": 5,
    "names": ["helmet", "no-helmet", "number-plate", "rider", "bike"]
}

yaml_path = os.path.join(MERGED_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"data.yaml saved at: {yaml_path}")

# Count total images
total = len(glob.glob(os.path.join(MERGED_DIR, "train", "images", "*")))
print(f"\nTotal training images: {total}")

# ─────────────────────────────────────────
# STEP 4 — RETRAIN MODEL (FINE TUNING)
# ─────────────────────────────────────────

print("\n" + "="*50)
print("STEP 3: Starting fine tuning on merged dataset...")
print("="*50)

# Use your existing best.pt as starting point (fine tuning)
CURRENT_MODEL = os.path.join(os.path.dirname(__file__), "runs", "detect", "helmet_v2", "weights", "best.pt")

if os.path.exists(CURRENT_MODEL):
    print(f"Fine tuning from: {CURRENT_MODEL}")
    model = YOLO(CURRENT_MODEL)
else:
    print("Current model not found, starting from yolov8n.pt base...")
    model = YOLO("yolov8n.pt")

model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    name="helmet_v3",
    patience=10,       # stop early if no improvement for 10 epochs
    save=True,
    plots=True,
)

print("\n" + "="*50)
print("✅ TRAINING COMPLETE!")
print("New model saved at:")
print("runs/detect/helmet_v3/weights/best.pt")
print("="*50)
print("\nUpdate MODEL_PATH in hi.py to:")
print(os.path.join(os.path.dirname(__file__), "runs", "detect", "helmet_v3", "weights", "best.pt"))
