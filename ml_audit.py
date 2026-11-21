"""Audit per-source label coverage in merged_dataset (detects partial annotation)."""
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
NAMES = ["helmet", "no-helmet", "number-plate", "rider", "bike"]


def audit(dataset=ROOT / "merged_dataset"):
    result = {}
    for split in ("train", "valid", "test"):
        stats = defaultdict(lambda: {"images": 0, "empty_label_images": 0,
                                     "instances": defaultdict(int), "images_with": defaultdict(int)})
        for image in sorted((dataset / split / "images").glob("*")):
            prefix = image.name.split("_", 1)[0]
            entry = stats[prefix]
            entry["images"] += 1
            label = dataset / split / "labels" / (image.stem + ".txt")
            classes = []
            if label.is_file():
                classes = [int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()]
            if not classes:
                entry["empty_label_images"] += 1
            for cls in classes:
                entry["instances"][NAMES[cls]] += 1
            for cls in set(classes):
                entry["images_with"][NAMES[cls]] += 1
        result[split] = {prefix: {"images": v["images"], "empty_label_images": v["empty_label_images"],
                                  "instances": {n: v["instances"].get(n, 0) for n in NAMES},
                                  "images_with": {n: v["images_with"].get(n, 0) for n in NAMES}}
                         for prefix, v in sorted(stats.items())}
    return result


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "merged_dataset"), indent=1))
