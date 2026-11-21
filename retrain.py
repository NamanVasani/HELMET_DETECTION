"""One-command retraining that never replaces the current model by itself.

  .venv/bin/python retrain.py --dry-run          # show what would happen
  .venv/bin/python retrain.py --hours 2          # export reviews, train, evaluate, compare

Steps:
  1. Export human-reviewed events (confirmed + false positives) as YOLO data (manage.training_export).
  2. Build a training set: original training images (+ pseudo-labelled rider/bike/plate set when present)
     + reviewed events. Validation and test stay the original human-labelled splits, so the comparison is fair.
  3. Fine-tune from the current best.pt with a low learning rate (default 0.001) inside a time budget.
  4. Evaluate current and candidate with evaluate_model.py on the same test split.
  5. Recommend the candidate only if no class gets worse (mAP50-95) and the average improves.
The current best.pt, model-manifest.json and thresholds.json are never modified.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import yaml

PROJECT = Path(__file__).resolve().parent
CLASSES = ["helmet", "no-helmet", "number-plate", "rider", "bike"]
TOLERANCE = 0.005  # mAP50-95 noise allowed before a class counts as "worse"


def build_dataset(base_yaml, reviewed_dir, out_dir):
    """Write a data.yaml whose train list = base train (+ reviewed images); val/test unchanged."""
    base = yaml.safe_load(Path(base_yaml).read_text())
    root = Path(base.get("path") or Path(base_yaml).parent)
    if not root.is_absolute():
        root = (Path(base_yaml).parent / root).resolve()

    def resolve(entry):
        entries = entry if isinstance(entry, list) else [entry]
        return [str(Path(item) if Path(item).is_absolute() else root / item) for item in entries]

    train = resolve(base["train"])
    if reviewed_dir and (Path(reviewed_dir) / "images").is_dir() and any((Path(reviewed_dir) / "images").iterdir()):
        train.append(str((Path(reviewed_dir) / "images").resolve()))
    data = {"path": str(root), "train": train, "val": resolve(base["val"])[0], "test": resolve(base["test"])[0],
            "names": {index: name for index, name in enumerate(CLASSES)}}
    names = base.get("names")
    listed = list(names.values()) if isinstance(names, dict) else names
    if listed != CLASSES:
        raise ValueError(f"Base dataset classes {listed} do not match {CLASSES}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return out_dir / "data.yaml"


def per_class(report_path):
    report = json.loads(Path(report_path).read_text())
    rows = report["splits"]["test"]["per_class"]
    return {row["Class"]: float(row["mAP50-95"]) for row in rows}


def compare(current, candidate, tolerance=TOLERANCE):
    """Return (adopt?, rows, reason). Adopt only if no class drops by more than `tolerance` and the mean rises."""
    rows, worse = [], []
    for name in CLASSES:
        old, new = current.get(name), candidate.get(name)
        if old is None or new is None:
            continue
        rows.append({"class": name, "current": round(old, 4), "candidate": round(new, 4), "change": round(new - old, 4)})
        if new < old - tolerance:
            worse.append(name)
    mean_old = sum(r["current"] for r in rows) / len(rows)
    mean_new = sum(r["candidate"] for r in rows) / len(rows)
    if worse:
        return False, rows, f"Not better: {', '.join(worse)} got worse. Keep the current model."
    if mean_new <= mean_old:
        return False, rows, "Not better on average. Keep the current model."
    return True, rows, f"Better: average mAP50-95 {mean_old:.4f} -> {mean_new:.4f} and no class got worse."


def run(command, dry_run):
    print("  $ " + " ".join(str(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in command], check=True, cwd=PROJECT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=2.0, help="training time budget")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--min-reviewed", type=int, default=100, help="refuse to train with fewer reviewed events")
    parser.add_argument("--force", action="store_true", help="train even below --min-reviewed")
    parser.add_argument("--dry-run", action="store_true", help="prepare data and print the plan; do not train")
    parser.add_argument("--storage", type=Path, default=None, help="event storage (default: HELMET_STORAGE / ./storage)")
    parser.add_argument("--base", type=Path, default=None,
                        help="training data.yaml (default: runs/datasets/merged_v4, else merged_dataset)")
    parser.add_argument("--eval-data", type=Path, default=PROJECT / "merged_dataset/data.yaml",
                        help="human-labelled data.yaml used to compare both models")
    parser.add_argument("--work", type=Path, default=None, help="output folder (default: runs/retrain/<timestamp>)")
    args = parser.parse_args(argv)

    import manage
    from event_store import STORAGE
    storage = args.storage or STORAGE
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    work = args.work or PROJECT / "runs" / "retrain" / stamp
    python = sys.executable

    print(f"1/5 Export reviewed events from {storage}")
    counts = manage.training_export(storage, work / "reviewed")
    reviewed = counts["confirmed"] + counts["false_positive"]
    print(f"    {counts['confirmed']} confirmed, {counts['false_positive']} false positives ({counts['skipped']} skipped)")
    if reviewed < args.min_reviewed and not args.force:
        print(f"    Only {reviewed} reviewed events; at least {args.min_reviewed} are needed for a useful retrain.\n"
              "    Review more events (Quick review), record real gate footage, or pass --force.")
        return 2

    base = args.base or PROJECT / "runs/datasets/merged_v4/data.yaml"
    if not base.is_file():
        base = PROJECT / "merged_dataset/data.yaml"
    print(f"2/5 Build training set from {base} + reviewed events")
    data_yaml = build_dataset(base, work / "reviewed", work / "dataset")
    print(f"    {data_yaml}")

    current = PROJECT / "best.pt"
    name = f"helmet_retrain_{stamp}"
    candidate = PROJECT / "runs/detect" / name / "weights/best.pt"
    print(f"3/5 Fine-tune from {current.name} (lr0={args.lr0}, {args.hours} h)")
    run([python, "train.py", "--data", data_yaml, "--model", current, "--name", name, "--hours", args.hours,
         "--lr0", args.lr0, "--batch", 16], args.dry_run)

    print("4/5 Evaluate current and candidate on the same human-labelled test split")
    for label, model in (("current", current), ("candidate", candidate)):
        run([python, "evaluate_model.py", "--model", model, "--data", args.eval_data, "--output", work / f"eval-{label}"],
            args.dry_run)

    print("5/5 Compare")
    if args.dry_run:
        print("    (dry run: nothing trained or evaluated)")
        return 0
    adopt, rows, reason = compare(per_class(work / "eval-current/evaluation.json"),
                                  per_class(work / "eval-candidate/evaluation.json"))
    for row in rows:
        print(f"    {row['class']:13} {row['current']:.4f} -> {row['candidate']:.4f} ({row['change']:+.4f})")
    print("    " + reason)
    (work / "comparison.json").write_text(json.dumps({"adopt": adopt, "reason": reason, "rows": rows,
                                                      "candidate": str(candidate)}, indent=2))
    if adopt:
        print("\n    To adopt it (your decision):\n"
              f"      1. Copy {candidate.relative_to(PROJECT)} to best.pt\n"
              "      2. Update model-manifest.json sha256, then run evaluate_model.py to bind new thresholds\n"
              "      3. Re-run the tests and python install_service.py --replace-pid ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
