import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import retrain
from event_store import associate_objects, save_event, update_review


def base_dataset(folder):
    for split in ("train", "valid", "test"):
        (folder / split / "images").mkdir(parents=True)
    (folder / "data.yaml").write_text(yaml.safe_dump({
        "path": str(folder), "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": ["helmet", "no-helmet", "number-plate", "rider", "bike"]}))
    return folder / "data.yaml"


class RetrainTests(unittest.TestCase):
    def test_compare_keeps_current_model_unless_every_class_holds(self):
        current = {"helmet": 0.48, "no-helmet": 0.52, "number-plate": 0.50, "rider": 0.32, "bike": 0.40}
        adopt, rows, reason = retrain.compare(current, dict(current, rider=0.45, bike=0.50))
        self.assertTrue(adopt, reason)
        adopt, _, reason = retrain.compare(current, dict(current, rider=0.60, helmet=0.469))  # last time's pattern
        self.assertFalse(adopt)
        self.assertIn("helmet", reason)
        adopt, _, _ = retrain.compare(current, dict(current, helmet=0.478))  # tiny noise, no gain
        self.assertFalse(adopt)
        self.assertEqual(len(rows), 5)

    def test_dataset_adds_reviewed_images_to_train_only(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            base = base_dataset(temp / "base")
            reviewed = temp / "reviewed" / "images"
            reviewed.mkdir(parents=True)
            (reviewed / "a.jpg").write_bytes(b"x")
            data = yaml.safe_load(retrain.build_dataset(base, temp / "reviewed", temp / "out").read_text())
            self.assertEqual(data["train"], [str(temp / "base/train/images"), str(reviewed.resolve())])
            self.assertEqual(data["val"], str(temp / "base/valid/images"))
            self.assertEqual(data["test"], str(temp / "base/test/images"))

    def test_refuses_with_too_few_reviews_and_dry_run_never_trains(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            storage = temp / "storage"
            head = {"class_name": "no-helmet", "confidence": 0.9, "track_id": 1, "box": [10, 10, 40, 40]}
            event = save_event(np.random.default_rng(1).integers(0, 255, (80, 100, 3), dtype=np.uint8), head, [head],
                               associate_objects(head, [head]), root=storage)
            update_review(event["date"], event["id"], "confirmed", root=storage)
            base = base_dataset(temp / "base")
            common = ["--storage", str(storage), "--base", str(base), "--eval-data", str(base)]
            with patch("retrain.subprocess.run") as run, redirect_stdout(io.StringIO()) as out:
                self.assertEqual(retrain.main(common + ["--work", str(temp / "w1")]), 2)
                self.assertIn("at least 100 are needed", out.getvalue())
                self.assertEqual(retrain.main(common + ["--work", str(temp / "w2"), "--force", "--dry-run"]), 0)
            run.assert_not_called()
            plan = out.getvalue()
            self.assertIn("train.py", plan)
            self.assertIn("--lr0 0.001", plan)
            self.assertIn("evaluate_model.py", plan)
            self.assertTrue((temp / "w2/dataset/data.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
