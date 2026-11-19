from pathlib import Path
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from evaluate_model import split_audit, thresholds_from_validation
from hi import load_thresholds


class EvaluationTests(unittest.TestCase):
    def test_calibration_is_bound_to_checkpoint_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.pt"
            model.write_bytes(b"checkpoint")
            config = Path(temp) / "thresholds.json"
            calibration = {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                           "full_frame": {"helmet": 0.5, "no-helmet": 0.4}}
            config.write_text(json.dumps(calibration))
            self.assertEqual(load_thresholds(model, config), calibration["full_frame"])
            calibration["full_frame"]["helmet"] = True
            config.write_text(json.dumps(calibration))
            with self.assertRaises(ValueError):
                load_thresholds(model, config)
            model.write_bytes(b"other checkpoint")
            with self.assertRaises(ValueError):
                load_thresholds(model, config)

    def test_thresholds_use_validation_class_curves(self):
        metrics = SimpleNamespace(names={0: "helmet", 1: "no-helmet"}, box=SimpleNamespace(
            ap_class_index=np.array([0, 1]), px=np.array([0.1, 0.5, 0.9]),
            f1_curve=np.array([[0.2, 0.8, 0.3], [0.9, 0.8, 0.1]])))
        self.assertEqual(thresholds_from_validation(metrics), {"helmet": 0.5, "no-helmet": 0.1})

    def test_overlapping_bytes_and_source_names_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "val", "test"):
                (root / split).mkdir()
            (root / "train/ds1_scene.rf.123.jpg").write_bytes(b"train")
            (root / "val/ds2_scene.rf.456.jpg").write_bytes(b"augmented")
            (root / "val/other.jpg").write_bytes(b"unique")
            (root / "test/renamed.jpg").write_bytes(b"train")
            (root / "test/independent.jpg").write_bytes(b"test")
            audit, eligible = split_audit({"path": temp, "train": "train", "val": "val", "test": "test"})
            self.assertEqual(audit["eligible_images"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(len(audit["excluded_overlaps"]), 2)
            self.assertTrue(eligible["test"][0].endswith("independent.jpg"))


if __name__ == "__main__":
    unittest.main()
