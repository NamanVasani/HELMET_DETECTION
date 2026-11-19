import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import auth
import manage
from event_store import save_event, update_review


def event(root, name="no-helmet"):
    head = {"class_name": name, "confidence": 0.9, "track_id": 1, "box": [40, 20, 80, 60]}
    rider = {"class_name": "rider", "confidence": 0.8, "track_id": 2, "box": [30, 10, 120, 150]}
    frame = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    return save_event(frame, head, [head, rider], {"rider": rider, "bike": None, "plate": None}, root=root)


class ManageTests(unittest.TestCase):
    def test_training_export_keeps_confirmed_labels_and_drops_false_positive_heads(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as out:
            good, bad = event(temp), event(temp)
            update_review(good["date"], good["id"], "confirmed", root=temp)
            update_review(bad["date"], bad["id"], "false_positive", root=temp)
            counts = manage.training_export(Path(temp), Path(out))
            self.assertEqual((counts["confirmed"], counts["false_positive"]), (1, 1))
            good_labels = (Path(out) / "labels" / f"{good['date']}_{good['id']}.txt").read_text().split("\n")
            bad_labels = (Path(out) / "labels" / f"{bad['date']}_{bad['id']}.txt").read_text().split("\n")
            self.assertTrue(any(line.startswith("1 ") for line in good_labels), "confirmed no-helmet label kept")
            self.assertFalse(any(line.startswith("1 ") for line in bad_labels), "false-positive head removed")
            self.assertTrue(any(line.startswith("3 ") for line in bad_labels), "rider kept as context")
            for line in filter(None, good_labels):
                values = [float(v) for v in line.split()[1:]]
                self.assertTrue(all(0 <= v <= 1 for v in values))
            self.assertIn("no-helmet", (Path(out) / "data.yaml").read_text())

    def test_prepare_model_verifies_manifest_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target, manifest = Path(temp) / "src.pt", Path(temp) / "best.pt", Path(temp) / "m.json"
            source.write_bytes(b"weights")
            manifest.write_text(json.dumps({"sha256": hashlib.sha256(b"weights").hexdigest()}))
            manage.prepare_model(source, target, manifest)
            self.assertEqual(target.read_bytes(), b"weights")
            source.write_bytes(b"tampered")
            with self.assertRaises(SystemExit):
                manage.prepare_model(source, target, manifest)

    def test_cli_creates_user_and_imports_backup(self):
        with tempfile.TemporaryDirectory() as backup, tempfile.TemporaryDirectory() as live:
            event(backup)
            with patch("manage.getpass.getpass", return_value="cli password 1"):
                manage.main(["--storage", live, "create-user", "ops", "--role", "admin"])
            self.assertEqual(auth.authenticate(Path(live), "ops", "cli password 1")["role"], "admin")
            manage.main(["--storage", live, "import-storage", backup])
            out = Path(live) / "out.csv"
            manage.main(["--storage", live, "export", "--out", str(out)])
            self.assertEqual(len(out.read_text().strip().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
