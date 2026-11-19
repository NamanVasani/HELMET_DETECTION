from datetime import date, datetime
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

import numpy as np

import backup
from event_store import associate_objects, get_setting, list_events, save_event, update_review


def make_event(root):
    head = {"class_name": "no-helmet", "confidence": 0.9, "track_id": 1, "box": [10, 10, 40, 40]}
    return save_event(np.zeros((80, 100, 3), dtype=np.uint8), head, [head], associate_objects(head, [head]), root=root)


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.root, self.dest = base / "storage", base / "backups"
        self.event = make_event(self.root)
        update_review(self.event["date"], self.event["id"], "confirmed", "GJ01AB1234", self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_backup_and_restore_round_trip(self):
        path = backup.create_backup(self.root, self.dest)
        self.assertRegex(path.name, r"helmet-backup-\d{8}-\d{6}\.zip")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        restored = backup.restore_backup(path, Path(self.temp.name) / "restored")
        events = list_events(self.event["date"], restored)
        self.assertEqual([(e["id"], e["status"], e["plate_text"]) for e in events],
                         [(self.event["id"], "confirmed", "GJ01AB1234")])
        folder = f"{self.event['date']}/no_helmet/{self.event['id']}"
        for name in ("person.jpg", "event.jpg", "event.json"):
            self.assertEqual((restored / folder / name).read_bytes(), (self.root / folder / name).read_bytes())

    def test_encrypted_backup_needs_the_passphrase(self):
        if not shutil.which("openssl"):
            self.skipTest("openssl unavailable")
        path = backup.create_backup(self.root, self.dest, passphrase="long secret phrase")
        self.assertTrue(path.name.endswith(".zip.enc"))
        self.assertFalse(zipfile.is_zipfile(path), "encrypted file must not be a readable zip")
        with self.assertRaises(ValueError):
            backup.restore_backup(path, Path(self.temp.name) / "r1")
        with self.assertRaises(ValueError):
            backup.restore_backup(path, Path(self.temp.name) / "r2", passphrase="wrong phrase")
        restored = backup.restore_backup(path, Path(self.temp.name) / "r3", passphrase="long secret phrase")
        self.assertEqual(len(list_events(self.event["date"], restored)), 1)

    def test_keeps_only_newest_backups_and_never_restores_over_data(self):
        for day in range(1, 10):
            backup.create_backup(self.root, self.dest, keep=7, now=datetime(2026, 11, day, 2, 0))
        names = [item["name"] for item in backup.list_backups(self.dest)]
        self.assertEqual(len(names), 7)
        self.assertEqual(names[0], "helmet-backup-20261109-020000.zip")
        self.assertEqual(names[-1], "helmet-backup-20261103-020000.zip")
        with self.assertRaises(ValueError):
            backup.restore_backup(self.dest / names[0], self.root)
        with self.assertRaises(ValueError):
            backup.create_backup(self.root, self.root / "inside")

    def test_scheduler_runs_once_per_day_after_the_set_time(self):
        clock = Mock(return_value=datetime(2026, 11, 20, 1, 59))
        scheduler = backup.BackupScheduler(self.root, self.dest, at="02:00", keep=7, now=clock)
        self.assertEqual(scheduler.tick(), "not_yet")
        clock.return_value = datetime(2026, 11, 20, 2, 0)
        self.assertEqual(scheduler.tick(), "done")
        self.assertEqual(scheduler.tick(), "already_done")
        clock.return_value = datetime(2026, 11, 21, 9, 15)
        self.assertEqual(scheduler.tick(), "done", "catches up after a missed night")
        last = get_setting(self.root, "backup_last")
        self.assertEqual(last["date"], "2026-11-21")
        self.assertEqual(len(backup.list_backups(self.dest)), 2)
        self.assertEqual(date.fromisoformat(last["date"]), date(2026, 11, 21))


if __name__ == "__main__":
    unittest.main()
