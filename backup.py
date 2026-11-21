"""Nightly backups of the event database and evidence, with optional AES-256 encryption (openssl)."""

from datetime import datetime
import logging
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
import zipfile

from event_store import DATE_RE, STORAGE, get_setting, record_audit, set_setting

BACKUP_DIR = Path(os.environ.get("HELMET_BACKUP_DIR", STORAGE.parent / "backups"))
BACKUP_TIME = os.environ.get("HELMET_BACKUP_TIME", "02:00")
BACKUP_KEEP = int(os.environ.get("HELMET_BACKUP_KEEP", "7"))
BACKUP_ENABLED = os.environ.get("HELMET_BACKUP", "1") != "0"
NAME_RE = re.compile(r"helmet-backup-\d{8}-\d{6}\.zip(\.enc)?\Z")


def _openssl(args, passphrase):
    environment = dict(os.environ, HELMET_BACKUP_PASSPHRASE=passphrase)
    subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-salt", *args,
                    "-pass", "env:HELMET_BACKUP_PASSPHRASE"], check=True, capture_output=True, env=environment)


def create_backup(root=STORAGE, dest=BACKUP_DIR, passphrase=None, keep=BACKUP_KEEP, now=None):
    """Write helmet-backup-YYYYMMDD-HHMMSS.zip[.enc] with events.db and all dated evidence; prune old ones."""
    root, dest = Path(root), Path(dest)
    if dest.resolve() == root.resolve() or root.resolve() in dest.resolve().parents:
        raise ValueError("Keep backups outside the data folder")
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    target = dest / f"helmet-backup-{stamp}.zip"
    with tempfile.TemporaryDirectory(prefix="helmet_backup_") as temp:
        snapshot = Path(temp) / "events.db"
        if (root / "events.db").is_file():
            source = sqlite3.connect(root / "events.db")
            copy = sqlite3.connect(snapshot)
            try:
                source.backup(copy)  # consistent copy even while the app writes (WAL)
            finally:
                copy.close()
                source.close()
        partial = target.with_suffix(".zip.partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if snapshot.is_file():
                archive.write(snapshot, "events.db")
            for folder in sorted(root.iterdir()):
                if folder.is_symlink() or not folder.is_dir() or not DATE_RE.fullmatch(folder.name):
                    continue
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        archive.write(path, path.relative_to(root).as_posix())
        if passphrase:
            encrypted = target.with_suffix(".zip.enc")
            _openssl(["-in", str(partial), "-out", str(encrypted)], passphrase)
            partial.unlink()
            target = encrypted
        else:
            partial.rename(target)
    target.chmod(0o600)
    backups = sorted((path for path in dest.iterdir() if NAME_RE.fullmatch(path.name)), key=lambda path: path.name)
    for old in backups[:-keep] if keep > 0 else []:
        old.unlink()
    return target


def restore_backup(path, target, passphrase=None):
    """Unpack a backup into a new, empty folder. Never overwrites live data."""
    path, target = Path(path), Path(target)
    if target.exists() and any(target.iterdir()):
        raise ValueError("Restore into a new or empty folder; it never overwrites existing data")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="helmet_restore_") as temp:
        archive_path = path
        if path.name.endswith(".enc"):
            if not passphrase:
                raise ValueError("This backup is encrypted; give the backup passphrase")
            archive_path = Path(temp) / "backup.zip"
            try:
                _openssl(["-d", "-in", str(path), "-out", str(archive_path)], passphrase)
            except subprocess.CalledProcessError as exc:
                raise ValueError("Wrong passphrase or damaged backup") from exc
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                resolved = (target / member).resolve()
                if target.resolve() not in resolved.parents:
                    raise ValueError(f"Unsafe path in backup: {member}")
            archive.extractall(target)
    return target


def list_backups(dest=BACKUP_DIR):
    dest = Path(dest)
    if not dest.is_dir():
        return []
    return [{"name": path.name, "bytes": path.stat().st_size, "encrypted": path.name.endswith(".enc")}
            for path in sorted(dest.iterdir(), reverse=True) if NAME_RE.fullmatch(path.name)]


class BackupScheduler:
    """Back up once a day at BACKUP_TIME (or at the next start after a missed time)."""

    def __init__(self, root=STORAGE, dest=BACKUP_DIR, at=BACKUP_TIME, keep=BACKUP_KEEP, now=None):
        self.root, self.dest, self.at, self.keep = Path(root), Path(dest), at, keep
        self.now = now or datetime.now

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="helmet-backup").start()
        return self

    def _run(self):
        while True:
            try:
                self.tick()
            except Exception:  # disk problems must not stop detection; retried next minute
                logging.exception("Backup failed")
            time.sleep(60)

    def tick(self):
        now = self.now()
        hour, minute = (int(part) for part in self.at.split(":"))
        if (now.hour, now.minute) < (hour, minute):
            return "not_yet"
        today = now.date().isoformat()
        if (get_setting(self.root, "backup_last") or {}).get("date") == today:
            return "already_done"
        path = create_backup(self.root, self.dest, os.environ.get("HELMET_BACKUP_PASSPHRASE") or None, self.keep, now)
        info = {"date": today, "file": path.name, "bytes": path.stat().st_size, "encrypted": path.name.endswith(".enc"),
                "at": now.isoformat(timespec="seconds"), "folder": str(self.dest)}
        set_setting(self.root, "backup_last", info)
        record_audit(self.root, "system", "backup_created", path.name, {"bytes": info["bytes"]})
        return "done"
