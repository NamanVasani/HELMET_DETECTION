"""Install a private macOS login runtime outside the protected Desktop folder."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

LABEL = "com.helmet.dashboard"


def service_config(runtime):
    return {"Label": LABEL,
            "ProgramArguments": [str(runtime / ".venv/bin/python"), "-u", str(runtime / "hi.py")],
            "WorkingDirectory": str(runtime), "RunAtLoad": True, "KeepAlive": True,
            "ThrottleInterval": 5, "Umask": 0o077,
            "EnvironmentVariables": {"HELMET_STORAGE": str(runtime / "storage"),
                                     "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
            "StandardOutPath": str(runtime / "service.log"),
            "StandardErrorPath": str(runtime / "service-error.log")}


def prepare_runtime(project, runtime):
    model = next((path for path in (project / "best.pt", project / "runs/detect/helmet_v3/weights/best.pt")
                  if path.is_file()), None)
    if model is None or not (project / ".venv/bin/python").is_file():
        raise ValueError("Install the Python environment and trained model before installing the service")
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime.chmod(0o700)
    for name in ("hi.py", "dashboard.py", "event_store.py", "model-manifest.json"):
        shutil.copy2(project / name, runtime / name)
    if (project / "thresholds.json").is_file():
        shutil.copy2(project / "thresholds.json", runtime / "thresholds.json")
    shutil.copy2(model, runtime / "best.pt")
    shutil.copytree(project / "models", runtime / "models", dirs_exist_ok=True)
    shutil.copytree(project / ".venv", runtime / ".venv", symlinks=True, dirs_exist_ok=True)
    subprocess.run([str(runtime / ".venv/bin/python"), "-B", "-c",
                    "import cv2, numpy, torch, ultralytics; print('Service environment imports passed')"], check=True)


def wait_ready(seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urlopen("http://127.0.0.1:8003/api/status", timeout=2) as response:
                if json.load(response)["camera_enabled"]:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Helmet did not become ready on its fixed port 8003")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--replace-pid", type=int, required=True, help="Existing Helmet PID to replace after preparation")
    args = parser.parse_args()
    if sys.platform != "darwin" or args.replace_pid <= 1:
        raise SystemExit("This installer requires macOS and an explicit existing Helmet PID")
    project = args.project.resolve()
    runtime = Path.home() / "Library/Application Support/Helmet"
    command = subprocess.check_output(["ps", "-p", str(args.replace_pid), "-o", "command="], text=True)
    if not any(str(root / "hi.py") in command for root in (project, runtime)):
        raise SystemExit("Refusing to stop a process that is not this Helmet app")
    prepare_runtime(project, runtime)
    agent = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    agent.parent.mkdir(parents=True, exist_ok=True)
    previous_plist = agent.read_bytes() if agent.exists() else None
    domain = f"gui/{os.getuid()}"
    job = f"{domain}/{LABEL}"
    storage = project / "storage"
    backup = project / "storage.before-login-service"
    linked = False
    stopped = False
    try:
        if subprocess.run(["launchctl", "print", job], capture_output=True).returncode == 0:
            subprocess.run(["launchctl", "bootout", job], check=True)
        try:
            os.kill(args.replace_pid, signal.SIGINT)
            for _ in range(40):
                if subprocess.run(["ps", "-p", str(args.replace_pid)], capture_output=True).returncode != 0:
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError("The existing process did not stop; no data location was changed")
        except ProcessLookupError:
            pass
        stopped = True
        if storage.is_symlink():
            if storage.resolve() != runtime / "storage":
                raise ValueError("Existing storage link points elsewhere; resolve it before installing")
        else:
            if backup.exists() or (runtime / "storage").exists():
                raise ValueError("A previous storage copy exists; refusing to overwrite it")
            if storage.exists():
                shutil.copytree(storage, runtime / "storage")
                storage.rename(backup)
                backup.chmod(0o700)
            else:
                (runtime / "storage").mkdir(mode=0o700)
            storage.symlink_to(runtime / "storage", target_is_directory=True)
            linked = True
        (runtime / "storage").chmod(0o700)
        agent.write_bytes(plistlib.dumps(service_config(runtime)))
        agent.chmod(0o600)
        subprocess.run(["launchctl", "bootstrap", domain, str(agent)], check=True)
        wait_ready()
        print(f"Helmet login service ready: http://127.0.0.1:8003; data: {runtime / 'storage'}")
        print("The original data folder remains in storage.before-login-service as a rollback copy.")
    except BaseException:
        subprocess.run(["launchctl", "bootout", job], capture_output=True)
        if linked:
            storage.unlink()
            if backup.exists():
                backup.rename(storage)
        if previous_plist is not None:
            agent.write_bytes(previous_plist)
        else:
            agent.unlink(missing_ok=True)
        if stopped:
            log = (project / "storage/service-rollback.log")
            log.parent.mkdir(exist_ok=True)
            with log.open("ab") as output:
                subprocess.Popen([str(project / ".venv/bin/python"), "-u", str(project / "hi.py")],
                                 cwd=project, stdout=output, stderr=output, start_new_session=True)
        raise


if __name__ == "__main__":
    main()
