"""Install a private macOS login runtime outside the protected Desktop folder.

  python install_service.py --check                 report whether the running service matches this project
  python install_service.py --replace-pid PID       install/update and replace the running Helmet process
"""
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
import ssl
from urllib.request import urlopen

LABEL = "com.helmet.dashboard"
ENTRY = "app.py"
RUNTIME_FILES = ("app.py", "dashboard.py", "event_store.py", "auth.py", "alerts.py", "cameras.py", "reports_pdf.py",
                 "backup.py", "ptz.py", "manage.py", "model-manifest.json")


def service_config(runtime, lan_hosts=None, tls=None):
    config = {"Label": LABEL,
            "ProgramArguments": [str(runtime / ".venv/bin/python"), "-u", str(runtime / ENTRY)],
            "WorkingDirectory": str(runtime), "RunAtLoad": True, "KeepAlive": True,
            "ThrottleInterval": 5, "Umask": 0o077,
            "EnvironmentVariables": {"HELMET_STORAGE": str(runtime / "storage"),
                                     "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
            "StandardOutPath": str(runtime / "service.log"),
            "StandardErrorPath": str(runtime / "service-error.log")}
    if lan_hosts:
        # Reachable from phones on the same Wi-Fi; login stays required (the dashboard refuses otherwise).
        config["EnvironmentVariables"].update({"HELMET_HOST": "0.0.0.0", "HELMET_AUTH": "1",
                                               "HELMET_ALLOWED_HOSTS": ",".join(lan_hosts)})
    if tls:
        config["EnvironmentVariables"].update({"HELMET_TLS_CERT": str(tls[0]), "HELMET_TLS_KEY": str(tls[1])})
    return config


def lan_hosts():
    """This Mac's Wi-Fi/Ethernet addresses and Bonjour name, as host:port entries for the Host check."""
    hosts = []
    for interface in ("en0", "en1"):
        result = subprocess.run(["ipconfig", "getifaddr", interface], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            hosts.append(f"{result.stdout.strip()}:8003")
    name = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True).stdout.strip()
    if name:
        hosts.append(f"{name.lower()}.local:8003")
    return hosts


def prepare_runtime(project, runtime):
    model = next((path for path in (project / "best.pt", project / "runs/detect/helmet_v3/weights/best.pt")
                  if path.is_file()), None)
    if model is None or not (project / ".venv/bin/python").is_file():
        raise ValueError("Install the Python environment and trained model before installing the service")
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime.chmod(0o700)
    for name in RUNTIME_FILES:
        shutil.copy2(project / name, runtime / name)
    shutil.copytree(project / "static", runtime / "static", dirs_exist_ok=True)
    (runtime / "hi.py").unlink(missing_ok=True)  # detector entry was renamed to app.py
    if (project / "thresholds.json").is_file():
        shutil.copy2(project / "thresholds.json", runtime / "thresholds.json")
    shutil.copy2(model, runtime / "best.pt")
    shutil.copytree(project / "models", runtime / "models", dirs_exist_ok=True)
    # Copy the environment beside the old one, verify it, then swap: copying over an existing venv fails on symlinks.
    staging, previous = runtime / ".venv.staging", runtime / ".venv.previous"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(project / ".venv", staging, symlinks=True)
    subprocess.run([str(staging / "bin/python"), "-B", "-c", "import cv2, numpy, torch, ultralytics, fastapi, uvicorn, fpdf"],
                   check=True, cwd=runtime)
    shutil.rmtree(previous, ignore_errors=True)
    if (runtime / ".venv").exists():
        (runtime / ".venv").rename(previous)
    staging.rename(runtime / ".venv")
    subprocess.run([str(runtime / ".venv/bin/python"), "-B", "-c",
                    "import cv2, numpy, torch, ultralytics, fastapi, uvicorn, fpdf; "
                    "print('Service environment imports passed')"], check=True, cwd=runtime)


def local_get(path, timeout):
    """GET the local service over HTTPS (self-signed, so unverified on loopback only) or HTTP."""
    insecure = ssl.create_default_context()
    insecure.check_hostname = False
    insecure.verify_mode = ssl.CERT_NONE
    try:
        return urlopen(f"https://127.0.0.1:8003{path}", timeout=timeout, context=insecure)
    except OSError:
        return urlopen(f"http://127.0.0.1:8003{path}", timeout=timeout)


def ensure_certificate(runtime, hosts):
    """Self-signed certificate for this Mac's LAN names/IPs (10 years). Regenerated when the names change."""
    folder = runtime / "tls"
    folder.mkdir(mode=0o700, exist_ok=True)
    cert, key, names_file = folder / "cert.pem", folder / "key.pem", folder / "names.txt"
    names = sorted({host.rsplit(":", 1)[0] for host in hosts} | {"127.0.0.1", "localhost"})
    if cert.is_file() and key.is_file() and names_file.is_file() and names_file.read_text() == ",".join(names):
        return cert, key
    san = ",".join(("IP:" if name.replace(".", "").isdigit() else "DNS:") + name for name in names)
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-days", "3650", "-nodes",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=Helmet local dashboard",
                    "-addext", f"subjectAltName={san}"], check=True, capture_output=True)
    key.chmod(0o600)
    names_file.write_text(",".join(names))
    return cert, key


def running_version(timeout=3):
    with local_get("/healthz", timeout) as response:
        return json.load(response)["version"]


def check(project):
    """Compare the project's code version with the installed runtime and the running service."""
    sys.path.insert(0, str(project))
    from dashboard import code_version
    runtime = Path.home() / "Library/Application Support/Helmet"
    expected = code_version(project)
    installed = code_version(runtime) if (runtime / ENTRY).is_file() else None
    try:
        running = running_version()
    except (OSError, ValueError, KeyError):
        running = None
    print(f"project  {expected}")
    print(f"installed {installed or 'not installed'}")
    print(f"running  {running or 'not reachable (or an older version without /healthz)'}")
    if expected == installed == running:
        print("The login service is up to date.")
        return 0
    print("The service is out of date. Re-run: python install_service.py --replace-pid "
          "$(lsof -tnP -iTCP:8003 -sTCP:LISTEN)")
    return 1


def wait_ready(seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with local_get("/healthz", 2) as response:
                if json.load(response)["ok"]:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Helmet did not become ready on its fixed port 8003")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--replace-pid", type=int, help="Existing Helmet PID to replace after preparation")
    parser.add_argument("--check", action="store_true", help="Only report whether the service is up to date")
    parser.add_argument("--lan", action="store_true", help="Also serve phones/computers on the same Wi-Fi (login required)")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check(args.project.resolve()))
    if args.replace_pid is None:
        parser.error("--replace-pid is required unless --check is used")
    if sys.platform != "darwin" or args.replace_pid <= 1:
        raise SystemExit("This installer requires macOS and an explicit existing Helmet PID")
    project = args.project.resolve()
    runtime = Path.home() / "Library/Application Support/Helmet"
    command = subprocess.check_output(["ps", "-p", str(args.replace_pid), "-o", "command="], text=True)
    if not any(str(root / name) in command for root in (project, runtime) for name in ("hi.py", ENTRY)):
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
        hosts = lan_hosts() if args.lan else None
        tls = ensure_certificate(runtime, hosts) if args.lan else None
        agent.write_bytes(plistlib.dumps(service_config(runtime, hosts, tls)))
        agent.chmod(0o600)
        subprocess.run(["launchctl", "bootstrap", domain, str(agent)], check=True)
        wait_ready()
        scheme = "https" if args.lan else "http"
        print(f"Helmet login service ready: {scheme}://127.0.0.1:8003; data: {runtime / 'storage'}")
        for host in hosts or []:
            print(f"  on your Wi-Fi: https://{host}")
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
                subprocess.Popen([str(project / ".venv/bin/python"), "-u", str(project / ENTRY)],
                                 cwd=project, stdout=output, stderr=output, start_new_session=True)
        raise


if __name__ == "__main__":
    main()
