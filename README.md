# Helmet — rider helmet compliance with human review

Helmet watches two-wheeler riders on a browser webcam, uploaded video, or CCTV/IP cameras (RTSP). A YOLOv8 model flags
**helmet** and **no-helmet** riders, links them to a nearby rider, bike and number plate where it can do so without
ambiguity, and saves evidence for a person to review. Nothing is treated as a violation until a reviewer confirms it.

It runs on one machine (a Mac, a Linux server or an edge box). All footage, evidence and accounts stay on that machine.

## Contents

- [Quick start (Mac)](#quick-start-mac)
- [Using the dashboard](#using-the-dashboard)
- [CCTV cameras](#cctv-cameras)
- [Alerts](#alerts)
- [Users and roles](#users-and-roles)
- [Privacy, retention and audit](#privacy-retention-and-audit)
- [Deploying on a server or edge box](#deploying-on-a-server-or-edge-box)
- [Command line](#command-line)
- [Model, accuracy and limits](#model-accuracy-and-limits)
- [Development](#development)
- [Configuration reference](#configuration-reference)

## Quick start (Mac)

```bash
uv venv --python 3.12 .venv            # or: python3.12 -m venv .venv
VIRTUAL_ENV=.venv uv pip install -r requirements-lock.txt
.venv/bin/python manage.py prepare-model   # copies the trusted checkpoint to best.pt and checks its hash
brew install tesseract                     # optional, for number-plate text
.venv/bin/python hi.py
```

Open **http://127.0.0.1:8003**. On the first visit you create the admin account (only possible from the same computer).
Port 8003 is fixed for this project; if it is busy the app stops with an error instead of picking another port.

`python dashboard.py` runs the review dashboard without the detector (no camera or video analysis).

## Using the dashboard

**Review** — the default view shows *No helmet* events that *Need review*, newest first.

- Filter by date range (Today / 7 days / 30 days), category, review status, camera, or plate/event search.
- Each card shows the person crop, the matched plate (if any), scores, source, and the rider/bike/plate association.
  Click an image to enlarge it; **Context** opens the wider evidence image.
- Set the review status (needs review / confirmed / false positive), correct the plate text, and **Save**.
  Plate text is checked against Indian registration formats (`GJ01AB1234`, `22BH1234AB`) and common OCR mix-ups
  (`O`/`0`, `I`/`1`, `B`/`8` …) are corrected.
- Select several cards for **bulk** confirm / false positive / needs review / delete.
- **Export** the current filter as CSV, a ZIP of evidence (images + JSON + CSV), or a PDF report.
- Keyboard: `j`/`k` move, `c` confirm, `f` false positive, `n` needs review, `x` select, `Delete` remove,
  `o` context image, `/` search, `?` help.

**Live** — start the browser camera (it is never started automatically), restart the detector, upload a video
(MP4/MOV/AVI/MKV/M4V up to 250 MB, with progress and **Cancel**), see per-source FPS, and watch CCTV streams.
Streams are only connected while this tab is open.

**Analytics** — totals, false-positive rate and a daily helmet / no-helmet chart for any range and camera.

**Settings** (admins) — CCTV cameras, alerts, users, privacy/retention summary and the audit log.
Everyone can change their own password from their name in the top bar.

### When an event is saved

- A helmet/no-helmet prediction must pass its calibrated full-frame threshold (`thresholds.json`, bound to the model's
  SHA-256), or the zoom-crop threshold for enlarged inference.
- The same class must be predicted on **three consecutive processed frames** for that tracked person.
- A tracked person is not captured again for 30 seconds.
- Each source (browser, video, each CCTV camera) has its own tracker, so IDs never mix between cameras.
- Associations require exactly one containing rider, one overlapping bike and one plate inside that bike; anything
  ambiguous stays unmatched. Geometry suggests, it does not prove, which plate belongs to whom.

## CCTV cameras

Settings → **CCTV cameras** → *Add camera*: a name, an optional site, and an `rtsp://`, `http(s)://` (MJPEG) URL or a
local device number. Most NVRs/IP cameras expose a URL like `rtsp://user:password@192.168.1.20:554/Streaming/Channels/101`.
Use the camera's sub-stream (e.g. 1280×720) — inference runs at 800 px, so 4K streams only cost bandwidth.

- Readers keep only the newest frame and reconnect automatically (1 s → 30 s backoff).
- Credentials in URLs are stored on the server and never sent back to the browser.
- Detection throughput is shared: one Apple M4 processes roughly 4–7 frames/s in total (see
  [Model, accuracy and limits](#model-accuracy-and-limits)), so plan about 1–3 cameras per machine.
- While an uploaded video is being analysed, live sources are paused.

## Alerts

Settings → **Alerts**: send a message for new events (by default only *No helmet*), at most once per *minimum interval*.

- **Webhook** — JSON `POST` with `{"text": "...", "event": {...}}` and an optional `Authorization: Bearer` token. Works
  directly with Slack/Teams incoming webhooks and with WhatsApp/SMS gateways that accept a JSON webhook.
- **Email** — SMTP with STARTTLS (port 587) or SSL (465); the person crop is attached.
- **Send test alert** checks the configuration. Delivery runs in the background; failures are shown in
  the status and recorded in the audit log, and never interrupt detection.

## Users and roles

| Role | Can |
|---|---|
| viewer | see events, analytics and live streams |
| reviewer | + review, delete, export, run the camera, upload video |
| admin | + cameras, alerts, users, audit log |

Passwords (10+ characters) are hashed with scrypt. Sessions last 12 hours in an `HttpOnly`, `SameSite=Strict` cookie.
Five wrong passwords lock that address out for a minute. The last admin cannot be removed or demoted.
`HELMET_AUTH=0` turns login off for single-user local use; the server then refuses to listen beyond `127.0.0.1`.

## Privacy, retention and audit

See [PRIVACY.md](PRIVACY.md) for the full policy and the India DPDP Act 2023 checklist. In short:

- **Privacy mode** (default on): for riders *wearing* a helmet, detected faces are blurred and no plate image or plate
  text is stored.
- **Retention**: evidence older than `HELMET_RETENTION_DAYS` (default 30) is deleted hourly. **Confirmed events are kept**
  until someone deletes them (`HELMET_KEEP_CONFIRMED=1`). `HELMET_RETENTION_DAYS=0` keeps everything.
- **Audit log**: sign-ins (including failures), reviews, deletions, exports, uploads, settings, user changes and alerts.
- **Security**: strict Content-Security-Policy (no inline script), `X-Frame-Options: DENY`, Host and Origin checks against
  DNS rebinding and cross-site requests, and login on every API and image.

## Deploying on a server or edge box

### Mac login service (starts at login, restarts on crash)

```bash
.venv/bin/python install_service.py --check                       # is the running service current?
.venv/bin/python install_service.py --replace-pid $(lsof -tnP -iTCP:8003 -sTCP:LISTEN)
```

The installer copies the code, model and environment to `~/Library/Application Support/Helmet` (launchd cannot
read the Desktop), verifies imports in a staging copy, swaps it in, and replaces the running process. Data lives in
`~/Library/Application Support/Helmet/storage`; the project's `storage` link points there.

### Docker (Linux amd64/arm64)

```bash
.venv/bin/python manage.py prepare-model          # the image needs best.pt
echo "HELMET_ADMIN_PASSWORD=choose-a-long-password" > .env
docker compose up -d --build
```

Browse to `http://<server>:8003` and sign in as `admin`. If you reach the server by a name or LAN IP, list it in
`HELMET_ALLOWED_HOSTS` (for example `helmet.local,192.168.1.50`) — other Host headers are rejected. Put the service
behind HTTPS (Caddy, nginx, Tailscale) before exposing it beyond a trusted network; the browser camera also needs HTTPS
when not on localhost.

### systemd (Ubuntu, Raspberry Pi OS, NVIDIA Jetson)

Copy the project to `/opt/helmet`, create its `.venv`, then install [deploy/helmet.service](deploy/helmet.service).
On a Jetson, install the NVIDIA-provided PyTorch wheel for your JetPack instead of the CPU wheel.

## Command line

```bash
python manage.py create-user NAME --role reviewer     # prompts for a password
python manage.py reset-password NAME
python manage.py list-users
python manage.py import-storage storage.before-login-service   # merge an old backup; never overwrites
python manage.py export --start 2026-11-01 --end 2026-11-30 --out november.zip
python manage.py training-export --out reviewed_dataset        # reviewed events → YOLO dataset (see below)
python manage.py cleanup                                       # apply retention now
python manage.py prepare-model
```

## Model, accuracy and limits

- Five classes: helmet, no-helmet, number-plate, rider, bike. Checkpoint: `runs/detect/helmet_v3/weights/best.pt`
  (YOLOv8n, 640 training size) — `model-manifest.json` records its SHA-256. No public download is provided.
- On the de-duplicated test split (280 images after excluding 446 overlapping files): **mAP50 0.745,
  mAP50–95 0.444**. At the frozen validation thresholds helmet precision is 0.94 / recall 0.62 and no-helmet precision
  0.82 / recall 0.77. Rider (recall 0.42) and bike (recall 0.46) are the weak classes, which is why many events have no
  plate match. See `reports/evaluation.json` and `reports/ml-audit-2026-11-19.json`.
- These numbers come from the source datasets, **not** from independent road footage. Night, rain, glare, motion blur
  and camera angles you deploy with must be checked on your own labelled clips before relying on the system.
- Plate text uses Tesseract and is often wrong on small, angled or dirty plates. Treat it as a hint and correct it.
- This is a review tool. It does not identify people, prove plate ownership, issue fines or contact authorities.

### Improving the model with your reviews

`python manage.py training-export --out reviewed_dataset` turns *confirmed* events into labelled examples and
*false positives* into hard negatives (YOLO format with `data.yaml`). Add them to the **training** split only, retrain
with `train.py`, then re-run `evaluate_model.py` — thresholds are re-bound to the new checkpoint's hash and the app
refuses mismatched calibration.

### Training and evaluation

Dataset download (`download_merge_retrain.py`) needs `ROBOFLOW_API_KEY` and `requirements-training.txt`. `train.py`
trains on `merged_dataset/data.yaml` (MPS when available). `evaluate_model.py` audits split overlap, tunes thresholds on
validation only and reports frozen-threshold test results. Never commit API keys, footage or `storage/`.

## Development

```bash
VIRTUAL_ENV=.venv uv pip install -r requirements-dev.txt && npm install
.venv/bin/ruff check .
.venv/bin/python -B -m unittest test_adaptive_zoom test_capture_state test_evaluation test_service test_event_store \
    test_dashboard test_server_features test_manage test_alerts_cameras
node test_camera_page.js          # dashboard in jsdom: login, review, keyboard, bulk, camera, video, roles, chart
.venv/bin/python check_usage.py   # real-model replay through upload, detection, storage and preview
```

CI runs the same lint and tests on every push (`.github/workflows/ci.yml`).

| File | Purpose |
|---|---|
| `hi.py` | detector: per-source pipelines, capture confirmation, zoom, video/CCTV/browser loop |
| `dashboard.py` | FastAPI server: auth, review/export/report APIs, live preview, cameras, alerts, users, audit |
| `event_store.py` | SQLite index + evidence folders, retention, export, privacy redaction, plate normalisation |
| `auth.py`, `alerts.py`, `cameras.py`, `reports_pdf.py` | accounts, notifications, RTSP readers, PDF reports |
| `static/` | dashboard UI (no inline script; works under strict CSP) |
| `manage.py`, `install_service.py` | admin CLI and Mac login service installer |
| `evaluate_model.py`, `train.py`, `calibrate_zoom.py`, `benchmark_speed.py` | model evaluation, training, calibration, speed |

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `HELMET_STORAGE` | `./storage` | data folder (database + evidence) |
| `HELMET_MODEL` | `best.pt`, then `runs/detect/helmet_v3/weights/best.pt` | checkpoint to load |
| `HELMET_HOST` | `127.0.0.1` | listen address (`0.0.0.0` in Docker) |
| `HELMET_AUTH` | `1` | require login |
| `HELMET_ALLOWED_HOSTS` | — | extra `host[:port]` names browsers use |
| `HELMET_ADMIN_USER` / `HELMET_ADMIN_PASSWORD` | — | create the first admin if none exists |
| `HELMET_RETENTION_DAYS` | `30` | delete evidence older than this (`0` = never) |
| `HELMET_KEEP_CONFIRMED` | `1` | keep confirmed events past retention |
| `HELMET_PRIVACY` | `1` | blur faces / drop plates for helmet-compliant riders |
