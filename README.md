# Helmet — rider helmet compliance with human review

**[Live product page](https://namanvasani.github.io/HELMET_DETECTION/)** · **[60-second demo video](https://namanvasani.github.io/HELMET_DETECTION/#demo)** · [CI](.github/workflows/ci.yml)

![Helmet review dashboard](docs/images/review.png)

| Quick review | Analytics | Dark mode |
|---|---|---|
| ![Quick review](docs/images/quick-review.png) | ![Analytics](docs/images/analytics.png) | ![Dark mode](docs/images/review-dark.png) |

**What it is:** an end-to-end computer-vision product — a custom-trained YOLOv8 model, a real-time detection pipeline
for webcams, phones and CCTV, and a secure web dashboard where people review every case before it counts.

**Highlights**

- Custom 5-class YOLOv8n model (helmet, no-helmet, number plate, rider, bike) trained on 5,273 merged images;
  mAP50 0.745 on a de-duplicated test split, with thresholds frozen on validation data only.
- Real-time pipeline: ByteTrack tracking, auto-zoom on distant riders (no-helmet recall 0.28 → 0.70 on small
  crops), 3-frame confirmation, "too close" guard, Apple-GPU inference with automatic CPU fallback.
- Number plates of every shape (long one-line plates 0/33 → 23/33 found) and two-line plate OCR with Indian
  state-code validation.
- Works with browser webcams, **phones** (front/back camera, on-device preview with live boxes), uploaded video and
  RTSP CCTV, including **ONVIF PTZ** cameras that turn and zoom toward riders.
- Product features: roles and audit log, quick review with keyboard shortcuts, analytics, CSV/ZIP/PDF exports,
  WhatsApp/email/webhook alerts, nightly encrypted backups, privacy mode (face blur), DPDP-oriented retention.
- Engineering: FastAPI + SQLite, strict CSP, scrypt passwords, HTTPS on the LAN, launchd/Docker/systemd deploys,
  128 Python tests + jsdom browser tests in GitHub Actions, and a guarded retraining loop that never swaps the model
  without proof it is better.

**Tech:** Python · PyTorch / Ultralytics YOLOv8 · OpenCV · FastAPI · SQLite · vanilla JS · Tesseract OCR · ONVIF ·
GitHub Actions · Docker

Helmet watches two-wheeler riders on a browser webcam, uploaded video, or CCTV/IP cameras (RTSP). A YOLOv8 model flags
**helmet** and **no-helmet** riders, links them to a nearby rider, bike and number plate where it can do so without
ambiguity, and saves evidence for a person to review. Nothing is treated as a violation until a reviewer confirms it.

It runs on one machine (a Mac, a Linux server or an edge box). All footage, evidence and accounts stay on that machine.

## Contents

- [Quick start (Mac)](#quick-start-mac)
- [Using the dashboard](#using-the-dashboard)
- [CCTV cameras](#cctv-cameras) (incl. PTZ)
- [Using it away from home](#using-it-away-from-home)
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
brew install tesseract                     # optional, for number-plate text
.venv/bin/python app.py
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

### PTZ cameras (optional)

For an ONVIF camera that can pan, tilt and zoom, add a `ptz` login to its camera entry (via `PUT /api/cameras`):

```json
{"name": "Gate", "url": "rtsp://…", "ptz": {"host": "192.168.1.50", "port": 80, "username": "admin", "password": "…"}}
```

Helmet then turns and zooms toward the distant rider it is focusing on — at most one small move per second — and
returns to the home position after 20 seconds without a target. Moves run in the background, so an offline or slow
camera never stalls detection; the PTZ password is never sent to the browser. Standard library only. It has been
tested against a simulated ONVIF camera (`test_ptz.py`), so check it on your camera model before relying on it.

## Using it away from home

`--lan` (below) covers phones on the same Wi-Fi. To open the dashboard from anywhere (mobile data, another network)
without router changes or an account, use a free Cloudflare quick tunnel:

```bash
brew install cloudflared
.venv/bin/python install_service.py --lan --tunnel --replace-pid $(lsof -tnP -iTCP:8003 -sTCP:LISTEN)
.venv/bin/python remote_access.py      # prints https://<random-words>.trycloudflare.com — Ctrl+C closes it
```

The link has a real certificate (phone cameras work without warnings), changes every run, and stops working when you
close it. **Change the admin password first**: the login page is reachable from the internet while the tunnel is open.
All tunnel visitors share one login rate limit (5 wrong passwords lock sign-in for a minute).

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

Add `--lan` to also open the dashboard to phones and laptops on the same Wi-Fi. It then serves **HTTPS** with a
self-signed certificate for this Mac's IP and `.local` name (stored in the runtime's `tls/` folder), so open
`https://<mac-ip>:8003` and accept the one-time browser warning. Login stays required, and the phone camera works
because the page is secure.

The installer copies the code, model and environment to `~/Library/Application Support/Helmet` (launchd cannot
read the Desktop), verifies imports in a staging copy, swaps it in, and replaces the running process. Data lives in
`~/Library/Application Support/Helmet/storage`; the project's `storage` link points there.

### Docker (Linux amd64/arm64)

```bash
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
python manage.py backup                                        # back up database + evidence now
python manage.py list-backups
python manage.py restore backups/helmet-backup-….zip --to restored-storage   # never overwrites live data
```

### Backups

Every night at 02:00 (or at the next start after a missed night) the service writes
`helmet-backup-YYYYMMDD-HHMMSS.zip` — a consistent copy of `events.db` plus all evidence — to a `backups` folder
beside the data (`~/Library/Application Support/Helmet/backups` for the Mac service) and keeps the newest 7.
Set `HELMET_BACKUP_PASSPHRASE` to encrypt them with AES-256 (`.zip.enc`); keep that passphrase safe, as an encrypted
backup cannot be restored without it. Copy backups to another disk regularly: a backup on the same disk does not
survive a disk failure. Settings → Privacy shows the last backup.

## Model, accuracy and limits

- Five classes: helmet, no-helmet, number-plate, rider, bike. The checkpoint `best.pt` (YOLOv8n, 640 training size) is
  in the repository; `model-manifest.json` records its SHA-256 and `python manage.py prepare-model --source PATH`
  verifies a replacement before installing it.
- On the de-duplicated test split (280 images after excluding 446 overlapping files): **mAP50 0.745,
  mAP50–95 0.444**. At the frozen validation thresholds helmet precision is 0.94 / recall 0.62 and no-helmet precision
  0.82 / recall 0.77. Rider (recall 0.42) and bike (recall 0.46) are the weak classes, which is why many events have no
  plate match. See `reports/evaluation.json` and `reports/ml-audit-2026-11-19.json`.
- Why rider/bike are weak: of the three merged datasets only one (555 of 5,273 training images) labels riders and bikes,
  so the model learned unlabelled riders as background. `pseudo_label.py` fills those labels; a 12-epoch fine-tune on
  that data (`runs/detect/helmet_v4`) lifted rider recall to 0.64 and bike recall to 0.67 but lowered helmet,
  no-helmet and plate mAP50-95 slightly, so it was **not adopted**. A longer run at a lower learning rate is the next
  step (`train.py --data runs/datasets/merged_v4/data.yaml --model runs/detect/helmet_v3/weights/best.pt`).
- Zoom: enlarged inference sizes heads to 20% and riders to 45% of the frame height (was 65% for every target). On
  upscaled test crops at the unchanged 0.50 threshold, no-helmet recall rises from 0.28 to 0.70 and precision from
  0.63 to 0.78 (`reports/zoom-calibration-2026-11-19.json`). Measured on dataset crops, not live footage.
- Speed: the model itself runs at 22–31 tracked frames/s on an M4 (`reports/speed-2026-11-19.json`); the app's 4–7
  frames/s is set by face detection, the second zoom pass, JPEG work and the browser's frame rate. CoreML and ONNX
  exports were not faster, so the `.pt` runtime stays. The full app pipeline runs 7.1 frames/s on the Apple GPU vs
  6.3 on the CPU at 1080p (3 runs each, no hangs), so the GPU is used automatically when present.
- These numbers come from the source datasets, **not** from independent road footage. Night, rain, glare, motion blur
  and camera angles you deploy with must be checked on your own labelled clips before relying on the system.
- Plates and bikes are kept from 0.35 confidence (helmet/no-helmet still need 0.80). At the old 0.80 no long one-line
  plate was found (0/33 on the test split); at 0.35 23/33 long and 67/96 square plates are found, 80% precise. Most
  remaining misses are tiny plates (2–3% of the image height) on vehicles with no detected rider; a second, zoomed
  look under riders found none of them, so finding those needs retraining with more small-plate images.
- Plate text uses Tesseract. Square two-line plates are split into rows and read line by line (hand-checked two-line
  plates: 0% → 53% characters right), and a reading is only accepted as valid with a real Indian state/UT code. It is
  still often wrong on small, angled or dirty plates — treat it as a hint and correct it.
- This is a review tool. It does not identify people, prove plate ownership, issue fines or contact authorities.

### Improving the model with your reviews

`python manage.py training-export --out reviewed_dataset` turns *confirmed* events into labelled examples and
*false positives* into hard negatives (YOLO format with `data.yaml`).

`python retrain.py` does the whole loop in one command and **never replaces `best.pt` by itself**:

```bash
.venv/bin/python retrain.py --dry-run     # show the plan
.venv/bin/python retrain.py --hours 2     # export reviews, fine-tune (lr0 0.001), evaluate, compare
```

It refuses to train with fewer than 100 reviewed events (`--force` overrides), adds reviewed events to the **training**
split only, fine-tunes from the current model with a low learning rate, evaluates both models on the same
human-labelled test split, and recommends the new model only if **no class gets worse** and the average improves.
Results are in `runs/retrain/<timestamp>/comparison.json`; adopting is a deliberate, manual step it prints for you.

### Training and evaluation

Dataset download (`download_merge_retrain.py`) needs `ROBOFLOW_API_KEY` and `requirements-training.txt`. `train.py`
trains on `merged_dataset/data.yaml` (MPS when available). `evaluate_model.py` audits split overlap, tunes thresholds on
validation only and reports frozen-threshold test results. Never commit API keys, footage or `storage/`.

## Development

```bash
VIRTUAL_ENV=.venv uv pip install -r requirements-dev.txt && npm install
.venv/bin/ruff check .
.venv/bin/python -B -m unittest test_adaptive_zoom test_capture_state test_evaluation test_service test_event_store \
    test_dashboard test_server_features test_manage test_alerts_cameras test_backup test_retrain test_ptz
node test_camera_page.js          # dashboard in jsdom: login, review, keyboard, bulk, camera, video, roles, chart
.venv/bin/python check_usage.py   # real-model replay through upload, detection, storage and preview
```

CI runs the same lint and tests on every push (`.github/workflows/ci.yml`); `.github/workflows/pages.yml` publishes
`site/` (the landing page) to GitHub Pages.

| File | Purpose |
|---|---|
| `app.py` | detector: per-source pipelines, capture confirmation, zoom, video/CCTV/browser loop |
| `dashboard.py` | FastAPI server: auth, review/export/report APIs, live preview, cameras, alerts, users, audit |
| `event_store.py` | SQLite index + evidence folders, retention, export, privacy redaction, plate normalisation |
| `auth.py`, `alerts.py`, `cameras.py`, `reports_pdf.py` | accounts, notifications, RTSP readers, PDF reports |
| `static/` | dashboard UI (no inline script; works under strict CSP) |
| `manage.py`, `install_service.py` | admin CLI and Mac login service installer |
| `ptz.py`, `remote_access.py` | ONVIF pan/tilt/zoom follow; Cloudflare quick tunnel for access away from home |
| `retrain.py` | guarded retraining from reviewed events (never replaces the model by itself) |
| `site/` | product landing page (published to GitHub Pages) |
| `evaluate_model.py`, `train.py`, `ml_audit.py`, `pseudo_label.py`, `calibrate_zoom.py`, `export_model.py`, `benchmark_speed.py` | evaluation, training, label audit/fill, zoom calibration, export, speed |

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `HELMET_STORAGE` | `./storage` | data folder (database + evidence) |
| `HELMET_MODEL` | `best.pt`, then `runs/detect/helmet_v3/weights/best.pt` | checkpoint to load |
| `HELMET_HOST` | `127.0.0.1` | listen address (`0.0.0.0` in Docker) |
| `HELMET_AUTH` | `1` | require login |
| `HELMET_ALLOWED_HOSTS` | — | extra `host[:port]` names browsers use; `*.domain` entries allow HTTPS tunnels |
| `HELMET_ADMIN_USER` / `HELMET_ADMIN_PASSWORD` | — | create the first admin if none exists |
| `HELMET_RETENTION_DAYS` | `30` | delete evidence older than this (`0` = never) |
| `HELMET_KEEP_CONFIRMED` | `1` | keep confirmed events past retention |
| `HELMET_PRIVACY` | `1` | blur faces / drop plates for helmet-compliant riders |
| `HELMET_FACE_EVERY` | `2` | run the face locator every Nth frame (busy scene 3.8 → 4.8 fps, same saved events) |
| `HELMET_MAX_HEAD_FRACTION` | `0.45` | heads taller than this share of the frame are "too close to judge" (close-up webcam hair was read as a helmet); `0` disables |
| `HELMET_DEVICE` | `auto` | `auto` = Apple GPU when present (7.1 vs 6.3 fps), else CPU; `cpu` turns it off; a GPU error falls back to the CPU |
| `HELMET_PLATE_CONFIDENCE` | `0.35` | minimum confidence for number-plate boxes |
| `HELMET_BACKUP` | `1` | nightly backups on/off |
| `HELMET_BACKUP_DIR` / `HELMET_BACKUP_TIME` / `HELMET_BACKUP_KEEP` | `../backups`, `02:00`, `7` | where, when and how many |
| `HELMET_BACKUP_PASSPHRASE` | — | encrypt backups (AES-256) |
