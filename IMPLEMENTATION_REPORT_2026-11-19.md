# Product upgrade — 19 November 2026

Earlier reports (2026-10-13, 2026-10-20) describe the previous capture algorithm and server. This report covers the
upgrade from a local demo to a deployable product.

## Data safety
- Retention deleted all 63 saved events (they were older than 30 days). Confirmed events are now kept past retention,
  retention is configurable (`HELMET_RETENTION_DAYS`, `0` = keep), and CSV / ZIP / PDF export exists.
- The 59-event rollback copy in `storage.before-login-service` is untouched; `python manage.py import-storage` can
  merge it back.

## Server and dashboard
- FastAPI replaces the standard-library HTTP server with the same detector interface.
- Login with scrypt-hashed passwords, viewer / reviewer / admin roles, sessions, login rate limiting, first-run admin
  setup (local only) and an audit log.
- Strict Content-Security-Policy (no inline script), frame denial, Host and Origin checks.
- Review: date range, category, status, camera and plate search, bulk actions, keyboard shortcuts, lightbox,
  pagination, no re-render on unchanged polls, cacheable evidence images.
- Live: opt-in camera, streams only while visible, video progress and cancel, per-source FPS.
- Analytics chart, settings for CCTV cameras, alerts, users and audit; light/dark theme; phone layout.

## Detection
- Per-source pipelines (browser, uploaded video, each RTSP camera) with separate trackers.
- RTSP / HTTP / device cameras with reconnect backoff and redacted credentials.
- Calibrated zoom geometry (heads 20%, riders 45% of frame height).
- Privacy mode blurs faces and drops plates for helmet-compliant riders.
- Plate text normalised to Indian formats.
- Webhook and email alerts in the background, rate-limited, with failures audited.

## Model work
- Audit: only ds1 labels rider/bike (555 of 5,273 training images); ds3 also lacks plates. This explains rider recall
  0.42 and bike recall 0.46.
- `helmet_v4` (pseudo-labelled rider/bike/plate, fine-tuned 12 epochs from v3): rider recall 0.64, bike 0.67, but
  helmet 0.481→0.469, no-helmet 0.521→0.497, plate 0.500→0.474 mAP50-95. Not adopted; `best.pt` stays `ad241f16…`.
- A zoom-crop threshold of 0.02 proposed by validation precision was rejected (test no-helmet precision 0.58);
  zoom_crop stays 0.50.
- CoreML/ONNX were not faster than `.pt`; the pipeline, not the model, limits FPS.

## Engineering
- Python 3.9 → 3.12; dependency pins and lock regenerated.
- `hi.py` renamed `app.py`; the detector loop is split into `SourcePipeline`.
- `manage.py` admin CLI; `install_service.py` stages the venv, copies all runtime files and has `--check`.
- Docker image built and run (healthy, login, detector loaded); Compose and systemd files; GitHub Actions CI passing.
- Tests: 55 → 87 Python tests plus jsdom browser tests; ruff clean.
- Real-model replay passed on the new server (3.5–5.3 processed FPS while training ran in parallel).
- Old files moved to `_archive/`; nothing deleted.

## Still requires the owner
- Independent labelled road footage (night, rain, glare, blur) for real-world accuracy.
- A physical reboot/login test of the Mac service.
- Pricing, contact details and dataset-licence check before publishing the landing page and demo video.
- A longer, lower-learning-rate retrain if rider/bike recall matters more than the current small helmet trade-off.
