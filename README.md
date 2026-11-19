# Helmet Detection with YOLOv8

A local Python webcam demo using a trained YOLOv8 model to detect helmets, no-helmet regions, number plates, riders, and bikes. It saves dated event snapshots, suggests nearby rider/vehicle/plate matches, and provides a local review dashboard.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python hi.py
```

Open **http://127.0.0.1:8003** for the live view and event review. The camera uses OpenCV YuNet to find and keep a face in view, including when the person is not on a bike; the helmet model still classifies helmet/no-helmet and rider objects. This detects face locations, not a person's identity. The view requests 1280×720 and uses 800-pixel inference. The checkpoint-bound `thresholds.json` selects full-frame scores of 0.4995 for helmet and 0.3714 for no-helmet from validation F1 curves after excluding known split overlaps. Zoomed crops retain a 0.50 score threshold; that crop threshold still needs independent calibration. Without a calibration file, full-frame scores fall back to 0.80. Saved pictures use the original camera/video frame as their source, then crop the person/head and surrounding event context. YuNet weights and their MIT license are in `models/face_detection_yunet/`.

The camera runs in your browser. Allow camera access for **http://127.0.0.1:8003**; after changing permissions, use **Start / retry camera**. On macOS, your browser also needs camera access in **System Settings → Privacy & Security → Camera**. Close other apps using the camera if it cannot start. Closing the page stops that page's camera stream. Press **Ctrl+C** in Terminal to close the app. To review saved events without the camera, run `python dashboard.py` and open the same address; this mode does not run detection, even when browser camera permission is allowed. Run only one of these commands on port 8003 at a time.

With `hi.py` running, choose **Analyze a saved video** on the dashboard to upload an MP4, MOV, AVI, MKV, or M4V file up to 250 MB. The webcam pauses while the same detector processes that clip, then resumes. The uploaded source video is deleted after processing; matching detection images and metadata are saved in the date folders and database. Each new video event records a clip ID, the original filename (without directories), a zero-based frame index, and its position in seconds. The event timestamp is the time of processing, not the original recording date. Older events without source metadata remain readable and are labelled accordingly. Only one video can be uploaded or processed at a time.

Event details and review status are indexed in the local SQLite database `storage/events.db`; images are filed by date and result under `storage/YYYY-MM-DD/helmet/event_...` or `storage/YYYY-MM-DD/no_helmet/event_...`. Each event contains a context crop, a person crop, an optional plate crop, timestamps, confidence scores, track IDs, and OCR text. Retention cleanup runs at startup and hourly while the server remains open, removing dated folders older than 30 days. Cleanup errors are logged and retried an hour later. The `storage/` folder is excluded from Git.

## When a picture is saved

- The browser must provide a camera frame, or an uploaded video must provide a readable frame. Camera permission alone does not start the Python detector.
- The preview chooses a face first (YuNet score threshold 0.60), then a rider if no faces are present, then a helmet/no-helmet box. Zoom works on one selected target at a time; full-frame capture confirmation runs independently for every qualifying tracked head or rider.
- A **helmet** or **no-helmet** prediction must meet its checkpoint-bound full-frame threshold, or **0.50 in the enlarged crop**. Finding a face alone does not save a picture. These are model scores, not measured probabilities of being correct.
- Full-frame predictions can confirm independently for all visible tracks. An enlarged-crop prediction must match the selected rider, fall near the selected face's head region, or match the selected helmet/no-helmet box. A bike, plate, or rider match is not required for a home camera test; captures without vehicle context are explicitly labelled as unverified.
- Three consecutive processed frames must agree on the class for each target independently. A missing or conflicting prediction resets that target's confirmation. Only the strongest qualifying prediction for each target counts in a frame. A 30-second cooldown follows known head/rider IDs even when they move or a rider association disappears. Distinct known tracks can capture at the same position. Untracked heads still use overlap. Zoom targets yield after eight processed frames and become eligible again after five seconds.
- `person.jpg` contains the rider crop if one is matched; otherwise it contains the detected head/helmet crop. `event.jpg` contains nearby context, and `plate.jpg` is saved only when a plate is matched. Both helmet and no-helmet events are saved.

Three processed frames are not a fixed duration or a guarantee of correctness. ID changes can still permit recaptures, and untracked overlapping people can still be confused. A source switch resets confirmation and cooldown. Review saved images before using them. Enlarged inference is skipped when the selected target already has a qualifying full-frame head prediction; its thresholds are unchanged. OpenCV uses at most two worker threads, measured locally to reduce face-detector overhead.

Detector errors trigger a retry after five seconds while the review dashboard stays available. **Restart detector** requests a new detector run without closing the HTTP listener; **Stop camera** releases browser camera tracks. Video upload shows byte progress, and analysis shows processed frame counts and errors. Uploaded sources are deleted after processing or interruption. Existing captures remain available. Editing Python files requires a deliberate app restart to load the new code.

## Plate matching and OCR

The app uses YOLO tracking and overlapping bounding boxes to suggest a rider, bike, and plate association. A head must be inside exactly one rider box; bike candidates must overlap that rider and pass the existing distance bound, and a plate centre must be inside the single matched bike. Ambiguous candidates stay unmatched. This deliberately favours an unmatched result over a guessed association. Review every suggestion in the dashboard; proximity does not prove that a plate belongs to a rider. Use **Review status**, **Plate text**, and **Save review** to record your decision or correct OCR. Status can be needs review, confirmed, or false positive. A false-positive label preserves the evidence; **Remove event** permanently deletes the record and pictures after confirmation. Polling preserves controls while you edit them. Returning an event to needs review resets its association label to suggested or unmatched. OCR command failures leave text blank.

Plate reading uses the local Tesseract OCR command. On macOS, install it with `brew install tesseract`. Without Tesseract, event capture and manual plate review still work, but plate text will be blank. Plate detection is not the same as reliable plate recognition.

## Model and training

The app looks for `best.pt` in the repository root, then `runs/detect/helmet_v3/weights/best.pt`; a trained model must be present at one of those paths. The merged training setup uses five classes: helmet, no-helmet, number-plate, rider, and bike. The live view displays all five. To train from scratch, `train.py` expects `merged_dataset/data.yaml` and the training dataset. `download_merge_retrain.py` downloads the Roboflow datasets and starts a 50-epoch run; training can take hours.

Roboflow download scripts read `ROBOFLOW_API_KEY` from the environment. Set your own key locally; never commit it.

## Limitations

This is a local prototype, not an enforcement system. It does not reliably prove rider-to-plate identity, issue fines, or alert authorities. Verify event images and plate text before using them. Captures may identify people and vehicles; keep the repository private if you ever choose to store footage in it (the provided storage folder is ignored by Git), and only use footage you are authorized to process.


## Reproducible setup and verification

The tested runtime is macOS arm64, Python 3.9.6, and Node.js 26.5.0 for JavaScript checks. `requirements.txt` pins runtime dependencies; `requirements-lock.txt` records the complete installed environment. Other platforms need their own verified lock. The current Python runtime is retained for compatibility; a supported Python upgrade remains a separate migration.

Copy the trusted existing `runs/detect/helmet_v3/weights/best.pt` to `best.pt` when preparing a fresh checkout. `model-manifest.json` records its SHA-256 and five required classes. No public checkpoint download is supplied. The default root/relative model paths work after moving the project; `HELMET_MODEL` can select a trusted local checkpoint and `HELMET_STORAGE` can select an explicit data directory. Calibration is refused if its checkpoint hash differs. Keep the bundled YuNet file and MIT license in `models/face_detection_yunet/`.

Dataset-download tools use `ROBOFLOW_API_KEY` from the environment. They require a separate Python 3.10+ environment with `requirements-training.txt`; Roboflow is not needed to run the dashboard. Training is deliberate and can take hours. `train.py` resolves its dataset from the current project folder and selects MPS when available, otherwise CPU.

```bash
.venv/bin/python -B -m unittest test_adaptive_zoom test_capture_state test_evaluation test_service test_event_store test_dashboard
node test_camera_page.js
.venv/bin/python evaluate_model.py --device cpu
.venv/bin/python check_usage.py
.venv/bin/python test_camera.py --seconds 5
```

Evaluation scans for identical bytes and shared source filenames, excludes known overlaps before tuning, selects full-frame thresholds on validation, and freezes them when reporting test results. `reports/evaluation.json` includes per-class AP, precision/recall curves at frozen thresholds, split counts, model hash, and limits. 496 validation and 280 test images remained in the current run; 446 later-split files were excluded (195 identical-byte and 251 filename overlaps). Test mAP50 is 0.7450 and mAP50–95 is 0.4439 across five classes. These are source-dataset results, not independent road-scene accuracy. Mixed labels, undetected augmented overlap, and temporal/crop behavior remain separate concerns. Summary precision/recall use the library's max-F1 point; the separately named interpolated values use the frozen validation thresholds. See [Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/).

`check_usage.py` replays a real dataset image through video upload, detection, storage, JPEG-frame input, and MJPEG preview using disposable data. Its low-light and distance cases are synthetic perturbations. `test_camera.py` is a bounded physical-camera diagnostic; it keeps frames in memory and writes only a result summary. Fresh independently annotated road footage is still required for an independent accuracy claim.

## macOS login service

```bash
# Find the existing Helmet listener's PID; this is not permission to stop another app.
lsof -nP -iTCP:8003 -sTCP:LISTEN
.venv/bin/python install_service.py --replace-pid HELMET_PID
```

The installer prepares a private runtime and copied Python environment at `~/Library/Application Support/Helmet`, checks imports, then replaces the existing Helmet process. Its LaunchAgent starts at login and restarts after process exit. This avoids launchd's previous access failures in the protected Desktop folder. Activation briefly interrupts the listener; detector retries afterward keep it open. Existing data is copied into the private runtime, the project `storage` path links to it, and `storage.before-login-service` retains the original data as a rollback copy. Both data locations are excluded from Git. Do not run a second detector alongside the service.

```bash
launchctl print gui/$(id -u)/com.helmet.dashboard
# Deliberately stop the service when you no longer want it running.
launchctl bootout gui/$(id -u)/com.helmet.dashboard
```

Runtime edits must be installed again to update the service's copied code. The service persists beyond closing this chat. Process-restart checks do not prove a complete physical reboot/login cycle. See [Apple's launchd documentation](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html).

## Current fixes and acceptance status

See `IMPLEMENTATION_REPORT_2026-10-20.md` for the section-3 fixes, fresh replay results, and remaining acceptance gaps. The older October-13 reports describe the previous capture algorithm. Independent road accuracy and reliable OCR are not established by passing regression tests. New modules remain deferred.
