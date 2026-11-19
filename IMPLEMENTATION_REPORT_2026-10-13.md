# Helmet implementation and verification — 13 October 2026

The existing localhost app was upgraded in place. Port 8003 remains the fixed project port. No new model training, external publication, or authority messaging was performed.

## Implemented

1. **Capture reliability:** three consecutive class agreements, reset on missing/conflicting evidence, one qualifying candidate per frame, spatial 30-second cooldown, eight-frame focus turn and five-second defer. Changing target resets confirmation. This reduces repeated captures and target starvation; it does not establish persistent face identity.
2. **Model evaluation:** class/schema checking, a checkpoint SHA-256 manifest, known-overlap exclusion, validation-only threshold tuning, and frozen-threshold test reporting. 5,273 training, 760 validation, and 462 test files were scanned. Excluding 195 identical-byte and 251 shared-source-filename cases left 496 validation and 280 test images. The native test AP results are mAP50 0.7450 and mAP50–95 0.4439. Class precision/recall at frozen thresholds are separately labeled interpolation estimates. Source-scene/augmentation independence, independent road footage, crop calibration, and event-level accuracy remain unverified.
3. **Recovery and feedback:** detector failures retry after five seconds without closing the HTTP listener; a restart endpoint and button request a new detector run. Video byte-upload progress, processed-frame totals, unreadable-video errors, interruption cleanup, and browser disconnect handling are implemented.
4. **Review controls:** the dashboard can save needs-review/confirmed/false-positive decisions and correct plate text through the existing API. Edits are preserved during focused input and on save failure. Image alternative text and labeled controls were added. False-positive status retains evidence; deletion still requires confirmation.
5. **Reproducible setup:** direct runtime versions and the installed macOS Python 3.9.6 environment are pinned, training-download requirements are separate, model provisioning/hash/class expectations are documented, training paths follow the project folder, and the existing interview guide was corrected. Python 3.9 is retained rather than claiming a newly tested supported-runtime migration. Other platforms and the separate Roboflow training installation were not verified.
6. **Persistent startup:** a standard-library installer prepares a private runtime/environment under Application Support, registers a login LaunchAgent with KeepAlive, and preserves original saved data in a private rollback folder. Activation/restart and data-preservation evidence are recorded below after deployment.
7. **Usage checks:** the real checkpoint processed a labeled multiple-person source image through actual video upload, detection, SQLite/image saving, JPEG-frame input, and MJPEG preview in disposable storage. Low-light and distance cases are synthetic image perturbations. Physical camera/browser checks are recorded separately; they cannot substitute for new annotated road footage.

## Checks completed before activation

- 46 Python regression tests passed (45 before activation; one additional empty-video regression passed afterward), including agreement/reset, target fairness, ID churn, overlap exclusion, threshold binding, review endpoints, server survival, retry, and login config.
- Browser JavaScript checks passed: review-only mode, denied-permission retry, duplicate start, stop, review/save failure, upload progress/retry, and detector failure.
- Python syntax and installed dependency consistency checks passed.
- Real-model replay: multiple people, 24 frames, 3.44 processed FPS, 5 saved events; synthetic low light, 24 frames, 3.70 FPS, 5 events; synthetic distance, 24 frames, 3.87 FPS, 7 events. Every saved replay event records three-frame confirmation.
- Replayed browser JPEG submission and MJPEG preview passed. These are pipeline timings on this Apple M4, not model-only FPS or fresh webcam FPS.
- The original database passed integrity checking with 59 events; 177 saved evidence files were hashed before service activation.

## Acceptance that needs external evidence

Independent, newly annotated road footage is still needed for accuracy/threshold acceptance in real traffic. The available sources overlap and have mixed category coverage. Complete augmentation/scene duplicate auditing and human annotation review were not performed. An actual machine reboot/login, additional physical devices/browsers, and an unsupported-Python migration were not performed. No claims of enforcement readiness, identity matching, guaranteed OCR correctness, or measured independent accuracy are made.

## Reproduce and inspect

See README.md for commands. `reports/evaluation.json` contains the current dataset audit, per-class results and threshold reporting; `reports/usage.json` contains replay measurements. Both are real local outputs, with their limitations recorded.

Primary references: [Ultralytics validation](https://docs.ultralytics.com/modes/val/), [Apple launchd jobs](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html), [Roboflow package requirements](https://pypi.org/project/roboflow/).

## Activated service and physical browser evidence

- The launchd login service is loaded and the detector is ready on 127.0.0.1:8003.
- A deliberate process termination changed the service PID from 30164 to 32262; the detector became ready again after 1.061 seconds. This confirms current restart-on-exit behavior, not a full reboot/login cycle.
- All 59 original events remain, the database integrity check passes, and all 177 evidence files match their before-activation SHA-256 in both the active and rollback locations. Both storage paths are excluded from Git, including the project's storage symlink.
- Safari camera permission was granted for the local app; the page showed live detection, and the returned annotated JPEG preview decoded to 1280×720. Stop camera visibly stopped it afterward. No new event was saved by that brief camera check.
- The real Safari dashboard visibly showed review status, plate text, Save review, and Remove event controls. Existing user records were not changed to fake test values.
- JavaScript checks include recovery after disconnect, preservation of permission errors, failed-save feedback, and upload retry/progress.
- The empty-video fix reports an error rather than successful completion when a readable video container supplies zero decodable frames, and removes the temporary source.

## Completion boundary

Capture, recovery, review controls, reproducible local setup, login-service activation, real camera access, and model replay are implemented and checked. Model evaluation tooling and filtered-source metrics are implemented and run. Independent annotated road-scene accuracy, independent zoom/temporal calibration, and a physical reboot remain unverified. The requested new-footage path was not supplied during this run, so those acceptance checks are not marked complete.
