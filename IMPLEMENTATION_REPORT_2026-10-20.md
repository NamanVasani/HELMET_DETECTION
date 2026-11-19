# Section 3 fixes — 20 October 2026

Section 3 is partially addressed. Passing tests does not establish perfect recognition, plate ownership, OCR, or independent road accuracy. Section 4 acceptance and section 5 additions have not started.

## Implemented software fixes

- Returning a review to needs review recalculates its association label. Review fields must be text; failed OCR commands leave the plate blank.
- Associations require exactly one containing rider, one overlapping bike within the existing distance bound, and one plate centre inside that bike. Ambiguity stays unmatched; geometry still does not prove ownership.
- Full-frame head/rider tracks confirm independently in three agreeing processed frames. Zoom selection no longer serializes full-frame capture. Missing predictions reset a track's agreement; known head/rider aliases carry confirmation and cooldown across temporary association loss.
- Known tracks retain their 30-second capture cooldown when moving. A distinct known track at the same position is eligible. ID changes and untracked overlapping people remain limitations.
- Enlarged inference runs only when the selected target lacks a qualifying full-frame head prediction. Resolution, checkpoint, and confidence thresholds are unchanged. OpenCV uses at most two workers, based on local face-detector timing.
- Video jobs retain a clip ID and sanitized original basename. Events record their zero-based frame index and clip position, using FPS as a fallback when the decoder omits timestamps. The dashboard distinguishes processing time from footage position and labels old records with unknown source.
- The existing HTTP server checks retention hourly while open, including review-only mode. Failures are logged and retried; no new process or dependency is needed.
- One upload/queued/processing video is accepted at a time, preventing overlapping job progress and source confusion.
- Dashboard notices identify captures without vehicle context. Home-camera testing remains available; those captures are not proof of a traffic violation.
- Old web.py and hi_show.py commands delegate to hi.main; nano train.py delegates to train.main, removing invalid checkpoint and machine-specific paths.

## Verification

- All 55 final Python regression tests passed (including multi-target confirmation, association loss, moving-track cooldown, review reset, OCR failures, video provenance, busy upload rejection, and scheduled retention). The superseded serial confirmation path and its duplicate tests were removed.
- Browser JavaScript checks pass, including escaped video filenames, clip positions, unknown legacy sources, camera retries, review failures, and upload feedback.
- Python source syntax checks pass.
- Fresh real-model HTTP/video replay passed using disposable event storage. Source metadata and three-frame confirmation are asserted. No physical camera was opened and no user events were created or edited for testing.
- Same multi-person replay: 3.80 to 5.03 processed FPS; 5 to 13 saved events. Synthetic low light: 3.83 to 5.56 FPS; 5 to 8 events. Synthetic distance: 3.97 to 6.68 FPS; 7 to 5 events. Changes in event counts can reflect identity grouping, different crop use, or changed predictions; they do not establish better recall or fewer false positives.
- See reports/comparison-2026-10-20.json and reports/usage-2026-10-20.json. The October-13 evaluation and usage reports are historical and have not been overwritten.

## Still open in section 3

- Helmet/no-helmet, rider, and bike model errors remain. The checkpoint and calibrated thresholds have not been changed without new accuracy evidence.
- Reliable OCR and plate ownership require labelled footage. Conservative matching can reduce incorrect suggestions while increasing unmatched events.
- Untracked overlap and track-ID changes cannot be resolved as person identity by these checks.
- Vehicle-context labels preserve existing home testing; a validated road-only capture policy has not been introduced.
- Processing remains below the rate needed to guarantee coverage of fast traffic.

## Acceptance boundary

New, independently labelled road footage is required before section 3 accuracy can be accepted. No such local path was supplied in this session. Section 4 additionally requires real night/rain/glare/blur scenes, crop/event calibration, a deliberate physical reboot/login, fresh training checks, and additional platforms. No reboot or long training run was started, and no section-5 module was added.

## Applied to the existing service

The three runtime source files were applied to the idle login service and it restarted successfully on port 8003. All 63 existing events and all 189 evidence files were preserved unchanged, and SQLite integrity passed. Runtime source bytes match the tested workspace. See reports/deployment-2026-10-20.json. This activation is not a physical reboot/login or an accuracy test.
