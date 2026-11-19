# Privacy and data protection

Helmet processes images of people and vehicles, which are personal data. This document describes what the software
does by default and what the **operator** (the organisation that installs it) must do. It is not legal advice; have it
reviewed for your deployment.

## What is collected

| Data | When | Where |
|---|---|---|
| Live frames | while a camera or video is being analysed | memory only; never written to disk |
| Uploaded video | during analysis | a temporary file, deleted when analysis ends, fails or is cancelled |
| Event evidence: person crop, context crop, optional plate crop, plate text, scores, boxes, track IDs, camera, time | when a detection is confirmed over 3 frames | `storage/YYYY-MM-DD/<category>/event_*` and `storage/events.db` |
| Accounts: username, scrypt password hash, role; sessions (hashed token, expiry) | when users are created / sign in | `storage/events.db` |
| Audit log: user, action, target, time; sign-in IP for login events | on each sign-in, review, deletion, export, upload, settings change, alert | `storage/events.db` |
| Alert settings (webhook token, SMTP password) | when an admin configures alerts | `storage/events.db`, never returned to the browser |

The software does not perform face recognition, does not identify people, and sends nothing to any external service
except the alert webhook / email server an admin configures.

## Defaults that reduce data

- **Privacy mode** (`HELMET_PRIVACY=1`): for riders wearing a helmet — who are not violating anything — detected faces are
  blurred in the saved images and no plate image or plate text is kept.
- **Retention** (`HELMET_RETENTION_DAYS=30`): unconfirmed and false-positive evidence is deleted after 30 days.
  Confirmed events are kept until deleted (`HELMET_KEEP_CONFIRMED=1`) because they may be needed for follow-up; set a
  review process to delete them when no longer needed.
- **Local only**: the dashboard listens on `127.0.0.1` unless configured otherwise, and refuses network access without
  login.
- **Least privilege**: viewers cannot change or export data; only admins manage cameras, alerts and users.

## Operator checklist (India — Digital Personal Data Protection Act, 2023)

- [ ] **Purpose and notice**: state the specific purpose (e.g. "helmet compliance at the campus gate") and put visible
      notices at monitored locations, in the languages people at that site read, with a contact for questions.
- [ ] **Lawful basis**: confirm the basis for processing (consent, or a legitimate use permitted under the Act) for each
      site, and record it.
- [ ] **Data minimisation**: keep privacy mode on; point cameras only at the area needed; use the shortest retention
      that meets the purpose.
- [ ] **Access control**: give each person their own account with the lowest role that works; remove leavers.
- [ ] **Security**: run behind HTTPS when accessed over a network; keep the host patched; back up `storage/` encrypted.
- [ ] **Rights requests**: be able to find and delete a person's events on request (search by date/camera/plate, then
      delete; deletions are recorded in the audit log).
- [ ] **Breach response**: know who to notify and how fast if evidence or accounts are exposed.
- [ ] **No automated decisions**: a person must confirm every event before any action; the model's output is a
      suggestion, and plate association does not prove ownership.

## Export and deletion

- Exports (CSV, ZIP, PDF) are logged. Treat exported files as personal data.
- **Remove** deletes an event's images and record permanently. Bulk delete is available for range clean-up.
- `python manage.py cleanup` applies the retention policy immediately.
