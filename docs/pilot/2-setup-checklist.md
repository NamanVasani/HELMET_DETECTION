# Pilot setup checklist (install day)

Written for the person installing Helmet at a customer site. Tick each line.

## Before the visit
- [ ] Customer has signed off the pilot offer (length, scope, contact person).
- [ ] Laptop/mini PC prepared: Helmet installed, `.venv/bin/python -B -m unittest ...` passes, `python install_service.py --check` shows up to date.
- [ ] Notice board text printed (see the end of this file).
- [ ] Ask the customer for: CCTV recorder brand/model and an admin login, or permission to mount a camera.

## At the site
1. **Camera view**
   - [ ] Camera sees riders **from the front or side, at 5–20 m**, not from directly above.
   - [ ] A rider's head is at least ~40 pixels tall at 1080p (zoom in on a test frame to check).
   - [ ] No strong backlight (sun behind riders); check morning and evening if possible.
2. **Stream**
   - [ ] Get the RTSP address (examples: Hikvision `rtsp://user:pass@IP:554/Streaming/Channels/101`, Dahua `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0`).
   - [ ] Test it opens in VLC (Media → Open Network Stream).
   - [ ] Set the stream to **1080p** in the recorder if it is lower.
3. **Helmet**
   - [ ] Settings → CCTV cameras → Add camera → name, site, RTSP address → Save.
   - [ ] Live tab shows the camera as **live** with FPS above 3.
   - [ ] Settings → **Draw zone** around the entry lane only.
   - [ ] Create accounts: one **reviewer** per guard, the customer contact as **admin** or **viewer**.
   - [ ] Settings → Alerts: WhatsApp/email to the customer contact; **Send test alert** works.
   - [ ] Daily report email at 08:00 turned on.
4. **Check with real traffic (30 minutes)**
   - [ ] Watch the Live tab while riders pass; note any missed no-helmet riders.
   - [ ] Review the first cases with the guard using **Quick review** (c / f / s).
   - [ ] Show the guard how to correct plate text and mark false positives.

## Before leaving
- [ ] Restart the computer once; confirm the dashboard comes back on its own.
- [ ] Note the computer's IP, the dashboard address and who has which account (store securely, not on paper at the gate).
- [ ] Put up the notice board.
- [ ] Agree a check-in call at the end of week 1.

## Week 1 accuracy check
- [ ] Export the week's events (Review → CSV and ZIP).
- [ ] Watch 30 minutes of recorded footage; count no-helmet riders a person sees vs. cases Helmet saved.
- [ ] Share detection rate and false-alarm rate with the customer.

## Notice board text
> **Helmet checks in progress.** Cameras at this gate check whether two-wheeler riders wear a helmet.
> Images of riders without a helmet are kept for review by [ORGANISATION] for up to 30 days.
> Riders wearing a helmet have their faces blurred. Questions: [CONTACT].
