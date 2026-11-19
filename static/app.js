'use strict';
/* Helmet review dashboard. No external libraries; works under a strict Content-Security-Policy. */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const PAGE_SIZE = 60;
const STATUS_LABELS = {needs_review: 'Needs review', confirmed: 'Confirmed', false_positive: 'False positive'};
const state = {
  user: null, perms: new Set(), tab: 'review', category: 'no_helmet', events: [], total: 0, signature: '',
  selected: new Set(), cameraStream: null, sendingFrame: false, lastFrameAt: 0, startingCamera: false,
  uploadingVideo: false, upload: null, status: null, cameras: [], focusEvent: null, ticks: 0, authMode: 'login',
};

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}
function today() { return new Date().toLocaleDateString('en-CA'); }
function addDays(iso, days) {
  const value = new Date(iso + 'T12:00:00');
  value.setDate(value.getDate() + days);
  return value.toLocaleDateString('en-CA');
}
function can(permission) { return state.perms.has(permission); }
function setText(selector, text) { const el = $(selector); if (el) el.textContent = text; }
function flash(message, isError = false) {
  const el = $('#status');
  if (!el) return;
  el.textContent = message;
  el.classList.toggle('error', isError);
}

async function api(url, options = {}) {
  const headers = {...(options.headers || {})};
  if (options.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    options = {...options, body: JSON.stringify(options.json)};
  }
  const response = await fetch(url, {credentials: 'same-origin', ...options, headers});
  let data = {};
  try { data = await response.json(); } catch { data = {}; }
  if (response.status === 401 && !url.startsWith('/api/login') && !url.startsWith('/api/account')) {
    showAuth('login');
    throw new Error('Sign in to continue.');
  }
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

/* ------------------------------------------------------------------ theme */
function applyTheme(theme) {
  try { if (theme === undefined) theme = localStorage.getItem('helmet-theme') || ''; } catch { theme = theme || ''; }
  if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
  setText('#theme-toggle', theme === 'dark' ? 'Dark' : theme === 'light' ? 'Light' : 'Auto theme');
}
function cycleTheme() {
  const order = ['', 'dark', 'light'];
  const current = document.documentElement.dataset.theme || '';
  const next = order[(order.indexOf(current) + 1) % order.length];
  try { localStorage.setItem('helmet-theme', next); } catch { /* storage unavailable */ }
  applyTheme(next);
}

/* ------------------------------------------------------------------ auth */
function showAuth(mode) {
  state.authMode = mode;
  stopCamera();
  $('#app').hidden = true;
  $('#auth-view').hidden = false;
  const setup = mode === 'setup';
  setText('#auth-title', setup ? 'Create the admin account' : 'Sign in');
  setText('#auth-help', setup ? 'First run: choose an admin username and a password of at least 10 characters.'
    : 'Sign in to review helmet detections.');
  setText('#auth-submit', setup ? 'Create admin' : 'Sign in');
  $('#auth-confirm-row').hidden = !setup;
  $('#auth-pass').autocomplete = setup ? 'new-password' : 'current-password';
  $('#auth-user').focus?.();
}

async function submitAuth(event) {
  event.preventDefault();
  const username = $('#auth-user').value.trim(), password = $('#auth-pass').value;
  setText('#auth-error', '');
  if (state.authMode === 'setup' && password !== $('#auth-confirm').value) {
    setText('#auth-error', 'Passwords do not match.');
    return;
  }
  try {
    await api(state.authMode === 'setup' ? '/api/setup' : '/api/login', {method: 'POST', json: {username, password}});
    $('#auth-pass').value = '';
    $('#auth-confirm').value = '';
    await boot();
  } catch (error) { setText('#auth-error', error.message); }
}

async function logout() {
  stopCamera();
  try { await api('/api/logout', {method: 'POST'}); } catch { /* already signed out */ }
  showAuth('login');
}

/* ------------------------------------------------------------------ tabs */
function selectTab(tab) {
  if (tab === 'settings' && !can('admin')) tab = 'review';
  state.tab = tab;
  $$('.tabs [role="tab"]').forEach(button => button.setAttribute('aria-selected', String(button.dataset.tab === tab)));
  $$('.tab').forEach(panel => { panel.hidden = panel.id !== `tab-${tab}`; });
  syncStreams();
  if (tab === 'analytics') loadAnalytics();
  if (tab === 'settings') loadSettings();
  if (tab === 'live') loadCameraStreams();
}

/* ------------------------------------------------------------------ review */
function filterParams(extra = {}) {
  const params = new URLSearchParams();
  const values = {start: $('#f-start').value, end: $('#f-end').value, category: state.category,
    status: $('#f-status').value, camera: $('#f-camera').value, q: $('#f-q').value.trim(), ...extra};
  Object.entries(values).forEach(([key, value]) => { if (value !== '' && value != null) params.set(key, value); });
  return params;
}

function updateExportLinks() {
  const params = filterParams();
  $('#export-csv').href = '/api/export.csv?' + params;
  $('#export-zip').href = '/api/export.zip?' + params;
  const report = new URLSearchParams({start: $('#f-start').value, end: $('#f-end').value});
  if ($('#f-camera').value) report.set('camera', $('#f-camera').value);
  $('#export-pdf').href = '/api/report.pdf?' + report;
}

function editing() {
  const active = document.activeElement;
  return Boolean(active && $('#events').contains(active) && /^(INPUT|SELECT|TEXTAREA)$/.test(active.tagName));
}

async function loadEvents({append = false, force = false} = {}) {
  const offset = append ? state.events.length : 0;
  try {
    const data = await api('/api/events?' + filterParams({limit: PAGE_SIZE, offset}));
    const events = append ? state.events.concat(data.events) : data.events;
    const signature = JSON.stringify([data.total, events.map(e => [e.id, e.status, e.plate_text])]);
    $('#load-more').hidden = events.length >= data.total;
    if (!force && !append && signature === state.signature) return;
    if (!force && !append && editing()) return;
    state.events = events;
    state.total = data.total;
    state.signature = signature;
    const keys = new Set(events.map(eventKey));
    state.selected.forEach(key => { if (!keys.has(key)) state.selected.delete(key); });
    renderEvents();
    flash('');
  } catch (error) { flash(`Could not load events: ${error.message} Retrying automatically.`, true); }
}

async function loadStats() {
  try {
    const params = new URLSearchParams({start: $('#f-start').value, end: $('#f-end').value});
    if ($('#f-camera').value) params.set('camera', $('#f-camera').value);
    const data = await api('/api/stats?' + params);
    renderTotals($('#stats'), data.totals);
    fillCameraSelects(data.cameras);
  } catch { /* stats are informational */ }
}

function renderTotals(target, totals) {
  const rate = totals.false_positive_rate == null ? '–' : Math.round(totals.false_positive_rate * 100) + '%';
  const tiles = [['Events', totals.events], ['No helmet', totals.no_helmet], ['Helmet', totals.helmet],
    ['Needs review', totals.needs_review], ['Confirmed', totals.confirmed], ['False positive', totals.false_positive],
    ['False-positive rate', rate]];
  target.innerHTML = tiles.map(([label, value]) => `<div class="stat"><small>${esc(label)}</small><b>${esc(value)}</b></div>`).join('');
}

function fillCameraSelects(cameras) {
  for (const id of ['#f-camera', '#a-camera']) {
    const select = $(id);
    if (!select) continue;
    const current = select.value;
    const names = new Set(cameras || []);
    if (current) names.add(current);
    const options = ['<option value="">All cameras</option>'].concat([...names].sort().map(
      name => `<option value="${esc(name)}">${esc(name === 'camera' ? 'Browser (older events)' : name)}</option>`));
    if (select.dataset.options !== options.join('')) {
      select.innerHTML = options.join('');
      select.dataset.options = options.join('');
      select.value = current;
    }
  }
}

function eventKey(event) { return `${event.date}/${event.id}`; }
function eventBase(event) {
  return `/media/${encodeURIComponent(event.date)}/${encodeURIComponent(event.category || 'no_helmet')}/${encodeURIComponent(event.id)}/`;
}

function sourceText(event) {
  const source = event.source || {};
  if (source.type === 'video') {
    return `Video: ${esc(source.filename)} · Frame ${esc(source.frame)} · ${esc(source.position_seconds ?? 'unknown')} seconds`;
  }
  if (source.type === 'rtsp') return `CCTV: ${esc(source.camera)}${source.site ? ' · ' + esc(source.site) : ''}`;
  if (source.type === 'camera') return 'Browser camera';
  return 'Source not recorded';
}

function cardHtml(event, index) {
  const category = event.category || 'no_helmet';
  const label = category === 'helmet' ? 'Helmet' : 'No helmet';
  const files = event.files || {};
  const detection = event.event_detection || event.helmet || event.no_helmet || {};
  const status = event.status || 'needs_review';
  const key = eventKey(event);
  const base = eventBase(event);
  const time = new Date(event.timestamp);
  const when = isNaN(time) ? esc(event.timestamp) : `${esc(event.date)} ${esc(time.toLocaleTimeString())}`;
  const selected = state.selected.has(key);
  const privacy = event.privacy && event.privacy.faces_blurred ? ` · ${esc(event.privacy.faces_blurred)} face(s) blurred` : '';
  const plateNote = event.plate_text ? (event.plate_format_valid ? ' ✓ format' : ' · unusual format') : '';
  const controls = can('review') ? `<div class="actions">
      <label>Review <select name="review-status">${Object.entries(STATUS_LABELS).map(([value, text]) =>
        `<option value="${value}" ${status === value ? 'selected' : ''}>${text}</option>`).join('')}</select></label>
      <label>Plate text <input name="plate-text" maxlength="16" value="${esc(event.plate_text || '')}" autocomplete="off"></label>
      <button type="button" class="ok save">Save</button>
      ${can('delete') ? '<button type="button" class="bad remove">Remove</button>' : ''}
    </div>` : '';
  return `<article class="card ${selected ? 'selected' : ''}" tabindex="0" data-key="${esc(key)}" data-index="${index}">
    <div class="card-head">
      ${can('review') ? `<label class="check"><input type="checkbox" class="pick" ${selected ? 'checked' : ''} aria-label="Select event"><b>${when}</b></label>` : `<b>${when}</b>`}
      <span class="badge ${esc(category)}">${label}</span>
    </div>
    <p class="notice"><span class="badge ${esc(status)}">${esc(STATUS_LABELS[status] || status)}</span> ${sourceText(event)} · time is processing time${privacy}</p>
    <div class="images">
      <figure><button type="button" class="zoom" data-src="${base}person.jpg" data-caption="${esc(label)} person crop"><img src="${base}person.jpg" alt="${esc(label)} person crop" loading="lazy"></button><figcaption>Person</figcaption></figure>
      <figure>${files.plate ? `<button type="button" class="zoom" data-src="${base}plate.jpg" data-caption="Suggested plate"><img src="${base}plate.jpg" alt="Suggested number plate" loading="lazy"></button>` : '<div class="noimg">No plate matched</div>'}
        <figcaption>Plate · ${esc(event.plate_text || 'not read')}${plateNote}</figcaption></figure>
    </div>
    <p class="notice">Score ${esc(Number(detection.confidence).toFixed(2))} · Rider ${event.rider ? '#' + esc(event.rider.track_id ?? '—') : 'not matched'} · Bike ${event.bike ? '#' + esc(event.bike.track_id ?? '—') : 'not matched'} · Association: ${esc(event.association_status)}
      · <button type="button" class="ghost zoom context" data-src="${base}event.jpg" data-caption="Event context">Context</button></p>
    <p class="notice">${event.rider && event.bike ? 'Rider and bike suggested; verify the traffic context.' : 'Vehicle context not verified; this may be a head-only test capture.'}</p>
    ${controls}
  </article>`;
}

function renderEvents() {
  const container = $('#events');
  container.innerHTML = state.events.map(cardHtml).join('') || '<p class="muted">No events match these filters.</p>';
  updateSelection();
  if (state.focusEvent) {
    const card = $$('.card', container).find(item => item.dataset.key.endsWith('/' + state.focusEvent));
    if (card) { card.focus(); card.scrollIntoView?.({block: 'center'}); }
    state.focusEvent = null;
  }
}

function updateSelection() {
  const count = state.selected.size;
  setText('#selected-count', `${count} selected`);
  $$('#bulkbar [data-bulk]').forEach(button => { button.disabled = count === 0; });
  const cards = $$('#events .card');
  $('#select-all').checked = cards.length > 0 && cards.every(card => state.selected.has(card.dataset.key));
  cards.forEach(card => card.classList.toggle('selected', state.selected.has(card.dataset.key)));
}

function eventFromCard(card) { return state.events[Number(card.dataset.index)]; }

async function reviewEvent(card, status) {
  const event = eventFromCard(card);
  const select = $('[name="review-status"]', card), plate = $('[name="plate-text"]', card);
  const buttons = $$('button', card);
  buttons.forEach(button => { button.disabled = true; });
  try {
    await api('/api/review', {method: 'POST', json: {date: event.date, id: event.id,
      status: status || select.value, plate_text: plate ? plate.value : (event.plate_text || '')}});
    flash('Review saved');
    document.activeElement?.blur?.();
    await loadEvents({force: true});
    loadStats();
  } catch (error) {
    flash(`Could not save review: ${error.message} Your edits are kept; retry.`, true);
  } finally { buttons.forEach(button => { button.disabled = false; }); }
}

async function removeEvent(card) {
  const event = eventFromCard(card);
  if (!confirm('Permanently remove this event and its saved pictures?')) return;
  try {
    await api('/api/delete', {method: 'POST', json: {date: event.date, id: event.id}});
    state.selected.delete(eventKey(event));
    flash('Removed');
    await loadEvents({force: true});
    loadStats();
  } catch (error) { flash(`Could not remove event: ${error.message} Retry.`, true); }
}

async function bulkAction(action) {
  const items = [...state.selected].map(key => { const [date, id] = key.split('/'); return {date, id}; });
  if (!items.length) return;
  if (action === 'delete' && !confirm(`Permanently remove ${items.length} event(s) and their pictures?`)) return;
  try {
    const body = action === 'delete' ? {action: 'delete', items} : {action: 'review', status: action, items};
    const data = await api('/api/bulk', {method: 'POST', json: body});
    const failed = data.results.filter(item => !item.ok).length;
    state.selected.clear();
    flash(failed ? `${items.length - failed} updated, ${failed} failed.` : `${items.length} event(s) updated.`, failed > 0);
    await loadEvents({force: true});
    loadStats();
  } catch (error) { flash(`Bulk action failed: ${error.message}`, true); }
}

function openLightbox(src, caption) {
  const dialog = $('#lightbox');
  $('#lightbox-img').src = src;
  $('#lightbox-img').alt = caption;
  setText('#lightbox-caption', caption);
  if (dialog.showModal) dialog.showModal(); else dialog.setAttribute('open', '');
}

function onEventsClick(event) {
  const card = event.target.closest('.card');
  if (!card) return;
  const zoom = event.target.closest('.zoom');
  if (zoom) { openLightbox(zoom.dataset.src, zoom.dataset.caption); return; }
  if (event.target.closest('.save')) reviewEvent(card);
  else if (event.target.closest('.remove')) removeEvent(card);
}

function onEventsChange(event) {
  if (!event.target.classList.contains('pick')) return;
  const key = event.target.closest('.card').dataset.key;
  if (event.target.checked) state.selected.add(key); else state.selected.delete(key);
  updateSelection();
}

function onKey(event) {
  const target = event.target;
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName);
  if (event.key === 'Escape' && typing) { target.blur(); return; }
  if (typing || event.metaKey || event.ctrlKey || event.altKey || $('#app').hidden) return;
  if (event.key === '?') { $('#help').showModal?.(); event.preventDefault(); return; }
  if (state.tab !== 'review') return;
  if (event.key === '/') { $('#f-q').focus(); event.preventDefault(); return; }
  const cards = $$('#events .card');
  if (!cards.length) return;
  const current = target.closest?.('.card');
  const index = current ? cards.indexOf(current) : -1;
  const move = step => { const next = cards[Math.max(0, Math.min(cards.length - 1, index + step))]; next.focus(); next.scrollIntoView?.({block: 'nearest'}); };
  if (event.key === 'j') { move(index < 0 ? 0 : 1); event.preventDefault(); return; }
  if (event.key === 'k') { move(index < 0 ? 0 : -1); event.preventDefault(); return; }
  if (!current) return;
  const quick = {c: 'confirmed', f: 'false_positive', n: 'needs_review'}[event.key];
  if (quick && can('review')) { reviewEvent(current, quick); event.preventDefault(); }
  else if (event.key === 'x' && can('review')) {
    const box = $('.pick', current); box.checked = !box.checked; box.dispatchEvent(new Event('change', {bubbles: true}));
  } else if ((event.key === 'Delete' || event.key === 'Backspace') && can('delete')) { removeEvent(current); event.preventDefault(); }
  else if (event.key === 'o') { const context = $('.context', current); openLightbox(context.dataset.src, context.dataset.caption); }
}

function filtersChanged() {
  state.signature = '';
  updateExportLinks();
  loadEvents({force: true});
  loadStats();
}

function setRange(days) {
  $('#f-end').value = today();
  $('#f-start').value = addDays(today(), -Number(days));
  filtersChanged();
}

/* ------------------------------------------------------------------ live camera */
function stopCamera(message = '') {
  if (state.cameraStream) state.cameraStream.getTracks().forEach(track => track.stop());
  state.cameraStream = null;
  const video = $('#camera-source');
  if (video) video.srcObject = null;
  if (message) setText('#camera-status', message);
}

async function restartDetector() {
  stopCamera('Restarting detector…');
  try {
    await api('/api/detector/restart', {method: 'POST'});
    setText('#camera-status', 'Detector restart requested. Start the camera again when it is ready.');
  } catch { setText('#camera-status', 'Could not restart detector. Check the connection and retry.'); }
}

async function sendCameraFrames(now) {
  const stream = state.cameraStream, video = $('#camera-source'), canvas = $('#camera-frame');
  if (!stream) return;
  if (now - state.lastFrameAt >= 120 && !state.sendingFrame && video.readyState >= 2) {
    state.lastFrameAt = now;
    state.sendingFrame = true;
    const scale = Math.min(1, 1280 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    try {
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.72));
      if (blob && state.cameraStream === stream) {
        const response = await fetch('/api/frame', {method: 'POST', headers: {'Content-Type': 'image/jpeg'}, body: blob,
          credentials: 'same-origin', signal: AbortSignal.timeout(10000)});
        if (!response.ok) throw new Error('Camera frame rejected');
      }
    } catch {
      if (state.cameraStream === stream) stopCamera('Camera detector unavailable. Retry the camera when the detector is ready.');
      return;
    } finally { state.sendingFrame = false; }
  }
  if (state.cameraStream === stream) requestAnimationFrame(sendCameraFrames);
}

async function startCamera() {
  if (state.cameraStream || state.startingCamera) return;
  state.startingCamera = true;
  try {
    setText('#camera-status', 'Checking camera detector…');
    let data;
    try { data = await api('/api/status', {cache: 'no-store'}); } catch {
      stopCamera('Cannot connect to the camera detector. Start the helmet app, then retry.');
      return;
    }
    if (!data.camera_enabled) {
      setText('#camera-status', 'Saved-event review is running. Start the helmet camera detector, then retry.');
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setText('#camera-status', 'Camera access is unavailable. Open this site on localhost or HTTPS.');
      return;
    }
    setText('#camera-status', 'Requesting browser camera permission…');
    state.cameraStream = await navigator.mediaDevices.getUserMedia({audio: false, video: {width: {ideal: 1280}, height: {ideal: 720}}});
    const video = $('#camera-source');
    video.srcObject = state.cameraStream;
    await video.play();
    setText('#camera-status', 'Camera allowed. Starting live detection…');
    requestAnimationFrame(sendCameraFrames);
    syncStreams();
  } catch (error) {
    const messages = {
      NotAllowedError: 'Camera permission denied. Allow camera access for this site and for your browser in system settings, then retry.',
      NotFoundError: 'No camera was found. Connect a camera, then retry.',
      NotReadableError: 'Camera could not start. Close other apps using it, check camera permission, then retry.',
    };
    stopCamera(messages[error.name] || 'Camera unavailable. Check browser and system camera settings, then retry.');
  } finally { state.startingCamera = false; }
}

/* Only hold MJPEG connections while the Live tab is visible. */
function syncStreams() {
  const live = state.tab === 'live' && state.status?.camera_enabled;
  const image = $('#live-stream');
  const wantLive = live && (state.status.sources || []).includes('live');
  if (wantLive && !image.getAttribute('src')) image.src = '/stream?source=live';
  if (!wantLive && image.getAttribute('src')) image.removeAttribute('src');
  image.hidden = !wantLive;
  $('#live-empty').hidden = wantLive;
  $$('#camera-streams img[data-source]').forEach(img => {
    if (live && !img.getAttribute('src')) img.src = '/stream?source=' + encodeURIComponent(img.dataset.source);
    if (!live && img.getAttribute('src')) img.removeAttribute('src');
  });
}

function analyzeVideo() {
  if (state.uploadingVideo) return;
  const file = $('#video-file').files[0];
  const status = $('#video-status');
  if (!file) { status.textContent = 'Choose a video first.'; return; }
  if (file.size > 250 * 1024 * 1024) { status.textContent = 'Video must be no larger than 250 MB.'; return; }
  if (!/\.(mp4|mov|avi|mkv|m4v)$/i.test(file.name)) { status.textContent = 'Choose an MP4, MOV, AVI, MKV, or M4V video.'; return; }
  state.uploadingVideo = true;
  $('#analyze-video').disabled = true;
  $('#cancel-video').hidden = false;
  const progress = $('#video-progress');
  progress.hidden = false;
  progress.value = 0;
  status.textContent = 'Uploading video…';
  const request = new XMLHttpRequest();
  state.upload = request;
  request.open('POST', '/api/video');
  request.timeout = 600000;
  request.setRequestHeader('Content-Type', file.type || 'application/octet-stream');
  request.setRequestHeader('X-File-Name', encodeURIComponent(file.name));
  const finish = () => { state.uploadingVideo = false; state.upload = null; $('#analyze-video').disabled = false; };
  request.upload.onprogress = event => {
    if (event.lengthComputable) {
      const percent = Math.round(event.loaded / event.total * 100);
      progress.value = percent;
      status.textContent = `Uploading video: ${percent}%`;
    }
  };
  request.onload = () => {
    finish();
    let data = {};
    try { data = JSON.parse(request.responseText); } catch { data = {}; }
    status.textContent = request.status === 202 ? 'Video queued. Waiting for analysis.' : (data.error || 'Could not queue video. Check the detector and retry.');
    if (request.status !== 202) { progress.hidden = true; $('#cancel-video').hidden = true; }
  };
  request.onerror = request.ontimeout = () => {
    finish();
    progress.hidden = true;
    $('#cancel-video').hidden = true;
    status.textContent = 'Upload interrupted. Check the connection and retry.';
  };
  request.onabort = () => { finish(); progress.hidden = true; $('#cancel-video').hidden = true; status.textContent = 'Upload canceled.'; };
  request.send(file);
}

async function cancelVideo() {
  if (state.upload) { state.upload.abort(); return; }
  try {
    await api('/api/video/cancel', {method: 'POST'});
    setText('#video-status', 'Canceling analysis…');
  } catch (error) { setText('#video-status', error.message); }
}

function renderVideo(video) {
  if (!video || state.uploadingVideo) return;
  const progress = $('#video-progress'), status = $('#video-status');
  const active = video.state === 'processing' || video.state === 'queued';
  $('#cancel-video').hidden = !active;
  progress.hidden = video.state !== 'processing';
  if (video.state === 'processing') {
    progress.value = video.total ? Math.round(video.frames / video.total * 100) : 0;
    status.textContent = `Analyzing video: ${video.frames} frames${video.total ? ' of ' + video.total : ''}. Live camera is paused.`;
  } else if (video.state === 'complete') status.textContent = 'Video analysis complete. Live camera can resume.';
  else if (video.state === 'failed') status.textContent = video.error;
  else if (video.state === 'queued') status.textContent = 'Video queued. Waiting for analysis.';
}

function renderMetrics(data) {
  const entries = Object.entries(data.metrics || {});
  $('#metrics').innerHTML = entries.length ? entries.map(([source, values]) => `<div class="stat"><small>${esc(source.replace('camera:', 'CCTV '))}</small>
    <b>${esc(values.fps)}</b> fps<br><small>${esc(values.frames)} frames · ${esc(values.events)} events</small></div>`).join('')
    : '<p class="muted">No frames processed yet.</p>';
  const pill = $('#detector-pill');
  pill.textContent = data.camera_enabled ? 'Detector live' : 'Review only';
  pill.className = 'pill ' + (data.camera_enabled ? 'live' : 'down');
  pill.title = data.status;
}

async function refreshStatus() {
  try {
    const data = await api('/api/status', {cache: 'no-store'});
    state.status = data;
    renderMetrics(data);
    renderVideo(data.video);
    if (state.cameraStream && !data.camera_enabled) stopCamera(data.status);
    else if (state.cameraStream) setText('#camera-status', data.status);
    else if (!state.startingCamera && !/denied|No camera|could not start/i.test($('#camera-status').textContent)) {
      setText('#camera-status', data.camera_enabled ? 'Detector ready. Press Start camera to begin.' : 'Saved-event review only. Run the detector (hi.py) for live detection.');
    }
    syncStreams();
  } catch {
    if (state.cameraStream) stopCamera('Cannot connect to the detector. Camera stopped; retry when the app is available.');
    const pill = $('#detector-pill');
    pill.textContent = 'Offline';
    pill.className = 'pill down';
  }
}

async function loadCameraStreams() {
  try {
    const data = await api('/api/cameras');
    state.cameras = data.cameras;
    $('#camera-streams').innerHTML = data.cameras.length ? data.cameras.map(camera => `<article class="card">
      <h3>${esc(camera.name)}${camera.site ? ' <small class="muted">' + esc(camera.site) + '</small>' : ''}</h3>
      <div class="viewer">${camera.enabled ? `<img data-source="camera:${esc(camera.id)}" alt="${esc(camera.name)} live">` : '<p>Disabled</p>'}</div>
      <p class="notice">${esc(camera.state || (camera.enabled ? 'starting' : 'disabled'))}${camera.error ? ' · ' + esc(camera.error) : ''}${camera.fps ? ' · ' + esc(camera.fps) + ' fps' : ''}</p>
    </article>`).join('') : `<p class="muted">No CCTV cameras configured.${can('admin') ? ' Add RTSP cameras in Settings.' : ''}</p>`;
    syncStreams();
  } catch { /* shown on next refresh */ }
}

/* ------------------------------------------------------------------ analytics */
function chartSvg(days, start, end) {
  const byDate = Object.fromEntries(days.map(day => [day.date, day]));
  const series = [];
  for (let day = start; day <= end && series.length < 400; day = addDays(day, 1)) {
    series.push(byDate[day] || {date: day, helmet: 0, no_helmet: 0, events: 0});
  }
  const width = Math.max(320, Math.round($('#a-chart').clientWidth || 900)), height = 260, left = 36, bottom = 24, top = 10;
  const peak = Math.max(1, ...series.map(day => day.helmet + day.no_helmet));
  const slot = (width - left) / Math.max(1, series.length);
  const bar = Math.max(1, slot * 0.72);
  const scale = value => (height - bottom - top) * value / peak;
  const labelEvery = Math.ceil(series.length / 10);
  const bars = series.map((day, index) => {
    const x = left + index * slot + (slot - bar) / 2;
    const helmet = scale(day.helmet), violation = scale(day.no_helmet);
    const base = height - bottom;
    const label = index % labelEvery === 0 ? `<text x="${x + bar / 2}" y="${height - 6}" text-anchor="middle">${esc(day.date.slice(5))}</text>` : '';
    return `<g><title>${esc(day.date)}: ${day.no_helmet} no helmet, ${day.helmet} helmet</title>
      ${helmet ? `<rect class="bar-helmet" x="${x}" y="${base - helmet}" width="${bar}" height="${helmet}" rx="1.5"></rect>` : ''}
      ${violation ? `<rect class="bar-violation" x="${x}" y="${base - helmet - violation}" width="${bar}" height="${violation}" rx="1.5"></rect>` : ''}${label}</g>`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Events per day from ${esc(start)} to ${esc(end)}">
    <line class="axis" x1="${left}" y1="${height - bottom}" x2="${width}" y2="${height - bottom}"></line>
    <text x="${left - 6}" y="${top + 8}" text-anchor="end">${peak}</text><text x="${left - 6}" y="${height - bottom}" text-anchor="end">0</text>
    ${bars}</svg>`;
}

async function loadAnalytics() {
  const start = $('#a-start').value, end = $('#a-end').value;
  try {
    const params = new URLSearchParams({start, end});
    if ($('#a-camera').value) params.set('camera', $('#a-camera').value);
    const data = await api('/api/stats?' + params);
    renderTotals($('#a-totals'), data.totals);
    fillCameraSelects(data.cameras);
    $('#a-chart').innerHTML = data.days.length ? chartSvg(data.days, start, end) : '<p class="muted">No events in this period.</p>';
  } catch (error) { $('#a-chart').innerHTML = `<p class="error">${esc(error.message)}</p>`; }
}

/* ------------------------------------------------------------------ settings */
function cameraRow(camera = {}) {
  const row = document.createElement('div');
  row.className = 'camera-row';
  row.dataset.id = camera.id || '';
  row.innerHTML = `<label>Name <input name="name" value="${esc(camera.name || '')}" maxlength="60" required></label>
    <label>Site <input name="site" value="${esc(camera.site || '')}" maxlength="60"></label>
    <label>Stream URL <input name="url" value="${esc(camera.url || '')}" placeholder="rtsp://user:pass@192.168.1.20/stream1" required></label>
    <label class="check"><input type="checkbox" name="enabled" ${camera.enabled === false ? '' : 'checked'}> On</label>
    <button type="button" class="bad drop" aria-label="Remove camera">Remove</button>`;
  return row;
}

async function loadSettings() {
  const tasks = [
    api('/api/cameras').then(data => {
      const rows = $('#camera-rows');
      rows.innerHTML = '';
      data.cameras.forEach(camera => rows.appendChild(cameraRow(camera)));
      if (!data.cameras.length) rows.innerHTML = '<p class="empty">No cameras yet. Add an RTSP stream from a CCTV recorder or IP camera.</p>';
    }),
    api('/api/settings/alerts').then(fillAlerts),
    loadUsers(),
    api('/api/audit?limit=200').then(data => {
      $('#audit-rows').innerHTML = data.entries.map(entry => `<tr><td>${esc(entry.at.replace('T', ' ').slice(0, 19))}</td>
        <td>${esc(entry.user)}</td><td>${esc(entry.action)}</td><td>${esc(entry.target)}</td></tr>`).join('')
        || '<tr><td colspan="4" class="muted">No entries yet.</td></tr>';
    }),
  ];
  const status = state.status || {};
  $('#privacy-info').innerHTML = `<p>Retention: ${status.retention_days ? esc(status.retention_days) + ' days' : 'events are kept until deleted'}${status.keep_confirmed ? '; confirmed events are kept until you delete them' : ''}.</p>
    <p>Privacy mode: ${status.privacy_mode ? 'on — faces are blurred and plates are not stored for riders wearing a helmet' : 'off'}.</p>
    <p>Data stays on this server. Use exports to keep records outside the retention window.</p>`;
  const results = await Promise.allSettled(tasks);
  const failed = results.find(result => result.status === 'rejected');
  if (failed) setText('#cameras-msg', failed.reason.message);
}

async function saveCameras(event) {
  event.preventDefault();
  const cameras = $$('#camera-rows .camera-row').map(row => ({
    id: row.dataset.id || undefined, name: $('[name="name"]', row).value.trim(), site: $('[name="site"]', row).value.trim(),
    url: $('[name="url"]', row).value.trim(), enabled: $('[name="enabled"]', row).checked,
  }));
  try {
    await api('/api/cameras', {method: 'PUT', json: {cameras}});
    setText('#cameras-msg', 'Cameras saved. The detector connects within a few seconds.');
    loadSettings();
  } catch (error) { setText('#cameras-msg', error.message); }
}

function fillAlerts(settings) {
  const form = $('#alerts-form');
  for (const [key, value] of Object.entries(settings)) {
    const input = form.elements[key];
    if (!input) continue;
    if (input.type === 'checkbox') input.checked = Boolean(value); else input.value = value ?? '';
  }
  form.elements.cat_no_helmet.checked = (settings.categories || []).includes('no_helmet');
  form.elements.cat_helmet.checked = (settings.categories || []).includes('helmet');
  form.elements.webhook_token.placeholder = settings.webhook_token_set ? 'saved — leave blank to keep' : 'optional';
  form.elements.smtp_password.placeholder = settings.smtp_password_set ? 'saved — leave blank to keep' : 'optional';
}

async function saveAlerts(event) {
  event.preventDefault();
  const form = $('#alerts-form').elements;
  const payload = {
    enabled: form.enabled.checked, email_enabled: form.email_enabled.checked,
    categories: [form.cat_no_helmet.checked && 'no_helmet', form.cat_helmet.checked && 'helmet'].filter(Boolean),
    min_interval_seconds: Number(form.min_interval_seconds.value || 0), smtp_port: Number(form.smtp_port.value || 587),
  };
  for (const key of ['dashboard_url', 'webhook_url', 'webhook_token', 'smtp_host', 'smtp_user', 'smtp_password', 'smtp_from', 'email_to']) {
    payload[key] = form[key].value.trim();
  }
  try {
    fillAlerts(await api('/api/settings/alerts', {method: 'PUT', json: payload}));
    setText('#alerts-msg', 'Alert settings saved.');
  } catch (error) { setText('#alerts-msg', error.message); }
}

async function testAlert() {
  setText('#alerts-msg', 'Sending test alert…');
  try {
    const data = await api('/api/alerts/test', {method: 'POST'});
    setText('#alerts-msg', `Test alert ${data.result.replace('sent:', 'sent via ')}.`);
  } catch (error) { setText('#alerts-msg', error.message); }
}

async function loadUsers() {
  const data = await api('/api/users');
  $('#user-rows').innerHTML = data.users.map(user => `<tr data-user="${esc(user.username)}"><td>${esc(user.username)}</td>
    <td><select class="role" aria-label="Role for ${esc(user.username)}">${data.roles.map(role => `<option ${role === user.role ? 'selected' : ''}>${role}</option>`).join('')}</select></td>
    <td><button type="button" class="reset">Reset password</button> ${user.username === state.user.username ? '' : '<button type="button" class="bad delete-user">Delete</button>'}</td></tr>`).join('');
}

async function onUserTable(event) {
  const row = event.target.closest('tr');
  if (!row) return;
  const username = row.dataset.user;
  try {
    if (event.type === 'change' && event.target.classList.contains('role')) {
      await api(`/api/users/${encodeURIComponent(username)}`, {method: 'PATCH', json: {role: event.target.value}});
      setText('#users-msg', `Role updated for ${username}.`);
    } else if (event.type === 'click' && event.target.classList.contains('reset')) {
      const password = prompt(`New password for ${username} (at least 10 characters):`);
      if (!password) return;
      await api(`/api/users/${encodeURIComponent(username)}`, {method: 'PATCH', json: {password}});
      setText('#users-msg', `Password reset for ${username}; their sessions were signed out.`);
    } else if (event.type === 'click' && event.target.classList.contains('delete-user')) {
      if (!confirm(`Delete user ${username}?`)) return;
      await api(`/api/users/${encodeURIComponent(username)}`, {method: 'DELETE'});
      setText('#users-msg', `${username} deleted.`);
    } else return;
  } catch (error) { setText('#users-msg', error.message); }
  loadUsers().catch(() => {});
}

async function addUser(event) {
  event.preventDefault();
  const form = event.target;
  try {
    await api('/api/users', {method: 'POST', json: {username: form.username.value.trim(), password: form.password.value, role: form.role.value}});
    setText('#users-msg', `User ${form.username.value.trim()} added.`);
    form.reset();
    loadUsers();
  } catch (error) { setText('#users-msg', error.message); }
}

async function changePassword(event) {
  event.preventDefault();
  const form = event.target;
  try {
    await api('/api/account/password', {method: 'POST', json: {current: form.current.value, password: form.password.value}});
    form.reset();
    setText('#password-msg', 'Password changed. Other sessions were signed out.');
  } catch (error) { setText('#password-msg', error.message); }
}

/* ------------------------------------------------------------------ startup */
let bound = false;
function bindOnce() {
  if (bound) return;
  bound = true;
  $$('.tabs [role="tab"]').forEach(button => button.addEventListener('click', () => selectTab(button.dataset.tab)));
  $('#logout').addEventListener('click', logout);
  $('#theme-toggle').addEventListener('click', cycleTheme);
  $('#help-button').addEventListener('click', () => $('#help').showModal?.());
  $('#who').addEventListener('click', () => { $('#account-panel').hidden = !$('#account-panel').hidden; });
  ['#f-start', '#f-end', '#f-status', '#f-camera'].forEach(id => $(id).addEventListener('change', filtersChanged));
  let searchTimer;
  $('#f-q').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(filtersChanged, 300); });
  $$('.quick [data-range]').forEach(button => button.addEventListener('click', () => setRange(button.dataset.range)));
  $$('.category-filter').forEach(button => button.addEventListener('click', () => {
    state.category = button.dataset.category;
    $$('.category-filter').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    filtersChanged();
  }));
  $('#events').addEventListener('click', onEventsClick);
  $('#events').addEventListener('change', onEventsChange);
  $('#select-all').addEventListener('change', event => {
    $$('#events .card').forEach(card => { if (event.target.checked) state.selected.add(card.dataset.key); else state.selected.delete(card.dataset.key); });
    $$('#events .pick').forEach(box => { box.checked = event.target.checked; });
    updateSelection();
  });
  $$('#bulkbar [data-bulk]').forEach(button => button.addEventListener('click', () => bulkAction(button.dataset.bulk)));
  $('#load-more').addEventListener('click', () => loadEvents({append: true}));
  $('#camera-start').addEventListener('click', startCamera);
  $('#camera-stop').addEventListener('click', () => stopCamera('Camera stopped.'));
  $('#detector-restart').addEventListener('click', restartDetector);
  $('#analyze-video').addEventListener('click', analyzeVideo);
  $('#cancel-video').addEventListener('click', cancelVideo);
  ['#a-start', '#a-end', '#a-camera'].forEach(id => $(id).addEventListener('change', loadAnalytics));
  $('#cameras-form').addEventListener('submit', saveCameras);
  $('#camera-add').addEventListener('click', () => { $('#camera-rows .empty')?.remove(); $('#camera-rows').appendChild(cameraRow()); });
  $('#camera-rows').addEventListener('click', event => { if (event.target.classList.contains('drop')) event.target.closest('.camera-row').remove(); });
  $('#alerts-form').addEventListener('submit', saveAlerts);
  $('#alert-test').addEventListener('click', testAlert);
  $('#user-rows').addEventListener('change', onUserTable);
  $('#user-rows').addEventListener('click', onUserTable);
  $('#user-form').addEventListener('submit', addUser);
  $('#password-form').addEventListener('submit', changePassword);
  document.addEventListener('keydown', onKey);
  window.addEventListener('pagehide', () => stopCamera());
  setInterval(tick, 3000);
}

function tick() {
  if ($('#app').hidden) return;
  state.ticks += 1;
  refreshStatus();
  if (state.tab === 'review' && $('#f-end').value >= today() && state.ticks % 2 === 0) { loadEvents(); loadStats(); }
  if (state.tab === 'live' && state.ticks % 3 === 0) loadCameraStreams();
}

function startApp(session) {
  state.user = session.user;
  state.perms = new Set(session.permissions);
  $('#auth-view').hidden = true;
  $('#app').hidden = false;
  $$('[data-perm]').forEach(el => { el.hidden = !can(el.dataset.perm); });
  setText('#who', session.auth_enabled ? `${session.user.username} · ${session.user.role}` : 'Local mode');
  $('#logout').hidden = !session.auth_enabled;
  const params = new URLSearchParams(location.search);
  const day = /^\d{4}-\d{2}-\d{2}$/.test(params.get('date') || '') ? params.get('date') : null;
  $('#f-end').value = day || today();
  $('#f-start').value = day || today();
  $('#a-end').value = today();
  $('#a-start').value = addDays(today(), -29);
  if (params.get('event')) {
    state.focusEvent = params.get('event');
    state.category = '';
    $('#f-status').value = '';
    $$('.category-filter').forEach(item => item.setAttribute('aria-pressed', String(item.dataset.category === '')));
  }
  bindOnce();
  updateExportLinks();
  selectTab(state.tab);
  loadEvents({force: true});
  loadStats();
  refreshStatus();
}

async function boot() {
  applyTheme();
  let session;
  try { session = await api('/api/session'); } catch (error) {
    $('#auth-view').hidden = false;
    setText('#auth-error', 'Cannot reach the Helmet server. ' + error.message);
    return;
  }
  if (session.setup_required) { showAuth('setup'); return; }
  if (!session.user) { showAuth('login'); return; }
  startApp(session);
}

$('#auth-form').addEventListener('submit', submitAuth);
boot();
