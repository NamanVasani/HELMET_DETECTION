// Run: node test_camera_page.js (after `npm install`). Loads static/index.html + static/app.js in jsdom with
// fake HTTP and camera APIs; never opens hardware or a network connection.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');

const html = fs.readFileSync(path.join(__dirname, 'static/index.html'), 'utf8').replace(/<script[^>]*><\/script>/, '');
const script = fs.readFileSync(path.join(__dirname, 'static/app.js'), 'utf8');
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
async function settle() { for (let i = 0; i < 8; i++) await tick(); }

function makePage({session, status, events = [], role = 'admin'}) {
  const dom = new JSDOM(html, {runScripts: 'dangerously', url: 'http://127.0.0.1:8003/', pretendToBeVisual: true});
  const {window} = dom;
  const env = {requests: [], fail: false, status, events, session, cameraRequests: 0, denied: false, stopped: 0,
    framesScheduled: 0, upload: null, bulk: null};
  const permissions = {admin: ['admin', 'camera', 'delete', 'export', 'review', 'video', 'view'],
    reviewer: ['camera', 'delete', 'export', 'review', 'video', 'view'], viewer: ['view']}[role];
  env.session = session || {auth_enabled: true, setup_required: false, user: {username: 'ana', role}, permissions};
  const stream = {getTracks: () => [{stop: () => env.stopped++}]};
  window.fetch = async (url, options = {}) => {
    env.requests.push({url, options});
    if (env.fail) throw new Error('offline');
    let body = {};
    let code = 200;
    if (url.startsWith('/api/session')) body = env.session;
    else if (url.startsWith('/api/setup') || url.startsWith('/api/login')) {
      env.session = {auth_enabled: true, setup_required: false, user: {username: 'ana', role: 'admin'}, permissions};
      body = {user: env.session.user};
    } else if (url.startsWith('/api/events')) body = {events: env.events, total: env.events.length};
    else if (url.startsWith('/api/stats')) body = {totals: {events: 3, helmet: 1, no_helmet: 2, needs_review: 2, confirmed: 1, false_positive: 0, false_positive_rate: 0},
      days: [{date: '2026-11-01', helmet: 1, no_helmet: 2, events: 3}], cameras: ['Gate 1']};
    else if (url.startsWith('/api/status')) body = env.status;
    else if (url.startsWith('/api/bulk')) { env.bulk = JSON.parse(options.body); body = {results: env.bulk.items.map(i => ({...i, ok: true}))}; }
    else if (url.startsWith('/api/cameras')) {
      if (options.method === 'PUT') env.savedCameras = JSON.parse(options.body).cameras;
      body = {cameras: env.cameras || []};
    }
    else if (url.startsWith('/api/offenders')) body = {offenders: [{plate_text: 'GJ01AB1234', events: 3, confirmed: 1,
      cameras: ['Gate 1'], last_seen: '2026-11-19T10:00:00+05:30', latest: {date: '2026-11-19', id: 'event_9'}}]};
    else if (url.startsWith('/api/review') && env.reviewFails) { code = 500; body = {error: 'disk full'}; }
    else if (url.startsWith('/api/review') && env.removeOnReview) {
      const reviewed = JSON.parse(options.body);
      env.events = env.events.filter(item => item.id !== reviewed.id);  // "needs review" filter drops it
    }
    return {ok: code < 400, status: code, json: async () => body};
  };
  window.navigator.mediaDevices = {async getUserMedia() {
    env.cameraRequests++;
    if (env.denied) throw Object.assign(new Error('denied'), {name: 'NotAllowedError'});
    return stream;
  }};
  window.HTMLMediaElement.prototype.play = async function play() {};
  window.requestAnimationFrame = () => env.framesScheduled++;
  window.setInterval = () => 0;
  window.confirm = () => true;
  window.AbortSignal.timeout = () => undefined;
  window.XMLHttpRequest = class {
    constructor() { env.upload = this; this.upload = {}; }
    open() {} setRequestHeader() {} send(file) { this.file = file; } abort() { this.onabort?.(); }
  };
  const element = window.document.createElement('script');
  element.textContent = script;
  window.document.body.appendChild(element);
  return {window, document: window.document, env, run: code => window.eval(code)};
}

const baseStatus = {status: 'Detector ready', camera_enabled: false, video: {state: 'idle'}, metrics: {}, sources: [],
  retention_days: 30, keep_confirmed: true, privacy_mode: true};

async function testSetupFlow() {
  const page = makePage({session: {auth_enabled: true, setup_required: true, user: null, permissions: []}, status: baseStatus});
  await settle();
  const {document} = page;
  assert.equal(document.querySelector('#app').hidden, true);
  assert.match(document.querySelector('#auth-title').textContent, /Create the admin/);
  document.querySelector('#auth-user').value = 'ana';
  document.querySelector('#auth-pass').value = 'long password 1';
  document.querySelector('#auth-confirm').value = 'different';
  document.querySelector('#auth-form').dispatchEvent(new page.window.Event('submit', {cancelable: true}));
  await settle();
  assert.match(document.querySelector('#auth-error').textContent, /do not match/);
  document.querySelector('#auth-confirm').value = 'long password 1';
  document.querySelector('#auth-form').dispatchEvent(new page.window.Event('submit', {cancelable: true}));
  await settle();
  assert.ok(page.env.requests.some(r => r.url === '/api/setup'));
  assert.equal(document.querySelector('#app').hidden, false, 'App shows after setup');
}

async function testReviewRenderingAndFilters() {
  const events = [{date: '2026-10-20', id: 'event_1', timestamp: '2026-10-20T10:00:00Z', category: 'no_helmet', status: 'needs_review',
    event_detection: {confidence: 0.9}, source: {type: 'video', filename: '<junction>.mp4', frame: 24, position_seconds: 2.4}},
  {date: '2026-10-20', id: 'event_2', timestamp: '2026-10-20T10:01:00Z', category: 'no_helmet', status: 'needs_review',
    event_detection: {confidence: 0.8}, files: {plate: 'plate.jpg'}, plate_text: 'GJ01AB1234', plate_format_valid: true}];
  const page = makePage({status: baseStatus, events});
  await settle();
  const {document, env} = page;
  const html = document.querySelector('#events').innerHTML;
  assert.match(html, /&lt;junction&gt;\.mp4/);
  assert.doesNotMatch(html, /<junction>/);
  assert.match(html, /Frame 24/);
  assert.match(html, /Vehicle context not verified/);
  assert.match(html, /Source not recorded/);
  assert.match(html, /✓ format/);
  const first = env.requests.find(r => r.url.startsWith('/api/events'));
  assert.match(first.url, /category=no_helmet/, 'Violations are shown first');
  assert.match(first.url, /status=needs_review/, 'Unreviewed events are shown first');
  assert.equal(env.cameraRequests, 0, 'Camera is opt-in; loading the page must not request it');
  assert.match(document.querySelector('#export-csv').href, /\/api\/export\.csv\?/);
  assert.match(document.querySelector('#stats').textContent, /No helmet\s*2/);

  // Polling with unchanged data must not rebuild the cards (keeps focus and avoids image reloads).
  document.querySelector('#events .card').dataset.marker = 'kept';
  await page.run('loadEvents()');
  await settle();
  assert.equal(document.querySelector('#events .card').dataset.marker, 'kept');

  // Lightbox.
  document.querySelector('#events .zoom').click();
  assert.ok(document.querySelector('#lightbox').hasAttribute('open'));
  assert.match(document.querySelector('#lightbox-img').src, /person\.jpg$/);
  // Zoom viewer: buttons, wheel, limits and reset.
  const zoomImg = document.querySelector('#lightbox-img');
  assert.equal(document.querySelector('#zoom-level').textContent, '100%');
  document.querySelector('#zoom-in').click();
  assert.equal(document.querySelector('#zoom-level').textContent, '150%');
  assert.match(zoomImg.style.transform, /scale\(1\.5\)/);
  const stage = document.querySelector('#lightbox-stage');
  for (let i = 0; i < 20; i++) stage.dispatchEvent(new page.window.WheelEvent('wheel', {deltaY: -100, bubbles: true, cancelable: true}));
  assert.equal(document.querySelector('#zoom-level').textContent, '500%', 'zoom is capped at 5x');
  document.querySelector('#zoom-out').click();
  assert.equal(document.querySelector('#zoom-level').textContent, '333%');
  document.querySelector('#zoom-reset').click();
  assert.equal(document.querySelector('#zoom-level').textContent, '100%');
  assert.match(zoomImg.style.transform, /translate\(0px, 0px\) scale\(1\)/);

  // Review with plate text and failure feedback.
  const card = document.querySelectorAll('#events .card')[1];
  card.querySelector('[name="plate-text"]').value = 'GJ05XY9999';
  card.querySelector('.save').click();
  await settle();
  const review = env.requests.filter(r => r.url === '/api/review').pop();
  assert.equal(JSON.parse(review.options.body).plate_text, 'GJ05XY9999');
  env.reviewFails = true;
  document.querySelectorAll('#events .card')[1].querySelector('.save').click();
  await settle();
  assert.match(document.querySelector('#status').textContent, /edits are kept/);
  env.reviewFails = false;

  // Keyboard shortcut: focus a card and press c to confirm.
  const target = document.querySelectorAll('#events .card')[0];
  target.focus();
  target.dispatchEvent(new page.window.KeyboardEvent('keydown', {key: 'c', bubbles: true}));
  await settle();
  const quick = env.requests.filter(r => r.url === '/api/review').pop();
  assert.equal(JSON.parse(quick.options.body).status, 'confirmed');
  assert.equal(JSON.parse(quick.options.body).id, 'event_1');

  // Bulk selection.
  const pick = document.querySelector('#events .pick');
  pick.checked = true;
  pick.dispatchEvent(new page.window.Event('change', {bubbles: true}));
  assert.match(document.querySelector('#selected-count').textContent, /1 selected/);
  document.querySelector('#bulkbar [data-bulk="false_positive"]').click();
  await settle();
  assert.deepEqual(env.bulk, {action: 'review', status: 'false_positive', items: [{date: '2026-10-20', id: 'event_1'}]});
}

async function testCameraAndVideo() {
  const page = makePage({status: {...baseStatus}});
  await settle();
  const {document, env, run} = page;
  await run('startCamera()');
  assert.match(document.querySelector('#camera-status').textContent, /Saved-event review/);
  assert.equal(env.cameraRequests, 0, 'Review mode must not request camera access');

  env.status.camera_enabled = true;
  env.denied = true;
  await run('startCamera()');
  assert.match(document.querySelector('#camera-status').textContent, /Camera permission denied/);
  env.denied = false;
  await Promise.all([run('startCamera()'), run('startCamera()')]);
  assert.equal(env.cameraRequests, 2, 'Retry should make one request after the denied request');
  assert.equal(env.framesScheduled, 1);
  run('stopCamera()');
  assert.equal(env.stopped, 1);

  const file = {name: 'sample.mp4', size: 1000, type: 'video/mp4'};
  Object.defineProperty(document.querySelector('#video-file'), 'files', {value: [file]});
  run('analyzeVideo()');
  assert.equal(env.upload.file, file);
  env.upload.upload.onprogress({lengthComputable: true, loaded: 500, total: 1000});
  assert.match(document.querySelector('#video-status').textContent, /50%/);
  assert.equal(document.querySelector('#video-progress').value, 50);
  assert.equal(document.querySelector('#cancel-video').hidden, false);
  env.upload.onerror();
  assert.equal(document.querySelector('#analyze-video').disabled, false);
  assert.match(document.querySelector('#video-status').textContent, /interrupted/);
  run('analyzeVideo()');
  env.upload.status = 202; env.upload.responseText = '{}'; env.upload.onload();
  assert.match(document.querySelector('#video-status').textContent, /queued/);
  env.status.video = {state: 'processing', frames: 30, total: 60};
  await run('refreshStatus()');
  assert.equal(document.querySelector('#video-progress').value, 50);

  await run('startCamera()');
  env.status.camera_enabled = false;
  await run('refreshStatus()');
  assert.equal(document.querySelector('#camera-source').srcObject, null, 'Detector failure stops camera');
  env.status.camera_enabled = true;
  await run('startCamera()');
  env.fail = true;
  await run('refreshStatus()');
  assert.match(document.querySelector('#camera-status').textContent, /Cannot connect/);
  assert.equal(document.querySelector('#detector-pill').textContent, 'Offline');
  env.fail = false;

  // Live stream is only attached while the Live tab is visible.
  env.status.sources = ['live'];
  await run('refreshStatus()');
  assert.equal(document.querySelector('#live-stream').getAttribute('src'), null);
  run("selectTab('live')");
  await settle();
  assert.equal(document.querySelector('#live-stream').getAttribute('src'), '/stream?source=live');
  run("selectTab('review')");
  assert.equal(document.querySelector('#live-stream').getAttribute('src'), null);
}

async function testViewerRoleAndAnalytics() {
  const events = [{date: '2026-10-20', id: 'event_1', timestamp: '2026-10-20T10:00:00Z', category: 'no_helmet', event_detection: {confidence: 0.9}}];
  const page = makePage({status: baseStatus, events, role: 'viewer'});
  await settle();
  const {document, run} = page;
  assert.equal(document.querySelector('#events .save'), null, 'Viewers get no review controls');
  assert.equal(document.querySelector('[data-tab="settings"]').hidden, true);
  assert.equal(document.querySelector('#bulkbar').hidden, true);
  run("selectTab('settings')");
  assert.equal(document.querySelector('#tab-settings').hidden, true, 'Viewers cannot open settings');
  run("selectTab('analytics')");
  await settle();
  assert.ok(document.querySelector('#a-chart svg'), 'Chart renders');
  assert.ok(document.querySelectorAll('#a-chart rect').length >= 2);
  // Repeat plates table, and clicking a plate opens Review filtered to it.
  assert.match(document.querySelector('#offender-rows').textContent, /GJ01AB1234\s*3\s*1\s*Gate 1/);
  document.querySelector('#offender-rows .plate-link').click();
  await settle();
  assert.equal(document.querySelector('#tab-review').hidden, false);
  assert.equal(document.querySelector('#f-q').value, 'GJ01AB1234');
  const filtered = page.env.requests.filter(r => r.url.startsWith('/api/events')).pop();
  assert.match(filtered.url, /q=GJ01AB1234/);
  assert.match(filtered.url, /category=no_helmet/);
}

async function testZoneEditor() {
  const page = makePage({status: baseStatus});
  page.env.cameras = [{id: 'c1', name: 'Gate 1', site: '', url: 'rtsp://***:***@10.0.0.5/s', enabled: true, zone: null}];
  await settle();
  const {document, env, window} = page;
  page.run("selectTab('settings')");
  await settle();
  const row = document.querySelector('#camera-rows .camera-row');
  assert.match(row.querySelector('.zone-info').textContent, /whole picture/);
  row.querySelector('.zone-edit').click();
  assert.ok(document.querySelector('#zone-dialog').hasAttribute('open'));
  assert.match(document.querySelector('#zone-img').src, /\/api\/cameras\/c1\/snapshot/);
  const svg = document.querySelector('#zone-svg');
  svg.getBoundingClientRect = () => ({left: 0, top: 0, width: 200, height: 100});
  for (const [x, y] of [[20, 50], [180, 50], [100, 100]]) {
    svg.dispatchEvent(new window.MouseEvent('click', {clientX: x, clientY: y, bubbles: true}));
  }
  assert.equal(document.querySelectorAll('#zone-svg circle').length, 3);
  assert.ok(document.querySelector('#zone-svg polygon'));
  document.querySelector('#zone-done').click();
  assert.match(row.querySelector('.zone-info').textContent, /3 points/);
  document.querySelector('#cameras-form').dispatchEvent(new window.Event('submit', {cancelable: true}));
  await settle();
  assert.deepEqual(env.savedCameras[0].zone, [[0.1, 0.5], [0.9, 0.5], [0.5, 1]]);
}

async function testQuickReview() {
  const make = id => ({date: '2026-11-19', id, timestamp: '2026-11-19T10:00:00Z', category: 'no_helmet',
    status: 'needs_review', event_detection: {confidence: 0.9}});
  const page = makePage({status: baseStatus, events: [make('event_1'), make('event_2'), make('event_3')]});
  page.env.removeOnReview = true;
  await settle();
  const {document, env, window} = page;
  document.querySelector('#quick-toggle').click();
  await settle();
  assert.equal(document.querySelectorAll('#events .card').length, 1, 'one event at a time');
  assert.match(document.querySelector('.quick-card').textContent, /1 of 3/);
  document.querySelector('.quick-act[data-status="confirmed"]').click();
  await settle();
  const first = JSON.parse(env.requests.filter(r => r.url === '/api/review').pop().options.body);
  assert.deepEqual([first.id, first.status], ['event_1', 'confirmed']);
  assert.match(document.querySelector('.quick-card').textContent, /1 of 2/, 'next event shown');
  assert.equal(document.querySelector('.quick-card').dataset.key, '2026-11-19/event_2');
  document.querySelector('.quick-skip').click();
  assert.equal(document.querySelector('.quick-card').dataset.key, '2026-11-19/event_3', 'skip moves on without saving');
  document.body.dispatchEvent(new window.KeyboardEvent('keydown', {key: 'f', bubbles: true}));
  await settle();
  const second = JSON.parse(env.requests.filter(r => r.url === '/api/review').pop().options.body);
  assert.deepEqual([second.id, second.status], ['event_3', 'false_positive']);
  env.events = [];
  await page.run('loadEvents({force: true})');
  await settle();
  assert.match(document.querySelector('#events').textContent, /All done/);
  document.querySelector('#quick-toggle').click();
  assert.match(document.querySelector('#quick-toggle').textContent, /^Quick review$/);
}

(async () => {
  await testQuickReview();
  await testZoneEditor();
  await testSetupFlow();
  await testReviewRenderingAndFilters();
  await testCameraAndVideo();
  await testViewerRoleAndAnalytics();
  console.log('Dashboard browser checks passed: setup/login, escaping, filters, no-rerender polling, lightbox, review, ' +
    'keyboard, bulk, camera opt-in/permissions/retry, video progress/cancel, detector failure, stream gating, roles, chart.');
})().catch(error => { console.error(error); process.exitCode = 1; });
