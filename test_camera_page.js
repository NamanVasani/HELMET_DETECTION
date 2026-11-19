// Run: node test_camera_page.js. Uses fake camera/HTTP responses; never opens hardware.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function main() {
  const elements = new Map();
  let enabled = false, denied = false, cameraRequests = 0, framesScheduled = 0, stopped = 0;
  let fail = false, requests = [], upload;
  let testEvents = [];
  const stream = {getTracks: () => [{stop: () => stopped++}]};
  const context = vm.createContext({
    document: {
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, {
          textContent: '', value: '', addEventListener() {}, play: async () => {}, contains: () => false,
        });
        return elements.get(selector);
      },
      querySelectorAll: () => [],
    },
    window: {addEventListener() {}},
    navigator: {mediaDevices: {async getUserMedia() {
      cameraRequests++;
      if (denied) throw Object.assign(new Error('denied'), {name: 'NotAllowedError'});
      return stream;
    }}},
    fetch: async (url, options) => {
      requests.push({url, options});
      if (fail) throw new Error('offline');
      return {ok: true, json: async () => url.startsWith('/api/events')
        ? {events: testEvents} : {status: 'Detector ready', camera_enabled: enabled}};
    },
    AbortSignal,
    XMLHttpRequest: class {
      constructor() {upload = this; this.upload = {};}
      open() {} setRequestHeader() {} send(file) {this.file = file;}
    },
    requestAnimationFrame: () => framesScheduled++, setInterval() {}, setTimeout() {},
  });
  const page = fs.readFileSync(path.join(__dirname, 'dashboard.py'), 'utf8');
  vm.runInContext(page.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  await new Promise(resolve => setImmediate(resolve));
  const status = elements.get('#camera-status');
  assert.equal(cameraRequests, 0, 'Review mode must not request camera access');
  assert.match(status.textContent, /Saved-event review/);

  testEvents = [{date: '2026-10-20', id: 'event_1', timestamp: '2026-10-20T10:00:00Z',
    category: 'helmet', event_detection: {confidence: 0.9},
    source: {type: 'video', filename: '<junction>.mp4', frame: 24, position_seconds: 2.4}}];
  await context.load();
  assert.match(elements.get('#events').innerHTML, /&lt;junction&gt;\.mp4/);
  assert.match(elements.get('#events').innerHTML, /Frame 24/);
  assert.match(elements.get('#events').innerHTML, /Vehicle context not verified/);
  delete testEvents[0].source;
  await context.load();
  assert.match(elements.get('#events').innerHTML, /Source not recorded/);
  testEvents = [];

  enabled = true;
  denied = true;
  await context.startCamera();
  assert.match(status.textContent, /Camera permission denied/);
  denied = false;
  await Promise.all([context.startCamera(), context.startCamera()]);
  assert.equal(cameraRequests, 2, 'Retry should make one request after the denied request');
  assert.equal(framesScheduled, 1);
  assert.equal(elements.get('#camera-source').srcObject, stream);
  context.stopCamera();
  assert.equal(stopped, 1);
  assert.equal(elements.get('#camera-source').srcObject, null);
  const card = {querySelector: selector => ({value: selector.includes('review-status') ? 'confirmed' : 'GJ01AB1234'})};
  const button = {dataset: {date: '2026-10-13', id: 'event_1'}, closest: () => card};
  await context.reviewEvent(button);
  assert.equal(button.disabled, false);
  const review = requests.find(x => x.url === '/api/review');
  assert.equal(JSON.parse(review.options.body).plate_text, 'GJ01AB1234');
  fail = true;
  await context.reviewEvent(button);
  assert.match(elements.get('#status').textContent, /edits are kept/);
  fail = false;
  const file = {name: 'sample.mp4', size: 1000, type: 'video/mp4'};
  elements.get('#video-file').files = [file];
  context.analyzeVideo();
  assert.equal(upload.file, file);
  upload.upload.onprogress({lengthComputable: true, loaded: 500, total: 1000});
  assert.match(elements.get('#video-status').textContent, /50%/);
  upload.onerror();
  assert.equal(elements.get('#analyze-video').disabled, false);
  assert.match(elements.get('#video-status').textContent, /interrupted/);
  context.analyzeVideo();
  upload.status = 202; upload.responseText = '{}'; upload.onload();
  assert.match(elements.get('#video-status').textContent, /queued/);
  await context.startCamera();
  enabled = false;
  await context.cameraStatus();
  assert.equal(elements.get('#camera-source').srcObject, null, 'Detector failure stops camera');
  fail = true;
  await context.cameraStatus();
  assert.match(status.textContent, /Cannot connect/);
  fail = false; enabled = true;
  await context.cameraStatus();
  assert.match(status.textContent, /Start \/ retry camera/);
  console.log('Camera and review checks passed: permissions, retries, review errors, video progress, detector failure.');
}

main().catch(error => {console.error(error); process.exitCode = 1;});
