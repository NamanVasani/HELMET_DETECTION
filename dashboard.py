"""Private, localhost-only review dashboard for saved detection events."""

from datetime import date
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import queue
import re
import threading
import tempfile
import time
from urllib.parse import parse_qs, unquote, urlparse

from event_store import STORAGE, clean_expired_events, delete_event, event_folder, list_events, update_review

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helmet Events</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#151d31;--muted:#9ba8bd;--line:#2a3550;--accent:#59d6b1;--red:#ff8077}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#f2f5fa;font:15px/1.5 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px 18px 60px}h1{margin:0;font-size:30px}h2{font-size:18px;margin:0 0 10px}.muted{color:var(--muted)}.top{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:22px 0}.panel,.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px}.live img{width:100%;max-height:400px;object-fit:contain;background:#05070d;border-radius:10px}.datebar{display:flex;align-items:center;gap:12px;margin:20px 0;flex-wrap:wrap}input,button,select{font:inherit;border-radius:9px;border:1px solid var(--line);padding:9px 12px;background:#0e1527;color:inherit}button{cursor:pointer}button.ok{background:#145541;border-color:#20765a}button.bad{background:#592a31;border-color:#85414a}.category-buttons{display:flex;gap:8px}.category-filter[aria-pressed="true"]{background:#145541;border-color:#20765a}.stats{display:flex;gap:10px;flex-wrap:wrap}.stat{background:#0e1527;border-radius:10px;padding:10px 14px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}.images{display:grid;grid-template-columns:1fr 1fr;gap:8px}.images figure{margin:0}.images img{width:100%;height:160px;object-fit:contain;background:#070a12;border-radius:8px}.images figcaption{font-size:12px;color:var(--muted);padding-top:3px}.badge{display:inline-block;border:1px solid #59647a;border-radius:999px;padding:2px 8px;font-size:12px}.badge.confirmed{color:var(--accent)}.badge.false_positive{color:var(--red)}.actions{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}.actions input{min-width:110px;flex:1}.notice{font-size:13px;color:var(--muted);margin-top:12px}#status{color:var(--accent)}@media(max-width:760px){.top{grid-template-columns:1fr}main{padding:20px 12px}.images img{height:130px}}
</style></head><body><main>
<h1>Helmet event review</h1><p class="muted">Review helmet and no-helmet captures; remove incorrect detections.</p>
<section class="top"><div class="panel live"><h2>Live camera</h2><img src="/stream" alt="Live camera detections"><div id="camera-status" class="notice" role="status">Checking camera detector…</div><button type="button" onclick="startCamera()">Start / retry camera</button><button type="button" onclick="stopCamera('Camera stopped.')">Stop camera</button><button type="button" onclick="restartDetector()">Restart detector</button><p class="notice">Camera frames are sent to this computer for live detection and local event storage. Captures require three agreeing processed frames.</p><video id="camera-source" autoplay muted playsinline hidden></video><canvas id="camera-frame" hidden></canvas></div>
<div class="panel"><h2>Daily summary</h2><div id="stats" class="stats"></div><div class="notice">Events are stored on this computer for 30 days. This dashboard only listens on this computer.</div></div></section>
<section class="panel" style="margin-bottom:18px"><h2>Analyze a saved video</h2><div class="actions"><input id="video-file" type="file" accept="video/*" aria-label="Choose a video file"><button id="analyze-video" class="ok" onclick="analyzeVideo()">Analyze video</button></div><div id="video-status" class="notice" role="status">The live camera pauses during analysis, then resumes. Uploaded video is deleted after processing; event pictures are saved by date.</div></section>
<div class="datebar"><label for="day">Review date</label><input id="day" type="date"><div class="category-buttons" role="group" aria-label="Filter events by category"><button type="button" class="category-filter" data-category="helmet" aria-pressed="true">Helmet</button><button type="button" class="category-filter" data-category="no_helmet" aria-pressed="false">No helmet</button></div><span id="status"></span></div>
<section id="events" class="grid"></section>
</main><script>
const day=document.querySelector('#day'),categoryButtons=[...document.querySelectorAll('.category-filter')],events=document.querySelector('#events'),stats=document.querySelector('#stats');
let activeCategory='helmet';
const videoFile=document.querySelector('#video-file'),videoStatus=document.querySelector('#video-status');
const cameraStatusText=document.querySelector('#camera-status'),cameraVideo=document.querySelector('#camera-source'),cameraCanvas=document.querySelector('#camera-frame');
let cameraStream=null,sendingFrame=false,lastFrameAt=0,startingCamera=false,uploadingVideo=false;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function load(){try{const res=await fetch('/api/events?date='+encodeURIComponent(day.value));if(!res.ok)throw new Error('Events unavailable');const data=await res.json();
const count=data.events.length,helmet=data.events.filter(x=>(x.category||'no_helmet')==='helmet').length,noHelmet=count-helmet;
stats.innerHTML=`<div class="stat">Events <b>${count}</b></div><div class="stat">Helmet <b>${helmet}</b></div><div class="stat">No helmet <b>${noHelmet}</b></div>`;
if(events.contains(document.activeElement))return;
events.innerHTML=data.events.filter(e=>(e.category||'no_helmet')===activeCategory).map(e=>{const category=e.category||'no_helmet',label=category==='helmet'?'Helmet':'No helmet',files=e.files||{},detection=e.event_detection||e.helmet||e.no_helmet||{};const base=`/media/${encodeURIComponent(e.date)}/${encodeURIComponent(category)}/${encodeURIComponent(e.id)}/`;
return `<article class="card"><div style="display:flex;justify-content:space-between;gap:8px"><b>${esc(new Date(e.timestamp).toLocaleTimeString())}</b><span class="badge ${esc(category)}">${label}</span></div>
<p class="notice">${e.source?.type==='video'?`Video: ${esc(e.source.filename)} · Frame ${esc(e.source.frame)} · ${esc(e.source.position_seconds??'unknown')} seconds · Time above is processing time`:e.source?.type==='camera'?'Camera capture · Time above is processing time':'Source not recorded · Time above is processing time'}<br>${e.rider&&e.bike?'Rider and bike suggested; verify the traffic context.':'Vehicle context not verified; this may be a head-only test capture.'}</p>
<div class="images"><figure><img src="${base}person.jpg" alt="${label} person crop"><figcaption>${label} person crop</figcaption></figure><figure>${files.plate?`<img src="${base}plate.jpg" alt="Suggested number plate">`:'<div class="muted">No plate matched</div>'}<figcaption>Suggested plate · ${esc(e.plate_text||'OCR not read')}</figcaption></figure></div>
<div class="notice">${label} score ${esc(Number(detection.confidence).toFixed(2))} · Rider ${e.rider?`#${esc(e.rider.track_id??'—')}`:'not matched'} · Bike ${e.bike?`#${esc(e.bike.track_id??'—')}`:'not matched'} · Association: ${esc(e.association_status)}</div>
<details><summary>View event context</summary><img style="width:100%;margin-top:8px;border-radius:8px" src="${base}event.jpg" alt="Detection context"></details>
<p class="notice">${category==='helmet'?'Remove this if the helmet detection is incorrect.':'If the rider is wearing a helmet or this is a false detection, remove the event and its saved pictures.'}</p>
<div class="actions"><label>Review status <select name="review-status">${['needs_review','confirmed','false_positive'].map(value=>`<option value="${value}" ${e.status===value?'selected':''}>${value.replaceAll('_',' ')}</option>`).join('')}</select></label><label>Plate text <input name="plate-text" maxlength="16" value="${esc(e.plate_text||'')}" autocomplete="off"></label><button class="ok" data-date="${esc(e.date)}" data-id="${esc(e.id)}" onclick="reviewEvent(this)">Save review</button><button class="bad" data-date="${esc(e.date)}" data-id="${esc(e.id)}" onclick="removeEvent(this.dataset.date,this.dataset.id)">Remove event</button></div><p class="notice">Review: ${esc((e.status||'needs_review').replaceAll('_',' '))}. Confirming is your review decision.</p></article>`}).join('')||'<p class="muted">No events saved for this date.</p>'}catch{document.querySelector('#status').textContent='Could not load events. Retrying automatically.'}}
async function reviewEvent(button){const card=button.closest('article');button.disabled=true;try{const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({date:button.dataset.date,id:button.dataset.id,status:card.querySelector('[name="review-status"]').value,plate_text:card.querySelector('[name="plate-text"]').value})});if(!response.ok)throw new Error('Review rejected');document.querySelector('#status').textContent='Review saved';document.activeElement?.blur();await load()}catch{document.querySelector('#status').textContent='Could not save review. Your edits are kept; retry.'}finally{button.disabled=false}}
async function removeEvent(d,id){if(!confirm('Permanently remove this event and its saved pictures?'))return;try{const r=await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({date:d,id})});if(!r.ok)throw new Error('Delete rejected');document.querySelector('#status').textContent='Removed';document.activeElement?.blur();await load()}catch{document.querySelector('#status').textContent='Could not remove event. Retry.'}}
function analyzeVideo(){if(uploadingVideo)return;const file=videoFile.files[0];if(!file){videoStatus.textContent='Choose a video first.';return}if(file.size>250*1024*1024){videoStatus.textContent='Video must be no larger than 250 MB.';return}if(!/\.(mp4|mov|avi|mkv|m4v)$/i.test(file.name)){videoStatus.textContent='Choose an MP4, MOV, AVI, MKV, or M4V video.';return}uploadingVideo=true;document.querySelector('#analyze-video').disabled=true;videoStatus.textContent='Uploading video…';const request=new XMLHttpRequest();request.open('POST','/api/video');request.timeout=120000;request.setRequestHeader('Content-Type',file.type||'application/octet-stream');request.setRequestHeader('X-File-Name',encodeURIComponent(file.name));const finish=()=>{uploadingVideo=false;document.querySelector('#analyze-video').disabled=false};request.upload.onprogress=event=>{if(event.lengthComputable)videoStatus.textContent='Uploading video: '+Math.round(event.loaded/event.total*100)+'%'};request.onload=()=>{finish();let data={};try{data=JSON.parse(request.responseText)}catch{}videoStatus.textContent=request.status===202?'Video queued. Waiting for analysis.':(data.error||'Could not queue video. Check the detector and retry.')};request.onerror=request.ontimeout=()=>{finish();videoStatus.textContent='Upload interrupted. Check the connection and retry.'};request.send(file)}
function stopCamera(message=''){if(cameraStream)cameraStream.getTracks().forEach(track=>track.stop());cameraStream=null;cameraVideo.srcObject=null;if(message)cameraStatusText.textContent=message}
async function restartDetector(){stopCamera('Restarting detector…');try{const response=await fetch('/api/detector/restart',{method:'POST'});if(!response.ok)throw new Error('Restart rejected');cameraStatusText.textContent='Detector restart requested. Retry camera when it is ready.'}catch{cameraStatusText.textContent='Could not restart detector. Check the connection and retry.'}}
async function sendCameraFrames(now){const stream=cameraStream;if(!stream)return;if(now-lastFrameAt>=120&&!sendingFrame&&cameraVideo.readyState>=2){lastFrameAt=now;sendingFrame=true;const scale=Math.min(1,1280/cameraVideo.videoWidth);cameraCanvas.width=Math.round(cameraVideo.videoWidth*scale);cameraCanvas.height=Math.round(cameraVideo.videoHeight*scale);cameraCanvas.getContext('2d').drawImage(cameraVideo,0,0,cameraCanvas.width,cameraCanvas.height);try{const blob=await new Promise(resolve=>cameraCanvas.toBlob(resolve,'image/jpeg',0.72));if(blob&&cameraStream===stream){const response=await fetch('/api/frame',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob,signal:AbortSignal.timeout(10000)});if(!response.ok)throw new Error('Camera frame rejected')}}catch{if(cameraStream===stream)stopCamera('Camera detector unavailable. It will retry automatically; retry camera when ready.');return}finally{sendingFrame=false}}if(cameraStream===stream)requestAnimationFrame(sendCameraFrames)}
async function startCamera(){
if(cameraStream||startingCamera)return;
startingCamera=true;
try{
  cameraStatusText.textContent='Checking camera detector…';
  let data;
  try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw new Error('Status unavailable');data=await response.json()}
  catch{stopCamera('Cannot connect to the camera detector. Start the helmet app, then retry.');return}
  if(!data.camera_enabled){cameraStatusText.textContent='Saved-event review is running. Start the helmet camera detector, then retry.';return}
  if(!navigator.mediaDevices?.getUserMedia){cameraStatusText.textContent='Camera access is unavailable. Open this site on localhost or HTTPS.';return}
  cameraStatusText.textContent='Requesting browser camera permission…';
  cameraStream=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:1280},height:{ideal:720}}});
  cameraVideo.srcObject=cameraStream;await cameraVideo.play();
  cameraStatusText.textContent='Camera allowed. Starting live detection…';requestAnimationFrame(sendCameraFrames);
}catch(error){
  const messages={NotAllowedError:'Camera permission denied. Allow camera access for this site and for your browser in macOS settings, then retry.',NotFoundError:'No camera was found. Connect a camera, then retry.',NotReadableError:'Camera could not start. Close other apps using it, check macOS camera permission, then retry.'};
  stopCamera(messages[error.name]||'Camera unavailable. Check browser and macOS camera settings, then retry.');
}finally{startingCamera=false}
}
async function cameraStatus(){try{const r=await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!r.ok)throw new Error('Status unavailable');const data=await r.json();if(cameraStream&&!data.camera_enabled)stopCamera(data.status);else if(cameraStream)cameraStatusText.textContent=data.status;else if(!startingCamera&&/^(Camera detector|Restarting detector|Detector restart|Cannot connect to the detector)/.test(cameraStatusText.textContent))cameraStatusText.textContent=data.status+' Use Start / retry camera when ready.';const video=data.video;if(video&&!uploadingVideo){if(video.state==='processing')videoStatus.textContent='Analyzing video: '+video.frames+' frames'+(video.total?' of '+video.total:'')+'. Live camera is paused.';else if(video.state==='complete')videoStatus.textContent='Video analysis complete. Live camera can resume.';else if(video.state==='failed')videoStatus.textContent=video.error;else if(video.state==='queued')videoStatus.textContent='Video queued. Waiting for analysis.'}}catch{stopCamera('Cannot connect to the detector. Camera stopped; retry when the app is available.')}}
window.addEventListener('pagehide',()=>stopCamera());day.value=new Date().toLocaleDateString('en-CA');day.addEventListener('change',load);categoryButtons.forEach(button=>button.addEventListener('click',()=>{activeCategory=button.dataset.category;categoryButtons.forEach(tab=>tab.setAttribute('aria-pressed',String(tab===button)));load()}));load();cameraStatus();startCamera();setInterval(()=>{if(day.value===new Date().toLocaleDateString('en-CA'))load();cameraStatus()},3000);
</script></body></html>"""


class DashboardServer:
    def __init__(self, host="127.0.0.1", port=8003, storage=STORAGE, accept_video=False, accept_frames=False):
        self.storage = Path(storage)
        self.frame_lock = threading.Lock()
        self.frame = None
        self.status = ("Waiting for browser camera permission." if accept_frames
                       else "Saved-event review only. Camera detector is not running.")
        self.accept_video = accept_video
        self.video_queue = queue.Queue(maxsize=1)
        self.frame_queue = queue.Queue(maxsize=1)
        self.accept_frames = accept_frames
        self.restart_detector = threading.Event()
        self.detector_managed = False
        self.video = {"state": "idle", "frames": 0, "total": 0, "error": ""}
        self.cleanup_after = time.monotonic() + 3600
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def parse_request(self):
                if not super().parse_request():
                    return False
                host = self.headers.get("Host", "").lower()
                port = self.server.server_port
                origin = self.headers.get("Origin")
                if host not in {f"127.0.0.1:{port}", f"localhost:{port}"} or (
                    origin is not None and origin != f"http://{host}"
                ):
                    self.send_error(403, "Use this dashboard from its localhost address")
                    return False
                return True

            def log_message(self, *_args):
                pass

            def send_body(self, status, content_type, body):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send_body(200, "text/html; charset=utf-8", PAGE.encode())
                elif parsed.path == "/api/status":
                    with owner.frame_lock:
                        status = owner.status
                        video = dict(owner.video)
                    self.send_body(200, "application/json", json.dumps({
                        "status": status, "camera_enabled": owner.accept_frames,
                        "video_enabled": owner.accept_video,
                        "video": video,
                    }).encode())
                elif parsed.path == "/api/events":
                    day = parse_qs(parsed.query).get("date", [date.today().isoformat()])[0]
                    try:
                        data = {"events": list_events(day, owner.storage)}
                    except ValueError:
                        self.send_body(400, "application/json", b'{"error":"invalid date"}')
                        return
                    self.send_body(200, "application/json", json.dumps(data).encode())
                elif parsed.path == "/stream":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with owner.frame_lock:
                                frame = owner.frame
                            if frame:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                            time.sleep(0.08)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                elif parsed.path.startswith("/media/"):
                    match = re.fullmatch(r"/media/(\d{4}-\d{2}-\d{2})/(helmet|no_helmet)/(event_[\w-]{1,80})/(person|plate|event)\.jpg", parsed.path)
                    if not match:
                        self.send_error(404)
                        return
                    try:
                        folder = event_folder(owner.storage, match.group(1), match.group(3), match.group(2))
                    except ValueError:
                        self.send_error(404)
                        return
                    path = folder / (match.group(4) + ".jpg")
                    if path.is_symlink() or not path.is_file():
                        self.send_error(404)
                        return
                    try:
                        self.send_body(200, "image/jpeg", path.read_bytes())
                    except OSError:
                        self.send_error(404)
                else:
                    self.send_error(404)

            def do_POST(self):
                endpoint = urlparse(self.path).path
                if endpoint == "/api/detector/restart":
                    if not owner.detector_managed:
                        self.send_error(503, "Run hi.py to manage the detector")
                        return
                    owner.restart_detector.set()
                    self.send_body(202, "application/json", b'{"restarting":true}')
                    return
                if endpoint == "/api/frame":
                    if not owner.accept_frames:
                        self.send_error(503)
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if self.headers.get_content_type() != "image/jpeg" or not 0 < length <= 5 * 1024 * 1024:
                            raise ValueError("Invalid camera frame")
                        frame = self.rfile.read(length)
                        if len(frame) != length or not frame.startswith(b"\xff\xd8"):
                            raise ValueError("Invalid JPEG frame")
                        owner.update_input_frame(frame)
                    except (ValueError, OSError) as exc:
                        self.send_body(400, "application/json", json.dumps({"error": str(exc)}).encode())
                        return
                    self.send_body(202, "application/json", b'{"accepted":true}')
                    return
                if endpoint not in {"/api/review", "/api/delete"}:
                    if endpoint != "/api/video":
                        self.send_error(404)
                        return
                    if not owner.accept_video:
                        self.send_error(503)
                        return
                    with owner.frame_lock:
                        busy = owner.video["state"] in {"uploading", "queued", "processing"}
                        if not busy:
                            owner.video = {"state": "uploading", "frames": 0, "total": 0, "error": ""}
                    if busy:
                        self.send_body(409, "application/json", b'{"error":"Wait for the current video to finish"}')
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        filename = unquote(self.headers.get("X-File-Name", ""))
                        suffix = Path(filename).suffix.lower()
                        if not 0 < length <= 250 * 1024 * 1024:
                            raise ValueError("Video must be no larger than 250 MB")
                        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
                            raise ValueError("Unsupported video format")
                        temporary = tempfile.NamedTemporaryFile(prefix="helmet_video_", suffix=suffix, delete=False)
                        try:
                            remaining = length
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise ValueError("Incomplete video upload")
                                temporary.write(chunk)
                                remaining -= len(chunk)
                        finally:
                            temporary.close()
                        with owner.frame_lock:
                            owner.video_queue.put_nowait({"path": temporary.name,
                                "filename": Path(filename.replace("\\", "/")).name[:255],
                                "id": Path(temporary.name).stem})
                            owner.video = {"state": "queued", "frames": 0, "total": 0, "error": ""}
                    except queue.Full:
                        Path(temporary.name).unlink(missing_ok=True)
                        owner.update_video("failed", error="A video is already queued")
                        self.send_error(409, "A video is already queued")
                        return
                    except (ValueError, OSError) as exc:
                        if "temporary" in locals():
                            Path(temporary.name).unlink(missing_ok=True)
                        owner.update_video("failed", error=str(exc))
                        self.send_body(400, "application/json", json.dumps({"error": str(exc)}).encode())
                        return
                    self.send_body(202, "application/json", b'{"queued":true}')
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("invalid request size")
                    data = json.loads(self.rfile.read(length))
                    result = (
                        {"removed": delete_event(data["date"], data["id"], owner.storage)}
                        if endpoint == "/api/delete"
                        else update_review(data["date"], data["id"], data["status"], data.get("plate_text", ""), owner.storage)
                    )
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self.send_body(400, "application/json", json.dumps({"error": str(exc)}).encode())
                    return
                except FileNotFoundError:
                    self.send_error(404)
                    return
                self.send_body(200, "application/json", json.dumps(result).encode())

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.server.service_actions = self.cleanup_expired
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def cleanup_expired(self):
        if time.monotonic() < self.cleanup_after:
            return
        self.cleanup_after = time.monotonic() + 3600
        try:
            clean_expired_events(self.storage)
        except Exception:
            logging.exception("Event cleanup failed; it will retry in one hour")

    def update_frame(self, frame):
        with self.frame_lock:
            self.frame = frame

    def update_input_frame(self, frame):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def next_frame(self, timeout=1):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def update_status(self, status):
        with self.frame_lock:
            self.status = status

    def update_video(self, state, **values):
        with self.frame_lock:
            self.video.update(state=state, **values)

    def next_video(self):
        try:
            return self.video_queue.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run the local helmet event dashboard")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    clean_expired_events()
    server = DashboardServer(port=args.port).start()
    print(f"Event review dashboard: http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.close()


if __name__ == "__main__":
    main()
