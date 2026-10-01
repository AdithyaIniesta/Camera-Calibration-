#!/usr/bin/env python3
"""
Simple camera viewer + capture button, served from the Jetson.

Same capture pipeline as apriltag_stereo_pose_logger.py (UYVY 1280x720 via
v4l2src -> nvvidconv -> BGR). Open http://<jetson-ip>:8080 from the PC to see
the live camera and click CAPTURE. Each capture is saved on the Jetson
(--outdir) AND downloaded by the PC browser, as a lossless PNG.

    python3 capture_server.py --device /dev/video0
"""
import argparse
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import _opencv_cuda  # noqa: F401  (must import before cv2, as in the logger)
except ImportError:
    pass
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="/dev/video0")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--outdir", default="captures")
parser.add_argument("--jpeg-quality", type=int, default=80, help="live view only")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

PIPELINE = (
    f"v4l2src device={args.device} io-mode=2 ! "
    "video/x-raw,format=UYVY,width=1280,height=720,framerate=60/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=1 max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise SystemExit(f"ERROR: cannot open {args.device}")

latest = None
lock = threading.Lock()


def grab_loop():
    global latest
    while True:
        ok, f = cap.read()
        if ok:
            with lock:
                latest = f
        else:
            time.sleep(0.001)


threading.Thread(target=grab_loop, daemon=True).start()

PAGE = b"""<!doctype html><title>Camera</title>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center">
<img src="/stream" style="max-width:100%;max-height:88vh"><br>
<button onclick="cap()" style="font-size:22px;padding:10px 40px;margin:10px">CAPTURE</button>
<span id="msg"></span>
<script>
async function cap(){
  const r = await fetch('/capture', {method:'POST'});
  if(!r.ok){ document.getElementById('msg').textContent = 'capture failed'; return; }
  const name = r.headers.get('X-Filename');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(await r.blob()); a.download = name; a.click();
  document.getElementById('msg').textContent = 'saved ' + name;
}
</script></body>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        f = None if latest is None else latest.copy()
                    if f is None:
                        time.sleep(0.02)
                        continue
                    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                     + jpg.tobytes() + b"\r\n")
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/capture":
            self.send_error(404)
            return
        with lock:
            f = None if latest is None else latest.copy()
        if f is None:
            self.send_error(503, "no frame yet")
            return
        name = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f.png")
        cv2.imwrite(os.path.join(args.outdir, name), f)
        ok, png = cv2.imencode(".png", f)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("X-Filename", name)
        self.send_header("Content-Length", str(len(png)))
        self.end_headers()
        self.wfile.write(png.tobytes())
        print(f"[CAPTURE] {name}")


print(f"Open http://<jetson-ip>:{args.port}  (saving to {os.path.abspath(args.outdir)})")
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
