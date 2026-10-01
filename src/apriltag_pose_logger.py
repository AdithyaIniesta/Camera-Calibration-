#!/usr/bin/env python3
"""
Live AprilTag pose logger for a single Jetson camera.

For every frame it:
  - grabs UYVY 1280x720@60 through the Jetson GStreamer VIC pipeline,
  - detects every AprilTag in view (chosen family),
  - runs solvePnP per tag → rvec, tvec, R (tag → camera),
  - builds the plane-induced homography H = K [r1 r2 t] (tag plane z=0
    in tag frame → image pixels, distorted with D on save),
  - draws outlines, IDs and axes on the RAW frame,
  - writes TWO JSON files: the poses (--out) and, next to it,
    <--out stem>_angles.json with each tag's corner/centre angles measured
    from the camera's own principal point (cx, cy).

Keys:
  S      snapshot the current frame's detections to the JSON log
  A      auto-log every frame that has ≥1 tag (toggle)
  W      write the accumulated log to --out and clear it
  Q/ESC  quit (also flushes if there is anything unwritten)

CLI:
  python3 apriltag_pose_logger.py intrinsics.json --tag-size 60
  python3 apriltag_pose_logger.py            # file-picker + tag-size prompt

No monitor on the Jetson: add a stream and view it from the PC browser.
  python3 apriltag_pose_logger.py intrinsics.json --tag-size 60 \\
      --stream-port 8080 --headless
  then open http://<jetson-ip>:8080  (SNAP / AUTO / WRITE buttons = S / A / W)
"""

import argparse
import io
import json
import queue
import threading
import time
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import _opencv_cuda  # noqa: F401  (must import before cv2)
import cv2
import numpy as np

from _angles import corner_angles_entry
from _argpick import parse_or_pick


# ============================================================
# CONFIG
# ============================================================

WINDOW_NAME = "AprilTag Pose Logger"

WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)
GREEN   = (0, 255, 0)
YELLOW  = (0, 255, 255)
CYAN    = (255, 255, 0)
RED     = (0, 0, 255)
ORANGE  = (0, 165, 255)


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser(
    description="Detect all AprilTags, compute homography+pose, log to JSON.")
parser.add_argument("intrinsic_json", help="Camera intrinsic JSON")
parser.add_argument("--tag-size", type=float, required=True,
                    help="AprilTag side length in mm (black square)")
parser.add_argument("--tag-family", default="36h11",
                    choices=["16h5", "25h9", "36h10", "36h11"])
parser.add_argument("--device", default="/dev/video0",
                    help="V4L2 device path (default: /dev/video0)")
parser.add_argument("--out", default="apriltag_pose_log.json",
                    help="Output JSON path")
parser.add_argument("--stream-port", type=int, default=0,
                    help="If >0, serve the annotated view as MJPEG on this port "
                         "(open http://<jetson-ip>:PORT; page has Snap/Auto/Write buttons)")
parser.add_argument("--headless", action="store_true",
                    help="No local window / keyboard (use with --stream-port)")

args = parse_or_pick(
    parser,
    [("intrinsic_json", "Camera intrinsic JSON", [("JSON", "*.json")])],
    ask_missing_options=[("tag-size", "AprilTag side length in mm (black square)")],
)
TAG_SIZE_MM = float(args.tag_size)


# ============================================================
# INTRINSICS
# ============================================================

def first_existing(data, keys):
    for k in keys:
        if k in data:
            return data[k]
    return None


with open(args.intrinsic_json, "r") as f:
    _cal = json.load(f)
K = np.asarray(
    first_existing(_cal, ["camera_matrix", "camera_matrix_left",
                          "camera_matrix_right", "K"]),
    dtype=np.float64,
)
D = np.asarray(
    first_existing(_cal, ["distortion_coefficients",
                          "distortion_coefficients_left",
                          "distortion_coefficients_right", "D"]),
    dtype=np.float64,
).reshape(-1, 1)
print(f"[K]\n{K}\n[D]={D.ravel()}")


# ============================================================
# CAPTURE (strict UYVY 1280x720@60)
# ============================================================

class Camera:
    def __init__(self, device):
        self.device = device
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        pipeline = (
            f"v4l2src device={self.device} io-mode=2 ! "
            "video/x-raw,format=UYVY,width=1280,height=720,framerate=60/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        )
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print(f"ERROR: cannot open GStreamer pipeline for {self.device}")
            return False
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.001)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# ============================================================
# APRILTAG
# ============================================================

FAMILY_MAP = {
    "16h5":  cv2.aruco.DICT_APRILTAG_16h5,
    "25h9":  cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}
aruco_dict = cv2.aruco.getPredefinedDictionary(FAMILY_MAP[args.tag_family])
try:
    _params = cv2.aruco.DetectorParameters()
    _det = cv2.aruco.ArucoDetector(aruco_dict, _params)
    def detect(gray):
        c, i, _ = _det.detectMarkers(gray)
        return c, i
except AttributeError:
    _params = cv2.aruco.DetectorParameters_create()
    def detect(gray):
        c, i, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=_params)
        return c, i


half = TAG_SIZE_MM / 2.0
TAG_OBJECT_POINTS = np.array([
    [-half,  half, 0.0],
    [ half,  half, 0.0],
    [ half, -half, 0.0],
    [-half, -half, 0.0],
], dtype=np.float64)

# Axis endpoints for drawing (tag_size long each)
AXIS_LEN = TAG_SIZE_MM
AXES_3D = np.array([
    [0, 0, 0],
    [AXIS_LEN, 0, 0],
    [0, AXIS_LEN, 0],
    [0, 0, -AXIS_LEN],  # -Z out of the tag toward camera
], dtype=np.float64)


def solve_tag(corners_2d):
    ok, rvec, tvec = cv2.solvePnP(
        TAG_OBJECT_POINTS, corners_2d.reshape(-1, 1, 2),
        K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return (rvec, tvec) if ok else (None, None)


def plane_homography(R, tvec):
    """H maps [X, Y, 1] on the tag plane (Z=0) to normalized camera coords,
    then K projects it to pixels. Distortion is NOT baked in (pure pinhole H)."""
    M = np.hstack([R[:, 0:1], R[:, 1:2], tvec])   # 3x3
    H = K @ M
    return H / H[2, 2]


# ============================================================
# DRAWING
# ============================================================

def put_text(img, text, pos, scale=0.6, color=WHITE, thickness=2):
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_tag(img, corners_2d, tag_id, rvec, tvec):
    pts = corners_2d.reshape(-1, 2).astype(int)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]),
                 GREEN, 2, cv2.LINE_AA)
    center = pts.mean(axis=0).astype(int)
    put_text(img, f"id {tag_id}", (center[0] - 20, center[1] - 6),
             scale=0.65, color=YELLOW, thickness=2)
    tz = float(tvec.ravel()[2])
    put_text(img, f"Z={tz/10:.1f} cm", (center[0] - 40, center[1] + 22),
             scale=0.5, color=CYAN, thickness=2)

    # Axes: red=X, green=Y, blue=Z (into the tag)
    proj, _ = cv2.projectPoints(AXES_3D, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    # Edge-on tags / strong distortion can project an axis end to NaN or a huge
    # value; that does not fit OpenCV's int32 pixel type and used to crash the
    # logger. Skip drawing the axes for such a tag.
    if not np.all(np.isfinite(proj)) or np.abs(proj).max() > 1e5:
        return
    o, x, y, z = [(int(p[0]), int(p[1])) for p in proj]
    cv2.line(img, o, x, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, o, y, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, o, z, (255, 0, 0), 2, cv2.LINE_AA)


# ============================================================
# STREAM (annotated view over HTTP, so no monitor is needed)
# ============================================================

_stream_frame = None             # latest annotated frame (BGR)
_stream_lock = threading.Lock()
web_cmds = queue.Queue()         # "s" / "a" / "w" / "q" from the web buttons

_PAGE = b"""<!doctype html><title>AprilTag logger</title>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center">
<img src="/stream" style="max-width:100%;max-height:88vh"><br>
<button onclick="c('s')">SNAP</button> <button onclick="c('a')">AUTO on/off</button>
<button onclick="c('w')">WRITE</button>
<button onclick="cap()" style="background:#2a7">CAPTURE (UYVY + JSON)</button> <span id="msg"></span>
<style>button{font-size:20px;padding:8px 28px;margin:8px}</style>
<script>
function c(k){fetch('/cmd/'+k,{method:'POST'})}
async function cap(){
  const r = await fetch('/capture',{method:'POST'});
  if(!r.ok){document.getElementById('msg').textContent='capture failed';return;}
  const n = r.headers.get('X-Filename');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(await r.blob()); a.download = n; a.click();
  document.getElementById('msg').textContent = 'saved ' + n;
}
</script></body>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_PAGE)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _stream_lock:
                        f = _stream_frame
                    if f is None:
                        time.sleep(0.05)
                        continue
                    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                     + jpg.tobytes() + b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/capture":
            res = capture_zip()
            if res is None:
                self.send_error(503, "no frame yet")
                return
            name, data = res
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("X-Filename", name + ".zip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith("/cmd/") and self.path[5:] in ("s", "a", "w", "q"):
            web_cmds.put(self.path[5:])
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)


def start_stream_server(port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[STREAM] http://<jetson-ip>:{port}")


# ============================================================
# LOG
# ============================================================

log_frames = []      # accumulated snapshots
frame_seq = 0


def snapshot(detections, frame_shape):
    global frame_seq
    frame_seq += 1
    entry = {
        "frame_index": frame_seq,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "image_width": int(frame_shape[1]),
        "image_height": int(frame_shape[0]),
        "tags": detections,
    }
    log_frames.append(entry)
    return entry


def _make_payloads(frames):
    """(poses payload, angles payload) for a list of snapshot entries."""
    payload = {
        "intrinsic_source": args.intrinsic_json,
        "camera_matrix": K.tolist(),
        "distortion_coefficients": D.ravel().tolist(),
        "tag_family": args.tag_family,
        "tag_size_mm": TAG_SIZE_MM,
        "device": args.device,
        "pipeline": "v4l2src UYVY 1280x720@60 -> nvvidconv -> BGR",
        "space": "raw_pixels",
        "num_frames_logged": len(frames),
        "frames": frames,
    }

    # Second payload: corner/centre angles from this camera's own principal
    # point (cx, cy), same convention as the C++ tracker (see _angles.py).
    angles_payload = {
        "intrinsic_source": args.intrinsic_json,
        "device": args.device,
        "angle_convention": (
            "alpha = atan2(u - cx, fx), positive right of the optical axis; "
            "beta = atan2(-(v - cy), fy), positive above it. Measured from "
            "this camera's own principal point (cx, cy), never the image "
            "centre. Raw pixels, no undistortion (same as the C++ tracker)."),
        "principal_point_px": {"cx": float(K[0, 2]), "cy": float(K[1, 2])},
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "K": K.tolist(),
        "tag_family": args.tag_family,
        "tag_size_mm": TAG_SIZE_MM,
        "num_frames_logged": len(frames),
        "frames": [{
            "frame_index": e["frame_index"],
            "timestamp_utc": e["timestamp_utc"],
            "image_width": e["image_width"],
            "image_height": e["image_height"],
            "tags": [dict(id=t["id"],
                          **corner_angles_entry(np.asarray(t["corners_2d_raw"]), K))
                     for t in e["tags"]],
        } for e in frames],
    }
    return payload, angles_payload


def flush(path):
    if not log_frames:
        print("[WRITE] nothing to flush")
        return
    payload, angles_payload = _make_payloads(log_frames)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    stem = path[:-5] if path.lower().endswith(".json") else path
    angles_path = stem + "_angles.json"
    with open(angles_path, "w") as f:
        json.dump(angles_payload, f, indent=2)
    print(f"[WRITE] {len(log_frames)} frame(s) -> {path}  +  {angles_path}")
    log_frames.clear()


# ============================================================
# CAPTURE (web button): UYVY image + pose JSONs, zipped to the PC
# ============================================================

def bgr_to_uyvy(bgr):
    """Packed UYVY (U Y0 V Y1, 2 bytes/pixel) from a BGR frame.
    BT.601 limited range, chroma averaged over each horizontal pixel pair.
    This is a conversion of the frame the poses were computed on, not the
    sensor's raw bytes."""
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    y = 16.0 + 0.257 * r + 0.504 * g + 0.098 * b
    u = 128.0 - 0.148 * r - 0.291 * g + 0.439 * b
    v = 128.0 + 0.439 * r - 0.368 * g - 0.071 * b
    u = (u[:, 0::2] + u[:, 1::2]) / 2.0
    v = (v[:, 0::2] + v[:, 1::2]) / 2.0
    out = np.empty((bgr.shape[0], bgr.shape[1] * 2), dtype=np.uint8)
    out[:, 0::4] = np.clip(u, 0, 255)
    out[:, 1::4] = np.clip(y[:, 0::2], 0, 255)
    out[:, 2::4] = np.clip(v, 0, 255)
    out[:, 3::4] = np.clip(y[:, 1::2], 0, 255)
    return out


_capture_seq = 0
_latest = {"frame": None, "dets": []}    # raw BGR frame + its detections, set by the main loop


def capture_zip():
    """Zip of <name>.uyvy + <name>_poses.json + <name>_angles.json for the
    latest frame, or None if there is no frame yet. Returns (name, bytes)."""
    global _capture_seq
    with _stream_lock:
        frame, dets = _latest["frame"], _latest["dets"]
    if frame is None:
        return None
    _capture_seq += 1
    name = f"capture_{_capture_seq:04d}"
    entry = {
        "frame_index": _capture_seq,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",   # device clock: unreliable
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "tags": dets,
    }
    payload, angles_payload = _make_payloads([entry])
    h, w = frame.shape[:2]
    payload["image_file"] = f"{name}.uyvy"
    payload["image_format"] = (f"raw packed UYVY, {w}x{h}, {w * h * 2} bytes, "
                               "converted from the BGR frame (BT.601 limited range)")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr(f"{name}.uyvy", bgr_to_uyvy(frame).tobytes())
        z.writestr(f"{name}_poses.json", json.dumps(payload, indent=2))
        z.writestr(f"{name}_angles.json", json.dumps(angles_payload, indent=2))
    print(f"[CAPTURE] {name}: {len(dets)} tag(s)")
    return name, buf.getvalue()


# ============================================================
# MAIN
# ============================================================

def main():
    cam = Camera(args.device)
    if not cam.start():
        return 1

    global _stream_frame
    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.stream_port > 0:
        start_stream_server(args.stream_port)
    auto = False
    print("\nControls: S=snapshot, A=auto, W=write, Q/ESC=quit "
          "(also the web buttons when --stream-port is set)\n")

    def poll_key():
        """Key from the local window or the web buttons, as a keycode (-1 = none)."""
        k = -1 if args.headless else (cv2.waitKey(1) & 0xFF)
        if k == 255:
            k = -1
        if k == -1:
            try:
                k = ord(web_cmds.get_nowait())
            except queue.Empty:
                if args.headless:
                    time.sleep(0.001)
        return k

    try:
        while True:
            frame = cam.get()
            if frame is None:
                if poll_key() in (ord("q"), 27):
                    break
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = detect(gray)

            detections = []
            disp = frame.copy()

            if ids is not None and len(ids) > 0:
                for i in range(len(ids)):
                    tag_id = int(ids[i][0])
                    c2d = corners[i].reshape(-1, 2).astype(np.float64)
                    rvec, tvec = solve_tag(c2d)
                    if rvec is None:
                        continue
                    R, _ = cv2.Rodrigues(rvec)
                    H = plane_homography(R, tvec)
                    T4 = np.eye(4)
                    T4[:3, :3] = R
                    T4[:3, 3] = tvec.ravel()

                    detections.append({
                        "id": tag_id,
                        "corners_2d_raw": c2d.tolist(),
                        "rvec": rvec.ravel().tolist(),
                        "tvec_mm": tvec.ravel().tolist(),
                        "R": R.tolist(),
                        "T_tag_to_camera_4x4": T4.tolist(),
                        "homography_tag_to_image": H.tolist(),
                        "distance_mm": float(np.linalg.norm(tvec)),
                    })
                    draw_tag(disp, c2d, tag_id, rvec, tvec)

            # HUD
            put_text(disp, f"tags: {len(detections)}",
                     (20, 35), scale=0.7, color=CYAN)
            put_text(disp, f"logged frames: {len(log_frames)}",
                     (20, 65), scale=0.55, color=WHITE)
            put_text(disp, f"AUTO: {'ON' if auto else 'OFF'}",
                     (20, 92), scale=0.55, color=GREEN if auto else ORANGE)
            put_text(disp, "S snap  A auto  W write  Q quit",
                     (20, disp.shape[0] - 20), scale=0.55, color=YELLOW)

            if not args.headless:
                cv2.imshow(WINDOW_NAME, disp)
            if args.stream_port > 0:
                with _stream_lock:
                    _stream_frame = disp
                    _latest["frame"] = frame        # raw frame (no overlays) + its detections
                    _latest["dets"] = detections

            if auto and detections:
                snapshot(detections, frame.shape)

            key = poll_key()
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                if detections:
                    e = snapshot(detections, frame.shape)
                    print(f"[SNAP] frame {e['frame_index']}: "
                          f"{len(e['tags'])} tag(s) ({[t['id'] for t in e['tags']]})")
                else:
                    print("[SNAP] no tags")
            elif key == ord("a"):
                auto = not auto
                print(f"[AUTO] {'ON' if auto else 'OFF'}")
            elif key == ord("w"):
                flush(args.out)
    finally:
        flush(args.out)   # write anything unsaved
        cam.stop()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
