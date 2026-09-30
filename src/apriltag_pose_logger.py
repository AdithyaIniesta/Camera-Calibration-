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
"""

import argparse
import json
import threading
import time
from datetime import datetime

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
    tz = float(tvec[2])
    put_text(img, f"Z={tz/10:.1f} cm", (center[0] - 40, center[1] + 22),
             scale=0.5, color=CYAN, thickness=2)

    # Axes: red=X, green=Y, blue=Z (into the tag)
    proj, _ = cv2.projectPoints(AXES_3D, rvec, tvec, K, D)
    o, x, y, z = proj.reshape(-1, 2).astype(int)
    cv2.line(img, tuple(o), tuple(x), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, tuple(o), tuple(y), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, tuple(o), tuple(z), (255, 0, 0), 2, cv2.LINE_AA)


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


def flush(path):
    if not log_frames:
        print("[WRITE] nothing to flush")
        return
    payload = {
        "intrinsic_source": args.intrinsic_json,
        "camera_matrix": K.tolist(),
        "distortion_coefficients": D.ravel().tolist(),
        "tag_family": args.tag_family,
        "tag_size_mm": TAG_SIZE_MM,
        "device": args.device,
        "pipeline": "v4l2src UYVY 1280x720@60 -> nvvidconv -> BGR",
        "space": "raw_pixels",
        "num_frames_logged": len(log_frames),
        "frames": log_frames,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    # Second file: corner/centre angles from this camera's own principal
    # point (cx, cy), same convention as the C++ tracker (see _angles.py).
    stem = path[:-5] if path.lower().endswith(".json") else path
    angles_path = stem + "_angles.json"
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
        "num_frames_logged": len(log_frames),
        "frames": [{
            "frame_index": e["frame_index"],
            "timestamp_utc": e["timestamp_utc"],
            "image_width": e["image_width"],
            "image_height": e["image_height"],
            "tags": [dict(id=t["id"],
                          **corner_angles_entry(np.asarray(t["corners_2d_raw"]), K))
                     for t in e["tags"]],
        } for e in log_frames],
    }
    with open(angles_path, "w") as f:
        json.dump(angles_payload, f, indent=2)
    print(f"[WRITE] {len(log_frames)} frame(s) -> {path}  +  {angles_path}")
    log_frames.clear()


# ============================================================
# MAIN
# ============================================================

def main():
    cam = Camera(args.device)
    if not cam.start():
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    auto = False
    print("\nControls: S=snapshot, A=auto, W=write, Q/ESC=quit\n")

    try:
        while True:
            frame = cam.get()
            if frame is None:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
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

            cv2.imshow(WINDOW_NAME, disp)

            if auto and detections:
                snapshot(detections, frame.shape)

            key = cv2.waitKey(1) & 0xFF
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
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
