#!/usr/bin/env python3
"""
Stereo AprilTag pose logger — SPLIT OUTPUT (one JSON per camera).

Each camera detects and solves poses INDEPENDENTLY in its OWN frame. On M
the current detections are captured. On W (or on quit) TWO separate JSON
files are written:

    <--out>_left.json   : tags seen by the LEFT camera,  poses in LEFT frame.
    <--out>_right.json  : tags seen by the RIGHT camera, poses in RIGHT frame.

Each file is self-contained for the downstream parametric-homography step
that will map corners from one image to the other. Every tag entry
includes:
    corners_raw       (pixel coords in that camera's image)
    R, tvec_mm        (tag → own-camera pose, from solvePnP)
    plane_normal      (n = R · [0,0,1]ᵀ, in own-camera frame)
    plane_distance_mm (d = nᵀ · tvec)
    distance_mm       (‖tvec‖)

Both files also carry the intrinsics + stereo extrinsic so any homography
script can consume one file alone and build

    H = K_dst · ( R_stereo − (T_stereo · nᵀ) / d ) · K_src⁻¹

to map corners from src → dst pixels.

Keys
    M    snapshot current frame — REPLACES the previous snapshot
    W    write both files to <--out>_{left,right}.json
    Q    quit (auto-flushes on exit; M then Q also saves)

CLI
    python3 apriltag_stereo_pose_logger.py \\
        left.json right.json stereo_extrinsic.json --tag-size 60

    (or no args -> file dialogs + tag-size prompt)
"""
import argparse
import json
import threading
import time
from datetime import datetime

import _opencv_cuda  # noqa: F401  (must import before cv2)
import cv2
import numpy as np

from _argpick import parse_or_pick


# ============================================================
# CONFIG
# ============================================================

WINDOW_NAME  = "Stereo AprilTag Pose Logger"
LEFT_DEVICE  = "/dev/video0"
RIGHT_DEVICE = "/dev/video2"

WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GREEN  = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)
RED    = (0, 0, 255)
ORANGE = (0, 165, 255)
BLUE   = (255, 0, 0)


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Stereo AprilTag pose logger (poses in LEFT frame)")
parser.add_argument("left_json")
parser.add_argument("right_json")
parser.add_argument("extrinsic_json")
parser.add_argument("--tag-size", type=float, required=True,
                    help="AprilTag side length in mm (black square)")
parser.add_argument("--tag-family", default="36h11",
                    choices=["16h5", "25h9", "36h10", "36h11"])
parser.add_argument("--left-device",  default=LEFT_DEVICE)
parser.add_argument("--right-device", default=RIGHT_DEVICE)
parser.add_argument("--out", default="apriltag_stereo_pose_log.json")

args = parse_or_pick(
    parser,
    [
        ("left_json",      "Left intrinsic JSON",  [("JSON", "*.json")]),
        ("right_json",     "Right intrinsic JSON", [("JSON", "*.json")]),
        ("extrinsic_json", "Stereo extrinsic JSON", [("JSON", "*.json")]),
    ],
    ask_missing_options=[("tag-size", "AprilTag side length in mm (black square)")],
)
TAG_SIZE_MM = float(args.tag_size)


# ============================================================
# LOAD JSON
# ============================================================

def first_existing(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(keys[0])


def load_K_D(path):
    with open(path) as f:
        d = json.load(f)
    K = np.asarray(first_existing(d, ["camera_matrix", "camera_matrix_left",
                                      "camera_matrix_right", "K"]), dtype=np.float64)
    D = np.asarray(first_existing(d, ["distortion_coefficients",
                                      "distortion_coefficients_left",
                                      "distortion_coefficients_right", "D"]),
                   dtype=np.float64).reshape(-1, 1)
    return K, D


def load_extrinsic(path):
    with open(path) as f:
        d = json.load(f)
    R = np.asarray(first_existing(d, ["rotation_left_to_right",
                                      "R_left_to_right", "R"]), dtype=np.float64)
    T = np.asarray(first_existing(d, ["translation_left_to_right_mm",
                                      "T_left_to_right_mm",
                                      "translation_left_to_right", "T"]),
                   dtype=np.float64).reshape(3, 1)
    if R.shape == (3,):
        R, _ = cv2.Rodrigues(np.deg2rad(R).reshape(3, 1))
    if R.shape == (3, 1):
        R, _ = cv2.Rodrigues(R)
    return R, T


K_L, D_L = load_K_D(args.left_json)
K_R, D_R = load_K_D(args.right_json)
R_stereo, T_stereo = load_extrinsic(args.extrinsic_json)

# Projection matrices used by triangulatePoints (in each camera's UNDISTORTED
# pixel space — we pre-undistort the corner coordinates before triangulation).
P_L = K_L @ np.hstack([np.eye(3),      np.zeros((3, 1))])
P_R = K_R @ np.hstack([R_stereo,       T_stereo])

print(f"K_L=\n{K_L}\nD_L={D_L.ravel()}")
print(f"K_R=\n{K_R}\nD_R={D_R.ravel()}")
print(f"R_stereo=\n{R_stereo}")
print(f"T_stereo(mm)={T_stereo.ravel()}   ‖T‖={float(np.linalg.norm(T_stereo)):.3f} mm")


# ============================================================
# CAMERA CAPTURE
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
            print(f"ERROR: cannot open {self.device}")
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
TAG_OBJ = np.array([
    [-half,  half, 0.0],
    [ half,  half, 0.0],
    [ half, -half, 0.0],
    [-half, -half, 0.0],
], dtype=np.float64)

AXES_3D = np.array([[0, 0, 0],
                    [TAG_SIZE_MM, 0, 0],
                    [0, TAG_SIZE_MM, 0],
                    [0, 0, -TAG_SIZE_MM]], dtype=np.float64)


def find_tags(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids = detect(gray)
    if ids is None:
        return {}
    return {int(i[0]): corners[k].reshape(-1, 2).astype(np.float64)
            for k, i in enumerate(ids)}


def pnp(corners, K, D):
    ok, rvec, tvec = cv2.solvePnP(TAG_OBJ, corners.reshape(-1, 1, 2),
                                  K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


def R_to_euler_xyz(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-8:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.rad2deg([rx, ry, rz]).tolist()


# ============================================================
# DRAWING
# ============================================================

def put_text(img, text, pos, scale=0.6, color=WHITE, thickness=2):
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_tag(img, corners, tag_id, dist_mm, color, rvec=None, tvec=None,
             K=None, D=None):
    pts = corners.reshape(-1, 2).astype(int)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i+1) % 4]),
                 color, 2, cv2.LINE_AA)
    ctr = pts.mean(axis=0).astype(int)
    put_text(img, f"id {tag_id}", (ctr[0]-20, ctr[1]-6),
             scale=0.6, color=YELLOW)
    put_text(img, f"{dist_mm/10:.1f} cm", (ctr[0]-30, ctr[1]+22),
             scale=0.5, color=CYAN)
    if rvec is not None and tvec is not None:
        proj, _ = cv2.projectPoints(AXES_3D, rvec, tvec, K, D)
        o, xa, ya, za = proj.reshape(-1, 2).astype(int)
        cv2.line(img, tuple(o), tuple(xa), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, tuple(o), tuple(ya), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(img, tuple(o), tuple(za), (255, 0, 0), 2, cv2.LINE_AA)


# ============================================================
# LOG
# ============================================================

last_snap_left  = None   # dict or None — latest LEFT-camera snapshot
last_snap_right = None   # dict or None — latest RIGHT-camera snapshot
frame_seq = 0


def _tag_entry(corners, R, t):
    """One tag entry — poses in own-camera frame + the plane fields the
    downstream parametric homography script needs."""
    if R is None:
        return None
    # Plane in own-camera frame: n = R · [0,0,1]^T, d = n^T · t
    n = R[:, 2].reshape(3, 1)
    d = float((n.T @ t)[0, 0])
    T4 = np.eye(4)
    T4[:3, :3] = R
    T4[:3, 3]  = t.ravel()
    return {
        "corners_raw":       corners.tolist(),
        "R":                 R.tolist(),
        "tvec_mm":           t.ravel().tolist(),
        "euler_xyz_deg":     R_to_euler_xyz(R),
        "distance_mm":       float(np.linalg.norm(t)),
        "plane_normal":      n.ravel().tolist(),
        "plane_distance_mm": d,
        "T_4x4":             T4.tolist(),
    }


def build_snapshot(tags, K, D, is_left, shape):
    """Assemble one camera's snapshot dict."""
    global frame_seq
    frame_seq += 1
    entries = {}
    for tid, corners in tags.items():
        R, t = pnp(corners, K, D)
        entries[tid] = _tag_entry(corners, R, t)
    return {
        "frame_index":    frame_seq,
        "timestamp_utc":  datetime.utcnow().isoformat() + "Z",
        "image":          {"width": int(shape[1]), "height": int(shape[0])},
        "tags":           entries,
    }


def _split_out_paths(base):
    base = base[:-5] if base.lower().endswith(".json") else base
    return base + "_left.json", base + "_right.json"


def _payload(snap, is_left):
    return {
        # Which camera this file is for.
        "camera":           "left" if is_left else "right",
        "device":           args.left_device if is_left else args.right_device,
        "pose_frame":       "left_camera" if is_left else "right_camera",
        "convention":       "OpenCV (X right, Y down, Z forward)",

        # This camera's intrinsics (used for solvePnP, and as K_src or K_dst
        # by the homography script — depending on direction).
        "K":                (K_L if is_left else K_R).tolist(),
        "D":                (D_L if is_left else D_R).ravel().tolist(),

        # The OTHER camera's intrinsics + stereo extrinsic — enough for the
        # homography script to build H = K_dst (R - t n^T / d) K_src^-1
        # from either direction using just this one file.
        "other_camera":     "right" if is_left else "left",
        "K_other":          (K_R if is_left else K_L).tolist(),
        "D_other":          (D_R if is_left else D_L).ravel().tolist(),
        "R_left_to_right":  R_stereo.tolist(),
        "T_left_to_right_mm": T_stereo.ravel().tolist(),
        "baseline_mm":      float(np.linalg.norm(T_stereo)),

        # Tag settings.
        "tag_family":       args.tag_family,
        "tag_size_mm":      TAG_SIZE_MM,

        # Source paths for provenance.
        "left_intrinsic":   args.left_json,
        "right_intrinsic":  args.right_json,
        "stereo_extrinsic": args.extrinsic_json,

        # The actual snapshot.
        "snapshot":         snap,
    }


def flush(base_out):
    global last_snap_left, last_snap_right
    if last_snap_left is None and last_snap_right is None:
        print("[WRITE] nothing to flush")
        return
    p_left, p_right = _split_out_paths(base_out)
    if last_snap_left is not None:
        with open(p_left, "w") as f:
            json.dump(_payload(last_snap_left, True), f, indent=2)
        print(f"[WRITE] LEFT  -> {p_left}  ({len(last_snap_left['tags'])} tag(s))")
    if last_snap_right is not None:
        with open(p_right, "w") as f:
            json.dump(_payload(last_snap_right, False), f, indent=2)
        print(f"[WRITE] RIGHT -> {p_right}  ({len(last_snap_right['tags'])} tag(s))")


# ============================================================
# MAIN
# ============================================================

def main():
    cam_L = Camera(args.left_device)
    cam_R = Camera(args.right_device)
    if not cam_L.start():
        return 1
    if not cam_R.start():
        cam_L.stop()
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("\nControls: M=snapshot, W=write, Q/ESC=quit  (no auto-log)\n")

    try:
        while True:
            fl = cam_L.get()
            fr = cam_R.get()
            if fl is None or fr is None:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            tags_L = find_tags(fl)
            tags_R = find_tags(fr)

            disp_L = fl.copy()
            disp_R = fr.copy()

            for tid, cL in tags_L.items():
                R_L_full, t_L_full = pnp(cL, K_L, D_L)
                rvec_L = cv2.Rodrigues(R_L_full)[0] if R_L_full is not None else None
                d = float(np.linalg.norm(t_L_full)) if t_L_full is not None else 0.0
                draw_tag(disp_L, cL, tid, d, GREEN,
                         rvec=rvec_L, tvec=t_L_full, K=K_L, D=D_L)

            for tid, cR in tags_R.items():
                R_R_full, t_R_full = pnp(cR, K_R, D_R)
                rvec_R = cv2.Rodrigues(R_R_full)[0] if R_R_full is not None else None
                d = float(np.linalg.norm(t_R_full)) if t_R_full is not None else 0.0
                draw_tag(disp_R, cR, tid, d, GREEN,
                         rvec=rvec_R, tvec=t_R_full, K=K_R, D=D_R)

            put_text(disp_L, f"LEFT  detected={len(tags_L)}",
                     (20, 35), scale=0.65, color=CYAN)
            put_text(disp_R, f"RIGHT detected={len(tags_R)}",
                     (20, 35), scale=0.65, color=ORANGE)
            saved_bits = []
            if last_snap_left  is not None: saved_bits.append(f"L={len(last_snap_left['tags'])}")
            if last_snap_right is not None: saved_bits.append(f"R={len(last_snap_right['tags'])}")
            put_text(disp_L, "snapshot: " + (", ".join(saved_bits) or "none"),
                     (20, 65), scale=0.55, color=WHITE)
            put_text(disp_L, "M=snap  W=write  Q=quit",
                     (20, disp_L.shape[0]-20), scale=0.55, color=YELLOW)

            combo = np.hstack((disp_L, disp_R))
            cv2.imshow(WINDOW_NAME, combo)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                # Replace the previous snapshot with the current one, PER CAMERA.
                last_snap_left  = build_snapshot(tags_L, K_L, D_L, True,  fl.shape) if tags_L else None
                last_snap_right = build_snapshot(tags_R, K_R, D_R, False, fr.shape) if tags_R else None

                nL = len(last_snap_left["tags"])  if last_snap_left  else 0
                nR = len(last_snap_right["tags"]) if last_snap_right else 0
                print(f"\n[SNAP] LEFT tags={nL}   RIGHT tags={nR}")

                def _print_side(name, snap):
                    if snap is None:
                        return
                    for tid, e in snap["tags"].items():
                        if e is None: continue
                        tv = e["tvec_mm"]; eu = e["euler_xyz_deg"]
                        print(f"  {name} id{tid:>3d}  "
                              f"t=({tv[0]:+8.1f},{tv[1]:+8.1f},{tv[2]:+8.1f}) mm  "
                              f"euler=({eu[0]:+6.1f},{eu[1]:+6.1f},{eu[2]:+6.1f})°  "
                              f"|t|={e['distance_mm']:.1f} mm")

                _print_side("L", last_snap_left)
                _print_side("R", last_snap_right)
            elif key == ord("w"):
                flush(args.out)
    finally:
        flush(args.out)
        cam_L.stop()
        cam_R.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
