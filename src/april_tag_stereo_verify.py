#!/usr/bin/env python3
"""
AprilTag stereo verification using refined extrinsics.

Detects an AprilTag in both cameras, estimates its pose, then builds a
plane-induced homography STRICTLY from camera parameters:

    H = K_right @ (R + t n^T / d) @ inv(K_left)

where (n, d) come from the tag plane expressed in the left camera frame
(from solvePnP), and (R, t) are the stereo extrinsics.

No point-correspondence homography is used for the mapping.

Usage:
  python3 april_tag_stereo_verify.py left.json right.json refined_extrinsics.json \\
      --tag-size 50 --tag-family 36h11

Keys:
  S          save current metrics to JSON
  Q / ESC    quit
"""

import argparse
import json
import threading
import time

import _opencv_cuda  # noqa: F401  (must import before cv2)
import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

LEFT_DEVICE = "/dev/video1"
RIGHT_DEVICE = "/dev/video3"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 60

WINDOW_NAME = "AprilTag Stereo Verify (H from camera params)"
PANEL_WIDTH = 560
MAX_DISPLAY_WIDTH = 1920
MAX_DISPLAY_HEIGHT = 1080

WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
BLUE = (255, 0, 0)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)
DARK = (15, 15, 15)


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser(description="AprilTag stereo pose + parametric homography verify")
parser.add_argument("left_json", help="Left intrinsic JSON")
parser.add_argument("right_json", help="Right intrinsic JSON")
parser.add_argument("extrinsic_json", help="Stereo extrinsic JSON (preferably refined)")
parser.add_argument("--tag-size", type=float, required=True, help="Tag side length in mm (black square)")
parser.add_argument(
    "--tag-family",
    default="36h11",
    choices=["16h5", "25h9", "36h10", "36h11", "25h10"],
    help="AprilTag family (default: 36h11)",
)
parser.add_argument("--out", default="apriltag_verify.json", help="Save path for metrics")
from _argpick import parse_or_pick
args = parse_or_pick(
    parser,
    [
        ("left_json",      "Left intrinsic JSON",  [("JSON", "*.json")]),
        ("right_json",     "Right intrinsic JSON", [("JSON", "*.json")]),
        ("extrinsic_json", "Stereo extrinsic JSON (refined)", [("JSON", "*.json")]),
    ],
    ask_missing_options=[("tag-size", "AprilTag side length in mm (black square)")],
)


# ============================================================
# CAMERA
# ============================================================

class CameraCapture:
    def __init__(self, device, width, height, fps):
        self.device = device
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.width = width
        self.height = height
        self.fps = fps

    def start(self):
        # Strict UYVY 1280x720@60 via Jetson GStreamer (VIC path).
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
        print(f"Camera started: {self.device}")
        return True

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


# ============================================================
# CALIBRATION LOAD
# ============================================================

def first_existing(data, keys):
    for k in keys:
        if k in data:
            return data[k]
    return None


def load_intrinsics(path, name):
    with open(path) as f:
        data = json.load(f)
    K = np.asarray(
        first_existing(data, ["camera_matrix", "K", "camera_matrix_left", "camera_matrix_right"]),
        dtype=np.float64,
    )
    D = np.asarray(
        first_existing(
            data,
            ["distortion_coefficients", "D", "distortion_coefficients_left", "distortion_coefficients_right"],
        ),
        dtype=np.float64,
    ).reshape(-1, 1)
    print(f"[{name}] fx={K[0,0]:.2f} fy={K[1,1]:.2f}")
    return K, D


def load_extrinsics(path):
    with open(path) as f:
        data = json.load(f)
    R = np.asarray(
        first_existing(data, ["rotation_left_to_right", "R_left_to_right", "rotation_matrix", "R"]),
        dtype=np.float64,
    )
    T = np.asarray(
        first_existing(
            data,
            [
                "translation_left_to_right_mm",
                "T_left_to_right_mm",
                "translation_mm",
                "translation_left_to_right",
                "T",
                "translation",
            ],
        ),
        dtype=np.float64,
    ).reshape(3, 1)
    if R.shape == (3,):
        R, _ = cv2.Rodrigues(np.deg2rad(R).reshape(3, 1))
    if R.shape == (3, 1):
        R, _ = cv2.Rodrigues(R)
    print(f"[EXTRINSIC] baseline={float(np.linalg.norm(T)):.3f} mm  T={T.ravel()}")
    return R, T


K_LEFT, D_LEFT = load_intrinsics(args.left_json, "LEFT")
K_RIGHT, D_RIGHT = load_intrinsics(args.right_json, "RIGHT")
R_LR, T_LR = load_extrinsics(args.extrinsic_json)


# ============================================================
# APRILTAG SETUP
# ============================================================

FAMILY_MAP = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "25h10": getattr(cv2.aruco, "DICT_APRILTAG_25h10", cv2.aruco.DICT_APRILTAG_25h9),
}

aruco_dict = cv2.aruco.getPredefinedDictionary(FAMILY_MAP[args.tag_family])
# Detector parameters (OpenCV 4.7+ API with fallback)
try:
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    def detect_tags(gray):
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids
except AttributeError:
    detector_params = cv2.aruco.DetectorParameters_create()
    def detect_tags(gray):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        return corners, ids

TAG_SIZE_MM = float(args.tag_size)

# Object points for a single tag (centre at origin, Z=0), order matches OpenCV aruco:
# TL, TR, BR, BL in the tag plane
HALF = TAG_SIZE_MM / 2.0
TAG_OBJECT_POINTS = np.array(
    [
        [-HALF,  HALF, 0.0],
        [ HALF,  HALF, 0.0],
        [ HALF, -HALF, 0.0],
        [-HALF, -HALF, 0.0],
    ],
    dtype=np.float64,
)


# ============================================================
# GEOMETRY: pose, plane, parametric H
# ============================================================

def estimate_pose(corners_2d, K, D):
    """corners_2d: (4,2) pixel coords. Returns rvec, tvec or (None, None)."""
    img = np.asarray(corners_2d, dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        TAG_OBJECT_POINTS, img, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            TAG_OBJECT_POINTS, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )
    if not ok:
        return None, None
    return rvec, tvec


def plane_from_pose(rvec, tvec):
    """
    Tag plane in camera frame: n · X + d = 0 with ||n||=1, d >= 0 style
    (actually n·X = d where d = n·t, plane through t with normal = R[:,2]).
    Returns unit normal n (3,) and d such that n·X = d for points on the plane.
    """
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2].copy()  # tag Z axis in camera
    # Ensure normal points toward camera (n · t < 0 typically for front-facing)
    t = tvec.ravel()
    if np.dot(n, t) > 0:
        n = -n
    d = float(np.dot(n, t))  # n · X = d on the plane
    return n, d


def homography_from_camera_params(K_src, K_dst, R_src_to_dst, T_src_to_dst, n_src, d_src):
    """
    Plane-induced homography from camera parameters only.

    Plane in SOURCE camera:  n · X = d  (d > 0, n into the scene).
    Extrinsic convention:    X_dst = R X_src + T

    On the plane (n·X)/d = 1, so:
        T = T (n·X)/d = (T n^T / d) X
        X_dst = R X + T = (R + T n^T / d) X

    Therefore:
        H = K_dst ( R + T n^T / d ) K_src^{-1}
    """
    if abs(d_src) < 1e-9:
        return None
    n = n_src.reshape(3, 1)
    # R + T n^T / d  (PLUS — matches X' = R X + T)
    H_euclid = R_src_to_dst + (T_src_to_dst @ n.T) / d_src
    H = K_dst @ H_euclid @ np.linalg.inv(K_src)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def map_points(points, H):
    if points is None or H is None:
        return None
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def project_tag_corners(rvec, tvec, K, D=None):
    """Project 3D tag corners into this camera (optionally with distortion)."""
    if D is None:
        D = np.zeros((5, 1), dtype=np.float64)
    proj, _ = cv2.projectPoints(TAG_OBJECT_POINTS, rvec, tvec, K, D)
    return proj.reshape(-1, 2)


def transform_pose_left_to_right(rvec_l, tvec_l, R_lr, T_lr):
    """Tag pose in right camera given pose in left and stereo extrinsics."""
    R_tag_l, _ = cv2.Rodrigues(rvec_l)
    R_tag_r = R_lr @ R_tag_l
    t_tag_r = R_lr @ tvec_l + T_lr
    rvec_r, _ = cv2.Rodrigues(R_tag_r)
    return rvec_r, t_tag_r


def error_stats(pred, actual):
    if pred is None or actual is None or len(pred) != len(actual):
        return None
    e = np.linalg.norm(pred - actual, axis=1)
    return {
        "mean": float(np.mean(e)),
        "median": float(np.median(e)),
        "rms": float(np.sqrt(np.mean(e ** 2))),
        "max": float(np.max(e)),
        "per_corner": e.tolist(),
    }


def to_ideal_pixels(points, K, D):
    """Distorted pixels → ideal pinhole pixels (same K, no distortion)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)


def to_distorted_pixels(points_ideal, K, D):
    """Ideal pinhole pixels → distorted pixels (for drawing on raw image)."""
    pts = np.asarray(points_ideal, dtype=np.float64).reshape(-1, 2)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy
    pts3d = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(
        pts3d, np.zeros(3), np.zeros(3), K, D
    )
    return proj.reshape(-1, 2)


# ============================================================
# DRAW
# ============================================================

def safe_pt(p):
    if not np.all(np.isfinite(p)):
        return None
    x, y = float(p[0]), float(p[1])
    if abs(x) > 1e5 or abs(y) > 1e5:
        return None
    return (int(round(x)), int(round(y)))


def draw_tag(img, corners, color, label=None):
    if corners is None:
        return
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    for i in range(4):
        p1 = safe_pt(pts[i])
        p2 = safe_pt(pts[(i + 1) % 4])
        if p1 and p2:
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        p = safe_pt(pt)
        if p:
            cv2.circle(img, p, 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if label and safe_pt(pts[0]):
        p = safe_pt(pts[0])
        cv2.putText(img, label, (p[0], p[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_stars(img, points, color):
    if points is None:
        return
    for pt in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        p = safe_pt(pt)
        if p is None:
            continue
        x, y = p
        cv2.putText(img, "*", (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLACK, 4, cv2.LINE_AA)
        cv2.putText(img, "*", (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def put_text(img, text, pos, scale=0.6, color=WHITE, thickness=2):
    x, y = map(int, pos)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fit_to_screen(img, max_w=MAX_DISPLAY_WIDTH, max_h=MAX_DISPLAY_HEIGHT):
    h, w = img.shape[:2]
    s = min(1.0, max_w / float(w), max_h / float(h))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def draw_panel(panel, left_det, right_det, err_lr, err_rl, pose_l, pose_r):
    panel[:] = DARK
    put_text(panel, "APRILTAG STEREO VERIFY", (16, 36), scale=0.75, color=CYAN, thickness=2)
    put_text(panel, "H from camera params only", (16, 62), scale=0.50, color=YELLOW, thickness=1)
    cv2.line(panel, (16, 75), (PANEL_WIDTH - 16, 75), CYAN, 1)

    put_text(panel, "DETECTION", (16, 110), scale=0.65, color=ORANGE, thickness=2)
    put_text(panel, f"Left  : {'YES id='+str(left_det['id']) if left_det else 'NO'}",
             (16, 140), scale=0.55, color=GREEN if left_det else RED, thickness=2)
    put_text(panel, f"Right : {'YES id='+str(right_det['id']) if right_det else 'NO'}",
             (16, 168), scale=0.55, color=GREEN if right_det else RED, thickness=2)

    put_text(panel, "POSE (left camera, mm)", (16, 215), scale=0.65, color=ORANGE, thickness=2)
    if pose_l is not None:
        t = pose_l[1].ravel()
        put_text(panel, f"t = ({t[0]:+.1f}, {t[1]:+.1f}, {t[2]:+.1f})", (16, 245), scale=0.52, color=WHITE, thickness=1)
        dist = float(np.linalg.norm(t))
        put_text(panel, f"|t| = {dist:.1f} mm", (16, 270), scale=0.52, color=CYAN, thickness=1)
    else:
        put_text(panel, "n/a", (16, 245), scale=0.52, color=YELLOW, thickness=1)

    put_text(panel, "HOMOGRAPHY ERROR (parametric H)", (16, 320), scale=0.58, color=ORANGE, thickness=2)
    put_text(panel, "L→R  (H maps left corners → right)", (16, 350), scale=0.50, color=YELLOW, thickness=1)
    if err_lr:
        put_text(panel, f"  RMS={err_lr['rms']:.2f}  mean={err_lr['mean']:.2f}  max={err_lr['max']:.2f}",
                 (16, 378), scale=0.50, color=GREEN if err_lr["rms"] < 5 else RED, thickness=1)
    else:
        put_text(panel, "  need both detections", (16, 378), scale=0.50, color=YELLOW, thickness=1)

    put_text(panel, "R→L  (H maps right corners → left)", (16, 415), scale=0.50, color=YELLOW, thickness=1)
    if err_rl:
        put_text(panel, f"  RMS={err_rl['rms']:.2f}  mean={err_rl['mean']:.2f}  max={err_rl['max']:.2f}",
                 (16, 443), scale=0.50, color=GREEN if err_rl["rms"] < 5 else RED, thickness=1)
    else:
        put_text(panel, "  need both detections", (16, 443), scale=0.50, color=YELLOW, thickness=1)

    put_text(panel, "LEGEND", (16, 500), scale=0.65, color=ORANGE, thickness=2)
    put_text(panel, "Green box  = detected tag", (16, 530), scale=0.50, color=GREEN, thickness=1)
    put_text(panel, "Blue *     = H-mapped corners", (16, 555), scale=0.50, color=BLUE, thickness=1)
    put_text(panel, "H = K2 (R + t n^T/d) K1^{-1}", (16, 590), scale=0.48, color=CYAN, thickness=1)

    put_text(panel, f"Tag family : {args.tag_family}", (16, 640), scale=0.50, color=WHITE, thickness=1)
    put_text(panel, f"Tag size   : {TAG_SIZE_MM:.1f} mm", (16, 665), scale=0.50, color=WHITE, thickness=1)
    put_text(panel, f"Baseline   : {float(np.linalg.norm(T_LR)):.2f} mm", (16, 690), scale=0.50, color=WHITE, thickness=1)

    put_text(panel, "S : save metrics", (16, 740), scale=0.55, color=CYAN, thickness=2)
    put_text(panel, "Q / ESC : quit", (16, 770), scale=0.55, color=RED, thickness=2)


# ============================================================
# MAIN LOOP LOGIC
# ============================================================

def process(frame_l, frame_r):
    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

    corners_l, ids_l = detect_tags(gray_l)
    corners_r, ids_r = detect_tags(gray_r)

    left_det = right_det = None
    # ids shape varies by OpenCV version (N,), (N,1), etc. — always ravel.
    if ids_l is not None and len(ids_l) > 0:
        ids_l_flat = np.asarray(ids_l).ravel()
        left_det = {
            "id": int(ids_l_flat[0]),
            "corners": np.asarray(corners_l[0], dtype=np.float64).reshape(4, 2),
        }
    if ids_r is not None and len(ids_r) > 0:
        ids_r_flat = np.asarray(ids_r).ravel()
        chosen = 0
        if left_det is not None:
            for i, tid in enumerate(ids_r_flat):
                if int(tid) == left_det["id"]:
                    chosen = i
                    break
        right_det = {
            "id": int(ids_r_flat[chosen]),
            "corners": np.asarray(corners_r[chosen], dtype=np.float64).reshape(4, 2),
        }

    # Display RAW frames — no cv2.undistort (avoids mushroom warping).
    raw_l = frame_l
    raw_r = frame_r

    pose_l = pose_r = None
    err_lr = err_rl = None
    mapped_to_r = mapped_to_l = None
    corners_l = corners_r = None

    if left_det is not None:
        corners_l = left_det["corners"]  # distorted pixel coords
        rvec_l, tvec_l = estimate_pose(corners_l, K_LEFT, D_LEFT)
        if rvec_l is not None:
            pose_l = (rvec_l, tvec_l)

    if right_det is not None:
        corners_r = right_det["corners"]  # distorted pixel coords
        rvec_r, tvec_r = estimate_pose(corners_r, K_RIGHT, D_RIGHT)
        if rvec_r is not None:
            pose_r = (rvec_r, tvec_r)

    # Parametric H is a pinhole relation. Pipeline:
    #   distorted src → ideal → H → ideal dst → distorted dst (for overlay)
    if pose_l is not None and right_det is not None:
        n, d = plane_from_pose(pose_l[0], pose_l[1])

        # Left → Right
        H_lr = homography_from_camera_params(K_LEFT, K_RIGHT, R_LR, T_LR, n, d)
        ideal_l = to_ideal_pixels(corners_l, K_LEFT, D_LEFT)
        ideal_mapped_r = map_points(ideal_l, H_lr)
        mapped_to_r = to_distorted_pixels(ideal_mapped_r, K_RIGHT, D_RIGHT)
        err_lr = error_stats(mapped_to_r, corners_r)

        # Right → Left
        rvec_r_from_l, tvec_r_from_l = transform_pose_left_to_right(
            pose_l[0], pose_l[1], R_LR, T_LR
        )
        n_r, d_r = plane_from_pose(rvec_r_from_l, tvec_r_from_l)
        R_rl = R_LR.T
        T_rl = -R_LR.T @ T_LR
        H_rl = homography_from_camera_params(K_RIGHT, K_LEFT, R_rl, T_rl, n_r, d_r)
        ideal_r = to_ideal_pixels(corners_r, K_RIGHT, D_RIGHT)
        ideal_mapped_l = map_points(ideal_r, H_rl)
        mapped_to_l = to_distorted_pixels(ideal_mapped_l, K_LEFT, D_LEFT)
        err_rl = error_stats(mapped_to_l, corners_l)

    return {
        "raw_l": raw_l,
        "raw_r": raw_r,
        "left_det": left_det,
        "right_det": right_det,
        "corners_l": corners_l,
        "corners_r": corners_r,
        "mapped_to_r": mapped_to_r,
        "mapped_to_l": mapped_to_l,
        "err_lr": err_lr,
        "err_rl": err_rl,
        "pose_l": pose_l,
        "pose_r": pose_r,
    }


def render(state):
    # Draw on RAW (distorted) frames — no mushroom from cv2.undistort
    disp_l = state["raw_l"].copy()
    disp_r = state["raw_r"].copy()

    if state["corners_l"] is not None:
        draw_tag(disp_l, state["corners_l"], GREEN, label="detected")
    if state["corners_r"] is not None:
        draw_tag(disp_r, state["corners_r"], GREEN, label="detected")

    # H-mapped corners as blue * (already re-distorted into raw pixel space)
    draw_stars(disp_r, state["mapped_to_r"], BLUE)
    draw_stars(disp_l, state["mapped_to_l"], BLUE)

    put_text(disp_l, "LEFT (raw)", (16, 36), scale=0.8, color=CYAN, thickness=2)
    put_text(disp_r, "RIGHT (raw)", (16, 36), scale=0.8, color=ORANGE, thickness=2)

    views = np.vstack((disp_l, disp_r))
    panel = np.zeros((views.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
    draw_panel(
        panel,
        state["left_det"],
        state["right_det"],
        state["err_lr"],
        state["err_rl"],
        state["pose_l"],
        state["pose_r"],
    )
    return fit_to_screen(np.hstack((views, panel)))


def save_metrics(path, state):
    payload = {
        "tag_family": args.tag_family,
        "tag_size_mm": TAG_SIZE_MM,
        "left_id": state["left_det"]["id"] if state["left_det"] else None,
        "right_id": state["right_det"]["id"] if state["right_det"] else None,
        "error_left_to_right": state["err_lr"],
        "error_right_to_left": state["err_rl"],
        "homography_source": "camera_parameters_only",
        "formula": "H = K_dst (R + T n^T / d) K_src^{-1}",
    }
    if state["pose_l"] is not None:
        payload["pose_left_t_mm"] = state["pose_l"][1].ravel().tolist()
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVE] {path}")
    if state["err_lr"]:
        el = state["err_lr"]
        print(f"       L→R  mean={el['mean']:.2f}  RMS={el['rms']:.2f}  max={el['max']:.2f} px")
        print(f"       L→R  per-corner: {[round(e, 2) for e in el['per_corner']]}")
    if state["err_rl"]:
        er = state["err_rl"]
        print(f"       R→L  mean={er['mean']:.2f}  RMS={er['rms']:.2f}  max={er['max']:.2f} px")
        print(f"       R→L  per-corner: {[round(e, 2) for e in er['per_corner']]}")
    if state["pose_l"] is not None:
        t = state["pose_l"][1].ravel()
        print(f"       pose_left t = ({t[0]:+.1f}, {t[1]:+.1f}, {t[2]:+.1f}) mm  |t|={np.linalg.norm(t):.1f}")


def main():
    left_cam = CameraCapture(LEFT_DEVICE, IMAGE_WIDTH, IMAGE_HEIGHT, FPS)
    right_cam = CameraCapture(RIGHT_DEVICE, IMAGE_WIDTH, IMAGE_HEIGHT, FPS)
    if not left_cam.start():
        return 1
    if not right_cam.start():
        left_cam.stop()
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print()
    print("=" * 64)
    print(f"AprilTag family={args.tag_family}  size={TAG_SIZE_MM} mm")
    print("H is computed from K, R, T and the tag plane — never from point matches.")
    print("S = save   Q/ESC = quit")
    print("=" * 64)
    print()
    print(f"{'id':>4}  {'dist_mm':>8}  {'L→R mean':>9}  {'L→R RMS':>9}  {'L→R max':>9}  "
          f"{'R→L mean':>9}  {'R→L RMS':>9}  {'R→L max':>9}")
    print("-" * 90)

    last_print = 0.0

    try:
        while True:
            fl = left_cam.get_frame()
            fr = right_cam.get_frame()
            if fl is None or fr is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            state = process(fl, fr)
            cv2.imshow(WINDOW_NAME, render(state))

            # Print errors ~2 Hz when both sides see the tag
            now = time.time()
            if (
                state["err_lr"] is not None
                and state["err_rl"] is not None
                and (now - last_print) > 0.5
            ):
                last_print = now
                el, er = state["err_lr"], state["err_rl"]
                tid = state["left_det"]["id"] if state["left_det"] else -1
                dist = float(np.linalg.norm(state["pose_l"][1])) if state["pose_l"] else 0.0
                print(
                    f"{tid:4d}  {dist:8.1f}  "
                    f"{el['mean']:9.2f}  {el['rms']:9.2f}  {el['max']:9.2f}  "
                    f"{er['mean']:9.2f}  {er['rms']:9.2f}  {er['max']:9.2f}"
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                if state["left_det"] and state["right_det"]:
                    save_metrics(args.out, state)
                else:
                    print("[SAVE] need tag visible in both cameras")
    finally:
        left_cam.stop()
        right_cam.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
