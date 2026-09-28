#!/usr/bin/env python3
"""
Live stereo extrinsic refinement.

Detects the chessboard on BOTH cameras, then refines R and T so that
3D points from the left camera reproject correctly into the right
(and optionally the other way).

Prior knowledge encoded as soft constraints:
  - Baseline length ≈ 45 mm  (vernier measurement)
  - Rx ≈ 20 deg              (design angle)
  - Ry ≈ 0, Rz ≈ 0           (CAD mount, small residual OK)

Usage:
  python3 refine_stereo_extrinsics.py left.json right.json extrinsics.json

Keys:
  O          run one optimization step (or hold to keep refining)
  A          auto-refine continuously while board is visible
  S          save refined extrinsics to refined_extrinsics.json
  R          reset to original extrinsics
  Q / ESC    quit
"""

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


# ============================================================
# CONFIGURATION
# ============================================================

LEFT_DEVICE = "/dev/video1"       # BORESIGHT / LEFT  (4 mm)
RIGHT_DEVICE = "/dev/video3"      # DEPRESSION / RIGHT (4 mm)

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 60

BOARD_COLS = 8
BOARD_ROWS = 5
SQUARE_SIZE_MM = 30.0

# Soft priors (tune weights below)
PRIOR_BASELINE_MM = 45.0
PRIOR_RX_DEG = 20.0
PRIOR_RY_DEG = 0.0
PRIOR_RZ_DEG = 0.0

# Weights for soft constraints in the cost (relative to pixel error)
W_BASELINE = 2.0      # mm residual weight
W_RX = 0.5            # deg residual weight
W_RY = 1.0
W_RZ = 1.0

WINDOW_NAME = "Stereo Extrinsic Refinement"

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

ROW_COLORS = [
    (0, 255, 255),
    (0, 255, 0),
    (255, 255, 0),
    (255, 165, 0),
    (255, 0, 255),
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Live refine stereo extrinsics from dual chessboard detections."
)
parser.add_argument("left_json", help="Left / boresight intrinsic JSON")
parser.add_argument("right_json", help="Right / depression intrinsic JSON")
parser.add_argument("extrinsic_json", help="Initial stereo extrinsic JSON")
parser.add_argument(
    "--out",
    default="refined_extrinsics.json",
    help="Output path for refined extrinsics (default: refined_extrinsics.json)",
)
parser.add_argument(
    "--baseline",
    type=float,
    default=PRIOR_BASELINE_MM,
    help=f"Target baseline in mm (default: {PRIOR_BASELINE_MM})",
)
parser.add_argument(
    "--rx",
    type=float,
    default=PRIOR_RX_DEG,
    help=f"Target Rx in degrees (default: {PRIOR_RX_DEG})",
)
args = parser.parse_args()

PRIOR_BASELINE_MM = args.baseline
PRIOR_RX_DEG = args.rx


# ============================================================
# CAMERA CAPTURE
# ============================================================

class CameraCapture:
    def __init__(self, device, width, height, fps):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

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
            print(f"ERROR: Could not open GStreamer pipeline for {self.device}")
            return False
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"Camera started: {self.device}")
        return True

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# ============================================================
# JSON HELPERS
# ============================================================

def first_existing(data, keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def load_camera_calibration(json_path, name):
    with open(json_path, "r") as f:
        data = json.load(f)

    K_raw = first_existing(
        data, ["camera_matrix", "camera_matrix_left", "camera_matrix_right", "K"]
    )
    D_raw = first_existing(
        data,
        [
            "distortion_coefficients",
            "distortion_coefficients_left",
            "distortion_coefficients_right",
            "D",
        ],
    )
    if K_raw is None:
        raise KeyError(f"{json_path}: no camera matrix found")
    if D_raw is None:
        raise KeyError(f"{json_path}: no distortion coefficients found")

    K = np.asarray(K_raw, dtype=np.float64)
    D = np.asarray(D_raw, dtype=np.float64).reshape(-1, 1)
    if K.shape != (3, 3):
        raise ValueError(f"{json_path}: camera matrix must be 3x3")

    square = data.get("square_size_mm", SQUARE_SIZE_MM)

    print()
    print("=" * 72)
    print(f"{name} CALIBRATION")
    print("=" * 72)
    print(f"File: {json_path}")
    print("K =")
    print(K)
    print("D =", D.ravel())
    if "reprojection_error" in data:
        print(f"Reprojection error = {float(data['reprojection_error']):.6f} px")
    return K, D, float(square)


def load_extrinsic_calibration(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    R_raw = first_existing(
        data,
        ["rotation_left_to_right", "R_left_to_right", "rotation_matrix", "R"],
    )
    T_raw = first_existing(
        data,
        [
            "translation_left_to_right_mm",
            "T_left_to_right_mm",
            "translation_mm",
            "translation_left_to_right",
            "T",
            "translation",
        ],
    )
    if R_raw is None:
        raise KeyError(f"{json_path}: no rotation found")
    if T_raw is None:
        raise KeyError(f"{json_path}: no translation found")

    R = np.asarray(R_raw, dtype=np.float64)
    T = np.asarray(T_raw, dtype=np.float64).reshape(3, 1)

    if R.shape == (3,):
        R, _ = cv2.Rodrigues(np.deg2rad(R).reshape(3, 1))
    if R.shape == (3, 1):
        R, _ = cv2.Rodrigues(R)
    if R.shape != (3, 3):
        raise ValueError(f"{json_path}: rotation must be 3x3 or 3-vector")

    print()
    print("=" * 72)
    print("INITIAL STEREO EXTRINSICS")
    print("=" * 72)
    print(f"File: {json_path}")
    print("R =")
    print(R)
    print("T [mm] =", T.ravel())
    print(f"Baseline = {float(np.linalg.norm(T)):.3f} mm")
    return R, T


def save_extrinsics(path, R, T, extra=None):
    payload = {
        "rotation_left_to_right": R.tolist(),
        "translation_left_to_right_mm": T.reshape(3, 1).tolist(),
        "baseline_mm": float(np.linalg.norm(T)),
        "euler_xyz_deg": rotation_matrix_to_euler_xyz(R).tolist(),
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVE] Wrote refined extrinsics → {path}")


# ============================================================
# LOAD CALIBRATION
# ============================================================

K_LEFT, D_LEFT, SQ_L = load_camera_calibration(args.left_json, "LEFT / BORESIGHT")
K_RIGHT, D_RIGHT, SQ_R = load_camera_calibration(args.right_json, "RIGHT / DEPRESSION")
SQUARE_SIZE_MM = SQ_L  # prefer left JSON value

R_INIT, T_INIT = load_extrinsic_calibration(args.extrinsic_json)
R = R_INIT.copy()
T = T_INIT.copy()


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def make_object_points(cols, rows, square_mm):
    obj = np.zeros((rows * cols, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            obj[r * cols + c] = [c * square_mm, r * square_mm, 0.0]
    return obj


OBJECT_POINTS = make_object_points(BOARD_COLS, BOARD_ROWS, SQUARE_SIZE_MM)


def rotation_matrix_to_euler_xyz(Rm):
    sy = np.sqrt(Rm[0, 0] ** 2 + Rm[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        rx = np.arctan2(Rm[2, 1], Rm[2, 2])
        ry = np.arctan2(-Rm[2, 0], sy)
        rz = np.arctan2(Rm[1, 0], Rm[0, 0])
    else:
        rx = np.arctan2(-Rm[1, 2], Rm[1, 1])
        ry = np.arctan2(-Rm[2, 0], sy)
        rz = 0.0
    return np.rad2deg([rx, ry, rz])


def euler_xyz_to_rotation_matrix(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]],
        dtype=np.float64,
    )
    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]],
        dtype=np.float64,
    )
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return Rz @ Ry @ Rx


def pack_params(R, T):
    """Pack R,T into optimization vector: [Rx,Ry,Rz, Tx,Ty,Tz] (deg, mm)."""
    e = rotation_matrix_to_euler_xyz(R)
    return np.array([e[0], e[1], e[2], T[0, 0], T[1, 0], T[2, 0]], dtype=np.float64)


def unpack_params(x):
    R = euler_xyz_to_rotation_matrix(x[0], x[1], x[2])
    T = np.array([[x[3]], [x[4]], [x[5]]], dtype=np.float64)
    return R, T


# ============================================================
# DETECTION / PROJECTION
# ============================================================

def find_chessboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        (BOARD_COLS, BOARD_ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners.reshape(-1, 2)


def solve_board_pose(corners_raw, K, D):
    img_pts = np.asarray(corners_raw, dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        OBJECT_POINTS, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None
    return rvec, tvec


def project_left_to_right(corners_left_raw, R_lr, T_lr):
    """
    solvePnP on left → 3D in left → transform to right → project (no distortion
    so points land on the undistorted right image).
    """
    rvec, tvec = solve_board_pose(corners_left_raw, K_LEFT, D_LEFT)
    if rvec is None:
        return None
    R_board, _ = cv2.Rodrigues(rvec)
    pts_left = (R_board @ OBJECT_POINTS.T + tvec).T  # (N,3)
    pts_right = (R_lr @ pts_left.T + T_lr).T
    projected, _ = cv2.projectPoints(
        pts_right.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K_RIGHT,
        np.zeros((5, 1)),
    )
    return projected.reshape(-1, 2)


def project_right_to_left(corners_right_raw, R_lr, T_lr):
    rvec, tvec = solve_board_pose(corners_right_raw, K_RIGHT, D_RIGHT)
    if rvec is None:
        return None
    R_board, _ = cv2.Rodrigues(rvec)
    pts_right = (R_board @ OBJECT_POINTS.T + tvec).T
    # X_left = R^T (X_right - T)
    pts_left = (R_lr.T @ (pts_right.T - T_lr)).T
    projected, _ = cv2.projectPoints(
        pts_left.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K_LEFT,
        np.zeros((5, 1)),
    )
    return projected.reshape(-1, 2)


def undistort_image(frame, K, D):
    return cv2.undistort(frame, K, D, None, K)


def undistort_points(points, K, D):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)


# ============================================================
# COST / OPTIMIZATION
# ============================================================

def reprojection_residuals(x, left_corners_raw, right_corners_raw):
    """
    Residuals = bidirectional pixel errors + soft priors.
    """
    R_lr, T_lr = unpack_params(x)

    residuals = []

    # Left → Right
    pred_r = project_left_to_right(left_corners_raw, R_lr, T_lr)
    if pred_r is not None:
        actual_r = undistort_points(right_corners_raw, K_RIGHT, D_RIGHT)
        residuals.append((pred_r - actual_r).ravel())

    # Right → Left
    pred_l = project_right_to_left(right_corners_raw, R_lr, T_lr)
    if pred_l is not None:
        actual_l = undistort_points(left_corners_raw, K_LEFT, D_LEFT)
        residuals.append((pred_l - actual_l).ravel())

    if not residuals:
        return np.zeros(6, dtype=np.float64)

    pix = np.concatenate(residuals)

    # Soft priors
    baseline = float(np.linalg.norm(T_lr))
    rx, ry, rz = x[0], x[1], x[2]

    priors = np.array(
        [
            W_BASELINE * (baseline - PRIOR_BASELINE_MM),
            W_RX * (rx - PRIOR_RX_DEG),
            W_RY * (ry - PRIOR_RY_DEG),
            W_RZ * (rz - PRIOR_RZ_DEG),
        ],
        dtype=np.float64,
    )

    return np.concatenate([pix, priors])


def mean_reprojection_error(R_lr, T_lr, left_corners_raw, right_corners_raw):
    errors = []
    pred_r = project_left_to_right(left_corners_raw, R_lr, T_lr)
    if pred_r is not None:
        actual_r = undistort_points(right_corners_raw, K_RIGHT, D_RIGHT)
        errors.append(np.linalg.norm(pred_r - actual_r, axis=1))
    pred_l = project_right_to_left(right_corners_raw, R_lr, T_lr)
    if pred_l is not None:
        actual_l = undistort_points(left_corners_raw, K_LEFT, D_LEFT)
        errors.append(np.linalg.norm(pred_l - actual_l, axis=1))
    if not errors:
        return None
    all_e = np.concatenate(errors)
    return {
        "mean": float(np.mean(all_e)),
        "median": float(np.median(all_e)),
        "rms": float(np.sqrt(np.mean(all_e ** 2))),
        "max": float(np.max(all_e)),
    }


def refine_once(R_lr, T_lr, left_corners_raw, right_corners_raw):
    """One least-squares refinement from current R,T."""
    x0 = pack_params(R_lr, T_lr)
    result = least_squares(
        reprojection_residuals,
        x0,
        args=(left_corners_raw, right_corners_raw),
        method="lm",
        max_nfev=80,
        verbose=0,
    )
    R_new, T_new = unpack_params(result.x)
    return R_new, T_new, result.cost


# ============================================================
# DRAWING
# ============================================================

def safe_point(point):
    if not np.all(np.isfinite(point)):
        return None
    x, y = float(point[0]), float(point[1])
    if abs(x) > 1e5 or abs(y) > 1e5:
        return None
    return (int(round(x)), int(round(y)))


def draw_points(image, points, filled=True, radius=5, color=None):
    if points is None:
        return
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    for i, pt in enumerate(points):
        p = safe_point(pt)
        if p is None:
            continue
        c = color if color is not None else ROW_COLORS[(i // BOARD_COLS) % len(ROW_COLORS)]
        if filled:
            cv2.circle(image, p, radius + 2, BLACK, -1, cv2.LINE_AA)
            cv2.circle(image, p, radius, c, -1, cv2.LINE_AA)
        else:
            cv2.circle(image, p, radius + 1, BLACK, -1, cv2.LINE_AA)
            cv2.circle(image, p, radius, c, 2, cv2.LINE_AA)


def draw_connections(image, points):
    if points is None:
        return
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    for row in range(BOARD_ROWS):
        start = row * BOARD_COLS
        for i in range(start, start + BOARD_COLS - 1):
            p1, p2 = safe_point(points[i]), safe_point(points[i + 1])
            if p1 is None or p2 is None:
                continue
            color = ROW_COLORS[row % len(ROW_COLORS)]
            cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)


def put_text(image, text, position, scale=0.65, color=WHITE, thickness=2):
    x, y = map(int, position)
    cv2.putText(image, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 4, cv2.LINE_AA)
    cv2.putText(image, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fit_to_screen(image, max_w=MAX_DISPLAY_WIDTH, max_h=MAX_DISPLAY_HEIGHT):
    h, w = image.shape[:2]
    scale = min(1.0, max_w / float(w), max_h / float(h))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def draw_panel(panel, left_ok, right_ok, err, auto_mode, last_cost):
    panel[:] = DARK
    put_text(panel, "EXTRINSIC REFINEMENT", (20, 40), scale=0.90, color=CYAN, thickness=3)
    cv2.line(panel, (20, 55), (PANEL_WIDTH - 20, 55), CYAN, 2)

    # Detection
    put_text(panel, "DETECTION", (20, 95), scale=0.70, color=YELLOW, thickness=2)
    put_text(
        panel,
        "Left  : DETECTED" if left_ok else "Left  : NOT DETECTED",
        (20, 125),
        scale=0.60,
        color=GREEN if left_ok else RED,
        thickness=2,
    )
    put_text(
        panel,
        "Right : DETECTED" if right_ok else "Right : NOT DETECTED",
        (20, 155),
        scale=0.60,
        color=GREEN if right_ok else RED,
        thickness=2,
    )

    # Current extrinsics
    angles = rotation_matrix_to_euler_xyz(R)
    baseline = float(np.linalg.norm(T))

    put_text(panel, "CURRENT R / T", (20, 200), scale=0.70, color=ORANGE, thickness=2)
    put_text(panel, f"Rx : {angles[0]:+.3f} deg   (prior {PRIOR_RX_DEG:.1f})", (20, 235), scale=0.58, color=GREEN, thickness=2)
    put_text(panel, f"Ry : {angles[1]:+.3f} deg   (prior {PRIOR_RY_DEG:.1f})", (20, 265), scale=0.58, color=WHITE, thickness=2)
    put_text(panel, f"Rz : {angles[2]:+.3f} deg   (prior {PRIOR_RZ_DEG:.1f})", (20, 295), scale=0.58, color=WHITE, thickness=2)
    put_text(panel, f"Tx : {T[0, 0]:+.3f} mm", (20, 330), scale=0.58, color=WHITE, thickness=2)
    put_text(panel, f"Ty : {T[1, 0]:+.3f} mm", (20, 360), scale=0.58, color=YELLOW, thickness=2)
    put_text(panel, f"Tz : {T[2, 0]:+.3f} mm", (20, 390), scale=0.58, color=WHITE, thickness=2)
    put_text(
        panel,
        f"Baseline : {baseline:.3f} mm   (prior {PRIOR_BASELINE_MM:.1f})",
        (20, 425),
        scale=0.58,
        color=CYAN,
        thickness=2,
    )

    # Error
    put_text(panel, "REPROJECTION ERROR (both directions)", (20, 475), scale=0.65, color=ORANGE, thickness=2)
    if err is None:
        put_text(panel, "Need board visible in BOTH cameras", (20, 510), scale=0.55, color=YELLOW, thickness=2)
    else:
        put_text(panel, f"Mean   : {err['mean']:.3f} px", (20, 510), scale=0.58, color=WHITE, thickness=2)
        put_text(panel, f"Median : {err['median']:.3f} px", (20, 540), scale=0.58, color=WHITE, thickness=2)
        put_text(panel, f"RMS    : {err['rms']:.3f} px", (20, 570), scale=0.58, color=GREEN, thickness=2)
        put_text(panel, f"Max    : {err['max']:.3f} px", (20, 600), scale=0.58, color=WHITE, thickness=2)

    if last_cost is not None:
        put_text(panel, f"Last LS cost : {last_cost:.4f}", (20, 640), scale=0.55, color=CYAN, thickness=2)

    # Legend
    put_text(panel, "LEGEND", (20, 690), scale=0.70, color=ORANGE, thickness=2)
    put_text(panel, "Filled dots  = detected corners", (20, 725), scale=0.55, color=WHITE, thickness=2)
    put_text(panel, "Blue rings   = projected via R/T", (20, 755), scale=0.55, color=BLUE, thickness=2)

    # Controls
    cv2.line(panel, (20, 790), (PANEL_WIDTH - 20, 790), CYAN, 2)
    put_text(panel, "CONTROLS", (20, 825), scale=0.70, color=YELLOW, thickness=2)
    put_text(panel, "O : one optimization step", (20, 860), scale=0.55, color=WHITE, thickness=2)
    put_text(
        panel,
        f"A : auto-refine  [{'ON' if auto_mode else 'OFF'}]",
        (20, 890),
        scale=0.55,
        color=GREEN if auto_mode else WHITE,
        thickness=2,
    )
    put_text(panel, "S : save refined_extrinsics.json", (20, 920), scale=0.55, color=CYAN, thickness=2)
    put_text(panel, "R : reset to original extrinsics", (20, 950), scale=0.55, color=ORANGE, thickness=2)
    put_text(panel, "Q / ESC : quit", (20, 980), scale=0.55, color=RED, thickness=2)

    # Initial values for reference
    a0 = rotation_matrix_to_euler_xyz(R_INIT)
    b0 = float(np.linalg.norm(T_INIT))
    put_text(panel, "INITIAL (reference)", (20, 1030), scale=0.60, color=YELLOW, thickness=2)
    put_text(
        panel,
        f"Rx={a0[0]:+.2f}  Ry={a0[1]:+.2f}  Rz={a0[2]:+.2f}",
        (20, 1060),
        scale=0.50,
        color=WHITE,
        thickness=1,
    )
    put_text(
        panel,
        f"T=({T_INIT[0,0]:+.1f}, {T_INIT[1,0]:+.1f}, {T_INIT[2,0]:+.1f})  base={b0:.1f}",
        (20, 1085),
        scale=0.50,
        color=WHITE,
        thickness=1,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    global R, T

    left_cam = CameraCapture(LEFT_DEVICE, IMAGE_WIDTH, IMAGE_HEIGHT, FPS)
    right_cam = CameraCapture(RIGHT_DEVICE, IMAGE_WIDTH, IMAGE_HEIGHT, FPS)

    if not left_cam.start():
        return 1
    if not right_cam.start():
        left_cam.stop()
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    auto_mode = False
    last_cost = None

    print()
    print("=" * 72)
    print("CONTROLS")
    print("=" * 72)
    print("O : run one optimization step")
    print("A : toggle auto-refine")
    print("S : save refined extrinsics")
    print("R : reset to original")
    print("Q / ESC : quit")
    print()
    print(f"Priors: baseline={PRIOR_BASELINE_MM} mm, Rx={PRIOR_RX_DEG} deg, Ry=Rz=0")
    print("=" * 72)
    print()

    try:
        while True:
            frame_l = left_cam.get_frame()
            frame_r = right_cam.get_frame()
            if frame_l is None or frame_r is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            undist_l = undistort_image(frame_l, K_LEFT, D_LEFT)
            undist_r = undistort_image(frame_r, K_RIGHT, D_RIGHT)

            left_ok, left_raw = find_chessboard(frame_l)
            right_ok, right_raw = find_chessboard(frame_r)

            left_undist = undistort_points(left_raw, K_LEFT, D_LEFT) if left_ok else None
            right_undist = undistort_points(right_raw, K_RIGHT, D_RIGHT) if right_ok else None

            pred_on_right = None
            pred_on_left = None
            err = None

            if left_ok and right_ok:
                pred_on_right = project_left_to_right(left_raw, R, T)
                pred_on_left = project_right_to_left(right_raw, R, T)
                err = mean_reprojection_error(R, T, left_raw, right_raw)

                if auto_mode:
                    R, T, last_cost = refine_once(R, T, left_raw, right_raw)
                    err = mean_reprojection_error(R, T, left_raw, right_raw)

            # Draw
            disp_l = undist_l.copy()
            disp_r = undist_r.copy()

            if left_ok:
                draw_connections(disp_l, left_undist)
                draw_points(disp_l, left_undist, filled=True, radius=5)
            if right_ok:
                draw_connections(disp_r, right_undist)
                draw_points(disp_r, right_undist, filled=True, radius=5)

            # Projected points (blue rings)
            draw_points(disp_r, pred_on_right, filled=False, radius=5, color=BLUE)
            draw_points(disp_l, pred_on_left, filled=False, radius=5, color=BLUE)

            put_text(disp_l, "LEFT / BORESIGHT", (20, 40), scale=0.80, color=CYAN, thickness=3)
            put_text(disp_r, "RIGHT / DEPRESSION", (20, 40), scale=0.80, color=ORANGE, thickness=3)

            views = np.vstack((disp_l, disp_r))
            panel = np.zeros((views.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
            draw_panel(panel, left_ok, right_ok, err, auto_mode, last_cost)

            combined = fit_to_screen(np.hstack((views, panel)))
            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("o"):
                if left_ok and right_ok:
                    R, T, last_cost = refine_once(R, T, left_raw, right_raw)
                    e = mean_reprojection_error(R, T, left_raw, right_raw)
                    a = rotation_matrix_to_euler_xyz(R)
                    print(
                        f"[OPT] cost={last_cost:.4f}  "
                        f"RMS={e['rms']:.3f}px  "
                        f"Rx={a[0]:+.3f} Ry={a[1]:+.3f} Rz={a[2]:+.3f}  "
                        f"T=({T[0,0]:+.2f},{T[1,0]:+.2f},{T[2,0]:+.2f})  "
                        f"base={np.linalg.norm(T):.2f}"
                    )
                else:
                    print("[OPT] Need board in BOTH cameras")

            elif key == ord("a"):
                auto_mode = not auto_mode
                print(f"[AUTO] {'ON' if auto_mode else 'OFF'}")

            elif key == ord("s"):
                e = None
                if left_ok and right_ok:
                    e = mean_reprojection_error(R, T, left_raw, right_raw)
                save_extrinsics(
                    args.out,
                    R,
                    T,
                    extra={
                        "source_extrinsic_json": str(args.extrinsic_json),
                        "left_json": str(args.left_json),
                        "right_json": str(args.right_json),
                        "priors": {
                            "baseline_mm": PRIOR_BASELINE_MM,
                            "rx_deg": PRIOR_RX_DEG,
                            "ry_deg": PRIOR_RY_DEG,
                            "rz_deg": PRIOR_RZ_DEG,
                        },
                        "reprojection_error_px": e,
                    },
                )

            elif key == ord("r"):
                R = R_INIT.copy()
                T = T_INIT.copy()
                last_cost = None
                print("[RESET] Restored original extrinsics")

    finally:
        left_cam.stop()
        right_cam.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
