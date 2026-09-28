#!/usr/bin/env python3
"""
Compare original vs refined stereo extrinsics using corresponding
chessboard points from both cameras.

Live mode (default):
  python3 compare_extrinsics.py left.json right.json original_ext.json refined_ext.json

Offline mode (image pair):
  python3 compare_extrinsics.py left.json right.json original_ext.json refined_ext.json \\
      --left-image left.png --right-image right.png

Keys (live):
  S          save current metrics + corner coordinates to JSON
  SPACE      freeze / unfreeze frame pair
  Q / ESC    quit
"""

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

LEFT_DEVICE = "/dev/video1"
RIGHT_DEVICE = "/dev/video3"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 60

BOARD_COLS = 8
BOARD_ROWS = 5
SQUARE_SIZE_MM = 30.0

WINDOW_NAME = "Extrinsic Comparison (Original vs Refined)"

PANEL_WIDTH = 580
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
    description="Quantify original vs refined extrinsics on corresponding points."
)
parser.add_argument("left_json", help="Left camera intrinsic JSON")
parser.add_argument("right_json", help="Right camera intrinsic JSON")
parser.add_argument("original_extrinsic_json", help="Original extrinsic JSON")
parser.add_argument("refined_extrinsic_json", help="Refined extrinsic JSON")
parser.add_argument("--left-image", default=None, help="Optional left image (offline)")
parser.add_argument("--right-image", default=None, help="Optional right image (offline)")
parser.add_argument(
    "--out",
    default="comparison_result.json",
    help="Where to save metrics + points (default: comparison_result.json)",
)
args = parser.parse_args()

OFFLINE = args.left_image is not None and args.right_image is not None


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
# JSON / CALIBRATION
# ============================================================

def first_existing(data, keys):
    for k in keys:
        if k in data:
            return data[k]
    return None


def load_intrinsics(path, name):
    with open(path, "r") as f:
        data = json.load(f)
    K_raw = first_existing(data, ["camera_matrix", "K", "camera_matrix_left", "camera_matrix_right"])
    D_raw = first_existing(
        data,
        ["distortion_coefficients", "D", "distortion_coefficients_left", "distortion_coefficients_right"],
    )
    if K_raw is None or D_raw is None:
        raise KeyError(f"{path}: missing K or D")
    K = np.asarray(K_raw, dtype=np.float64)
    D = np.asarray(D_raw, dtype=np.float64).reshape(-1, 1)
    sq = float(data.get("square_size_mm", SQUARE_SIZE_MM))
    print(f"[{name}] K fx={K[0,0]:.2f} fy={K[1,1]:.2f}  square={sq} mm")
    return K, D, sq


def load_extrinsics(path, name):
    with open(path, "r") as f:
        data = json.load(f)
    R_raw = first_existing(
        data, ["rotation_left_to_right", "R_left_to_right", "rotation_matrix", "R"]
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
    if R_raw is None or T_raw is None:
        raise KeyError(f"{path}: missing R or T")
    R = np.asarray(R_raw, dtype=np.float64)
    T = np.asarray(T_raw, dtype=np.float64).reshape(3, 1)
    if R.shape == (3,):
        R, _ = cv2.Rodrigues(np.deg2rad(R).reshape(3, 1))
    if R.shape == (3, 1):
        R, _ = cv2.Rodrigues(R)
    if R.shape != (3, 3):
        raise ValueError(f"{path}: bad R shape")
    baseline = float(np.linalg.norm(T))
    print(f"[{name}] baseline={baseline:.3f} mm  T={T.ravel()}")
    return R, T, data


K_LEFT, D_LEFT, SQ = load_intrinsics(args.left_json, "LEFT")
K_RIGHT, D_RIGHT, _ = load_intrinsics(args.right_json, "RIGHT")
SQUARE_SIZE_MM = SQ

R_ORIG, T_ORIG, _ = load_extrinsics(args.original_extrinsic_json, "ORIGINAL")
R_REF, T_REF, _ = load_extrinsics(args.refined_extrinsic_json, "REFINED")


# ============================================================
# GEOMETRY
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


def undistort_image(frame, K, D):
    return cv2.undistort(frame, K, D, None, K)


def undistort_points(points, K, D):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)


def solve_board_pose(corners_raw, K, D):
    img = np.asarray(corners_raw, dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        OBJECT_POINTS, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None
    return rvec, tvec


def project_left_to_right(corners_left_raw, R_lr, T_lr):
    rvec, tvec = solve_board_pose(corners_left_raw, K_LEFT, D_LEFT)
    if rvec is None:
        return None
    R_b, _ = cv2.Rodrigues(rvec)
    pts_l = (R_b @ OBJECT_POINTS.T + tvec).T
    pts_r = (R_lr @ pts_l.T + T_lr).T
    proj, _ = cv2.projectPoints(
        pts_r.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K_RIGHT,
        np.zeros((5, 1)),
    )
    return proj.reshape(-1, 2)


def project_right_to_left(corners_right_raw, R_lr, T_lr):
    rvec, tvec = solve_board_pose(corners_right_raw, K_RIGHT, D_RIGHT)
    if rvec is None:
        return None
    R_b, _ = cv2.Rodrigues(rvec)
    pts_r = (R_b @ OBJECT_POINTS.T + tvec).T
    pts_l = (R_lr.T @ (pts_r.T - T_lr)).T
    proj, _ = cv2.projectPoints(
        pts_l.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K_LEFT,
        np.zeros((5, 1)),
    )
    return proj.reshape(-1, 2)


def error_stats(predicted, actual):
    if predicted is None or actual is None:
        return None
    if len(predicted) != len(actual):
        return None
    e = np.linalg.norm(predicted - actual, axis=1)
    return {
        "mean": float(np.mean(e)),
        "median": float(np.median(e)),
        "rms": float(np.sqrt(np.mean(e ** 2))),
        "max": float(np.max(e)),
        "per_point": e.tolist(),
    }


def bidirectional_error(R_lr, T_lr, left_raw, right_raw):
    actual_l = undistort_points(left_raw, K_LEFT, D_LEFT)
    actual_r = undistort_points(right_raw, K_RIGHT, D_RIGHT)
    pred_r = project_left_to_right(left_raw, R_lr, T_lr)
    pred_l = project_right_to_left(right_raw, R_lr, T_lr)
    e_lr = error_stats(pred_r, actual_r)
    e_rl = error_stats(pred_l, actual_l)
    # Combined
    errs = []
    if e_lr is not None:
        errs.extend(e_lr["per_point"])
    if e_rl is not None:
        errs.extend(e_rl["per_point"])
    if not errs:
        return None, None, None
    errs = np.asarray(errs)
    combined = {
        "mean": float(np.mean(errs)),
        "median": float(np.median(errs)),
        "rms": float(np.sqrt(np.mean(errs ** 2))),
        "max": float(np.max(errs)),
    }
    return combined, e_lr, e_rl, pred_l, pred_r, actual_l, actual_r


# ============================================================
# DRAWING
# ============================================================

def safe_point(pt):
    if not np.all(np.isfinite(pt)):
        return None
    x, y = float(pt[0]), float(pt[1])
    if abs(x) > 1e5 or abs(y) > 1e5:
        return None
    return (int(round(x)), int(round(y)))


def draw_points(img, points, filled=True, radius=5, color=None):
    if points is None:
        return
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    for i, pt in enumerate(pts):
        p = safe_point(pt)
        if p is None:
            continue
        c = color if color is not None else ROW_COLORS[(i // BOARD_COLS) % len(ROW_COLORS)]
        if filled:
            # Actual detected corners
            cv2.circle(img, p, radius + 2, BLACK, -1, cv2.LINE_AA)
            cv2.circle(img, p, radius, c, -1, cv2.LINE_AA)
        else:
            # Projected points — asterisk (*)
            x, y = p
            cv2.putText(
                img, "*", (x - 8, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 4, cv2.LINE_AA,
            )
            cv2.putText(
                img, "*", (x - 8, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA,
            )


def draw_connections(img, points):
    if points is None:
        return
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    for row in range(BOARD_ROWS):
        start = row * BOARD_COLS
        for i in range(start, start + BOARD_COLS - 1):
            p1, p2 = safe_point(pts[i]), safe_point(pts[i + 1])
            if p1 is None or p2 is None:
                continue
            cv2.line(img, p1, p2, ROW_COLORS[row % len(ROW_COLORS)], 2, cv2.LINE_AA)


def put_text(img, text, pos, scale=0.65, color=WHITE, thickness=2):
    x, y = map(int, pos)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 4, cv2.LINE_AA)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fit_to_screen(img, max_w=MAX_DISPLAY_WIDTH, max_h=MAX_DISPLAY_HEIGHT):
    h, w = img.shape[:2]
    s = min(1.0, max_w / float(w), max_h / float(h))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def draw_panel(panel, left_ok, right_ok, err_orig, err_ref, frozen):
    panel[:] = DARK
    put_text(panel, "ORIGINAL vs REFINED", (20, 40), scale=0.85, color=CYAN, thickness=3)
    cv2.line(panel, (20, 55), (PANEL_WIDTH - 20, 55), CYAN, 2)

    put_text(panel, "DETECTION", (20, 95), scale=0.70, color=YELLOW, thickness=2)
    put_text(
        panel,
        "Left  : DETECTED" if left_ok else "Left  : NOT DETECTED",
        (20, 125), scale=0.58, color=GREEN if left_ok else RED, thickness=2,
    )
    put_text(
        panel,
        "Right : DETECTED" if right_ok else "Right : NOT DETECTED",
        (20, 155), scale=0.58, color=GREEN if right_ok else RED, thickness=2,
    )
    if frozen:
        put_text(panel, "FROZEN", (20, 185), scale=0.58, color=ORANGE, thickness=2)

    # Original metrics
    put_text(panel, "ORIGINAL EXTRINSICS", (20, 230), scale=0.70, color=ORANGE, thickness=2)
    a0 = rotation_matrix_to_euler_xyz(R_ORIG)
    b0 = float(np.linalg.norm(T_ORIG))
    put_text(panel, f"Rx={a0[0]:+.2f}  Ry={a0[1]:+.2f}  Rz={a0[2]:+.2f}", (20, 260), scale=0.52, color=WHITE, thickness=1)
    put_text(panel, f"T=({T_ORIG[0,0]:+.1f},{T_ORIG[1,0]:+.1f},{T_ORIG[2,0]:+.1f})  base={b0:.1f}", (20, 285), scale=0.50, color=WHITE, thickness=1)

    if err_orig is None:
        put_text(panel, "Need both boards...", (20, 320), scale=0.55, color=YELLOW, thickness=2)
    else:
        put_text(panel, f"Mean   : {err_orig['mean']:.3f} px", (20, 320), scale=0.58, color=WHITE, thickness=2)
        put_text(panel, f"Median : {err_orig['median']:.3f} px", (20, 348), scale=0.58, color=WHITE, thickness=2)
        put_text(panel, f"RMS    : {err_orig['rms']:.3f} px", (20, 376), scale=0.58, color=RED, thickness=2)
        put_text(panel, f"Max    : {err_orig['max']:.3f} px", (20, 404), scale=0.58, color=WHITE, thickness=2)

    # Refined metrics
    put_text(panel, "REFINED EXTRINSICS", (20, 460), scale=0.70, color=GREEN, thickness=2)
    a1 = rotation_matrix_to_euler_xyz(R_REF)
    b1 = float(np.linalg.norm(T_REF))
    put_text(panel, f"Rx={a1[0]:+.2f}  Ry={a1[1]:+.2f}  Rz={a1[2]:+.2f}", (20, 490), scale=0.52, color=WHITE, thickness=1)
    put_text(panel, f"T=({T_REF[0,0]:+.1f},{T_REF[1,0]:+.1f},{T_REF[2,0]:+.1f})  base={b1:.1f}", (20, 515), scale=0.50, color=WHITE, thickness=1)

    if err_ref is None:
        put_text(panel, "Need both boards...", (20, 550), scale=0.55, color=YELLOW, thickness=2)
    else:
        put_text(panel, f"Mean   : {err_ref['mean']:.3f} px", (20, 550), scale=0.58, color=WHITE, thickness=2)
        put_text(panel, f"Median : {err_ref['median']:.3f} px", (20, 578), scale=0.58, color=WHITE, thickness=2)
        put_text(panel, f"RMS    : {err_ref['rms']:.3f} px", (20, 606), scale=0.58, color=GREEN, thickness=2)
        put_text(panel, f"Max    : {err_ref['max']:.3f} px", (20, 634), scale=0.58, color=WHITE, thickness=2)

    # Improvement
    if err_orig is not None and err_ref is not None:
        put_text(panel, "IMPROVEMENT", (20, 690), scale=0.70, color=CYAN, thickness=2)
        d_rms = err_orig["rms"] - err_ref["rms"]
        d_mean = err_orig["mean"] - err_ref["mean"]
        ratio = err_orig["rms"] / max(err_ref["rms"], 1e-9)
        put_text(panel, f"RMS  drop : {d_rms:+.3f} px", (20, 725), scale=0.58, color=GREEN if d_rms > 0 else RED, thickness=2)
        put_text(panel, f"Mean drop : {d_mean:+.3f} px", (20, 753), scale=0.58, color=GREEN if d_mean > 0 else RED, thickness=2)
        put_text(panel, f"RMS ratio : {ratio:.2f}x better", (20, 781), scale=0.58, color=YELLOW, thickness=2)

    # Layout guide
    put_text(panel, "2x2 LAYOUT", (20, 840), scale=0.70, color=ORANGE, thickness=2)
    put_text(panel, "TL: R→L ORIGINAL (magenta *)", (20, 870), scale=0.50, color=MAGENTA, thickness=2)
    put_text(panel, "TR: R→L REFINED  (blue *)", (20, 895), scale=0.50, color=BLUE, thickness=2)
    put_text(panel, "BL: L→R ORIGINAL (magenta *)", (20, 920), scale=0.50, color=MAGENTA, thickness=2)
    put_text(panel, "BR: L→R REFINED  (blue *)", (20, 945), scale=0.50, color=BLUE, thickness=2)
    put_text(panel, "Filled dots = real corners", (20, 975), scale=0.50, color=WHITE, thickness=2)
    put_text(panel, "* = projected points", (20, 1000), scale=0.50, color=WHITE, thickness=2)

    put_text(panel, "S : save metrics + points", (20, 1020), scale=0.55, color=CYAN, thickness=2)
    put_text(panel, "SPACE : freeze / unfreeze", (20, 1048), scale=0.55, color=WHITE, thickness=2)
    put_text(panel, "Q / ESC : quit", (20, 1076), scale=0.55, color=RED, thickness=2)


def save_comparison(path, left_raw, right_raw, err_orig, err_ref, pred_l_o, pred_r_o, pred_l_r, pred_r_r, actual_l, actual_r):
    payload = {
        "left_corners_raw": np.asarray(left_raw).tolist(),
        "right_corners_raw": np.asarray(right_raw).tolist(),
        "left_corners_undistorted": np.asarray(actual_l).tolist() if actual_l is not None else None,
        "right_corners_undistorted": np.asarray(actual_r).tolist() if actual_r is not None else None,
        "original": {
            "error_combined": err_orig,
            "projected_on_left": np.asarray(pred_l_o).tolist() if pred_l_o is not None else None,
            "projected_on_right": np.asarray(pred_r_o).tolist() if pred_r_o is not None else None,
            "euler_xyz_deg": rotation_matrix_to_euler_xyz(R_ORIG).tolist(),
            "T_mm": T_ORIG.ravel().tolist(),
            "baseline_mm": float(np.linalg.norm(T_ORIG)),
        },
        "refined": {
            "error_combined": err_ref,
            "projected_on_left": np.asarray(pred_l_r).tolist() if pred_l_r is not None else None,
            "projected_on_right": np.asarray(pred_r_r).tolist() if pred_r_r is not None else None,
            "euler_xyz_deg": rotation_matrix_to_euler_xyz(R_REF).tolist(),
            "T_mm": T_REF.ravel().tolist(),
            "baseline_mm": float(np.linalg.norm(T_REF)),
        },
    }
    if err_orig is not None and err_ref is not None:
        payload["improvement"] = {
            "rms_drop_px": err_orig["rms"] - err_ref["rms"],
            "mean_drop_px": err_orig["mean"] - err_ref["mean"],
            "rms_ratio": err_orig["rms"] / max(err_ref["rms"], 1e-9),
        }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVE] {path}")
    if err_orig and err_ref:
        print(f"       Original RMS={err_orig['rms']:.3f}  Refined RMS={err_ref['rms']:.3f}  "
              f"drop={err_orig['rms']-err_ref['rms']:+.3f} px")


# ============================================================
# MAIN
# ============================================================

def process_pair(frame_l, frame_r):
    """Returns everything needed for display + metrics."""
    undist_l = undistort_image(frame_l, K_LEFT, D_LEFT)
    undist_r = undistort_image(frame_r, K_RIGHT, D_RIGHT)

    left_ok, left_raw = find_chessboard(frame_l)
    right_ok, right_raw = find_chessboard(frame_r)

    err_orig = err_ref = None
    pred_l_o = pred_r_o = pred_l_r = pred_r_r = None
    actual_l = actual_r = None

    if left_ok and right_ok:
        (err_orig, _, _, pred_l_o, pred_r_o, actual_l, actual_r) = bidirectional_error(
            R_ORIG, T_ORIG, left_raw, right_raw
        )
        (err_ref, _, _, pred_l_r, pred_r_r, _, _) = bidirectional_error(
            R_REF, T_REF, left_raw, right_raw
        )

    return {
        "undist_l": undist_l,
        "undist_r": undist_r,
        "left_ok": left_ok,
        "right_ok": right_ok,
        "left_raw": left_raw,
        "right_raw": right_raw,
        "actual_l": actual_l,
        "actual_r": actual_r,
        "err_orig": err_orig,
        "err_ref": err_ref,
        "pred_l_o": pred_l_o,
        "pred_r_o": pred_r_o,
        "pred_l_r": pred_l_r,
        "pred_r_r": pred_r_r,
    }


def _make_view(base_img, actual_pts, projected_pts, title, title_color, proj_color, proj_label):
    """One panel: actual corners + one projection method (no overlap of methods)."""
    img = base_img.copy()
    if actual_pts is not None:
        draw_connections(img, actual_pts)
        draw_points(img, actual_pts, filled=True, radius=5)
    if projected_pts is not None:
        draw_points(img, projected_pts, filled=False, radius=6, color=proj_color)
    put_text(img, title, (16, 36), scale=0.70, color=title_color, thickness=2)
    put_text(img, proj_label, (16, 68), scale=0.55, color=proj_color, thickness=2)
    return img


def render(state, frozen):
    """
    2x2 grid — no overlapping original/refined markers:

      [0,0] LEFT  + RIGHT→LEFT  ORIGINAL (magenta)
      [0,1] LEFT  + RIGHT→LEFT  REFINED  (blue)
      [1,0] RIGHT + LEFT→RIGHT  ORIGINAL (magenta)
      [1,1] RIGHT + LEFT→RIGHT  REFINED  (blue)

    Side panel: quantitative comparison.
    """
    ul = state["undist_l"]
    ur = state["undist_r"]
    al = state["actual_l"]
    ar = state["actual_r"]

    # Top row: projections onto LEFT (mapped from right)
    top_left = _make_view(
        ul, al, state["pred_l_o"],
        "LEFT camera", CYAN,
        MAGENTA, "R→L  ORIGINAL",
    )
    top_right = _make_view(
        ul, al, state["pred_l_r"],
        "LEFT camera", CYAN,
        BLUE, "R→L  REFINED",
    )

    # Bottom row: projections onto RIGHT (mapped from left)
    bot_left = _make_view(
        ur, ar, state["pred_r_o"],
        "RIGHT camera", ORANGE,
        MAGENTA, "L→R  ORIGINAL",
    )
    bot_right = _make_view(
        ur, ar, state["pred_r_r"],
        "RIGHT camera", ORANGE,
        BLUE, "L→R  REFINED",
    )

    top = np.hstack((top_left, top_right))
    bot = np.hstack((bot_left, bot_right))
    grid = np.vstack((top, bot))

    panel = np.zeros((grid.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
    draw_panel(panel, state["left_ok"], state["right_ok"], state["err_orig"], state["err_ref"], frozen)

    return fit_to_screen(np.hstack((grid, panel)))


def main():
    if OFFLINE:
        frame_l = cv2.imread(args.left_image)
        frame_r = cv2.imread(args.right_image)
        if frame_l is None or frame_r is None:
            print("ERROR: could not read image pair")
            return 1
        state = process_pair(frame_l, frame_r)
        combined = render(state, frozen=True)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, combined)

        if state["left_ok"] and state["right_ok"]:
            save_comparison(
                args.out,
                state["left_raw"],
                state["right_raw"],
                state["err_orig"],
                state["err_ref"],
                state["pred_l_o"],
                state["pred_r_o"],
                state["pred_l_r"],
                state["pred_r_r"],
                state["actual_l"],
                state["actual_r"],
            )
            print()
            print("=" * 60)
            print(f"ORIGINAL  RMS = {state['err_orig']['rms']:.3f} px")
            print(f"REFINED   RMS = {state['err_ref']['rms']:.3f} px")
            print(f"IMPROVEMENT   = {state['err_orig']['rms'] - state['err_ref']['rms']:+.3f} px")
            print("=" * 60)

        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    # Live mode
    left_cam = CameraCapture(LEFT_DEVICE, IMAGE_WIDTH, IMAGE_HEIGHT, FPS)
    right_cam = CameraCapture(RIGHT_DEVICE, IMAGE_WIDTH, IMAGE_HEIGHT, FPS)
    if not left_cam.start():
        return 1
    if not right_cam.start():
        left_cam.stop()
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    frozen = False
    frozen_state = None

    print()
    print("=" * 72)
    print("CONTROLS")
    print("=" * 72)
    print("S     : save metrics + corresponding points to JSON")
    print("SPACE : freeze / unfreeze current pair")
    print("Q/ESC : quit")
    print("=" * 72)
    print()

    try:
        while True:
            if not frozen:
                fl = left_cam.get_frame()
                fr = right_cam.get_frame()
                if fl is None or fr is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    continue
                state = process_pair(fl, fr)
            else:
                state = frozen_state

            combined = render(state, frozen)
            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if frozen:
                    frozen = False
                    frozen_state = None
                    print("[LIVE]")
                else:
                    frozen = True
                    frozen_state = state
                    print("[FROZEN]")
            elif key == ord("s"):
                if state["left_ok"] and state["right_ok"]:
                    save_comparison(
                        args.out,
                        state["left_raw"],
                        state["right_raw"],
                        state["err_orig"],
                        state["err_ref"],
                        state["pred_l_o"],
                        state["pred_r_o"],
                        state["pred_l_r"],
                        state["pred_r_r"],
                        state["actual_l"],
                        state["actual_r"],
                    )
                else:
                    print("[SAVE] Need board in both cameras")
    finally:
        left_cam.stop()
        right_cam.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
