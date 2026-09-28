#!/usr/bin/env python3
"""
Live stereo extrinsic refinement using an AprilTag.

Same idea as refine_stereo_extrinsics.py, but instead of a chessboard the
correspondences come from a single AprilTag visible in both cameras. The
tag gives 4 corner points per frame; each is a left↔right correspondence
plus a 3D reference (its own 4 tag-frame corners), so solvePnP still
works and the R, T refinement is identical in spirit.

Prior knowledge encoded as soft constraints:
  - Baseline length ≈ 45 mm
  - Rx ≈ 20 deg
  - Ry ≈ 0, Rz ≈ 0

Usage:
  # CLI:
  python3 refine_stereo_extrinsics_apriltag.py \\
      left.json right.json extrinsics.json --tag-size 60

  # No args -> file dialogs pop up and a prompt asks for --tag-size.

Keys:
  O          run one optimization step
  A          auto-refine continuously while tag is visible
  S          save refined extrinsics to refined_extrinsics_apriltag.json
  R          reset to original extrinsics
  Q / ESC    quit
"""

import argparse
import json
import threading
import time

import _opencv_cuda  # noqa: F401  (must import before cv2)
import cv2
import numpy as np
from scipy.optimize import least_squares


# ============================================================
# CONFIGURATION
# ============================================================

LEFT_DEVICE = "/dev/video1"       # BORESIGHT / LEFT
RIGHT_DEVICE = "/dev/video3"      # DEPRESSION / RIGHT

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 60

# Soft priors
PRIOR_BASELINE_MM = 45.0
PRIOR_RX_DEG = 20.0
PRIOR_RY_DEG = 0.0
PRIOR_RZ_DEG = 0.0

W_BASELINE = 2.0
W_RX = 0.5
W_RY = 1.0
W_RZ = 1.0

WINDOW_NAME = "Stereo Extrinsic Refinement (AprilTag)"

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
BLACK = (0, 0, 0)
DARK = (15, 15, 15)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Live refine stereo extrinsics from dual AprilTag detections."
)
parser.add_argument("left_json", help="Left intrinsic JSON")
parser.add_argument("right_json", help="Right intrinsic JSON")
parser.add_argument("extrinsic_json", help="Initial stereo extrinsic JSON")
parser.add_argument("--tag-size", type=float, required=True,
                    help="AprilTag side length in mm (black square)")
parser.add_argument("--tag-family", default="36h11",
                    choices=["16h5", "25h9", "36h10", "36h11"],
                    help="AprilTag family (default: 36h11)")
parser.add_argument("--out", default="refined_extrinsics_apriltag.json",
                    help="Output path (default: refined_extrinsics_apriltag.json)")
parser.add_argument("--baseline", type=float, default=PRIOR_BASELINE_MM,
                    help=f"Prior baseline in mm (default: {PRIOR_BASELINE_MM})")
parser.add_argument("--rx", type=float, default=PRIOR_RX_DEG,
                    help=f"Prior Rx in degrees (default: {PRIOR_RX_DEG})")

from _argpick import parse_or_pick
args = parse_or_pick(
    parser,
    [
        ("left_json",      "Left intrinsic JSON",  [("JSON", "*.json")]),
        ("right_json",     "Right intrinsic JSON", [("JSON", "*.json")]),
        ("extrinsic_json", "Initial stereo extrinsic JSON", [("JSON", "*.json")]),
    ],
    ask_missing_options=[("tag-size", "AprilTag side length in mm (black square)")],
)

PRIOR_BASELINE_MM = args.baseline
PRIOR_RX_DEG = args.rx
TAG_SIZE_MM = float(args.tag_size)


# ============================================================
# CAMERA CAPTURE
# ============================================================

class CameraCapture:
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
            return None if self.frame is None else self.frame.copy()

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
    for k in keys:
        if k in data:
            return data[k]
    return None


def load_camera_calibration(path, name):
    with open(path, "r") as f:
        data = json.load(f)
    K_raw = first_existing(data, ["camera_matrix", "camera_matrix_left",
                                  "camera_matrix_right", "K"])
    D_raw = first_existing(data, ["distortion_coefficients",
                                  "distortion_coefficients_left",
                                  "distortion_coefficients_right", "D"])
    if K_raw is None or D_raw is None:
        raise KeyError(f"{path}: missing K or D")
    K = np.asarray(K_raw, dtype=np.float64)
    D = np.asarray(D_raw, dtype=np.float64).reshape(-1, 1)
    print(f"\n=== {name} ===\nK=\n{K}\nD={D.ravel()}")
    return K, D


def load_extrinsic_calibration(path):
    with open(path, "r") as f:
        data = json.load(f)
    R_raw = first_existing(data, ["rotation_left_to_right", "R_left_to_right",
                                  "rotation_matrix", "R"])
    T_raw = first_existing(data, ["translation_left_to_right_mm",
                                  "T_left_to_right_mm", "translation_mm",
                                  "translation_left_to_right", "T",
                                  "translation"])
    if R_raw is None or T_raw is None:
        raise KeyError(f"{path}: missing R or T")
    R = np.asarray(R_raw, dtype=np.float64)
    T = np.asarray(T_raw, dtype=np.float64).reshape(3, 1)
    if R.shape == (3,):
        R, _ = cv2.Rodrigues(np.deg2rad(R).reshape(3, 1))
    if R.shape == (3, 1):
        R, _ = cv2.Rodrigues(R)
    print(f"\n=== INITIAL EXTRINSICS ===\nR=\n{R}\nT[mm]={T.ravel()}   "
          f"baseline={float(np.linalg.norm(T)):.3f} mm")
    return R, T


def save_extrinsics(path, R, T, extra=None):
    payload = {
        "rotation_left_to_right": R.tolist(),
        "translation_left_to_right_mm": T.reshape(3, 1).tolist(),
        "baseline_mm": float(np.linalg.norm(T)),
        "euler_xyz_deg": rotation_matrix_to_euler_xyz(R).tolist(),
        "source": "apriltag_correspondences",
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVE] Wrote refined extrinsics → {path}")


K_LEFT, D_LEFT = load_camera_calibration(args.left_json, "LEFT")
K_RIGHT, D_RIGHT = load_camera_calibration(args.right_json, "RIGHT")
R_INIT, T_INIT = load_extrinsic_calibration(args.extrinsic_json)
R = R_INIT.copy()
T = T_INIT.copy()


# ============================================================
# APRILTAG SETUP
# ============================================================

FAMILY_MAP = {
    "16h5":  cv2.aruco.DICT_APRILTAG_16h5,
    "25h9":  cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}
aruco_dict = cv2.aruco.getPredefinedDictionary(FAMILY_MAP[args.tag_family])
try:
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    def _detect(gray):
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids
except AttributeError:
    detector_params = cv2.aruco.DetectorParameters_create()
    def _detect(gray):
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params)
        return corners, ids


# Tag 3D corner coordinates (in tag frame, z=0), order matches cv2.aruco:
# TL, TR, BR, BL — as seen from the tag's viewer side.
half = TAG_SIZE_MM / 2.0
TAG_OBJECT_POINTS = np.array([
    [-half,  half, 0.0],
    [ half,  half, 0.0],
    [ half, -half, 0.0],
    [-half, -half, 0.0],
], dtype=np.float64)


def find_tag(frame):
    """Return (found, corners_2d) — 4x2 pixel corners of the first tag."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids = _detect(gray)
    if ids is None or len(ids) == 0:
        return False, None
    return True, corners[0].reshape(-1, 2).astype(np.float64)


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def rotation_matrix_to_euler_xyz(Rm):
    sy = np.sqrt(Rm[0, 0] ** 2 + Rm[1, 0] ** 2)
    if sy > 1e-8:
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
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pack_params(R, T):
    e = rotation_matrix_to_euler_xyz(R)
    return np.array([e[0], e[1], e[2], T[0, 0], T[1, 0], T[2, 0]])


def unpack_params(x):
    R = euler_xyz_to_rotation_matrix(x[0], x[1], x[2])
    T = np.array([[x[3]], [x[4]], [x[5]]])
    return R, T


def solve_tag_pose(corners_2d, K, D):
    ok, rvec, tvec = cv2.solvePnP(
        TAG_OBJECT_POINTS, corners_2d.reshape(-1, 1, 2),
        K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return (rvec, tvec) if ok else (None, None)


def project_left_to_right(corners_left, R_lr, T_lr):
    rvec, tvec = solve_tag_pose(corners_left, K_LEFT, D_LEFT)
    if rvec is None:
        return None
    R_tag, _ = cv2.Rodrigues(rvec)
    pts_left = (R_tag @ TAG_OBJECT_POINTS.T + tvec).T
    pts_right = (R_lr @ pts_left.T + T_lr).T
    proj, _ = cv2.projectPoints(
        pts_right.reshape(-1, 1, 3),
        np.zeros((3, 1)), np.zeros((3, 1)),
        K_RIGHT, np.zeros((5, 1)),
    )
    return proj.reshape(-1, 2)


def project_right_to_left(corners_right, R_lr, T_lr):
    rvec, tvec = solve_tag_pose(corners_right, K_RIGHT, D_RIGHT)
    if rvec is None:
        return None
    R_tag, _ = cv2.Rodrigues(rvec)
    pts_right = (R_tag @ TAG_OBJECT_POINTS.T + tvec).T
    pts_left = (R_lr.T @ (pts_right.T - T_lr)).T
    proj, _ = cv2.projectPoints(
        pts_left.reshape(-1, 1, 3),
        np.zeros((3, 1)), np.zeros((3, 1)),
        K_LEFT, np.zeros((5, 1)),
    )
    return proj.reshape(-1, 2)


def undistort_image(frame, K, D):
    return cv2.undistort(frame, K, D, None, K)


def undistort_points(pts, K, D):
    return cv2.undistortPoints(
        np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2),
        K, D, P=K,
    ).reshape(-1, 2)


# ============================================================
# COST / OPTIMIZATION
# ============================================================

def reprojection_residuals(x, left_c, right_c):
    R_lr, T_lr = unpack_params(x)
    parts = []
    pred_r = project_left_to_right(left_c, R_lr, T_lr)
    if pred_r is not None:
        actual_r = undistort_points(right_c, K_RIGHT, D_RIGHT)
        parts.append((pred_r - actual_r).ravel())
    pred_l = project_right_to_left(right_c, R_lr, T_lr)
    if pred_l is not None:
        actual_l = undistort_points(left_c, K_LEFT, D_LEFT)
        parts.append((pred_l - actual_l).ravel())
    if not parts:
        return np.zeros(6)
    pix = np.concatenate(parts)
    priors = np.array([
        W_BASELINE * (float(np.linalg.norm(T_lr)) - PRIOR_BASELINE_MM),
        W_RX * (x[0] - PRIOR_RX_DEG),
        W_RY * (x[1] - PRIOR_RY_DEG),
        W_RZ * (x[2] - PRIOR_RZ_DEG),
    ])
    return np.concatenate([pix, priors])


def mean_reprojection_error(R_lr, T_lr, left_c, right_c):
    errs = []
    pred_r = project_left_to_right(left_c, R_lr, T_lr)
    if pred_r is not None:
        actual_r = undistort_points(right_c, K_RIGHT, D_RIGHT)
        errs.append(np.linalg.norm(pred_r - actual_r, axis=1))
    pred_l = project_right_to_left(right_c, R_lr, T_lr)
    if pred_l is not None:
        actual_l = undistort_points(left_c, K_LEFT, D_LEFT)
        errs.append(np.linalg.norm(pred_l - actual_l, axis=1))
    if not errs:
        return None
    all_e = np.concatenate(errs)
    return {
        "mean":   float(np.mean(all_e)),
        "median": float(np.median(all_e)),
        "rms":    float(np.sqrt(np.mean(all_e ** 2))),
        "max":    float(np.max(all_e)),
    }


def refine_once(R_lr, T_lr, left_c, right_c):
    x0 = pack_params(R_lr, T_lr)
    res = least_squares(reprojection_residuals, x0,
                        args=(left_c, right_c),
                        method="lm", max_nfev=80, verbose=0)
    R_new, T_new = unpack_params(res.x)
    return R_new, T_new, res.cost


# ============================================================
# DRAWING
# ============================================================

def safe_point(p):
    if not np.all(np.isfinite(p)):
        return None
    x, y = float(p[0]), float(p[1])
    if abs(x) > 1e5 or abs(y) > 1e5:
        return None
    return int(round(x)), int(round(y))


def draw_tag(image, corners_2d, color=GREEN, filled=True):
    if corners_2d is None:
        return
    pts = [safe_point(c) for c in corners_2d]
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        if a and b:
            cv2.line(image, a, b, color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if p is None:
            continue
        r = 6
        if filled:
            cv2.circle(image, p, r + 2, BLACK, -1, cv2.LINE_AA)
            cv2.circle(image, p, r, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(image, p, r + 1, BLACK, -1, cv2.LINE_AA)
            cv2.circle(image, p, r, color, 2, cv2.LINE_AA)
        cv2.putText(image, str(i), (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def put_text(img, text, pos, scale=0.65, color=WHITE, thickness=2):
    x, y = map(int, pos)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thickness + 4, cv2.LINE_AA)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def fit_to_screen(img, max_w=MAX_DISPLAY_WIDTH, max_h=MAX_DISPLAY_HEIGHT):
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def draw_panel(panel, left_ok, right_ok, err, auto_mode, last_cost):
    panel[:] = DARK
    put_text(panel, "APRILTAG STEREO REFINE", (20, 40), scale=0.85, color=CYAN, thickness=3)
    cv2.line(panel, (20, 55), (PANEL_WIDTH - 20, 55), CYAN, 2)

    put_text(panel, f"Tag: {args.tag_family}  size={TAG_SIZE_MM:.1f} mm",
             (20, 90), scale=0.55, color=YELLOW, thickness=2)

    put_text(panel, "Left  : " + ("DETECTED" if left_ok else "NOT DETECTED"),
             (20, 130), scale=0.60, color=GREEN if left_ok else RED)
    put_text(panel, "Right : " + ("DETECTED" if right_ok else "NOT DETECTED"),
             (20, 160), scale=0.60, color=GREEN if right_ok else RED)

    ang = rotation_matrix_to_euler_xyz(R)
    baseline = float(np.linalg.norm(T))

    put_text(panel, "CURRENT R / T", (20, 205), scale=0.65, color=ORANGE)
    put_text(panel, f"Rx : {ang[0]:+.3f} deg  (prior {PRIOR_RX_DEG:.1f})", (20, 235), scale=0.55, color=GREEN)
    put_text(panel, f"Ry : {ang[1]:+.3f} deg  (prior {PRIOR_RY_DEG:.1f})", (20, 262), scale=0.55)
    put_text(panel, f"Rz : {ang[2]:+.3f} deg  (prior {PRIOR_RZ_DEG:.1f})", (20, 289), scale=0.55)
    put_text(panel, f"Tx : {T[0,0]:+.3f} mm", (20, 320), scale=0.55)
    put_text(panel, f"Ty : {T[1,0]:+.3f} mm", (20, 347), scale=0.55, color=YELLOW)
    put_text(panel, f"Tz : {T[2,0]:+.3f} mm", (20, 374), scale=0.55)
    put_text(panel, f"Baseline : {baseline:.3f} mm  (prior {PRIOR_BASELINE_MM:.1f})",
             (20, 405), scale=0.55, color=CYAN)

    put_text(panel, "REPROJECTION ERROR", (20, 450), scale=0.65, color=ORANGE)
    if err is None:
        put_text(panel, "Need tag in BOTH cameras", (20, 480), scale=0.55, color=YELLOW)
    else:
        put_text(panel, f"Mean   : {err['mean']:.3f} px", (20, 480), scale=0.55)
        put_text(panel, f"Median : {err['median']:.3f} px", (20, 507), scale=0.55)
        put_text(panel, f"RMS    : {err['rms']:.3f} px", (20, 534), scale=0.55, color=GREEN)
        put_text(panel, f"Max    : {err['max']:.3f} px", (20, 561), scale=0.55)
    if last_cost is not None:
        put_text(panel, f"Last LS cost : {last_cost:.4f}", (20, 595), scale=0.55, color=CYAN)

    cv2.line(panel, (20, 630), (PANEL_WIDTH - 20, 630), CYAN, 2)
    put_text(panel, "CONTROLS", (20, 665), scale=0.65, color=YELLOW)
    put_text(panel, "O : one optimization step", (20, 695), scale=0.55)
    put_text(panel, f"A : auto-refine  [{'ON' if auto_mode else 'OFF'}]",
             (20, 722), scale=0.55, color=GREEN if auto_mode else WHITE)
    put_text(panel, f"S : save {args.out}", (20, 749), scale=0.50, color=CYAN)
    put_text(panel, "R : reset to original", (20, 776), scale=0.55, color=ORANGE)
    put_text(panel, "Q / ESC : quit", (20, 803), scale=0.55, color=RED)

    put_text(panel, "Filled dots  = detected corners", (20, 850), scale=0.50)
    put_text(panel, "Blue rings   = projected via R/T", (20, 875), scale=0.50, color=BLUE)


# ============================================================
# MAIN
# ============================================================

def main():
    global R, T

    left_cam = CameraCapture(LEFT_DEVICE)
    right_cam = CameraCapture(RIGHT_DEVICE)
    if not left_cam.start():
        return 1
    if not right_cam.start():
        left_cam.stop()
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    auto_mode = False
    last_cost = None

    print("\nControls: O=optimize, A=auto, S=save, R=reset, Q=quit")
    print(f"Priors: baseline={PRIOR_BASELINE_MM} mm, Rx={PRIOR_RX_DEG} deg\n")

    try:
        while True:
            fl = left_cam.get_frame()
            fr = right_cam.get_frame()
            if fl is None or fr is None:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            und_l = undistort_image(fl, K_LEFT, D_LEFT)
            und_r = undistort_image(fr, K_RIGHT, D_RIGHT)

            l_ok, l_raw = find_tag(fl)
            r_ok, r_raw = find_tag(fr)

            l_un = undistort_points(l_raw, K_LEFT, D_LEFT) if l_ok else None
            r_un = undistort_points(r_raw, K_RIGHT, D_RIGHT) if r_ok else None

            pred_r = pred_l = None
            err = None
            if l_ok and r_ok:
                pred_r = project_left_to_right(l_raw, R, T)
                pred_l = project_right_to_left(r_raw, R, T)
                err = mean_reprojection_error(R, T, l_raw, r_raw)
                if auto_mode:
                    R, T, last_cost = refine_once(R, T, l_raw, r_raw)
                    err = mean_reprojection_error(R, T, l_raw, r_raw)

            disp_l = und_l.copy()
            disp_r = und_r.copy()
            if l_ok:
                draw_tag(disp_l, l_un, GREEN, filled=True)
            if r_ok:
                draw_tag(disp_r, r_un, GREEN, filled=True)
            if pred_r is not None:
                draw_tag(disp_r, pred_r, BLUE, filled=False)
            if pred_l is not None:
                draw_tag(disp_l, pred_l, BLUE, filled=False)

            put_text(disp_l, "LEFT / BORESIGHT", (20, 40), scale=0.80, color=CYAN, thickness=3)
            put_text(disp_r, "RIGHT / DEPRESSION", (20, 40), scale=0.80, color=ORANGE, thickness=3)

            views = np.vstack((disp_l, disp_r))
            panel = np.zeros((views.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
            draw_panel(panel, l_ok, r_ok, err, auto_mode, last_cost)

            cv2.imshow(WINDOW_NAME, fit_to_screen(np.hstack((views, panel))))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("o"):
                if l_ok and r_ok:
                    R, T, last_cost = refine_once(R, T, l_raw, r_raw)
                    e = mean_reprojection_error(R, T, l_raw, r_raw)
                    a = rotation_matrix_to_euler_xyz(R)
                    print(f"[OPT] cost={last_cost:.4f}  RMS={e['rms']:.3f}px  "
                          f"Rx={a[0]:+.3f} Ry={a[1]:+.3f} Rz={a[2]:+.3f}  "
                          f"T=({T[0,0]:+.2f},{T[1,0]:+.2f},{T[2,0]:+.2f})  "
                          f"base={np.linalg.norm(T):.2f}")
                else:
                    print("[OPT] Need tag in BOTH cameras")
            elif key == ord("a"):
                auto_mode = not auto_mode
                print(f"[AUTO] {'ON' if auto_mode else 'OFF'}")
            elif key == ord("s"):
                e = None
                if l_ok and r_ok:
                    e = mean_reprojection_error(R, T, l_raw, r_raw)
                save_extrinsics(args.out, R, T, extra={
                    "source_extrinsic_json": str(args.extrinsic_json),
                    "left_json": str(args.left_json),
                    "right_json": str(args.right_json),
                    "tag_family": args.tag_family,
                    "tag_size_mm": TAG_SIZE_MM,
                    "priors": {
                        "baseline_mm": PRIOR_BASELINE_MM,
                        "rx_deg": PRIOR_RX_DEG,
                        "ry_deg": PRIOR_RY_DEG,
                        "rz_deg": PRIOR_RZ_DEG,
                    },
                    "reprojection_error_px": e,
                })
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
