#!/usr/bin/env python3
"""
Stereo AprilTag pose logger — poses expressed in the LEFT camera frame.

Given:
  - LEFT camera intrinsic JSON  (K_L, D_L)
  - RIGHT camera intrinsic JSON (K_R, D_R)
  - Stereo extrinsic JSON       (R, T  left→right)

For every AprilTag visible in BOTH cameras it computes the tag pose
THREE independent ways, all expressed in the LEFT camera frame:

  1. pnp_left        : solvePnP on left image (direct).
  2. pnp_right_xf    : solvePnP on right image, transformed into left frame
                       using the stereo extrinsic: X_L = R^T (X_R - T).
  3. triangulated    : cv2.triangulatePoints on the 4 tag corners in both
                       cameras, then solvePnP on the 3D points to recover
                       a rigid tag→camera transform.

Agreement between these three is the sanity check. If they diverge more
than ~1–2 cm in position or ~2° in orientation, one of the calibrations
is off (intrinsics, extrinsics, or tag_size).

Keys
    S    snapshot current frame's stereo detections to the JSON log
    A    auto-log every frame with ≥1 matched tag (toggle)
    W    write log to --out and clear
    Q    quit (also flushes)

CLI
    python3 apriltag_stereo_pose_logger.py \\
        left.json right.json stereo_extrinsic.json --tag-size 60

    (or no args → file dialogs + tag-size prompt)
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


def transform_pose_right_to_left(R_r, t_r):
    """Given tag pose (R_r, t_r) in right camera, express it in left camera."""
    R_l = R_stereo.T @ R_r
    t_l = R_stereo.T @ (t_r - T_stereo)
    return R_l, t_l


def triangulate_corners(corners_L, corners_R):
    """Undistort corners then triangulate 4 3D points in left camera frame."""
    und_L = cv2.undistortPoints(corners_L.reshape(-1, 1, 2), K_L, D_L, P=K_L).reshape(-1, 2)
    und_R = cv2.undistortPoints(corners_R.reshape(-1, 1, 2), K_R, D_R, P=K_R).reshape(-1, 2)
    pts4d = cv2.triangulatePoints(P_L, P_R, und_L.T, und_R.T)  # 4x4
    pts3d = (pts4d[:3] / pts4d[3]).T                            # (4,3)
    return pts3d


def rigid_fit(src, dst):
    """
    Umeyama-style rigid fit: find R, t such that dst ≈ R * src + t.
    Both are (N,3) arrays.
    """
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    S = src - src_c
    D = dst - dst_c
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    Rf = Vt.T @ U.T
    if np.linalg.det(Rf) < 0:
        Vt[-1] *= -1
        Rf = Vt.T @ U.T
    tf = dst_c - Rf @ src_c
    return Rf, tf.reshape(3, 1)


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


def rot_angle_diff(R1, R2):
    """Angle in degrees between two rotation matrices."""
    Rrel = R1.T @ R2
    cos_ang = (np.trace(Rrel) - 1.0) / 2.0
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


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

log_frames = []
frame_seq = 0


def compute_stereo_pose_entry(tag_id, corners_L, corners_R):
    """Compute the three pose estimates + disagreement metrics."""
    # 1. PnP in left camera
    R_L,  t_L  = pnp(corners_L, K_L, D_L)
    # 2. PnP in right camera, transformed to left
    R_Rr, t_Rr = pnp(corners_R, K_R, D_R)
    R_Rxf, t_Rxf = (transform_pose_right_to_left(R_Rr, t_Rr)
                    if R_Rr is not None else (None, None))
    # 3. Triangulate corners → rigid fit to tag object points
    pts3d = triangulate_corners(corners_L, corners_R)
    R_tri, t_tri = rigid_fit(TAG_OBJ, pts3d)

    def pose_block(R, t):
        if R is None:
            return None
        T4 = np.eye(4)
        T4[:3, :3] = R
        T4[:3, 3]  = t.ravel()
        return {
            "tvec_mm":         t.ravel().tolist(),
            "R":               R.tolist(),
            "euler_xyz_deg":   R_to_euler_xyz(R),
            "distance_mm":     float(np.linalg.norm(t)),
            "T_tag_to_left_4x4": T4.tolist(),
        }

    disagreements = {}
    if R_L is not None and R_Rxf is not None:
        disagreements["pnp_left_vs_pnp_right_xf"] = {
            "translation_mm": float(np.linalg.norm(t_L - t_Rxf)),
            "rotation_deg":   rot_angle_diff(R_L, R_Rxf),
        }
    if R_L is not None:
        disagreements["pnp_left_vs_triangulated"] = {
            "translation_mm": float(np.linalg.norm(t_L - t_tri)),
            "rotation_deg":   rot_angle_diff(R_L, R_tri),
        }
    if R_Rxf is not None:
        disagreements["pnp_right_xf_vs_triangulated"] = {
            "translation_mm": float(np.linalg.norm(t_Rxf - t_tri)),
            "rotation_deg":   rot_angle_diff(R_Rxf, R_tri),
        }

    return {
        "id": tag_id,
        "corners_left_raw":  corners_L.tolist(),
        "corners_right_raw": corners_R.tolist(),
        "pnp_left":          pose_block(R_L,   t_L),
        "pnp_right_xf":      pose_block(R_Rxf, t_Rxf),
        "triangulated":      pose_block(R_tri, t_tri),
        "disagreements":     disagreements,
    }


def snapshot(entries, shape_L, shape_R):
    global frame_seq
    frame_seq += 1
    e = {
        "frame_index": frame_seq,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "left_image":  {"width": int(shape_L[1]), "height": int(shape_L[0])},
        "right_image": {"width": int(shape_R[1]), "height": int(shape_R[0])},
        "tags": entries,
    }
    log_frames.append(e)
    return e


def flush(path):
    if not log_frames:
        print("[WRITE] nothing to flush")
        return
    payload = {
        "left_intrinsic":   args.left_json,
        "right_intrinsic":  args.right_json,
        "stereo_extrinsic": args.extrinsic_json,
        "K_left":  K_L.tolist(),  "D_left":  D_L.ravel().tolist(),
        "K_right": K_R.tolist(),  "D_right": D_R.ravel().tolist(),
        "R_left_to_right":            R_stereo.tolist(),
        "T_left_to_right_mm":         T_stereo.ravel().tolist(),
        "baseline_mm":                float(np.linalg.norm(T_stereo)),
        "tag_family":                 args.tag_family,
        "tag_size_mm":                TAG_SIZE_MM,
        "pose_frame":                 "left_camera",
        "convention":                 "OpenCV (X right, Y down, Z forward)",
        "left_device":                args.left_device,
        "right_device":               args.right_device,
        "num_frames_logged":          len(log_frames),
        "frames":                     log_frames,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[WRITE] {len(log_frames)} frame(s) -> {path}")
    log_frames.clear()


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
    auto = False
    print("\nControls: S=snap, A=auto, W=write, Q/ESC=quit\n")

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
            common = sorted(set(tags_L) & set(tags_R))

            entries = []
            disp_L = fl.copy()
            disp_R = fr.copy()

            for tid in common:
                cL = tags_L[tid]
                cR = tags_R[tid]
                entry = compute_stereo_pose_entry(tid, cL, cR)
                entries.append(entry)

                # Draw left with axes from pnp_left, right with axes from pnp_right
                R_L_full, t_L_full = pnp(cL, K_L, D_L)
                R_Rr_full, t_Rr_full = pnp(cR, K_R, D_R)
                rvec_L = cv2.Rodrigues(R_L_full)[0] if R_L_full is not None else None
                rvec_R = cv2.Rodrigues(R_Rr_full)[0] if R_Rr_full is not None else None
                d_L = entry["pnp_left"]["distance_mm"] if entry["pnp_left"] else 0
                d_R = entry["pnp_right_xf"]["distance_mm"] if entry["pnp_right_xf"] else 0
                draw_tag(disp_L, cL, tid, d_L, GREEN,
                         rvec=rvec_L, tvec=t_L_full, K=K_L, D=D_L)
                draw_tag(disp_R, cR, tid, d_R, GREEN,
                         rvec=rvec_R, tvec=t_Rr_full, K=K_R, D=D_R)

            # Draw un-matched detections in orange
            for tid, c in tags_L.items():
                if tid not in common:
                    draw_tag(disp_L, c, tid, 0.0, ORANGE)
            for tid, c in tags_R.items():
                if tid not in common:
                    draw_tag(disp_R, c, tid, 0.0, ORANGE)

            # HUD
            put_text(disp_L, f"LEFT  detected={len(tags_L)} matched={len(common)}",
                     (20, 35), scale=0.65, color=CYAN)
            put_text(disp_R, f"RIGHT detected={len(tags_R)} matched={len(common)}",
                     (20, 35), scale=0.65, color=ORANGE)
            put_text(disp_L, f"logged={len(log_frames)}  AUTO={'ON' if auto else 'OFF'}",
                     (20, 65), scale=0.55, color=GREEN if auto else WHITE)
            put_text(disp_L, "S=snap  A=auto  W=write  Q=quit",
                     (20, disp_L.shape[0]-20), scale=0.55, color=YELLOW)

            combo = np.hstack((disp_L, disp_R))
            cv2.imshow(WINDOW_NAME, combo)

            if auto and entries:
                snapshot(entries, fl.shape, fr.shape)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                if entries:
                    e = snapshot(entries, fl.shape, fr.shape)
                    print(f"[SNAP] frame {e['frame_index']}: "
                          f"{len(e['tags'])} matched tag(s) "
                          f"({[t['id'] for t in e['tags']]})")
                    for t in entries:
                        d = t["disagreements"]
                        if "pnp_left_vs_pnp_right_xf" in d:
                            k = d["pnp_left_vs_pnp_right_xf"]
                            print(f"   id{t['id']}: L-vs-Rxf  Δt={k['translation_mm']:.2f} mm  "
                                  f"Δrot={k['rotation_deg']:.2f} deg")
                        if "pnp_left_vs_triangulated" in d:
                            k = d["pnp_left_vs_triangulated"]
                            print(f"   id{t['id']}: L-vs-Tri  Δt={k['translation_mm']:.2f} mm  "
                                  f"Δrot={k['rotation_deg']:.2f} deg")
                else:
                    print("[SNAP] no matched tags")
            elif key == ord("a"):
                auto = not auto
                print(f"[AUTO] {'ON' if auto else 'OFF'}")
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
