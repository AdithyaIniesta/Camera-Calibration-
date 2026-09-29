#!/usr/bin/env python3

import argparse
import json

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

LEFT_DEVICE = "/dev/video0"       # BORESIGHT / LEFT
RIGHT_DEVICE = "/dev/video2"      # DEPRESSION / RIGHT

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 60

TAG_FAMILY = "36h11"

# ------------------------------------------------------------
# FIXED PHYSICAL APRILTAG GEOMETRY
# ------------------------------------------------------------

TAG_SIZE_M = 0.150                # 150 mm
TAG_CENTER_SPACING_M = 0.220      # 220 mm

WINDOW_NAME = "AprilTag 3D PnP Stereo Pose"

OUTPUT_FILE = "apriltag_pnp_stereo_pose.json"

# Display colors - BGR
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "AprilTag 3D PnP stereo pose estimation. "
        "Automatically selects two common 36h11 tags."
    )
)

parser.add_argument(
    "--out",
    default=OUTPUT_FILE,
    help="Output JSON file",
)

args = parser.parse_args()


# ============================================================
# TERMINAL INPUT
# ============================================================

def ask_float(prompt, default=None):

    while True:

        if default is None:
            text = input(
                f"{prompt}: "
            ).strip()

        else:

            text = input(
                f"{prompt} [{default}]: "
            ).strip()

            if text == "":
                return float(default)

        try:
            return float(text)

        except ValueError:
            print(
                "Please enter a valid number."
            )


# ============================================================
# CAMERA INTRINSICS
# ============================================================

def get_camera_intrinsics(name):

    print()
    print("=" * 70)
    print(f"{name} CAMERA INTRINSICS")
    print("=" * 70)

    print()
    print(
        "Enter the calibrated camera intrinsic parameters."
    )

    print(
        "fx/fy/cx/cy are in pixels."
    )

    fx = ask_float(
        f"{name} fx",
        IMAGE_WIDTH,
    )

    fy = ask_float(
        f"{name} fy",
        IMAGE_WIDTH,
    )

    cx = ask_float(
        f"{name} cx",
        IMAGE_WIDTH / 2.0,
    )

    cy = ask_float(
        f"{name} cy",
        IMAGE_HEIGHT / 2.0,
    )

    print()
    print(
        "Distortion coefficients:"
    )

    print(
        "Enter your calibrated values."
    )

    print(
        "Use zero only if distortion is negligible "
        "or images are already undistorted."
    )

    k1 = ask_float(
        f"{name} k1",
        0.0,
    )

    k2 = ask_float(
        f"{name} k2",
        0.0,
    )

    p1 = ask_float(
        f"{name} p1",
        0.0,
    )

    p2 = ask_float(
        f"{name} p2",
        0.0,
    )

    k3 = ask_float(
        f"{name} k3",
        0.0,
    )

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    D = np.array(
        [k1, k2, p1, p2, k3],
        dtype=np.float64,
    )

    return K, D


# ============================================================
# BOARD GEOMETRY
# ============================================================

def get_board_geometry(tag_ids):

    """
    Fixed board geometry.

    First detected tag:
        center = (0, 0, 0)

    Second detected tag:
        center = (0.220, 0, 0) meters

    Therefore:

        tag center spacing = 220 mm

    Both tags are assumed to lie in the same physical plane
    and have the same orientation.
    """

    tag_a = int(tag_ids[0])
    tag_b = int(tag_ids[1])

    centers = {

        tag_a: np.array(
            [
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),

        tag_b: np.array(
            [
                TAG_CENTER_SPACING_M,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),
    }

    return centers


def tag_corners_3d(
    center,
    tag_size,
):

    """
    AprilTag corner order:

        0 = top-left
        1 = top-right
        2 = bottom-right
        3 = bottom-left

    Board coordinate system:

        +X = right
        +Y = up
        +Z = out of tag plane
    """

    half = tag_size / 2.0

    cx, cy, cz = center

    return np.array(
        [
            [
                cx - half,
                cy + half,
                cz,
            ],

            [
                cx + half,
                cy + half,
                cz,
            ],

            [
                cx + half,
                cy - half,
                cz,
            ],

            [
                cx - half,
                cy - half,
                cz,
            ],
        ],
        dtype=np.float64,
    )


def build_board_points(
    tag_ids,
    tag_centers,
):

    points = []

    for tag_id in tag_ids:

        corners = tag_corners_3d(
            tag_centers[tag_id],
            TAG_SIZE_M,
        )

        points.extend(
            corners
        )

    return np.asarray(
        points,
        dtype=np.float64,
    )


# ============================================================
# APRILTAG DETECTOR
# ============================================================

FAMILY_MAP = {
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

aruco_dict = cv2.aruco.getPredefinedDictionary(
    FAMILY_MAP[TAG_FAMILY]
)

try:

    detector_params = (
        cv2.aruco.DetectorParameters()
    )

    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        detector_params,
    )

    def detect_tags(gray):

        corners, ids, _ = (
            detector.detectMarkers(
                gray
            )
        )

        return corners, ids

except AttributeError:

    detector_params = (
        cv2.aruco.DetectorParameters_create()
    )

    def detect_tags(gray):

        corners, ids, _ = (
            cv2.aruco.detectMarkers(
                gray,
                aruco_dict,
                parameters=detector_params,
            )
        )

        return corners, ids


# ============================================================
# CAMERA CAPTURE
# ============================================================

class CameraCapture:

    def __init__(self, device):

        self.device = device
        self.cap = None

    def start(self):

        pipeline = (
            f"v4l2src device={self.device} io-mode=2 ! "
            "video/x-raw,format=UYVY,"
            "width=1280,height=720,framerate=60/1 ! "
            "nvvidconv ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        )

        self.cap = cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER,
        )

        if not self.cap.isOpened():

            print(
                f"ERROR: Could not open camera "
                f"{self.device}"
            )

            return False

        print(
            f"Camera started: {self.device}"
        )

        return True

    def read(self):

        if self.cap is None:

            return False, None

        return self.cap.read()

    def stop(self):

        if self.cap is not None:

            self.cap.release()


# ============================================================
# APRILTAG DETECTION
# ============================================================

def detect_tag_dict(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    corners, ids = detect_tags(
        gray
    )

    result = {}

    if ids is None:

        return result

    for i, tag_id in enumerate(
        ids.flatten()
    ):

        pts = (
            corners[i]
            .reshape(4, 2)
            .astype(np.float64)
        )

        result[int(tag_id)] = pts

    return result


# ============================================================
# AUTOMATIC TAG SELECTION
# ============================================================

def choose_two_common_tags(
    left_tags,
    right_tags,
):

    """
    Find tags visible in both cameras.

    The first two common IDs are selected.

    No --tag-ids argument is required.
    """

    common = sorted(
        set(left_tags.keys())
        &
        set(right_tags.keys())
    )

    if len(common) < 2:

        return None

    return common[:2]


# ============================================================
# IMAGE POINTS
# ============================================================

def build_image_points(
    left_tags,
    right_tags,
    tag_ids,
):

    left_points = []
    right_points = []

    for tag_id in tag_ids:

        left_points.extend(
            left_tags[tag_id]
        )

        right_points.extend(
            right_tags[tag_id]
        )

    return (
        np.asarray(
            left_points,
            dtype=np.float64,
        ),

        np.asarray(
            right_points,
            dtype=np.float64,
        ),
    )


# ============================================================
# PNP
# ============================================================

def solve_camera_pose(
    object_points,
    image_points,
    K,
    D,
):

    """
    Solve:

        X_camera =
            R * X_board + t
    """

    if len(object_points) < 4:

        return (
            False,
            None,
            None,
            None,
        )

    success, rvec, tvec = (
        cv2.solvePnP(
            object_points,
            image_points,
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    )

    if not success:

        return (
            False,
            None,
            None,
            None,
        )

    R, _ = cv2.Rodrigues(
        rvec
    )

    return (
        True,
        rvec,
        tvec,
        R,
    )


# ============================================================
# REPROJECTION ERROR
# ============================================================

def reprojection_errors(
    object_points,
    image_points,
    rvec,
    tvec,
    K,
    D,
):

    projected, _ = (
        cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            K,
            D,
        )
    )

    projected = (
        projected.reshape(-1, 2)
    )

    errors = np.linalg.norm(
        projected - image_points,
        axis=1,
    )

    return (
        projected,
        errors,
    )


# ============================================================
# TRANSFORM
# ============================================================

def relative_right_from_left(
    R_left,
    t_left,
    R_right,
    t_right,
):

    """
    PnP gives:

        X_left =
            R_left X_board + t_left

        X_right =
            R_right X_board + t_right

    Therefore:

        X_right =
            R_rel X_left + t_rel

    with:

        R_rel =
            R_right R_left^T

        t_rel =
            t_right -
            R_rel t_left
    """

    R_rel = (
        R_right
        @
        R_left.T
    )

    t_rel = (
        t_right.reshape(3)
        -
        R_rel
        @
        t_left.reshape(3)
    )

    return (
        R_rel,
        t_rel,
    )


def make_transform(
    R,
    t,
):

    T = np.eye(
        4,
        dtype=np.float64,
    )

    T[:3, :3] = R

    T[:3, 3] = (
        t.reshape(3)
    )

    return T


def rotation_matrix_to_euler_deg(
    R,
):

    """
    Returns XYZ-style:

        roll
        pitch
        yaw

    in degrees.
    """

    sy = np.sqrt(
        R[0, 0] ** 2
        +
        R[1, 0] ** 2
    )

    singular = (
        sy < 1e-6
    )

    if not singular:

        roll = np.arctan2(
            R[2, 1],
            R[2, 2],
        )

        pitch = np.arctan2(
            -R[2, 0],
            sy,
        )

        yaw = np.arctan2(
            R[1, 0],
            R[0, 0],
        )

    else:

        roll = np.arctan2(
            -R[1, 2],
            R[1, 1],
        )

        pitch = np.arctan2(
            -R[2, 0],
            sy,
        )

        yaw = 0.0

    return np.degrees(
        [
            roll,
            pitch,
            yaw,
        ]
    )


# ============================================================
# DRAWING
# ============================================================

def safe_point(p):

    if not np.all(
        np.isfinite(p)
    ):

        return None

    return (
        int(round(float(p[0]))),
        int(round(float(p[1]))),
    )


def draw_tag(
    image,
    corners,
    tag_id,
    color,
):

    pts = [
        safe_point(p)
        for p in corners
    ]

    for i in range(4):

        a = pts[i]
        b = pts[
            (i + 1) % 4
        ]

        if (
            a is not None
            and b is not None
        ):

            cv2.line(
                image,
                a,
                b,
                color,
                2,
                cv2.LINE_AA,
            )

    for i, p in enumerate(
        pts
    ):

        if p is None:
            continue

        cv2.circle(
            image,
            p,
            7,
            BLACK,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            p,
            4,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{tag_id}:{i}",
            (
                p[0] + 8,
                p[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_text(
    image,
    text,
    position,
    color=WHITE,
    scale=0.65,
):

    x, y = position

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        BLACK,
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_axis(
    image,
    rvec,
    tvec,
    K,
    D,
    axis_length,
):

    axis = np.array(
        [
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ],
        dtype=np.float64,
    )

    projected, _ = (
        cv2.projectPoints(
            axis,
            rvec,
            tvec,
            K,
            D,
        )
    )

    projected = (
        projected.reshape(-1, 2)
    )

    origin = safe_point(
        projected[0]
    )

    x_axis = safe_point(
        projected[1]
    )

    y_axis = safe_point(
        projected[2]
    )

    z_axis = safe_point(
        projected[3]
    )

    if origin is None:
        return

    if x_axis is not None:

        cv2.line(
            image,
            origin,
            x_axis,
            RED,
            3,
            cv2.LINE_AA,
        )

    if y_axis is not None:

        cv2.line(
            image,
            origin,
            y_axis,
            GREEN,
            3,
            cv2.LINE_AA,
        )

    if z_axis is not None:

        cv2.line(
            image,
            origin,
            z_axis,
            BLUE,
            3,
            cv2.LINE_AA,
        )


# ============================================================
# SAVE
# ============================================================

def save_result(
    path,
    tag_ids,
    tag_centers,
    K_left,
    D_left,
    K_right,
    D_right,
    R_rel,
    t_rel,
    T_rel,
    rvec_left,
    tvec_left,
    rvec_right,
    tvec_right,
    errors_left,
    errors_right,
):

    left_rms = np.sqrt(
        np.mean(
            errors_left ** 2
        )
    )

    right_rms = np.sqrt(
        np.mean(
            errors_right ** 2
        )
    )

    euler = (
        rotation_matrix_to_euler_deg(
            R_rel
        )
    )

    data = {

        "source":
            "AprilTag_3D_PnP_stereo",

        "tag_family":
            TAG_FAMILY,

        "tag_ids":
            [
                int(x)
                for x in tag_ids
            ],

        "tag_size_m":
            TAG_SIZE_M,

        "tag_size_mm":
            TAG_SIZE_M * 1000.0,

        "tag_center_spacing_m":
            TAG_CENTER_SPACING_M,

        "tag_center_spacing_mm":
            TAG_CENTER_SPACING_M * 1000.0,

        "tag_centers_board_m":
            {
                str(k): v.tolist()
                for k, v
                in tag_centers.items()
            },

        "left_camera_matrix":
            K_left.tolist(),

        "left_distortion":
            D_left.tolist(),

        "right_camera_matrix":
            K_right.tolist(),

        "right_distortion":
            D_right.tolist(),

        "left_board_rvec":
            rvec_left.reshape(3).tolist(),

        "left_board_tvec_m":
            tvec_left.reshape(3).tolist(),

        "right_board_rvec":
            rvec_right.reshape(3).tolist(),

        "right_board_tvec_m":
            tvec_right.reshape(3).tolist(),

        "left_to_right_rotation":
            R_rel.tolist(),

        "left_to_right_translation_m":
            t_rel.tolist(),

        "left_to_right_translation_mm":
            (
                t_rel * 1000.0
            ).tolist(),

        "left_to_right_transform":
            T_rel.tolist(),

        "relative_roll_deg":
            float(euler[0]),

        "relative_pitch_deg":
            float(euler[1]),

        "relative_yaw_deg":
            float(euler[2]),

        "baseline_m":
            float(
                np.linalg.norm(t_rel)
            ),

        "baseline_mm":
            float(
                np.linalg.norm(t_rel)
                * 1000.0
            ),

        "left_reprojection_errors_px":
            errors_left.tolist(),

        "right_reprojection_errors_px":
            errors_right.tolist(),

        "left_rms_reprojection_px":
            float(left_rms),

        "right_rms_reprojection_px":
            float(right_rms),

        "uses_camera_intrinsics":
            True,

        "uses_stereo_extrinsics":
            False,

        "method":
            (
                "Two common AprilTags, "
                "known 3D board geometry, "
                "independent solvePnP, "
                "relative camera transform"
            ),
    }

    with open(
        path,
        "w",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

    print()
    print(
        f"[SAVE] Saved to: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # FIXED GEOMETRY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("APRILTAG 3D PNP STEREO POSE")
    print("=" * 70)

    print()
    print(
        f"AprilTag family       : {TAG_FAMILY}"
    )

    print(
        f"AprilTag size         : "
        f"{TAG_SIZE_M * 1000:.1f} mm"
    )

    print(
        f"Tag center spacing    : "
        f"{TAG_CENTER_SPACING_M * 1000:.1f} mm"
    )

    print()
    print(
        "Two common AprilTags will be selected automatically."
    )

    # --------------------------------------------------------
    # CAMERA INTRINSICS
    # --------------------------------------------------------

    K_left, D_left = (
        get_camera_intrinsics(
            "LEFT"
        )
    )

    K_right, D_right = (
        get_camera_intrinsics(
            "RIGHT"
        )
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    left_cam = CameraCapture(
        LEFT_DEVICE
    )

    right_cam = CameraCapture(
        RIGHT_DEVICE
    )

    if not left_cam.start():

        return 1

    if not right_cam.start():

        left_cam.stop()

        return 1

    # --------------------------------------------------------
    # WINDOW
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    pose_valid = False

    last_tag_ids = None

    last_tag_centers = None

    last_R_rel = None
    last_t_rel = None
    last_T_rel = None

    last_rvec_left = None
    last_tvec_left = None

    last_rvec_right = None
    last_tvec_right = None

    last_errors_left = None
    last_errors_right = None

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONTROLS")
    print("=" * 70)
    print()
    print("H  = solve 3D PnP")
    print("S  = save pose")
    print("R  = reset")
    print("Q  = quit")
    print("ESC = quit")
    print()
    print(
        "CLICK THE VIDEO WINDOW BEFORE PRESSING KEYS."
    )
    print()

    try:

        while True:

            # =================================================
            # READ CAMERAS
            # =================================================

            ret_l, frame_l = (
                left_cam.read()
            )

            ret_r, frame_r = (
                right_cam.read()
            )

            if (
                not ret_l
                or not ret_r
            ):

                raw_key = (
                    cv2.waitKeyEx(1)
                )

                if raw_key == 27:

                    break

                if raw_key != -1:

                    k = (
                        raw_key
                        & 0xFF
                    )

                    if k in (
                        ord("q"),
                        ord("Q"),
                    ):

                        break

                continue

            # =================================================
            # DETECT
            # =================================================

            left_tags = (
                detect_tag_dict(
                    frame_l
                )
            )

            right_tags = (
                detect_tag_dict(
                    frame_r
                )
            )

            common_ids = (
                choose_two_common_tags(
                    left_tags,
                    right_tags,
                )
            )

            display_l = (
                frame_l.copy()
            )

            display_r = (
                frame_r.copy()
            )

            # =================================================
            # DRAW TAGS
            # =================================================

            for tag_id, corners in (
                left_tags.items()
            ):

                selected = (
                    common_ids is not None
                    and tag_id
                    in common_ids
                )

                draw_tag(
                    display_l,
                    corners,
                    tag_id,
                    GREEN
                    if selected
                    else CYAN,
                )

            for tag_id, corners in (
                right_tags.items()
            ):

                selected = (
                    common_ids is not None
                    and tag_id
                    in common_ids
                )

                draw_tag(
                    display_r,
                    corners,
                    tag_id,
                    GREEN
                    if selected
                    else ORANGE,
                )

            # =================================================
            # CURRENT IMAGE POINTS
            # =================================================

            current_left = None
            current_right = None

            if common_ids is not None:

                (
                    current_left,
                    current_right,
                ) = build_image_points(
                    left_tags,
                    right_tags,
                    common_ids,
                )

                for p in current_left:

                    pt = safe_point(p)

                    if pt is not None:

                        cv2.circle(
                            display_l,
                            pt,
                            8,
                            YELLOW,
                            2,
                            cv2.LINE_AA,
                        )

                for p in current_right:

                    pt = safe_point(p)

                    if pt is not None:

                        cv2.circle(
                            display_r,
                            pt,
                            8,
                            YELLOW,
                            2,
                            cv2.LINE_AA,
                        )

            # =================================================
            # AXES
            # =================================================

            if (
                pose_valid
                and last_rvec_left
                is not None
            ):

                draw_axis(
                    display_l,
                    last_rvec_left,
                    last_tvec_left,
                    K_left,
                    D_left,
                    TAG_SIZE_M * 2.0,
                )

            if (
                pose_valid
                and last_rvec_right
                is not None
            ):

                draw_axis(
                    display_r,
                    last_rvec_right,
                    last_tvec_right,
                    K_right,
                    D_right,
                    TAG_SIZE_M * 2.0,
                )

            # =================================================
            # LEFT TEXT
            # =================================================

            draw_text(
                display_l,
                "LEFT / BORESIGHT",
                (20, 40),
                CYAN,
                0.8,
            )

            if common_ids is None:

                draw_text(
                    display_l,
                    "Need 2 common 36h11 tags",
                    (20, 75),
                    RED,
                )

            else:

                draw_text(
                    display_l,
                    f"Common tags: {common_ids}",
                    (20, 75),
                    GREEN,
                )

                draw_text(
                    display_l,
                    "8 x 3D/2D PnP points ready",
                    (20, 105),
                    YELLOW,
                )

            # =================================================
            # RIGHT TEXT
            # =================================================

            draw_text(
                display_r,
                "RIGHT / DEPRESSION",
                (20, 40),
                ORANGE,
                0.8,
            )

            if pose_valid:

                draw_text(
                    display_r,
                    "3D PnP ACTIVE",
                    (20, 75),
                    GREEN,
                )

                if (
                    last_t_rel
                    is not None
                ):

                    baseline = (
                        np.linalg.norm(
                            last_t_rel
                        )
                    )

                    draw_text(
                        display_r,
                        (
                            f"Baseline: "
                            f"{baseline * 1000:.2f} mm"
                        ),
                        (20, 105),
                        BLUE,
                    )

                if (
                    last_errors_left
                    is not None
                    and last_errors_right
                    is not None
                ):

                    left_rms = np.sqrt(
                        np.mean(
                            last_errors_left
                            ** 2
                        )
                    )

                    right_rms = np.sqrt(
                        np.mean(
                            last_errors_right
                            ** 2
                        )
                    )

                    draw_text(
                        display_r,
                        (
                            f"RMS L/R: "
                            f"{left_rms:.2f}/"
                            f"{right_rms:.2f} px"
                        ),
                        (20, 135),
                        GREEN,
                    )

            else:

                draw_text(
                    display_r,
                    "Press H to solve PnP",
                    (20, 75),
                    YELLOW,
                )

            # =================================================
            # SIDE BY SIDE
            # =================================================

            view = np.hstack(
                (
                    display_l,
                    display_r,
                )
            )

            cv2.imshow(
                WINDOW_NAME,
                view,
            )

            # =================================================
            # KEYBOARD
            # =================================================

            raw_key = (
                cv2.waitKeyEx(1)
            )

            # ESC
            if raw_key == 27:

                break

            # No key
            if raw_key == -1:

                continue

            key = (
                raw_key & 0xFF
            )

            # Upper/lowercase safe
            key = chr(key).lower()

            # =================================================
            # H = PNP
            # =================================================

            if key == "h":

                if (
                    current_left is None
                    or current_right is None
                ):

                    print()
                    print(
                        "[PNP] Need two common "
                        "AprilTags visible "
                        "in BOTH cameras."
                    )

                    continue

                print()
                print("=" * 70)
                print(
                    "SOLVING 3D PNP"
                )
                print("=" * 70)

                # ------------------------------------------------
                # BOARD GEOMETRY
                # ------------------------------------------------

                tag_centers = (
                    get_board_geometry(
                        common_ids
                    )
                )

                board_points = (
                    build_board_points(
                        common_ids,
                        tag_centers,
                    )
                )

                # ------------------------------------------------
                # LEFT PNP
                # ------------------------------------------------

                (
                    success_l,
                    rvec_l,
                    tvec_l,
                    R_l,
                ) = solve_camera_pose(
                    board_points,
                    current_left,
                    K_left,
                    D_left,
                )

                if not success_l:

                    print(
                        "[PNP] LEFT solvePnP failed."
                    )

                    continue

                # ------------------------------------------------
                # RIGHT PNP
                # ------------------------------------------------

                (
                    success_r,
                    rvec_r,
                    tvec_r,
                    R_r,
                ) = solve_camera_pose(
                    board_points,
                    current_right,
                    K_right,
                    D_right,
                )

                if not success_r:

                    print(
                        "[PNP] RIGHT solvePnP failed."
                    )

                    continue

                # ------------------------------------------------
                # REPROJECTION
                # ------------------------------------------------

                _, errors_l = (
                    reprojection_errors(
                        board_points,
                        current_left,
                        rvec_l,
                        tvec_l,
                        K_left,
                        D_left,
                    )
                )

                _, errors_r = (
                    reprojection_errors(
                        board_points,
                        current_right,
                        rvec_r,
                        tvec_r,
                        K_right,
                        D_right,
                    )
                )

                # ------------------------------------------------
                # LEFT -> RIGHT
                # ------------------------------------------------

                (
                    R_rel,
                    t_rel,
                ) = relative_right_from_left(
                    R_l,
                    tvec_l,
                    R_r,
                    tvec_r,
                )

                T_rel = make_transform(
                    R_rel,
                    t_rel,
                )

                euler = (
                    rotation_matrix_to_euler_deg(
                        R_rel
                    )
                )

                baseline = (
                    np.linalg.norm(
                        t_rel
                    )
                )

                # ------------------------------------------------
                # SAVE STATE
                # ------------------------------------------------

                pose_valid = True

                last_tag_ids = list(
                    common_ids
                )

                last_tag_centers = (
                    tag_centers
                )

                last_R_rel = (
                    R_rel.copy()
                )

                last_t_rel = (
                    t_rel.copy()
                )

                last_T_rel = (
                    T_rel.copy()
                )

                last_rvec_left = (
                    rvec_l.copy()
                )

                last_tvec_left = (
                    tvec_l.copy()
                )

                last_rvec_right = (
                    rvec_r.copy()
                )

                last_tvec_right = (
                    tvec_r.copy()
                )

                last_errors_left = (
                    errors_l.copy()
                )

                last_errors_right = (
                    errors_r.copy()
                )

                # ------------------------------------------------
                # PRINT RESULT
                # ------------------------------------------------

                print()
                print(
                    "SELECTED TAGS:"
                )

                print(
                    common_ids
                )

                print()
                print(
                    "TAG SIZE:"
                )

                print(
                    f"{TAG_SIZE_M * 1000:.3f} mm"
                )

                print()
                print(
                    "TAG CENTER SPACING:"
                )

                print(
                    f"{TAG_CENTER_SPACING_M * 1000:.3f} mm"
                )

                print()
                print(
                    "LEFT BOARD TRANSLATION:"
                )

                print(
                    tvec_l.reshape(3)
                )

                print()
                print(
                    "RIGHT BOARD TRANSLATION:"
                )

                print(
                    tvec_r.reshape(3)
                )

                print()
                print("=" * 70)
                print(
                    "LEFT -> RIGHT ROTATION"
                )
                print("=" * 70)

                print(
                    R_rel
                )

                print()
                print(
                    "LEFT -> RIGHT TRANSLATION (m)"
                )

                print(
                    t_rel
                )

                print()
                print(
                    "LEFT -> RIGHT TRANSLATION (mm)"
                )

                print(
                    t_rel * 1000.0
                )

                print()
                print(
                    "BASELINE:"
                )

                print(
                    f"{baseline * 1000.0:.3f} mm"
                )

                print()
                print(
                    "RELATIVE ROTATION:"
                )

                print(
                    f"Roll  : {euler[0]:.4f} deg"
                )

                print(
                    f"Pitch : {euler[1]:.4f} deg"
                )

                print(
                    f"Yaw   : {euler[2]:.4f} deg"
                )

                print()
                print(
                    "REPROJECTION ERROR:"
                )

                print(
                    f"LEFT RMS  : "
                    f"{np.sqrt(np.mean(errors_l ** 2)):.4f} px"
                )

                print(
                    f"RIGHT RMS : "
                    f"{np.sqrt(np.mean(errors_r ** 2)):.4f} px"
                )

                print(
                    f"LEFT MAX  : "
                    f"{np.max(errors_l):.4f} px"
                )

                print(
                    f"RIGHT MAX : "
                    f"{np.max(errors_r):.4f} px"
                )

                print()
                print(
                    "4x4 LEFT -> RIGHT TRANSFORM:"
                )

                print(
                    T_rel
                )

                print()

            # =================================================
            # S = SAVE
            # =================================================

            elif key == "s":

                if not pose_valid:

                    print()
                    print(
                        "[SAVE] No valid PnP pose."
                    )

                    print(
                        "[SAVE] Press H first."
                    )

                    continue

                save_result(
                    args.out,
                    last_tag_ids,
                    last_tag_centers,
                    K_left,
                    D_left,
                    K_right,
                    D_right,
                    last_R_rel,
                    last_t_rel,
                    last_T_rel,
                    last_rvec_left,
                    last_tvec_left,
                    last_rvec_right,
                    last_tvec_right,
                    last_errors_left,
                    last_errors_right,
                )

            # =================================================
            # R = RESET
            # =================================================

            elif key == "r":

                pose_valid = False

                last_tag_ids = None
                last_tag_centers = None

                last_R_rel = None
                last_t_rel = None
                last_T_rel = None

                last_rvec_left = None
                last_tvec_left = None

                last_rvec_right = None
                last_tvec_right = None

                last_errors_left = None
                last_errors_right = None

                print()
                print(
                    "[RESET] PnP pose cleared."
                )

            # =================================================
            # Q = QUIT
            # =================================================

            elif key == "q":

                break

    finally:

        left_cam.stop()
        right_cam.stop()

        cv2.destroyAllWindows()

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )

