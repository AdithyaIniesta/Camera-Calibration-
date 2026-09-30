"""
Corner / centre bearings measured from a camera's own principal point.

Same convention as the C++ tracker (calibration.cpp: computeAngleOffsets):

    alpha = atan2(u - cx, fx)       positive = right of the optical axis
    beta  = atan2(-(v - cy), fy)    positive = above the optical axis
                                    (image Y grows downward)

The origin is the camera's own (cx, cy) from its intrinsics — never the
image centre (W/2, H/2). Raw (distorted) pixels are used as-is, exactly
like the tracker; nothing is undistorted.

For a camera that is NOT the boresight, the bearing can also be expressed
in the boresight frame with the stereo extrinsic, exactly like the tracker
does for the depression camera:

    ray_own     = ((u - cx)/fx, (v - cy)/fy, 1)
    ray_bore    = R_left_to_right^T . ray_own
    alpha_bore  = atan2( ray_bore.x, ray_bore.z)
    beta_bore   = atan2(-ray_bore.y, ray_bore.z)
"""
import math

import numpy as np


def bearing_deg(u, v, K):
    """(alpha, beta) in degrees from the camera's own principal point."""
    dx = float(u) - K[0, 2]
    dy = float(v) - K[1, 2]
    return (math.degrees(math.atan2(dx, K[0, 0])),
            math.degrees(math.atan2(-dy, K[1, 1])))


def bearing_in_boresight_deg(u, v, K, R_left_to_right):
    """Bearing of pixel (u, v) of the RIGHT camera, in the LEFT (boresight)
    camera's frame."""
    ray = np.array([(float(u) - K[0, 2]) / K[0, 0],
                    (float(v) - K[1, 2]) / K[1, 1],
                    1.0])
    b = np.asarray(R_left_to_right, dtype=np.float64).T @ ray
    return (math.degrees(math.atan2(b[0], b[2])),
            math.degrees(math.atan2(-b[1], b[2])))


def _point(u, v, K, R_left_to_right):
    a, b = bearing_deg(u, v, K)
    p = {
        "u": float(u),
        "v": float(v),
        "du_from_cx": float(u) - float(K[0, 2]),
        "dv_from_cy": float(v) - float(K[1, 2]),
        "alpha_deg": a,
        "beta_deg": b,
    }
    if R_left_to_right is not None:
        ab, bb = bearing_in_boresight_deg(u, v, K, R_left_to_right)
        p["alpha_boresight_deg"] = ab
        p["beta_boresight_deg"] = bb
    return p


def corner_angles_entry(corners, K, R_left_to_right=None):
    """Angles for one tag: its 4 corners (detector order, which follows the
    tag's own TL, TR, BR, BL and is NOT necessarily the on-screen order) plus
    the centre of the quad. Pass R_left_to_right only for the non-boresight
    camera to also get the boresight-frame bearings."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    ctr = c.mean(axis=0)
    return {
        "corners": [dict(index=i, **_point(x, y, K, R_left_to_right))
                    for i, (x, y) in enumerate(c)],
        "center": _point(ctr[0], ctr[1], K, R_left_to_right),
    }
