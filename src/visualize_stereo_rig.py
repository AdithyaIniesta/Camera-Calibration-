#!/usr/bin/env python3
"""
3D stereo rig visualizer with mouse rotate / pan / zoom.

Loads a stereo extrinsic JSON (rotation_left_to_right + translation_left_to_right_mm)
and draws both camera frames in 3D. Matplotlib's own 3D toolbar gives you:

  - LEFT drag   : rotate
  - RIGHT drag  : zoom
  - MIDDLE drag : pan
  - keyboard 'p': toggle pan mode
  - keyboard 'o': toggle zoom-to-rectangle
  - toolbar     : save PNG, home, back, forward

Usage:
  python3 visualize_stereo_rig.py extrinsic.json
  python3 visualize_stereo_rig.py            # file-picker dialog pops up
"""
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt


# ----- args ---------------------------------------------------------

parser = argparse.ArgumentParser(description="3D stereo rig visualizer")
parser.add_argument("extrinsic_json", nargs="?", default=None,
                    help="Stereo extrinsic JSON (contains R + T left->right).")
parser.add_argument("--axis-len", type=float, default=30.0,
                    help="Camera axis arrow length in mm (default 30).")
parser.add_argument("--frustum-depth", type=float, default=60.0,
                    help="Camera frustum depth in mm (default 60).")
args = parser.parse_args()

if args.extrinsic_json is None:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    root = tk.Tk(); root.withdraw()
    picked = filedialog.askopenfilename(
        title="Stereo extrinsic JSON",
        filetypes=[("JSON", "*.json")])
    if not picked:
        messagebox.showerror("Cancelled", "No file chosen.")
        sys.exit(1)
    args.extrinsic_json = picked


# ----- load ---------------------------------------------------------

def first_existing(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(keys[0])

with open(args.extrinsic_json, "r") as f:
    data = json.load(f)

R = np.asarray(first_existing(
    data, ["rotation_left_to_right", "R_left_to_right", "R"]), dtype=np.float64)
T = np.asarray(first_existing(
    data, ["translation_left_to_right_mm", "T_left_to_right_mm",
           "translation_left_to_right", "T"]), dtype=np.float64).ravel()
baseline = float(np.linalg.norm(T))

print(f"Loaded  : {args.extrinsic_json}")
print(f"R =\n{R}")
print(f"T (mm)  = {T}")
print(f"baseline = {baseline:.3f} mm")


# ----- helpers ------------------------------------------------------

AXIS = args.axis_len
FDEPTH = args.frustum_depth

def draw_camera(ax, origin, R_cam, label, sensor_hw=(24.0, 18.0),
                axis_color=('red', 'green', 'blue'), body_color='k'):
    """Draw axes + a pyramidal frustum representing the camera FOV."""
    o = np.asarray(origin, dtype=np.float64)

    # Axes: +X red, +Y green, +Z blue (OpenCV: X right, Y down, Z forward)
    for i, col in enumerate(axis_color):
        v = R_cam @ np.eye(3)[i] * AXIS
        ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2],
                  color=col, arrow_length_ratio=0.15, linewidth=1.6)

    # Frustum: 4 rays from origin to the sensor rectangle at Z = FDEPTH
    hw, hh = sensor_hw[0] / 2.0, sensor_hw[1] / 2.0
    corners_cam = np.array([
        [ hw,  hh, FDEPTH],
        [-hw,  hh, FDEPTH],
        [-hw, -hh, FDEPTH],
        [ hw, -hh, FDEPTH],
    ])
    corners_world = (R_cam @ corners_cam.T).T + o
    for c in corners_world:
        ax.plot([o[0], c[0]], [o[1], c[1]], [o[2], c[2]],
                color=body_color, linewidth=0.8, alpha=0.6)
    loop = np.vstack([corners_world, corners_world[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
            color=body_color, linewidth=1.0, alpha=0.7)

    # Origin + label
    ax.scatter(*o, color=body_color, s=45)
    ax.text(o[0], o[1] - 4, o[2] - 4, label,
            fontsize=10, fontweight='bold', color=body_color)


# ----- figure -------------------------------------------------------

O_L = np.zeros(3)
O_R = T.copy()

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')

draw_camera(ax, O_L, np.eye(3),  "LEFT (boresight)",  body_color='#0a5fa8')
draw_camera(ax, O_R, R,           "RIGHT (depression)", body_color='#c1620c')

# Baseline vector (mount → mount)
ax.plot([O_L[0], O_R[0]], [O_L[1], O_R[1]], [O_L[2], O_R[2]],
        color='purple', linestyle='--', linewidth=2,
        label=f"Baseline ‖T‖ = {baseline:.2f} mm")

# World origin marker
ax.scatter(0, 0, 0, color='k', marker='+', s=60)

ax.set_title(f"Stereo Rig  —  {args.extrinsic_json}\n"
             f"T = ({T[0]:+.2f}, {T[1]:+.2f}, {T[2]:+.2f}) mm    "
             f"‖T‖ = {baseline:.2f} mm")
ax.set_xlabel("X (mm) — right")
ax.set_ylabel("Y (mm) — down")
ax.set_zlabel("Z (mm) — forward")
ax.legend(loc="upper left", fontsize=9)

# Equal aspect so the geometry is honest
pad = 20.0
xs = np.array([O_L[0], O_R[0]]);  ys = np.array([O_L[1], O_R[1]]);  zs = np.array([O_L[2], O_R[2]])
span = max(xs.ptp(), ys.ptp(), zs.ptp(), 3 * AXIS) + pad
mid = np.array([(xs.min() + xs.max()) / 2,
                (ys.min() + ys.max()) / 2,
                (zs.min() + zs.max()) / 2])
ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
try:
    ax.set_box_aspect((1, 1, 1))   # equal aspect on 3D (matplotlib >= 3.3)
except Exception:
    pass

# Match OpenCV convention: +Y down, +Z forward
ax.invert_yaxis()

plt.tight_layout()
plt.show()
