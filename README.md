# Camera Calibration — Stereo Refinement Toolkit

Small set of scripts for refining and verifying the extrinsic calibration of a
stereo camera rig on a Jetson. Capture uses a strict Jetson GStreamer pipeline:
`v4l2src → UYVY 1280×720@60 → nvvidconv → BGR → appsink`.

## Programs

- **`refine_stereo_extrinsics.py`** — Live stereo capture; click matching
  points in the left/right feeds and re-optimize the stereo extrinsic (R, t)
  against the existing per-camera intrinsics. Writes `refined_extrinsics.json`.

- **`april_tag_stereo_verify.py`** — Uses an AprilTag visible in both cameras
  as a 3D reference to check how well a given extrinsic
  reprojects/triangulates. Writes `apriltag_verify.json`.

- **`compare_extrinsics.py`** — Quantifies original vs. refined extrinsics on
  corresponding points (epipolar / reprojection / triangulation error). Runs
  live or offline with `--left-image` / `--right-image`. Writes
  `comparison_result.json`.

## Requirements

- Jetson with GStreamer + `nvvidconv` (JetPack)
- OpenCV built with GStreamer support
- Two UYVY-capable USB/CSI cameras at `/dev/videoN`
