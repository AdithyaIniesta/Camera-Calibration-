# Camera Calibration — Stereo Refinement Toolkit

Scripts for refining and verifying the extrinsic calibration of a stereo
camera rig on a Jetson. Capture uses a strict Jetson GStreamer pipeline:
`v4l2src → UYVY 1280×720@60 → nvvidconv → BGR → appsink`.

## Layout

```
.
├── src/                        # Programs
│   ├── refine_stereo_extrinsics.py
│   ├── april_tag_stereo_verify.py
│   └── compare_extrinsics.py
├── data/                       # Intrinsics, extrinsics, results
│   ├── left_four.json          # left  camera intrinsics
│   ├── right_four.json         # right camera intrinsics
│   ├── stereo_calibration.json # original stereo extrinsic
│   ├── refined_extrinsics.json # output of refine
│   ├── apriltag_verify.json    # output of verify
│   ├── comparison_result.json  # output of compare
│   └── samples/                # sample stereo pair (imgL0/imgR0)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Programs

- **`src/refine_stereo_extrinsics.py`** — Live stereo capture; click matching
  points in the left/right feeds and re-optimize the stereo extrinsic (R, t)
  against the existing per-camera intrinsics. Writes `refined_extrinsics.json`.

- **`src/april_tag_stereo_verify.py`** — Uses an AprilTag visible in both
  cameras as a 3D reference to check how well a given extrinsic
  reprojects/triangulates. Writes `apriltag_verify.json`.

- **`src/compare_extrinsics.py`** — Quantifies original vs. refined extrinsics
  on corresponding points (epipolar / reprojection / triangulation error). Runs
  live or offline with `--left-image` / `--right-image`. Writes
  `comparison_result.json`.

## Requirements

- Jetson with JetPack — provides GStreamer, `nvvidconv`, and a GStreamer-enabled
  OpenCV. The pip `opencv-*` wheels are built **without** GStreamer support and
  will silently fail to open the pipeline, so use the system OpenCV:

  ```bash
  sudo apt install python3-opencv
  ```

- Two UYVY-capable cameras at `/dev/videoN`
- Python deps:

  ```bash
  pip3 install -r requirements.txt   # numpy, scipy
  ```

## Run

```bash
python3 src/refine_stereo_extrinsics.py \
  data/left_four.json data/right_four.json data/stereo_calibration.json
```
