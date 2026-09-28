"""
Prepend the CUDA-built OpenCV to sys.path so `import cv2` picks it up
instead of the system/pip one. Import this BEFORE `import cv2`.

Point OPENCV_CUDA_PREFIX at the install prefix from build_opencv_cuda.sh,
e.g. /opt/opencv-4.10.0-cuda. If unset, we fall back to system OpenCV.
"""
import os
import sys
from pathlib import Path


def _activate() -> None:
    prefix = os.environ.get("OPENCV_CUDA_PREFIX")
    if not prefix:
        return
    # opencv installs its python module under <prefix>/lib/python3/dist-packages
    candidate = Path(prefix) / "lib" / "python3" / "dist-packages"
    if not candidate.is_dir():
        print(f"[opencv-cuda] WARNING: {candidate} not found; using system cv2")
        return
    sys.path.insert(0, str(candidate))


_activate()
