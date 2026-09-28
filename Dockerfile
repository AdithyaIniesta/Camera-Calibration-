# Jetson (JetPack) base image — L4T r35.4.1 has GStreamer + nvvidconv preinstalled.
# On x86 dev hosts you can swap this for python:3.10-slim, but capture won't work
# without a Jetson (no nvvidconv). The scripts still run in offline mode.
FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps matched exactly to the capture pipeline:
#   v4l2src  -> gstreamer1.0-plugins-good
#   videoconvert, appsink -> gstreamer1.0-plugins-base
#   nvvidconv -> nvidia-l4t-gstreamer (already in the l4t-jetpack base image)
# Plus Python and the libs OpenCV's GUI + video I/O need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY README.md ./

# Default: show the refine tool's help. Override with e.g.:
#   docker run --rm -it --runtime nvidia --device /dev/video0 --device /dev/video1 \
#       -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix camcal
CMD ["python3", "src/refine_stereo_extrinsics.py", "--help"]
