#!/usr/bin/env python3
"""
Chessboard Camera Calibration Studio
------------------------------------
Single-camera and stereo-pair chessboard calibration, restyled to match
the AprilTag Multi-Zoom Calibration Studio industrial UI.

    pip install opencv-python numpy
"""

import os
import glob
import json
import math
import queue
import pickle
import threading
import tkinter as tk

from pathlib import Path
from datetime import datetime
from tkinter import ttk, filedialog, messagebox

import _opencv_cuda  # noqa: F401  (must import before cv2)
import cv2
import numpy as np


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "Chessboard Camera Calibration Studio"

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT = SCRIPT_DIR / "chessboard_calibration"

DEFAULT_BOARD_COLS = 8   # internal corners
DEFAULT_BOARD_ROWS = 5
DEFAULT_SQUARE_MM = 30.0

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

MIN_IMAGES_RECOMMENDED = 10


# ============================================================
# COLORS (mirrors the zoom tool)
# ============================================================

BG = "#eef1f4"
PANEL = "#ffffff"
PANEL2 = "#e2e8ee"

ACCENT = "#0a5fa8"
ACCENT2 = "#c1620c"

TEXT = "#1b2530"
MUTED = "#5b6b7a"

SUCCESS = "#1c7d43"
WARNING = "#a8650a"
ERROR = "#b3261e"

BORDER = "#c3ccd6"

PREVIEW_BG = "#12181f"
PREVIEW_TEXT = "#aab6c2"
CONSOLE_BG = "#12181f"
CONSOLE_TEXT = "#cfe3f2"


# ============================================================
# FOV
# ============================================================

def calculate_fov(fx, fy, width, height):

    hfov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))

    f_eq = math.sqrt(fx * fy)
    diag = math.sqrt(width * width + height * height)
    dfov = math.degrees(2.0 * math.atan(diag / (2.0 * f_eq)))

    return hfov, vfov, dfov


# ============================================================
# REPROJECTION ERROR
# ============================================================

def calculate_reprojection_errors(
    object_points,
    image_points,
    rvecs,
    tvecs,
    camera_matrix,
    distortion
):

    all_errs = []
    per_image = []

    for i in range(len(object_points)):

        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            distortion
        )

        projected = projected.reshape(-1, 2)
        observed = image_points[i].reshape(-1, 2)

        errs = np.linalg.norm(observed - projected, axis=1)

        all_errs.extend(errs.tolist())

        per_image.append({
            "index": i,
            "mean_error_px": float(np.mean(errs)),
            "rms_error_px": float(math.sqrt(np.mean(errs ** 2))),
            "max_error_px": float(np.max(errs)),
        })

    if not all_errs:
        return 0.0, 0.0, 0.0, []

    arr = np.asarray(all_errs)

    return (
        float(np.mean(arr)),
        float(math.sqrt(np.mean(arr ** 2))),
        float(np.max(arr)),
        per_image,
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

class ChessboardCalibrationApp:

    def __init__(self, root):

        self.root = root

        self.root.title(APP_NAME)
        self.root.geometry("1450x900")
        self.root.minsize(1200, 760)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.capL = None
        self.capR = None

        self.preview_running = False

        self.last_frameL = None
        self.last_frameR = None

        self.calibration_running = False
        self.stop_calibration_event = threading.Event()

        self.log_queue = queue.Queue()

        self.img_count = 0

        # ----------------------------------------------------
        # Variables
        # ----------------------------------------------------

        self.mode_var = tk.StringVar(value="single")

        self.cam_left_var = tk.StringVar(value="0")
        self.cam_right_var = tk.StringVar(value="1")

        self.resolution_var = tk.StringVar(value="1280x720")
        self.width_var = tk.StringVar(value=str(DEFAULT_WIDTH))
        self.height_var = tk.StringVar(value=str(DEFAULT_HEIGHT))

        self.camera_role_var = tk.StringVar(value="boresight")

        self.board_cols_var = tk.StringVar(value=str(DEFAULT_BOARD_COLS))
        self.board_rows_var = tk.StringVar(value=str(DEFAULT_BOARD_ROWS))
        self.square_var = tk.StringVar(value=str(DEFAULT_SQUARE_MM))

        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT))

        self.status_var = tk.StringVar(value="READY")
        self.corner_status_var = tk.StringVar(value="Corners: --")
        self.resolution_status_var = tk.StringVar(value="Resolution: ---")
        self.capture_status_var = tk.StringVar(value="CAPTURE READY")
        self.saved_status_var = tk.StringVar(value="Saved: 0")

        # ----------------------------------------------------
        # Build UI
        # ----------------------------------------------------

        self.configure_style()
        self.build_header()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        self.capture_tab = tk.Frame(self.notebook, bg=BG)
        self.calibration_tab = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.capture_tab, text="  ①  CAPTURE DATA  ")
        self.notebook.add(self.calibration_tab, text="  ②  CALIBRATE  ")

        self.build_capture_tab()
        self.build_calibration_tab()

        self.root.bind("<KeyPress-s>", self.save_frame_key)
        self.root.bind("<space>", self.save_frame_key)
        self.root.bind("<Escape>", lambda e: self.stop_preview())

        self.process_log_queue()

    # ========================================================
    # STYLE
    # ========================================================

    def configure_style(self):

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=BG, foreground=TEXT, font=("Ubuntu", 10))

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)

        style.configure(
            "TButton",
            background=PANEL2, foreground=TEXT,
            padding=(12, 8), borderwidth=0,
        )
        style.map(
            "TButton",
            background=[("active", "#d2dbe4")],
            foreground=[("active", ACCENT)],
        )

        style.configure(
            "Accent.TButton",
            background=ACCENT, foreground="#ffffff",
            font=("Ubuntu", 10, "bold"), padding=(14, 9),
        )
        style.map("Accent.TButton", background=[("active", "#0a4f8c")])

        style.configure(
            "Danger.TButton",
            background="#f2d4d1", foreground="#7d1712", padding=(12, 8),
        )
        style.map("Danger.TButton", background=[("active", "#e8b9b4")])

        style.configure(
            "TEntry",
            fieldbackground=PANEL2, foreground=TEXT,
            insertcolor=TEXT, bordercolor=BORDER,
        )
        style.configure(
            "TCombobox",
            fieldbackground=PANEL2, background=PANEL2, foreground=TEXT,
        )

        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=PANEL, foreground=MUTED,
            padding=(20, 10), font=("Ubuntu", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", PANEL2)],
            foreground=[("selected", ACCENT)],
        )

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=PANEL2, background=ACCENT, borderwidth=0,
        )

        style.configure(
            "TRadiobutton",
            background=PANEL, foreground=TEXT,
        )
        style.map(
            "TRadiobutton",
            background=[("active", PANEL)],
            foreground=[("active", ACCENT)],
        )

    # ========================================================
    # HEADER
    # ========================================================

    def build_header(self):

        header = tk.Frame(self.root, bg=BG, height=75)
        header.pack(fill="x", padx=20, pady=(15, 8))

        left = tk.Frame(header, bg=BG)
        left.pack(side="left")

        tk.Label(
            left, text="CHESSBOARD",
            bg=BG, fg=ACCENT, font=("Ubuntu", 20, "bold"),
        ).pack(anchor="w")

        tk.Label(
            left, text="CAMERA CALIBRATION STUDIO",
            bg=BG, fg=TEXT, font=("Ubuntu", 12, "bold"),
        ).pack(anchor="w")

        tk.Label(
            left,
            text="OpenCV cv2.findChessboardCorners • Single & Stereo",
            bg=BG, fg=MUTED, font=("Ubuntu", 9),
        ).pack(anchor="w")

        right = tk.Frame(header, bg=BG)
        right.pack(side="right")

        self.status_badge = tk.Label(
            right, textvariable=self.status_var,
            bg=PANEL2, fg=ACCENT, padx=15, pady=7,
            font=("Ubuntu", 10, "bold"),
        )
        self.status_badge.pack(side="right")

    # ========================================================
    # PANEL / FIELD helpers
    # ========================================================

    def make_panel(self, parent, title):

        panel = tk.Frame(
            parent, bg=PANEL,
            highlightbackground=BORDER, highlightthickness=1,
        )

        tk.Label(
            panel, text=title.upper(),
            bg=PANEL, fg=ACCENT, font=("Ubuntu", 10, "bold"),
        ).pack(anchor="w", padx=15, pady=(13, 8))

        return panel

    def field(self, parent, label, variable, width=None):

        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="x", padx=15, pady=4)

        tk.Label(
            frame, text=label,
            bg=PANEL, fg=MUTED, font=("Ubuntu", 9),
        ).pack(anchor="w")

        entry = ttk.Entry(frame, textvariable=variable)

        if width:
            entry.configure(width=width)

        entry.pack(fill="x", pady=(3, 0))
        return entry

    # ========================================================
    # CAPTURE TAB
    # ========================================================

    def build_capture_tab(self):

        self.capture_tab.columnconfigure(1, weight=1)
        self.capture_tab.rowconfigure(0, weight=1)

        # ----- LEFT column -----
        left = tk.Frame(self.capture_tab, bg=BG, width=320)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        # Mode panel
        mode_panel = self.make_panel(left, "Mode")
        mode_panel.pack(fill="x", pady=(0, 10))

        mode_row = tk.Frame(mode_panel, bg=PANEL)
        mode_row.pack(fill="x", padx=15, pady=(0, 10))

        ttk.Radiobutton(
            mode_row, text="Single", variable=self.mode_var, value="single",
            command=self.on_mode_change,
        ).pack(side="left", padx=(0, 10))

        ttk.Radiobutton(
            mode_row, text="Stereo", variable=self.mode_var, value="stereo",
            command=self.on_mode_change,
        ).pack(side="left")

        # Camera panel
        camera_panel = self.make_panel(left, "Camera")
        camera_panel.pack(fill="x", pady=(0, 10))

        self.field(camera_panel, "LEFT / CAMERA INDEX", self.cam_left_var)

        self.right_entry_row = tk.Frame(camera_panel, bg=PANEL)
        # packed dynamically in on_mode_change
        tk.Label(
            self.right_entry_row, text="RIGHT CAMERA INDEX",
            bg=PANEL, fg=MUTED, font=("Ubuntu", 9),
        ).pack(anchor="w")
        ttk.Entry(
            self.right_entry_row, textvariable=self.cam_right_var,
        ).pack(fill="x", pady=(3, 0))

        ttk.Button(
            camera_panel, text="SCAN CAMERAS",
            command=self.on_scan_cameras,
        ).pack(fill="x", padx=15, pady=8)

        tk.Label(
            camera_panel, text="RESOLUTION",
            bg=PANEL, fg=MUTED, font=("Ubuntu", 9),
        ).pack(anchor="w", padx=15, pady=(5, 3))

        self.resolution_combo = ttk.Combobox(
            camera_panel, textvariable=self.resolution_var, state="readonly",
            values=["640x480", "1280x720", "1920x1080", "Custom"],
        )
        self.resolution_combo.pack(fill="x", padx=15)
        self.resolution_combo.bind("<<ComboboxSelected>>", self.resolution_changed)

        self.field(camera_panel, "WIDTH", self.width_var)
        self.field(camera_panel, "HEIGHT", self.height_var)

        tk.Label(
            camera_panel, text="CAMERA ROLE",
            bg=PANEL, fg=MUTED, font=("Ubuntu", 9),
        ).pack(anchor="w", padx=15, pady=(8, 3))

        ttk.Combobox(
            camera_panel, textvariable=self.camera_role_var, state="readonly",
            values=["boresight", "depression"],
        ).pack(fill="x", padx=15, pady=(0, 12))

        # Board panel
        board_panel = self.make_panel(left, "Chessboard")
        board_panel.pack(fill="x", pady=(0, 10))

        self.field(board_panel, "INTERNAL CORNERS (COLS)", self.board_cols_var)
        self.field(board_panel, "INTERNAL CORNERS (ROWS)", self.board_rows_var)
        self.field(board_panel, "SQUARE SIZE (mm)", self.square_var)

        tk.Label(
            board_panel,
            text="Internal corners = squares − 1 in each direction.",
            bg=PANEL, fg=MUTED, font=("Ubuntu", 8), wraplength=260, justify="left",
        ).pack(anchor="w", padx=15, pady=(0, 12))

        # Dataset panel
        dataset_panel = self.make_panel(left, "Dataset")
        dataset_panel.pack(fill="x", pady=(0, 10))

        self.field(dataset_panel, "OUTPUT DIRECTORY", self.output_var)

        ttk.Button(
            dataset_panel, text="BROWSE",
            command=self.browse_output,
        ).pack(fill="x", padx=15, pady=(7, 12))

        # ----- RIGHT column -----
        right = tk.Frame(self.capture_tab, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # Action bar
        action_bar = tk.Frame(
            right, bg=PANEL,
            highlightbackground=BORDER, highlightthickness=1,
        )
        action_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        tk.Label(
            action_bar, text="LIVE CAPTURE",
            bg=PANEL, fg=MUTED, font=("Ubuntu", 9, "bold"),
        ).pack(side="left", padx=(15, 15), pady=12)

        ttk.Button(
            action_bar, text="START PREVIEW",
            command=self.start_preview,
        ).pack(side="left", padx=4)

        ttk.Button(
            action_bar, text="STOP", style="Danger.TButton",
            command=self.stop_preview,
        ).pack(side="left", padx=4)

        ttk.Button(
            action_bar, text="CAPTURE (S)", style="Accent.TButton",
            command=self.save_frame,
        ).pack(side="left", padx=4)

        ttk.Button(
            action_bar, text="CLEAR OUTPUT", style="Danger.TButton",
            command=self.clear_output,
        ).pack(side="left", padx=4)

        # Preview area
        preview_panel = tk.Frame(
            right, bg=PREVIEW_BG,
            highlightbackground=BORDER, highlightthickness=1,
        )
        preview_panel.grid(row=1, column=0, sticky="nsew")

        self.preview_left_label = tk.Label(
            preview_panel, bg=PREVIEW_BG, fg=PREVIEW_TEXT,
            text="CAMERA OFFLINE\n\nStart Preview",
            font=("Ubuntu", 15),
        )
        self.preview_left_label.pack(side="left", fill="both", expand=True)

        self.preview_right_label = tk.Label(
            preview_panel, bg=PREVIEW_BG, fg=PREVIEW_TEXT,
            text="RIGHT OFFLINE",
            font=("Ubuntu", 15),
        )
        # only packed in stereo mode

        # Bottom info bar
        info = tk.Frame(right, bg=PANEL)
        info.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        tk.Label(
            info, textvariable=self.resolution_status_var,
            bg=PANEL, fg=MUTED,
        ).pack(side="left", padx=15, pady=10)

        tk.Label(
            info, textvariable=self.corner_status_var,
            bg=PANEL, fg=ACCENT,
        ).pack(side="left", padx=20)

        tk.Label(
            info, textvariable=self.saved_status_var,
            bg=PANEL, fg=SUCCESS,
        ).pack(side="left", padx=20)

        tk.Label(
            info, textvariable=self.capture_status_var,
            bg=PANEL, fg=ACCENT2,
        ).pack(side="right", padx=15)

    # ========================================================
    # CALIBRATION TAB
    # ========================================================

    def build_calibration_tab(self):

        self.calibration_tab.columnconfigure(1, weight=1)
        self.calibration_tab.rowconfigure(0, weight=1)

        # ----- LEFT -----
        left = tk.Frame(self.calibration_tab, bg=BG, width=300)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        settings = self.make_panel(left, "Calibration")
        settings.pack(fill="x", pady=(0, 10))

        self.field(settings, "INTERNAL CORNERS (COLS)", self.board_cols_var)
        self.field(settings, "INTERNAL CORNERS (ROWS)", self.board_rows_var)
        self.field(settings, "SQUARE SIZE (mm)", self.square_var)
        self.field(settings, "IMAGES FOLDER", self.output_var)

        tk.Label(
            settings, text="Mode follows the Capture tab.",
            bg=PANEL, fg=MUTED, font=("Ubuntu", 8),
        ).pack(anchor="w", padx=15, pady=(4, 8))

        self.calibrate_button = ttk.Button(
            settings, text="RUN CALIBRATION", style="Accent.TButton",
            command=self.run_calibration_thread,
        )
        self.calibrate_button.pack(fill="x", padx=15, pady=5)

        self.stop_calibration_button = ttk.Button(
            settings, text="STOP CALIBRATION", style="Danger.TButton",
            command=self.stop_calibration,
        )
        self.stop_calibration_button.pack(fill="x", padx=15, pady=5)

        ttk.Button(
            settings, text="OPEN DATASET FOLDER",
            command=self.open_dataset,
        ).pack(fill="x", padx=15, pady=(5, 12))

        # ----- MIDDLE -----
        middle = tk.Frame(self.calibration_tab, bg=BG)
        middle.grid(row=0, column=1, sticky="nsew")
        middle.columnconfigure(0, weight=1)
        middle.rowconfigure(1, weight=1)

        tk.Label(
            middle, text="DETECTION PREVIEW",
            bg=BG, fg=ACCENT, font=("Ubuntu", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        preview = tk.Frame(
            middle, bg=PREVIEW_BG,
            highlightbackground=BORDER, highlightthickness=1,
        )
        preview.grid(row=1, column=0, sticky="nsew")

        self.detected_label = tk.Label(
            preview, bg=PREVIEW_BG, fg=PREVIEW_TEXT,
            text="Run calibration to see detected corners.",
            font=("Ubuntu", 14),
        )
        self.detected_label.pack(fill="both", expand=True)

        self.detected_filename_label = tk.Label(
            middle, text="", bg=BG, fg=MUTED, font=("Ubuntu", 9),
        )
        self.detected_filename_label.grid(row=2, column=0, sticky="w", pady=(4, 0))

        # Result cards
        result_bar = tk.Frame(middle, bg=PANEL)
        result_bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        self.result_labels = {}

        for key, title in [
            ("fx", "FX"),
            ("fy", "FY"),
            ("cx", "CX"),
            ("cy", "CY"),
            ("rms", "RMS"),
            ("hfov", "HFOV"),
            ("vfov", "VFOV"),
            ("dfov", "DFOV"),
        ]:

            card = tk.Frame(result_bar, bg=PANEL2)
            card.pack(side="left", fill="both", expand=True, padx=2, pady=5)

            tk.Label(
                card, text=title, bg=PANEL2, fg=MUTED,
                font=("Ubuntu", 8, "bold"),
            ).pack(pady=(6, 0))

            label = tk.Label(
                card, text="---", bg=PANEL2, fg=TEXT,
                font=("Ubuntu", 10, "bold"),
            )
            label.pack(pady=(2, 7))

            self.result_labels[key] = label

        # ----- RIGHT LOG -----
        right = tk.Frame(self.calibration_tab, bg=BG, width=390)
        right.grid(row=0, column=2, sticky="ns", padx=(10, 0))

        tk.Label(
            right, text="CALIBRATION LOG",
            bg=BG, fg=ACCENT, font=("Ubuntu", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        log_frame = tk.Frame(
            right, bg=PANEL,
            highlightbackground=BORDER, highlightthickness=1,
        )
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_frame, bg=CONSOLE_BG, fg=CONSOLE_TEXT,
            insertbackground=TEXT, relief="flat", borderwidth=0,
            font=("Ubuntu Mono", 9), padx=10, pady=10, wrap="word",
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    # ========================================================
    # STATUS / LOG
    # ========================================================

    def set_status(self, text, color=ACCENT):

        def update():
            self.status_var.set(text)
            self.status_badge.configure(fg=color)

        self.root.after(0, update)

    def log(self, text):
        self.log_queue.put(str(text))

    def process_log_queue(self):

        try:
            while True:
                text = self.log_queue.get_nowait()
                self.log_text.insert("end", text + "\n")
                self.log_text.see("end")
        except queue.Empty:
            pass

        self.root.after(50, self.process_log_queue)

    # ========================================================
    # MODE
    # ========================================================

    def on_mode_change(self):

        if self.mode_var.get() == "stereo":
            self.right_entry_row.pack(fill="x", padx=15, pady=4)
            self.preview_right_label.pack(side="left", fill="both", expand=True)
        else:
            self.right_entry_row.pack_forget()
            self.preview_right_label.pack_forget()

        self.ensure_output_dirs()

    # ========================================================
    # RESOLUTION
    # ========================================================

    def resolution_changed(self, event=None):

        v = self.resolution_var.get()

        if v == "640x480":
            self.width_var.set("640"); self.height_var.set("480")
        elif v == "1280x720":
            self.width_var.set("1280"); self.height_var.set("720")
        elif v == "1920x1080":
            self.width_var.set("1920"); self.height_var.set("1080")

        if self.preview_running:
            self._apply_resolution()

    def _apply_resolution(self):
        try:
            w = int(self.width_var.get())
            h = int(self.height_var.get())
        except ValueError:
            return
        for cap in (self.capL, self.capR):
            if cap is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # ========================================================
    # OUTPUT
    # ========================================================

    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_var.set(folder)
            self.ensure_output_dirs()

    def ensure_output_dirs(self):
        out = self.output_var.get().strip()
        if not out:
            return
        try:
            Path(out).mkdir(parents=True, exist_ok=True)
            if self.mode_var.get() == "stereo":
                (Path(out) / "stereoLeft").mkdir(exist_ok=True)
                (Path(out) / "stereoRight").mkdir(exist_ok=True)
        except OSError as e:
            self.log(f"Could not create output dir '{out}': {e}")

    def open_dataset(self):
        folder = Path(self.output_var.get())
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(folder))
            else:
                os.system(f'xdg-open "{folder}"')
        except Exception:
            pass

    def clear_output(self):

        out = self.output_var.get().strip()
        if not out or not Path(out).exists():
            return

        if not messagebox.askyesno(
            "Clear output",
            f"Delete every .png in\n\n{out}\n\n(subfolders included)?",
            icon="warning",
        ):
            return

        removed = 0

        for pattern in ("*.png", "stereoLeft/*.png", "stereoRight/*.png"):
            for p in Path(out).glob(pattern):
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass

        self.img_count = 0
        self.saved_status_var.set("Saved: 0")
        self.log(f"CLEARED: removed {removed} image(s) from {out}")
        self.set_status("OUTPUT CLEARED", WARNING)

    # ========================================================
    # CAMERA OPEN / SCAN
    # ========================================================

    def _open_capture(self, index):
        # On Windows: keep the local dev flow (DirectShow / Media Foundation).
        # On Linux (Jetson): use a strict UYVY 1280x720@60 GStreamer pipeline
        # via v4l2src -> nvvidconv (VIC) -> BGR -> appsink. Requires an OpenCV
        # built with GStreamer (see OPENCV_CUDA_PREFIX / build_opencv_cuda.sh).
        if os.name == "nt":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            return cap

        device = f"/dev/video{index}"
        # STRICT: UYVY 1280x720@60 only. No fallback — if the camera can't
        # deliver this mode, opening fails and the caller reports it.
        pipeline = (
            f"v4l2src device={device} io-mode=2 ! "
            "video/x-raw,format=UYVY,width=1280,height=720,framerate=60/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    def on_scan_cameras(self):

        self.set_status("SCANNING CAMERAS", WARNING)

        working = []
        for i in range(6):
            cap = self._open_capture(i)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    working.append((i, w, h))
            cap.release()

        if working:
            summary = ", ".join(f"idx {i} ({w}x{h})" for i, w, h in working)
            self.set_status(f"{len(working)} CAMERA(S)", SUCCESS)
            self.log(f"Cameras found: {summary}")
            messagebox.showinfo("Cameras", "Working cameras:\n\n" + summary)
        else:
            self.set_status("NO CAMERA", ERROR)
            messagebox.showwarning("Cameras", "No working cameras on indices 0-5.")

    # ========================================================
    # PREVIEW
    # ========================================================

    def start_preview(self):

        if self.preview_running:
            return

        try:
            left_idx = int(self.cam_left_var.get())
        except ValueError:
            messagebox.showerror("Camera", "Invalid left camera index.")
            return

        self.capL = self._open_capture(left_idx)
        if not self.capL.isOpened():
            messagebox.showerror("Camera", f"Cannot open camera {left_idx}")
            self.capL = None
            return

        try:
            w = int(self.width_var.get())
            h = int(self.height_var.get())
            self.capL.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.capL.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        except ValueError:
            pass

        if self.mode_var.get() == "stereo":
            try:
                right_idx = int(self.cam_right_var.get())
            except ValueError:
                messagebox.showerror("Camera", "Invalid right camera index.")
                self.capL.release(); self.capL = None
                return

            self.capR = self._open_capture(right_idx)
            if not self.capR.isOpened():
                messagebox.showerror("Camera", f"Cannot open camera {right_idx}")
                self.capL.release(); self.capL = None
                self.capR = None
                return

            try:
                self.capR.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.capR.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            except Exception:
                pass

        self.ensure_output_dirs()
        self.preview_running = True
        self.set_status("CAMERA LIVE", SUCCESS)
        self.update_preview()

    def stop_preview(self):

        self.preview_running = False

        if self.capL is not None:
            self.capL.release(); self.capL = None
        if self.capR is not None:
            self.capR.release(); self.capR = None

        self.set_status("CAMERA STOPPED", MUTED)

    def update_preview(self):

        if not self.preview_running or self.capL is None:
            return

        cols = self._int(self.board_cols_var, DEFAULT_BOARD_COLS)
        rows = self._int(self.board_rows_var, DEFAULT_BOARD_ROWS)

        ret, frameL = self.capL.read()

        if ret and frameL is not None:
            self.last_frameL = frameL.copy()

            gray = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
            )

            display = frameL.copy()
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)

            h, w = display.shape[:2]

            cv2.rectangle(display, (12, 12), (340, 82), (8, 14, 20), -1)
            cv2.putText(
                display, f"{w} x {h}",
                (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (200, 210, 220), 2,
            )
            cv2.putText(
                display,
                f"BOARD {cols}x{rows}  {'FOUND' if found else 'NO CORNERS'}",
                (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (0, 220, 150) if found else (0, 90, 220), 2,
            )

            self.display_frame(self.preview_left_label, display, 1050, 700)

            self.resolution_status_var.set(f"Resolution: {w} x {h}")
            self.corner_status_var.set(
                f"Corners: {cols * rows if found else 0} / {cols * rows}"
            )
            self.capture_status_var.set(
                "GOOD CAPTURE" if found else "MOVE BOARD / ADJUST LIGHT"
            )

        if self.mode_var.get() == "stereo" and self.capR is not None:
            retR, frameR = self.capR.read()
            if retR and frameR is not None:
                self.last_frameR = frameR.copy()

                grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
                foundR, cornersR = cv2.findChessboardCorners(
                    grayR, (cols, rows),
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
                )
                displayR = frameR.copy()
                cv2.drawChessboardCorners(displayR, (cols, rows), cornersR, foundR)
                self.display_frame(self.preview_right_label, displayR, 520, 700)

        self.root.after(30, self.update_preview)

    def display_frame(self, label, frame, max_w, max_h):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        scale = min(max_w / w, max_h / h, 1.0)

        if scale < 1.0:
            rgb = cv2.resize(
                rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        h, w = rgb.shape[:2]

        ppm = f"P6\n{w} {h}\n255\n".encode() + rgb.tobytes()

        photo = tk.PhotoImage(data=ppm, format="PPM")
        label.configure(image=photo, text="")
        label.image = photo

    # ========================================================
    # SAVE
    # ========================================================

    def save_frame_key(self, event=None):
        if self.notebook.index(self.notebook.select()) != 0:
            return
        self.save_frame()

    def save_frame(self):

        if self.last_frameL is None:
            messagebox.showwarning("Capture", "Start preview first.")
            return

        out = self.output_var.get().strip()
        self.ensure_output_dirs()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        if self.mode_var.get() == "single":
            role = self.camera_role_var.get()
            path = Path(out) / f"{role}_{self.img_count:03d}_{ts}.png"
            cv2.imwrite(str(path), self.last_frameL)
            self.log(f"CAPTURED | {path.name}")

        else:
            if self.last_frameR is None:
                messagebox.showwarning("Capture", "No right frame yet.")
                return
            pL = Path(out) / "stereoLeft"  / f"imgL_{self.img_count:03d}_{ts}.png"
            pR = Path(out) / "stereoRight" / f"imgR_{self.img_count:03d}_{ts}.png"
            cv2.imwrite(str(pL), self.last_frameL)
            cv2.imwrite(str(pR), self.last_frameR)
            self.log(f"CAPTURED | {pL.name} + {pR.name}")

        self.img_count += 1
        self.saved_status_var.set(f"Saved: {self.img_count}")
        self.capture_status_var.set("SAVED")

    # ========================================================
    # CALIBRATION
    # ========================================================

    def _int(self, var, default):
        try:
            return int(var.get())
        except ValueError:
            return default

    def _float(self, var, default):
        try:
            return float(var.get())
        except ValueError:
            return default

    def run_calibration_thread(self):

        if self.calibration_running:
            return

        self.stop_calibration_event.clear()
        self.calibration_running = True
        self.calibrate_button.configure(state="disabled")
        self.log_text.delete("1.0", "end")

        threading.Thread(target=self.run_calibration, daemon=True).start()

    def stop_calibration(self):
        if self.calibration_running:
            self.stop_calibration_event.set()
            self.log("STOP REQUESTED...")

    def run_calibration(self):

        try:
            mode = self.mode_var.get()
            cols = self._int(self.board_cols_var, DEFAULT_BOARD_COLS)
            rows = self._int(self.board_rows_var, DEFAULT_BOARD_ROWS)
            square = self._float(self.square_var, DEFAULT_SQUARE_MM)
            out = self.output_var.get().strip()

            self.ensure_output_dirs()

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001,
            )

            objp = np.zeros((cols * rows, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            objp *= square

            if mode == "single":
                self.calibrate_single(out, cols, rows, objp, criteria)
            else:
                self.calibrate_stereo(out, cols, rows, objp, criteria)

        except Exception as e:
            self.log(f"CALIBRATION ERROR: {e}")
            self.root.after(0, lambda: messagebox.showerror("Calibration", str(e)))

        finally:
            self.calibration_running = False
            self.root.after(0, self._finish_calibration_ui)

    def _finish_calibration_ui(self):
        self.calibrate_button.configure(state="normal")
        if self.stop_calibration_event.is_set():
            self.set_status("CALIBRATION STOPPED", WARNING)
        else:
            self.set_status("CALIBRATION COMPLETE", SUCCESS)

    # -------- Single camera --------

    def calibrate_single(self, out, cols, rows, objp, criteria):

        self.log("=" * 42)
        self.log("SINGLE CAMERA CALIBRATION")
        self.log("=" * 42)

        images = sorted(glob.glob(os.path.join(out, "*.png")))

        if not images:
            self.log(f"No .png images found in {out}")
            return

        detected_dir = Path(out) / "detected"
        detected_dir.mkdir(exist_ok=True)

        object_points, image_points = [], []
        frame_size = None

        for fname in images:

            if self.stop_calibration_event.is_set():
                self.log("Calibration stopped.")
                return

            img = cv2.imread(fname)
            if img is None:
                self.log(f"  FAIL {os.path.basename(fname)}: unreadable")
                continue

            if frame_size is None:
                frame_size = (img.shape[1], img.shape[0])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

            if found:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria,
                )
                object_points.append(objp)
                image_points.append(corners)
                self.log(f"  OK   {os.path.basename(fname)}")

                annotated = img.copy()
                cv2.drawChessboardCorners(annotated, (cols, rows), corners, True)
                cv2.imwrite(str(detected_dir / os.path.basename(fname)), annotated)
                self._show_detected(annotated, os.path.basename(fname), True)
            else:
                self.log(f"  FAIL {os.path.basename(fname)}: no corners")
                self._show_detected(img, os.path.basename(fname), False)

        if not object_points:
            self.log("\nNo valid images.")
            return

        if len(object_points) < MIN_IMAGES_RECOMMENDED:
            self.log(
                f"\nWARNING: only {len(object_points)} valid images "
                f"({MIN_IMAGES_RECOMMENDED}+ recommended)."
            )

        self.log(f"\nRunning cv2.calibrateCamera() on {len(object_points)} image(s)...")

        rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, frame_size, None, None,
        )

        mean_err, rms_err, max_err, _ = calculate_reprojection_errors(
            object_points, image_points, rvecs, tvecs,
            camera_matrix, dist,
        )

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])

        hfov, vfov, dfov = calculate_fov(fx, fy, frame_size[0], frame_size[1])

        self.log("")
        self.log("------------ RESULT ------------")
        self.log(f"Resolution : {frame_size[0]} x {frame_size[1]}")
        self.log(f"Images     : {len(object_points)}")
        self.log(f"OpenCV RMS : {rms:.6f} px")
        self.log(f"Mean error : {mean_err:.6f} px")
        self.log(f"RMS error  : {rms_err:.6f} px")
        self.log(f"Max error  : {max_err:.6f} px")
        self.log(f"fx = {fx:.4f}   fy = {fy:.4f}")
        self.log(f"cx = {cx:.4f}   cy = {cy:.4f}")
        self.log(f"HFOV = {hfov:.4f} deg   VFOV = {vfov:.4f} deg   DFOV = {dfov:.4f} deg")
        self.log(f"Distortion: {dist.reshape(-1)}")

        # persist
        out_p = Path(out)

        with open(out_p / "calibration.pkl", "wb") as f:
            pickle.dump((camera_matrix, dist), f)

        result = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist.reshape(-1).tolist(),
            "image_width": int(frame_size[0]),
            "image_height": int(frame_size[1]),
            "board_cols": int(cols),
            "board_rows": int(rows),
            "square_size_mm": float(self._float(self.square_var, DEFAULT_SQUARE_MM)),
            "num_images_used": len(object_points),
            "opencv_rms": float(rms),
            "rms_reprojection_error_px": float(rms_err),
            "mean_reprojection_error_px": float(mean_err),
            "max_reprojection_error_px": float(max_err),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "hfov_deg": float(hfov),
            "vfov_deg": float(vfov),
            "dfov_deg": float(dfov),
            "aspect_fy_over_fx": float(fy / fx if fx > 0 else 0.0),
            "calibration_timestamp": datetime.now().isoformat(),
        }

        with open(out_p / "calibration.json", "w") as f:
            json.dump(result, f, indent=4)

        # OpenCV YAML mirror
        fs = cv2.FileStorage(str(out_p / "calibration.yaml"), cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", camera_matrix)
        fs.write("distortion_coefficients", dist)
        fs.write("image_width", int(frame_size[0]))
        fs.write("image_height", int(frame_size[1]))
        fs.write("board_cols", int(cols))
        fs.write("board_rows", int(rows))
        fs.write("square_size_mm", float(self._float(self.square_var, DEFAULT_SQUARE_MM)))
        fs.write("opencv_rms", float(rms))
        fs.write("rms_reprojection_error_px", float(rms_err))
        fs.write("mean_reprojection_error_px", float(mean_err))
        fs.write("max_reprojection_error_px", float(max_err))
        fs.write("fx", fx); fs.write("fy", fy)
        fs.write("cx", cx); fs.write("cy", cy)
        fs.write("hfov_deg", float(hfov))
        fs.write("vfov_deg", float(vfov))
        fs.write("dfov_deg", float(dfov))
        fs.release()

        self.root.after(
            0, self._update_result_cards,
            fx, fy, cx, cy, rms_err, hfov, vfov, dfov,
        )

        self.log(f"\nSaved: calibration.pkl / calibration.json / calibration.yaml -> {out}")
        self.log(f"Annotated images -> {detected_dir}")
        self.log("CALIBRATION COMPLETE.")

    # -------- Stereo --------

    def calibrate_stereo(self, out, cols, rows, objp, criteria):

        self.log("=" * 42)
        self.log("STEREO CAMERA CALIBRATION")
        self.log("=" * 42)

        left_dir = Path(out) / "stereoLeft"
        right_dir = Path(out) / "stereoRight"

        images_L = sorted(glob.glob(str(left_dir  / "*.png")))
        images_R = sorted(glob.glob(str(right_dir / "*.png")))

        if not images_L or not images_R:
            self.log(f"No images in {left_dir} or {right_dir}")
            return

        if len(images_L) != len(images_R):
            self.log("WARNING: left/right counts differ - pairing by sorted order.")

        detected_L_dir = Path(out) / "detected_left"
        detected_R_dir = Path(out) / "detected_right"
        detected_L_dir.mkdir(exist_ok=True)
        detected_R_dir.mkdir(exist_ok=True)

        object_points, points_L, points_R = [], [], []
        frame_size = None

        for fL, fR in zip(images_L, images_R):

            if self.stop_calibration_event.is_set():
                self.log("Calibration stopped.")
                return

            imgL = cv2.imread(fL)
            imgR = cv2.imread(fR)
            if imgL is None or imgR is None:
                continue

            if frame_size is None:
                frame_size = (imgL.shape[1], imgL.shape[0])

            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            retL, cornersL = cv2.findChessboardCorners(grayL, (cols, rows), None)
            retR, cornersR = cv2.findChessboardCorners(grayR, (cols, rows), None)

            if retL and retR:
                cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
                cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

                object_points.append(objp)
                points_L.append(cornersL)
                points_R.append(cornersR)

                self.log(f"  OK   {os.path.basename(fL)} / {os.path.basename(fR)}")

                annotL = imgL.copy()
                annotR = imgR.copy()
                cv2.drawChessboardCorners(annotL, (cols, rows), cornersL, True)
                cv2.drawChessboardCorners(annotR, (cols, rows), cornersR, True)
                cv2.imwrite(str(detected_L_dir / os.path.basename(fL)), annotL)
                cv2.imwrite(str(detected_R_dir / os.path.basename(fR)), annotR)
                self._show_detected(annotL, os.path.basename(fL), True)
            else:
                self.log(f"  FAIL {os.path.basename(fL)} / {os.path.basename(fR)}")
                self._show_detected(imgL, os.path.basename(fL), False)

        if not object_points:
            self.log("\nNo valid pairs.")
            return

        self.log(f"\nCalibrating each camera separately on {len(object_points)} pair(s)...")

        _, K_L, D_L, _, _ = cv2.calibrateCamera(
            object_points, points_L, frame_size, None, None,
        )
        _, K_R, D_R, _, _ = cv2.calibrateCamera(
            object_points, points_R, frame_size, None, None,
        )

        self.log("Running cv2.stereoCalibrate() with CALIB_FIX_INTRINSIC...")

        flags = cv2.CALIB_FIX_INTRINSIC
        stereo_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100, 1e-5,
        )

        rms_stereo, K_L, D_L, K_R, D_R, R, T, E, F = cv2.stereoCalibrate(
            object_points, points_L, points_R,
            K_L, D_L, K_R, D_R, frame_size,
            criteria=stereo_criteria, flags=flags,
        )

        rect_L, rect_R, proj_L, proj_R, Q, roi_L, roi_R = cv2.stereoRectify(
            K_L, D_L, K_R, D_R, frame_size, R, T, alpha=1, newImageSize=(0, 0),
        )

        map_L = cv2.initUndistortRectifyMap(
            K_L, D_L, rect_L, proj_L, frame_size, cv2.CV_16SC2,
        )
        map_R = cv2.initUndistortRectifyMap(
            K_R, D_R, rect_R, proj_R, frame_size, cv2.CV_16SC2,
        )

        baseline = float(np.linalg.norm(T))

        fx_L = float(K_L[0, 0]); fy_L = float(K_L[1, 1])
        cx_L = float(K_L[0, 2]); cy_L = float(K_L[1, 2])
        hfov, vfov, dfov = calculate_fov(fx_L, fy_L, frame_size[0], frame_size[1])

        self.log("")
        self.log("------------ RESULT ------------")
        self.log(f"Resolution   : {frame_size[0]} x {frame_size[1]}")
        self.log(f"Pairs used   : {len(object_points)}")
        self.log(f"Stereo RMS   : {rms_stereo:.6f} px")
        self.log(f"Baseline |T| : {baseline:.4f} mm")
        self.log(f"Rotation R   :\n{R}")
        self.log(f"Translation T:\n{T.reshape(-1)}")

        out_p = Path(out)

        cv_file = cv2.FileStorage(str(out_p / "stereoMap.xml"), cv2.FILE_STORAGE_WRITE)
        cv_file.write("stereoMapL_x", map_L[0])
        cv_file.write("stereoMapL_y", map_L[1])
        cv_file.write("stereoMapR_x", map_R[0])
        cv_file.write("stereoMapR_y", map_R[1])
        cv_file.write("Q", Q)
        cv_file.release()

        with open(out_p / "stereo_calibration.pkl", "wb") as f:
            pickle.dump((K_L, D_L, K_R, D_R, R, T, Q), f)

        result = {
            "camera_matrix_left":  K_L.tolist(),
            "camera_matrix_right": K_R.tolist(),
            "distortion_coefficients_left":  D_L.reshape(-1).tolist(),
            "distortion_coefficients_right": D_R.reshape(-1).tolist(),
            "rotation_left_to_right":      R.tolist(),
            "translation_left_to_right":   T.reshape(-1).tolist(),
            "essential_matrix":            E.tolist(),
            "fundamental_matrix":          F.tolist(),
            "Q":                           Q.tolist(),
            "stereo_rms":                  float(rms_stereo),
            "baseline_mm":                 baseline,
            "image_width":  int(frame_size[0]),
            "image_height": int(frame_size[1]),
            "board_cols": int(cols),
            "board_rows": int(rows),
            "square_size_mm": float(self._float(self.square_var, DEFAULT_SQUARE_MM)),
            "num_pairs_used": len(object_points),
            "hfov_deg_left": float(hfov),
            "vfov_deg_left": float(vfov),
            "dfov_deg_left": float(dfov),
            "calibration_timestamp": datetime.now().isoformat(),
        }

        with open(out_p / "stereo_calibration.json", "w") as f:
            json.dump(result, f, indent=4)

        self.root.after(
            0, self._update_result_cards,
            fx_L, fy_L, cx_L, cy_L, rms_stereo, hfov, vfov, dfov,
        )

        self.log(
            f"\nSaved: stereo_calibration.pkl / stereo_calibration.json / "
            f"stereoMap.xml -> {out}"
        )
        self.log(f"Annotated -> {detected_L_dir}, {detected_R_dir}")
        self.log("CALIBRATION COMPLETE.")

    # ========================================================
    # UI updates from worker
    # ========================================================

    def _show_detected(self, img, filename, found):

        color = (0, 200, 0) if found else (0, 0, 220)
        bordered = cv2.copyMakeBorder(img, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=color)

        def _do():
            self.display_frame(self.detected_label, bordered, 780, 620)
            status = "FOUND" if found else "NOT FOUND"
            self.detected_filename_label.configure(text=f"{filename}  —  {status}")

        self.root.after(0, _do)

    def _update_result_cards(self, fx, fy, cx, cy, rms, hfov, vfov, dfov):

        values = {
            "fx": f"{fx:.2f}",
            "fy": f"{fy:.2f}",
            "cx": f"{cx:.2f}",
            "cy": f"{cy:.2f}",
            "rms": f"{rms:.3f}",
            "hfov": f"{hfov:.2f}°",
            "vfov": f"{vfov:.2f}°",
            "dfov": f"{dfov:.2f}°",
        }

        for key, value in values.items():
            self.result_labels[key].configure(text=value)

    # ========================================================
    # CLOSE
    # ========================================================

    def on_close(self):

        self.preview_running = False
        self.stop_calibration_event.set()

        for cap in (self.capL, self.capR):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    app = ChessboardCalibrationApp(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()


if __name__ == "__main__":
    main()
