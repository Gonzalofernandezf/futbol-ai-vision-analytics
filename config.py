"""
Configuration management for Football AI Vision Analytics.

Centralized settings for paths, model parameters, and processing options.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# File paths
VIDEO_PATH = os.getenv("VIDEO_PATH", "video_OG.mp4")
MODEL_PATH = os.getenv("MODEL_PATH", "best_100e.pt")
STUB_PATH = os.getenv("STUB_PATH", os.path.join(PROJECT_ROOT, "stubs", "track_stubs.pkl"))

# Output directories
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output_videos"))
DEMO_DIR = os.getenv("DEMO_DIR", os.path.join(PROJECT_ROOT, "demo_dashboard"))

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STUB_PATH), exist_ok=True)

# YOLO model parameters
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.10"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.5"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")
YOLO_TRACKER = os.getenv("YOLO_TRACKER", "bytetrack.yaml")

# Processing parameters
FRAME_WINDOW = int(os.getenv("FRAME_WINDOW", "5"))
MAX_SPEED_KMH = float(os.getenv("MAX_SPEED_KMH", "45"))
MIN_TRACK_DURATION = float(os.getenv("MIN_TRACK_DURATION", "0.5"))
MAX_SPEED_GAP_FRAMES = int(os.getenv("MAX_SPEED_GAP_FRAMES", "30"))

# Calibration parameters
FIELD_WIDTH_METERS = float(os.getenv("FIELD_WIDTH_METERS", "68"))

# Ball tracking
BALL_MAX_SPEED_MPS  = float(os.getenv("BALL_MAX_SPEED_MPS",  "55.0"))  # ~200 km/h, world-record upper bound

# Ball detection gates (applied per-frame before any tracking)
BALL_MIN_CONF       = float(os.getenv("BALL_MIN_CONF",       "0.50"))  # higher than player threshold (0.35)
BALL_MAX_BBOX_PX    = int  (os.getenv("BALL_MAX_BBOX_PX",    "90"))    # max width OR height in pixels (socks/stains are larger)
BALL_MIN_ASPECT     = float(os.getenv("BALL_MIN_ASPECT",     "0.35"))  # min width/height ratio (socks are tall & thin)
BALL_MAX_ASPECT     = float(os.getenv("BALL_MAX_ASPECT",     "2.80"))  # max width/height ratio (stains are wide & flat)

# Pitch field dimensions for out-of-bounds guard (FIFA standard, meters)
PITCH_LENGTH_M      = float(os.getenv("PITCH_LENGTH_M",      "100.0"))
PITCH_WIDTH_M       = float(os.getenv("PITCH_WIDTH_M",       "64.0"))
PITCH_MARGIN_M      = float(os.getenv("PITCH_MARGIN_M",      "5.0"))   # tolerance beyond edge before discarding

# Visualization
DISTANCE_THRESHOLD_PIXELS = float(os.getenv("DISTANCE_THRESHOLD_PIXELS", "60"))

# Logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
