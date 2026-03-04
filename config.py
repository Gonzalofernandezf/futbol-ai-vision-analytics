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

# Visualization
DISTANCE_THRESHOLD_PIXELS = float(os.getenv("DISTANCE_THRESHOLD_PIXELS", "60"))

# Logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
