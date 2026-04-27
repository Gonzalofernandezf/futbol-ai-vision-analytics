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
BALL_MAX_SPEED_MPS  = float(os.getenv("BALL_MAX_SPEED_MPS",  "40.0"))  # tuned for U-20 footage, well above realistic shots

# YOLO inference thresholds for the ball pass (separate from the global YOLO_CONF/IOU)
YOLO_BALL_CONF      = float(os.getenv("YOLO_BALL_CONF",      "0.35"))  # confidence floor for the YOLO ball pass
YOLO_BALL_IOU       = float(os.getenv("YOLO_BALL_IOU",       "0.4"))   # NMS IoU for the YOLO ball pass

# Crowd / stands mask: ignore detections whose top-Y is above this many pixels
CROWD_MASK_Y_PX     = int  (os.getenv("CROWD_MASK_Y_PX",     "80"))

# Ball detection gates (applied per-frame before any tracking)
BALL_MIN_CONF       = float(os.getenv("BALL_MIN_CONF",       "0.35"))  # post-detection confidence gate
BALL_MAX_BBOX_PX    = int  (os.getenv("BALL_MAX_BBOX_PX",    "90"))    # max width OR height in pixels (socks/stains are larger)
BALL_MIN_BBOX_PX    = int  (os.getenv("BALL_MIN_BBOX_PX",    "4"))     # min width AND height in pixels (sub-pixel noise)
BALL_MIN_ASPECT     = float(os.getenv("BALL_MIN_ASPECT",     "0.70"))  # min width/height ratio (ball is roughly square)
BALL_MAX_ASPECT     = float(os.getenv("BALL_MAX_ASPECT",     "1.40"))  # max width/height ratio (ball is roughly square)

# Ball interpolation
BALL_INTERP_LIMIT     = int(os.getenv("BALL_INTERP_LIMIT",     "10"))     # max consecutive NaNs to fill
BALL_INTERP_DIRECTION = os.getenv("BALL_INTERP_DIRECTION",     "both")    # 'forward' | 'backward' | 'both'

# Static-ball cluster filter (post-transform): if the ball barely moves in real-world
# meters across a window, those detections are almost certainly a stain / socks.
BALL_STATIC_RADIUS_M       = float(os.getenv("BALL_STATIC_RADIUS_M",       "0.5"))
BALL_STATIC_WINDOW_FRAMES  = int  (os.getenv("BALL_STATIC_WINDOW_FRAMES",  "30"))

# Pitch field dimensions for out-of-bounds guard (FIFA standard, meters)
PITCH_LENGTH_M      = float(os.getenv("PITCH_LENGTH_M",      "105.0"))
PITCH_WIDTH_M       = float(os.getenv("PITCH_WIDTH_M",       "68.0"))
PITCH_MARGIN_M      = float(os.getenv("PITCH_MARGIN_M",      "5.0"))   # tolerance beyond edge before discarding

# Visualization
DISTANCE_THRESHOLD_PIXELS = float(os.getenv("DISTANCE_THRESHOLD_PIXELS", "60"))

# Logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
