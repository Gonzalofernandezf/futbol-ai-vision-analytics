"""
Configuration management for Football AI Vision Analytics.

Centralized settings for paths, model parameters, and processing options.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Versión del pipeline — mantener en sync con setup.py. Se reporta en
# processing_meta.json para que el dashboard (/admin) identifique el run.
PIPELINE_VERSION = "1.0.0"

# File paths
VIDEO_PATH = os.getenv("VIDEO_PATH", "video_OG.mp4")
MODEL_PATH = os.getenv("MODEL_PATH", "best_100e.pt")
MODELO_CANCHA_PATH = os.getenv("MODELO_CANCHA_PATH", "modelo_cancha.pt")
# Dedicated YOLO model for ball detection (trained separately, e.g. best_ball_v1.pt).
# Lives next to MODEL_PATH in the project root by default.
BALL_MODEL_PATH = os.getenv("BALL_MODEL_PATH", "best_ball_v1.pt")
STUB_PATH = os.getenv("STUB_PATH", os.path.join(PROJECT_ROOT, "stubs", "track_stubs.pkl"))
BALL_STUB_PATH = os.getenv("BALL_STUB_PATH", os.path.join(PROJECT_ROOT, "stubs", "ball_stub.pkl"))

# Output directories
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output_videos"))
# Carpeta pública del dashboard React/Vite — el pipeline deposita aquí los
# artefactos de runtime (match_data.json, demo_video.mp4, eval_*.json).
DEMO_DIR = os.getenv("DEMO_DIR", os.path.join(PROJECT_ROOT, "futbol-ai-dashboard", "public"))

# Artefactos de evaluación (eval/eval_keypoints.py escribe aquí por defecto).
# El pipeline los copia condicionalmente a DEMO_DIR en el deploy — relevante
# cuando DEMO_DIR apunta a otro sitio (p.ej. runs en Kaggle).
EVAL_ARTIFACTS_DIR = os.getenv("EVAL_ARTIFACTS_DIR", os.path.join(PROJECT_ROOT, "futbol-ai-dashboard", "public"))

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STUB_PATH), exist_ok=True)

# YOLO model parameters
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.10"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.5"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")
YOLO_TRACKER = os.getenv("YOLO_TRACKER", "bytetrack.yaml")
# FP16 inference: ~2x faster on CUDA GPUs (T4, P100, etc.) with no perceptible quality loss.
# Auto-disabled when running on CPU (FP16 on CPU is slower than FP32).
YOLO_HALF = os.getenv("YOLO_HALF", "true").lower() == "true" and YOLO_DEVICE != "cpu"

# Inference batch size. Larger = fewer kernel launches = better GPU saturation,
# at the cost of VRAM. On Kaggle T4 (15 GB) 40-60 is safe with FP16; OOM → lower.
YOLO_BATCH_SIZE_TRACKER = int(os.getenv("YOLO_BATCH_SIZE_TRACKER", "40"))
YOLO_BATCH_SIZE_FIELD   = int(os.getenv("YOLO_BATCH_SIZE_FIELD",   "40"))

# Per-model device override. Useful on multi-GPU hosts (Kaggle T4 x2):
#   YOLO_DEVICE_TRACKER=cuda:0  YOLO_DEVICE_FIELD=cuda:1  BALL_DEVICE=cuda:1
# Default: fall back to YOLO_DEVICE so single-GPU setups keep working.
YOLO_DEVICE_TRACKER = os.getenv("YOLO_DEVICE_TRACKER", YOLO_DEVICE)
YOLO_DEVICE_FIELD   = os.getenv("YOLO_DEVICE_FIELD",   YOLO_DEVICE)
BALL_DEVICE         = os.getenv("BALL_DEVICE",         YOLO_DEVICE)

# Ball detector batch size — model is small (YOLOv11m, ~40MB) so a larger batch
# fits comfortably alongside the field model on the same GPU.
BALL_BATCH_SIZE     = int(os.getenv("BALL_BATCH_SIZE", "40"))

# Run tracker, field-keypoint detection and camera-movement optical flow in
# parallel threads (they don't share state; YOLO + OpenCV release the GIL).
# Auto-disabled on CPU since three CPU-bound threads contend for the same core.
PARALLEL_INFERENCE = (
    os.getenv("PARALLEL_INFERENCE", "true").lower() == "true"
    and YOLO_DEVICE != "cpu"
)

# Processing parameters
FRAME_WINDOW = int(os.getenv("FRAME_WINDOW", "5"))
MAX_SPEED_KMH = float(os.getenv("MAX_SPEED_KMH", "32"))   # techo biomecánico para fútbol base/semipro
SPEED_MEDIAN_WINDOW = int(os.getenv("SPEED_MEDIAN_WINDOW", "3"))  # ventanas a medianizar (1 ventana = FRAME_WINDOW frames)
MIN_TRACK_DURATION = float(os.getenv("MIN_TRACK_DURATION", "0.5"))
MAX_SPEED_GAP_FRAMES = int(os.getenv("MAX_SPEED_GAP_FRAMES", "30"))

# Calibration parameters
FIELD_WIDTH_METERS = float(os.getenv("FIELD_WIDTH_METERS", "68"))

# Field keypoint detection (modelo_cancha.pt) — supports both pose & detect modes
FIELD_KP_CONF = float(os.getenv("FIELD_KP_CONF", "0.35"))  # confidence floor for field keypoints

# Ball tracking
BALL_MAX_SPEED_MPS  = float(os.getenv("BALL_MAX_SPEED_MPS",  "40.0"))  # tuned for U-20 footage, well above realistic shots

# YOLO inference thresholds for the ball pass (separate from the global YOLO_CONF/IOU)
YOLO_BALL_CONF      = float(os.getenv("YOLO_BALL_CONF",      "0.35"))  # confidence floor for the YOLO ball pass
YOLO_BALL_IOU       = float(os.getenv("YOLO_BALL_IOU",       "0.4"))   # NMS IoU for the YOLO ball pass
YOLO_AGNOSTIC_NMS   = os.getenv("YOLO_AGNOSTIC_NMS", "true").lower() == "true"

# ByteTrack parameters — exposed here so they can be tuned without touching tracker.py.
# lost_track_buffer ~7s at 30fps (200 frames) keeps IDs through real occlusions in the area.
# minimum_matching_threshold 0.7 is less strict than 0.8, helping lateral-camera footage.
BYTETRACK_ACTIVATION      = float(os.getenv("BYTETRACK_ACTIVATION",      "0.25"))
BYTETRACK_LOST_BUFFER     = int  (os.getenv("BYTETRACK_LOST_BUFFER",     "200"))
BYTETRACK_MATCH_THRESHOLD = float(os.getenv("BYTETRACK_MATCH_THRESHOLD", "0.7"))

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
PITCH_LENGTH_M      = float(os.getenv("PITCH_LENGTH_M",      "100.0"))
PITCH_WIDTH_M       = float(os.getenv("PITCH_WIDTH_M",       "64.0"))
PITCH_MARGIN_M      = float(os.getenv("PITCH_MARGIN_M",      "5.0"))   # tolerance beyond edge before discarding

# Visualization
DISTANCE_THRESHOLD_PIXELS = float(os.getenv("DISTANCE_THRESHOLD_PIXELS", "60"))

# Logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Chunked processing
CHUNK_DURATION_MIN = float(os.getenv("CHUNK_DURATION_MIN", "4"))    # minutes per chunk
CHUNK_OVERLAP_SEC  = float(os.getenv("CHUNK_OVERLAP_SEC",  "10"))   # overlap prepended to each non-first chunk
VIDEO_START_SEC    = float(os.getenv("VIDEO_START_SEC",    "0"))    # analysis start offset (seconds)
_video_end_raw     = os.getenv("VIDEO_END_SEC")
VIDEO_END_SEC      = float(_video_end_raw) if _video_end_raw is not None else None  # None = full video

# Frame scaling — reduce RAM when loading video
# 1.0 = original (~6 MB/frame at 1080p)   0.5 = half res (~1.5 MB, safe for 13 GB Kaggle + 4-min chunks)
# 0.33 = YOLO-native width (~0.67 MB, fallback for <8 GB environments)
FRAME_SCALE = float(os.getenv("FRAME_SCALE", "0.5"))

# MatchAnalytics v2 — derived-metric thresholds
SPRINT_THRESHOLD_KMH     = float(os.getenv("SPRINT_THRESHOLD_KMH",      "21.0"))
HI_THRESHOLD_KMH         = float(os.getenv("HI_THRESHOLD_KMH",          "15.0"))
HIGH_ACCEL_THRESHOLD_MS2 = float(os.getenv("HIGH_ACCEL_THRESHOLD_MS2",   "3.0"))
PEAK_WINDOW_SEC          = int  (os.getenv("PEAK_WINDOW_SEC",            "300"))
ANALYTICS_GRID_ROWS      = int  (os.getenv("ANALYTICS_GRID_ROWS",        "10"))
ANALYTICS_GRID_COLS      = int  (os.getenv("ANALYTICS_GRID_COLS",        "10"))
SPEED_ZONE_THRESHOLDS    = {
    "walk":   float(os.getenv("SPEED_ZONE_WALK_KMH", "7.0")),
    "jog":    float(os.getenv("SPEED_ZONE_JOG_KMH",  "15.0")),
    "run":    float(os.getenv("SPEED_ZONE_RUN_KMH",   "21.0")),
}

# Skip annotated video output — set True on memory-constrained envs (Kaggle, CI).
# JSON stats are always exported regardless of this flag.
SKIP_VIDEO_OUTPUT = os.getenv("SKIP_VIDEO_OUTPUT", "false").lower() == "true"

# Video output codec. vp09 = VP9 (browser-compatible but slow to encode ~0.5 fps on CPU).
# Use mp4v for fast encoding (~100 fps) when browser compatibility is not needed (e.g. Kaggle).
VIDEO_CODEC = os.getenv("VIDEO_CODEC", "vp09")
