"""
Dedicated YOLO ball detector.

Lives separately from the player tracker so the ball can use its own model
weights (best_ball_v1.pt), its own confidence/IoU thresholds, and its own
GPU device — without polluting the player detection thresholds.

Inference output matches the shape Main.py already expects for tracks["ball"]:
    list[dict]  — one entry per frame
    each entry is {} (no ball this frame) or {1: {"bbox", "position", "confidence"}}

Downstream ball post-processing (interpolation, speed filter, static-cluster
filter) lives in Trackers/tracker.py and consumes this list directly — no
changes needed there.
"""

import os
import pickle

import numpy as np
from ultralytics import YOLO

import config as _cfg


class BallDetector:
    """YOLO inference + per-frame geometric gates dedicated to the ball."""

    def __init__(self, model_path=None):
        self.device = _cfg.BALL_DEVICE
        path = model_path or _cfg.BALL_MODEL_PATH
        self.model = YOLO(path)
        # Move weights to the target device up-front so the first predict()
        # call doesn't pay a cold-start cost (matters when running in a thread).
        if "cuda" in self.device:
            self.model.to(self.device)
        if _cfg.YOLO_HALF and "cuda" in self.device:
            self.model.model.half()
        print(
            f"⚽ Modelo balón: path={path}  device={self.device}  "
            f"clases={self.model.names}"
        )

    def detect_frames(self, frames):
        """Batched YOLO inference over `frames`. Returns a list of Results."""
        batch_size = _cfg.BALL_BATCH_SIZE
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_results = self.model.predict(
                frames[i:i + batch_size],
                conf=_cfg.YOLO_BALL_CONF,
                iou=_cfg.YOLO_BALL_IOU,
                imgsz=_cfg.YOLO_IMGSZ,
                device=self.device,
                verbose=False,
            )
            detections += batch_results
        return detections

    def get_ball_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Returns per-frame ball tracks in the same shape as the old
        Tracker.get_object_tracks output for the "ball" key:
            [{} or {1: {bbox, position, confidence}}, ...]
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_tracks = pickle.load(f)
            print("✅ Ball tracks loaded from cache (stub).")
            return ball_tracks

        print("⚽ Detecting ball in video...")
        detections = self.detect_frames(frames)

        ball_tracks = []
        stats = {
            "accepted":   0,
            "rej_conf":   0,
            "rej_size":   0,
            "rej_aspect": 0,
            "rej_bounds": 0,  # crowd-mask reject (stands / sky)
        }

        for detection in detections:
            boxes = detection.boxes
            best_conf = -1.0
            best_entry = None

            if boxes is not None and len(boxes) > 0:
                xyxy  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for box, conf in zip(xyxy, confs):
                    bbox = box.tolist()
                    conf = float(conf)

                    # Gate 0: crowd mask (top of frame = stands / sky)
                    if bbox[1] <= _cfg.CROWD_MASK_Y_PX:
                        stats["rej_bounds"] += 1
                        continue

                    # Gate 1: ball-specific confidence floor
                    if conf < _cfg.BALL_MIN_CONF:
                        stats["rej_conf"] += 1
                        continue

                    # Gate 2: size — too large = sock/leg/head, too small = noise
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if (w > _cfg.BALL_MAX_BBOX_PX or h > _cfg.BALL_MAX_BBOX_PX or
                        w < _cfg.BALL_MIN_BBOX_PX or h < _cfg.BALL_MIN_BBOX_PX):
                        stats["rej_size"] += 1
                        continue

                    # Gate 3: aspect — ball is roughly square
                    aspect = w / h if h > 0 else 999
                    if aspect < _cfg.BALL_MIN_ASPECT or aspect > _cfg.BALL_MAX_ASPECT:
                        stats["rej_aspect"] += 1
                        continue

                    if conf > best_conf:
                        best_conf = conf
                        best_entry = {
                            "bbox": bbox,
                            "position": ((bbox[0] + bbox[2]) / 2, bbox[3]),
                            "confidence": conf,
                        }

            if best_entry is not None:
                ball_tracks.append({1: best_entry})
                stats["accepted"] += 1
            else:
                ball_tracks.append({})

        print(
            "⚽ Ball pipeline (detector): "
            f"aceptados {stats['accepted']}, "
            f"rechazados por conf {stats['rej_conf']}, "
            f"por size {stats['rej_size']}, "
            f"por aspect {stats['rej_aspect']}, "
            f"por bounds {stats['rej_bounds']}"
        )

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_tracks, f)

        return ball_tracks
