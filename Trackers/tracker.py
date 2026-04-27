from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd

import config as _cfg

"""
Object Tracking and Visualization Module

Handles multi-object tracking across video frames using YOLOv8 detection and ByteTrack.
Also manages drawing and annotation functions for visualization output.
"""

class Tracker:
    """
    Multi-object tracker for players and ball detection across video frames.
    
    Uses YOLOv8 for detection and ByteTrack for temporal association.
    Provides caching to avoid recomputing detections on subsequent runs.
    
    Attributes:
        model (YOLO): YOLOv8 detection model
        tracker (ByteTrack): Multi-object tracker instance
    """
    
    def __init__(self, model_path):
        """
        Initialize tracker with YOLO model.
        
        Args:
            model_path (str): Path to YOLO model weights (e.g., 'best_100e.pt')
        """
        self.model = YOLO(model_path)
        # We increase the buffer to 90 frames (approx 3-4 seconds) to keep IDs 
        # even if the player is occluded or blurred for a while.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=100, 
            minimum_matching_threshold=0.8
        )

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # 'iou' obliga a YOLO a borrar cajas que se solapen más de ese ratio.
            # Esto elimina el "doble bbox" de raíz.
            detections_batch = self.model.predict(
                frames[i:i+batch_size],
                conf=_cfg.YOLO_BALL_CONF,
                iou=_cfg.YOLO_BALL_IOU,
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        # 1. If we already have saved data, read it and save time
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            print("✅ Data loaded from cache! (stub)")
            return tracks

        # 2. Otherwise, we need to work: detect objects in frames
        print("🔍 Detecting objects in video (this may take a while)...")
        detections = self.detect_frames(frames)
        tracks = {
            "players": [],  # Sets of Bboxes per frame for players
            "referees": [], # Sets of Bboxes per frame for referees
            "ball": []      # Sets of Bboxes per frame for the ball
        }

        # Ball pipeline counters (logged once at the end of detection)
        ball_stats = {
            "accepted":   0,
            "rej_conf":   0,
            "rej_size":   0,   # too large OR too small
            "rej_aspect": 0,
            "rej_bounds": 0,   # crowd-mask (handled later by filter_ball_positions_by_speed for pitch bounds)
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # --- FILTRO ANTI-FANTASMAS ---
            # 1. Aplicamos NMS extra por si YOLO falló
            # detection_supervision = detection_supervision.with_nms(threshold=0.5)
            
            # 2. Ignoramos todo lo que pase en el tercio superior de la pantalla (gradas)
            # Asumiendo que y=0 es arriba. Ajusta CROWD_MASK_Y_PX en config si corta cabezas de jugadores.
            mask = detection_supervision.xyxy[:, 1] > _cfg.CROWD_MASK_Y_PX
            ball_class_id = cls_names_inv.get('ball')
            if ball_class_id is not None:
                rejected_balls = (~mask) & (detection_supervision.class_id == ball_class_id)
                ball_stats["rej_bounds"] += int(rejected_balls.sum())
            detection_supervision = detection_supervision[mask]

            # This prevents the tracker from getting confused if the label changes
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Tracking
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv['player']:
                    # Calculate foot position (center-bottom)
                    position = ( (bbox[0] + bbox[2])/2, bbox[3] )
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": position 
                        }
                if cls_id == cls_names_inv['referee']:
                    position = ( (bbox[0] + bbox[2])/2, bbox[3] )
                    tracks["referees"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": position 
                        }

            # --- BALL DETECTION: pick best candidate per frame ---
            # Problems with the naive loop:
            #   1. Multiple "ball" detections (real ball + sock + pitch stain) all get
            #      written to key 1, so the LAST one (lowest confidence, since YOLO sorts
            #      desc) overwrites the real ball.
            #   2. No geometry check → elongated sock boxes or large stain boxes pass.
            #   3. No ball-specific confidence gate.
            # Fix: collect all candidates, apply shape/confidence gates, keep best conf.
            best_ball_conf = -1
            best_ball_entry = None

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                conf  = float(frame_detection[2]) if frame_detection[2] is not None else 0.0

                if cls_id != cls_names_inv['ball']:
                    continue

                # Gate 1: minimum confidence for ball
                if conf < _cfg.BALL_MIN_CONF:
                    ball_stats["rej_conf"] += 1
                    continue

                # Gate 2: size — ball cannot be large (socks, stains, heads are bigger)
                # nor sub-pixel small (compression noise on grass)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if (w > _cfg.BALL_MAX_BBOX_PX or h > _cfg.BALL_MAX_BBOX_PX or
                    w < _cfg.BALL_MIN_BBOX_PX or h < _cfg.BALL_MIN_BBOX_PX):
                    ball_stats["rej_size"] += 1
                    continue

                # Gate 3: shape — ball is roughly square; socks are tall, stains are flat
                aspect = w / h if h > 0 else 999
                if aspect < _cfg.BALL_MIN_ASPECT or aspect > _cfg.BALL_MAX_ASPECT:
                    ball_stats["rej_aspect"] += 1
                    continue

                # Keep the highest-confidence detection that passed all gates
                if conf > best_ball_conf:
                    best_ball_conf = conf
                    best_ball_entry = {
                        "bbox": bbox,
                        "position": ((bbox[0] + bbox[2]) / 2, bbox[3]),
                        "confidence": conf,
                    }

            if best_ball_entry is not None:
                tracks["ball"][frame_num][1] = best_ball_entry
                ball_stats["accepted"] += 1

        print(
            "⚽ Ball pipeline: "
            f"aceptados {ball_stats['accepted']}, "
            f"rechazados por conf {ball_stats['rej_conf']}, "
            f"por size {ball_stats['rej_size']}, "
            f"por aspect {ball_stats['rej_aspect']}, "
            f"por bounds {ball_stats['rej_bounds']}"
        )

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    # Drawing functions
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # Bottom part of the box (feet) 
        x_center, _ = self.get_center_of_bbox(bbox)
        width = self.get_bbox_width(bbox)

        # Draw the ellipse under the feet
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Optional: Draw rectangle with the ID number
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = (y2 - rectangle_height//2) + 15
            y2_rect = (y2 + rectangle_height//2) + 15

            if track_id is not None:

                cv2.rectangle(frame,
                              (int(x1_rect), int(y1_rect)),
                              (int(x2_rect), int(y2_rect)),
                              color,
                              cv2.FILLED)

                x1_text = x1_rect + 12
                if track_id > 99:
                    x1_text -= 10

                cv2.putText(
                    frame,
                    f"{track_id}",
                    (int(x1_text), int(y1_rect+15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,0),
                    2
                )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1]) # Upper part of the box 
        x, _ = self.get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        # Draw inverted triangle over the ball
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # Black outlline 
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() # Copy so as not to spoil the original 
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players (Dynamic)
            for track_id, player in player_dict.items():
                # Retrieve the color computed by Main. If not found, use red as a fallback.
                color = player.get("team_color", (0, 0, 255))
                # Now pass that 'color' variable instead of the fixed one
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # If they have the ball, draw an extra red triangle on top
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Referees (Yellow circle)

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255), track_id)

            # Draw Ball (Green triangle)
            for track_id, ball in ball_dict.items():

                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            
            
            output_video_frames.append(frame)



        return output_video_frames

# -------------------------------------
    # Utilities for export and calculations
    def get_center_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1+x2)/2), int((y1+y2)/2)
    def get_bbox_width(self, bbox):
        return bbox[2] - bbox[0]

    # Ball interpolation
    def interpolate_ball_positions(self, ball_positions):
        # 1. Data extraction
        ball_bboxes = []
        for frame_tracks in ball_positions:
            if frame_tracks:
                bbox = list(frame_tracks.values())[0].get("bbox", [])
                ball_bboxes.append(bbox)
            else:
                ball_bboxes.append([np.nan, np.nan, np.nan, np.nan])

        df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # 2. Interpolation — gap length and direction live in config so we can tune
        # short-occlusion behaviour without touching code.
        df_ball_positions = df_ball_positions.interpolate(
            method='linear',
            limit=_cfg.BALL_INTERP_LIMIT,
            limit_direction=_cfg.BALL_INTERP_DIRECTION,
        )

        # 3. Reconstruction 
        ball_positions_interpolated = []
        for i, row in df_ball_positions.iterrows():
            ball_positions_interpolated.append({})
            
            # Only save if we have valid data
            if not np.isnan(row['x1']):
                bbox = [row['x1'], row['y1'], row['x2'], row['y2']]
                
                # We need to recalculate position based on the new interpolated bbox
                position = ((bbox[0] + bbox[2])/2, bbox[3])

                ball_positions_interpolated[i][1] = {
                    "bbox": bbox,
                    "position": position  # <--- AHORA SÍ LO GUARDAMOS
                }

        return ball_positions_interpolated

    def filter_ball_positions_by_speed(self, ball_tracks, fps, max_speed_mps=55.0,
                                       pitch_length=105.0, pitch_width=68.0, pitch_margin=5.0):
        """
        Two-stage ball false-positive filter using real-world meter coordinates.

        Stage A — Pitch bounds guard
            Any detection whose position_transformed falls outside the pitch rectangle
            (plus a small margin) is removed immediately.  "Sky" and stands false
            positives always land far outside the field in homography space.
            Critically, this prevents a sky false-positive from becoming the speed-filter
            anchor and then rejecting all real subsequent detections.

        Stage B — Speed plausibility guard
            Consecutive detections that imply the ball travelled faster than
            max_speed_mps are removed.  The anchor is only updated by detections that
            passed Stage A AND are reachable from the previous valid position.

        Must be called AFTER view_transformer.add_transformed_position_to_tracks()
        so that 'position_transformed' is present in each track entry.

        Args:
            ball_tracks (list[dict]): Per-frame ball track dicts (tracks["ball"]).
            fps (float): Video frame rate.
            max_speed_mps (float): Max allowed speed in m/s (default 55 ≈ 200 km/h).
            pitch_length (float): Pitch length in metres (default 105).
            pitch_width (float): Pitch width in metres (default 68).
            pitch_margin (float): Tolerance beyond pitch edge before discarding (default 5 m).

        Returns:
            list[dict]: Filtered ball_tracks.
        """
        max_dist_per_frame = max_speed_mps / fps
        x_min = -pitch_margin
        x_max = pitch_length + pitch_margin
        y_min = -pitch_margin
        y_max = pitch_width  + pitch_margin

        last_valid_frame = None
        last_valid_pos   = None
        removed_bounds   = 0
        removed_speed    = 0

        for frame_num, frame_ball in enumerate(ball_tracks):
            if not frame_ball or 1 not in frame_ball:
                continue

            pos_transformed = frame_ball[1].get('position_transformed')
            if pos_transformed is None:
                # No homography for this frame — skip, don't update anchor
                continue

            pos = np.array(pos_transformed, dtype=float).flatten()

            # --- Stage A: pitch bounds ---
            px, py = pos[0], pos[1]
            if not (x_min <= px <= x_max and y_min <= py <= y_max):
                ball_tracks[frame_num] = {}
                removed_bounds += 1
                continue

            # --- Stage B: speed plausibility ---
            if last_valid_pos is not None:
                frames_elapsed = frame_num - last_valid_frame
                dist           = np.linalg.norm(pos - last_valid_pos)
                max_allowed    = max_dist_per_frame * frames_elapsed

                if dist > max_allowed:
                    ball_tracks[frame_num] = {}
                    removed_speed += 1
                    continue  # Don't update anchor — wait for next reachable detection

            last_valid_pos   = pos
            last_valid_frame = frame_num

        kept = sum(1 for f in ball_tracks if f and 1 in f)
        print(
            "⚽ Ball pipeline (post-transform): "
            f"aceptados {kept}, "
            f"rechazados por bounds {removed_bounds}, "
            f"por speed {removed_speed} (limit {max_speed_mps} m/s)"
        )

        return ball_tracks

    def filter_static_ball_clusters(self, ball_tracks, fps=None,
                                    radius_m=None, window_frames=None):
        """
        Drop ball detections that barely move in real-world meters across a sliding
        window — those are almost always pitch stains, white socks or the centre
        circle, not the actual ball.

        Operates on `position_transformed` (metres), so it must run AFTER
        view_transformer.add_transformed_position_to_tracks().

        Args:
            ball_tracks (list[dict]): Per-frame ball track dicts (tracks["ball"]).
            fps (float|None): Unused, accepted for API symmetry with the speed filter.
            radius_m (float|None): Override BALL_STATIC_RADIUS_M.
            window_frames (int|None): Override BALL_STATIC_WINDOW_FRAMES.

        Returns:
            list[dict]: ball_tracks with static clusters cleared in-place.
        """
        radius        = _cfg.BALL_STATIC_RADIUS_M       if radius_m       is None else radius_m
        window        = _cfg.BALL_STATIC_WINDOW_FRAMES  if window_frames  is None else window_frames

        n = len(ball_tracks)
        if n == 0 or window <= 1:
            print("⚽ Ball pipeline (static-cluster): aceptados 0, rechazados por static 0")
            return ball_tracks

        # Pre-extract transformed positions per frame, NaN where missing.
        positions = np.full((n, 2), np.nan, dtype=float)
        for i, frame_ball in enumerate(ball_tracks):
            if frame_ball and 1 in frame_ball:
                p = frame_ball[1].get('position_transformed')
                if p is not None:
                    arr = np.asarray(p, dtype=float).flatten()
                    if arr.size >= 2:
                        positions[i, 0] = arr[0]
                        positions[i, 1] = arr[1]

        to_drop = np.zeros(n, dtype=bool)

        # Sliding window: if every detection inside [i, i+window) lies within `radius`
        # of the window centroid, the whole window is static → drop those frames.
        for start in range(0, n - window + 1):
            end = start + window
            block = positions[start:end]
            valid = ~np.isnan(block[:, 0])
            # Need the window to be (mostly) populated; require at least half full
            # so a single sticky detection in an empty stretch doesn't trigger it.
            if valid.sum() < max(2, window // 2):
                continue

            pts = block[valid]
            centroid = pts.mean(axis=0)
            dists = np.linalg.norm(pts - centroid, axis=1)
            if dists.max() < radius:
                idxs = np.where(valid)[0] + start
                to_drop[idxs] = True

        removed = 0
        for i in range(n):
            if to_drop[i] and ball_tracks[i] and 1 in ball_tracks[i]:
                ball_tracks[i] = {}
                removed += 1

        kept = sum(1 for f in ball_tracks if f and 1 in f)
        print(
            "⚽ Ball pipeline (static-cluster): "
            f"aceptados {kept}, rechazados por static {removed} "
            f"(radius {radius} m, window {window} frames)"
        )

        return ball_tracks

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # 1. Transparent overlay (same as before)
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 2. NumPy logic (THE IMPORTANT CHANGE)
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
        # Count using NumPy (much faster)
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)
        
        # 3. Calculate percentages (same as before)
        total_frames = team_1_num_frames + team_2_num_frames
        
        if total_frames == 0:
            team_1_perc = 0
            team_2_perc = 0
        else:
            team_1_perc = team_1_num_frames / total_frames
            team_2_perc = team_2_num_frames / total_frames

        # 4. Write text (same as before)
        cv2.putText(frame, f"Team 1 Possession: {team_1_perc*100:.1f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Possession: {team_2_perc*100:.1f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame