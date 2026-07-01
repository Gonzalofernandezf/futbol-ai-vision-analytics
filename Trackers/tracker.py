from ultralytics import YOLO
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import yaml
import tempfile

import config as _cfg

class Tracker:
    def __init__(self, model_path):
        self.device = _cfg.YOLO_DEVICE_TRACKER
        self.model = YOLO(model_path)
        if "cuda" in self.device:
            self.model.to(self.device)
        if _cfg.YOLO_HALF and "cuda" in self.device:
            self.model.model.half()
        self._botsort_yaml = self._build_botsort_yaml()

    def __del__(self):
        try:
            if hasattr(self, '_botsort_yaml') and os.path.exists(self._botsort_yaml):
                os.unlink(self._botsort_yaml)
        except Exception:
            pass

    def _build_botsort_yaml(self):
        """Generate a BoT-SORT YAML from config.py env-overridable params."""
        cfg = {
            "tracker_type":     "botsort",
            "track_high_thresh": _cfg.BOTSORT_TRACK_HIGH_THRESH,
            "track_low_thresh":  _cfg.BOTSORT_TRACK_LOW_THRESH,
            "new_track_thresh":  _cfg.BOTSORT_NEW_TRACK_THRESH,
            "track_buffer":      _cfg.BOTSORT_TRACK_BUFFER,
            "match_thresh":      _cfg.BOTSORT_MATCH_THRESH,
            "gmc_method":        _cfg.BOTSORT_GMC_METHOD,
            "proximity_thresh":  _cfg.BOTSORT_PROXIMITY_THRESH,
            "appearance_thresh": _cfg.BOTSORT_APPEARANCE_THRESH,
            "with_reid":         _cfg.BOTSORT_WITH_REID,
            "model":             _cfg.BOTSORT_REID_MODEL,
            # Explicit defaults for fields added in newer ultralytics so our YAML
            # is self-contained and doesn't depend on the runtime's default.yaml.
            "fuse_score":        False,
        }
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="botsort_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f)
        return path

    def track_frames(self, frames):
        """Run YOLO tracking with BoT-SORT + ReID across all frames."""
        batch_size = _cfg.YOLO_BATCH_SIZE_TRACKER
        results = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = self.model.track(
                batch,
                persist=True,
                tracker=self._botsort_yaml,
                conf=_cfg.YOLO_CONF,
                iou=_cfg.YOLO_IOU,
                imgsz=_cfg.YOLO_IMGSZ,
                device=self.device,
                verbose=False,
                agnostic_nms=_cfg.YOLO_AGNOSTIC_NMS,
            )
            results += batch_results
        return results

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            print("✅ Data loaded from cache! (stub)")
            return tracks

        print("🔍 Tracking players/referees with BoT-SORT + ReID...")
        detections = self.track_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            boxes = detection.boxes
            if boxes is None or boxes.id is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)

            for idx in range(len(track_ids)):
                bbox = xyxy[idx].tolist()

                # Crowd mask
                if bbox[1] <= _cfg.CROWD_MASK_Y_PX:
                    continue

                cls_id = cls_ids[idx]
                track_id = int(track_ids[idx])
                cls_name = cls_names.get(cls_id, "")

                # Remap goalkeeper → player
                if cls_name == "goalkeeper":
                    cls_name = "player"

                position = ((bbox[0] + bbox[2]) / 2, bbox[3])

                if cls_name == "player":
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": position,
                    }
                elif cls_name == "referee":
                    tracks["referees"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": position,
                    }

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

    def draw_annotations_frame(self, frame, frame_num, tracks):
        """Dibuja anotaciones sobre un único frame (para render en streaming)."""
        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get('has_ball', False):
                frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

        for track_id, referee in referee_dict.items():
            frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id)

        for track_id, ball in ball_dict.items():
            frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

        return frame

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