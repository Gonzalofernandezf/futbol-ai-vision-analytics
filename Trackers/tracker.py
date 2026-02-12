from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd

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
        self.tracker = sv.ByteTrack(lost_track_buffer=60)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
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

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

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

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    position = ( (bbox[0] + bbox[2])/2, bbox[3] )
                    tracks["ball"][frame_num][1] = {
                        "bbox": bbox, 
                        "position": position 
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

        # 2. Interpolation
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit=5, limit_direction='both')
        df_ball_positions = df_ball_positions.bfill()

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