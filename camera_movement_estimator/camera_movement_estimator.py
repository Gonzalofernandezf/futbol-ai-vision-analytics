import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')

"""
Camera Movement Estimation Module

Estimates camera motion across frames using optical flow on field features.
Adjusts player positions by subtracting camera movement to create a 
stationary camera perspective.
"""

class CameraMovementEstimator():
    """
    Estimates and compensates for camera movement in video.
    
    Uses Lucas-Kanade optical flow on detected field features (corners, lines)
    to calculate frame-to-frame camera movement. Subtracts this movement
    from player positions to normalize coordinates.
    
    Attributes:
        features (dict): Parameters for goodFeaturesToTrack detection
        lk_params (dict): Parameters for Lucas-Kanade optical flow
    """
    
    def __init__(self, frame):
        # Parameters to detect features (corners, white field lines)
        # It uses goodFeaturesToTrack with these values:
        self.features = dict(maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7, mask=None)
        
        # Parameters for Optical Flow (Lucas-Kanade)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        # This function is KEY: Adjusts player positions by subtracting camera movement
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    
                    # SUBTRACT the camera movement from the player's position
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read from cache if it exists (to avoid recalculating each time)
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        # Convertir primer frame a escala de grises
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Detect initial key points
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            
            # Compute Optical Flow
            new_features, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            if new_features is not None:
                good_new = new_features[status==1]
                good_old = old_features[status==1]
                
                for new, old in zip(good_new, good_old):
                    diff_camera_movement_x = new[0] - old[0]
                    diff_camera_movement_y = new[1] - old[1]
                    
                    camera_movement_x += diff_camera_movement_x
                    camera_movement_y += diff_camera_movement_y
                    
                    max_distance = max(max_distance, (diff_camera_movement_x**2 + diff_camera_movement_y**2)**0.5 )

                # Filter: If points move a lot, discard them (may be noise)
                if max_distance > 100:
                    camera_movement_x = 0
                    camera_movement_y = 0

                if len(good_new) > 0:
                    camera_movement_x /= len(good_new)
                    camera_movement_y /= len(good_new)
                
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                
                # Update points for the next frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            
            cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames