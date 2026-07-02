import cv2
import sys
import numpy as np
sys.path.append("../")

from config import MAX_SPEED_KMH, SPEED_MEDIAN_WINDOW, ANNOTATION_SCALE

"""
Speed and Distance Calculation Module

Computes player speed (km/h), distance traveled, and acceleration (m/s²)
from tracked positions across frames using physics formulas.
"""

class SpeedAndDistance_Estimator():
    """
    Calculates speed, distance, and acceleration metrics from tracked positions.
    
    Uses a sliding window approach over multiple frames to smooth velocity
    estimates and reduce noise. Implements smart forward-fill to handle
    short tracking gaps while respecting maximum gap thresholds.
    
    Attributes:
        frame_window (int): Number of frames to average for velocity (default: 5)
        frame_rate (int): Video frame rate in fps (updated from video metadata)
    """
    
    def __init__(self):
        self.frame_window = 5 # We calculate velocity every 5 frames to smooth out noise.
        self.frame_rate = 24  # We assume 24fps by default, we will overwrite it if necessary.
    
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees": continue # We don't care the referees or balls speed for now 
            
            number_of_frames = len(object_tracks)
            
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames-1)

                for track_id, _ in object_tracks[frame_num].items():
                    # Only compute if the object exists in both frames (start and end of the window)
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Get the transformed positions (METERS)
                    start_position = object_tracks[frame_num][track_id].get('position_transformed')
                    end_position = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue
                    
                    # 1. Calculate Euclidean distance (basic mathematical formula)
                    distance = self.measure_distance(start_position, end_position)
                    
                    # 2. Calculate Speed
                    # Time = Number of frames / FPS
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    
                    if time_elapsed == 0: continue
                        
                    speed_meters_per_second = distance / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Techo biomecánico: descarta ventanas imposibles antes de acumular distancia.
                    # Causas típicas: jitter de homografía, ID switch, corte de vídeo.
                    if speed_km_per_hour > MAX_SPEED_KMH:
                        continue

                    # 3. Save data in the Tracker
                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

        # MEDIAN FILTER POR JUGADOR
        # Suaviza picos de velocidad causados por jitter de homografía o ID switches
        # que sobrevivieron al techo biomecánico (ej: 31 km/h cuando el jugador va a 12).
        # Opera sobre las velocidades ya asignadas por ventana antes del forward-fill.
        self._apply_speed_median_filter(tracks)

        # SMART FORWARD FILL (With tolerance limit)
        # Why 45: position_transformed returns None whenever the homography is
        # degraded (camera pan, pitch keypoints occluded). The speed window is
        # skipped for every such frame, so a 1-second pan at 30fps generates a
        # 30-frame gap with no speed data. 12 frames (~0.4 s) wasn't enough to
        # bridge that. 45 frames (~1.5 s) covers typical broadcast camera pans
        # while still "dying" cleanly when a player genuinely leaves the pitch.
        MAX_FRAME_GAP = 45

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees": continue
            
            # Temporary variables to control 'memory'
            last_valid_speed = None
            last_valid_distance = None
            gap_counter = 0 

            for frame_num in range(len(object_tracks)):
                # Iterate over the tracks of this frame
                # Note: There may be multiple players, so we search for the one we're tracking
                # But since tracks is structured by frame, we need to keep memory PER PLAYER ID.
                pass 
            
            # --- FIX: The data structure requires iterating by player ID first ---
            # To do this correctly, first collect all unique IDs appearing in the video
            unique_track_ids = set()
            for frame_data in object_tracks:
                unique_track_ids.update(frame_data.keys())
            
            # Now process the timeline of EACH individual player
            for track_id in unique_track_ids:
                last_speed = None
                last_dist = None
                frames_since_last_valid = 0
                
                for frame_num in range(len(object_tracks)):
                    if track_id in object_tracks[frame_num]:
                        track_info = object_tracks[frame_num][track_id]
                        
                        # CASE 1: We have real data. Reset counters.
                        if "speed" in track_info:
                            last_speed = track_info['speed']
                            last_dist = track_info['distance']
                            frames_since_last_valid = 0
                        
                        # CASE 2: No data, but we have recent memory (< MAX_GAP)
                        elif last_speed is not None and frames_since_last_valid < MAX_FRAME_GAP:
                            track_info['speed'] = last_speed
                            track_info['distance'] = last_dist
                            frames_since_last_valid += 1
                            
                        # CASE 3: A long time passed without data. Do not fill (let it die).
                        else:
                            frames_since_last_valid += 1

    def _apply_speed_median_filter(self, tracks):
        """
        Aplica un filtro de mediana deslizante sobre la serie de velocidad de cada jugador.
        Elimina picos aislados (1-2 ventanas) que pasaron el techo biomecánico pero
        siguen siendo inconsistentes con el movimiento real (ej. 30 km/h rodeado de 12 km/h).
        La ventana opera en unidades de frame_window para no romper los saltos naturales
        entre mediciones no contiguas.
        """
        half = SPEED_MEDIAN_WINDOW // 2

        for object, object_tracks in tracks.items():
            if object in ("ball", "referees"):
                continue

            unique_ids = set()
            for frame_data in object_tracks:
                unique_ids.update(frame_data.keys())

            for track_id in unique_ids:
                # Recoger frames donde hay velocidad calculada (no forward-fill aún)
                frames_with_speed = [
                    fn for fn in range(len(object_tracks))
                    if track_id in object_tracks[fn]
                    and "speed" in object_tracks[fn][track_id]
                ]
                if len(frames_with_speed) < 3:
                    continue

                speeds = [object_tracks[fn][track_id]["speed"] for fn in frames_with_speed]

                # Mediana deslizante sobre la lista de velocidades
                smoothed = []
                for i, fn in enumerate(frames_with_speed):
                    lo = max(0, i - half)
                    hi = min(len(speeds), i + half + 1)
                    smoothed.append(float(np.median(speeds[lo:hi])))

                for i, fn in enumerate(frames_with_speed):
                    object_tracks[fn][track_id]["speed"] = smoothed[i]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy() # Don't overwrite original

            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees": continue

                for track_id, track_info in object_tracks[frame_num].items():
                    # Try to get the data. If they don't exist, return None.
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)

                        # If there is no data (neither calculated nor inherited), skip to the next.
                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        offset = round(10 * ANNOTATION_SCALE)
                        position = (int(bbox[2]), int(bbox[3])) # bottom right corner
                        position = (position[0] + offset, position[1]) # a little to the right

                        # Draw texts
                        font_scale, thickness_bg, thickness_fg, line_gap = self._text_style()
                        cv2.putText(frame, f"{speed:.1f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_bg)
                        cv2.putText(frame, f"{speed:.1f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_fg)

                        cv2.putText(frame, f"{distance:.1f} m", (position[0], position[1]+line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_bg)
                        cv2.putText(frame, f"{distance:.1f} m", (position[0], position[1]+line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_fg)

            output_frames.append(frame)

        return output_frames

    def draw_speed_and_distance_frame(self, frame, frame_num, tracks):
        """Dibuja velocidad y distancia sobre un único frame (para render en streaming)."""
        offset = round(10 * ANNOTATION_SCALE)
        font_scale, thickness_bg, thickness_fg, line_gap = self._text_style()
        for object, object_tracks in tracks.items():
            if object in ("ball", "referees"):
                continue
            for track_id, track_info in object_tracks[frame_num].items():
                speed = track_info.get('speed', None)
                distance = track_info.get('distance', None)
                if speed is None or distance is None:
                    continue
                bbox = track_info['bbox']
                position = (int(bbox[2]) + offset, int(bbox[3]))
                cv2.putText(frame, f"{speed:.1f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_bg)
                cv2.putText(frame, f"{speed:.1f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_fg)
                cv2.putText(frame, f"{distance:.1f} m", (position[0], position[1] + line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_bg)
                cv2.putText(frame, f"{distance:.1f} m", (position[0], position[1] + line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_fg)
        return frame

    @staticmethod
    def _text_style():
        """Tamaño/grosor de fuente escalados por ANNOTATION_SCALE (ver config.py),
        con piso mínimo para que el texto siga siendo legible a escalas bajas."""
        font_scale = max(0.3, round(0.5 * ANNOTATION_SCALE, 3))
        thickness_bg = max(1, round(2 * ANNOTATION_SCALE))
        thickness_fg = max(1, round(1 * ANNOTATION_SCALE))
        line_gap = max(10, round(20 * ANNOTATION_SCALE))
        return font_scale, thickness_bg, thickness_fg, line_gap

    def measure_distance(self, p1, p2):
        # Pythagoras: a^2 + b^2 = c^2
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5