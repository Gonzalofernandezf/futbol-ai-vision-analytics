import json
import numpy as np
from analytics.match_analytics import compute_player_derived, compute_team_stats

"""
Data Export Module

Exports tracking data and match statistics to JSON format.
Handles NumPy type conversions for JSON serialization and calculates
per-player metrics (speed, distance, acceleration).
"""

# This class helps convert odd NumPy objects into normal Python types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class GameStatsExporter:
    """
    Exports match tracking data and statistics to JSON format.
    
    Handles data transformation from internal tracking format to
    JSON-serializable output with position metrics, team data,
    possession statistics, and per-player performance indicators.
    
    Attributes:
        fps (int): Video frame rate for time-based metric conversions
    """
    
    def __init__(self, fps=24):
        self.fps = fps

    def _export_ball(self, ball_tracks):
        """Balón, en el mismo formato que un jugador (position_history en metros,
        un punto por frame; speed_over_time en km/h, uno por segundo). Necesario
        para el análisis de balón parado — ver analytics/set_piece_detector.py."""
        position_history = []
        speeds_by_second = {}
        prev_pos = None
        prev_frame = None
        fps_int = round(self.fps) or 1

        for frame_num, frame_ball in enumerate(ball_tracks):
            pos = None
            if frame_ball and 1 in frame_ball:
                pos = frame_ball[1].get('position_transformed')
                if pos is not None:
                    pos = [round(float(pos[0]), 2), round(float(pos[1]), 2)]
            position_history.append(pos)

            if pos is not None and prev_pos is not None:
                dt = (frame_num - prev_frame) / self.fps
                if dt > 0:
                    dist_m = ((pos[0] - prev_pos[0]) ** 2 + (pos[1] - prev_pos[1]) ** 2) ** 0.5
                    speed_kmh = (dist_m / dt) * 3.6
                    speeds_by_second[frame_num // fps_int] = round(speed_kmh, 2)
            if pos is not None:
                prev_pos, prev_frame = pos, frame_num

        total_seconds = len(ball_tracks) // fps_int + 1
        speed_over_time = [speeds_by_second.get(s) for s in range(total_seconds)]

        return {
            "position_history": position_history,
            "speed_over_time": speed_over_time,
        }

    def _export_set_pieces(self, set_pieces):
        """Convierte frames -> segundos para que el dashboard pueda saltar el
        vídeo directamente (currentTime = frame_start_sec)."""
        out = []
        fps = self.fps or 1
        for event in set_pieces:
            out.append({
                "type": event["type"],
                "frame_start": event["frame_start"],
                "frame_end": event["frame_end"],
                "start_sec": round(event["frame_start"] / fps, 2),
                "end_sec": round(event["frame_end"] / fps, 2),
                "position_m": event.get("position_m"),
            })
        return out

    # NOW (Add arguments with default value 0 just in case):
    def export_json(self, tracks, output_path="match_data.json", home_possession=0, away_possession=0,
                     view_transformer=None, set_pieces=None):
        # Final structure
        export_data = {
            "schema_version": 2,
            "match_meta": {
                "duration_seconds": 0,
                "fps": self.fps,
                "home_possession": round(home_possession, 1),
                "away_possession": round(away_possession, 1)
            },
            "players": {},
            "ball": self._export_ball(tracks.get("ball", [])),
            "set_pieces": self._export_set_pieces(set_pieces or []),
        }

        player_stats_temp = {}
        total_frames = len(tracks['players'])
        export_data["match_meta"]["duration_seconds"] = total_frames / self.fps

        print(" 💾  Procesando datos para exportación web...")

        for frame_num, frame_players in enumerate(tracks['players']):
            for player_id, info in frame_players.items():
                
                # CRITICAL FIX: Convert the ID to a string (text)
                # This avoids the "int64" error because JSON prefers string keys
                pid_str = str(player_id)

                if pid_str not in player_stats_temp:
                    player_stats_temp[pid_str] = {
                        "team": info.get('team', 0),
                        "speeds": [],
                        "distances": [],
                        "speed_by_second": {}
                    }

                speed = info.get('speed', 0)
                distance = info.get('distance', 0)
                bbox = info.get('bbox', None) # Recuperamos la caja del jugador

                if speed is None: speed = 0
                if distance is None: distance = 0

                player_stats_temp[pid_str]["speeds"].append(speed)
                player_stats_temp[pid_str]["distances"].append(distance)

                # NEW: CALCULATION OF POSITION (X, Y) IN METERS
                position = None
                if bbox is not None and view_transformer is not None:
                    try:
                        import numpy as np # Importamos la herramienta matemática
                        
                        # 1. FIX: Convert the list to a Numpy array
                        foot_position = np.array([(bbox[0] + bbox[2])/2, bbox[3]])
                        
                        # 2. Transform
                        position = view_transformer.transform_point(foot_position, frame_num)
                        
                        # 3. Convert and clean (take out extra parenthesis)
                        if position is not None:
                            # A. Convert Numpy to list
                            if hasattr(position, 'tolist'):
                                position = position.tolist()
                            elif hasattr(position, '__iter__'):
                                position = list(position)
                            
                            # B. FLATTEN: If it's [[x,y]], keep [x,y]
                            # Check if it is a list within a list
                            if isinstance(position, list) and len(position) > 0 and isinstance(position[0], list):
                                position = position[0]
                                
                    except Exception as e:
                        # If it fails, print a warning but DO NOT break the program
                        if frame_num == 0: print(f"⚠️ Warning en transformación: {e}")
                        position = None
                    
                    # If the transformation fails (returns None), save None
                    if position is not None:
                        # Convert numpy array to a regular list  [x, y]
                        position = position.tolist() if hasattr(position, 'tolist') else list(position)
                
                if "positions" not in player_stats_temp[pid_str]:
                    player_stats_temp[pid_str]["positions"] = []

                # Append the position (or None when the player isn't visible / homography failed).
                # We deliberately do NOT forward-fill None with the last known position:
                # a bad transform (garbage coords) would get propagated to many frames and
                # appear as a cluster at a pitch corner in the heatmap.
                # The HTML already skips null entries, so honest gaps are safe.
                player_stats_temp[pid_str]["positions"].append(position)

                fps_int = round(self.fps)
                if frame_num % fps_int == 0:
                    speed_val = info.get('speed', None)
                    if speed_val is not None:
                        second_idx = frame_num // fps_int
                        player_stats_temp[pid_str]["speed_by_second"][second_idx] = round(speed_val, 2)

        # Calculate final metrics
        # 2. Calculate final metrics and FILTER GHOSTS
        # REPLACE WHAT YOU HAVE INSIDE THE LOOP WITH THIS:
        for player_id, stats in player_stats_temp.items():
            
            # FILTER A: Appears for a very short time? (Noise / Ghost)
            # FIX CTO: Lowered from 1.5s to 0.5s to avoid losing real players with short tracks
            speeds = stats["speeds"]
            total_distance = max(stats["distances"]) if stats["distances"] else 0

            # eliminar jugadores que aparecen poco tiempo,
            if len(speeds) < (self.fps * 0.3): 
                continue # Mínimo 0.3 segundos (7 frames a 24fps)
            if total_distance < 0.5:  # Mínimo 0.5 metros recorridos
                continue
            
            # FILTER B: Impossible speed? (Teleportation)
            # Filter absurd spikes greater than 45 km/h for calculations
            # 1. KEEP THE FILTERS (This doesn't change)
            valid_speeds = [s for s in speeds if s < 45]
            max_speed = max(valid_speeds) if valid_speeds else 0

            
            # Fix: Allow goalkeepers that move little (lower from 5m to 1m)
            #comentamos para no eliminar
            #if total_distance < 1: continue

            # 2. NEW: ACCELERATION CALCULATION (Physics)
            max_accel = 0
            # We need a minimum amount of data to compute changes
            if len(speeds) > 10: 
                # Convert from km/h to m/s (dividing by 3.6)
                speeds_ms = [s / 3.6 for s in speeds]
                
                # Use a half-second window to smooth the calculation
                window = int(self.fps / 2)
                if window < 1: window = 1
                
                accels = []
                # Iterate the list comparing current speed vs speed 0.5s ago
                for i in range(window, len(speeds_ms)):
                    v_final = speeds_ms[i]
                    v_inicial = speeds_ms[i - window]
                    time_delta = window / self.fps
                    
                    # Physical formula: a = (vf - vi) / t
                    a = (v_final - v_inicial) / time_delta
                    accels.append(a)
                
                if accels:
                    max_accel = max(accels)

            # 3. Build full-duration speed timeline (one entry per second of video).
            # Missing seconds → None so Chart.js renders visible gaps between segments.
            total_video_seconds = int(total_frames / self.fps) + 1
            speed_by_second = stats.get("speed_by_second", {})
            speed_over_time = [speed_by_second.get(s, None) for s in range(total_video_seconds)]

            export_data["players"][player_id] = {
                "team": int(stats["team"]),
                "max_speed_kmh": round(float(max_speed), 2),
                "total_distance_m": round(float(total_distance), 2),
                "max_acceleration_ms2": round(float(max_accel), 2),
                "speed_over_time": speed_over_time,
                "position_history": stats["positions"]
            }

        # ── MatchAnalytics v2: derived metrics + team aggregates ─────────────
        duration = export_data["match_meta"]["duration_seconds"]
        for pdata in export_data["players"].values():
            pdata["derived"] = compute_player_derived(pdata, duration)
        export_data["team_stats"] = compute_team_stats(export_data["players"])

        # Save file using the special encoder we defined above
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4, cls=NumpyEncoder)
        
        print(f" ✅  Datos exportados exitosamente a: {output_path}")