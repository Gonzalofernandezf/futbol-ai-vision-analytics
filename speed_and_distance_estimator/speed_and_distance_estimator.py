import cv2
import sys 
sys.path.append("../")

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5 # Calculamos velocidad cada 5 frames para suavizar ruido
        self.frame_rate = 24  # Asumimos 24fps por defecto, lo sobreescribiremos si hace falta
    
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees": continue # No nos importa la velocidad del árbitro o balón por ahora
            
            number_of_frames = len(object_tracks)
            
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames-1)

                for track_id, _ in object_tracks[frame_num].items():
                    # Solo calculamos si el objeto existe en ambos frames (inicio y fin de la ventana)
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Obtenemos las posiciones transformadas (METROS)
                    start_position = object_tracks[frame_num][track_id].get('position_transformed')
                    end_position = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue
                    
                    # 1. Calcular Distancia Euclidiana (Fórmula matemática básica)
                    distance = self.measure_distance(start_position, end_position)
                    
                    # 2. Calcular Velocidad
                    # Tiempo = Cantidad de frames / FPS
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    
                    if time_elapsed == 0: continue
                        
                    speed_meters_per_second = distance / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # 3. Guardar datos en el Tracker
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

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # FORWARD FILL INTELIGENTE (Con límite de tolerancia)
        # ------------------------------------------------------------------------------
        MAX_FRAME_GAP = 12  # Límite: 0.5 segundos (a 24fps). Más de esto es mentir.

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees": continue
            
            # Variables temporales para controlar la "memoria"
            last_valid_speed = None
            last_valid_distance = None
            gap_counter = 0 

            for frame_num in range(len(object_tracks)):
                # Iteramos sobre los tracks de este frame
                # Nota: Puede haber varios jugadores, así que buscamos al que estamos rastreando
                # Pero como tracks está estructurado por frame, necesitamos mantener memoria POR ID DE JUGADOR.
                pass 
            
            # --- CORRECCIÓN: La estructura de datos requiere iterar por ID de jugador primero ---
            # Para hacer esto bien, primero recolectamos todos los IDs únicos que aparecen en el video
            unique_track_ids = set()
            for frame_data in object_tracks:
                unique_track_ids.update(frame_data.keys())
            
            # Ahora procesamos la línea de tiempo de CADA jugador individualmente
            for track_id in unique_track_ids:
                last_speed = None
                last_dist = None
                frames_since_last_valid = 0
                
                for frame_num in range(len(object_tracks)):
                    if track_id in object_tracks[frame_num]:
                        track_info = object_tracks[frame_num][track_id]
                        
                        # CASO 1: Tenemos dato real. Reseteamos contadores.
                        if "speed" in track_info:
                            last_speed = track_info['speed']
                            last_dist = track_info['distance']
                            frames_since_last_valid = 0
                        
                        # CASO 2: No hay dato, pero tenemos memoria reciente (< MAX_GAP)
                        elif last_speed is not None and frames_since_last_valid < MAX_FRAME_GAP:
                            track_info['speed'] = last_speed
                            track_info['distance'] = last_dist
                            frames_since_last_valid += 1
                            
                        # CASO 3: Pasó mucho tiempo sin datos. No rellenamos (dejamos que muera).
                        else:
                            frames_since_last_valid += 1

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy() # No sobreescribir original

            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees": continue

                for track_id, track_info in object_tracks[frame_num].items():
                    # Intentamos obtener los datos. Si no existen, devuelve None.
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        
                        # Si no hay datos (ni calculados ni heredados), saltamos al siguiente.
                        if speed is None or distance is None: 
                            continue
                        
                        bbox = track_info['bbox']
                        position = (int(bbox[2]), int(bbox[3])) # Esquina inferior derecha
                        position = (position[0] + 10, position[1]) # Un poco a la derecha

                        # Dibujar textos
                        cv2.putText(frame, f"{speed:.1f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{speed:.1f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.putText(frame, f"{distance:.1f} m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.1f} m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            output_frames.append(frame)
        
        return output_frames

    def measure_distance(self, p1, p2):
        # Pitágoras: a^2 + b^2 = c^2
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5