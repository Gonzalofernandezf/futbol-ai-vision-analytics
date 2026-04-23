import numpy as np 
import cv2 
from ultralytics import YOLO # AÑADIDO: Para cargar tu best.pt

"""
Perspective Transformation Module

Converts 2D pixel coordinates from video to real-world meter coordinates
on the football field using homography transformation.
Calibration is done via interactive tool (calibrate_pitch.py).
"""

class ViewTransformer():
    """
    Transforms video coordinates to real-world field coordinates.
    
    Uses a 4-point perspective transformation (homography) to map video pixels
    to real meters on the pitch. The transformation is calibrated using the
    penalty box corners which have known FIFA dimensions.
    
    Attributes:
        pixel_vertices (np.array): 4 corner points in video (source)
        target_vertices (np.array): 4 corner points in real meters (target)
        H (np.array): Homography matrix for perspective transformation
    """
    
    def __init__(self, model_path='modelo_cancha.pt'):
        # 1. Cargamos el modelo YOLO que entrenaste con tus 50 fotos
        self.modelo_cancha = YOLO(model_path)
        
        # 2. Diccionario de la Realidad (Mundo Real en Metros)
        # Reemplaza tus 4 puntos fijos por los 13 puntos oficiales de tu esqueleto.
        # Las llaves (0, 1, 2...) deben coincidir con el orden que les diste en Roboflow.
        self.target_vertices_dict = {
            0: [0.0, 11.0],  1: [16.5, 11.0],  2: [0.0, 53.0],   3: [16.5, 53.0],  4: [11.0, 32.0],
            5: [50.0, 22.85], 6: [50.0, 32.0],  7: [50.0, 41.15],
            8: [83.5, 11.0],  9: [100.0, 11.0], 10: [83.5, 53.0], 11: [100.0, 53.0], 12: [89.0, 32.0]
        }
        
        # 3. Almacenamiento Dinámico
        # En lugar de guardar una matriz, guardaremos un diccionario con las matrices de TODO el video
        # Ejemplo: { frame_0: matriz_0, frame_1: matriz_1 ... }
        self.matrices_por_frame = {}
        self.ultima_matriz_valida = None

    def calcular_matrices_para_video(self, video_frames):
        """
        NUEVA FUNCIÓN: Recorre todo el video antes de hacer el tracking, 
        busca los puntos clave con la IA en cada fotograma y calcula su matriz correspondiente.
        """
        print("⚽ Calculando matrices de perspectiva con IA para todo el video...")
        
        # Recorremos cada fotograma del video
        for frame_num, frame in enumerate(video_frames):
            # Le pasamos el fotograma a tu IA entrenada
            resultados_cancha = self.modelo_cancha(frame, verbose=False)[0]
            
            pts_pixeles = [] # Aquí guardaremos los puntos que la IA vea
            pts_metros = []  # Aquí sus equivalentes en la vida real
            
            # Si la IA detecta al menos un esqueleto de cancha...
            if len(resultados_cancha.keypoints.xy) > 0:
                xy_coords = resultados_cancha.keypoints.xy[0].cpu().numpy()
                confianzas = resultados_cancha.keypoints.conf[0].cpu().numpy()
                
                # Revisamos cada uno de los 13 puntos
                for i, (xy, conf) in enumerate(zip(xy_coords, confianzas)):
                    # Si la IA está más de un 50% segura de que lo está viendo...
                    if conf > 0.5: 
                        pts_pixeles.append(xy) # Guardamos la coordenada del pixel
                        pts_metros.append(self.target_vertices_dict[i]) # Guardamos su medida real
            
            # Las matemáticas requieren mínimo 4 puntos para calcular la perspectiva.
            if len(pts_pixeles) >= 4:
                # Calculamos la matriz para ESTE frame específico
                matriz, _ = cv2.findHomography(
                    np.array(pts_pixeles, dtype=np.float32), 
                    np.array(pts_metros, dtype=np.float32), 
                    cv2.RANSAC, 5.0
                )
                
                if matriz is not None:
                    # Si el cálculo fue exitoso, lo guardamos para este frame
                    self.matrices_por_frame[frame_num] = matriz
                    self.ultima_matriz_valida = matriz # Actualizamos nuestro "salvavidas"
                else:
                    # Si falló el cálculo, reciclamos la matriz del frame anterior
                    self.matrices_por_frame[frame_num] = self.ultima_matriz_valida
            else:
                # Si la IA vio menos de 4 puntos (cámara borrosa, paneo brusco), reciclamos la matriz anterior
                self.matrices_por_frame[frame_num] = self.ultima_matriz_valida
    
    # MODIFICACIÓN: Ahora la función necesita saber en qué frame estamos (frame_num)
    def transform_point(self, point, frame_num,
                        _x_min=-5.0, _x_max=105.0, _y_min=-5.0, _y_max=69.0):
        """
        Transform a pixel coordinate to real-world meters using the per-frame homography.

        Returns None (instead of garbage values) when:
          - No homography matrix exists for this frame.
          - The homography is degraded (recycled from a camera-pan frame) and produces
            coordinates far outside the pitch.  Values like [975, 568] (pixel-space
            leakage) or [-628, -345] (inverted/degenerate matrix) are silently rejected
            here so they never reach downstream consumers (exporter, heatmap, ball filter).
        """
        matriz_actual = self.matrices_por_frame.get(frame_num)
        if matriz_actual is None:
            return None

        reshaped_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)
        result = cv2.perspectiveTransform(reshaped_point, matriz_actual)
        result = result.reshape(-1, 2)

        # Bounds guard: reject degenerate homography outputs.
        # A valid pitch position is within [−5, 105] m × [−5, 69] m (100×64 + 5m margin).
        x, y = float(result[0, 0]), float(result[0, 1])
        if not (_x_min <= x <= _x_max and _y_min <= y <= _y_max):
            return None

        return result

    def add_transformed_position_to_tracks(self, tracks):
        # Iterate over all objects (ball, players) and add their position in meters
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted'] # Usamos la posición de los pies (ajustada)
                    
                    # Convert to numpy for the function
                    position = np.array(position)
                    
                    # Calculate position in meters
                    position_transformed = self.transform_point(position, frame_num)

                    if position_transformed is not None:
                        # squeeze() removes extra useless dimensions
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed