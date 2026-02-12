import numpy as np 
import cv2 

class ViewTransformer():
    def __init__(self):
        # ANCHO DEL CAMPO REAL (en metros)
        # Un campo estándar tiene 68 metros de ancho.
        # La línea de banda (x=0) hasta la otra banda (x=68).
        court_width = 68
        court_length = 23.32 # Largo del segmento que vamos a usar (ej: área grande)

        # 1. PUNTOS EN EL VIDEO (SOURCE)
        # Tienes que buscar 4 puntos fijos en tu video que formen un rectángulo en la vida real (área grande derecha)
        self.pixel_vertices = np.array([[922, 346], [1212, 328], [1754, 510], [1367, 552]]).astype(np.float32)

        # 2. PUNTOS REALES (ÁREA GRANDE DERECHA - FIFA)
        self.target_vertices = np.array([
            [88.5, 13.84],  # Pico Arriba
            [105,  13.84],  # Fondo Arriba
            [105,  54.16],  # Fondo Abajo
            [88.5, 54.16]   # Pico Abajo
        ]).astype(np.float32)

        # 3. Calcular la Matriz de Homografía (La fórmula mágica)
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        # Convertir un punto (x, y) de píxeles a metros
        p = (int(point[0]), int(point[1]))
        
        # --- ELIMINAR RESTRICCIÓN DE POLÍGONO ---
        # Si limitas esto, borras a todo jugador que salga del área calibrada.
        #is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 
        #if not is_inside: 
        #    return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        # Recorrer todos los objetos (balón, jugadores) y añadir su posición en metros
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted'] # Usamos la posición de los pies (ajustada)
                    
                    # Convertir a numpy para la función
                    position = np.array(position)
                    
                    # Calcular posición en metros
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        # squeeze() elimina dimensiones extra inútiles
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed