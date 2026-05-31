import numpy as np
import cv2
from ultralytics import YOLO

import config

"""
Perspective Transformation Module

Convierte coordenadas en píxeles a metros reales sobre la cancha vía homografía.
Soporta dos tipos de modelo de cancha auto-detectados al cargar:

  • 'pose'   — modelo legacy de 25 keypoints (cada instancia = la cancha completa).
  • 'detect' — modelo nuevo de bounding boxes por clase (22 clases, cada bbox
               es un único punto de referencia). El centro del bbox se trata
               como la posición del keypoint.

La API pública (calcular_matrices_para_video, transform_point,
add_transformed_position_to_tracks) es idéntica para ambos modos.
"""


# Coordenadas canónicas para el modelo pose (25 keypoints, sub-18/20 100m × 64m)
POSE_TARGET_VERTICES = {
    0:  [0.0,   0.0],   # L_corner_tl
    1:  [0.0,  22.84],  # L_areas_tl
    2:  [5.5,  22.84],  # L_areas_tr
    3:  [0.0,  11.84],  # L_areab_tl
    4:  [16.5, 11.84],  # L_areab_tr
    5:  [50.0,  0.0],   # Center_top
    6:  [83.5, 11.84],  # R_areab_tl
    7:  [100.0,11.84],  # R_areab_tr
    8:  [94.5, 22.84],  # R_areas_tl
    9:  [100.0,22.84],  # R_areas_tr
    10: [100.0, 0.0],   # R_corner_tr
    11: [11.0, 32.0],   # L_penalspot
    12: [50.0, 32.0],   # Center
    13: [89.0, 32.0],   # R_penalspot
    14: [0.0,  64.0],   # L_corner_bl
    15: [0.0,  41.16],  # L_areas_bl
    16: [5.5,  41.16],  # L_areas_br
    17: [0.0,  52.16],  # L_areab_bl
    18: [16.5, 52.16],  # L_areab_br
    19: [50.0, 64.0],   # Center_bottom
    20: [83.5, 52.16],  # R_areab_bl
    21: [100.0,52.16],  # R_areab_br
    22: [94.5, 41.16],  # R_areas_bl
    23: [100.0,41.16],  # R_areas_br
    24: [100.0,64.0],   # R_corner_br
}

# Coordenadas canónicas para el modelo detect (22 clases por nombre)
# Convención: x=0 → portería izquierda, x=100 → portería derecha
#             y=0 → banda superior,      y=64 → banda inferior
DETECT_CLASS_TO_FIELD_COORDS = {
    # Esquinas exteriores de la cancha
    "corner_tl":         [0.0,    0.0],
    "corner_tr":         [100.0,  0.0],
    "corner_bl":         [0.0,   64.0],
    "corner_br":         [100.0, 64.0],

    # Área grande izquierda (16.5m x 40.32m)
    "big_l_goal_top":    [0.0,   11.84],
    "big_l_goal_bot":    [0.0,   52.16],
    "big_l_far_top":     [16.5,  11.84],
    "big_l_far_bot":     [16.5,  52.16],

    # Área grande derecha
    "big_r_goal_top":    [100.0, 11.84],
    "big_r_goal_bot":    [100.0, 52.16],
    "big_r_far_top":     [83.5,  11.84],
    "big_r_far_bot":     [83.5,  52.16],

    # Área chica izquierda (5.5m x 18.32m) — sólo se anotan los vértices "far"
    "small_l_far_top":   [5.5,   22.84],
    "small_l_far_bot":   [5.5,   41.16],

    # Área chica derecha
    "small_r_far_top":   [94.5,  22.84],
    "small_r_far_bot":   [94.5,  41.16],

    # Puntos de penalti (11m desde portería)
    "penalty_l":         [11.0,  32.0],
    "penalty_r":         [89.0,  32.0],

    # Punto central
    "center_spot":       [50.0,  32.0],

    # Intersecciones del círculo central con la línea de medio campo (radio 9.15m)
    "circle_top":        [50.0,  22.85],
    "circle_bot":        [50.0,  41.15],

    # Línea de medio campo cruzando la banda superior
    # (halfline_bot no se anotó nunca en el dataset actual)
    "halfline_top":      [50.0,   0.0],
}


class ViewTransformer():
    """
    Construye una homografía por frame para mapear píxeles → metros reales.

    Auto-detecta el tipo de modelo:
      - YOLO pose:    usa los 25 keypoints en orden fijo.
      - YOLO detect:  usa bbox centers, mapeando cada nombre de clase a su
                      posición canónica vía DETECT_CLASS_TO_FIELD_COORDS.

    En ambos modos el resultado guarda una matriz 3×3 por frame en
    `matrices_por_frame`, con fallback a la última matriz válida cuando
    el frame no produce suficientes puntos.
    """

    def __init__(self, model_path='modelo_cancha.pt'):
        self.modelo_cancha = YOLO(model_path)
        self.task = getattr(self.modelo_cancha, "task", None)

        if self.task == "pose":
            self.target_vertices_dict = POSE_TARGET_VERTICES
        elif self.task == "detect":
            self.target_vertices_dict = DETECT_CLASS_TO_FIELD_COORDS
        else:
            raise ValueError(
                f"Tipo de modelo no soportado: '{self.task}'. "
                "Se espera 'pose' o 'detect'."
            )

        print(f"📐 modelo_cancha cargado en modo '{self.task}' "
              f"({len(self.target_vertices_dict)} referencias)")

        self.matrices_por_frame = {}
        self.ultima_matriz_valida = None

    # ── Extracción de puntos por frame ───────────────────────────────────────

    def _extract_pose_points(self, resultados, conf_threshold):
        """Extrae (pts_pixeles, pts_metros) desde un resultado YOLO pose."""
        pts_pixeles, pts_metros = [], []
        if resultados.keypoints is None or len(resultados.keypoints.xy) == 0:
            return pts_pixeles, pts_metros

        xy_coords = resultados.keypoints.xy[0].cpu().numpy()
        confianzas = resultados.keypoints.conf[0].cpu().numpy()

        for i, (xy, conf) in enumerate(zip(xy_coords, confianzas)):
            if conf < conf_threshold:
                continue
            if i not in self.target_vertices_dict:
                continue
            pts_pixeles.append(xy)
            pts_metros.append(self.target_vertices_dict[i])
        return pts_pixeles, pts_metros

    def _extract_detect_points(self, resultados, conf_threshold):
        """
        Extrae (pts_pixeles, pts_metros) desde un resultado YOLO detect.

        Si el modelo detecta varias instancias de la misma clase en un frame
        (físicamente imposible para keypoints únicos como 'center_spot'),
        se queda con la de mayor confianza.
        """
        pts_pixeles, pts_metros = [], []
        if resultados.boxes is None or len(resultados.boxes) == 0:
            return pts_pixeles, pts_metros

        names_map = resultados.names  # {class_id: class_name}
        boxes = resultados.boxes

        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        # Mejor detección por clase
        best_per_class = {}  # class_name → (conf, cx, cy)
        for box_xyxy, conf, cls_id in zip(xyxy, confs, clses):
            if conf < conf_threshold:
                continue
            cls_name = names_map.get(cls_id) if isinstance(names_map, dict) else names_map[cls_id]
            if cls_name not in self.target_vertices_dict:
                continue
            cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
            cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
            prev = best_per_class.get(cls_name)
            if prev is None or conf > prev[0]:
                best_per_class[cls_name] = (float(conf), float(cx), float(cy))

        for cls_name, (_conf, cx, cy) in best_per_class.items():
            pts_pixeles.append([cx, cy])
            pts_metros.append(self.target_vertices_dict[cls_name])

        return pts_pixeles, pts_metros

    # ── Cálculo principal ────────────────────────────────────────────────────

    def calcular_matrices_para_video(self, video_frames):
        """
        Recorre el video y calcula una homografía por frame, con fallback a la
        última matriz válida cuando el frame no produce ≥4 puntos.
        """
        print("⚽ Calculando matrices de perspectiva con IA para todo el video...")
        conf_threshold = config.FIELD_KP_CONF

        for frame_num, frame in enumerate(video_frames):
            resultados = self.modelo_cancha(frame, verbose=False)[0]

            if self.task == "pose":
                pts_pixeles, pts_metros = self._extract_pose_points(resultados, conf_threshold)
            else:
                pts_pixeles, pts_metros = self._extract_detect_points(resultados, conf_threshold)

            if len(pts_pixeles) >= 4:
                matriz, _ = cv2.findHomography(
                    np.array(pts_pixeles, dtype=np.float32),
                    np.array(pts_metros, dtype=np.float32),
                    cv2.RANSAC, 5.0,
                )
                if matriz is not None:
                    self.matrices_por_frame[frame_num] = matriz
                    self.ultima_matriz_valida = matriz
                else:
                    self.matrices_por_frame[frame_num] = self.ultima_matriz_valida
            else:
                self.matrices_por_frame[frame_num] = self.ultima_matriz_valida

    def transform_point(self, point, frame_num,
                        _x_min=-5.0, _x_max=105.0, _y_min=-5.0, _y_max=69.0):
        """
        Transforma una coordenada en píxeles a metros usando la homografía del frame.

        Devuelve None cuando:
          - No hay matriz para este frame.
          - La salida cae fuera del campo (homografía degradada / reciclada de
            un frame con paneo brusco).
        """
        matriz_actual = self.matrices_por_frame.get(frame_num)
        if matriz_actual is None:
            return None

        reshaped_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)
        result = cv2.perspectiveTransform(reshaped_point, matriz_actual)
        result = result.reshape(-1, 2)

        x, y = float(result[0, 0]), float(result[0, 1])
        if not (_x_min <= x <= _x_max and _y_min <= y <= _y_max):
            return None

        return result

    def add_transformed_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = np.array(track_info['position_adjusted'])
                    position_transformed = self.transform_point(position, frame_num)

                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()

                    tracks[object_name][frame_num][track_id]['position_transformed'] = position_transformed
