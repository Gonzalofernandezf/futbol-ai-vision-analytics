import numpy as np
import cv2
from ultralytics import YOLO # AÑADIDO: Para cargar tu best.pt
import config as _cfg

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
        self.modelo_cancha = YOLO(model_path)
        print(f"🏟️  Modelo cancha: task={self.modelo_cancha.task}  clases={self.modelo_cancha.names}")

        # Mapeo class_id → coordenadas reales (metros).
        # Orden según data.yaml exportado de Roboflow (alfabético por nombre de clase).
        # Sistema de coordenadas: x=0 línea de gol izquierda, x=100 línea de gol derecha,
        # y=0 banda superior, y=64 banda inferior.
        self.target_vertices_dict = {
            0:  [16.5,  52.16],  # big_l_far_bot
            1:  [16.5,  11.84],  # big_l_far_top
            2:  [0.0,   52.16],  # big_l_goal_bot
            3:  [0.0,   11.84],  # big_l_goal_top
            4:  [83.5,  52.16],  # big_r_far_bot
            5:  [83.5,  11.84],  # big_r_far_top
            6:  [100.0, 52.16],  # big_r_goal_bot
            7:  [100.0, 11.84],  # big_r_goal_top
            8:  [50.0,  32.0],   # center_spot
            9:  [50.0,  41.15],  # circle_bot
            10: [50.0,  22.85],  # circle_top
            11: [0.0,   64.0],   # corner_bl
            12: [100.0, 64.0],   # corner_br
            13: [0.0,   0.0],    # corner_tl
            14: [100.0, 0.0],    # corner_tr
            15: [50.0,  0.0],    # halfline_top
            16: [11.0,  32.0],   # penalty_l
            17: [89.0,  32.0],   # penalty_r
            18: [5.5,   41.16],  # small_l_far_bot
            19: [5.5,   22.84],  # small_l_far_top
            20: [94.5,  41.16],  # small_r_far_bot
            21: [94.5,  22.84],  # small_r_far_top
        }
        
        # 3. Almacenamiento Dinámico
        # En lugar de guardar una matriz, guardaremos un diccionario con las matrices de TODO el video
        # Ejemplo: { frame_0: matriz_0, frame_1: matriz_1 ... }
        self.matrices_por_frame = {}
        self.ultima_matriz_valida = None

    def calcular_matrices_para_video(self, video_frames):
        """
        Recorre todo el video, detecta keypoints de cancha con el modelo de detección
        (una clase por keypoint, class_id → target_vertices_dict) y calcula la homografía
        por frame.
        """
        print("⚽ Calculando matrices de perspectiva con IA para todo el video...")
        frames_con_matriz = 0
        class_hit_count = {cls_id: 0 for cls_id in self.target_vertices_dict}

        # Batched inference: launching the field model frame-by-frame leaves the GPU
        # idle between launches. Process in chunks so CUDA kernels stay saturated.
        batch_size = 20
        total_frames = len(video_frames)

        for batch_start in range(0, total_frames, batch_size):
            batch = video_frames[batch_start:batch_start + batch_size]
            batch_results = self.modelo_cancha(
                batch,
                verbose=False,
                device=_cfg.YOLO_DEVICE,
                half=_cfg.YOLO_HALF,
            )

            for offset, resultados_cancha in enumerate(batch_results):
                frame_num = batch_start + offset

                pts_pixeles = []
                pts_metros = []

                # Modelo de detección: cada bbox es un keypoint identificado por su class_id.
                # Si el mismo class_id aparece varias veces, nos quedamos con el de mayor confianza.
                if len(resultados_cancha.boxes) > 0:
                    boxes  = resultados_cancha.boxes
                    xyxy   = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    confs  = boxes.conf.cpu().numpy()

                    best_per_class = {}  # {class_id: (conf, center_x, center_y)}
                    for box, cls_id, conf in zip(xyxy, cls_ids, confs):
                        if conf < 0.25 or cls_id not in self.target_vertices_dict:
                            continue
                        if cls_id not in best_per_class or conf > best_per_class[cls_id][0]:
                            cx = (box[0] + box[2]) / 2.0
                            cy = (box[1] + box[3]) / 2.0
                            best_per_class[cls_id] = (conf, cx, cy)

                    for cls_id, (_, cx, cy) in best_per_class.items():
                        pts_pixeles.append([cx, cy])
                        pts_metros.append(self.target_vertices_dict[cls_id])
                        class_hit_count[cls_id] += 1

                if len(pts_pixeles) >= 4:
                    matriz, _ = cv2.findHomography(
                        np.array(pts_pixeles, dtype=np.float32),
                        np.array(pts_metros,  dtype=np.float32),
                        cv2.RANSAC, 5.0
                    )
                    if matriz is not None:
                        self.matrices_por_frame[frame_num] = matriz
                        self.ultima_matriz_valida = matriz
                        frames_con_matriz += 1
                    else:
                        self.matrices_por_frame[frame_num] = self.ultima_matriz_valida
                else:
                    self.matrices_por_frame[frame_num] = self.ultima_matriz_valida

                # Diagnóstico solo en el primer frame para no saturar el log
                if frame_num == 0:
                    print(f"   Frame 0: {len(pts_pixeles)} puntos de cancha detectados con conf>0.25")

        total = total_frames
        print(f"   Matrices calculadas: {frames_con_matriz}/{total} frames ({frames_con_matriz/total*100:.0f}%)")
        print("   Detecciones por clase (cuántos frames detectó cada punto):")
        class_names = self.modelo_cancha.names
        for cls_id, count in sorted(class_hit_count.items(), key=lambda x: x[1]):
            pct = count / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"   {cls_id:2d} {class_names[cls_id]:20s} {bar} {pct:5.1f}%  ({count} frames)")
    
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