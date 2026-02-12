from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model_path):
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

        # 1. Si ya tenemos los datos guardados, los leemos y ahorramos tiempo
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            print("¡Datos cargados desde la caché! (stub)")
            return tracks

        # 2. Si no, nos toca trabajar: Detectamos objetos en los frames
        print("Detectando objetos en el video (esto puede tardar)...")
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

            # Esto evita que el tracker se confunda si cambia la etiqueta
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Rastreo (Tracking)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv['player']:
                    # 👇 NUEVO: Calcular la posición de los pies (Centro-Abajo) 👇
                    position = ( (bbox[0] + bbox[2])/2, bbox[3] )
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": position # <--- AÑADIR ESTA LÍNEA
                        }
                if cls_id == cls_names_inv['referee']:
                     # 👇 NUEVO: Calcular la posición de los pies (Centro-Abajo) 👇
                    position = ( (bbox[0] + bbox[2])/2, bbox[3] )
                    tracks["referees"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": position # <--- AÑADIR ESTA LÍNEA
                        }

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    # 👇 NUEVO: Lo mismo para la pelota 👇
                    position = ( (bbox[0] + bbox[2])/2, bbox[3] )
                    tracks["ball"][frame_num][1] = {
                        "bbox": bbox, 
                        "position": position # <--- AÑADIR ESTA LÍNEA
                        }

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    # --- FUNCIONES DE DIBUJO (NUEVO) ---
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # Parte inferior de la caja (pies)
        x_center, _ = self.get_center_of_bbox(bbox)
        width = self.get_bbox_width(bbox)

        # Dibujar la elipse bajo los pies
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

        # Opcional: Dibujar rectángulo con el número de ID
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
        y = int(bbox[1]) # Parte superior de la caja
        x, _ = self.get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        # Dibujar triángulo invertido sobre la pelota
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # Borde negro
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() # Copiamos para no manchar el original
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Dibujar Jugadores (Dinámico)
            for track_id, player in player_dict.items():
                # Recuperamos el color que calculó el Main. Si no lo encuentra, usa rojo por seguridad.
                color = player.get("team_color", (0, 0, 255))
                
                # Ahora pasamos esa variable 'color' en lugar del fijo
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # Si tiene el balón, dibujamos un triángulo rojo extra encima 
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Dibujar Árbitros (Círculo Amarillo)

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255), track_id)

            # Dibujar Pelota (Triángulo Verde)
            for track_id, ball in ball_dict.items():

                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            
            
            output_video_frames.append(frame)



        return output_video_frames

    # --- ÚTILES ---
    def get_center_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1+x2)/2), int((y1+y2)/2)
    def get_bbox_width(self, bbox):
        return bbox[2] - bbox[0]

    #Interpolación de pelota
    def interpolate_ball_positions(self, ball_positions):
        # 1. Extracción de datos
        ball_bboxes = []
        for frame_tracks in ball_positions:
            if frame_tracks:
                bbox = list(frame_tracks.values())[0].get("bbox", [])
                ball_bboxes.append(bbox)
            else:
                ball_bboxes.append([np.nan, np.nan, np.nan, np.nan])

        df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # 2. Interpolación
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit=5, limit_direction='both')
        df_ball_positions = df_ball_positions.bfill()

        # 3. Reconstrucción (AQUÍ ESTABA EL ERROR)
        ball_positions_interpolated = []
        for i, row in df_ball_positions.iterrows():
            ball_positions_interpolated.append({})
            
            # Solo guardamos si tenemos datos válidos
            if not np.isnan(row['x1']):
                bbox = [row['x1'], row['y1'], row['x2'], row['y2']]
                
                # 👇 ¡¡ESTO ES LO QUE FALTABA!! 👇
                # Tenemos que recalcular la posición basada en el nuevo bbox interpolado
                position = ((bbox[0] + bbox[2])/2, bbox[3])

                ball_positions_interpolated[i][1] = {
                    "bbox": bbox,
                    "position": position  # <--- AHORA SÍ LO GUARDAMOS
                }

        return ball_positions_interpolated

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # 1. Overlay Transparente (Igual que antes)
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 2. Lógica NumPy (EL CAMBIO IMPORTANTE)
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
        # Contamos usando NumPy (mucho más rápido)
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)
        
        # 3. Calcular porcentajes (Igual que antes)
        total_frames = team_1_num_frames + team_2_num_frames
        
        if total_frames == 0:
            team_1_perc = 0
            team_2_perc = 0
        else:
            team_1_perc = team_1_num_frames / total_frames
            team_2_perc = team_2_num_frames / total_frames

        # 4. Escribir texto (Igual que antes)
        cv2.putText(frame, f"Team 1 Possession: {team_1_perc*100:.1f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Possession: {team_2_perc*100:.1f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame