from utils.video_utils import read_video, save_video
from Trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from datetime import datetime
import cv2
import numpy as np
import os
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
from data_exporter.data_exporter import GameStatsExporter
import shutil 
from utils.video_utils import read_video, save_video

def get_dynamic_output_path(output_dir, base_name="output_elipses"):
    """
    Genera un nombre de archivo único con fecha y versión incremental.
    Formato: YYYY-MM-DD_output_elipses_vX.mp4
    """
    # 1. Obtener fecha de hoy (Ej: 2024-02-06)
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 2. Asegurar que existe el directorio
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    version = 1
    while True:
        # Construimos el nombre candidato
        file_name = f"{today}_{base_name}_v{version}.mp4"
        full_path = os.path.join(output_dir, file_name)
        
        # 3. Si no existe, ¡ese es el nuestro!
        if not os.path.exists(full_path):
            return full_path
        
        # Si existe, probamos con la siguiente versión
        version += 1

def main():
    # 1. Configuración
    video_path = 'football video analysis_1.mp4'
    model_path = 'best_100e.pt'
    stub_path = 'stubs/track_stubs.pkl' 
    # Definimos la carpeta donde queremos guardar los videos
    output_dir = 'output_videos' 
    
    # La función calcula automáticamente si toca v1, v2, v3...
    output_path = get_dynamic_output_path(output_dir)
    
    print(f"📁 El video se guardará como: {output_path}")

    # 2. Leer Video
    video_frames, fps = read_video(video_path)

    # 3. Inicializar Tracker
    tracker = Tracker(model_path)

    # 4. Obtener Tracks (¡Esto será instantáneo ahora!)
    tracks = tracker.get_object_tracks(
        video_frames, 
        read_from_stub=True, 
        stub_path=stub_path
    )

    # --- NUEVO: INTERPOLACIÓN DE PELOTA ---
    print("⚽ Interpolando posiciones de la pelota...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    # --------------------------------------

    # 5. BLOQUE DE RECORTE DE JUGADOR (DEBUG) 
    # 1. Definir carpeta y crearla si no existe (AUTOMATIZACIÓN)
    output_dir = 'output_videos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Carpeta '{output_dir}' creada automáticamente.")

    # 2. Verificar que existan jugadores detectados en el primer frame
    if tracks['players'] and len(tracks['players']) > 0 and tracks['players'][0]:
        # Tomamos el primer frame y el primer jugador detectado
        frame = video_frames[0]
        # Obtenemos el ID y datos del primer jugador que encontremos
        player_id = list(tracks['players'][0].keys())[0]
        player_data = tracks['players'][0][player_id]
        
        bbox = player_data['bbox']

        # 3. Coordenadas seguras (Evita crash si el jugador está medio fuera de pantalla)
        h, w, _ = frame.shape
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        # Asegurar que no sean negativos ni mayores al tamaño del video (Clamping)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 4. Recortar y Guardar
        # Verificamos que el recorte tenga tamaño válido (width > 0 y height > 0)
        if x2 > x1 and y2 > y1:
            cropped_image = frame[y1:y2, x1:x2]
            save_path = os.path.join(output_dir, 'player_crop_debug.jpg')
            
            cv2.imwrite(save_path, cropped_image)
            print(f"✅ Imagen de prueba guardada en: {save_path}")
        else:
            print("⚠️ El bounding box del jugador es inválido (tamaño 0).")
    else:
        print("⚠️ No se detectaron jugadores en el frame 0 para recortar.")

    # 6. NUEVO: Estimación de Movimiento de Cámara 
    
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    
    # 1. Calcular cuánto se mueve la cámara en X e Y
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )

    # 2. Ajustar la posición de los jugadores restando el movimiento
    # (Esto crea el campo 'position_adjusted' que usaremos luego)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # 6.5. TRANSFORMACIÓN DE PERSPECTIVA (PIXELES -> METROS)
    # Inicializamos el transformador
    view_transformer = ViewTransformer()
    
    # Añadimos la posición transformada (en metros) a cada objeto rastreado
    # NOTA: Esto usa 'position_adjusted' que acabamos de calcular en el paso 6
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    print(" 📐  Perspectiva transformada: Coordenadas en metros calculadas.")

    # 6.8. ESTIMACIÓN DE VELOCIDAD Y DISTANCIA
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    # Importante: Le pasamos los fps reales del video para que el cálculo sea exacto
    speed_and_distance_estimator.frame_rate = fps 
    
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    print(" 🚀  Velocidad y Distancia calculadas.")
    
    # 7. LÓGICA DE EQUIPOS 
    team_assigner = TeamAssigner()
    
    # Le mandamos el primer frame donde haya jugadores para que "aprenda" los colores de las camisetas
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    print("🧠 Asignando equipos a cada jugador...")
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Identificamos el equipo (1 o 2)
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            
            # Guardamos el dato en el diccionario tracks
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    print(f"🎨 Colores detectados - Equipo 1: {team_assigner.team_colors[1]}, Equipo 2: {team_assigner.team_colors[2]}")

    # --- 7.5 CORRECCIÓN DE EQUIPOS (VOTACIÓN) ---
    # Esto evita que el color parpadee si la IA se confunde en un frame suelto.
    print("⚖️ Ajustando equipos por votación mayoritaria...")
    
    player_team_votes = {} # Diccionario para guardar los votos: {ID_Jugador: [1, 1, 2, 1...]}

    # 1. Recolectamos todos los "votos" de cada frame
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = track['team']
            
            if player_id not in player_team_votes:
                player_team_votes[player_id] = []
            
            player_team_votes[player_id].append(team)

    # 2. Contamos votos y corregimos el pasado
    for player_id, votes in player_team_votes.items():
        # Calculamos la MODA (el valor que más se repite)
        # Ejemplo: Si votes es [1, 1, 1, 2, 1], el ganador es 1.
        team_winner = max(set(votes), key=votes.count)
        
        # 3. Sobreescribimos el equipo definitivo en TODOS los frames
        for frame_num in range(len(tracks['players'])):
            if player_id in tracks['players'][frame_num]:
                tracks['players'][frame_num][player_id]['team'] = team_winner
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_winner]
    
    # 8. ASIGNAR POSESIÓN DE BALÓN
    player_assigner = PlayerBallAssigner()
    team_ball_control = [] # Para guardar qué equipo tiene el balón en cada frame

    print("⚽ Calculando posesión del balón...")
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

            # Guardamos qué equipo tiene el balón para estadísticas
            team_id = tracks['players'][frame_num][assigned_player]['team']
            team_ball_control.append(team_id)
        else:
            # Si nadie lo tiene, añadimos el último equipo que lo tuvo (para evitar parpadeos en estadísticas)
            # O simplemente None si prefieres exactitud pura
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(None) # Nadie lo ha tenido aún

    # Imprimimos un resumen rápido en consola
    team1_num_frames = team_ball_control.count(1)
    team2_num_frames = team_ball_control.count(2)
    # Evitar división por cero
    total = team1_num_frames + team2_num_frames
    if total == 0: total = 1 

    print(f"📊 Posesión estimada: Equipo 1: {team1_num_frames/total*100:.1f}% - Equipo 2: {team2_num_frames/total*100:.1f}%")
    
    
    # 👇 BLOQUE DE HARDCODE INTELIGENTE (Auto-Detección de Color) 👇
    # Objetivo: Saber cuál ID (1 o 2) corresponde al equipo blanco/claro basándonos en la suma de sus colores RGB.
    
    # 1. Recuperamos los colores que aprendió el TeamAssigner
    color_team_1 = team_assigner.team_colors[1]
    color_team_2 = team_assigner.team_colors[2]
    
    # 2. Comparamos "Luminosidad" (Suma R+G+B)
    # El blanco (255,255,255) suma ~765. El verde oscuro o negro suma mucho menos.
    if sum(color_team_1) > sum(color_team_2):
        id_equipo_blanco = 1
        id_equipo_color = 2
        print(f"⚪ Auto-Detect: El equipo BLANCO es el ID {id_equipo_blanco} (Más brillante)")
    else:
        id_equipo_blanco = 2
        id_equipo_color = 1
        print(f"⚪ Auto-Detect: El equipo BLANCO es el ID {id_equipo_blanco} (Más brillante)")

    # 3. Asignación Forzada a tus Jugadores Clave
    # Lista de jugadores que TÚ sabes que juegan de blanco (ej: 135, 17)
    jugadores_blancos_ids = [135, 17]

    print(f"🔧 Forzando jugadores {jugadores_blancos_ids} al equipo BLANCO (ID {id_equipo_blanco})...")

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            
            # Si el jugador está en tu lista de "Blancos"
            if player_id in jugadores_blancos_ids:
                # Le asignamos el ID que detectamos como blanco
                tracks['players'][frame_num][player_id]['team'] = id_equipo_blanco
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[id_equipo_blanco]
    
    # 9. DIBUJAR (NUEVO)
    print("🎨 Dibujando elipses y triángulos...")
    # Primero dibujamos las anotaciones del tracker (elipses, etc.)
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # DIBUJAR MOVIMIENTO DE CÁMARA
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    team_ball_control = np.array(team_ball_control)
    
    # CAMBIO AQUÍ: Dibujar Posesión Frame a Fram
    print("📊 Estampando estadísticas de posesión...")
    for frame_num, frame in enumerate(output_video_frames):
        # Llamamos a la función nueva pasando el frame individual y el número de frame
        output_video_frames[frame_num] = tracker.draw_team_ball_control(frame, frame_num, team_ball_control)
    
    # DIBUJAR VELOCIDAD Y DISTANCIA 
    # Hacemos esto ANTES de dibujar la posesión para que el texto quede "debajo" de los overlays gráficos si hay
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    # 10. Guardar Video
    print(f"💾 Guardando video final en {output_path}...")
    save_video(output_video_frames, output_path, fps)
    print("✅ ¡Terminado! Revisa el video nuevo.")

    # ==============================================================
    # 11. EXPORTACIÓN DE DATOS (HISTÓRICO)
    # ==============================================================
    # Inicializamos el exportador
    exporter = GameStatsExporter(fps=fps)

    # 1. Definimos el nombre HISTÓRICO (basado en el nombre del video v6, v7...)
    # Esto guardará '..._output_elipses_v6_stats.json' en la carpeta output_videos
    json_output_path = output_path.replace('.mp4', '_stats.json')

    # 1. Recuperamos la lista completa de quién tuvo el balón
    # team_ball_control es una lista tipo [1, 1, 1, 2, 2, 1...]
    team_ball_control_np = np.array(team_ball_control)
    
    # 2. Contamos cuántas veces aparece el 1 y el 2 en TODO el partido
    team1_frames = np.sum(team_ball_control_np == 1)
    team2_frames = np.sum(team_ball_control_np == 2)
    total_valid_frames = team1_frames + team2_frames

    # 3. Calculamos porcentajes (evitando división por cero)
    if total_valid_frames > 0:
        home_poss = (team1_frames / total_valid_frames) * 100
        away_poss = (team2_frames / total_valid_frames) * 100
    else:
        home_poss, away_poss = 0, 0

    # 4. LLAMADA AL EXPORTADOR (Aquí conectamos con el Paso 1)
    exporter.export_json(tracks, json_output_path, home_possession=home_poss, away_possession=away_poss, view_transformer=view_transformer)
    print(f" 💾  Datos históricos guardados en: {json_output_path}")

    # ==============================================================
    # 12. AUTO-DEPLOY A LA CARPETA DEMO (COPIA GENÉRICA)
    # ==============================================================
    print(" 🚀  Actualizando la Demo Web...")

    # Definimos la carpeta de la demo
    demo_dir = 'demo_dashboard'
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    # Definimos los nombres GENÉRICOS que espera el HTML
    # El HTML siempre buscará 'demo_video.mp4' y 'match_data.json'
    demo_video_dest = os.path.join(demo_dir, 'demo_video.mp4')
    demo_json_dest = os.path.join(demo_dir, 'match_data.json')

    # COPIAR Y RENOMBRAR AUTOMÁTICAMENTE
    # Copiamos el video v6 -> demo_video.mp4
    shutil.copy(output_path, demo_video_dest)
    
    # Copiamos el json v6 -> match_data.json
    shutil.copy(json_output_path, demo_json_dest)

    print(f" ✅  Demo actualizada en la carpeta '{demo_dir}'")
    print(f"     Video: {demo_video_dest}")
    print(f"     JSON:  {demo_json_dest}")

if __name__ == '__main__':
    main()