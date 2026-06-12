from concurrent.futures import ThreadPoolExecutor
from utils.video_utils import read_video, save_video
from utils.perf_monitor import PhaseTimer, run_sanity_checks, tracking_metrics
from Trackers.tracker import Tracker
from ball_detection.ball_detector import BallDetector
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
import config

"""
Football AI Vision Analytics - Main Processing Pipeline

This module orchestrates the complete video analysis workflow:
1. Player detection and tracking (YOLOv8 + ByteTrack)
2. Team assignment based on jersey colors
3. Ball possession tracking
4. Perspective transformation (pixels -> real meters)
5. Speed and acceleration calculation
6. Output generation (video + JSON statistics)

Usage:
    python Main.py
    
Output:
    - Annotated video: output_videos/YYYY-MM-DD_output_elipses_vX.mp4
    - Statistics JSON: output_videos/YYYY-MM-DD_output_elipses_vX_stats.json
"""

def get_dynamic_output_path(output_dir, base_name="output_elipses"):
    """
    Generates a unique file name with date and incremental version to avoid overwriting.
    
    Args:
        output_dir (str): Directory to save output videos
        base_name (str): Base name for output file (default: "output_elipses")
    
    Returns:
        str: Full path with format YYYY-MM-DD_output_elipses_vX.mp4
    
    Example:
        path = get_dynamic_output_path('output_videos')
        # Returns: 'output_videos/2024-02-06_output_elipses_v1.mp4'
    """
    # 1. Obtain today's date (e.g., 2024-02-06)
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 2. Ensure that the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    version = 1
    while True:
        # Build candidates name
        file_name = f"{today}_{base_name}_v{version}.mp4"
        full_path = os.path.join(output_dir, file_name)
        
        # 3. checks if it doesn't exists
        if not os.path.exists(full_path):
            return full_path
        
        # if it exists, we try the next version
        version += 1

def main():
    timer = PhaseTimer()

    # 1. Configuration - Import from config module
    video_path = config.VIDEO_PATH
    model_path = config.MODEL_PATH
    stub_path = config.STUB_PATH
    output_dir = config.OUTPUT_DIR

    # The function automatically calculates whether it touches v1, v2, v3...
    output_path = get_dynamic_output_path(output_dir)

    print(f"📁 Video will be saved as: {output_path}")

    # 2. Leer Video — rango controlado por VIDEO_START_SEC / VIDEO_END_SEC (config / .env)
    if config.VIDEO_END_SEC is not None:
        segmentos_partido = [(config.VIDEO_START_SEC, config.VIDEO_END_SEC)]
    else:
        segmentos_partido = None  # lee el vídeo completo

    with timer.phase("Read video"):
        video_frames, fps = read_video(video_path, segments=segmentos_partido)

    # 3. Initialize all GPU/CPU-heavy components up front so we can dispatch them
    #    in parallel below. Tracker (GPU), ViewTransformer (GPU on possibly a
    #    different device), and CameraMovementEstimator (CPU optical flow) all
    #    read video_frames read-only and write to disjoint state — safe to run
    #    in threads. YOLO + OpenCV release the GIL during heavy work.
    tracker = Tracker(model_path)
    ball_detector = BallDetector(model_path=config.BALL_MODEL_PATH)
    view_transformer = ViewTransformer(model_path=config.MODELO_CANCHA_PATH)
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    if config.PARALLEL_INFERENCE:
        with timer.phase("Parallel inference (tracker+ball+field+camera)"):
            with ThreadPoolExecutor(max_workers=4) as ex:
                f_tracks = ex.submit(
                    tracker.get_object_tracks,
                    video_frames, False, stub_path,
                )
                f_ball = ex.submit(
                    ball_detector.get_ball_tracks,
                    video_frames, False, config.BALL_STUB_PATH,
                )
                f_field = ex.submit(
                    view_transformer.calcular_matrices_para_video,
                    video_frames,
                )
                f_camera = ex.submit(
                    camera_movement_estimator.get_camera_movement,
                    video_frames, False, 'stubs/camera_movement_stub.pkl',
                )
                tracks                    = f_tracks.result()
                ball_tracks               = f_ball.result()
                f_field.result()
                camera_movement_per_frame = f_camera.result()
    else:
        with timer.phase("Tracker detect"):
            tracks = tracker.get_object_tracks(
                video_frames, read_from_stub=False, stub_path=stub_path,
            )
        with timer.phase("Ball detect"):
            ball_tracks = ball_detector.get_ball_tracks(
                video_frames, read_from_stub=False, stub_path=config.BALL_STUB_PATH,
            )
        with timer.phase("Field keypoints"):
            view_transformer.calcular_matrices_para_video(video_frames)
        with timer.phase("Camera movement (flow)"):
            camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
                video_frames, read_from_stub=False,
                stub_path='stubs/camera_movement_stub.pkl',
            )

    # Merge ball tracks (from the dedicated detector) into the main tracks dict.
    # tracker.get_object_tracks left tracks["ball"] as a list of empty {} so
    # downstream code that indexes by frame_num still works.
    tracks["ball"] = ball_tracks

    # Ball interpolation
    print("⚽ Interpolating ball positions...")
    with timer.phase("Ball interpolate"):
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 5. Player crop block 
    # 1. Create output directory if it doesn't exist (AUTOMATION)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Directory '{output_dir}' created automatically.")

    # 2. Verify that players are detected in the first frame
    if tracks['players'] and len(tracks['players']) > 0 and tracks['players'][0]:
        # Take the first frame and the first detected player
        frame = video_frames[0]
        # Get the ID and data of the first player we find
        player_id = list(tracks['players'][0].keys())[0]
        player_data = tracks['players'][0][player_id]
        
        bbox = player_data['bbox']

        # 3. Safe coordinates (prevents crash if the player is partially off-screen)
        h, w, _ = frame.shape
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        # Ensure values are not negative or larger than the video size (clamping)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 4. Crop and Save
        # Verify that the crop has valid size (width > 0 and height > 0)
        if x2 > x1 and y2 > y1:
            cropped_image = frame[y1:y2, x1:x2]
            save_path = os.path.join(output_dir, 'player_crop_debug.jpg')
            
            cv2.imwrite(save_path, cropped_image)
            if config.DEBUG_MODE:
                print(f"✅ Debug image saved to: {save_path}")
        else:
            print("⚠️ Player bounding box is invalid (size 0).")
    else:
        print("⚠️ No players detected in frame 0 for cropping.")

    # 6. Camera movement adjustment + perspective transform → metres
    #    (Both inputs already computed above; this block only writes derived
    #    fields into tracks.)
    with timer.phase("Camera adjust"):
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    with timer.phase("Transform tracks"):
        view_transformer.add_transformed_position_to_tracks(tracks)

    print(" 📐  Perspective transformed: Coordinates in meters calculated.")

    # Ball post-transform pipeline:
    #   detect → interpolate → transform → speed_filter → static_cluster_filter
    # The speed filter removes physically impossible jumps; the static-cluster filter
    # then removes long stretches where the "ball" doesn't move in real-world metres
    # (pitch stains, socks, centre-spot artefacts the speed filter can't catch).
    with timer.phase("Ball filters"):
        tracks["ball"] = tracker.filter_ball_positions_by_speed(
            tracks["ball"],
            fps,
            max_speed_mps  = config.BALL_MAX_SPEED_MPS,
            pitch_length   = config.PITCH_LENGTH_M,
            pitch_width    = config.PITCH_WIDTH_M,
            pitch_margin   = config.PITCH_MARGIN_M,
        )

        tracks["ball"] = tracker.filter_static_ball_clusters(
            tracks["ball"],
            fps           = fps,
            radius_m      = config.BALL_STATIC_RADIUS_M,
            window_frames = config.BALL_STATIC_WINDOW_FRAMES,
        )

    # 6.8. Speed and distance estimation
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    # Important: Pass the video's real FPS so the calculation is accurate
    speed_and_distance_estimator.frame_rate = fps

    with timer.phase("Speed/distance"):
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    print(" 🚀  Speed and distance calculated.")

    # 7. Team logic
    team_assigner = TeamAssigner()

    # Send the first frame containing players so it can 'learn' jersey colors
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    print("🧠 Assigning teams to each player...")

    with timer.phase("Team assignment"):
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                # Identify the team (1 or 2)
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)

                # Save the data in the tracks dictionary
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    print(f"🎨 Colors detected - Team 1: {team_assigner.team_colors[1]}, Team 2: {team_assigner.team_colors[2]}")

    # 7.5 Team correction (VOTING)
    # This prevents the color from flickering if the AI is confused in a single frame.
    print("⚖️ Adjusting teams by majority voting...")

    with timer.phase("Team voting"):
        player_team_votes = {} # Dictionary to save the votes: {player_id: [1, 1, 2, 1...]}

        # 1. Collect all the "votes" from each frame
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = track['team']

                if player_id not in player_team_votes:
                    player_team_votes[player_id] = []

                player_team_votes[player_id].append(team)

        # 2. Count votes and correct the past
        for player_id, votes in player_team_votes.items():
            # Compute the MODE (the most frequent value)
            # Example: If votes is [1, 1, 1, 2, 1], the winner is 1.
            team_winner = max(set(votes), key=votes.count)

            # 3. Overwrite the definitive team in ALL frames
            for frame_num in range(len(tracks['players'])):
                if player_id in tracks['players'][frame_num]:
                    tracks['players'][frame_num][player_id]['team'] = team_winner
                    tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_winner]

    # 8. Assign Ball possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = [] # To save what team has the ball on each frame

    print("⚽ Calculating ball possession...")
    with timer.phase("Ball possession"):
        for frame_num, player_track in enumerate(tracks['players']):
            # --- FIX: ACCESO SEGURO A LA PELOTA ---
            ball_dict = tracks['ball'][frame_num]

            # Verificamos si la IA realmente detectó una pelota con el ID 1 en este frame
            if 1 in ball_dict:
                ball_bbox = ball_dict[1]['bbox']
                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

                if assigned_player != -1:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_id = tracks['players'][frame_num][assigned_player]['team']
                    team_ball_control.append(team_id)
                else:
                    # Nadie la tiene clara, mantenemos posesión anterior
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else None)
            else:
                # La pelota desapareció o no se detectó en este frame.
                # No explotamos, simplemente mantenemos la posesión del último equipo.
                team_ball_control.append(team_ball_control[-1] if team_ball_control else None)
            # --------------------------------------

    # Print a quick summary to the console
    team1_num_frames = team_ball_control.count(1)
    team2_num_frames = team_ball_control.count(2)
    # Avoid division by zero
    total = team1_num_frames + team2_num_frames
    if total == 0: total = 1 

    print(f"📊 Estimated possession: Team 1: {team1_num_frames/total*100:.1f}% - Team 2: {team2_num_frames/total*100:.1f}%")
    
    # -----------COMENTAMOS BLOQUE COMPLETO PORQUE APLICA PARA OTRO VIDEO--------------------------
    # Smart hardcode block(Auto Color Detection)
    # Goal: Determine which ID (1 or 2) corresponds to the white/bright team based on the sum of their RGB colors.
    
    # 1. Retrieve the colors learned by the TeamAssigner
    #color_team_1 = team_assigner.team_colors[1]
    #color_team_2 = team_assigner.team_colors[2]
    
    # 2. Compare "Luminosity" (Sum R+G+B)
    #White (255,255,255) sums to ~765. Dark green or black sums much less.
    #if sum(color_team_1) > sum(color_team_2):
    #    id_equipo_blanco = 1
    #    id_equipo_color = 2
    #    print(f"⚪ Auto-Detect: WHITE team is ID {id_equipo_blanco} (Brighter)")
    #else:
    #    id_equipo_blanco = 2
    #    id_equipo_color = 1
    #    print(f"⚪ Auto-Detect: WHITE team is ID {id_equipo_blanco} (Brighter)")

    # 3. Forced assignment to your key players
    # List of players YOU know play in white (e.g., 135, 17)
    # jugadores_blancos_ids = [135, 17]

    #print(f"🔧 Forcing players {jugadores_blancos_ids} to WHITE team (ID {id_equipo_blanco})...")

    #for frame_num, player_track in enumerate(tracks['players']):
    #    for player_id, track in player_track.items():
            
            # If the player is in your 'White' list
    #        if player_id in jugadores_blancos_ids:
    #            # Assign the ID we detected as white
    #            tracks['players'][frame_num][player_id]['team'] = id_equipo_blanco
    #            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[id_equipo_blanco]
    # -----------FIN BLOQUE COMENTADO COMPLETO PORQUE APLICA PARA OTRO VIDEO--------------------------

    # 9. Data export — siempre, ANTES de dibujar, para no perder stats si hay OOM al dibujar
    exporter = GameStatsExporter(fps=fps)
    json_output_path = output_path.replace('.mp4', '_stats.json')

    team_ball_control_np = np.array(team_ball_control)
    team1_frames = np.sum(team_ball_control_np == 1)
    team2_frames = np.sum(team_ball_control_np == 2)
    total_valid_frames = team1_frames + team2_frames
    if total_valid_frames > 0:
        home_poss = (team1_frames / total_valid_frames) * 100
        away_poss = (team2_frames / total_valid_frames) * 100
    else:
        home_poss, away_poss = 0, 0

    with timer.phase("Export JSON"):
        exporter.export_json(tracks, json_output_path, home_possession=home_poss, away_possession=away_poss, view_transformer=view_transformer)
    print(f" 💾  Historical data saved to: {json_output_path}")

    # Dump the full tracks dict to pickle for post-hoc rendering.
    # The JSON loses per-frame pixel bboxes (only stores metres), so an external
    # renderer can only approximate overlay positions. The pickle keeps the
    # exact bbox/team/team_color/has_ball state per frame — render_annotated_video.py
    # uses it via --tracks-pkl to draw overlays exactly at the players' feet.
    import pickle as _pickle
    tracks_pkl_path = output_path.replace('.mp4', '_tracks.pkl')
    with open(tracks_pkl_path, 'wb') as f:
        _pickle.dump({
            "tracks":             tracks,
            "team_ball_control":  team_ball_control,
            "fps":                fps,
        }, f)
    print(f" 💾  Tracks pickle saved to: {tracks_pkl_path}")

    # 10. Draw + save video — streaming frame-by-frame to avoid doubling RAM
    # (skip entirely with SKIP_VIDEO_OUTPUT=true for JSON-only runs)
    if config.SKIP_VIDEO_OUTPUT:
        print("⏭️  SKIP_VIDEO_OUTPUT=true — skipping annotated video render.")
    else:
        with timer.phase("Render video"):
            print("🎨 Rendering annotated video (streaming)...")
            h, w = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            team_ball_control_arr = np.array(team_ball_control)
            total_frames = len(video_frames)
            for frame_num, orig_frame in enumerate(video_frames):
                frame = orig_frame.copy()
                frame = tracker.draw_annotations_frame(frame, frame_num, tracks)
                frame = camera_movement_estimator.draw_camera_movement_frame(frame, frame_num, camera_movement_per_frame)
                frame = tracker.draw_team_ball_control(frame, frame_num, team_ball_control_arr)
                frame = speed_and_distance_estimator.draw_speed_and_distance_frame(frame, frame_num, tracks)
                writer.write(frame)
                if frame_num % 500 == 0:
                    print(f"  ✍️  {frame_num}/{total_frames} frames rendered...")

            writer.release()
            print(f"✅ Video saved to {output_path}")

    # 11. Auto-deploy to demo folder
    print(" 🚀  Updating web demo...")
    demo_dir = config.DEMO_DIR
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    demo_json_dest = os.path.join(demo_dir, 'match_data.json')
    shutil.copy(json_output_path, demo_json_dest)

    if not config.SKIP_VIDEO_OUTPUT:
        demo_video_dest = os.path.join(demo_dir, 'demo_video.mp4')
        shutil.copy(output_path, demo_video_dest)
        print(f"     Video: {demo_video_dest}")

    # Artefactos de evaluación: copiar solo si existen (puede no haberse corrido
    # eval_keypoints.py todavía) y si origen y destino no son el mismo archivo.
    for eval_fname in ('eval_report.json', 'eval_history.json'):
        eval_src = os.path.join(config.EVAL_ARTIFACTS_DIR, eval_fname)
        eval_dst = os.path.join(demo_dir, eval_fname)
        if os.path.exists(eval_src) and \
           os.path.normcase(os.path.abspath(eval_src)) != os.path.normcase(os.path.abspath(eval_dst)):
            shutil.copy(eval_src, eval_dst)
            print(f"     Eval:  {eval_dst}")

    print(f" ✅  Demo actualizada en la carpeta '{demo_dir}'")
    print(f"     JSON:  {demo_json_dest}")

    # 12. Perf + sanity report — surfaces regressions when stacking optimisations.
    timer.report()
    run_sanity_checks(tracks, json_output_path)

    metrics = tracking_metrics(tracks, fps)
    print("📊 TRACKING METRICS")
    print("─" * 55)
    print(f"  {'unique player IDs':<26}: {metrics['num_unique_player_ids']}")
    print(f"  {'long tracks (≥5s)':<26}: {metrics['num_long_tracks']}")
    print(f"  {'avg track duration':<26}: {metrics['avg_track_duration_s']:.1f}s")
    print(f"  {'id churn ratio':<26}: {metrics['id_churn_ratio']:.2f}  (target ≤3.0)")
    print(f"  {'team flip rate':<26}: {metrics['team_flip_rate']*100:.1f}%  (target <2%)")
    print("═" * 55)

if __name__ == '__main__':
    main()