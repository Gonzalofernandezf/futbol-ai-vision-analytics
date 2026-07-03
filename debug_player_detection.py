"""
Vista rápida de detección de jugador/portero/árbitro + asignación de equipo,
sin balón/campo/velocidad/posesión — para comparar detectores (p.ej. T3:
best_100e.pt vs best_jugadores_v2.pt) sin correr el pipeline completo de
Main.py ni competir por VRAM con ball_detector/view_transformer.

Uso:
    VIDEO_PATH=... MODEL_PATH=... python debug_player_detection.py

Respeta las mismas env vars que Main.py para esto: VIDEO_PATH, MODEL_PATH,
VIDEO_START_SEC/VIDEO_END_SEC, OUTPUT_DIR, YOLO_DEVICE_TRACKER, FRAME_SCALE,
SKIP_VIDEO_OUTPUT.

Salida: <OUTPUT_DIR>/debug_players_<fecha>_vN.mp4 (sin JSON, sin deploy a
DEMO_DIR — es una herramienta de depuración visual, no parte del pipeline
de producción). Con SKIP_VIDEO_OUTPUT=true no se renderiza nada — solo se
imprimen las métricas de tracking (útil para ablations rápidos de
parámetros, p.ej. comparar BOTSORT_TRACK_BUFFER, donde el vídeo no aporta
nada y cuesta tiempo de Kaggle).
"""
import os
from datetime import datetime

import cv2

import config
from utils.video_utils import read_video
from utils.perf_monitor import tracking_metrics
from Trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner


def get_output_path(output_dir, base_name="debug_players"):
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(output_dir, exist_ok=True)
    version = 1
    while True:
        path = os.path.join(output_dir, f"{today}_{base_name}_v{version}.mp4")
        if not os.path.exists(path):
            return path
        version += 1


def main():
    video_path = config.VIDEO_PATH
    model_path = config.MODEL_PATH

    segments = [(config.VIDEO_START_SEC, config.VIDEO_END_SEC)] if config.VIDEO_END_SEC is not None else None
    video_frames, fps = read_video(video_path, segments=segments)

    tracker = Tracker(model_path)
    print("🔍 Detectando jugador/portero/árbitro (sin balón/campo/velocidad)...")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path=None)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    print("🧠 Asignando equipos...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Votación por mayoría (igual que Main.py) para que el color no parpadee.
    player_team_votes = {}
    for player_track in tracks['players']:
        for player_id, track in player_track.items():
            player_team_votes.setdefault(player_id, []).append(track['team'])

    for player_id, votes in player_team_votes.items():
        team_winner = max(set(votes), key=votes.count)
        for frame_num in range(len(tracks['players'])):
            if player_id in tracks['players'][frame_num]:
                tracks['players'][frame_num][player_id]['team'] = team_winner
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_winner]

    if config.SKIP_VIDEO_OUTPUT:
        print("⏭️  SKIP_VIDEO_OUTPUT=true — saltando el render, solo métricas.")
    else:
        output_path = get_output_path(config.OUTPUT_DIR)
        print(f"📁 Video de salida: {output_path}")
        print("🎨 Renderizando vídeo (streaming)...")
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame_num, orig_frame in enumerate(video_frames):
            frame = tracker.draw_annotations_frame(orig_frame.copy(), frame_num, tracks)
            writer.write(frame)
            if frame_num % 500 == 0:
                print(f"  ✍️  {frame_num}/{len(video_frames)} frames renderizados...")

        writer.release()
        print(f"✅ Vídeo guardado en {output_path}")

    metrics = tracking_metrics(tracks, fps)
    print("📊 TRACKING METRICS")
    print("─" * 55)
    print(f"  {'unique player IDs':<26}: {metrics['num_unique_player_ids']}")
    print(f"  {'long tracks (≥5s)':<26}: {metrics['num_long_tracks']}")
    print(f"  {'avg track duration':<26}: {metrics['avg_track_duration_s']:.1f}s")
    print(f"  {'id churn ratio':<26}: {metrics['id_churn_ratio']:.2f}  (target ≤3.0)")
    print(f"  {'team flip rate':<26}: {metrics['team_flip_rate']*100:.1f}%  (target <2%)")


if __name__ == '__main__':
    main()
