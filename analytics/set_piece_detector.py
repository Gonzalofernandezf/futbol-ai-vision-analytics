"""
Set-piece (balón parado) detection — pure functions over already-computed ball
tracks. No modelo nuevo: clasifica córners/saques de banda/saques de meta/faltas
combinando dos señales geométricas/temporales ya disponibles tras el tracking.

Ver docs/PLAN_player_tracking.md, sección "Análisis de balón parado", para el
diseño y las decisiones (por qué se combinan dos señales, por qué se clasifica
por el punto de salida y no por el de reaparición, etc.).
"""
import math
from typing import Optional

import config

CORNER    = "corner"
THROW_IN  = "banda"
GOAL_KICK = "meta"
FOUL      = "falta"


def _classify_boundary_position(pos, pitch_length=None, pitch_width=None, margin=None):
    """Clasifica un punto [x, y] en metros como córner/banda/meta, o None si no
    está cerca de ningún borde de cancha (probable falso positivo del detector,
    no una salida real de balón)."""
    if pos is None:
        return None
    pitch_length = pitch_length if pitch_length is not None else config.PITCH_LENGTH_M
    pitch_width  = pitch_width  if pitch_width  is not None else config.PITCH_WIDTH_M
    margin       = margin       if margin       is not None else config.SET_PIECE_BOUNDARY_MARGIN_M
    x, y = pos[0], pos[1]

    corners = [(0.0, 0.0), (0.0, pitch_width), (pitch_length, 0.0), (pitch_length, pitch_width)]
    if any(math.hypot(x - cx, y - cy) <= margin for cx, cy in corners):
        return CORNER

    near_goal_line = x <= margin or x >= pitch_length - margin
    near_touchline = y <= margin or y >= pitch_width - margin
    if near_goal_line:
        return GOAL_KICK
    if near_touchline:
        return THROW_IN
    return None


def detect_boundary_events(ball_tracks, out_of_bounds_events, fps, min_gap_frames=None):
    """Señal 1: el balón sale de la cancha y reaparece → córner / banda / meta.

    Se clasifica por el punto de SALIDA (la posición del primer frame rechazado
    por el guard de límites dentro del hueco), no por el de reaparición: un
    córner o saque de meta no reaparece donde salió — se saca desde ahí hacia
    dentro del campo — así que el punto de salida es la señal fiable, la de
    reaparición no.

    Args:
        ball_tracks: tracks["ball"] ya filtrado (list[dict]; balón id 1 con
            'position_transformed' cuando hay detección válida, {} en huecos).
        out_of_bounds_events: [(frame_num, [x, y]), ...] — frames rechazados
            específicamente por el guard de límites (Stage A de
            Tracker.filter_ball_positions_by_speed), no oclusiones normales.
        fps: frame rate del vídeo.
    """
    if min_gap_frames is None:
        min_gap_frames = config.SET_PIECE_MIN_GAP_FRAMES

    oob_by_frame = dict(out_of_bounds_events)
    n = len(ball_tracks)
    events = []

    i = 0
    while i < n:
        if ball_tracks[i] and 1 in ball_tracks[i]:
            i += 1
            continue

        gap_start = i
        while i < n and not (ball_tracks[i] and 1 in ball_tracks[i]):
            i += 1
        gap_end = i  # exclusivo: primer frame con detección válida de nuevo, o n

        if gap_end - gap_start < min_gap_frames:
            continue  # hueco demasiado corto — ruido de detección, no una salida real

        oob_frames_in_gap = [f for f in range(gap_start, gap_end) if f in oob_by_frame]
        if not oob_frames_in_gap:
            continue  # hueco normal (oclusión), no una salida de cancha

        exit_frame = oob_frames_in_gap[0]
        exit_pos = oob_by_frame[exit_frame]
        event_type = _classify_boundary_position(exit_pos)
        if event_type is None:
            continue  # salió lejos de cualquier borde — probable falso positivo

        events.append({
            "type": event_type,
            "frame_start": gap_start,
            "frame_end": min(gap_end, n - 1),
            "position_m": [round(exit_pos[0], 2), round(exit_pos[1], 2)],
        })

    return events


def detect_stationary_events(ball_tracks, fps, speed_threshold_kmh=None, min_duration_sec=None):
    """Señal 2: balón casi inmóvil sin haber salido de cancha, durante un tramo
    sostenido → falta / tiro libre."""
    if speed_threshold_kmh is None:
        speed_threshold_kmh = config.SET_PIECE_STATIONARY_KMH
    if min_duration_sec is None:
        min_duration_sec = config.SET_PIECE_STATIONARY_SEC
    min_frames = max(1, round(min_duration_sec * fps))

    events = []
    run_start = None
    prev_pos = None
    prev_frame = None

    def flush(end_frame):
        if run_start is not None and prev_pos is not None and end_frame - run_start >= min_frames:
            events.append({
                "type": FOUL,
                "frame_start": run_start,
                "frame_end": end_frame,
                "position_m": [round(prev_pos[0], 2), round(prev_pos[1], 2)],
            })

    for frame_num, frame_ball in enumerate(ball_tracks):
        if not frame_ball or 1 not in frame_ball:
            flush(frame_num)
            run_start, prev_pos, prev_frame = None, None, None
            continue

        pos = frame_ball[1].get('position_transformed')
        if pos is None:
            continue

        if prev_pos is not None:
            dt = (frame_num - prev_frame) / fps
            if dt > 0:
                dist_m = math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1])
                speed_kmh = (dist_m / dt) * 3.6
                if speed_kmh < speed_threshold_kmh:
                    if run_start is None:
                        run_start = prev_frame
                else:
                    flush(frame_num)
                    run_start = None

        prev_pos, prev_frame = pos, frame_num

    flush(len(ball_tracks) - 1 if ball_tracks else 0)
    return events


def detect_set_pieces(ball_tracks, out_of_bounds_events, fps):
    """Combina las dos señales y devuelve la lista de eventos ordenada por inicio."""
    events = detect_boundary_events(ball_tracks, out_of_bounds_events, fps)
    events += detect_stationary_events(ball_tracks, fps)
    events.sort(key=lambda e: e["frame_start"])
    return events
