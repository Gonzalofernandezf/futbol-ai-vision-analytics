"""
T6 — Suavizado cinemático de la trayectoria del balón (Savitzky-Golay).

Funciones puras sobre tracks["ball"] ya filtrado (post speed-filter y
static-cluster-filter). Elimina el ruido de localización de alta frecuencia
que la derivada frame-a-frame convierte en picos de velocidad irreales —
lo que a su vez rompe ventanas de "balón inmóvil" en el detector de balón
parado y ensucia la curva de velocidad del dashboard.

Decisiones de diseño (ver docs/PLAN_player_tracking.md, Tier T6):
- Solo se suaviza position_transformed (metros). bbox y position (píxeles)
  quedan crudos: el vídeo anotado y la asignación de posesión usan píxeles
  y no se benefician del suavizado.
- NO rellena huecos ni los crea: detect_boundary_events depende de la
  estructura exacta de huecos (frames sin balón) para detectar salidas de
  cancha, así que cada tramo continuo de detecciones se suaviza por separado
  y los huecos quedan intactos. Rellenar huecos es trabajo de
  interpolate_ball_positions (BALL_INTERP_*), que corre antes.
- Tramos más cortos que la ventana se suavizan con la mayor ventana impar
  que quepa; si ni siquiera cabe una ventana válida (> polyorder), el tramo
  queda crudo — mejor cruda que una extrapolación forzada.
"""
import numpy as np
from scipy.signal import savgol_filter

import config


def _smooth_segment(positions, window, polyorder):
    """Suaviza un tramo continuo de posiciones [x, y] (lista de listas/tuplas).

    Devuelve una lista de [x, y] suavizados con la misma longitud. Si el tramo
    es demasiado corto para una ventana válida, lo devuelve tal cual.
    """
    n = len(positions)
    # Mayor ventana impar que quepa en el tramo, sin superar la configurada.
    eff_window = min(window, n)
    if eff_window % 2 == 0:
        eff_window -= 1
    if eff_window <= polyorder:
        return positions  # tramo demasiado corto — se deja crudo

    arr = np.asarray(positions, dtype=float)
    xs = savgol_filter(arr[:, 0], eff_window, polyorder)
    ys = savgol_filter(arr[:, 1], eff_window, polyorder)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def smooth_ball_positions(ball_tracks, window=None, polyorder=None):
    """Suaviza in-place position_transformed del balón por tramos continuos.

    Args:
        ball_tracks: tracks["ball"] (list[dict] por frame; balón bajo la clave 1
            con 'position_transformed' cuando hay detección válida, {} en huecos).
        window: ventana Savitzky-Golay en frames (impar). Default: config.
        polyorder: orden polinomial del filtro. Default: config.

    Returns:
        La misma lista ball_tracks, con position_transformed suavizado. La
        estructura de huecos no cambia: ningún frame gana ni pierde detección.
    """
    if window is None:
        window = config.BALL_SAVGOL_WINDOW
    if polyorder is None:
        polyorder = config.BALL_SAVGOL_POLYORDER

    n = len(ball_tracks)
    smoothed_segments = 0
    i = 0
    while i < n:
        frame_ball = ball_tracks[i]
        has_pos = bool(frame_ball) and 1 in frame_ball \
            and frame_ball[1].get('position_transformed') is not None
        if not has_pos:
            i += 1
            continue

        seg_start = i
        while i < n and ball_tracks[i] and 1 in ball_tracks[i] \
                and ball_tracks[i][1].get('position_transformed') is not None:
            i += 1
        seg_end = i  # exclusivo

        segment = [
            list(np.asarray(ball_tracks[f][1]['position_transformed'], dtype=float).flatten())
            for f in range(seg_start, seg_end)
        ]
        smoothed = _smooth_segment(segment, window, polyorder)
        if smoothed is not segment:
            smoothed_segments += 1
        for f, pos in zip(range(seg_start, seg_end), smoothed):
            ball_tracks[f][1]['position_transformed'] = pos

    if smoothed_segments:
        print(f"⚽ Ball smoothing (Savitzky-Golay w={window}, p={polyorder}): "
              f"{smoothed_segments} tramo(s) suavizado(s).")
    return ball_tracks
