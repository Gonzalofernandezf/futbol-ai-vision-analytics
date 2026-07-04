import numpy as np
import cv2
from collections import deque
from sklearn.cluster import KMeans

import config as _cfg

"""
Team Assignment Module (T7 — ver docs/PLAN_player_tracking.md, Tier T7)

Clasifica jugadores en equipos por color de camiseta con K-Means, con:
- Máscara HSV de césped antes de promediar el color del jersey (T7.a).
- Centroides entrenados con un burn-in multi-frame, no solo el frame 0 (T7.b).
- Re-cluster periódico con anclaje de labels para resistir cambios de luz
  sin intercambiar equipo 1<->2 (T7.c).
- Voto deslizante por track ID en vez de cacheo permanente (T7.d) — el voto
  global anterior era un no-op: el equipo se decidía en el primer frame de
  cada ID y todos los "votos" posteriores eran el mismo valor cacheado.
- El portero (flag is_goalkeeper del tracker) se excluye del entrenamiento
  de centroides para no contaminarlos con un kit distinto (T7.e); su equipo
  se asigna por cercanía de color + voto deslizante, como a cualquiera.
"""


class TeamAssigner:
    """
    Attributes:
        team_colors (dict): color BGR dominante por equipo (1/2), enteros para OpenCV.
        kmeans (KMeans): modelo global de 2 clusters sobre colores de jersey.
    """

    def __init__(self):
        self.team_colors = {}
        self.kmeans = None

    # ------------------------------------------------------------------
    # T7.a — extracción de color
    def get_player_color(self, frame, bbox):
        """Color medio del jersey: mitad superior del bbox, 60% central, con
        máscara HSV de césped (fallback a media sin máscara si la máscara
        deja pocos píxeles — jersey verdoso, sombra rara)."""
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if image.size == 0:
            return np.array([128, 128, 128], dtype=float)

        top_half = image[0:int(image.shape[0] / 2), :]
        height, width = top_half.shape[:2]
        player_crop = top_half[:, int(width * 0.20):int(width * 0.80)]
        if player_crop.size == 0:
            return np.array([128, 128, 128], dtype=float)

        if _cfg.TEAM_HSV_MASK_ENABLED:
            hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]
            not_grass = (hue < _cfg.TEAM_GRASS_HUE_LOW) | (hue > _cfg.TEAM_GRASS_HUE_HIGH)
            kept_pct = 100.0 * not_grass.sum() / not_grass.size
            if kept_pct >= _cfg.TEAM_GRASS_MIN_PIXELS_PCT:
                return player_crop[not_grass].reshape(-1, 3).mean(axis=0)

        # Sin máscara (desactivada o demasiado agresiva): media del crop entero.
        return player_crop.reshape(-1, 3).mean(axis=0)

    # ------------------------------------------------------------------
    # T7.b/c — clustering
    def _fit_clusters(self, colors, anchor=False):
        """Entrena K-Means(2) sobre una lista de colores. Con anchor=True, los
        centroides nuevos se asignan a equipo 1/2 por cercanía a los centroides
        anteriores (los labels de sklearn son arbitrarios — sin anclaje, un
        re-fit puede intercambiar los equipos a mitad de partido)."""
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(colors)

        if anchor and self.kmeans is not None:
            old = np.array([self.team_colors[1], self.team_colors[2]], dtype=float)
            new = kmeans.cluster_centers_
            # Dos permutaciones posibles (2x2): elegir la de menor distancia total.
            direct  = np.linalg.norm(new[0] - old[0]) + np.linalg.norm(new[1] - old[1])
            swapped = np.linalg.norm(new[1] - old[0]) + np.linalg.norm(new[0] - old[1])
            if swapped < direct:
                kmeans.cluster_centers_ = new[::-1].copy()
                kmeans.labels_ = 1 - kmeans.labels_

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0].astype(int).tolist()
        self.team_colors[2] = kmeans.cluster_centers_[1].astype(int).tolist()

    def _fallback_colors(self):
        print("⚠️ Not enough player colors to train teams — using defaults.")
        self.team_colors[1] = (255, 255, 255)
        self.team_colors[2] = (0, 0, 0)
        self.kmeans = None

    def _predict_team(self, color):
        if self.kmeans is None:
            return 1
        return int(self.kmeans.predict(np.asarray(color, dtype=float).reshape(1, -1))[0]) + 1

    # ------------------------------------------------------------------
    # Entry point único (compartido por Main.py y debug_player_detection.py)
    def assign_teams(self, video_frames, players_tracks, fps):
        """Escribe 'team' y 'team_color' en players_tracks, in-place.

        Pipeline: burn-in inicial (T7.b) -> por frame: color + voto deslizante
        (T7.d) con re-cluster periódico anclado (T7.c). Las entradas con
        is_goalkeeper=True nunca entrenan centroides (T7.e).
        """
        n_frames = len(players_tracks)
        stride = max(1, _cfg.TEAM_SAMPLE_EVERY_N_FRAMES)
        burnin_frames = min(n_frames, max(1, round(_cfg.TEAM_BURNIN_SEC * fps)))
        recluster_every = round(_cfg.TEAM_RECLUSTER_SEC * fps) if _cfg.TEAM_RECLUSTER_SEC > 0 else 0
        vote_window = max(1, round(_cfg.TEAM_VOTE_WINDOW_SEC * fps))

        # --- T7.b: burn-in — pool de colores de campo (sin porteros) ---
        burnin_colors = []
        for frame_num in range(0, burnin_frames, stride):
            for player_id, track in players_tracks[frame_num].items():
                if track.get('is_goalkeeper', False):
                    continue
                burnin_colors.append(self.get_player_color(video_frames[frame_num], track['bbox']))

        if len(burnin_colors) < 2:
            self._fallback_colors()
        else:
            self._fit_clusters(burnin_colors)

        # --- por frame: voto deslizante + buffer de re-cluster ---
        votes_by_id = {}        # track_id -> deque de votos (1/2) de la ventana
        recluster_buffer = []   # colores recientes de jugadores de campo
        frames_since_recluster = 0

        for frame_num, player_track in enumerate(players_tracks):
            for player_id, track in player_track.items():
                color = self.get_player_color(video_frames[frame_num], track['bbox'])
                vote = self._predict_team(color)

                if player_id not in votes_by_id:
                    votes_by_id[player_id] = deque(maxlen=vote_window)
                votes_by_id[player_id].append(vote)

                votes = votes_by_id[player_id]
                team = max(set(votes), key=lambda t: (list(votes).count(t), -t))

                track['team'] = team
                track['team_color'] = self.team_colors[team]

                if not track.get('is_goalkeeper', False) and frame_num % stride == 0:
                    recluster_buffer.append(color)

            # --- T7.c: re-cluster periódico con anclaje ---
            frames_since_recluster += 1
            if recluster_every and frames_since_recluster >= recluster_every:
                if len(recluster_buffer) >= 2 and self.kmeans is not None:
                    self._fit_clusters(recluster_buffer, anchor=True)
                recluster_buffer = []
                frames_since_recluster = 0

        return players_tracks
