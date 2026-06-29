from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans

import config as _cfg

"""
Team Assignment Module (Tier 4 — robust jersey-colour clustering)

Classifies players into teams from jersey colour with four robustness measures
over the naive frame-0 K-Means:

  1. HSV segmentation — pitch grass (and optionally white lines) is masked out
     of the torso crop before averaging the colour, so green bleed at the bbox
     edges doesn't drag the cluster centroids towards the grass.
  2. Periodic re-clustering — the 2-team K-Means is re-fit every N seconds
     (config.TEAM_RECLUSTER_INTERVAL_SEC) on recently seen colours, so slow
     illumination / shadow drift doesn't lock the teams to frame-0 lighting.
     New centroids are aligned to the previous ones so team labels stay stable.
  3. Moving voting — instead of caching team_id permanently on first sight, a
     per-track vote histogram is kept and the running mode is returned. A single
     confused frame can no longer flip an ID for the rest of its track.
  4. Goalkeeper handling — keepers wear a distinct kit, so they are excluded
     from the 2-team K-Means training (they no longer contaminate the centroids)
     but are still assigned to the nearest team so downstream possession logic,
     which assumes team ∈ {1, 2}, keeps working.
"""


class TeamAssigner:
    """
    Assigns players to teams (1 or 2) from jersey colour.

    Attributes:
        team_colors (dict): Dominant BGR colour per team {1: [...], 2: [...]}.
        kmeans (KMeans): Active 2-cluster model (re-fit periodically).
        team_votes (dict): {player_id: Counter({team_id: votes})} — moving vote.
        fps (float): Video FPS, set by the caller; drives the re-cluster cadence.
    """

    def __init__(self):
        self.team_colors = {}
        self.kmeans = None

        # Moving voting: per-track histogram of team votes (replaces the old
        # permanent player_id -> team_id cache).
        self.team_votes = {}

        # Re-clustering state: buffer of recently observed outfield colours and
        # the frame at which we last re-fit.
        self._recent_colors = []        # list[(frame_num, np.ndarray[3])]
        self._last_recluster_frame = 0
        self.fps = 30.0                 # overwritten by the caller

    # ── Colour extraction ──────────────────────────────────────────────────

    def _foreground_mask(self, crop_bgr):
        """Boolean (H, W) mask of pixels that are NOT pitch grass (and, when
        TEAM_MASK_WHITE is on, not white lines either)."""
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        grass = (
            (h >= _cfg.TEAM_GRASS_HUE_LOW) & (h <= _cfg.TEAM_GRASS_HUE_HIGH)
            & (s >= _cfg.TEAM_GRASS_SAT_MIN) & (v >= _cfg.TEAM_GRASS_VAL_MIN)
        )
        keep = ~grass

        if _cfg.TEAM_MASK_WHITE:
            white = (s <= _cfg.TEAM_WHITE_SAT_MAX) & (v >= _cfg.TEAM_WHITE_VAL_MIN)
            keep &= ~white

        return keep

    def _jersey_pixels(self, frame, bbox):
        """Return the jersey-candidate pixels (N×3, BGR) for a bbox with grass
        masked out. Falls back to the raw centred crop when masking would leave
        too few pixels (heavy occlusion / a player standing on pure grass)."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return None

        # Torso: top half, horizontally centred (chest), away from the grass at
        # the bbox sides.
        top_half = image[0:image.shape[0] // 2, :]
        h, w = top_half.shape[:2]
        if h == 0 or w == 0:
            return None
        crop = top_half[:, int(w * 0.20):int(w * 0.80)]
        if crop.size == 0:
            return None

        pixels = crop.reshape(-1, 3)
        keep = self._foreground_mask(crop).reshape(-1)
        fg = pixels[keep]

        # If the mask ate almost everything, the crop is mostly grass / the
        # player is occluded — trust the raw crop over a few stray pixels.
        min_px = max(1, int(pixels.shape[0] * _cfg.TEAM_MIN_JERSEY_PIXEL_FRAC))
        if fg.shape[0] < min_px:
            return pixels
        return fg

    def get_player_color(self, frame, bbox):
        """Average jersey (BGR) colour for a player bbox, grass-masked."""
        fg = self._jersey_pixels(frame, bbox)
        if fg is None or fg.shape[0] == 0:
            return np.array([128, 128, 128], dtype=float)
        return fg.mean(axis=0)

    # ── Clustering ─────────────────────────────────────────────────────────

    def _adopt_kmeans(self, km):
        """Adopt a freshly fit 2-cluster model, aligning its cluster order to
        the existing team_colors so that label 0 stays team 1 and label 1 stays
        team 2. Without this, K-Means' arbitrary label order would flip every
        team on each re-cluster."""
        centers = km.cluster_centers_
        if self.team_colors:
            old1 = np.array(self.team_colors[1], dtype=float)
            old2 = np.array(self.team_colors[2], dtype=float)
            straight = np.linalg.norm(centers[0] - old1) + np.linalg.norm(centers[1] - old2)
            swapped = np.linalg.norm(centers[0] - old2) + np.linalg.norm(centers[1] - old1)
            if swapped < straight:
                centers = centers[::-1].copy()
                km.cluster_centers_ = centers

        self.kmeans = km
        self.team_colors[1] = centers[0].astype(int).tolist()
        self.team_colors[2] = centers[1].astype(int).tolist()

    def assign_team_color(self, frame, player_detections):
        """Train the initial 2-team model on the first frame.

        Goalkeepers (role == "goalkeeper") are skipped: their distinct kit would
        pull a centroid away from the two outfield colours.
        """
        player_colors = []
        for _, det in player_detections.items():
            if det.get("role") == "goalkeeper":
                continue
            player_colors.append(self.get_player_color(frame, det["bbox"]))

        # Need at least 2 outfield colours to split into two teams.
        if len(player_colors) < 2:
            print("⚠️ Not enough outfield players in frame 0 to train colours.")
            self.team_colors[1] = [255, 255, 255]
            self.team_colors[2] = [0, 0, 0]
            self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
            if player_colors:
                # Duplicate the single colour so fit() has >= n_clusters points.
                self.kmeans.fit(np.array(player_colors + player_colors))
            else:
                self.kmeans.fit(np.array([[255, 255, 255], [0, 0, 0]], dtype=float))
            return

        km = KMeans(n_clusters=2, init="k-means++", n_init=10)
        km.fit(np.array(player_colors))
        self.kmeans = km
        # First fit — no prior to align against, just seed team_colors.
        self.team_colors[1] = km.cluster_centers_[0].astype(int).tolist()
        self.team_colors[2] = km.cluster_centers_[1].astype(int).tolist()

    def _maybe_recluster(self, frame_num):
        """Re-fit the 2-team model on the most recent window of outfield colours
        if TEAM_RECLUSTER_INTERVAL_SEC has elapsed."""
        interval = _cfg.TEAM_RECLUSTER_INTERVAL_SEC
        if interval <= 0 or self.kmeans is None:
            return
        interval_frames = max(1, int(interval * self.fps))
        if frame_num - self._last_recluster_frame < interval_frames:
            return

        # Keep only colours from the most recent window (bounds memory too).
        window_start = frame_num - interval_frames
        self._recent_colors = [(fn, c) for (fn, c) in self._recent_colors if fn >= window_start]
        self._last_recluster_frame = frame_num

        recent = [c for (_, c) in self._recent_colors]
        if len(recent) < 2:
            return
        km = KMeans(n_clusters=2, init="k-means++", n_init=10)
        km.fit(np.array(recent))
        self._adopt_kmeans(km)

    # ── Per-frame assignment ───────────────────────────────────────────────

    def _predict_team(self, color):
        if self.kmeans is None:
            return 1
        return int(self.kmeans.predict(color.reshape(1, -1))[0]) + 1

    def get_player_team(self, frame, player_bbox, player_id, frame_num=0, role="player"):
        """Predict the team for a player in this frame, record the vote, and
        return the running mode of that track's votes.

        Goalkeepers are assigned to the nearest of the two team colours (so
        team ∈ {1, 2} downstream) but are NOT fed into the re-cluster buffer, so
        they never contaminate the outfield centroids.
        """
        color = self.get_player_color(frame, player_bbox)
        team_id = self._predict_team(color)

        if role != "goalkeeper":
            self._recent_colors.append((frame_num, color))
            self._maybe_recluster(frame_num)

        votes = self.team_votes.setdefault(player_id, Counter())
        votes[team_id] += 1
        return votes.most_common(1)[0][0]

    def final_team(self, player_id):
        """Final team for a track: the mode over its whole vote history. Used to
        lock one stable team per ID across every frame once the pass is done."""
        votes = self.team_votes.get(player_id)
        if not votes:
            return 1
        return votes.most_common(1)[0][0]
