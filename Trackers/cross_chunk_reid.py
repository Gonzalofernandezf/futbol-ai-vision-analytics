"""
Cross-chunk ReID (Tier 2) — reconcilia IDs de jugadores entre chunks de
run_chunked.py usando apariencia (embeddings ReID de BoT-SORT) + posición +
equipo, vía Hungarian assignment. Ver docs/PLAN_player_tracking.md, sección
"Tier 2 — Cross-chunk ReID".

Reemplaza el matching ingenuo (_match_ids en run_chunked.py: solo posición,
umbral fijo, greedy no-óptimo) por uno que respeta equipo como restricción dura
y usa apariencia además de posición — pero deja intacta la lógica de fusión
(_merge_chunks) que ya recorta el solape y concatena position_history/speed_over_time
correctamente.
"""
import pickle

import numpy as np
from scipy.optimize import linear_sum_assignment

import config


def build_chunk_state(tracks, embeddings=None):
    """Construye el estado de un chunk a partir de tracks['players'] (ya con
    'team' y 'position_transformed' asignados) y embeddings (dict
    track_id -> np.ndarray|None, de Tracker.get_track_embeddings()). Llamar en
    Main.py después de la asignación de equipos (Team voting), antes de
    exportar."""
    embeddings = embeddings or {}
    state = {}
    for frame_num, frame_players in enumerate(tracks['players']):
        for player_id, info in frame_players.items():
            pos = info.get('position_transformed')
            if player_id not in state:
                state[player_id] = {
                    "team_id": info.get('team'),
                    "mean_embedding": embeddings.get(player_id),
                    "first_position_m": pos,
                    "last_position_m": pos,
                    "frame_start": frame_num,
                    "frame_end": frame_num,
                }
            else:
                entry = state[player_id]
                entry["frame_end"] = frame_num
                if pos is not None:
                    entry["last_position_m"] = pos
                    if entry["first_position_m"] is None:
                        entry["first_position_m"] = pos
                if entry["team_id"] is None:
                    entry["team_id"] = info.get('team')

    for entry in state.values():
        entry["track_duration_frames"] = entry["frame_end"] - entry["frame_start"] + 1

    return state


def _cosine_distance(a, b):
    """[0, 1], 0 = idénticos. 1.0 (máxima distancia) si falta algún embedding —
    sin apariencia, el match depende solo del coste espacial."""
    if a is None or b is None:
        return 1.0
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos_sim = float(np.dot(a, b) / (na * nb))
    return (1.0 - cos_sim) / 2.0


def _euclidean(p1, p2):
    if p1 is None or p2 is None:
        return None
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


class ChunkBridge:
    """Encadena los chunk_state de un partido y calcula, para cada chunk, un
    mapeo local_id -> global_id consistente a lo largo de todo el partido."""

    def __init__(self, appearance_weight=None, spatial_weight=None,
                 max_spatial_m=None, max_cost=None):
        self.appearance_weight = (config.CROSS_CHUNK_APPEARANCE_WEIGHT
                                   if appearance_weight is None else appearance_weight)
        self.spatial_weight = (config.CROSS_CHUNK_SPATIAL_WEIGHT
                                if spatial_weight is None else spatial_weight)
        self.max_spatial_m = (config.CROSS_CHUNK_MAX_SPATIAL_M
                               if max_spatial_m is None else max_spatial_m)
        self.max_cost = config.CROSS_CHUNK_MAX_COST if max_cost is None else max_cost
        self.chunk_states = []

    def load_chunk_states(self, chunk_state_paths):
        """chunk_state_paths: lista de rutas a chunk_state.pkl en orden
        cronológico. Una ruta None/inexistente se trata como chunk vacío
        (no rompe la cadena, ese chunk simplemente no aporta matches)."""
        self.chunk_states = []
        for path in chunk_state_paths:
            state = {}
            if path is not None:
                try:
                    with open(path, 'rb') as f:
                        state = pickle.load(f)
                except (OSError, pickle.PickleError) as e:
                    print(f"⚠️  ChunkBridge: no se pudo leer {path} ({e}), chunk tratado como vacío.")
            self.chunk_states.append(state)
        return self.chunk_states

    def _match_pair(self, prev_state, curr_state):
        """Hungarian assignment entre el final de prev_state y el inicio de
        curr_state. Devuelve {curr_local_id: prev_local_id} solo para pares
        aceptados (mismo equipo, dentro de max_spatial_m, coste bajo max_cost)."""
        prev_ids = list(prev_state.keys())
        curr_ids = list(curr_state.keys())
        if not prev_ids or not curr_ids:
            return {}

        cost = np.full((len(prev_ids), len(curr_ids)), np.inf)
        for i, pid in enumerate(prev_ids):
            p = prev_state[pid]
            for j, cid in enumerate(curr_ids):
                c = curr_state[cid]

                # Restricción dura: nunca emparejar entre equipos distintos.
                if p.get("team_id") is not None and c.get("team_id") is not None \
                        and p["team_id"] != c["team_id"]:
                    continue

                spatial_m = _euclidean(p.get("last_position_m"), c.get("first_position_m"))
                if spatial_m is not None and spatial_m > self.max_spatial_m:
                    continue

                app_cost = _cosine_distance(p.get("mean_embedding"), c.get("mean_embedding"))
                spatial_cost = 0.0 if spatial_m is None else min(spatial_m / self.max_spatial_m, 1.0)

                cost[i, j] = (self.appearance_weight * app_cost
                              + self.spatial_weight * spatial_cost)

        row_idx, col_idx = linear_sum_assignment(cost)

        id_map = {}
        for i, j in zip(row_idx, col_idx):
            c = cost[i, j]
            if not np.isfinite(c) or c > self.max_cost:
                continue
            id_map[curr_ids[j]] = prev_ids[i]
        return id_map

    def compute_global_ids(self):
        """Devuelve una lista de dicts (uno por chunk): local_id -> global_id.
        El chunk 0 usa sus propios IDs locales como globales."""
        if not self.chunk_states:
            return []

        first_ids = list(self.chunk_states[0].keys())
        global_maps = [{lid: lid for lid in first_ids}]
        last_seen_global = dict(global_maps[0])  # local_id del chunk anterior -> global_id
        used_global_ids = set(global_maps[0].values())

        for i in range(1, len(self.chunk_states)):
            prev_state, curr_state = self.chunk_states[i - 1], self.chunk_states[i]
            local_to_prev = self._match_pair(prev_state, curr_state)

            curr_map = {}
            for local_id in curr_state.keys():
                prev_local = local_to_prev.get(local_id)
                if prev_local is not None and prev_local in last_seen_global:
                    curr_map[local_id] = last_seen_global[prev_local]
                else:
                    # Sin match aceptado: jugador "nuevo" desde la perspectiva
                    # global. Desambiguar si el ID local colisiona con uno ya
                    # en uso (mismo criterio que usaba _merge_chunks).
                    candidate = local_id
                    if candidate in used_global_ids:
                        candidate = f"{local_id}_c{i + 1}"
                    curr_map[local_id] = candidate

            used_global_ids.update(curr_map.values())
            global_maps.append(curr_map)
            last_seen_global = curr_map

        return global_maps
