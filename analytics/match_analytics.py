"""
MatchAnalytics v2 — pure derived-metric functions.

Each function takes only Python scalars/lists (no NumPy, no I/O) and is
fully deterministic: same input → same output. Null values in time-series
are handled gracefully; unreachable metrics return None, never raise.
"""

import math
from typing import Optional
import config

# A sprint segment requires at least this many consecutive qualifying seconds.
_SPRINT_MIN_SEC = 1


# ── internal helpers ──────────────────────────────────────────────────────────

def _active(speed_over_time: list) -> list:
    return [v for v in speed_over_time if v is not None]


def _get(obj, *keys):
    """Safe nested dict accessor; returns None on any missing key or wrong type."""
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


# ── per-player functions ──────────────────────────────────────────────────────

def compute_active_time(speed_over_time: list) -> float:
    return float(len(_active(speed_over_time)))


def compute_missing_data_pct(speed_over_time: list) -> Optional[float]:
    total = len(speed_over_time)
    if total == 0:
        return None
    return round(100.0 * sum(1 for v in speed_over_time if v is None) / total, 2)


def compute_sprints(
    speed_over_time: list,
    threshold_kmh: Optional[float] = None,
) -> dict:
    if threshold_kmh is None:
        threshold_kmh = config.SPRINT_THRESHOLD_KMH

    segments: list = []
    current: list = []

    for v in speed_over_time:
        if v is not None and v >= threshold_kmh:
            current.append(v)
        else:
            if len(current) >= _SPRINT_MIN_SEC:
                segments.append(current)
            current = []
    if len(current) >= _SPRINT_MIN_SEC:
        segments.append(current)

    if not segments:
        return {"count": 0, "longest_sec": 0, "total_distance_m": 0.0}

    return {
        "count": len(segments),
        "longest_sec": max(len(s) for s in segments),
        "total_distance_m": round(sum(v / 3.6 for s in segments for v in s), 2),
    }


def compute_speed_zones(
    speed_over_time: list,
    thresholds: Optional[dict] = None,
) -> dict:
    if thresholds is None:
        thresholds = config.SPEED_ZONE_THRESHOLDS

    zones = {"walk": 0, "jog": 0, "run": 0, "sprint": 0}
    for v in speed_over_time:
        if v is None:
            continue
        if v < thresholds["walk"]:
            zones["walk"] += 1
        elif v < thresholds["jog"]:
            zones["jog"] += 1
        elif v < thresholds["run"]:
            zones["run"] += 1
        else:
            zones["sprint"] += 1
    return zones


def compute_high_intensity_pct(
    speed_over_time: list,
    hi_threshold_kmh: Optional[float] = None,
) -> Optional[float]:
    if hi_threshold_kmh is None:
        hi_threshold_kmh = config.HI_THRESHOLD_KMH

    active = _active(speed_over_time)
    if not active:
        return None
    hi = sum(1 for v in active if v >= hi_threshold_kmh)
    return round(100.0 * hi / len(active), 2)


def compute_accelerations(
    speed_over_time: list,
    threshold_ms2: Optional[float] = None,
) -> dict:
    if threshold_ms2 is None:
        threshold_ms2 = config.HIGH_ACCEL_THRESHOLD_MS2

    # Discrete accelerations over consecutive valid-second pairs.
    # A null value resets the chain (no cross-gap derivatives).
    accels: list = []
    prev: Optional[float] = None
    for v in speed_over_time:
        if v is None:
            prev = None
            continue
        if prev is not None:
            accels.append((v - prev) / 3.6)  # Δv km/h → m/s, Δt = 1 s
        prev = v

    if not accels:
        return {"high_count": 0, "max_ms2": None}

    # Count upward threshold crossings (transition from below to at/above threshold).
    high_count = sum(
        1
        for i in range(1, len(accels))
        if accels[i - 1] < threshold_ms2 and accels[i] >= threshold_ms2
    )

    return {
        "high_count": high_count,
        "max_ms2": round(max(accels), 2),
    }


def _half_stats(speeds: list) -> dict:
    valid = [v for v in speeds if v is not None]
    if not valid:
        return {
            "distance_m": None,
            "avg_speed_kmh": None,
            "sprints": None,
            "high_intensity_pct": None,
        }
    return {
        "distance_m": round(sum(v / 3.6 for v in valid), 2),
        "avg_speed_kmh": round(sum(valid) / len(valid), 2),
        "sprints": compute_sprints(speeds)["count"],
        "high_intensity_pct": compute_high_intensity_pct(speeds),
    }


def compute_halves(speed_over_time: list, duration_seconds: float) -> dict:
    mid = int(duration_seconds / 2)
    return {
        "first": _half_stats(speed_over_time[:mid]),
        "second": _half_stats(speed_over_time[mid:]),
    }


def compute_drop_off(halves: dict) -> Optional[float]:
    """Caída de velocidad media de la primera a la segunda mitad.

    Positivo = el jugador bajó el ritmo en la segunda mitad (declive);
    negativo = lo subió. Misma convención que el dashboard (insights.ts).
    """
    avg1 = _get(halves, "first", "avg_speed_kmh")
    avg2 = _get(halves, "second", "avg_speed_kmh")
    if avg1 is None or avg2 is None or avg1 == 0:
        return None
    return round((avg1 - avg2) / avg1 * 100, 2)


def compute_peak_window(
    speed_over_time: list,
    window_sec: Optional[int] = None,
) -> Optional[dict]:
    if window_sec is None:
        window_sec = config.PEAK_WINDOW_SEC

    n = len(speed_over_time)
    if n == 0:
        return None

    w = min(window_sec, n)
    best_avg, best_start = -1.0, 0

    for start in range(n - w + 1):
        valid = [v for v in speed_over_time[start: start + w] if v is not None]
        if not valid:
            continue
        avg = sum(valid) / len(valid)
        if avg > best_avg:
            best_avg, best_start = avg, start

    if best_avg < 0:
        return None

    return {
        "start_sec": best_start,
        "end_sec": best_start + w,
        "avg_speed_kmh": round(best_avg, 2),
    }


def compute_position_metrics(
    position_history: list,
    pitch_length: Optional[float] = None,
    pitch_width: Optional[float] = None,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
) -> dict:
    if pitch_length is None:
        pitch_length = config.PITCH_LENGTH_M
    if pitch_width is None:
        pitch_width = config.PITCH_WIDTH_M
    if grid_rows is None:
        grid_rows = config.ANALYTICS_GRID_ROWS
    if grid_cols is None:
        grid_cols = config.ANALYTICS_GRID_COLS

    _null = {
        "centroid_m": None,
        "dispersion_m": None,
        "coverage_pct": None,
        "dominant_cell": None,
        "lateral_band": None,
    }

    valid = [p for p in position_history if p is not None]
    if len(valid) < 2:
        return _null

    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    # 2-D dispersion: sqrt(var_x + var_y)
    var_x = sum((x - cx) ** 2 for x in xs) / len(xs)
    var_y = sum((y - cy) ** 2 for y in ys) / len(ys)

    # Grid coverage
    cell_h = pitch_length / grid_rows
    cell_w = pitch_width / grid_cols
    cell_counts: dict = {}
    for p in valid:
        row = max(0, min(int(p[0] / cell_h), grid_rows - 1))
        col = max(0, min(int(p[1] / cell_w), grid_cols - 1))
        cell_counts[(row, col)] = cell_counts.get((row, col), 0) + 1

    dominant = max(cell_counts, key=cell_counts.get)

    # Lateral band from centroid y
    left_bound = pitch_width / 3
    right_bound = pitch_width * 2 / 3
    if cy < left_bound:
        band = "left"
    elif cy > right_bound:
        band = "right"
    else:
        band = "center"

    return {
        "centroid_m": [round(cx, 2), round(cy, 2)],
        "dispersion_m": round(math.sqrt(var_x + var_y), 2),
        "coverage_pct": round(len(cell_counts) / (grid_rows * grid_cols) * 100, 2),
        "dominant_cell": list(dominant),
        "lateral_band": band,
    }


def compute_player_derived(player_data: dict, duration_seconds: float) -> dict:
    sot = player_data.get("speed_over_time") or []
    ph = player_data.get("position_history") or []

    halves = compute_halves(sot, duration_seconds)
    return {
        "active_time_sec": compute_active_time(sot),
        "missing_data_pct": compute_missing_data_pct(sot),
        "sprints": compute_sprints(sot),
        "speed_zones_sec": compute_speed_zones(sot),
        "high_intensity_pct": compute_high_intensity_pct(sot),
        "accelerations": compute_accelerations(sot),
        "halves": halves,
        "drop_off_pct": compute_drop_off(halves),
        "peak_window": compute_peak_window(sot),
        "position": compute_position_metrics(ph),
    }


def attach_player_percentiles(players_export: dict) -> None:
    """Añade derived.percentiles a cada jugador del export.

    Rank intra-equipo 0-100 por métrica: 100 * (#valores del equipo <= valor) / n,
    la misma convención que usa el dashboard en PlayerRadar (pctRank).
    Solo se exportan las métricas que consume el frontend (insights.ts):
    sprints, high_intensity_pct y max_speed_kmh.
    Muta players_export in place; jugadores sin valor para una métrica
    simplemente no reciben esa clave.
    """
    metric_paths = {
        "max_speed_kmh":      ("max_speed_kmh",),
        "sprints":            ("derived", "sprints", "count"),
        "high_intensity_pct": ("derived", "high_intensity_pct"),
    }

    teams: dict = {}
    for pdata in players_export.values():
        teams.setdefault(pdata.get("team"), []).append(pdata)

    for members in teams.values():
        for metric, path in metric_paths.items():
            values = [v for v in (_get(p, *path) for p in members) if v is not None]
            n = len(values)
            if n == 0:
                continue
            for p in members:
                v = _get(p, *path)
                if v is None or not isinstance(p.get("derived"), dict):
                    continue
                pct = 100.0 * sum(1 for x in values if x <= v) / n
                p["derived"].setdefault("percentiles", {})[metric] = round(pct, 1)


# ── team aggregates ───────────────────────────────────────────────────────────

def _percentiles(values: list, pcts: tuple = (5, 25, 50, 75, 95)) -> dict:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return {f"p{p}": None for p in pcts}
    n = len(clean)
    result = {}
    for p in pcts:
        idx = (p / 100) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        result[f"p{p}"] = round(clean[lo] * (1 - frac) + clean[hi] * frac, 2)
    return result


def compute_team_stats(players_export: dict) -> dict:
    teams: dict = {1: [], 2: []}
    for pid, pdata in players_export.items():
        t = pdata.get("team")
        if t in teams:
            teams[t].append((pid, pdata))

    out = {}
    for tid, members in teams.items():
        if not members:
            out[str(tid)] = None
            continue

        def _leader(*keys):
            best_pid, best_val = None, None
            for pid, p in members:
                v = _get(p, *keys)
                if v is not None and (best_val is None or v > best_val):
                    best_val, best_pid = v, pid
            return {"player_id": best_pid, "value": best_val}

        out[str(tid)] = {
            "players_count": len(members),
            "percentiles": {
                "total_distance_m":   _percentiles([_get(p, "total_distance_m")        for _, p in members]),
                "max_speed_kmh":      _percentiles([_get(p, "max_speed_kmh")           for _, p in members]),
                "sprints_count":      _percentiles([_get(p, "derived", "sprints", "count") for _, p in members]),
                "high_intensity_pct": _percentiles([_get(p, "derived", "high_intensity_pct") for _, p in members]),
                "drop_off_pct":       _percentiles([_get(p, "derived", "drop_off_pct") for _, p in members]),
                "active_time_sec":    _percentiles([_get(p, "derived", "active_time_sec") for _, p in members]),
            },
            "leaders": {
                "total_distance_m": _leader("total_distance_m"),
                "max_speed_kmh":    _leader("max_speed_kmh"),
                "sprints_count":    _leader("derived", "sprints", "count"),
            },
        }

    return out
