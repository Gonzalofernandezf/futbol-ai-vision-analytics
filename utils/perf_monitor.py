"""
perf_monitor — Per-chunk timing and output sanity checks.

Used to spot regressions when stacking multiple performance changes in a
single branch: if a phase suddenly takes 5x longer or a sanity metric goes
red, we know which optimisation broke something without having to bisect.

Two pieces:
  - PhaseTimer: context manager that accumulates wall-clock time per named
    phase, prints a sorted report at the end.
  - run_sanity_checks: reads the exported JSON + tracks and validates
    output ranges (player count, ball detection rate, max speed, etc.).
    Returns the number of issues but never raises — we'd rather finish the
    chunk and surface warnings than abort 1h of compute.
"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import config


class PhaseTimer:
    def __init__(self):
        self.phases = {}
        self.order  = []

    @contextmanager
    def phase(self, name):
        if name not in self.phases:
            self.phases[name] = 0.0
            self.order.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] += time.perf_counter() - start

    def report(self):
        total = sum(self.phases.values())
        if total <= 0:
            return
        print("═" * 55)
        print("⏱️  CHUNK TIMING REPORT")
        print("─" * 55)
        for name in self.order:
            t   = self.phases[name]
            pct = t / total * 100
            print(f"  {name:<22}: {t:7.1f}s  ({pct:5.1f}%)")
        print("─" * 55)
        print(f"  {'Total':<22}: {total:7.1f}s  ({total/60:.2f} min)")
        print("═" * 55)


def _check(label, ok, value_str, expected_str, severity="warn"):
    status = "ok" if ok else severity
    icon = {"ok": "✅", "warn": "⚠️ ", "error": "❌"}[status]
    print(f"  {icon} {label:<22}: {value_str}  (expected {expected_str})")
    return {"label": label, "value": value_str, "status": status, "expected": expected_str}


def run_sanity_checks(tracks, json_path):
    """
    Validate per-chunk output against expected ranges.

    Prints a report and returns the list of structured checks
    ({label, value, status, expected}) — same shape the dashboard's admin
    page consumes via processing_meta.json.
    Designed to never raise — we never want a check to abort a long run.

    `tracks` is optional: pass None (e.g. from run_chunked.py, which only has
    the merged JSON) and the tracks-dependent checks are skipped.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"🔍 SANITY CHECKS — could not read {json_path}: {e}")
        return [{"label": "Stats JSON readable", "value": "no", "status": "error", "expected": "yes"}]

    players = data.get("players", {})
    meta    = data.get("match_meta", {})

    print("🔍 SANITY CHECKS")
    print("─" * 55)
    checks = []

    # 1. Player count
    n = len(players)
    checks.append(_check(
        "Players detected", 14 <= n <= 30, str(n), "14-30",
    ))

    # 2. Ball detection rate (post all filters)
    if tracks and "ball" in tracks and len(tracks["ball"]) > 0:
        total = len(tracks["ball"])
        with_ball = sum(1 for f in tracks["ball"] if f and 1 in f)
        rate = with_ball / total * 100
        checks.append(_check(
            "Ball detection rate", rate >= 40, f"{rate:.0f}%", ">40%",
            severity="warn" if rate >= 20 else "error",
        ))

    # 3. Homography coverage (transformed position present)
    pos_covered = 0
    pos_total   = 0
    for p in players.values():
        for pos in p.get("position_history", []) or []:
            pos_total += 1
            if pos is not None:
                pos_covered += 1
    if pos_total > 0:
        cov = pos_covered / pos_total * 100
        checks.append(_check(
            "Homography coverage", cov >= 60, f"{cov:.0f}%", ">60%",
            severity="warn" if cov >= 30 else "error",
        ))

    # 4. Max speed plausibility
    max_speed = max((p.get("max_speed_kmh", 0) for p in players.values()), default=0)
    checks.append(_check(
        "Max speed (km/h)", max_speed <= 35, f"{max_speed:.1f}", "<35",
    ))

    # 5. Possession sum ~ 100
    home = meta.get("home_possession", 0)
    away = meta.get("away_possession", 0)
    total_poss = home + away
    checks.append(_check(
        "Possession sum", 95 <= total_poss <= 105, f"{total_poss:.1f}%", "~100%",
    ))

    # 6. Average distance per player
    distances = [p.get("total_distance_m", 0) for p in players.values()]
    avg_dist  = sum(distances) / len(distances) if distances else 0
    checks.append(_check(
        "Avg distance per pl.", 20 <= avg_dist <= 2000, f"{avg_dist:.0f} m", "20-2000",
    ))

    issues = sum(1 for c in checks if c["status"] != "ok")
    print("═" * 55)
    if issues == 0:
        print("✅ All sanity checks passed.")
    else:
        print(f"⚠️  {issues} sanity check(s) outside expected range — review output.")
    print("═" * 55)
    return checks


def _detect_gpu_model():
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            return f"{name} x{n}" if n > 1 else name
    except Exception:
        pass
    return "CPU"


def export_processing_meta(output_path, phases, sanity_checks, video_path,
                           chunk_seconds=None):
    """
    Escribe processing_meta.json (consumido por la ruta /admin del dashboard).

    Args:
        output_path (str): destino del JSON (normalmente DEMO_DIR/processing_meta.json).
        phases (dict): {nombre_fase: segundos} — PhaseTimer.phases, o la suma
            de fases de todos los chunks en run_chunked.py.
        sanity_checks (list): salida de run_sanity_checks().
        video_path (str): vídeo de entrada, usado para derivar match_id.
        chunk_seconds (list[float] | None): duración wall-clock de cada chunk.
            None = run de un solo paso (Main.py directo).
    """
    if chunk_seconds:
        total = sum(chunk_seconds)
        per_chunk = chunk_seconds
    else:
        total = sum(phases.values())
        per_chunk = [total]

    meta = {
        "match_id": f"{datetime.now():%Y-%m-%d}_{Path(video_path).stem}",
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": config.PIPELINE_VERSION,
        "models": {
            "players": os.path.basename(config.MODEL_PATH),
            "ball":    os.path.basename(config.BALL_MODEL_PATH),
            "field":   os.path.basename(config.MODELO_CANCHA_PATH),
        },
        "timing": {
            "total_seconds":         round(total),
            "chunks_processed":      len(per_chunk),
            "per_chunk_avg_seconds": round(sum(per_chunk) / len(per_chunk)),
            "slowest_chunk_seconds": round(max(per_chunk)),
            "fastest_chunk_seconds": round(min(per_chunk)),
            "phases": {name: round(secs, 1) for name, secs in phases.items()},
        },
        "sanity_checks": sanity_checks,
        # gpu_util y coste no se miden todavía — 0 explícito, no inventado.
        "infra": {
            "gpu_model": _detect_gpu_model(),
            "gpu_util_avg_pct": 0,
            "estimated_cost_usd": 0.0,
        },
    }

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"     Meta:  {output_path}")
    return meta


def tracking_metrics(tracks, fps):
    """Calcula métricas de tracking sobre tracks['players'].

    Útil para comparar antes/después de cambios en ByteTrack o el detector:
    un id_churn_ratio cercano a 1.0 indica que casi no hay IDs duplicados
    por jugador real.

    Args:
        tracks (dict): Resultado de Tracker.get_object_tracks() (clave 'players').
        fps (float): FPS del vídeo procesado.

    Returns:
        dict: Métricas de tracking listas para imprimir o comparar.
    """
    track_durations = {}   # {track_id: frame_count}
    track_teams     = {}   # {track_id: [team_id, ...]}

    for frame in tracks.get("players", []):
        for tid, info in frame.items():
            track_durations[tid] = track_durations.get(tid, 0) + 1
            if "team" in info:
                track_teams.setdefault(tid, []).append(info["team"])

    long_tracks = [tid for tid, d in track_durations.items() if d >= fps * 5]
    flips = sum(1 for teams in track_teams.values() if len(set(teams)) > 1)

    return {
        "num_unique_player_ids": len(track_durations),
        "num_long_tracks":       len(long_tracks),
        "avg_track_duration_s":  sum(track_durations.values()) / max(len(track_durations), 1) / fps,
        "id_churn_ratio":        len(track_durations) / max(len(long_tracks), 1),
        "team_flip_rate":        flips / max(len(track_teams), 1),
    }
