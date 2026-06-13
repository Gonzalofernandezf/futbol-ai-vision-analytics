"""
run_chunked.py

Process long football match videos by splitting into N-minute chunks,
running Main.py on each, then merging the per-chunk JSON statistics
into one combined match_data.json.

Configuration (via .env or environment variables):
    CHUNK_DURATION_MIN   minutes per chunk                        (default 5)
    CHUNK_OVERLAP_SEC    seconds prepended to each non-first chunk (default 3)
    VIDEO_START_SEC      analysis start offset in seconds          (default 0)
    VIDEO_END_SEC        analysis end offset; unset = full video   (default unset)

Usage:
    python run_chunked.py
    CHUNK_DURATION_MIN=0.5 python run_chunked.py   # 30-second chunks for testing
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
from analytics.match_analytics import (
    attach_player_percentiles,
    compute_player_derived,
    compute_team_stats,
)
from utils.perf_monitor import export_processing_meta, run_sanity_checks


# ── helpers ────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def _video_meta(path):
    cap         = cv2.VideoCapture(path)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (frame_count / fps if fps > 0 else 0.0), fps


def _compute_chunks(start, end, chunk_dur, overlap):
    """
    Return [(actual_start, actual_end), ...].
    All chunks except the first are extended `overlap` seconds backwards
    so ByteTrack has context frames at chunk boundaries.
    """
    chunks = []
    logical = start
    while logical < end - 0.5:  # 0.5 s guard against floating-point creep
        logical_end  = min(logical + chunk_dur, end)
        actual_start = max(start, logical - overlap) if chunks else logical
        chunks.append((actual_start, logical_end))
        logical = logical_end
    return chunks


def _find_stats_json(directory):
    hits = glob.glob(os.path.join(directory, "*_stats.json"))
    return max(hits, key=os.path.getmtime) if hits else None


def _find_mp4(directory):
    hits = glob.glob(os.path.join(directory, "*.mp4"))
    return max(hits, key=os.path.getmtime) if hits else None


def _mean_pos(pos_list, n):
    """Mean [x, y] over the first n non-None entries of pos_list."""
    pts = [p for p in pos_list[:n] if p is not None]
    if not pts:
        return None
    return np.array(pts, dtype=float).mean(axis=0)


def _match_ids(prev_json, curr_json, overlap_sec, fps):
    """
    Build {curr_id: prev_id} by matching mean player positions in the
    overlap window (the same real-world seconds appear at the end of
    prev_json and the start of curr_json).
    Only accepts matches within 5 real-world metres.
    """
    THRESHOLD_M = 5.0
    n = max(1, int(overlap_sec * fps))

    prev_pos = {}
    for pid, pdata in prev_json.get("players", {}).items():
        pos = _mean_pos(pdata.get("position_history", [])[-n:], n)
        if pos is not None:
            prev_pos[pid] = pos

    # Use the second half of the overlap window to compute curr positions.
    # ByteTrack needs ~15 frames to stabilise IDs; skipping the first half avoids
    # averaging over unreliable early tracks.
    half = n // 2
    curr_pos = {}
    for pid, pdata in curr_json.get("players", {}).items():
        pos = _mean_pos(pdata.get("position_history", [])[half:n], n - half)
        if pos is not None:
            curr_pos[pid] = pos

    mapping = {}
    used    = set()
    for cid, cpos in curr_pos.items():
        best_pid, best_d = None, THRESHOLD_M
        for pid, ppos in prev_pos.items():
            if pid in used:
                continue
            d = float(np.linalg.norm(cpos - ppos))
            if d < best_d:
                best_d, best_pid = d, pid
        if best_pid is not None:
            mapping[cid] = best_pid
            used.add(best_pid)

    total     = len(curr_pos)
    matched   = len(mapping)
    print(f"   🔗 ID match: {matched}/{total} players matched "
          f"({total - matched} unmatched → new IDs)")

    return mapping


def _merge_chunks(chunk_jsons, overlaps):
    """
    Combine per-chunk player stats into one match JSON.
    overlaps[i] = seconds prepended to chunk i (0 for chunk 0).

    Per-player combination rules:
      - max_speed_kmh, max_acceleration_ms2 : maximum across chunks
      - total_distance_m                    : sum (overlap window adds ~1 %
                                              but granular per-frame data
                                              is not in the JSON)
      - position_history, speed_over_time   : concatenate after trimming
                                              the overlap window from
                                              each non-first chunk
    """
    fps = chunk_jsons[0]["match_meta"]["fps"]

    global_players = {}
    total_eff_dur  = 0.0
    home_w = away_w = 0.0
    prev_json = None

    for i, (cj, ov) in enumerate(zip(chunk_jsons, overlaps)):
        chunk_dur     = cj["match_meta"]["duration_seconds"]
        effective_dur = max(0.0, chunk_dur - ov)
        trim_frames   = int(ov * fps)
        trim_secs     = int(ov)

        id_map = _match_ids(prev_json, cj, ov, fps) if (prev_json and ov > 0) else {}
        seen   = set()

        for curr_id, cdata in cj["players"].items():
            global_id = id_map.get(curr_id, curr_id)

            # Collision: global_id already exists but is NOT this matched player
            if global_id in global_players and global_id not in id_map.values():
                global_id = f"{curr_id}_c{i + 1}"

            trimmed_pos = cdata["position_history"][trim_frames:]
            trimmed_sot = cdata["speed_over_time"][trim_secs:]

            if global_id in global_players:
                g = global_players[global_id]
                g["max_speed_kmh"]        = max(g["max_speed_kmh"],        cdata["max_speed_kmh"])
                g["max_acceleration_ms2"] = max(g["max_acceleration_ms2"], cdata["max_acceleration_ms2"])
                g["total_distance_m"]    += cdata["total_distance_m"]
                g["position_history"].extend(trimmed_pos)
                g["speed_over_time"].extend(trimmed_sot)
            else:
                pad_f = int(total_eff_dur * fps)
                pad_s = int(total_eff_dur)
                global_players[global_id] = {
                    "team":                 int(cdata["team"]),
                    "max_speed_kmh":        cdata["max_speed_kmh"],
                    "total_distance_m":     cdata["total_distance_m"],
                    "max_acceleration_ms2": cdata["max_acceleration_ms2"],
                    "position_history":     [None] * pad_f + trimmed_pos,
                    "speed_over_time":      [None] * pad_s + trimmed_sot,
                }

            seen.add(global_id)

        # Pad players absent from this chunk so all arrays stay aligned
        for gid in list(global_players.keys()):
            if gid not in seen:
                g       = global_players[gid]
                eff_f   = int(effective_dur * fps)
                eff_s   = int(effective_dur)
                g["position_history"].extend([None] * eff_f)
                g["speed_over_time"].extend([None]  * eff_s)

        home_w        += cj["match_meta"]["home_possession"] * effective_dur
        away_w        += cj["match_meta"]["away_possession"] * effective_dur
        total_eff_dur += effective_dur
        prev_json = cj

    total_eff_dur = max(total_eff_dur, 0.001)

    # Recalcular las analíticas v2 sobre los jugadores ya fusionados: los
    # bloques `derived`/`team_stats` de cada chunk solo cubren ese chunk y
    # no son válidos para el partido completo (mitades, peak_window,
    # percentiles intra-equipo, etc.). Mismo contrato que data_exporter.
    for pdata in global_players.values():
        pdata["derived"] = compute_player_derived(pdata, total_eff_dur)
    attach_player_percentiles(global_players)

    return {
        "schema_version": 2,
        "match_meta": {
            "duration_seconds": round(total_eff_dur, 2),
            "fps":              fps,
            "home_possession":  round(home_w / total_eff_dur, 1),
            "away_possession":  round(away_w / total_eff_dur, 1),
        },
        "players": global_players,
        "team_stats": compute_team_stats(global_players),
    }


# ── main ───────────────────────────────────────────────────────────────────

def main():
    video_path    = config.VIDEO_PATH
    chunk_dur_sec = config.CHUNK_DURATION_MIN * 60.0
    overlap_sec   = config.CHUNK_OVERLAP_SEC
    start_sec     = config.VIDEO_START_SEC

    video_dur, fps = _video_meta(video_path)
    if video_dur <= 0:
        print(f"❌ Cannot read video: {video_path}")
        sys.exit(1)

    end_sec = config.VIDEO_END_SEC if config.VIDEO_END_SEC is not None else video_dur

    print(f"📹 Video : {video_path}  ({video_dur:.1f}s at {fps:.2f} fps)")
    print(f"⏱️  Range : {start_sec:.1f}s → {end_sec:.1f}s")
    print(f"🧩 Chunk : {config.CHUNK_DURATION_MIN} min   Overlap: {overlap_sec}s")

    chunks   = _compute_chunks(start_sec, end_sec, chunk_dur_sec, overlap_sec)
    overlaps = [0.0] + [overlap_sec] * (len(chunks) - 1)

    print(f"📦 {len(chunks)} chunk(s) to process\n")

    project_root = Path(config.PROJECT_ROOT)
    output_dir   = str(config.OUTPUT_DIR)
    chunks_dir   = os.path.join(output_dir, "_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    chunk_jsons  = []
    chunk_times  = []  # wall-clock per chunk for the final summary
    phase_totals = {}  # suma de fases de todos los chunks (para processing_meta)

    for i, (cs, ce) in enumerate(chunks):
        tag      = f"chunk{i + 1:02d}"
        cdir     = os.path.join(chunks_dir, tag)
        demo_tmp = os.path.join(chunks_dir, f"{tag}_demo")

        print(f"{'─' * 60}")
        print(f"🎬 {tag}: {cs:.1f}s → {ce:.1f}s  ({ce - cs:.1f}s, overlap={overlaps[i]:.1f}s)")

        env = os.environ.copy()
        env["VIDEO_START_SEC"] = str(cs)
        env["VIDEO_END_SEC"]   = str(ce)
        env["OUTPUT_DIR"]      = cdir
        env["DEMO_DIR"]        = demo_tmp  # suppress real-demo deploy during chunk runs

        t0 = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(project_root / "Main.py")],
            env=env,
            cwd=str(project_root),
        )
        chunk_times.append(time.perf_counter() - t0)

        if result.returncode != 0:
            print(f"❌ Main.py failed on {tag} (exit {result.returncode}). Aborting.")
            sys.exit(1)

        # Move annotated video to output_dir with _chunkXX suffix
        mp4 = _find_mp4(cdir)
        if mp4:
            dest = os.path.join(
                output_dir,
                os.path.basename(mp4).replace(".mp4", f"_{tag}.mp4"),
            )
            shutil.move(mp4, dest)
            print(f"   📹 Video  → {dest}")

        json_path = _find_stats_json(cdir)
        if not json_path:
            print(f"❌ No *_stats.json found in {cdir}. Aborting.")
            sys.exit(1)

        with open(json_path) as f:
            chunk_jsons.append(json.load(f))

        # Acumular fases del chunk (Main.py deja processing_meta.json en su DEMO_DIR tmp)
        chunk_meta_path = os.path.join(demo_tmp, "processing_meta.json")
        if os.path.exists(chunk_meta_path):
            with open(chunk_meta_path) as f:
                for phase, secs in json.load(f)["timing"]["phases"].items():
                    phase_totals[phase] = phase_totals.get(phase, 0.0) + secs

        print(f"   📊 Stats  → {json_path}")

    print(f"\n{'=' * 60}")
    print(f"🔗 Merging {len(chunk_jsons)} chunk(s)...")

    merged = _merge_chunks(chunk_jsons, overlaps)

    today         = datetime.now().strftime("%Y-%m-%d")
    combined_path = os.path.join(output_dir, f"{today}_combined_stats.json")

    with open(combined_path, "w") as f:
        json.dump(merged, f, indent=4, cls=_NumpyEncoder)

    print(f"✅ Combined JSON → {combined_path}")

    # Deploy combined JSON to demo dashboard
    demo_dir      = str(config.DEMO_DIR)
    os.makedirs(demo_dir, exist_ok=True)
    demo_json_dst = os.path.join(demo_dir, "match_data.json")
    shutil.copy(combined_path, demo_json_dst)
    print(f"🚀 Demo deployed → {demo_json_dst}")

    # Artefactos de evaluación: copiar solo si existen (puede no haberse corrido
    # eval_keypoints.py todavía) y si origen y destino no son el mismo archivo.
    for eval_fname in ("eval_report.json", "eval_history.json"):
        eval_src = os.path.join(config.EVAL_ARTIFACTS_DIR, eval_fname)
        eval_dst = os.path.join(demo_dir, eval_fname)
        if os.path.exists(eval_src) and \
           os.path.normcase(os.path.abspath(eval_src)) != os.path.normcase(os.path.abspath(eval_dst)):
            shutil.copy(eval_src, eval_dst)
            print(f"   📈 Eval   → {eval_dst}")

    # Metadata de procesamiento combinada para la ruta /admin del dashboard.
    # Sanity checks sobre el JSON ya mergeado (sin tracks: esos checks se omiten).
    sanity_checks = run_sanity_checks(None, combined_path)
    export_processing_meta(
        os.path.join(demo_dir, "processing_meta.json"),
        phase_totals, sanity_checks, config.VIDEO_PATH,
        chunk_seconds=chunk_times,
    )

    print(f"\n📁 Per-chunk files  → {chunks_dir}/")
    print(f"ℹ️  Chunk videos saved in {output_dir}/ with _chunk01, _chunk02 ... suffix")
    print(f"ℹ️  To concatenate: ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4")

    # ── Run summary — quick sanity that all chunks completed in similar time ──
    if chunk_times:
        total_min  = sum(chunk_times) / 60
        avg_min    = (sum(chunk_times) / len(chunk_times)) / 60
        slowest_i  = int(np.argmax(chunk_times))
        fastest_i  = int(np.argmin(chunk_times))
        print("\n" + "═" * 60)
        print("📊 RUN SUMMARY")
        print("─" * 60)
        print(f"  Total time           : {total_min:.1f} min")
        print(f"  Chunks processed     : {len(chunk_times)} / {len(chunks)}")
        print(f"  Per-chunk avg        : {avg_min:.2f} min")
        print(f"  Slowest chunk        : {slowest_i + 1:02d} ({chunk_times[slowest_i]/60:.2f} min)")
        print(f"  Fastest chunk        : {fastest_i + 1:02d} ({chunk_times[fastest_i]/60:.2f} min)")
        print("═" * 60)


if __name__ == "__main__":
    main()
