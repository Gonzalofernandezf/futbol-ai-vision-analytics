#!/usr/bin/env python3
"""
render_annotated_video.py

Re-renders an annotated football match video from a pre-computed stats JSON
(combined_stats.json from run_chunked.py, or *_stats.json from Main.py),
without touching the ML/detection pipeline.

Player positions are back-projected from pitch metres to approximate pixel
coordinates via a linear mapping (pitch_length_m → frame_w, pitch_width_m → frame_h).
Ellipses / speed labels will appear near — not exactly at — player locations.

Usage
-----
    python render_annotated_video.py \\
        --video  video_OG.mp4 \\
        --json   output_videos/2024-01-01_combined_stats.json \\
        --output output_videos/render.mp4

    # Optional clip window
    python render_annotated_video.py \\
        --video video_OG.mp4 --json stats.json --output clip.mp4 \\
        --start-sec 30 --end-sec 90
"""

import argparse
import json
import pickle
import sys

import cv2
import numpy as np

import config
from Trackers.tracker import Tracker
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator


# ── constants ────────────────────────────────────────────────────────────────

_BBOX_W = 30    # synthetic player bbox width  (pixels)
_BBOX_H = 60    # synthetic player bbox height (pixels)

# BGR fallback colours; the stats JSON stores team index (1/2) but not the
# actual jersey colour that TeamAssigner learned from the frame.
_TEAM_COLORS: dict[int, tuple] = {
    1: (0,   0, 255),   # red
    2: (255, 50,  50),  # blue
}


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _synthetic_bbox(x_px: int, y_px: int) -> list:
    """Return [x1, y1, x2, y2] with player feet at (x_px, y_px)."""
    return [
        float(x_px - _BBOX_W // 2),
        float(y_px - _BBOX_H),
        float(x_px + _BBOX_W // 2),
        float(y_px),
    ]


def build_tracks(
    players_json: dict,
    json_fps: float,
    json_start_frame: int,
    clip_frames: int,
    frame_w: int,
    frame_h: int,
    pitch_length: float,
    pitch_width: float,
) -> dict:
    """
    Reconstruct a minimal ``tracks`` dict from the stats JSON for a clip window.

    Available from the JSON
    -----------------------
    - team assignment + approximate team colour
    - position_history : metres on pitch, one entry per original video frame
    - speed_over_time  : speed in km/h, one entry per second

    Approximated
    ------------
    - pixel bbox        : linearly back-projected from metres
    - cumulative dist   : linearly interpolated from total_distance_m

    tracks["ball"] and tracks["referees"] are always empty (not exported to JSON).
    """
    players_frames = [{} for _ in range(clip_frames)]
    ball_frames    = [{} for _ in range(clip_frames)]
    referee_frames = [{} for _ in range(clip_frames)]

    for pid_str, pdata in players_json.items():
        try:
            pid = int(pid_str)
        except ValueError:
            pid = pid_str

        team       = int(pdata.get("team", 0))
        team_color = _TEAM_COLORS.get(team, (0, 0, 255))
        total_dist = float(pdata.get("total_distance_m", 0.0))
        pos_hist   = pdata.get("position_history", [])
        sot        = pdata.get("speed_over_time",  [])   # one value per second

        for local_f in range(clip_frames):
            global_f = json_start_frame + local_f
            if global_f >= len(pos_hist):
                continue
            pos = pos_hist[global_f]
            if pos is None:
                continue

            x_m, y_m = float(pos[0]), float(pos[1])

            # Linear back-projection: metres → approximate pixels
            x_px = int(np.clip(x_m / pitch_length * frame_w, 1, frame_w - 1))
            y_px = int(np.clip(y_m / pitch_width  * frame_h, _BBOX_H, frame_h - 1))

            # Speed: use per-second bucket for this global frame
            sec_idx   = int(global_f / json_fps) if json_fps > 0 else 0
            speed_val = (
                float(sot[sec_idx])
                if sec_idx < len(sot) and sot[sec_idx] is not None
                else 0.0
            )

            # Distance: linear approximation of cumulative value
            dist_val = total_dist * (local_f / max(clip_frames - 1, 1))

            players_frames[local_f][pid] = {
                "bbox":       _synthetic_bbox(x_px, y_px),
                "team":       team,
                "team_color": team_color,
                "speed":      speed_val,
                "distance":   round(dist_val, 1),
            }

    return {
        "players":  players_frames,
        "ball":     ball_frames,
        "referees": referee_frames,
    }


def build_team_ball_control(
    match_meta: dict,
    clip_frames: int,
    fps: float,
) -> np.ndarray:
    """
    Synthesise a per-frame team_ball_control array from possession percentages.

    Frames are distributed in ~1-second blocks so the running cumulative
    percentage stays stable throughout the clip rather than snapping
    from 100 % to the final value in one step.
    """
    home_poss = float(match_meta.get("home_possession", 50.0)) / 100.0
    block     = max(1, int(fps))
    arr: list[int] = []

    for start in range(0, clip_frames, block):
        n  = min(block, clip_frames - start)
        n1 = max(0, min(n, round(n * home_poss)))
        arr.extend([1] * n1 + [2] * (n - n1))

    return np.array(arr[:clip_frames], dtype=int)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-render annotated football video from a pre-computed stats JSON or tracks pickle"
    )
    parser.add_argument("--video",     required=True,
                        help="Path to the original (unannotated) video")
    parser.add_argument("--json",      default=None,
                        help="Path to combined_stats.json (or *_stats.json). "
                             "Used as fallback when --tracks-pkl is not given. "
                             "Overlays will be approximate (back-projected from metres).")
    parser.add_argument("--tracks-pkl", default=None,
                        help="Path to *_tracks.pkl produced by Main.py. "
                             "Preferred over --json: gives exact pixel-perfect overlays.")
    parser.add_argument("--output",    required=True,
                        help="Output path for the annotated video (.mp4)")
    parser.add_argument("--start-sec", type=float, default=0.0,
                        help="Clip start in seconds (default: 0)")
    parser.add_argument("--end-sec",   type=float, default=None,
                        help="Clip end in seconds (default: full video)")
    args = parser.parse_args()

    if args.tracks_pkl is None and args.json is None:
        print("❌ Must pass either --tracks-pkl (preferred) or --json.", file=sys.stderr)
        sys.exit(1)

    # ── load source ──────────────────────────────────────────────────────────
    # Two paths:
    #   A. --tracks-pkl (preferred): exact bboxes, exact team colours, exact
    #      ball positions, exact per-frame possession array. Pixel-perfect.
    #   B. --json (fallback): back-projects metres → approximate pixels,
    #      synthesises possession blocks from the overall %. Overlay positions
    #      are close but not exact. Use only when the pickle is unavailable.
    use_pickle = args.tracks_pkl is not None
    if use_pickle:
        with open(args.tracks_pkl, 'rb') as fh:
            bundle = pickle.load(fh)
        precomputed_tracks    = bundle["tracks"]
        precomputed_tbc       = np.asarray(bundle["team_ball_control"])
        json_fps              = float(bundle["fps"])
        match_meta            = {}
        players               = {}
        print(f"📦 Using tracks pickle: {args.tracks_pkl}  (pixel-perfect mode)")
    else:
        data       = _load_json(args.json)
        match_meta = data.get("match_meta", {})
        players    = data.get("players",   {})
        json_fps   = float(match_meta.get("fps", 25.0))
        precomputed_tracks = None
        precomputed_tbc    = None
        print(f"📊 Using JSON (approximate mode): {args.json}")

    # ── open source video ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    video_fps   = cap.get(cv2.CAP_PROP_FPS) or json_fps
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(args.start_sec * video_fps)
    end_frame   = (
        int(args.end_sec * video_fps)
        if args.end_sec is not None
        else total_frames_video
    )
    end_frame   = min(end_frame, total_frames_video)
    clip_frames = end_frame - start_frame

    if clip_frames <= 0:
        print("❌ Empty clip range — check --start-sec / --end-sec.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    # Frame index into position_history that corresponds to clip start
    json_start_frame = int(args.start_sec * json_fps)

    print(f"📹 Video  : {args.video}  ({total_frames_video} frames @ {video_fps:.2f} fps)")
    print(f"📊 JSON   : {args.json}  (json_fps={json_fps})")
    print(f"✂️  Clip   : frames {start_frame} → {end_frame}  ({clip_frames} frames)")
    print(f"💾 Output : {args.output}")

    # ── reconstruct tracks ────────────────────────────────────────────────────
    if use_pickle:
        # Slice the per-frame lists from the pickle to the requested clip window.
        # Pickle frames are 0-indexed from the run that produced it (i.e. relative
        # to that pipeline call's VIDEO_START_SEC), which matches the video frame
        # ordering here as long as the caller passes the same video.
        def _slice(lst):
            return list(lst[json_start_frame:json_start_frame + clip_frames])

        tracks = {
            "players":  _slice(precomputed_tracks.get("players",  [])),
            "ball":     _slice(precomputed_tracks.get("ball",     [])),
            "referees": _slice(precomputed_tracks.get("referees", [])),
        }
        # Pad short slices so the render loop doesn't IndexError on the last frames.
        for key in ("players", "ball", "referees"):
            while len(tracks[key]) < clip_frames:
                tracks[key].append({})

        team_ball_control = np.asarray(
            precomputed_tbc[json_start_frame:json_start_frame + clip_frames]
        )
        if len(team_ball_control) < clip_frames:
            pad = np.zeros(clip_frames - len(team_ball_control), dtype=int)
            team_ball_control = np.concatenate([team_ball_control, pad])
    else:
        tracks = build_tracks(
            players_json     = players,
            json_fps         = json_fps,
            json_start_frame = json_start_frame,
            clip_frames      = clip_frames,
            frame_w          = frame_w,
            frame_h          = frame_h,
            pitch_length     = config.PITCH_LENGTH_M,
            pitch_width      = config.PITCH_WIDTH_M,
        )
        team_ball_control = build_team_ball_control(match_meta, clip_frames, video_fps)

    # ── drawing helpers — no YOLO or ByteTrack loaded ─────────────────────────
    # Tracker.__new__ skips __init__ (avoids loading YOLO weights).
    # All drawing methods only use cv2 / numpy — no instance state required.
    tracker = Tracker.__new__(Tracker)
    sde     = SpeedAndDistance_Estimator()
    sde.frame_rate = video_fps

    # ── video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    writer = cv2.VideoWriter(args.output, fourcc, video_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"❌ Cannot open VideoWriter: {args.output}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    # ── seek to clip start ────────────────────────────────────────────────────
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ── render frame-by-frame ─────────────────────────────────────────────────
    print("🎨 Rendering...")
    for local_f in range(clip_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  Video stream ended at local frame {local_f}")
            break

        frame = tracker.draw_annotations_frame(frame, local_f, tracks)
        frame = tracker.draw_team_ball_control(frame, local_f, team_ball_control)
        frame = sde.draw_speed_and_distance_frame(frame, local_f, tracks)
        writer.write(frame)

        if local_f > 0 and local_f % 500 == 0:
            print(f"  ✍️  {local_f}/{clip_frames} frames rendered...")

    cap.release()
    writer.release()
    print(f"✅ Done → {args.output}")


if __name__ == "__main__":
    main()
