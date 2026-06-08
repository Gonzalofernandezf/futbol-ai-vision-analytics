"""
Unit tests for analytics/match_analytics.py (MatchAnalytics v2).

Run with:  python -m pytest tests/ -v
"""

import pytest
from analytics.match_analytics import (
    compute_active_time,
    compute_missing_data_pct,
    compute_sprints,
    compute_speed_zones,
    compute_high_intensity_pct,
    compute_accelerations,
    compute_halves,
    compute_drop_off,
    compute_peak_window,
    compute_position_metrics,
    compute_player_derived,
    compute_team_stats,
)


# ── compute_active_time ───────────────────────────────────────────────────────

def test_active_time_basic():
    assert compute_active_time([1.0, None, 3.0]) == 2.0

def test_active_time_all_null():
    assert compute_active_time([None, None]) == 0.0

def test_active_time_empty():
    assert compute_active_time([]) == 0.0


# ── compute_missing_data_pct ──────────────────────────────────────────────────

def test_missing_data_pct_partial():
    assert compute_missing_data_pct([1.0, None, 3.0]) == pytest.approx(33.33, abs=0.01)

def test_missing_data_pct_empty():
    assert compute_missing_data_pct([]) is None

def test_missing_data_pct_all_present():
    assert compute_missing_data_pct([1.0, 2.0]) == 0.0


# ── compute_sprints ───────────────────────────────────────────────────────────

def test_sprints_two_segments():
    sot = [22.0, 23.0, 5.0, 25.0, 26.0, 27.0]
    r = compute_sprints(sot, threshold_kmh=21.0)
    assert r["count"] == 2
    assert r["longest_sec"] == 3

def test_sprints_single_second_counts():
    r = compute_sprints([25.0], threshold_kmh=21.0)
    assert r["count"] == 1
    assert r["longest_sec"] == 1

def test_sprints_null_breaks_segment():
    # null in the middle should split one long segment into two
    r = compute_sprints([22.0, None, 22.0], threshold_kmh=21.0)
    assert r["count"] == 2

def test_sprints_none():
    r = compute_sprints([5.0, 10.0, 15.0], threshold_kmh=21.0)
    assert r["count"] == 0
    assert r["total_distance_m"] == 0.0

def test_sprints_total_distance():
    # 2 s at 21.6 km/h = 2 * (21.6/3.6) = 12 m
    r = compute_sprints([21.6, 21.6], threshold_kmh=21.0)
    assert r["total_distance_m"] == pytest.approx(12.0, abs=0.01)


# ── compute_speed_zones ───────────────────────────────────────────────────────

_ZONES = {"walk": 7.0, "jog": 15.0, "run": 21.0}

def test_speed_zones_one_per_zone():
    r = compute_speed_zones([3.0, 10.0, 18.0, 25.0, None], thresholds=_ZONES)
    assert r == {"walk": 1, "jog": 1, "run": 1, "sprint": 1}

def test_speed_zones_ignores_nulls():
    r = compute_speed_zones([None, None, 5.0], thresholds=_ZONES)
    assert sum(r.values()) == 1

def test_speed_zones_boundary_walk_jog():
    # exactly 7 is jog (walk is strictly < 7)
    r = compute_speed_zones([7.0], thresholds=_ZONES)
    assert r["walk"] == 0
    assert r["jog"] == 1


# ── compute_high_intensity_pct ────────────────────────────────────────────────

def test_hi_pct_basic():
    # active=[10,15,20], hi(>=15)=2 → 66.67 %
    r = compute_high_intensity_pct([10.0, 15.0, 20.0, None], hi_threshold_kmh=15.0)
    assert r == pytest.approx(66.67, abs=0.01)

def test_hi_pct_all_null():
    assert compute_high_intensity_pct([None, None], hi_threshold_kmh=15.0) is None

def test_hi_pct_zero():
    assert compute_high_intensity_pct([5.0, 10.0], hi_threshold_kmh=15.0) == pytest.approx(0.0)


# ── compute_accelerations ─────────────────────────────────────────────────────

def test_accelerations_upward_crossing():
    # speeds: 0 → 3.6 → 14.4 km/h  →  accels: 1.0, 3.0 m/s²
    # crossing: accels[0]=1.0 < 3.0, accels[1]=3.0 >= 3.0  → count=1
    r = compute_accelerations([0.0, 3.6, 14.4], threshold_ms2=3.0)
    assert r["high_count"] == 1
    assert r["max_ms2"] == pytest.approx(3.0, abs=0.01)

def test_accelerations_no_crossing():
    # all accels < 3 m/s²
    r = compute_accelerations([10.0, 13.0, 16.0], threshold_ms2=3.0)
    assert r["high_count"] == 0

def test_accelerations_null_resets_chain():
    # null breaks the consecutive pair: no accels computable
    r = compute_accelerations([0.0, None, 14.4], threshold_ms2=3.0)
    assert r["high_count"] == 0
    assert r["max_ms2"] is None

def test_accelerations_empty():
    r = compute_accelerations([], threshold_ms2=3.0)
    assert r["high_count"] == 0
    assert r["max_ms2"] is None


# ── compute_halves ────────────────────────────────────────────────────────────

def test_halves_split_at_midpoint():
    sot = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    r = compute_halves(sot, duration_seconds=6.0)
    assert r["first"]["avg_speed_kmh"] == pytest.approx(10.0)
    assert r["second"]["avg_speed_kmh"] == pytest.approx(20.0)

def test_halves_distance():
    # 2 s at 3.6 km/h = 2 * 1 m = 2.0 m per half
    sot = [3.6, 3.6, 3.6, 3.6]
    r = compute_halves(sot, duration_seconds=4.0)
    assert r["first"]["distance_m"] == pytest.approx(2.0, abs=0.01)

def test_halves_all_null_second_half():
    r = compute_halves([10.0, 10.0, None, None], duration_seconds=4.0)
    assert r["second"]["avg_speed_kmh"] is None
    assert r["second"]["distance_m"] is None


# ── compute_drop_off ──────────────────────────────────────────────────────────

def test_drop_off_negative():
    halves = {
        "first":  {"avg_speed_kmh": 10.0, "distance_m": 10.0, "sprints": 0},
        "second": {"avg_speed_kmh":  8.0, "distance_m":  8.0, "sprints": 0},
    }
    assert compute_drop_off(halves) == pytest.approx(-20.0)

def test_drop_off_positive():
    halves = {
        "first":  {"avg_speed_kmh":  8.0, "distance_m": 8.0, "sprints": 0},
        "second": {"avg_speed_kmh": 10.0, "distance_m": 10.0, "sprints": 0},
    }
    assert compute_drop_off(halves) == pytest.approx(25.0)

def test_drop_off_zero_first_half():
    halves = {
        "first":  {"avg_speed_kmh": 0.0, "distance_m": 0.0, "sprints": 0},
        "second": {"avg_speed_kmh": 5.0, "distance_m": 5.0, "sprints": 0},
    }
    assert compute_drop_off(halves) is None

def test_drop_off_null_half():
    halves = {
        "first":  {"avg_speed_kmh": None, "distance_m": None, "sprints": None},
        "second": {"avg_speed_kmh":  5.0, "distance_m":  5.0, "sprints": 0},
    }
    assert compute_drop_off(halves) is None


# ── compute_peak_window ───────────────────────────────────────────────────────

def test_peak_window_finds_best():
    # window=2; slices: [5,30]=17.5, [30,25]=27.5, [25,5]=15 → best starts at 1
    r = compute_peak_window([5.0, 30.0, 25.0, 5.0], window_sec=2)
    assert r["start_sec"] == 1
    assert r["end_sec"] == 3
    assert r["avg_speed_kmh"] == pytest.approx(27.5)

def test_peak_window_clamps_to_video_length():
    r = compute_peak_window([10.0, 20.0, 30.0], window_sec=100)
    assert r["start_sec"] == 0
    assert r["end_sec"] == 3

def test_peak_window_all_null():
    assert compute_peak_window([None, None], window_sec=2) is None

def test_peak_window_empty():
    assert compute_peak_window([], window_sec=300) is None


# ── compute_position_metrics ──────────────────────────────────────────────────

_PITCH = dict(pitch_length=100.0, pitch_width=64.0, grid_rows=10, grid_cols=10)

def test_position_centroid():
    ph = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
    r = compute_position_metrics(ph, **_PITCH)
    assert r["centroid_m"] == pytest.approx([10.0, 10.0], abs=0.01)

def test_position_lateral_left():
    # y=10 < 64/3 ≈ 21.33
    r = compute_position_metrics([[50.0, 10.0]] * 3, **_PITCH)
    assert r["lateral_band"] == "left"

def test_position_lateral_right():
    # y=50 > 64*2/3 ≈ 42.67
    r = compute_position_metrics([[50.0, 50.0]] * 3, **_PITCH)
    assert r["lateral_band"] == "right"

def test_position_lateral_center():
    # y=32 (middle)
    r = compute_position_metrics([[50.0, 32.0]] * 3, **_PITCH)
    assert r["lateral_band"] == "center"

def test_position_full_coverage():
    # One point per cell centre → all 100 cells touched
    ph = [[(r * 10.0 + 5.0), (c * 6.4 + 3.2)] for r in range(10) for c in range(10)]
    r = compute_position_metrics(ph, **_PITCH)
    assert r["coverage_pct"] == pytest.approx(100.0)

def test_position_insufficient_data():
    r = compute_position_metrics([None, None], **_PITCH)
    assert r["centroid_m"] is None

def test_position_single_point_returns_null():
    # needs >= 2 valid points
    r = compute_position_metrics([[50.0, 32.0]], **_PITCH)
    assert r["centroid_m"] is None


# ── compute_player_derived (integration) ─────────────────────────────────────

_REQUIRED_KEYS = {
    "active_time_sec", "missing_data_pct", "sprints", "speed_zones_sec",
    "high_intensity_pct", "accelerations", "halves", "drop_off_pct",
    "peak_window", "position",
}


def test_player_derived_all_keys():
    p = {
        "speed_over_time":  [5.0, 10.0, 22.0, 22.0, None, 8.0],
        "position_history": [[50.0, 32.0], None, [51.0, 33.0], [52.0, 30.0]],
    }
    assert set(compute_player_derived(p, 6.0).keys()) == _REQUIRED_KEYS


def test_player_derived_all_null_no_crash():
    p = {"speed_over_time": [None] * 3, "position_history": [None] * 3}
    r = compute_player_derived(p, 3.0)
    assert r["active_time_sec"] == 0.0
    assert r["high_intensity_pct"] is None
    assert r["peak_window"] is None


def test_player_derived_empty_no_crash():
    p = {"speed_over_time": [], "position_history": []}
    r = compute_player_derived(p, 0.0)
    assert set(r.keys()) == _REQUIRED_KEYS


# ── schema shape tests ────────────────────────────────────────────────────────

def _build_export(n_players: int = 4, seconds: int = 20) -> dict:
    """Synthetic v2 player export for schema validation."""
    players = {}
    for i in range(1, n_players + 1):
        team = 1 if i <= n_players // 2 else 2
        sot = [float(5 + i) if j % 3 != 0 else None for j in range(seconds)]
        ph = [[float(i * 10 % 90), 32.0] if j % 4 != 0 else None for j in range(seconds * 24)]
        pdata = {
            "team": team,
            "max_speed_kmh": float(20 + i),
            "total_distance_m": float(100 + i * 10),
            "max_acceleration_ms2": 2.5,
            "speed_over_time": sot,
            "position_history": ph,
        }
        pdata["derived"] = compute_player_derived(pdata, float(seconds))
        players[str(i)] = pdata
    return players


def test_team_stats_top_level_keys():
    players = _build_export()
    ts = compute_team_stats(players)
    for team_key in ("1", "2"):
        assert team_key in ts
        assert ts[team_key] is not None
        assert set(ts[team_key].keys()) == {"players_count", "percentiles", "leaders"}


def test_team_stats_percentile_metrics():
    ts = compute_team_stats(_build_export())
    expected_metrics = {
        "total_distance_m", "max_speed_kmh", "sprints_count",
        "high_intensity_pct", "drop_off_pct", "active_time_sec",
    }
    for team_key in ("1", "2"):
        pct = ts[team_key]["percentiles"]
        assert set(pct.keys()) == expected_metrics
        for metric in expected_metrics:
            for pn in ("p5", "p25", "p50", "p75", "p95"):
                assert pn in pct[metric], f"Missing {pn} in {metric} for team {team_key}"


def test_team_stats_leader_keys():
    ts = compute_team_stats(_build_export())
    for team_key in ("1", "2"):
        leaders = ts[team_key]["leaders"]
        for field in ("total_distance_m", "max_speed_kmh", "sprints_count"):
            assert field in leaders
            assert "player_id" in leaders[field]
            assert "value" in leaders[field]


def test_derived_sub_keys():
    p = {
        "speed_over_time":  [5.0, 22.0, 22.0, 8.0, None, 10.0, 5.0, 9.0, 20.0, 15.0],
        "position_history": [[50.0, 32.0]] * 30,
    }
    d = compute_player_derived(p, 10.0)

    assert set(d["sprints"].keys())       == {"count", "longest_sec", "total_distance_m"}
    assert set(d["speed_zones_sec"].keys()) == {"walk", "jog", "run", "sprint"}
    assert set(d["accelerations"].keys()) == {"high_count", "max_ms2"}
    assert set(d["halves"].keys())        == {"first", "second"}
    for half in ("first", "second"):
        assert set(d["halves"][half].keys()) == {"distance_m", "avg_speed_kmh", "sprints"}
    assert set(d["position"].keys())      == {
        "centroid_m", "dispersion_m", "coverage_pct", "dominant_cell", "lateral_band"
    }
    if d["peak_window"] is not None:
        assert set(d["peak_window"].keys()) == {"start_sec", "end_sec", "avg_speed_kmh"}
