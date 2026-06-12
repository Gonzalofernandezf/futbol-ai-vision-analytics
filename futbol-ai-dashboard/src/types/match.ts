export interface MatchMeta {
  duration_seconds: number;
  fps: number;
  home_possession: number;
  away_possession: number;
}

export type TeamId = 1 | 2;

/**
 * Percentage distribution across speed zones — computed client-side from
 * raw speeds or from `SpeedZonesSec` counts (see lib/insights.ts).
 */
export interface SpeedZones {
  walk_pct: number;
  jog_pct: number;
  run_pct: number;
  sprint_pct: number;
}

/** Seconds spent in each speed zone, as exported by analytics (`speed_zones_sec`) */
export interface SpeedZonesSec {
  walk: number;
  jog: number;
  run: number;
  sprint: number;
}

export interface SprintStats {
  count: number;
  longest_sec: number;
  total_distance_m: number;
}

export interface AccelerationStats {
  high_count: number;
  max_ms2: number | null;
}

export interface HalfDerived {
  distance_m: number | null;
  avg_speed_kmh: number | null;
  sprints: number | null;
  high_intensity_pct: number | null;
}

export interface PeakWindow {
  start_sec: number;
  end_sec: number;
  avg_speed_kmh: number;
}

export interface PositionMetrics {
  /** [x, y] mean position in meters */
  centroid_m: [number, number] | null;
  dispersion_m: number | null;
  coverage_pct: number | null;
  /** [row, col] of the most-occupied grid cell */
  dominant_cell: [number, number] | null;
  lateral_band: "left" | "center" | "right" | null;
}

/**
 * Precomputed analytics per player — mirrors the output of
 * analytics/match_analytics.py::compute_player_derived (the Python side is
 * the source of truth for this contract).
 */
export interface PlayerDerived {
  active_time_sec: number;
  /** Percentage of match seconds with no detection (0–100) */
  missing_data_pct: number | null;
  sprints: SprintStats;
  speed_zones_sec: SpeedZonesSec;
  high_intensity_pct: number | null;
  accelerations: AccelerationStats;
  halves: {
    first: HalfDerived;
    second: HalfDerived;
  };
  /** Drop in avg speed from first to second half (positive = decline) */
  drop_off_pct: number | null;
  peak_window: PeakWindow | null;
  position: PositionMetrics;
  /**
   * Intra-team percentile rank (0–100) per metric.
   * Keys: "sprints", "high_intensity_pct", "max_speed_kmh".
   */
  percentiles?: Record<string, number>;
}

export interface TeamLeader {
  player_id: string | null;
  value: number | null;
}

export interface TeamStats {
  players_count: number;
  /** p5/p25/p50/p75/p95 per metric, keyed by metric name */
  percentiles: Record<string, Record<string, number | null>>;
  /** Top performer per metric: "total_distance_m", "max_speed_kmh", "sprints_count" */
  leaders: Record<string, TeamLeader>;
}

export interface PlayerStats {
  team: TeamId;
  max_speed_kmh: number;
  total_distance_m: number;
  max_acceleration_ms2: number;
  /** One value per second of the match. May contain nulls when the player was not detected. */
  speed_over_time: Array<number | null>;
  /** [x, y] in meters on a 100m x 64m pitch. May contain nulls. */
  position_history: Array<[number, number] | null>;
  /** Precomputed analytics — present only in schema v2 JSON */
  derived?: PlayerDerived;
}

export interface MatchData {
  match_meta: MatchMeta;
  players: Record<string, PlayerStats>;
  /** Absent in v1 JSON; set to 2 for schema v2 */
  schema_version?: number;
  /** Team-level stats keyed by team ID string — present only in schema v2. Null for empty teams. */
  team_stats?: Record<string, TeamStats | null>;
}

/** Fully-typed v2 match document — all derived fields are guaranteed present */
export interface MatchAnalyticsV2 extends Omit<MatchData, "schema_version" | "team_stats"> {
  schema_version: 2;
  team_stats: Record<string, TeamStats | null>;
}

export interface Player extends PlayerStats {
  id: string;
}
