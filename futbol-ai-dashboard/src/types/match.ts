export interface MatchMeta {
  duration_seconds: number;
  fps: number;
  home_possession: number;
  away_possession: number;
}

export type TeamId = 1 | 2;

export interface SpeedZones {
  walk_pct: number;
  jog_pct: number;
  run_pct: number;
  sprint_pct: number;
}

export interface HalfDerived {
  distance_m: number;
  sprints: number;
  high_intensity_pct: number | null;
}

export interface PlayerDerived {
  sprints: number;
  speed_zones: SpeedZones;
  high_intensity_pct: number | null;
  halves: {
    first: HalfDerived;
    second: HalfDerived;
  };
  /** Percentage drop in high-intensity from first to second half (positive = decline) */
  drop_off_pct: number;
  peak_window: {
    start_min: number;
    end_min: number;
    avg_speed: number;
  };
  position: string;
  /** Intra-team percentile (0–100) per metric key */
  percentiles: Record<string, number>;
  /** Fraction of match_seconds with no detection (0–100) */
  missing_data_pct: number;
}

export interface TeamStats {
  team: TeamId;
  /** Player ID of the top performer per metric */
  leaders: Record<string, string>;
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
  /** Team-level stats keyed by team ID string — present only in schema v2 */
  team_stats?: Record<string, TeamStats>;
}

/** Fully-typed v2 match document — all derived fields are guaranteed present */
export interface MatchAnalyticsV2 extends Omit<MatchData, "schema_version" | "team_stats"> {
  schema_version: 2;
  team_stats: Record<string, TeamStats>;
}

export interface Player extends PlayerStats {
  id: string;
}
