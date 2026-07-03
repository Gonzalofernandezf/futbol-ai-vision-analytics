import { computeSpeedZones, computeHighIntensityPct, countSprints, computeDropOff } from "@/lib/insights";
import type { MatchData, PlayerStats, SetPieceEvent, TeamId, TeamStats } from "@/types/match";

/** Mulberry32 PRNG — deterministic so the demo looks the same on every build. */
function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-9);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const DURATION_SECONDS = 5700; // 95' partido con tiempo añadido
const FPS = 25;
const PITCH_W = 100;
const PITCH_H = 64;

interface RoleSpec {
  role: string;
  x: number;
  y: number;
  sprintiness: number; // 0..1, más alto = más sprints/velocidad
  roam: number; // metros de amplitud de movimiento respecto a la posición base
}

const FORMATION: RoleSpec[] = [
  { role: "GK", x: 5, y: 32, sprintiness: 0.05, roam: 5 },
  { role: "CB", x: 18, y: 22, sprintiness: 0.25, roam: 10 },
  { role: "CB", x: 18, y: 42, sprintiness: 0.25, roam: 10 },
  { role: "LB", x: 26, y: 8, sprintiness: 0.55, roam: 18 },
  { role: "RB", x: 26, y: 56, sprintiness: 0.55, roam: 18 },
  { role: "CDM", x: 36, y: 32, sprintiness: 0.35, roam: 14 },
  { role: "CM", x: 48, y: 20, sprintiness: 0.5, roam: 16 },
  { role: "CAM", x: 52, y: 44, sprintiness: 0.6, roam: 18 },
  { role: "LW", x: 68, y: 10, sprintiness: 0.8, roam: 20 },
  { role: "RW", x: 68, y: 54, sprintiness: 0.8, roam: 20 },
  { role: "ST", x: 78, y: 32, sprintiness: 0.7, roam: 16 },
];

function generateSpeedSeries(rand: () => number, spec: RoleSpec): Array<number | null> {
  const speeds: Array<number | null> = [];
  let sprintCooldown = 0;
  for (let t = 0; t < DURATION_SECONDS; t++) {
    const restCycle = 0.5 + 0.5 * Math.sin(t / 140 + spec.x);
    const halftimeDip = t > DURATION_SECONDS / 2 - 60 && t < DURATION_SECONDS / 2 + 90 ? 0.2 : 1;
    const fatigue = 1 - 0.15 * (t / DURATION_SECONDS) * (1 - spec.sprintiness * 0.4);
    let base = (2 + restCycle * 6 * spec.sprintiness + 2 * (1 - spec.sprintiness)) * halftimeDip * fatigue;

    if (sprintCooldown > 0) {
      sprintCooldown--;
      base = 22 + rand() * 10 * spec.sprintiness + 4;
    } else if (rand() < 0.004 * (0.3 + spec.sprintiness) * halftimeDip) {
      sprintCooldown = 2 + Math.floor(rand() * 3);
    }

    const noisy = Math.max(0, base + gaussian(rand) * 1.2);
    const missing = rand() < 0.03 || (t > 2400 && t < 2450); // huecos de detección + un corte
    speeds.push(missing ? null : Math.min(34, Number(noisy.toFixed(2))));
  }
  return speeds;
}

function generatePositionSeries(
  rand: () => number,
  spec: RoleSpec,
  speeds: Array<number | null>,
): Array<[number, number] | null> {
  const positions: Array<[number, number] | null> = [];
  let x = spec.x;
  let y = spec.y;
  for (let t = 0; t < DURATION_SECONDS; t++) {
    x += (spec.x - x) * 0.04 + gaussian(rand) * (spec.roam / 12);
    y += (spec.y - y) * 0.04 + gaussian(rand) * (spec.roam / 12);
    x = Math.min(PITCH_W - 1, Math.max(1, x));
    y = Math.min(PITCH_H - 1, Math.max(1, y));
    positions.push(speeds[t] === null ? null : [Number(x.toFixed(2)), Number(y.toFixed(2))]);
  }
  return positions;
}

function computeDerived(speeds: Array<number | null>) {
  const half = Math.floor(speeds.length / 2);
  const firstHalf = speeds.slice(0, half);
  const secondHalf = speeds.slice(half);

  const firstZones = computeSpeedZones(firstHalf);
  const secondZones = computeSpeedZones(secondHalf);
  const firstHI = firstZones.run_pct + firstZones.sprint_pct;
  const secondHI = secondZones.run_pct + secondZones.sprint_pct;

  const distance = (arr: Array<number | null>) =>
    arr.reduce((sum, v) => sum + (v ?? 0) / 3.6, 0);

  const zones = computeSpeedZones(speeds);
  const highIntensityPct = computeHighIntensityPct(speeds);
  const sprints = countSprints(speeds);
  const missing = speeds.filter((v) => v === null).length;

  let bestWindowStart = 0;
  let bestWindowAvg = -1;
  const windowSec = 300;
  for (let start = 0; start + windowSec <= speeds.length; start += 60) {
    const slice = speeds.slice(start, start + windowSec).filter((v): v is number => v !== null);
    if (slice.length === 0) continue;
    const avg = slice.reduce((s, v) => s + v, 0) / slice.length;
    if (avg > bestWindowAvg) {
      bestWindowAvg = avg;
      bestWindowStart = start;
    }
  }

  return {
    sprints,
    speed_zones: zones,
    high_intensity_pct: highIntensityPct,
    halves: {
      first: { distance_m: distance(firstHalf), sprints: countSprints(firstHalf), high_intensity_pct: firstHI },
      second: { distance_m: distance(secondHalf), sprints: countSprints(secondHalf), high_intensity_pct: secondHI },
    },
    drop_off_pct: computeDropOff(firstHI, secondHI),
    peak_window: {
      start_min: Math.floor(bestWindowStart / 60),
      end_min: Math.floor((bestWindowStart + windowSec) / 60),
      avg_speed: Number((bestWindowAvg < 0 ? 0 : bestWindowAvg).toFixed(2)),
    },
    missing_data_pct: Number(((missing / speeds.length) * 100).toFixed(2)),
    totalDistance: distance(speeds),
  };
}

function percentileRank(values: number[], value: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const below = sorted.filter((v) => v < value).length;
  return Math.round((below / (sorted.length - 1 || 1)) * 100);
}

function buildTeam(rand: () => number, team: TeamId): Map<string, PlayerStats> {
  const players = new Map<string, PlayerStats>();
  const offset = team === 1 ? 0 : 11;
  FORMATION.forEach((spec, idx) => {
    const id = String(offset + idx + 1);
    const mirroredSpec: RoleSpec =
      team === 1 ? spec : { ...spec, x: PITCH_W - spec.x, y: PITCH_H - spec.y };
    const speeds = generateSpeedSeries(rand, mirroredSpec);
    const positions = generatePositionSeries(rand, mirroredSpec, speeds);
    const derived = computeDerived(speeds);
    const maxSpeed = Math.max(0, ...speeds.filter((v): v is number => v !== null));
    let maxAccel = 0;
    for (let t = 1; t < speeds.length; t++) {
      const a = speeds[t];
      const b = speeds[t - 1];
      if (a === null || b === null) continue;
      maxAccel = Math.max(maxAccel, Math.abs(a - b) / 3.6);
    }

    players.set(id, {
      team,
      max_speed_kmh: Number(maxSpeed.toFixed(2)),
      total_distance_m: Number(derived.totalDistance.toFixed(1)),
      max_acceleration_ms2: Number(maxAccel.toFixed(2)),
      speed_over_time: speeds,
      position_history: positions,
      derived: {
        sprints: derived.sprints,
        speed_zones: derived.speed_zones,
        high_intensity_pct: derived.high_intensity_pct,
        halves: derived.halves,
        drop_off_pct: derived.drop_off_pct,
        peak_window: derived.peak_window,
        position: mirroredSpec.role,
        percentiles: {}, // se rellena después de conocer a todo el equipo
        missing_data_pct: derived.missing_data_pct,
      },
    });
  });
  return players;
}

function fillPercentilesAndLeaders(
  players: Map<string, PlayerStats>,
  team: TeamId,
): TeamStats {
  const teamPlayers = [...players.entries()].filter(([, p]) => p.team === team);
  const metrics: Array<keyof PlayerStats | "sprints" | "high_intensity_pct"> = [
    "max_speed_kmh",
    "total_distance_m",
    "max_acceleration_ms2",
  ];
  const leaders: Record<string, string> = {};

  for (const metric of metrics) {
    let bestId = teamPlayers[0][0];
    let bestVal = -Infinity;
    for (const [id, p] of teamPlayers) {
      const v = p[metric as "max_speed_kmh"] as number;
      if (v > bestVal) {
        bestVal = v;
        bestId = id;
      }
    }
    leaders[metric] = bestId;
    const values = teamPlayers.map(([, p]) => p[metric as "max_speed_kmh"] as number);
    for (const [id, p] of teamPlayers) {
      p.derived!.percentiles[metric] = percentileRank(values, p[metric as "max_speed_kmh"] as number);
    }
  }

  for (const derivedMetric of ["sprints", "high_intensity_pct"] as const) {
    const values = teamPlayers.map(([, p]) => p.derived![derivedMetric]);
    let bestId = teamPlayers[0][0];
    let bestVal = -Infinity;
    for (const [id, p] of teamPlayers) {
      const v = p.derived![derivedMetric];
      if (v > bestVal) {
        bestVal = v;
        bestId = id;
      }
    }
    leaders[derivedMetric] = bestId;
    for (const [, p] of teamPlayers) {
      p.derived!.percentiles[derivedMetric] = percentileRank(values, p.derived![derivedMetric]);
    }
  }

  return { team, leaders };
}

function generateBall(rand: () => number): { position_history: Array<[number, number] | null>; speed_over_time: Array<number | null> } {
  const positions: Array<[number, number] | null> = [];
  const speeds: Array<number | null> = [];
  let x = PITCH_W / 2;
  let y = PITCH_H / 2;
  for (let t = 0; t < DURATION_SECONDS; t++) {
    const kick = rand() < 0.02;
    const stepX = (kick ? gaussian(rand) * 14 : gaussian(rand) * 3);
    const stepY = (kick ? gaussian(rand) * 10 : gaussian(rand) * 2);
    x = Math.min(PITCH_W, Math.max(0, x + stepX));
    y = Math.min(PITCH_H, Math.max(0, y + stepY));
    const missing = rand() < 0.05;
    positions.push(missing ? null : [Number(x.toFixed(2)), Number(y.toFixed(2))]);
    speeds.push(missing ? null : Number(Math.min(95, Math.abs(stepX) * 3.6 + gaussian(rand) * 4 + 8).toFixed(2)));
  }
  return { position_history: positions, speed_over_time: speeds };
}

function generateSetPieces(rand: () => number): SetPieceEvent[] {
  const types: SetPieceEvent["type"][] = ["corner", "banda", "meta", "falta"];
  const events: SetPieceEvent[] = [];
  let t = 40;
  while (t < DURATION_SECONDS - 30) {
    const type = types[Math.floor(rand() * types.length)];
    const frameStart = Math.floor(t * FPS);
    const frameEnd = frameStart + FPS * 3;
    let position: [number, number];
    if (type === "corner") position = [rand() < 0.5 ? 1 : 99, rand() < 0.5 ? 1 : 63];
    else if (type === "meta") position = [rand() < 0.5 ? 4 : 96, 20 + rand() * 24];
    else if (type === "banda") position = [rand() * 100, rand() < 0.5 ? 1 : 63];
    else position = [10 + rand() * 80, rand() * 64];

    events.push({
      type,
      frame_start: frameStart,
      frame_end: frameEnd,
      start_sec: Number(t.toFixed(1)),
      end_sec: Number((t + 3).toFixed(1)),
      position_m: [Number(position[0].toFixed(1)), Number(position[1].toFixed(1))],
    });
    t += 60 + rand() * 240;
  }
  return events;
}

export function generateMockMatchData(): MatchData {
  const rand = mulberry32(20260703);

  const team1Players = buildTeam(rand, 1);
  const team2Players = buildTeam(rand, 2);
  const allPlayers = new Map([...team1Players, ...team2Players]);

  const team1Stats = fillPercentilesAndLeaders(allPlayers, 1);
  const team2Stats = fillPercentilesAndLeaders(allPlayers, 2);

  const players: MatchData["players"] = {};
  for (const [id, p] of allPlayers) players[id] = p;

  return {
    schema_version: 2,
    match_meta: {
      duration_seconds: DURATION_SECONDS,
      fps: FPS,
      home_possession: 54.2,
      away_possession: 45.8,
    },
    players,
    team_stats: {
      "1": team1Stats,
      "2": team2Stats,
    },
    ball: generateBall(rand),
    set_pieces: generateSetPieces(rand),
  };
}
