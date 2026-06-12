import type { Player, SpeedZones, SpeedZonesSec, TeamStats } from "@/types/match";

export const SPRINT_KMH = 21;
export const RUN_KMH = 15;
export const JOG_KMH = 7;

/** Count sprint events: each transition from sub-sprint into sprint zone = 1 sprint */
export function countSprints(speeds: Array<number | null>): number {
  let sprints = 0;
  let inSprint = false;
  for (const v of speeds) {
    if (v === null) continue;
    if (!inSprint && v >= SPRINT_KMH) {
      sprints++;
      inSprint = true;
    } else if (inSprint && v < SPRINT_KMH) {
      inSprint = false;
    }
  }
  return sprints;
}

/** Distribution of non-null samples across walk / jog / run / sprint zones */
export function computeSpeedZones(speeds: Array<number | null>): SpeedZones {
  let walk = 0, jog = 0, run = 0, sprint = 0, total = 0;
  for (const v of speeds) {
    if (v === null) continue;
    total++;
    if (v < JOG_KMH) walk++;
    else if (v < RUN_KMH) jog++;
    else if (v < SPRINT_KMH) run++;
    else sprint++;
  }
  if (total === 0) return { walk_pct: 0, jog_pct: 0, run_pct: 0, sprint_pct: 0 };
  return {
    walk_pct: (walk / total) * 100,
    jog_pct: (jog / total) * 100,
    run_pct: (run / total) * 100,
    sprint_pct: (sprint / total) * 100,
  };
}

/** Percentage of time in high-intensity zones (run + sprint) */
export function computeHighIntensityPct(speeds: Array<number | null>): number {
  const z = computeSpeedZones(speeds);
  return z.run_pct + z.sprint_pct;
}

/** Convert per-zone second counts (schema v2 `speed_zones_sec`) into percentages */
export function zonePctsFromCounts(z: SpeedZonesSec): SpeedZones {
  const total = z.walk + z.jog + z.run + z.sprint;
  if (total === 0) return { walk_pct: 0, jog_pct: 0, run_pct: 0, sprint_pct: 0 };
  return {
    walk_pct: (z.walk / total) * 100,
    jog_pct: (z.jog / total) * 100,
    run_pct: (z.run / total) * 100,
    sprint_pct: (z.sprint / total) * 100,
  };
}

/**
 * Performance drop-off from first to second half.
 * Positive = intensity declined; negative = intensity increased second half.
 */
export function computeDropOff(firstHalfHIPct: number, secondHalfHIPct: number): number {
  if (firstHalfHIPct === 0) return 0;
  return ((firstHalfHIPct - secondHalfHIPct) / firstHalfHIPct) * 100;
}

/** Dominant field zone based on average Y coordinate of position history */
export function dominantZone(positions: Array<[number, number] | null>): string {
  let sy = 0, n = 0;
  for (const p of positions) {
    if (!p) continue;
    sy += p[1];
    n++;
  }
  if (n === 0) return "sin datos posicionales";
  const cy = sy / n;
  if (cy < 64 * 0.33) return "banda izquierda";
  if (cy > 64 * 0.66) return "banda derecha";
  return "zona central";
}

/** Slice speed_over_time to the minutes window [rangeMin[0], rangeMin[1]] */
export function sliceSpeedByRange(
  speeds: Array<number | null>,
  rangeMin: [number, number],
  totalMin: number,
): Array<number | null> {
  if (totalMin <= 0) return speeds;
  const n = speeds.length;
  const start = Math.floor((rangeMin[0] / totalMin) * n);
  const end = Math.ceil((rangeMin[1] / totalMin) * n);
  return speeds.slice(start, end);
}

export function isFullMatch(range: [number, number], totalMin: number): boolean {
  return range[0] === 0 && Math.abs(range[1] - totalMin) <= 1;
}

export function isFirstHalf(range: [number, number], totalMin: number): boolean {
  return range[0] === 0 && Math.abs(range[1] - totalMin / 2) <= 1;
}

export function isSecondHalf(range: [number, number], totalMin: number): boolean {
  return Math.abs(range[0] - totalMin / 2) <= 1 && Math.abs(range[1] - totalMin) <= 1;
}

export interface InsightCandidate {
  key: string;
  severity: "high" | "low";
  message: string;
}

function toSeverity(percentile: number | null): "high" | "low" {
  if (percentile === null) return "low";
  return percentile >= 80 || percentile <= 20 ? "high" : "low";
}

/**
 * Build the full list of insight candidates for a player.
 * Uses precomputed `derived` fields when the active range matches a known period
 * (full match / first half / second half); otherwise recalculates from raw data.
 * Insights that depend on peak_window or drop_off are suppressed when
 * missing_data_pct > 30 to avoid misleading conclusions.
 */
export function buildInsightCandidates(
  player: Player,
  teamStats: TeamStats | null,
  rangeMin: [number, number],
  matchDurationMin: number,
): InsightCandidate[] {
  const d = player.derived;
  const candidates: InsightCandidate[] = [];
  const teamLabel = `Equipo ${player.team}`;
  const dataLimited = (d?.missing_data_pct ?? 0) > 30;

  const usingPrecomputed =
    d != null &&
    (isFullMatch(rangeMin, matchDurationMin) ||
      isFirstHalf(rangeMin, matchDurationMin) ||
      isSecondHalf(rangeMin, matchDurationMin));

  let sprints: number;
  let hiPct: number;
  let zones: SpeedZones;
  let dropOff: number | null = null;

  if (usingPrecomputed && d) {
    const sliced = sliceSpeedByRange(player.speed_over_time, rangeMin, matchDurationMin);
    if (isFirstHalf(rangeMin, matchDurationMin) && d.halves) {
      sprints = d.halves.first.sprints ?? countSprints(sliced);
      hiPct = d.halves.first.high_intensity_pct ?? computeHighIntensityPct(sliced);
      zones = computeSpeedZones(sliced);
    } else if (isSecondHalf(rangeMin, matchDurationMin) && d.halves) {
      sprints = d.halves.second.sprints ?? countSprints(sliced);
      hiPct = d.halves.second.high_intensity_pct ?? computeHighIntensityPct(sliced);
      zones = computeSpeedZones(sliced);
    } else {
      sprints = d.sprints.count;
      hiPct = d.high_intensity_pct ?? computeHighIntensityPct(player.speed_over_time);
      zones = zonePctsFromCounts(d.speed_zones_sec);
      dropOff = d.drop_off_pct;
    }
  } else {
    const sliced = sliceSpeedByRange(player.speed_over_time, rangeMin, matchDurationMin);
    sprints = countSprints(sliced);
    zones = computeSpeedZones(sliced);
    hiPct = zones.run_pct + zones.sprint_pct;
  }

  const pSprints = d?.percentiles?.sprints ?? null;
  const pHI = d?.percentiles?.high_intensity_pct ?? null;
  const pSpeed = d?.percentiles?.max_speed_kmh ?? null;

  // --- Sprint candidate ---
  if (teamStats?.leaders?.sprints_count?.player_id === player.id) {
    candidates.push({
      key: "sprints_leader",
      severity: "high",
      message: `Top sprinter del ${teamLabel}: ${sprints} sprints${pSprints !== null ? `, p${Math.round(pSprints)} del plantel` : ""}.`,
    });
  } else if (pSprints !== null && pSprints <= 20) {
    candidates.push({
      key: "sprints_low",
      severity: "high",
      message: `Solo ${sprints} sprints, p${Math.round(pSprints)} del ${teamLabel}.`,
    });
  } else {
    candidates.push({
      key: "sprints",
      severity: toSeverity(pSprints),
      message: `${sprints} sprints en el período.`,
    });
  }

  // --- High-intensity candidate ---
  if (pHI !== null && pHI >= 80) {
    candidates.push({
      key: "hi_top",
      severity: "high",
      message: `Alta intensidad: ${hiPct.toFixed(0)}% del tiempo, top del ${teamLabel} (p${Math.round(pHI)}).`,
    });
  } else if (pHI !== null && pHI <= 20) {
    candidates.push({
      key: "hi_low",
      severity: "high",
      message: `Baja intensidad: ${hiPct.toFixed(0)}% en carrera/sprint (p${Math.round(pHI)} del ${teamLabel}).`,
    });
  } else {
    candidates.push({
      key: "hi",
      severity: "low",
      message: `${hiPct.toFixed(0)}% del tiempo en alta intensidad.`,
    });
  }

  // --- Drop-off candidate (full match, not limited data) ---
  if (dropOff !== null && !dataLimited && Math.abs(dropOff) >= 15) {
    candidates.push({
      key: "drop_off",
      severity: "high",
      message:
        dropOff > 0
          ? `Bajó un ${dropOff.toFixed(0)}% de intensidad en la segunda mitad.`
          : `Intensificó su juego en la segunda mitad (+${Math.abs(dropOff).toFixed(0)}%).`,
    });
  }

  // --- Peak window candidate (full match, not limited data) ---
  if (usingPrecomputed && isFullMatch(rangeMin, matchDurationMin) && d?.peak_window && !dataLimited) {
    const pw = d.peak_window;
    candidates.push({
      key: "peak_window",
      severity: "low",
      message: `Mejor ventana: min ${Math.round(pw.start_sec / 60)}–${Math.round(pw.end_sec / 60)} (${pw.avg_speed_kmh.toFixed(1)} km/h prom).`,
    });
  }

  // --- Speed percentile (contextual, not repeating the raw number) ---
  if (pSpeed !== null && pSpeed >= 80) {
    candidates.push({
      key: "speed_top",
      severity: "high",
      message: `Uno de los más rápidos del ${teamLabel} (p${Math.round(pSpeed)} en vel. máxima).`,
    });
  } else if (pSpeed !== null && pSpeed <= 20) {
    candidates.push({
      key: "speed_low",
      severity: "high",
      message: `Velocidad máxima en p${Math.round(pSpeed)} del ${teamLabel}.`,
    });
  }

  // --- Zone candidate (always available as low-severity fallback) ---
  const n = player.position_history.length;
  const posStart = matchDurationMin > 0 ? Math.floor((rangeMin[0] / matchDurationMin) * n) : 0;
  const posEnd = matchDurationMin > 0 ? Math.ceil((rangeMin[1] / matchDurationMin) * n) : n;
  const zone = dominantZone(player.position_history.slice(posStart, posEnd));
  candidates.push({
    key: "zone",
    severity: "low",
    message: `Actividad concentrada en ${zone}.`,
  });

  return candidates;
}

/**
 * Pick the top n insights, always preferring high-severity over low-severity.
 * Low-severity candidates fill the remaining slots when there aren't enough high ones.
 */
export function selectTopInsights(candidates: InsightCandidate[], n = 4): InsightCandidate[] {
  const high = candidates.filter((c) => c.severity === "high");
  if (high.length >= n) return high.slice(0, n);
  const low = candidates.filter((c) => c.severity === "low");
  return [...high, ...low].slice(0, n);
}
