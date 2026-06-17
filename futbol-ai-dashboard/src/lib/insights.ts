import type { Player, TeamStats } from "@/types/match";

export interface InsightCandidate {
  key: string;
  message: string;
  severity: "high" | "low";
}

export function dominantZone(
  positionHistory: Array<[number, number] | null>,
): string {
  const valid = positionHistory.filter(
    (p): p is [number, number] => p !== null,
  );
  if (valid.length === 0) return "zona desconocida";

  const avgX = valid.reduce((s, p) => s + p[0], 0) / valid.length;
  const avgY = valid.reduce((s, p) => s + p[1], 0) / valid.length;

  const xZone = avgX < 33 ? "defensiva" : avgX < 66 ? "central" : "ofensiva";
  const yZone = avgY < 21 ? "banda izquierda" : avgY > 43 ? "banda derecha" : "centro";

  if (xZone === "central" && yZone === "centro") return "zona central";
  if (yZone === "centro") return `zona ${xZone}`;
  return `${yZone} ${xZone}`;
}

export function buildInsightCandidates(
  player: Player,
  teamStats: TeamStats | null,
  timeRange: [number, number],
  matchDurationMin: number,
): InsightCandidate[] {
  const d = player.derived;
  if (!d) return [];

  const candidates: InsightCandidate[] = [];

  candidates.push({
    key: "distance",
    message: `Recorrió ${(player.total_distance_m / 1000).toFixed(1)} km en total.`,
    severity: "low",
  });

  candidates.push({
    key: "top_speed",
    message: `Pico de velocidad: ${player.max_speed_kmh.toFixed(1)} km/h.`,
    severity: "low",
  });

  candidates.push({
    key: "zone",
    message: `Zona predominante: ${dominantZone(player.position_history)}.`,
    severity: "low",
  });

  if (d.sprints > 0) {
    candidates.push({
      key: "sprints",
      message: `Realizó ${d.sprints} sprint${d.sprints !== 1 ? "s" : ""} durante el partido.`,
      severity: d.sprints >= 5 ? "high" : "low",
    });
  }

  if (d.high_intensity_pct > 15) {
    candidates.push({
      key: "high_intensity",
      message: `${d.high_intensity_pct.toFixed(0)}% del tiempo en alta intensidad.`,
      severity: "high",
    });
  }

  if (d.drop_off_pct > 20) {
    candidates.push({
      key: "drop_off",
      message: `Caída de intensidad del ${d.drop_off_pct.toFixed(0)}% en el 2do tiempo.`,
      severity: "high",
    });
  }

  if (d.peak_window) {
    candidates.push({
      key: "peak",
      message: `Pico de actividad entre min ${d.peak_window.start_min}–${d.peak_window.end_min} (${d.peak_window.avg_speed.toFixed(1)} km/h prom).`,
      severity: "low",
    });
  }

  if (d.speed_zones) {
    if (d.speed_zones.sprint_pct > 5) {
      candidates.push({
        key: "sprint_pct",
        message: `Pasó ${d.speed_zones.sprint_pct.toFixed(0)}% del tiempo en sprint.`,
        severity: "high",
      });
    }
    if (d.speed_zones.walk_pct > 60) {
      candidates.push({
        key: "walk_pct",
        message: `Caminó el ${d.speed_zones.walk_pct.toFixed(0)}% del partido.`,
        severity: "low",
      });
    }
  }

  const pctDist = d.percentiles?.total_distance_m;
  if (pctDist !== undefined && pctDist >= 90) {
    candidates.push({
      key: "pctl_dist",
      message: `Top ${(100 - pctDist).toFixed(0)}% en distancia recorrida de su equipo.`,
      severity: "high",
    });
  }

  const pctSpeed = d.percentiles?.max_speed_kmh;
  if (pctSpeed !== undefined && pctSpeed >= 90) {
    candidates.push({
      key: "pctl_speed",
      message: `Top ${(100 - pctSpeed).toFixed(0)}% en velocidad máxima de su equipo.`,
      severity: "high",
    });
  }

  return candidates;
}

export function selectTopInsights(
  candidates: InsightCandidate[],
): InsightCandidate[] {
  const high = candidates.filter((c) => c.severity === "high");
  const low = candidates.filter((c) => c.severity === "low");
  const picked = [...high, ...low];
  return picked.slice(0, 4);
}
