import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { Player } from "@/types/match";

interface Props {
  player: Player;
  teamPlayers: Player[];
}

function pctRank(value: number, sorted: number[]): number {
  if (sorted.length === 0) return 0;
  let i = 0;
  while (i < sorted.length && sorted[i] <= value) i += 1;
  return (i / sorted.length) * 100;
}

function activeRatio(speed: Array<number | null>): number {
  const total = speed.length || 1;
  const active = speed.reduce<number>((s, v) => s + (v != null ? 1 : 0), 0);
  return active / total;
}

function avgSpeed(speed: Array<number | null>): number {
  let sum = 0;
  let n = 0;
  for (const v of speed) {
    if (v != null) {
      sum += v;
      n += 1;
    }
  }
  return n > 0 ? sum / n : 0;
}

export function PlayerRadar({ player, teamPlayers }: Props) {
  // Axes normalised to percentile rank within the player's team (0-100).
  // Using percentiles avoids hardcoded thresholds and adapts to each match.
  const team = teamPlayers.filter((p) => p.team === player.team);

  const distances = team.map((p) => p.total_distance_m).sort((a, b) => a - b);
  const maxSpeeds = team.map((p) => p.max_speed_kmh).sort((a, b) => a - b);
  const maxAccels = team.map((p) => p.max_acceleration_ms2).sort((a, b) => a - b);
  const avgSpeeds = team.map((p) => avgSpeed(p.speed_over_time)).sort((a, b) => a - b);
  const actives = team.map((p) => activeRatio(p.speed_over_time)).sort((a, b) => a - b);

  const data = [
    {
      axis: "Distancia",
      v: pctRank(player.total_distance_m, distances),
    },
    {
      axis: "Vel. máx",
      v: pctRank(player.max_speed_kmh, maxSpeeds),
    },
    {
      axis: "Vel. media",
      v: pctRank(avgSpeed(player.speed_over_time), avgSpeeds),
    },
    {
      axis: "Aceleración",
      v: pctRank(player.max_acceleration_ms2, maxAccels),
    },
    {
      axis: "Presencia",
      v: pctRank(activeRatio(player.speed_over_time), actives),
    },
  ];

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 24 }}>
          <PolarGrid stroke="rgba(255,255,255,0.12)" />
          <PolarAngleAxis dataKey="axis" tick={{ fontSize: 10, fill: "rgba(255,255,255,0.7)" }} />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fontSize: 9, fill: "rgba(255,255,255,0.4)" }}
            tickCount={5}
          />
          <Radar
            dataKey="v"
            stroke="var(--primary)"
            fill="var(--primary)"
            fillOpacity={0.35}
            isAnimationActive={false}
          />
          <Tooltip
            contentStyle={{
              background: "var(--popover)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(v) => {
              const num = typeof v === "number" ? v : 0;
              return [`${num.toFixed(0)} pctl`, "Equipo"];
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
