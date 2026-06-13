import type { Player, TeamStats } from "@/types/match";
import { Gauge, Route as RouteIcon, Zap, User } from "lucide-react";
import { SpeedChart } from "./SpeedChart";
import { InsightsList } from "./InsightsList";
import { useTeam } from "@/hooks/useTeam";
import { daysUntil, getInitials } from "@/lib/playerEnrichment";
import { StatusBadge } from "@/components/team/StatusBadge";

interface Props {
  player: Player;
  rangeMin: [number, number];
  teamStats?: Record<string, TeamStats | null> | null;
  /** Placeholder position label (not yet in JSON) */
  position?: string;
}

function formatDistance(m: number): string {
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(0)} m`;
}

export function PlayerCard({ player, rangeMin, teamStats, position = "DEF" }: Props) {
  const playerTeamStats = teamStats ? (teamStats[String(player.team)] ?? null) : null;
  const teamColor = player.team === 1 ? "var(--team-1)" : "var(--team-2)";
  const { getPlayerByTrackerId } = useTeam();
  const td = getPlayerByTrackerId(Number(player.id));
  const displayName = td?.name && td.name.trim() ? td.name : `Jugador #${player.id}`;
  const displayJersey = td?.jerseyNumber ?? Number(player.id);
  const displayPosition = td?.position ?? position;
  return (
    <div className="flex flex-col gap-4 rounded-xl border border-border bg-card p-4 shadow-lg">
      {/* Header */}
      <div
        className="relative overflow-hidden rounded-lg p-4"
        style={{
          background: `linear-gradient(135deg, ${teamColor} 0%, color-mix(in oklab, ${teamColor} 50%, black) 100%)`,
        }}
      >
        <div className="flex items-center gap-4 text-white">
          <div className="relative flex h-24 w-20 shrink-0 items-center justify-center overflow-hidden rounded-md bg-black/30 ring-1 ring-white/20">
            {td?.avatarBase64 ? (
              <img
                src={td.avatarBase64}
                alt={`Foto ${displayName}`}
                className="h-full w-full object-cover"
                loading="lazy"
              />
            ) : td ? (
              <span className="text-2xl font-bold">{getInitials(td.name)}</span>
            ) : (
              <User className="h-10 w-10 opacity-70" />
            )}
          </div>
          <div className="flex min-w-0 flex-1 items-start justify-between">
            <div>
              <div className="text-4xl font-black leading-none">#{displayJersey}</div>
              <div className="mt-1 truncate text-sm font-semibold">{displayName}</div>
              <div className="text-[10px] font-semibold uppercase tracking-wider opacity-80">
                Equipo {player.team}
              </div>
            </div>
            <div className="rounded-md bg-black/30 px-2 py-1 text-xs font-bold tracking-wider">
              {displayPosition}
            </div>
          </div>
        </div>
        {td && (
          <div className="mt-3 flex items-center gap-2">
            <StatusBadge status={td.status} />
            {td.status === "injured" && td.recoveryEndDate && (
              <span className="text-[11px] text-white/90">
                ⚠️ Lesionado — recupera en {daysUntil(td.recoveryEndDate)} días
              </span>
            )}
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        <StatBox
          icon={<Gauge className="h-4 w-4" />}
          label="Vel. máx"
          value={`${player.max_speed_kmh.toFixed(1)}`}
          unit="km/h"
        />
        <StatBox
          icon={<RouteIcon className="h-4 w-4" />}
          label="Distancia"
          value={formatDistance(player.total_distance_m)}
        />
        <StatBox
          icon={<Zap className="h-4 w-4" />}
          label="Acel. máx"
          value={`${player.max_acceleration_ms2.toFixed(1)}`}
          unit="m/s²"
        />
      </div>

      {/* Speed chart */}
      <div className="flex flex-col gap-1">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
          Velocidad en el tiempo
        </div>
        <SpeedChart speedOverTime={player.speed_over_time} rangeMin={rangeMin} />
      </div>

      <InsightsList player={player} teamStats={playerTeamStats} />
    </div>
  );
}

function StatBox({
  icon,
  label,
  value,
  unit,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  unit?: string;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-md border border-border bg-muted/30 p-2.5">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        {icon}
        {label}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-bold leading-tight text-foreground">{value}</span>
        {unit && <span className="text-[10px] text-muted-foreground">{unit}</span>}
      </div>
    </div>
  );
}