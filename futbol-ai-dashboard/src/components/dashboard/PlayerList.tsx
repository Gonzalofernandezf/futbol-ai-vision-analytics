import type { Player } from "@/types/match";
import { Gauge, Route as RouteIcon, User } from "lucide-react";
import { useTeam } from "@/hooks/useTeam";
import { getInitials } from "@/lib/playerEnrichment";

interface Props {
  players: Player[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function PlayerList({ players, selectedId, onSelect }: Props) {
  const { getPlayerByTrackerId } = useTeam();
  if (players.length === 0) {
    return <p className="text-xs text-muted-foreground">Sin jugadores</p>;
  }
  return (
    <ul className="flex flex-col gap-1.5 overflow-y-auto pr-1">
      {players.map((p) => {
        const active = p.id === selectedId;
        const td = getPlayerByTrackerId(Number(p.id));
        const teamColor = p.team === 1 ? "var(--team-1)" : "var(--team-2)";
        return (
          <li key={p.id}>
            <button
              onClick={() => onSelect(p.id)}
              className={`flex w-full items-center gap-3 rounded-md border px-3 py-2 text-left transition-colors ${
                active
                  ? "border-primary/60 bg-primary/10"
                  : "border-border bg-card hover:border-primary/40 hover:bg-primary/5"
              }`}
            >
              <div
                className="flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-md text-xs font-bold text-white"
                style={{ background: td?.avatarBase64 ? undefined : teamColor }}
              >
                {td?.avatarBase64 ? (
                  <img src={td.avatarBase64} alt="" className="h-full w-full object-cover" />
                ) : td?.name?.trim() ? (
                  getInitials(td.name)
                ) : (
                  <User className="h-4 w-4 opacity-80" />
                )}
              </div>
              <div className="flex min-w-0 flex-1 flex-col">
                <span className="truncate text-xs font-medium text-foreground">
                  {td?.name?.trim() ? `${td.name} · #${td.jerseyNumber}` : `Jugador #${p.id}`}
                </span>
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  <RouteIcon className="h-3 w-3" />
                  {(p.total_distance_m / 1000).toFixed(1)} km
                  <span className="mx-1">·</span>
                  <Gauge className="h-3 w-3" />
                  {p.max_speed_kmh.toFixed(1)} km/h
                </span>
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}