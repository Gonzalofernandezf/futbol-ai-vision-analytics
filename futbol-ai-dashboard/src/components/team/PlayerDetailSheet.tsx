import { Pencil, User, Gauge, Route as RouteIcon, Zap } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "./StatusBadge";
import { AvatarUploader } from "./AvatarUploader";
import { SpeedChart } from "@/components/player/SpeedChart";
import { PlayerRadar } from "@/components/player/PlayerRadar";
import { daysUntil, getInitials } from "@/lib/playerEnrichment";
import { useMatchData } from "@/hooks/useMatchData";
import type { TeamPlayer } from "@/types/team";

interface Props {
  player: TeamPlayer | null;
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onEdit: (p: TeamPlayer) => void;
  onAvatar: (file: File) => void;
}

export function PlayerDetailSheet({ player, open, onOpenChange, onEdit, onAvatar }: Props) {
  const { players: matchPlayers } = useMatchData();
  if (!player) return null;

  const linked = player.trackerId != null
    ? matchPlayers.find((m) => Number(m.id) === player.trackerId)
    : undefined;

  const teamColor = player.team === 1 ? "var(--team-1)" : "var(--team-2)";

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-md">
        <SheetHeader>
          <SheetTitle className="sr-only">Detalle de {player.name}</SheetTitle>
          <SheetDescription className="sr-only">
            Resumen de estado, métricas e historial del jugador
          </SheetDescription>
        </SheetHeader>

        <div className="mt-2 flex items-center gap-4">
          <div
            className="flex h-20 w-20 shrink-0 items-center justify-center overflow-hidden rounded-full text-2xl font-bold text-white ring-2 ring-border"
            style={{ background: player.avatarBase64 ? undefined : teamColor }}
          >
            {player.avatarBase64 ? (
              <img src={player.avatarBase64} alt="" className="h-full w-full object-cover" />
            ) : player.name ? (
              getInitials(player.name)
            ) : (
              <User className="h-8 w-8" />
            )}
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xl font-bold leading-tight">{player.name}</div>
            <div className="text-xs text-muted-foreground">
              Dorsal #{player.jerseyNumber} · Equipo {player.team}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              <Badge variant="secondary">{player.position}</Badge>
              <StatusBadge status={player.status} />
            </div>
          </div>
        </div>

        {player.status === "injured" && player.recoveryEndDate && (
          <div className="mt-4 rounded-md border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">
            ⚠️ Lesionado — recupera en {daysUntil(player.recoveryEndDate)} días
          </div>
        )}

        <div className="mt-4 flex gap-2">
          <Button variant="outline" size="sm" onClick={() => onEdit(player)}>
            <Pencil className="h-4 w-4" /> Editar datos
          </Button>
          <AvatarUploader onFile={onAvatar} label="Cambiar avatar" />
        </div>

        <Section title="Métricas del último partido">
          {linked ? (
            <>
              <div className="grid grid-cols-3 gap-2">
                <Metric
                  icon={<Gauge className="h-3.5 w-3.5" />}
                  label="Vel. máx"
                  value={`${linked.max_speed_kmh.toFixed(1)} km/h`}
                />
                <Metric
                  icon={<RouteIcon className="h-3.5 w-3.5" />}
                  label="Distancia"
                  value={
                    linked.total_distance_m >= 1000
                      ? `${(linked.total_distance_m / 1000).toFixed(2)} km`
                      : `${linked.total_distance_m.toFixed(0)} m`
                  }
                />
                <Metric
                  icon={<Zap className="h-3.5 w-3.5" />}
                  label="Acel. máx"
                  value={`${linked.max_acceleration_ms2.toFixed(1)} m/s²`}
                />
              </div>
              <div className="mt-3">
                <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                  Velocidad en el tiempo
                </div>
                <SpeedChart
                  speedOverTime={linked.speed_over_time}
                  rangeMin={[0, Math.max(1, Math.round(linked.speed_over_time.length / 60))]}
                />
              </div>
              <div className="mt-3">
                <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                  Perfil vs Equipo {player.team} (percentiles)
                </div>
                <PlayerRadar player={linked} teamPlayers={matchPlayers} />
              </div>
            </>
          ) : (
            <p className="text-xs text-muted-foreground">
              {player.trackerId == null
                ? "Sin Tracker ID — asigna uno en edición para enlazar las estadísticas del pipeline."
                : `No se encontró el tracker ${player.trackerId} en el partido cargado.`}
            </p>
          )}
        </Section>

        <Section title="Historial / Notas">
          <p className="whitespace-pre-wrap text-xs text-muted-foreground">
            {player.notes?.trim() || "Sin notas registradas."}
          </p>
        </Section>
      </SheetContent>
    </Sheet>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-6">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </div>
      <div className="rounded-lg border border-border bg-card p-3">{children}</div>
    </div>
  );
}

function Metric({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-md border border-border bg-muted/30 p-2">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        {icon}
        {label}
      </div>
      <div className="mt-0.5 text-sm font-bold">{value}</div>
    </div>
  );
}