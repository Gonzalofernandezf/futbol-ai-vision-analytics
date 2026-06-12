import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";
import { useMatchData } from "@/hooks/useMatchData";
import { useSelection } from "@/contexts/SelectionContext";
import { PlayerCard } from "@/components/player/PlayerCard";
import { Heatmap } from "@/components/dashboard/Heatmap";
import { LoadingSkeleton, ErrorPanel } from "@/components/LoadingSkeleton";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Player } from "@/types/match";
import { useTeam } from "@/hooks/useTeam";
import { getInitials } from "@/lib/playerEnrichment";

export const Route = createFileRoute("/compare")({
  head: () => ({
    meta: [
      { title: "Comparar jugadores — Futbol AI" },
      { name: "description", content: "Compara métricas tácticas de dos jugadores lado a lado." },
    ],
  }),
  component: ComparePage,
});

function ComparePage() {
  const { players, loading, error } = useMatchData();
  const { timeRange, matchDurationMin, setMatchDurationMin, setTimeRange } = useSelection();
  const [leftId, setLeftId] = useState<string | null>(null);
  const [rightId, setRightId] = useState<string | null>(null);

  useEffect(() => {
    if (players.length > 0 && (!leftId || !rightId)) {
      const t1 = players.find((p) => p.team === 1);
      const t2 = players.find((p) => p.team === 2);
      if (t1) setLeftId(t1.id);
      if (t2) setRightId(t2.id);
    }
  }, [players, leftId, rightId]);

  useEffect(() => {
    if (matchDurationMin === 90 && players.length > 0) {
      const len = players[0]?.speed_over_time.length ?? 0;
      const mins = Math.max(1, Math.round(len / 60));
      setMatchDurationMin(mins);
      setTimeRange([0, mins]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [players]);

  if (loading) return <LoadingSkeleton />;
  if (error) return <ErrorPanel message={error} />;

  const left = players.find((p) => p.id === leftId) ?? null;
  const right = players.find((p) => p.id === rightId) ?? null;

  return (
    <div className="flex min-h-screen w-full flex-col gap-4 p-4">
      <div className="flex items-center gap-3">
        <Link
          to="/"
          className="flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm hover:bg-muted"
        >
          <ArrowLeft className="h-4 w-4" />
          Volver al dashboard
        </Link>
        <h1 className="text-lg font-semibold">Comparativa</h1>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ComparePane
          label="Jugador A"
          players={players}
          selectedId={leftId}
          onChange={setLeftId}
          player={left}
          rangeMin={timeRange}
        />
        <ComparePane
          label="Jugador B"
          players={players}
          selectedId={rightId}
          onChange={setRightId}
          player={right}
          rangeMin={timeRange}
        />
      </div>
    </div>
  );
}

interface PaneProps {
  label: string;
  players: Player[];
  selectedId: string | null;
  onChange: (id: string) => void;
  player: Player | null;
  rangeMin: [number, number];
}

function ComparePane({ label, players, selectedId, onChange, player, rangeMin }: PaneProps) {
  const team1 = players.filter((p) => p.team === 1);
  const team2 = players.filter((p) => p.team === 2);
  const { getPlayerByTrackerId } = useTeam();

  const renderItem = (p: Player) => {
    const td = getPlayerByTrackerId(Number(p.id));
    return (
      <SelectItem key={p.id} value={p.id}>
        <span className="flex items-center gap-2">
          <span
            className="flex h-5 w-5 items-center justify-center overflow-hidden rounded-full text-[9px] font-bold text-white"
            style={{
              background: td?.avatarBase64
                ? undefined
                : p.team === 1
                  ? "var(--team-1)"
                  : "var(--team-2)",
            }}
          >
            {td?.avatarBase64 ? (
              <img src={td.avatarBase64} alt="" className="h-full w-full object-cover" />
            ) : td?.name?.trim() ? (
              getInitials(td.name)
            ) : (
              `#${p.id}`
            )}
          </span>
          {td?.name?.trim() ? `${td.name} · #${td.jerseyNumber}` : `Jugador #${p.id}`}
          <span className="text-xs text-muted-foreground">
            {(p.total_distance_m / 1000).toFixed(1)} km
          </span>
        </span>
      </SelectItem>
    );
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <span className="text-xs uppercase tracking-wider text-muted-foreground">{label}</span>
        <Select value={selectedId ?? undefined} onValueChange={onChange}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Selecciona un jugador" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Equipo 1</SelectLabel>
              {team1.map(renderItem)}
            </SelectGroup>
            <SelectGroup>
              <SelectLabel>Equipo 2</SelectLabel>
              {team2.map(renderItem)}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>
      {player ? (
        <>
          <PlayerCard player={player} rangeMin={rangeMin} />
          <div className="rounded-lg border border-border bg-card p-3">
            <div className="mb-2 text-[10px] uppercase tracking-wider text-muted-foreground">
              Mapa de calor
            </div>
            <Heatmap positions={player.position_history} width={600} height={384} />
          </div>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">Selecciona un jugador</p>
      )}
    </div>
  );
}