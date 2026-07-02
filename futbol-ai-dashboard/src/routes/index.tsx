import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { useMatchData } from "@/hooks/useMatchData";
import { useSelection } from "@/contexts/SelectionContext";
import { DashboardSidebar } from "@/components/dashboard/DashboardSidebar";
import { MatchHeader } from "@/components/dashboard/MatchHeader";
import { Heatmap } from "@/components/dashboard/Heatmap";
import { VideoPlayer } from "@/components/dashboard/VideoPlayer";
import { PlayerCard } from "@/components/player/PlayerCard";
import { SetPiecesCard } from "@/components/player/SetPiecesCard";
import { LoadingSkeleton, ErrorPanel } from "@/components/LoadingSkeleton";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard — Futbol AI" },
      { name: "description", content: "Análisis táctico de fútbol con IA por jugador." },
    ],
  }),
  component: DashboardPage,
});

function DashboardPage() {
  const { meta, players, teamStats, setPieces, loading, error } = useMatchData();
  const {
    selectedPlayerId,
    setSelectedPlayerId,
    selectedTeam,
    timeRange,
    setTimeRange,
    setMatchDurationMin,
    matchDurationMin,
  } = useSelection();

  // Initialize defaults once data arrives
  useEffect(() => {
    if (meta) {
      const mins = Math.round(meta.duration_seconds / 60);
      setMatchDurationMin(mins);
      setTimeRange([0, mins]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta]);

  useEffect(() => {
    if (!selectedPlayerId && players.length > 0) {
      const first = players.find((p) => p.team === selectedTeam) ?? players[0];
      setSelectedPlayerId(first.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [players, selectedPlayerId]);

  if (loading) return <LoadingSkeleton />;
  if (error || !meta) return <ErrorPanel message={error ?? "Datos no disponibles"} />;

  const selected = players.find((p) => p.id === selectedPlayerId) ?? null;

  // Filter positions by time range. position_history has no timestamps; assume
  // even temporal distribution across the match for the placeholder dataset.
  const filteredPositions = selected
    ? sliceByRange(selected.position_history, timeRange, matchDurationMin)
    : [];

  return (
    <div className="flex min-h-screen w-full">
      <DashboardSidebar players={players} />

      <main className="grid min-w-0 flex-1 grid-cols-1 items-stretch gap-4 p-4 xl:grid-cols-[280px_1fr_360px]">
        {/* Balón parado: columna propia entre el listado de jugadores y el vídeo,
            con toda la altura disponible para el listado (en vez de un recuadro
            apretado bajo la tarjeta de jugador). */}
        <div className="flex min-w-0 flex-col">
          <SetPiecesCard setPieces={setPieces} />
        </div>

        <section className="flex min-w-0 flex-col gap-4">
          <MatchHeader meta={meta} />
          <VideoPlayer meta={meta} />
          {selected ? (
            <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Mapa de calor · Jugador #{selected.id}
              </div>
              <Heatmap positions={filteredPositions} />
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Selecciona un jugador</p>
          )}
        </section>

        <aside className="flex flex-col gap-4">
          {selected ? (
            <PlayerCard player={selected} rangeMin={timeRange} teamStats={teamStats} />
          ) : (
            <p className="text-sm text-muted-foreground">Selecciona un jugador</p>
          )}
        </aside>
      </main>
    </div>
  );
}

function sliceByRange(
  positions: Array<[number, number] | null>,
  range: [number, number],
  totalMin: number,
): Array<[number, number] | null> {
  if (totalMin <= 0) return positions;
  const n = positions.length;
  const start = Math.floor((range[0] / totalMin) * n);
  const end = Math.ceil((range[1] / totalMin) * n);
  return positions.slice(start, end);
}
