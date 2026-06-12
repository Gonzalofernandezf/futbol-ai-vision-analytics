import type { Player } from "@/types/match";
import { TeamTabs } from "./TeamTabs";
import { PlayerList } from "./PlayerList";
import { TimeRangeSlider } from "./TimeRangeSlider";
import { useSelection } from "@/contexts/SelectionContext";

interface Props {
  players: Player[];
}

export function DashboardSidebar({ players }: Props) {
  const {
    selectedTeam,
    setSelectedTeam,
    selectedPlayerId,
    setSelectedPlayerId,
    timeRange,
    setTimeRange,
    matchDurationMin,
  } = useSelection();

  const teamPlayers = players.filter((p) => p.team === selectedTeam);

  return (
    <aside className="flex w-[250px] shrink-0 flex-col gap-4 border-r border-border bg-sidebar p-4">
      <div>
        <h1 className="text-lg font-bold tracking-tight">Futbol AI</h1>
        <p className="text-xs text-muted-foreground">Análisis táctico</p>
      </div>

      <TeamTabs
        selected={selectedTeam}
        onSelect={(t) => {
          setSelectedTeam(t);
          const first = players.find((p) => p.team === t);
          setSelectedPlayerId(first?.id ?? null);
        }}
      />

      <div className="flex min-h-0 flex-1 flex-col gap-2">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
          Jugadores
        </div>
        <PlayerList
          players={teamPlayers}
          selectedId={selectedPlayerId}
          onSelect={setSelectedPlayerId}
        />
      </div>

      <div className="border-t border-border pt-3">
        <TimeRangeSlider value={timeRange} max={matchDurationMin} onChange={setTimeRange} />
      </div>
    </aside>
  );
}