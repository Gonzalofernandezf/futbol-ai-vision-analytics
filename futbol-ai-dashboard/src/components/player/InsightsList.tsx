import type { Player, TeamStats } from "@/types/match";
import { Sparkles, AlertTriangle } from "lucide-react";
import { useSelection } from "@/contexts/SelectionContext";
import { buildInsightCandidates, selectTopInsights, dominantZone } from "@/lib/insights";

/** Legacy 3-insight view for v1 JSON (no derived data) */
function InsightsV1({ player }: { player: Player }) {
  const insights = [
    `Recorrió ${(player.total_distance_m / 1000).toFixed(1)} km en el partido.`,
    `Alcanzó ${player.max_speed_kmh.toFixed(1)} km/h en su pico de velocidad.`,
    `Concentró su actividad en la ${dominantZone(player.position_history)}.`,
  ];
  return (
    <ul className="flex flex-col gap-1.5 text-sm text-foreground/90">
      {insights.map((i) => (
        <li key={i} className="flex gap-2">
          <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-primary" />
          <span>{i}</span>
        </li>
      ))}
    </ul>
  );
}

export function InsightsList({
  player,
  teamStats,
}: {
  player: Player;
  teamStats: TeamStats | null;
}) {
  const { timeRange, matchDurationMin } = useSelection();
  const missingPct = player.derived?.missing_data_pct ?? 0;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
        <Sparkles className="h-3.5 w-3.5 text-primary" />
        Insights
      </div>

      {!player.derived ? (
        <InsightsV1 player={player} />
      ) : (
        <>
          {missingPct > 30 && (
            <div className="flex items-center gap-1.5 rounded-md border border-yellow-500/40 bg-yellow-500/10 px-2.5 py-1.5 text-xs text-yellow-600 dark:text-yellow-400">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
              Datos limitados en {missingPct.toFixed(0)}% del partido
            </div>
          )}
          <ul className="flex flex-col gap-1.5 text-sm text-foreground/90">
            {selectTopInsights(
              buildInsightCandidates(player, teamStats, timeRange, matchDurationMin),
            ).map((insight) => (
              <li key={insight.key} className="flex gap-2">
                <span
                  className={`mt-2 h-1 w-1 shrink-0 rounded-full ${
                    insight.severity === "high" ? "bg-primary" : "bg-muted-foreground/60"
                  }`}
                />
                <span>{insight.message}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
