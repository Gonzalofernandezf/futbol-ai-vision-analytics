import type { MatchMeta } from "@/types/match";
import { Clock } from "lucide-react";

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function MatchHeader({ meta }: { meta: MatchMeta }) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Clock className="h-4 w-4" />
          <span>Duración</span>
          <span className="font-mono text-foreground">{formatDuration(meta.duration_seconds)}</span>
        </div>
        <div className="text-xs text-muted-foreground">{meta.fps.toFixed(2)} fps</div>
      </div>
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between text-xs">
          <span style={{ color: "var(--team-1)" }} className="font-semibold">
            Equipo 1 · {meta.home_possession.toFixed(1)}%
          </span>
          <span style={{ color: "var(--team-2)" }} className="font-semibold">
            {meta.away_possession.toFixed(1)}% · Equipo 2
          </span>
        </div>
        <div className="flex h-2 overflow-hidden rounded-full">
          <div
            style={{
              width: `${meta.home_possession}%`,
              backgroundColor: "var(--team-1)",
            }}
          />
          <div
            style={{
              width: `${meta.away_possession}%`,
              backgroundColor: "var(--team-2)",
            }}
          />
        </div>
      </div>
    </div>
  );
}