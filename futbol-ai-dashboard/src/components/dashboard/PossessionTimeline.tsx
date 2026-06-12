import type { MatchMeta } from "@/types/match";

export function PossessionTimeline({ meta }: { meta: MatchMeta }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>Timeline de posesión</span>
        <span>aprox.</span>
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded-sm">
        <div
          style={{ width: `${meta.home_possession}%`, backgroundColor: "var(--team-1)" }}
        />
        <div
          style={{ width: `${meta.away_possession}%`, backgroundColor: "var(--team-2)" }}
        />
      </div>
    </div>
  );
}