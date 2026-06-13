import type { Player } from "@/types/match";
import type { Team, TeamPlayer } from "@/types/team";

export type EnrichedPlayer = Player & { teamData?: TeamPlayer };

export function enrichPlayerStats(stats: Player, team: Team): EnrichedPlayer {
  const trackerId = Number(stats.id);
  if (!Number.isFinite(trackerId)) return stats;
  const teamData = team.players.find((p) => p.trackerId === trackerId);
  return teamData ? { ...stats, teamData } : stats;
}

export function getInitials(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}

export function daysUntil(iso: string): number {
  const end = new Date(iso).getTime();
  const now = Date.now();
  return Math.max(0, Math.ceil((end - now) / 86400000));
}
