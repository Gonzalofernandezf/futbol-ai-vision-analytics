import { useEffect, useState } from "react";
import type { MatchData, Player, TeamStats, SetPieceEvent } from "@/types/match";

interface UseMatchDataResult {
  meta: MatchData["match_meta"] | null;
  players: Player[];
  teamStats: Record<string, TeamStats> | null;
  setPieces: SetPieceEvent[];
  schemaVersion: number;
  loading: boolean;
  error: string | null;
}

export function useMatchData(): UseMatchDataResult {
  const [data, setData] = useState<MatchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/match_data.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<MatchData>;
      })
      .then((json) => {
        if (!cancelled) {
          setData(json);
          setLoading(false);
          if (import.meta.env.DEV && !json.schema_version) {
            console.warn(
              "[useMatchData] schema v1 detected — derived and team_stats fields are unavailable",
            );
          }
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Unknown error");
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const players: Player[] = data
    ? Object.entries(data.players).map(([id, p]) => ({ id, ...p }))
    : [];

  return {
    meta: data?.match_meta ?? null,
    players,
    teamStats: data?.team_stats ?? null,
    setPieces: data?.set_pieces ?? [],
    schemaVersion: data?.schema_version ?? 1,
    loading,
    error,
  };
}
