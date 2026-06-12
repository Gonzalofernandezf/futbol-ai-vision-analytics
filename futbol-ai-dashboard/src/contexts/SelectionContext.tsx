import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { TeamId } from "@/types/match";

interface SelectionState {
  selectedTeam: TeamId;
  setSelectedTeam: (t: TeamId) => void;
  selectedPlayerId: string | null;
  setSelectedPlayerId: (id: string | null) => void;
  /** Time range in minutes [start, end] */
  timeRange: [number, number];
  setTimeRange: (r: [number, number]) => void;
  matchDurationMin: number;
  setMatchDurationMin: (m: number) => void;
}

const SelectionContext = createContext<SelectionState | null>(null);

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [selectedTeam, setSelectedTeam] = useState<TeamId>(1);
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<[number, number]>([0, 90]);
  const [matchDurationMin, setMatchDurationMin] = useState<number>(90);

  const value = useMemo<SelectionState>(
    () => ({
      selectedTeam,
      setSelectedTeam,
      selectedPlayerId,
      setSelectedPlayerId,
      timeRange,
      setTimeRange,
      matchDurationMin,
      setMatchDurationMin,
    }),
    [selectedTeam, selectedPlayerId, timeRange, matchDurationMin],
  );

  return <SelectionContext.Provider value={value}>{children}</SelectionContext.Provider>;
}

export function useSelection(): SelectionState {
  const ctx = useContext(SelectionContext);
  if (!ctx) throw new Error("useSelection must be used within SelectionProvider");
  return ctx;
}