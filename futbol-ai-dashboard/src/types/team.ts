export type PlayerStatus = "fit" | "training" | "doubtful" | "injured";

export interface TeamPlayer {
  id: string;
  name: string;
  jerseyNumber: number;
  position: string;
  team: 1 | 2;
  status: PlayerStatus;
  recoveryEndDate?: string;
  avatarBase64?: string;
  trackerId?: number;
  notes?: string;
}

export interface Team {
  players: TeamPlayer[];
  lastUpdated: string;
}

export const STATUS_LABELS: Record<PlayerStatus, string> = {
  fit: "En forma",
  training: "Puesta a punto",
  doubtful: "Duda",
  injured: "Lesionado",
};

export const POSITION_OPTIONS = ["POR", "DEF", "MED", "DEL"] as const;