import { Badge } from "@/components/ui/badge";
import type { PlayerStatus } from "@/types/team";
import { STATUS_LABELS } from "@/types/team";

const COLORS: Record<PlayerStatus, string> = {
  fit: "bg-emerald-500/15 text-emerald-400 border-emerald-500/40",
  training: "bg-blue-500/15 text-blue-400 border-blue-500/40",
  doubtful: "bg-amber-500/15 text-amber-400 border-amber-500/40",
  injured: "bg-red-500/15 text-red-400 border-red-500/40",
};

export function StatusBadge({ status }: { status: PlayerStatus }) {
  return (
    <Badge variant="outline" className={`${COLORS[status]} font-medium`}>
      {STATUS_LABELS[status]}
    </Badge>
  );
}
