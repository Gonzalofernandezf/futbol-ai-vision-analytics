import { Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "./StatusBadge";
import { AvatarUploader } from "./AvatarUploader";
import { daysUntil, getInitials } from "@/lib/playerEnrichment";
import type { TeamPlayer } from "@/types/team";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

interface Props {
  player: TeamPlayer;
  onEdit: () => void;
  onDelete: () => void;
  onAvatar: (file: File) => void;
}

export function TeamPlayerCard({ player, onEdit, onDelete, onAvatar }: Props) {
  const teamColor = player.team === 1 ? "var(--team-1)" : "var(--team-2)";
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-border bg-card p-5 shadow-sm">
      <div
        className="flex h-32 w-32 items-center justify-center overflow-hidden rounded-full ring-2 ring-border"
        style={{ background: player.avatarBase64 ? undefined : teamColor }}
      >
        {player.avatarBase64 ? (
          <img
            src={player.avatarBase64}
            alt={`Avatar ${player.name}`}
            className="h-full w-full object-cover"
          />
        ) : (
          <span className="text-3xl font-bold text-white">{getInitials(player.name)}</span>
        )}
      </div>
      <div className="text-center">
        <div className="text-lg font-semibold leading-tight">{player.name}</div>
        <div className="text-xs text-muted-foreground">Dorsal #{player.jerseyNumber} · Equipo {player.team}</div>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Badge variant="secondary">{player.position}</Badge>
        <StatusBadge status={player.status} />
      </div>
      {player.status === "injured" && player.recoveryEndDate && (
        <p className="text-xs text-red-400">
          Recupera en {daysUntil(player.recoveryEndDate)} días
        </p>
      )}
      <div className="mt-auto flex w-full items-center justify-center gap-1 border-t border-border pt-3">
        <AvatarUploader onFile={onAvatar} />
        <Button variant="ghost" size="icon" aria-label="Editar jugador" onClick={onEdit}>
          <Pencil className="h-4 w-4" />
        </Button>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button variant="ghost" size="icon" aria-label="Borrar jugador">
              <Trash2 className="h-4 w-4" />
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>¿Borrar a {player.name}?</AlertDialogTitle>
              <AlertDialogDescription>
                Esta acción no se puede deshacer.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancelar</AlertDialogCancel>
              <AlertDialogAction onClick={onDelete}>Borrar</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}
