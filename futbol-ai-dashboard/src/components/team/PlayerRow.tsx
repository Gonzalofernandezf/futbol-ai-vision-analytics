import { Pencil, Trash2, User } from "lucide-react";
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
  onOpen: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onAvatar: (file: File) => void;
}

export function PlayerRow({ player, onOpen, onEdit, onDelete, onAvatar }: Props) {
  const teamColor = player.team === 1 ? "var(--team-1)" : "var(--team-2)";
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen();
        }
      }}
      className="group flex cursor-pointer items-center gap-3 border-b border-border px-3 py-2.5 transition-colors hover:bg-muted/40 focus:bg-muted/40 focus:outline-none"
    >
      <div
        className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-full text-xs font-bold text-white"
        style={{ background: player.avatarBase64 ? undefined : teamColor }}
      >
        {player.avatarBase64 ? (
          <img src={player.avatarBase64} alt="" className="h-full w-full object-cover" />
        ) : player.name ? (
          getInitials(player.name)
        ) : (
          <User className="h-4 w-4" />
        )}
      </div>

      <div className="flex w-10 shrink-0 items-center justify-center font-mono text-sm font-bold text-muted-foreground">
        #{player.jerseyNumber}
      </div>

      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">{player.name}</div>
        <div className="truncate text-[11px] text-muted-foreground">
          Equipo {player.team}
          {player.trackerId != null && ` · Tracker ${player.trackerId}`}
        </div>
      </div>

      <Badge variant="secondary" className="hidden shrink-0 sm:inline-flex">
        {player.position}
      </Badge>

      <div className="hidden shrink-0 md:block">
        <StatusBadge status={player.status} />
      </div>

      <div className="hidden w-28 shrink-0 text-right text-[11px] text-muted-foreground md:block">
        {player.status === "injured" && player.recoveryEndDate
          ? `Recupera en ${daysUntil(player.recoveryEndDate)}d`
          : ""}
      </div>

      <div
        className="flex shrink-0 items-center gap-0.5 opacity-60 transition-opacity group-hover:opacity-100"
        onClick={(e) => e.stopPropagation()}
      >
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