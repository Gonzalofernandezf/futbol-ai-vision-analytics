import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Plus, Search } from "lucide-react";
import { useTeam } from "@/hooks/useTeam";
import { PlayerRow } from "@/components/team/PlayerRow";
import { PlayerDetailSheet } from "@/components/team/PlayerDetailSheet";
import { PlayerFormDialog } from "@/components/team/PlayerFormDialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { PlayerStatus, TeamPlayer } from "@/types/team";

export const Route = createFileRoute("/team")({
  head: () => ({
    meta: [
      { title: "Gestión del equipo — Futbol AI" },
      { name: "description", content: "Gestiona el plantel: dorsales, posiciones, estado físico y avatares." },
    ],
  }),
  component: TeamPage,
});

function TeamPage() {
  const { team, addPlayer, updatePlayer, deletePlayer, uploadAvatar } = useTeam();
  const [teamFilter, setTeamFilter] = useState<"all" | "1" | "2">("all");
  const [statusFilter, setStatusFilter] = useState<"all" | PlayerStatus>("all");
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<TeamPlayer | null>(null);
  const [detailId, setDetailId] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return team.players.filter((p) => {
      if (teamFilter !== "all" && String(p.team) !== teamFilter) return false;
      if (statusFilter !== "all" && p.status !== statusFilter) return false;
      if (q) {
        const inName = p.name.toLowerCase().includes(q);
        const inDorsal = String(p.jerseyNumber).includes(q);
        if (!inName && !inDorsal) return false;
      }
      return true;
    });
  }, [team.players, teamFilter, statusFilter, search]);

  const openAdd = () => {
    setEditing(null);
    setOpen(true);
  };
  const openEdit = (p: TeamPlayer) => {
    setEditing(p);
    setOpen(true);
  };

  const detail = detailId ? team.players.find((p) => p.id === detailId) ?? null : null;

  return (
    <div className="flex w-full flex-col gap-5 p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Gestión del equipo</h1>
          <p className="text-sm text-muted-foreground">
            {team.players.length} jugadores en el plantel
          </p>
        </div>
        <Button onClick={openAdd}>
          <Plus className="h-4 w-4" /> Añadir jugador
        </Button>
      </header>

      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-card p-3">
        <Select value={teamFilter} onValueChange={(v) => setTeamFilter(v as typeof teamFilter)}>
          <SelectTrigger className="w-[160px]"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Todos los equipos</SelectItem>
            <SelectItem value="1">Equipo 1</SelectItem>
            <SelectItem value="2">Equipo 2</SelectItem>
          </SelectContent>
        </Select>
        <Select value={statusFilter} onValueChange={(v) => setStatusFilter(v as typeof statusFilter)}>
          <SelectTrigger className="w-[180px]"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Todos los estados</SelectItem>
            <SelectItem value="fit">En forma</SelectItem>
            <SelectItem value="training">Puesta a punto</SelectItem>
            <SelectItem value="doubtful">Duda</SelectItem>
            <SelectItem value="injured">Lesionado</SelectItem>
          </SelectContent>
        </Select>
        <div className="relative ml-auto min-w-[220px] flex-1 sm:max-w-xs">
          <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            className="pl-8"
            placeholder="Buscar por nombre o dorsal"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Buscar jugadores"
          />
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border p-12 text-center text-sm text-muted-foreground">
          No hay jugadores que coincidan con los filtros.
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-border bg-card">
          <div className="hidden items-center gap-3 border-b border-border bg-muted/30 px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground sm:flex">
            <div className="w-10" />
            <div className="w-10 text-center">#</div>
            <div className="flex-1">Jugador</div>
            <div className="w-[60px]">Pos.</div>
            <div className="hidden w-[140px] md:block">Estado</div>
            <div className="hidden w-28 text-right md:block">Recuperación</div>
            <div className="w-[124px] text-right">Acciones</div>
          </div>
          {filtered.map((p) => (
            <PlayerRow
              key={p.id}
              player={p}
              onOpen={() => setDetailId(p.id)}
              onEdit={() => openEdit(p)}
              onDelete={() => deletePlayer(p.id)}
              onAvatar={(file) => uploadAvatar(p.id, file)}
            />
          ))}
        </div>
      )}

      <PlayerFormDialog
        open={open}
        onOpenChange={setOpen}
        initial={editing}
        onSubmit={(data) => {
          if (editing) updatePlayer(editing.id, data);
          else addPlayer(data);
        }}
      />

      <PlayerDetailSheet
        player={detail}
        open={detailId !== null}
        onOpenChange={(o) => !o && setDetailId(null)}
        onEdit={(p) => {
          setDetailId(null);
          openEdit(p);
        }}
        onAvatar={(file) => detail && uploadAvatar(detail.id, file)}
      />
    </div>
  );
}