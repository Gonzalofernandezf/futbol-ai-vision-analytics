import { useEffect, useState } from "react";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { PlayerStatus, TeamPlayer } from "@/types/team";
import { POSITION_OPTIONS } from "@/types/team";

const schema = z
  .object({
    name: z.string().trim().min(1, "Nombre requerido").max(80),
    jerseyNumber: z.number().int().min(1, "Dorsal 1-99").max(99, "Dorsal 1-99"),
    position: z.string().min(1),
    team: z.union([z.literal(1), z.literal(2)]),
    status: z.enum(["fit", "training", "doubtful", "injured"]),
    recoveryEndDate: z.string().optional(),
    trackerId: z.number().int().min(0).max(9999).optional(),
    notes: z.string().max(500).optional(),
  })
  .refine((v) => v.status !== "injured" || !!v.recoveryEndDate, {
    message: "Indica fecha de recuperación",
    path: ["recoveryEndDate"],
  });

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  initial?: TeamPlayer | null;
  onSubmit: (data: Omit<TeamPlayer, "id">) => void;
}

type FormState = {
  name: string;
  jerseyNumber: string;
  position: string;
  team: "1" | "2";
  status: PlayerStatus;
  recoveryEndDate: string;
  trackerId: string;
  notes: string;
};

const empty: FormState = {
  name: "",
  jerseyNumber: "",
  position: "MED",
  team: "1",
  status: "fit",
  recoveryEndDate: "",
  trackerId: "",
  notes: "",
};

export function PlayerFormDialog({ open, onOpenChange, initial, onSubmit }: Props) {
  const [form, setForm] = useState<FormState>(empty);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (open) {
      setErrors({});
      setForm(
        initial
          ? {
              name: initial.name,
              jerseyNumber: String(initial.jerseyNumber),
              position: initial.position,
              team: String(initial.team) as "1" | "2",
              status: initial.status,
              recoveryEndDate: initial.recoveryEndDate?.slice(0, 10) ?? "",
              trackerId: initial.trackerId != null ? String(initial.trackerId) : "",
              notes: initial.notes ?? "",
            }
          : empty,
      );
    }
  }, [open, initial]);

  const update = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((f) => ({ ...f, [k]: v }));

  const handleSave = () => {
    const parsed = schema.safeParse({
      name: form.name,
      jerseyNumber: Number(form.jerseyNumber),
      position: form.position,
      team: Number(form.team) as 1 | 2,
      status: form.status,
      recoveryEndDate: form.recoveryEndDate
        ? new Date(form.recoveryEndDate).toISOString()
        : undefined,
      trackerId: form.trackerId ? Number(form.trackerId) : undefined,
      notes: form.notes || undefined,
    });
    if (!parsed.success) {
      const errs: Record<string, string> = {};
      for (const issue of parsed.error.issues) {
        const k = String(issue.path[0] ?? "");
        if (!errs[k]) errs[k] = issue.message;
      }
      setErrors(errs);
      return;
    }
    onSubmit(parsed.data);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{initial ? "Editar jugador" : "Añadir jugador"}</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4">
          <Field label="Nombre completo" error={errors.name}>
            <Input value={form.name} onChange={(e) => update("name", e.target.value)} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Dorsal" error={errors.jerseyNumber}>
              <Input
                type="number"
                min={1}
                max={99}
                value={form.jerseyNumber}
                onChange={(e) => update("jerseyNumber", e.target.value)}
              />
            </Field>
            <Field label="Posición">
              <Select value={form.position} onValueChange={(v) => update("position", v)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {POSITION_OPTIONS.map((p) => (
                    <SelectItem key={p} value={p}>{p}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Equipo">
              <Select value={form.team} onValueChange={(v) => update("team", v as "1" | "2")}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Equipo 1</SelectItem>
                  <SelectItem value="2">Equipo 2</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            <Field label="Estado físico">
              <Select value={form.status} onValueChange={(v) => update("status", v as PlayerStatus)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="fit">En forma</SelectItem>
                  <SelectItem value="training">Puesta a punto</SelectItem>
                  <SelectItem value="doubtful">Duda</SelectItem>
                  <SelectItem value="injured">Lesionado</SelectItem>
                </SelectContent>
              </Select>
            </Field>
          </div>
          {form.status === "injured" && (
            <Field label="Fecha de recuperación" error={errors.recoveryEndDate}>
              <Input
                type="date"
                value={form.recoveryEndDate}
                onChange={(e) => update("recoveryEndDate", e.target.value)}
              />
            </Field>
          )}
          <Field
            label="Tracker ID (opcional)"
            hint="Si conoces el ID que el pipeline asignó a este jugador, ingrésalo para enlazar las estadísticas"
          >
            <Input
              type="number"
              value={form.trackerId}
              onChange={(e) => update("trackerId", e.target.value)}
            />
          </Field>
          <Field label="Notas (opcional)">
            <Textarea
              rows={3}
              value={form.notes}
              onChange={(e) => update("notes", e.target.value)}
            />
          </Field>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancelar</Button>
          <Button onClick={handleSave}>Guardar</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string;
  hint?: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid gap-1.5">
      <Label className="text-xs">{label}</Label>
      {children}
      {hint && !error && <p className="text-[11px] text-muted-foreground">{hint}</p>}
      {error && <p className="text-[11px] text-red-400">{error}</p>}
    </div>
  );
}