import { Flag, MoveHorizontal, Target, Hand } from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useVideo } from "@/contexts/VideoContext";
import type { SetPieceEvent, SetPieceType } from "@/types/match";

interface Props {
  setPieces: SetPieceEvent[];
}

const TYPE_CONFIG: Record<SetPieceType, { label: string; icon: typeof Flag }> = {
  falta: { label: "Falta", icon: Hand },
  corner: { label: "Córner", icon: Flag },
  banda: { label: "Banda", icon: MoveHorizontal },
  meta: { label: "Meta", icon: Target },
};

const TYPES = Object.keys(TYPE_CONFIG) as SetPieceType[];

function fmtMinSec(sec: number): string {
  const total = Math.max(0, Math.round(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function SetPiecesCard({ setPieces }: Props) {
  const { seekToMinute } = useVideo();

  const byType: Record<SetPieceType, SetPieceEvent[]> = {
    falta: [],
    corner: [],
    banda: [],
    meta: [],
  };
  for (const event of setPieces) {
    byType[event.type]?.push(event);
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 rounded-xl border border-border bg-card p-4 shadow-lg">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        Balón parado
      </div>

      <Tabs defaultValue="falta" className="flex min-h-0 flex-1 flex-col">
        {/* h-auto en vez de la h-9 fija del TabsList base: con icono + etiqueta en
            dos líneas, 36px no alcanza y el contenido se desbordaba hacia abajo. */}
        <TabsList className="grid h-auto w-full grid-cols-4 gap-1">
          {TYPES.map((type) => {
            const Icon = TYPE_CONFIG[type].icon;
            return (
              <TabsTrigger
                key={type}
                value={type}
                className="flex-col gap-1 py-2 text-[11px] leading-none"
              >
                <Icon className="h-3.5 w-3.5" />
                {TYPE_CONFIG[type].label}
              </TabsTrigger>
            );
          })}
        </TabsList>

        {TYPES.map((type) => (
          <TabsContent key={type} value={type} className="flex min-h-0 flex-1 flex-col">
            <SetPieceList events={byType[type]} onSelect={(sec) => seekToMinute(sec / 60)} />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}

function SetPieceList({
  events,
  onSelect,
}: {
  events: SetPieceEvent[];
  onSelect: (startSec: number) => void;
}) {
  if (events.length === 0) {
    return (
      <p className="flex flex-1 items-center justify-center text-center text-xs text-muted-foreground">
        Sin eventos detectados en este partido.
      </p>
    );
  }

  return (
    <ul className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto">
      {events.map((event, idx) => (
        <li key={`${event.type}-${event.frame_start}-${idx}`}>
          <button
            type="button"
            onClick={() => onSelect(event.start_sec)}
            className="flex w-full cursor-pointer items-center justify-between rounded-md border border-border bg-muted/30 px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-muted/60"
          >
            <span className="font-medium text-foreground">{fmtMinSec(event.start_sec)}</span>
            {event.position_m && (
              <span className="text-[10px] text-muted-foreground">
                ({event.position_m[0].toFixed(0)}, {event.position_m[1].toFixed(0)}) m
              </span>
            )}
          </button>
        </li>
      ))}
    </ul>
  );
}
