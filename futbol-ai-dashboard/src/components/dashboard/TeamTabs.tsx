import type { TeamId } from "@/types/match";

interface Props {
  selected: TeamId;
  onSelect: (t: TeamId) => void;
}

export function TeamTabs({ selected, onSelect }: Props) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {([1, 2] as TeamId[]).map((t) => {
        const active = selected === t;
        const color = t === 1 ? "var(--team-1)" : "var(--team-2)";
        return (
          <button
            key={t}
            onClick={() => onSelect(t)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold transition-all ${
              active
                ? "border-transparent text-white shadow"
                : "border-border bg-muted/40 text-muted-foreground hover:text-foreground"
            }`}
            style={active ? { backgroundColor: color } : undefined}
          >
            Equipo {t}
          </button>
        );
      })}
    </div>
  );
}