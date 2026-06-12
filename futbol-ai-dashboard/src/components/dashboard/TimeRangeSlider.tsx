import { Slider } from "@/components/ui/slider";

interface Props {
  value: [number, number];
  max: number;
  onChange: (v: [number, number]) => void;
}

export function TimeRangeSlider({ value, max, onChange }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Rango</span>
        <span className="font-mono">
          {value[0]}′ — {value[1]}′
        </span>
      </div>
      <Slider
        value={value}
        min={0}
        max={max}
        step={1}
        onValueChange={(v) => onChange([v[0] ?? 0, v[1] ?? max] as [number, number])}
      />
    </div>
  );
}